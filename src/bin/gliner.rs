//! GLiNER: zero-shot NER, where the entity types are *input text* rather than
//! a fixed set of output classes.
//!
//! The contrast with `bert-base-NER` (see `main.rs`) is the point of this file:
//!
//! |            | bert-base-NER          | GLiNER                       |
//! | ---------- | ---------------------- | ---------------------------- |
//! | inputs     | 3                      | 6                            |
//! | labels     | frozen in the weights  | supplied per call, as text   |
//! | output     | `[batch, seq, labels]` | `[batch, words, width, types]` |
//! | activation | softmax over labels    | sigmoid per (span, type)     |
//! | decode     | BIO state machine      | threshold + overlap resolution |
//!
//! Half the domain logic moves from *after* the forward pass to *before* it:
//! the model scores candidate spans that we enumerate, so building `span_idx`
//! correctly is a precondition for calling it at all.

#![feature(iter_intersperse)]
#![feature(iter_advance_by)]

use anyhow::{Result, anyhow};
use ndarray::prelude::*;
use ort::{session::Session, value::TensorRef};
use std::collections::HashSet;
use std::println;
use tokenizers::{Encoding, Tokenizer};

/// Baked into the exported graph as six `Constant` nodes — `gliner_config.json`
/// reports it, but tracing froze it, so this is not configurable here.
const MAX_WIDTH: usize = 12;

const MODEL_DIR: &str = "./onnx/gliner-small-v2.1";

/// A word of the source text, paired with the byte range it occupies.
///
/// GLiNER's `words_splitter_type` is `whitespace`, and span indices refer to
/// *these* units — not to subword tokens. Keeping the byte range lets us slice
/// the original string when reporting an entity.
#[derive(Debug)]
struct Word {
    text: String,
    start: usize,
    end: usize,
}

/// Split `text` into the word units GLiNER's span indices refer to.
///
/// Despite `words_splitter_type: "whitespace"`, the reference implementation is
/// a regex, and matching it is mandatory — a different word count shifts every
/// span index and silently corrupts the scores:
///
/// ```text
/// \w+(?:[-_]\w+)*|\S
/// ```
///
/// Two alternatives: a word optionally continued by `-` or `_` joined words, or
/// any single non-whitespace character. So `Sekondi-Takoradi` is deliberately
/// ONE unit (which is why the hyphen that defeated the BIO tagger never becomes
/// a decision point here), while a trailing `.` becomes its own.
///
/// Expected for the fixture sentence, 8 units:
///   ["Kwame", "Nkrumah", "founded", "Osagyefo", "Studios", "in",
///    "Sekondi-Takoradi", "."]
fn split_words(text: &str) -> Vec<Word> {
    regex::regex!(r"\w+(?:[-_]\w+)*|\S")
        .find_iter(text)
        .map(|m| Word {
            text: m.as_str().into(),
            start: m.start(),
            end: m.end(),
        })
        .collect()
}

/// Ground truth captured from the reference Python implementation, so that a
/// mismatch surfaces at *construction* time rather than as a mysteriously wrong
/// score after inference.
///
/// Every tensor here is batch-wrapped (outer length 1) exactly as the model
/// expects it, which is why each field is a `Vec<Vec<_>>`.
#[derive(serde::Deserialize)]
struct Fixture {
    text: String,
    labels: Vec<String>,
    words: Vec<String>,
    input_ids: Vec<Vec<i64>>,
    attention_mask: Vec<Vec<i64>>,
    words_mask: Vec<Vec<i64>>,
    text_lengths: Vec<Vec<i64>>,
    span_idx: Vec<Vec<[i64; 2]>>,
    span_mask: Vec<Vec<bool>>,
    logits_shape: Vec<usize>,
    expected: Vec<Expected>,
}

#[derive(serde::Deserialize, Debug)]
struct Expected {
    text: String,
    label: String,
    score: f32,
}

/// Compare one constructed tensor against the fixture, reporting the first
/// divergence with its index — a bare `assert_eq!` on two 96-element vectors
/// prints both in full and leaves you to diff them by eye.
fn check<T: PartialEq + std::fmt::Debug>(name: &str, got: &[T], want: &[T]) -> Result<()> {
    if got.len() != want.len() {
        return Err(anyhow!(
            "{name}: length {} != expected {}",
            got.len(),
            want.len()
        ));
    }
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g != w {
            return Err(anyhow!("{name}[{i}]: {g:?} != expected {w:?}"));
        }
    }
    println!("  ok  {name} ({} elements)", got.len());
    Ok(())
}

/// Enumerate every candidate span as `[start_word, end_word]`, **inclusive**.
///
/// Layout is all `MAX_WIDTH` widths for word 0, then all for word 1, and so on,
/// so flat index `s * MAX_WIDTH + w` lines up with `logits[s][w]`. Spans that
/// run past the end of the sentence are still emitted — `span_mask` culls them.
fn span_indices(num_words: usize) -> (Vec<[i64; 2]>, Vec<bool>) {
    let mut idx = Vec::with_capacity(num_words * MAX_WIDTH);
    let mut mask = Vec::with_capacity(num_words * MAX_WIDTH);

    for start in 0..num_words {
        for width in 0..MAX_WIDTH {
            let end = start + width;
            idx.push([start as i64, end as i64]);
            mask.push(end < num_words);
        }
    }

    (idx, mask)
}

/// Build the token sequence GLiNER expects, plus the mask that maps token
/// positions back to word indices.
///
/// The sequence is assembled by hand rather than by a single `encode()` call:
///
/// ```text
/// [CLS] <<ENT>> person <<ENT>> organization <<ENT>> location <<SEP>> Kwame N krum ah ... [SEP]
///   0      0      0       0         0          0       0        0      1   2  0   0      0     <- words_mask
/// ```
///
/// `words_mask` is **1-based** and marks only the *first* subword of each word;
/// every other position is 0. The model uses it to scatter-gather 24 token
/// positions down to 8 word representations — the same subword-to-word merge
/// that `main.rs` performs by hand with `starts_with("##")`, except here it is a
/// tensor the graph consumes rather than decode logic that runs afterward.
///
/// Returns `(input_ids, words_mask)`.
fn encode(tokenizer: &Tokenizer, labels: &[&str], words: &[Word]) -> Result<(Vec<i64>, Vec<i64>)> {
    let mut bracketed_labels: Vec<&str> = Vec::from(labels)
        .into_iter()
        .intersperse("<<ENT>>")
        .collect();
    bracketed_labels.insert(0, "<<ENT>>");
    bracketed_labels.push("<<SEP>>");

    let tokens: Vec<&str> = bracketed_labels
        .into_iter()
        .chain(words.iter().map(|w| w.text.as_str()))
        .collect();

    let encoding = tokenizer.encode(tokens, true).map_err(|e| anyhow!(e))?;

    // `<<ENT>>` before each label, plus a trailing `<<SEP>>`: the elements of
    // `tokens` that precede the sentence itself.
    let preamble = labels.len() * 2 + 1;

    let mut words_mask: Vec<i64> = Vec::with_capacity(encoding.len());
    // TODO(human): emit one value per token position from `encoding.get_word_ids()`.
    // 0 for `None`, 0 for preamble elements, 0 for repeated ids, and
    // `id - preamble + 1` for the first token of each sentence word.

    let mut active_id = -1i64;
    let word_ids = encoding.get_word_ids();

    for (idx, wid) in word_ids.iter().copied().enumerate() {
        if idx < preamble {
            words_mask.push(0);
            continue;
        }

        if let Some(id) = wid {
            let id64 = id as i64;
            let tk = id64 - preamble as i64 + 1;

            if tk == active_id {
                words_mask.push(0);
                continue;
            }

            active_id = tk;
            words_mask.push(tk);
        } else if wid.is_none() {
            words_mask.push(0);
            continue;
        }
    }

    let input_ids = encoding.get_ids().iter().map(|&o| i64::from(o)).collect();

    Ok((input_ids, words_mask))
}

fn main() -> Result<()> {
    let raw = std::fs::read_to_string("./tests/fixtures/gliner-small-v2.1.json")?;
    let fixture: Fixture = serde_json::from_str(&raw)?;

    let text = fixture.text.as_str();
    let labels = ["person", "organization", "location"];

    let words = split_words(text);
    let (span_idx, span_mask) = span_indices(words.len());

    println!("checking input construction against the fixture:");

    let word_texts: Vec<String> = words.iter().map(|w| w.text.clone()).collect();
    check("words", &word_texts, &fixture.words)?;
    check("span_idx", &span_idx, &fixture.span_idx[0])?;
    check("span_mask", &span_mask, &fixture.span_mask[0])?;
    check(
        "text_lengths",
        &[words.len() as i64],
        &fixture.text_lengths[0],
    )?;

    let tokenizer =
        Tokenizer::from_file(format!("{MODEL_DIR}/tokenizer.json")).map_err(|e| anyhow!("{e}"))?;
    let (input_ids, words_mask) = encode(&tokenizer, &labels, &words)?;

    check("input_ids", &input_ids, &fixture.input_ids[0])?;
    check("words_mask", &words_mask, &fixture.words_mask[0])?;

    let attention_mask = vec![1i64; input_ids.len()];
    check(
        "attention_mask",
        &attention_mask,
        &fixture.attention_mask[0],
    )?;

    println!(
        "\nall six inputs match; expected output {:?}",
        fixture.logits_shape
    );
    for e in &fixture.expected {
        println!("  {} \t{} \t{:.3}", e.text, e.label, e.score);
    }

    Ok(())
}
