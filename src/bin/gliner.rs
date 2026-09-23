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

use anyhow::{Result, anyhow};
use ndarray::Ix4;
use ort::session::Session;
use ort::value::TensorRef;
use tokenizers::Tokenizer;

/// Minimum sigmoid score for a (span, type) pair to be considered at all.
/// GLiNER's reference decoder default.
const SCORE_THRESHOLD: f32 = 0.5;

/// A scored candidate entity, before overlap resolution.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    start_word: usize,
    end_word: usize,
    label: usize,
    score: f32,
}

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

    // Emit one value per token position from `encoding.get_word_ids()`: 0 for
    // `None`, 0 for preamble elements, 0 for repeated ids, and
    // `id - preamble + 1` for the first token of each sentence word.
    let words_mask: Vec<i64> = encoding
        .get_word_ids()
        .iter()
        .copied()
        .scan(None::<i64>, |active_id, word_id| {
            let tk = word_id
                .map(|id| id as i64)
                .filter(|&id| id >= preamble as i64)
                .map(|id| id - preamble as i64 + 1)
                .filter(|&tk| Some(tk) != *active_id);

            if let Some(tk) = tk {
                *active_id = Some(tk);
            }

            Some(tk.unwrap_or(0))
        })
        .collect();

    let input_ids = encoding.get_ids().iter().map(|&o| i64::from(o)).collect();

    Ok((input_ids, words_mask))
}

/// Sigmoid activation: GLiNER scores each (span, type) pair independently —
/// a span can plausibly match no label or, in principle, more than one — so
/// there's no softmax-style competition *across types* the way `bert-base-NER`
/// competes across BIO tags. Only overlap resolution (below) makes spans
/// compete, and it competes on (start, end) alone, blind to type.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Walk every `(start_word, width, type)` triple in `logits`, applying sigmoid
/// and keeping only those above [`SCORE_THRESHOLD`] whose span survives
/// `span_mask` (i.e. doesn't run past the sentence).
///
/// `logits` and `span_mask` share the same flat `(start_word, width)` grid —
/// `span_mask[s * MAX_WIDTH + w]` gates `logits[0][s][w][_]` — which is exactly
/// the correspondence `span_indices`'s doc comment sets up.
fn gather_candidates(logits: &ndarray::ArrayView4<f32>, span_mask: &[bool]) -> Vec<Candidate> {
    let (_, num_words, num_widths, num_types) = logits.dim();
    let mut candidates = Vec::new();

    for start_word in 0..num_words {
        for width in 0..num_widths {
            let flat = start_word * num_widths + width;
            if !span_mask[flat] {
                continue;
            }
            let end_word = start_word + width;
            for label in 0..num_types {
                let score = sigmoid(logits[[0, start_word, width, label]]);
                if score > SCORE_THRESHOLD {
                    candidates.push(Candidate {
                        start_word,
                        end_word,
                        label,
                        score,
                    });
                }
            }
        }
    }

    candidates
}

/// Resolve overlapping candidates to GLiNER's final entity list.
///
/// GLiNER's reference decoder treats this as greedy non-max suppression over
/// word ranges: sort every above-threshold `(span, type)` candidate by score
/// descending, then walk the list keeping a candidate only if its word range
/// doesn't intersect any range already kept. A span that overlaps a
/// higher-scoring keeper is dropped outright — even a different label doesn't
/// save it, since the check is on `(start_word, end_word)` alone.
///
/// The winners are re-sorted by `start_word` at the end, so entities print in
/// reading order (the reference decoder does the same).
fn resolve_overlaps(mut candidates: Vec<Candidate>) -> Vec<Candidate> {
    // `total_cmp` rather than `partial_cmp().unwrap()`: a total order on f32
    // (NaN included), so there's no panic path to reason about.
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));

    // Plain data massage. Every candidate is checked against *all* keepers,
    // not just the last one pushed: after sorting by score, keepers can be
    // anywhere in the sentence, so `acc.last()` has no special meaning.
    // Ranges are inclusive, so two spans are disjoint iff one ends strictly
    // before the other starts. An identical range with a different label
    // counts as overlapping, so each span gets at most one label.
    let mut kept: Vec<Candidate> = Vec::new();
    for c in candidates {
        let overlaps = kept
            .iter()
            .any(|k| !(c.end_word < k.start_word || k.end_word < c.start_word));
        if !overlaps {
            kept.push(c);
        }
    }

    kept.sort_by_key(|c| c.start_word);
    kept
}

fn main() -> Result<()> {
    let raw = std::fs::read_to_string("./tests/fixtures/gliner-small-v2.1.json")?;
    let fixture: Fixture = serde_json::from_str(&raw)?;

    let text = fixture.text.as_str();
    let labels: Vec<&str> = fixture.labels.iter().map(String::as_str).collect();

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

    // Batch size is always 1 here; every tensor's leading dim is that batch.
    let seq_len = input_ids.len();
    let num_spans = span_idx.len();
    let text_lengths_val = [words.len() as i64];

    let a_input_ids = TensorRef::from_array_view(([1, seq_len], &*input_ids))?;
    let a_attention_mask = TensorRef::from_array_view(([1, seq_len], &*attention_mask))?;
    let a_words_mask = TensorRef::from_array_view(([1, seq_len], &*words_mask))?;
    let a_text_lengths = TensorRef::from_array_view(([1, 1], &text_lengths_val[..]))?;
    let a_span_mask = TensorRef::from_array_view(([1, num_spans], &*span_mask))?;

    // build `a_span_idx`, a TensorRef over `span_idx: Vec<[i64; 2]>`
    let idx = span_idx.into_flattened();
    let a_span_idx = TensorRef::from_array_view(([1, num_spans, 2], &*idx))?;

    let mut session = Session::builder()?.commit_from_file(format!("{MODEL_DIR}/model.onnx"))?;

    let outputs = session.run(ort::inputs![
        "input_ids" => a_input_ids,
        "attention_mask" => a_attention_mask,
        "words_mask" => a_words_mask,
        "text_lengths" => a_text_lengths,
        "span_idx" => a_span_idx,
        "span_mask" => a_span_mask,
    ])?;

    // `[batch, words, width, types]`: 8 words × 12 widths × 3 types = 288 scores
    //
    // "Why logits's last dimension is 3: it's not num_types compiled into the
    // graph, it's num_types as it fell out of this specific input's label
    // count."
    //
    //
    // ""That's also, not coincidentally, most of why interpretability research on
    // transformers is hard — 'where is the fact stored' doesn't have a stable
    // answer when storage and computation are the same operation, running on
    // data that's mixed in with the query."
    let logits = outputs
        .get("logits")
        .ok_or_else(|| anyhow!("model produced no `logits` output"))?
        .try_extract_array::<f32>()?
        .into_dimensionality::<Ix4>()?;

    println!("\nlogits shape: {:?}", logits.shape());

    let candidates = gather_candidates(&logits.view(), &span_mask);
    let entities = resolve_overlaps(candidates);

    println!("\ndecoded entities:");
    for c in &entities {
        let text_span = &text[words[c.start_word].start..words[c.end_word].end];
        println!("  {text_span} \t{} \t{:.3}", labels[c.label], c.score);
    }

    Ok(())
}
