mod sem;

use std::collections::BTreeMap;

use anyhow::Result;
use ndarray::prelude::*;
use ndarray::s;
use ort::{session::Session, value::TensorRef};
use serde_json;
use tokenizers::{Encoding, PaddingParams, Tokenizer};
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

/// Flatten one column of a batch of encodings into the contiguous `i64` buffer
/// that a `[batch, sequence]` tensor expects.
///
/// `Encoding` stores `ids`, `attention_mask`, and `type_ids` as parallel arrays
/// of equal length (a struct-of-arrays), so `column` selects which one to read.
fn flatten(encodings: &[Encoding], column: impl Fn(&Encoding) -> &[u32]) -> Vec<i64> {
    encodings
        .iter()
        .map(column)
        .flatten()
        .map(|&e| i64::from(e))
        .collect()
}

/// Pick the winning label for one token position.
///
/// `lane` is the `[labels]` slice of logits the model emitted for a single
/// token — raw scores, not probabilities, one per entry in `config.json`'s
/// `id2label`.
fn argmax(lane: ArrayView1<f32>) -> Option<usize> {
    let out = lane
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i);

    // println!("lane: {:?}", lane);

    out
}

#[derive(serde::Deserialize)]
struct Config {
    id2label: BTreeMap<usize, String>,
}

/// One entity span, recovered from the source text rather than rebuilt from
/// WordPiece spelling.
struct Entity {
    text: String,
    kind: String,
}

/// Walk one sentence's `[sequence, labels]` logits and collect BIO spans.
///
/// Two merges happen here, and they are different problems:
///   1. subword → word: `N ##k ##rum ##ah` is one word with one tag.
///   2. word → span:    `B-LOC I-LOC` joins two words into one entity.
///
/// Byte offsets from the encoding let us slice `source` directly, so the
/// reassembled text keeps its original casing and punctuation.
fn entities(
    encoding: &Encoding,
    sentence: ArrayView2<f32>,
    id2label: &BTreeMap<usize, String>,
    source: &str,
) -> Vec<Entity> {
    let specials = encoding.get_special_tokens_mask();
    let offsets = encoding.get_offsets();
    let tokens = encoding.get_tokens();

    let mut out: Vec<Entity> = Vec::new();
    // The span currently being built: (kind, start byte, end byte).
    let mut open: Option<(String, usize, usize)> = None;

    for (position, lane) in sentence.axis_iter(Axis(0)).enumerate() {
        // `[CLS]`, `[SEP]`, and `[PAD]` stand for nothing in the source text.
        if specials[position] == 1 || encoding.get_attention_mask()[position] == 0 {
            continue;
        }

        let Some(id) = argmax(lane) else { continue };
        let label = &id2label[&id];
        let (start, end) = offsets[position];
        let continues_word = tokens[position].starts_with("##");
        let _fragment = &source[start..end];

        if continues_word {
            if let Some((kind, running_start, _)) = open {
                open = Some((kind, running_start, end));
            }
            continue;
        }

        if label.starts_with("O") {
            if let Some((kind, running_start, running_end)) = open {
                let text = &source[running_start..running_end];
                let entity = Entity {
                    kind,
                    text: text.into(),
                };
                out.push(entity);
                open = None;
            };
            continue;
        }

        if label.starts_with("B") {
            if let Some((kind, running_start, running_end)) = open {
                let text = &source[running_start..running_end];
                let entity = Entity {
                    kind,
                    text: text.into(),
                };
                out.push(entity);
            };

            let entity_kind = label.strip_prefix("B-").unwrap();
            open = Some((entity_kind.into(), start, end));
            continue;
        }

        if label.starts_with("I") {
            let entity_kind = label.strip_prefix("I-").unwrap();

            match open {
                // The classifier reported that we're inside an entity, but no
                // entity was previously extracted. Open a new span.
                None => {
                    open = Some((entity_kind.into(), start, end));
                    continue;
                }
                Some((running_entity, running_start, _)) => {
                    open = Some((running_entity, running_start, end));
                }
            };
        }
    }
    out
}

fn main() -> Result<()> {
    // Tracing goes to stderr so structured logs don't mix with any stdout
    // output (e.g. health-check scripts that parse the server's stdout).
    tracing_subscriber::registry()
        .with(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ort=debug,ort_with_claude=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
        .init();

    let mut session = Session::builder()?.commit_from_file("./onnx/bert-base-NER/model.onnx")?;

    let inputs = vec![
        "My name is Wolfgang and I live in Berlin",
        "Kwame Nkrumah founded Osagyefo Studios in Sekondi-Takoradi.",
    ];

    let mut tokenizer = Tokenizer::from_file("./onnx/bert-base-NER/tokenizer.json")
        .map_err(|e| anyhow::anyhow!(e))?;
    tokenizer.with_padding(Some(PaddingParams::default()));

    let encodings = tokenizer
        .encode_batch(inputs.clone(), true)
        .map_err(|e| anyhow::anyhow!(e))?;

    let ids = flatten(&encodings, Encoding::get_ids);
    let masks = flatten(&encodings, Encoding::get_attention_mask);
    let token_types = flatten(&encodings, Encoding::get_type_ids);

    let padded_token_length = ids.len() / inputs.len();

    let a_ids = TensorRef::from_array_view(([inputs.len(), padded_token_length], &*ids))?;
    let a_masks = TensorRef::from_array_view(([inputs.len(), padded_token_length], &*masks))?;
    let a_token_types =
        TensorRef::from_array_view(([inputs.len(), padded_token_length], &*token_types))?;
    let outputs = session.run(ort::inputs![
        "input_ids" => a_ids,
        "attention_mask" => a_masks,
        "token_type_ids" => a_token_types,
    ])?;

    // `[batch, sequence, labels]` — one 9-wide lane of logits per token position.
    let logits = outputs
        .get("logits")
        .unwrap()
        .try_extract_array::<f32>()?
        .into_dimensionality::<Ix3>()
        .unwrap();

    let config_string = &std::fs::read_to_string("./onnx/bert-base-NER/config.json")?;
    let cfg: Config = serde_json::from_str(config_string)?;
    let id2label = cfg.id2label;

    for (i, encoding) in encodings.iter().enumerate() {
        // `[sequence, labels]` for this one sentence — a zero-copy view.
        let sentence = logits.index_axis(Axis(0), i);

        println!("\n{}", inputs[i]);
        for entity in entities(encoding, sentence, &id2label, inputs[i]) {
            println!("\t{}\t{}", entity.text, entity.kind);
        }
    }

    Ok(())
}

fn print_metadata(session: &mut Session) -> Result<()> {
    let meta = session.metadata()?;
    if let Some(x) = meta.name() {
        println!("Name: {x}");
    }
    if let Some(x) = meta.description() {
        println!("Description: {x}");
    }
    if let Some(x) = meta.producer() {
        println!("Produced by {x}");
    }

    if let Ok(custom_keys) = meta.custom_keys()
        && !custom_keys.is_empty()
    {
        println!("=== Custom keys ===");
        for key in custom_keys {
            if let Some(value) = meta.custom(&key) {
                println!("    {key}: {value}");
            }
        }
    };

    println!("=== Inputs ===");
    for (i, input) in session.inputs().iter().enumerate() {
        println!("    {i} {}: {}", input.name(), input.dtype());
    }
    println!("=== Outputs ===");
    for (i, output) in session.outputs().iter().enumerate() {
        println!("    {i} {}: {}", output.name(), output.dtype());
    }
    let metadata = session.metadata().unwrap();
    let name = session.metadata()?.name().unwrap();
    let inputs = session.inputs();
    let outputs = session.outputs();

    println!("bert/base-NER");
    println!("\tname\t\t{:?}", name);
    println!("\tinputs\t\t{:?}", inputs.len());
    println!("\toutputs\t\t{:?}", outputs.len());
    println!("\tcustom keys\t{:?}", metadata.custom_keys()?);

    println!("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Inputs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
    for input in inputs.iter() {
        let input_name = input.name();
        let dtype = input.dtype();

        let dimension_symbols = dtype.tensor_shape().unwrap();

        println!("\n{input_name}");
        match dtype {
            ort::value::ValueType::Tensor {
                ty,
                shape,
                dimension_symbols,
            } => {
                let msg = format!("{:?}", ty);
                println!("\tdtype\t{:?}", dimension_symbols);
                println!("\tdtype\t{:?}", msg);
            }
            _ => {
                tracing::error!("Unexpected dtype")
            }
        };
        println!("\tdimension_symbols\t{:?}", dimension_symbols);
    }

    println!("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Outputs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
    for output in outputs.iter() {
        let output_name = output.name();
        let dtype = output.dtype();
        let dimension_symbols = dtype.tensor_shape().unwrap();
        println!("{output_name}");
        println!("\tdtype\t{:?}", dtype);
        println!("\tdimension_symbols\t{:?}", dimension_symbols);
    }

    Ok(())
}
