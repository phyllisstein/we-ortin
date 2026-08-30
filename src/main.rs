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
        for (position, lane) in sentence.axis_iter(Axis(0)).enumerate() {
            // Skip padded positions: the model emitted logits for them, but they
            // stand for nothing in the source text.
            if encoding.get_attention_mask()[position] == 0 {
                continue;
            }

            let token = &encoding.get_tokens()[position];
            let label: String = match argmax(lane) {
                Some(id) => id2label.get(&id).unwrap().into(),
                None => "undefined".into(),
            };

            println!("\t{token}\t{label}");
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
