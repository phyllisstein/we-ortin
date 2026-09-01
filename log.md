# Learning log

Running notes. Newest at the top.

---
## 2026-09-01 — `bert-base-NER` end to end: tensors in, entities out

First complete pipeline. Two sentences → tokenizer → three `[batch, sequence]`
int64 tensors → `session.run` → `[batch, sequence, labels]` logits → decoded
BIO spans. Everything below was learned by breaking it.

### The theme: the tokenizer's configuration is part of the model's contract

This bit three times, and the pattern is worth naming. A tokenizer is not a
generic text-to-numbers function. `tokenizer.json` encodes decisions that the
*model weights were trained under*, and copying inference code between models
transplants assumptions that do not survive the move.

**Failure 1 — loud.** `encode_batch` returned ragged encodings (9 and 23
tokens), so no rectangle described them:

```
shape [2, 9] (18 elements) is different from the length of the data provided (32 elements)
```

`bert-base-NER`'s `tokenizer.json` has `padding: null`. Through serde that
becomes `Tokenizer.padding = None`, and `encode_batch` then *batches without
padding*. The upstream `ort` MiniLM example carries a comment asserting that
`encode_batch` pads — true only when the tokenizer's own padding config is
populated. Fix: `tokenizer.with_padding(Some(PaddingParams::default()))`
(default strategy is already `BatchLongest`).

**Failure 2 — silent, and the dangerous one.** `encode_batch(inputs, false)`
sets `add_special_tokens: false`, so no `[CLS]`/`[SEP]`. Every token in both
sentences then decoded as `O`. Shapes correct, dtypes correct, masks correct,
no panic, no warning — just uniformly plausible wrong answers.

*Why* it collapses is the good part. Softmax forces attention mass to sum to 1
with no option to abstain, so heads with nothing useful to attend to dump their
mass onto the semantically empty `[CLS]`/`[SEP]` positions — **attention
sinks**. Clark et al. (2019) found >50% of attention in many heads lands on
`[SEP]`. Remove them and that mass redistributes onto real tokens, shifting
every hidden state off the manifold the classification head was fit to. The
head collapses toward the majority class, which in NER is `O` (~80–90% of real
tokens).

The loud failure is the lucky one.

### Tensors in, tensors out

The inputs are tensors too, not just the outputs — `input_ids`,
`attention_mask`, `token_type_ids`, all `[batch, sequence]` int64. Flowing
tensors the whole way down.

`input_ids` is the one stream in the forward pass that is **coordinates, not
values**: row numbers into a 30,522-line `vocab.txt`, consumed by ONNX's
`Gather` op. Arithmetic on them is meaningless. Every other tensor holds
quantities.

`attention_mask` does double duty. Going in, it becomes an additive `0` /
`-10000` bias on attention *scores* before softmax, so `exp(-10000) ≈ 0` and
padded positions contribute nothing. Coming out, it is the filter marking which
positions were real.

`token_type_ids` is BERT's segment embedding, left over from next-sentence-
prediction pretraining: all zeros for a single sentence, `0`s then `1`s for a
pair. ONNX Runtime requires *every* declared input bound — there are no
defaults — so it must be supplied even when it is entirely zeros.

Contrast worth keeping, since crossing domains is the point of this repo:

| | inputs | outputs |
| --- | --- | --- |
| `bert-base-NER` | 3 (`input_ids`, `attention_mask`, `token_type_ids`) | 1 (`logits`) |
| `all-MiniLM-L6-v2` | 2 (no `token_type_ids` — the export dropped it) | 2 (`last_hidden_state` + pooled) |

Same `ort` surface, different contract. The plumbing transfers; the contract
does not.

### Flat buffer + shape tuple

`TensorRef::from_array_view((shape, &[T]))` — the buffer holds values, the shape
holds the interpretation. A flat buffer can only address a *rectangle*, which is
why padding is mandatory rather than a convenience. `[2,16,9]` has strides
`[144, 9, 1]`; element `[i][j][k]` lives at `i*144 + j*9 + k`. Slicing and
transposing are stride manipulations, zero-copy.

Corollary that cost a debugging round: `.iter()` on an `Array3` walks every
scalar (414 of them), not rows. `index_axis` / `axis_iter` walk *structure*. A
`zip(masks)` against `.iter()` silently paired logit-scalars with unrelated mask
values. Collapse `[2,23,9] → [2,23]` with argmax first, and then the mask lines
up shape-for-shape.

### Decoding logits

The exported graph ends at logits — softmax was deliberately left out, because
argmax is monotonic under softmax and the probabilities are never needed.

`f32` is not `Ord`, only `PartialOrd`: NaN breaks both totality and
reflexivity, so `max_by_key` will not compile. Use
`max_by(|a, b| a.total_cmp(b))` — IEEE 754 totalOrder, cannot panic — rather
than `partial_cmp().unwrap()`. Note `max_by` returns the **last** maximum on
ties while `min_by` returns the first.

Label names come from `config.json`'s `id2label`. Deserializing it as
`BTreeMap<usize, String>` works even though JSON object keys are always strings,
because serde's integer deserializers accept the string form *in key position*.
`BTreeMap` iterates in key order, so `.into_values().collect()` would give a
dense `Vec<String>` if the ordering is ever needed.

One misread worth recording: `"0": "O"` is the **letter O**, for *Outside*, not
the digit zero. Slicing it off with `s![.., 1..]` made every lane 8 wide, so
argmax returned 0–7 meaning labels 1–8 — every tag silently shifted by one. It
*looked* better only because removing `O` forces every token to claim some
entity type.

### Two merges, one traversal

`entities()` reconstructs spans from per-token tags. Two distinct concerns, both
run-detection over the *same* sequence axis — they are not two ndarray axes, and
`Axis(1)` (labels) is already collapsed by argmax before any of this runs:

1. **subword → word.** `Nkrumah` tokenizes to `N ##k ##rum ##ah`. Four
   positions, four independent tags, one word. Under HuggingFace's `first`
   strategy a `##` token never makes a decision — it just extends whatever the
   word's first subword decided. This check must come *before* the BIO dispatch,
   since the `##` token's own label is noise being deliberately discarded.
2. **word → span.** `B-ORG I-ORG` joins `Osagyefo` and `Studios` into one
   entity.

The test sentence exercises three distinct paths:

| span | merged by |
| --- | --- |
| `Kwame Nkrumah` | `I-PER` **and** `##` continuation |
| `Osagyefo Studios` | `I-ORG` only |
| `Sekondi` / `Takoradi` | neither — see below |

Missing the `I-` branch entirely still produced `Kwame Nkrumah` correctly, via
the `##` path, which is exactly the kind of coincidence that hides a bug.
`Osagyefo Studios` — two whole words, no `##` anywhere — is what exposed it.

Use `Encoding::get_offsets()` (byte offsets into the original string) rather
than reassembling WordPiece spelling. Slicing `source[start..end]` preserves
casing and punctuation for free; a detokenizer would have to *guess* whether to
put spaces around a hyphen.

### `Sekondi-Takoradi`: not a decode bug

The model emits `Sekondi → B-LOC`, `- → O`, `Ta → B-LOC`. Two separate `B-LOC`
spans is the correct reading of that tag sequence under any BIO decoder,
HuggingFace's included. The decode is faithful; the *prediction* is what splits.

Why: this is a token-classification head with **no CRF layer**, so every
position argmaxes independently and nothing enforces that a well-formed BIO
sequence comes out. `B-LOC O B-LOC` inside one orthographic word is not
forbidden by anything.

> **CRF (Conditional Random Field).** A layer bolted onto the *output* of a
> token classifier to make tag decisions depend on each other. It adds a learned
> **transition matrix** — 9×9 for this label set — where `T[a][b]` scores how
> plausible tag `b` is immediately after tag `a`. You then stop arg-maxing each
> position alone and instead score whole tag *sequences*:
>
> ```
> score(tags) = Σᵢ emission[i][tagᵢ] + Σᵢ transition[tagᵢ₋₁][tagᵢ]
> ```
>
> The emissions are the existing logits; the transitions are new parameters.
> Finding the best sequence looks like it needs 9²³ ≈ 10²¹ evaluations, but
> **Viterbi** does it in O(n·k²) — 23 × 81 steps — because the score decomposes
> into terms spanning only two adjacent positions, so the best path *ending in
> tag b at position i* depends only on the best path ending in each tag at
> `i-1`. Same dynamic program as the forward algorithm in HMMs.
>
> Nobody hand-writes BIO's grammar: fine-tuning discovers that `T[O][I-LOC]`
> should be strongly negative (you cannot be Inside a location you never Began)
> and that `T[B-PER][I-PER]` is favorable.
>
> A CRF would not automatically fix `Sekondi-Takoradi`, but it changes the
> calculus — `B-LOC O B-LOC` would pay a transition penalty that
> `B-LOC I-LOC I-LOC` does not, so the model would have to be *confident* about
> the `O` to keep it, not merely ahead by a hair.
>
> Why `bert-base-NER` has none, two separate reasons: (1) CRFs were near
> universal atop BiLSTMs (2015–2018) because an LSTM's state at position *i* is
> genuinely weak on what was tagged at `i-1`; self-attention already gives every
> position a view of the whole sequence, so the emissions carry much of that
> consistency themselves and the measured gain over BERT is typically under a
> point of F1. (2) Viterbi is a loop with data-dependent control flow, awkward
> to express as ONNX ops, so exports routinely drop it and leave decoding to the
> host — the recurring shape of this whole exercise: **the graph gives you
> emissions; structured decoding is your problem.**

Punctuation is overwhelmingly `O` in CoNLL-2003, a
lexical prior strong enough to override span context; that `O` breaks the
contextual frame, so `Ta` re-decides from scratch and its prior for a
capitalized unknown in location context is `B-LOC`.

Related, and encouraging: *Osagyefo Studios* is fictional and neither word is
in-vocab as a unit, yet it still classified as `ORG`. The signal is the
syntactic frame `founded X in Y`, not memorized surface forms. That is the
whole value of a fine-tuned transformer over a gazetteer.

Merging across the hyphen would be a post-hoc heuristic layered on top — a
legitimate production move, but it lives strictly *outside* the graph.

### Rust notes

- **`as` is a real conversion**, not TypeScript's erased assertion: it emits a
  zero/sign-extension instruction and doubles the memory footprint. Prefer
  `i64::from(x)` for widening — it only compiles when provably lossless.
- **`&*ids`** converts `Vec<i64> → &[i64]`: `*` names the `Deref::Target` (the
  unsized `[i64]`) so `&` can bind to it. Needed because `from_array_view` is
  generic — there is no concrete target type for deref coercion to fire toward.
- **`&e` in a pattern** *removes* a reference, because patterns mirror
  constructors: `&` in an expression adds, `&` in a pattern strips. (`e` is
  already the address; `*e` follows it to read the value.)
- **`Default::default()`** in struct-update position creates an inference
  variable `?S: Default`, defers trait selection, and lets the *use site*
  constrain it. Information flows outward-in.
- **`Option<T>` vs `T::default()`**: `Option` answers "do you want this at all";
  `Default` answers "you want it but don't care how." `padding: null` ↔ `None`
  round-trips through serde with no attribute.
- **rustc suggestion calibration**: borrow-checker suggestions are purely
  syntactic lifetime fixes. `MachineApplicable` means "this compiles," not
  "this is what you meant." Its fix for the `E0716` on chained
  `from_file(...)?.with_padding(...)` introduced a binding rather than pointing
  at the actual issue, which is that `with_padding` returns `&mut Self` and so
  cannot terminate an expression.
- Filtering padding **first** beats filtering last. Both are safe — padded
  logits are well-formed `f32`s, nothing crashes either way — but filtering
  last is a screen at the end of the pipe, and every intermediate stage still
  saw the padding. Filtering first makes bad states unrepresentable.

---
## 2026-08-27 — Aside: a second model exported (`embeddinggemma-export`)

### A Python dependency conflict, and why Cargo doesn't have them

`optimum-cli export onnx -m Qwen/Qwen3-Embedding-4B` died with:

```
ImportError: cannot import name 'get_parameter_dtype' from 'transformers.modeling_utils'
```

The traceback pointed at `optimum/exporters/onnx/convert.py:30`, but the real
cause was two packages further out.

**What the environment actually contained:**

| package | required `transformers` |
| --- | --- |
| `sentence-transformers 6.0.0` | `>=5.0.0, <6.0.0` |
| `optimum-onnx 0.1.0` | `>=4.36, <4.58.0` |

Those windows do not overlap. There is no `transformers` version that satisfies
both, yet both were installed — which means they went in via separate
`uv pip install` commands. A second install does not re-resolve the environment;
it upgrades a shared dependency straight past the first package's ceiling.
`transformers 5.x` had renamed `get_parameter_dtype` to `_get_dtype`, so
`optimum-onnx`'s import found nothing.

**The fix** was to notice that `sentence-transformers` was never needed. Nothing
depended on it, and it is a *PyTorch* inference library — but the whole point
here is to export to ONNX and run inference in Rust via `ort`. It had been
installed reflexively because Qwen3-Embedding is a sentence-similarity model.
Removing it, then pinning `transformers>=4.51,<4.58` (4.51 is where Qwen3
architecture support landed), resolved everything. `uv` correctly cascaded the
downgrade to `tokenizers` and `huggingface-hub`, which are version-locked to the
`transformers` major.

**The transferable lesson.** Python's `site-packages` is a *flat namespace*:
exactly one version of a package name can exist at a time. So a violated version
bound is not a duplicated dependency, it is an unsatisfiable conflict, and the
failure surfaces at import time in an unrelated file.

Cargo makes this class of bug structurally impossible. `Cargo.lock` resolves the
entire graph in one pass, and semver-incompatible versions of the same crate are
*both* compiled in, with symbols kept distinct. Two crates wanting incompatible
versions of a shared dependency is a non-event. That is the same guarantee that
made pinning `ort`/`ort-sys` viable earlier.

A secondary contributor: this venv has no `pyproject.toml`, `requirements.txt`,
or `uv.lock`. Nothing recorded intent, so nothing could detect the violation.

Two lazy-loading details worth keeping:

- `transformers` uses a `_LazyModule` shim, so `__getattr__` triggers the real
  import. That is why an `ImportError` about `transformers` appeared inside a
  line that read `from optimum.exporters.onnx import main_export`.
- `optimum 2.x` split the ONNX exporter into a separate distribution,
  `optimum-onnx`, while keeping the `optimum.exporters.onnx` *import path*. Two
  distributions writing into one namespace package makes `pip list` output a
  poor guide to which code is actually on disk. `importlib.metadata.files()`
  answers the ownership question directly.

---

### Exporting EmbeddingGemma

Exported `google/embeddinggemma-300m` (Gemma3 encoder, 300M params) alongside
the existing BERT NER model. The point of the exercise was to see which parts
of the tooling transfer and which parts are per-model.

### The export

```
optimum-cli export onnx -m google/embeddinggemma-300m embeddinggemma_300m \
  --task feature-extraction --opset 17
```

**There is no `sentence-similarity` ONNX task.** The valid task for an embedding
model is `feature-extraction`, and the reason is exactly the boundary noted in
Step 1: the graph does not compute similarity. It emits per-token hidden states.
Pooling and cosine similarity are the caller's job.

Discovering the valid task list has a trap. `TasksManager._SUPPORTED_MODEL_TYPE`
is **empty until `optimum.exporters.onnx.model_configs` is imported** — each
`OnnxConfig` subclass registers itself via decorator at class-definition time,
so the registry is populated by import side-effect. Query it too early and every
architecture reports `not supported yet ... Only [] are supported`, which reads
like missing support rather than an unpopulated table. Import first, then query:

```python
import optimum.exporters.onnx.model_configs
from optimum.exporters.tasks import TasksManager as T
T.get_supported_tasks_for_model_type("gemma3_text", "onnx", library_name="transformers")
# -> ['feature-extraction', 'feature-extraction-with-past',
#     'text-generation', 'text-generation-with-past']
```

The `-with-past` variants take KV-cache tensors as extra inputs, for
autoregressive generation. Embedding encodes the whole sequence in one pass, so
plain `feature-extraction` is correct.

### The architecture, and where the parameters went

`hidden_size` is 768 — identical to BERT-base. Everything else diverges:

- `vocab_size: 262144` vs BERT's 28,996. The embedding matrix alone is
  262144 x 768 ~= 201M params: **two-thirds of the entire model**.
- `intermediate_size: 1152` is only 1.5x hidden, where BERT uses 4x (3072).

Gemma spent its budget on vocabulary coverage rather than per-layer computation.

- `use_bidirectional_attention: true` is the key tell. Gemma3 is normally a
  *causal* decoder; EmbeddingGemma flips every token to attend over the full
  sequence, which is what makes it an encoder suitable for embeddings.
- `layer_types` + `sliding_window: 512` — Gemma3 interleaves local
  sliding-window attention with global layers. It survived export fine.

### Numerical validation is evidence, not an obstacle

The export warned:

```
values not close enough, max diff: 0.0004487 (atol: 1e-05)
```

This is expected. ONNX fuses and reorders operations, float addition is not
associative, and the error compounds across 24 layers. Against
`last_hidden_state` values of order ~1 that is ~1 part in 2000, far below
anything that reorders cosine rankings.

The discipline worth keeping: **read the number before loosening `--atol`.**
4e-4 is float reassociation. 0.5 would be a broken graph. The flag that silences
the warning also silences the diagnostic.

### The contract comparison — the actual lesson

|  | BERT NER | EmbeddingGemma |
| --- | --- | --- |
| inputs | `input_ids`, `attention_mask`, `token_type_ids` | `input_ids`, `attention_mask` |
| output | `[batch, seq, 9]` logits | `[batch, seq, 768]` hidden states |
| post-processing | argmax -> `id2label` | mean-pool -> L2 normalize |
| semantics live in | `config.json` `id2label` | **no exported file** |

No `token_type_ids` here: Gemma has no segment embeddings, so there is nothing
to mark. The runtime tooling is unchanged between the two models; only the
contract varies. That is the transfer.

The last row is the sharp one. For NER the missing semantics were at least
*written down* — `id2label` sits in `config.json`. For embeddings the pooling
strategy is recorded in **nothing we exported**; it lived in the
sentence-transformers `1_Pooling/` directory, which we bypassed on purpose to
keep both models on the same `transformers` code path. Choose wrong and nothing
raises — the embeddings are just quietly worse. A strictly harder failure mode
than a mislabelled entity.

(Deliberately skipped `--library-name sentence_transformers`. It bakes pooling
into the graph, which hides the mechanism and takes a different code path from
the BERT export. Also omitted from that graph: the Matryoshka dense layers that
project 768 -> 512/256/128.)

### Smoke test

Mean-pool over non-pad tokens, then L2 normalize:

```
"The cat sat on the mat."                        \  0.8312
"A feline rested upon the rug."                  /
"Quarterly earnings exceeded analyst forecasts."    0.4427 / 0.4717
```

Paraphrases at 0.83 despite sharing almost no vocabulary; unrelated pair at
~0.44. The model encodes meaning rather than surface overlap.

Two notes from the token dump:

- Tokens decode as `['<bos>', 'The', '▁cat', '▁sat', ...]`. The `▁` (U+2581)
  **is** the leading space, encoded into the token — SentencePiece treats input
  as a raw byte stream with no pre-tokenization, so whitespace needs to be a
  representable character. BERT's WordPiece instead used `##` to mark
  continuations. The `tokenizers` crate handles this, but detokenizing in Rust
  means mapping `▁` back to a space.
- `<bos>`/`<eos>` are included in the mean, since `attention_mask` is 1 there.
  That matches sentence-transformers, but it is a *choice made in pooling code*,
  not something the graph dictates. CLS-pooling or excluding sentinels yields
  different vectors from the identical `.onnx` file.

### Artefacts

`model.onnx` (1.2G fp32), `tokenizer.json` (32M — large because of the 262k
vocab), `config.json`, `tokenizer_config.json`, `special_tokens_map.json`.
Still no single-file "model": the same three-way split as Step 1.

## Next up

Run from the **package root** — `cargo run` inherits the shell's cwd, and the
model paths are relative (`./onnx/bert-base-NER/...`).

**Step 3 — tokenize and look.** `encode_batch` already runs; nothing reads the
result yet. Print, in aligned columns for both sentences: token strings, ids,
attention mask, offsets.

Two different splitting mechanisms should be visible side by side:

- `Sekondi-Takoradi` — the **pre-tokenizer** splits on punctuation, before
  WordPiece ever runs. Tokens can never span a word boundary.
- `Osagyefo` — **WordPiece** splits *inside* a word, greedy longest-match from
  the left, with `##` marking continuations. The `##` is a positional type tag,
  not decoration: those 6,477 entries are only matchable in continuation
  position.

`Wolfgang` (id 14326) and `Berlin` are both single tokens, so the first sentence
does **not** exercise the alignment problem. That's why the second sentence
exists — the 2018 Wikipedia vocab has never seen `Osagyefo`.

**Step 4 — build tensors and run.** Three inputs are required, not two;
`token_type_ids` needs an all-zeros `[N, L]` tensor (vestigial segment marker
from next-sentence-prediction pretraining). `ort::inputs![...]` binds
**positionally**, so order matters. Derive `padded_token_length` from
`encodings[0].len()` rather than asserting it.

**Step 5 — decode.** `serde_json` must be added to `Cargo.toml` explicitly to
parse `id2label`; it's currently only a transitive dep, and those aren't
importable.

**Step 6 — merge spans.** `logits` is `[batch, seq, 9]` — one prediction per
*token*, not per word, and nothing forces a split word's pieces to agree.
Offsets are the only way back to the original string.

**Candidate next step (deferred) — the second model.** `embeddinggemma_300m/`
is exported and smoke-tested (see the 2026-08-27 entry); nothing in Rust reads
it yet. Deliberately parked until Steps 3-6 finish, because the design question
it raises can't be answered from one worked example:

> NER and embeddings have *incompatible post-processing shapes*. NER wants
> per-token argmax against `id2label` and **preserves** the sequence axis.
> Embedding wants mean-pool + L2 normalize, which **destroys** it. Is pooling a
> post-processing step that belongs behind a shared trait alongside
> argmax-decoding, or different enough in kind that a common abstraction hides
> more than it shows?

That asymmetry — one path keeps `[batch, seq, *]`, the other collapses to
`[batch, *]` — is the real boundary this repo is trying to locate: where the
tooling stops transferring between domains. Answer it *after* Step 6, with two
working paths to compare rather than one plus a guess.

Also parked: pooling strategy is recorded in **no exported file** (it lived in
the sentence-transformers `1_Pooling/` dir we bypassed). Unlike `id2label`,
getting it wrong raises nothing — the embeddings are just quietly worse.

### Cleanup carried forward

- Two printing blocks do the same job. `ValueType` implements `Display`
  (`Tensor<f32>(batch_size, sequence_length, 9)`) — keep that, delete the Debug
  block.
- In the Debug block, `dimension_symbols` is bound to `dtype.tensor_shape()`,
  and two rows are both labelled `dtype`. Dies with the block.

---

## 2026-08-23 — Step 2: reading the graph's contract

The declared signature, straight from `session.inputs()` / `session.outputs()`:

```
input_ids       Tensor<i64>(batch_size, sequence_length)
attention_mask  Tensor<i64>(batch_size, sequence_length)
token_type_ids  Tensor<i64>(batch_size, sequence_length)
logits          Tensor<f32>(batch_size, sequence_length, 9)
```

Producer: `torch_jit` / pytorch. No custom metadata keys.

**Symbolic dimensions are carried in two parallel fields, not one.** `shape` is
the numeric view where `-1` means "unknown"; `dimension_symbols` is the identity
view. Neither subsumes the other — shape says *whether* a dim is free, the
symbol says *which other dims it's tied to*. All three inputs share the symbols
`batch_size` and `sequence_length`, and that shared naming is a **constraint**:
feed `input_ids` of length 8 and `attention_mask` of length 7 and the runtime
binds `sequence_length` to 8, then fails to bind it to 7.

The output's third axis has `shape: 9` and symbol `""`. **The empty string is
the tell** — fixed dims carry no symbol because there's nothing to unify. That's
ONNX's `dim_value` vs `dim_param` distinction surfacing directly in the Rust
type: the two fields are positionally zipped and exactly one is meaningful per
axis. `9` is the only genuinely architectural number here — the label count.

`ValueType` is a four-variant enum (`Tensor`, `Sequence`, `Map`, `Optional`).
`Map` exists because scikit-learn exports emit label→probability dicts via
`ZipMap`; transformers are tensors-only, so we'll never see the other three.
It also has `tensor_type()` / `tensor_shape()` / `is_tensor()` accessors that
return `Option`, which are the `match` prepackaged — and a `Display` impl that
renders the whole contract on one line.

**The wrapper pattern worth stealing:** `ValueType::from_type_info` takes an
opaque `OrtTypeInfo*`, reads a tag via `GetOnnxTypeFromTypeInfo`, then casts
with `CastTypeInfoToTensorInfo`. That's a runtime-tagged C union, checked only
by convention — cast wrongly and it's UB. `ort` does that dance exactly once, in
one `unsafe` block at session load, and hands the rest of the program a Rust
enum where the tag *is* the discriminant. The unsafety isn't eliminated, it's
**concentrated**.

### Rust asides

- `impl` is a separate item from the type definition. "Has methods" implies
  nothing about being class-shaped — the three axes (`struct`/`enum` = what data
  *is*, `impl` = what it *can do*, `trait` = what it *promises*) are orthogonal.
- `&self` really is just the first parameter, sugar for `self: &Self`. Other
  receiver types exist (`self: Rc<Self>`, `self: Pin<&mut Self>`). The only
  magic is the dot operator's auto-ref/auto-deref at the call site.
- `?` is not "unwrap or return" — it's "unwrap or return `From::from(e)`". Every
  `?` is a type-directed conversion. `ort::Error` has
  `From<Box<dyn Error + Send + Sync>>`, and `tokenizers::Error` *is* that box,
  so the upstream example's `.map_err(|e| Error::new(e.to_string()))` is both
  redundant and lossy (it discards the `source()` chain).
- To discover an unknown type fast: `let _: () = expr;` and read the mismatch.
- Prefer `i64::from(x)` over `x as i64`. `From` only exists for lossless
  conversions, so it breaks the build if the source type ever widens; `as`
  silently truncates.

### On the upstream `ort` example

`examples/sentence-transformers/semantic-similarity.rs` has a line that cannot
be reconstructed from the code: `outputs[1]`. Why 1 is unknowable without
running Step 2 against that specific model — the premise lives in the graph, not
the source. Same for `inputs![a_ids, a_mask]`, which binds **positionally**;
that model dropped `token_type_ids`, ours didn't.

Also: it computes a bare dot product and calls it cosine similarity. That's only
valid because MiniLM's export bakes in L2 normalization. **CLIP's exports
generally don't** — copy that loop into Phase 2 unchanged and similarities will
exceed 100%.

`ndarray` earns its place on the *output* side only. Input: we have a shape and
no need to manipulate it, so `TensorRef::from_array_view((shape, &slice))` is
right. Output: `try_extract_array` returns `ArrayViewD` (rank as runtime data),
and `.into_dimensionality::<Ix2>()` lifts rank into the type system — which is
what makes `index_axis` / `axis_iter` available at all.

## 2026-08-22 — Setup: Steps 0 and 1

### Step 0 — the build

`cargo build` succeeded unmodified. Two things surfaced:

**Cargo unified `ndarray`.** `cargo tree -i ndarray` shows one `ndarray v0.17.2`
with two dependents — `ort` and this crate. Because both wanted a
semver-compatible version, Cargo built it *once*, so `ort`'s `Array<f32, _>` and
ours are the same type.

Had `ort` wanted 0.16, Cargo would have built **both** versions side by side —
that's legal and intentional, since semver-incompatible majors are treated as
distinct crates. The symptom is the famously unhelpful error:

```
expected `ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>`
   found `ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>`
```

Diagnostic if it ever appears: `cargo tree -d` (duplicates).

**`ort` doesn't compile ONNX Runtime.** `ort-sys`'s build-dependencies are
`ureq` + `hmac-sha256` + `lzma-rust2` — a downloader, a checksum verifier, and a
decompressor. It fetches a prebuilt C++ binary at build time and links against
it. Hence a 6-second build for something that would take ~an hour from source.

### Step 1 — the artefacts

Downloaded fp32 `model.onnx` (412M), `tokenizer.json`, `config.json`. Commands
recorded in `readme.md`.

**The central realisation of this step:** a "model" is three detached files, and
ONNX only owns one of them.

The graph's contract is *purely numeric*: `int64` tensors in, `float32` tensors
out. It does not know that its inputs came from text, and it does not know that
output index 3 means `B-PER`. Both halves of that meaning live outside the
graph — in `tokenizer.json` and in `config.json`'s `id2label`. Serialization
formats for computation graphs deliberately stop at the tensor boundary; the
semantics are somebody else's problem.

From `config.json`:

- 9 labels, BIO scheme over 4 entity types: `O`, and B-/I- for `PER`, `ORG`,
  `LOC`, `MISC`.
- `hidden_size: 768`, `num_hidden_layers: 12` — plain BERT-base.
- `vocab_size: 28996` → this is BERT **cased**. Correct choice for NER, where
  capitalisation is a strong signal. The uncased vocab is 30522.
- The whole task-specific part of this model is one `768 -> 9` linear layer on
  top of a general encoder. Everything else is transferred.

### Open item

`serde_json` is currently only a transitive dependency. It'll need to be added
to `Cargo.toml` explicitly to parse `id2label` in Step 5 — transitive deps
aren't importable.

---
