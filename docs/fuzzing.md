# Fuzzing

`tests/fuzz/` is a cargo-fuzz crate with fourteen targets: `container` (container, model and recognizer
loaders on arbitrary bytes), `unicharset`, `recoder`, `dawg`, `lstm` (the network parser; its first byte is a selector, not part of the component, whose low bit
picks the byte-swapped reader, so the real components in the seeds are read natively),
`network` (an `lstm` component written from the input: a layer tree of every type with its
dimensions and weights, infinities and NaN included, parsed, built and run on fuzzed activations),
`model_mutate` (the mini model with edits), `pgm`, `imaging` (every operator, under tiny budgets and
a set cancel flag, with invariants; also on bitmaps whose public fields disagree with their pixels,
on the wrong depth and without a source, plus `normalize` with fuzzed `input_height` and
`x_scale`), `threads` (planes of 65K to 261K pixels, so the operators really split into bands,
run with `max_threads` 1 and with 2 to 9, on the shared pool, a caller's pool or per-call workers,
and a fuzzed `max_alloc` of 64 KiB to 256 MiB, errors included, and the results must be equal; a
FullyConnected layer is compared the same way), `grid` (one grid with a fuzzed origin driven through many inserts and queries,
checked against a scan), `dict` (a lexicon built from a fuzzed unicharset and dawgs and walked
from both the positions it produces and hand-made ones), `recognize` (a recognizer shared by all
inputs: forward with a tap, greedy and `recognize_text`, beam with and without the lexicon
(on the mini model, `recognize` is also compared with two `recognize_into` calls on one `Workspace` and `Line`, and under a fuzzed `max_alloc` a limit `recognize` passes must be passed by a warm `recognize_into`; the real model is not compared, only checked for no panic or unbounded allocation), and
lexicon walks) and
`tensor` (the decoders and `score_symbols` on a caller-built tensor with fuzzed activations
whose public fields disagree with its data). The real eng network takes about a second to run a
line, so the recognizer targets use a mini model: the real model's charset, recoder and dawgs
around a network of a few hundred weights, built in `tests/fuzz/fuzz_targets/common.rs` (`recognize`
uses the real model for 1 input in 32, when bits 3-7 of the mode byte are set). There is no SIMD
kernel, so no differential target; `fuzz-arm64` runs `imaging` and `threads` natively on an arm64
runner (`ubuntu-24.04-arm`), with the same `cargo +nightly fuzz run` as the x86_64 job. Seeds are in `tests/fuzz/seeds/<target>`:
the components of the eng model, heads of the line corpus and small hand-made inputs (the Go repo
has no fuzz seeds). `.github/workflows/fuzz.yml` runs them weekly (Saturday 18:23 UTC) and on dispatch, 600 s per
target by default, two at a time (`model_mutate` inputs are capped at 64 bytes, one keep byte and
ten edits, with `-len_control=0`, and its calls run with `max_pixels` 1 << 16, so an edited input
height past 181 is refused at the scaler instead of costing a serial beam search of hundreds of
thousands of steps), with `-rss_limit_mb=2048 -malloc_limit_mb=512` (`imaging`, `threads` and `tensor` also count the heap with a global allocator and fail if a call's peak goes over its `max_alloc`); the corpus is
kept between runs (one cache per architecture and target); the targets that need the model read it from `CADMUS_MODEL`, which the image
sets.
