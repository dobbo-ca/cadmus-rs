# Status

The whole library is ported (`nn`, `imaging`, `tessdata`, `dict`, `recog` and the `cadmusdump`
tool, `src/bin/cadmusdump.rs`), with resource limits and threading, Go parity (`tests/parity/`),
fuzzing (`tests/fuzz/`) and the benchmark (`examples/bench.rs`; [benchmarks.md](benchmarks.md)). The Go repo is at L1 of its
design: a line crop in, text out, matching `tesseract --psm 13`; there is no page-level entry
point yet, so the public surface is the five modules, as in Go (where they are `internal/`), plus
`par` (`for_each_chunk_mut`, the pool helper the threaded calls use, and `ThreadPool`) and the
ledger types `Budget`, `Buf` and `Hold` that `Limits` goes with. There is no SIMD tier.

The Go library has only internal packages, so each is a public module here: `nn` (the tensor
runtime), `imaging`, `tessdata` (the model loader), `dict` and `recog`.

Entry points: `tessdata::Model::load` (container, network weights, unicharset, recoder, dawgs),
`recog::Recognizer::new(model_bytes, &Limits, &cancel)` then `recognize(&Bitmap, &Limits, &cancel)`
(beam search with the dictionary, words and confidences), `recognize_into(&mut Workspace, &Bitmap,
&mut Line, &Limits, &cancel)` (the same on buffers the caller keeps from one call to the next, for a
worker that recognizes line after line; see [performance.md](performance.md)) or `recognize_text`
(greedy). Images are
8bpp `imaging::Bitmap`s (`from_gray8`, `from_rgba8`, `from_pgm`).
