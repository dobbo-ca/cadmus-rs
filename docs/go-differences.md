# Differences from the Go library

- No PNG: `cadmusdump -activations` and the line corpus read binary PGM. The corpus
  (`tests/data/lines`) is the Go repo's PNGs converted with Go's `color.GrayModel`.
- Go panics on bad arguments (wrong depth, empty bitmap, bad connectivity) become `Error`s.
- `tessdata`: a unicharset must be UTF-8; layer nesting is capped at 64; strings read from the
  model (layer names, the spec and the version string) are lossily decoded.
  `imaging::scale_gray` rejects an empty source.
- `nn::tanh` and `nn::logistic` return NaN for a NaN input, where Go's table lookup panics (a
  forward pass that reaches one fails with `Error::Network`).
- `Recognizer` owns the dawgs and builds a `dict::Dict` view on demand (`Go: r.Dict`, now
  `set_lexicon(bool)`).
- The tap that `recog.BuildWithTap` wraps around every layer is `nn::Ctx::tap`
  (`Recognizer::forward_tap`).
- `Recognizer::recognize_into` and `recog::Workspace` have no Go counterpart: Go has no API that
  reuses buffers between calls ([performance.md](performance.md)). The output is the same.

Not ported: `internal/nn/gen` (the tables generator, `//go:build ignore`; the tables are carried
over as `src/nn/tables.rs`), `internal/nofma_test.go` (it checks Go's arm64 output for fused
multiply-adds; Rust never fuses, and `matrix_dot_vector_does_not_fuse_multiply_add` checks the
dot product), `tests/data/fetch.sh` (the CI image fetches the model instead, see [testing.md](testing.md)),
`tests/data/lines/gen.sh`, the C golden generators (kept under `tests/data/golden/gen` for
provenance); `BenchmarkRecognizeLine` is `examples/bench.rs` (see [benchmarks.md](benchmarks.md)).

## Errors

Errors: Go's errors are all `fmt.Errorf` strings per package, so `Error` has one variant per Go
package (`Tessdata`, `Network`, `Imaging`, `Recog`) plus the limit variants (`ModelLimit`,
`PixelLimit`, `AllocLimit`, `OutOfMemory`, `Cancelled`).
