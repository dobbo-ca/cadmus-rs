# Tests and CI

## Tests

The Go repo has 189 test functions (nn 36, imaging 58, tessdata 52, dict 6, recog 29, cadmusdump
7, and `internal/nofma_test.go` 1, not ported); the Rust crate has 215 `#[test]` functions (nn 49, imaging 53, tessdata 36, dict 8, recog 32,
cadmusdump 6, and `limits` 9 and `par` 10, all in `src/`, 1 in `tests/limits_peak.rs`, 4 in
`tests/threads.rs`, 1 in `tests/shared_pool.rs` and 6 in `tests/reuse_alloc.rs`: a reused `Workspace` against fresh calls, its zero-allocation call, its `max_alloc` rules and its forced warm retry), plus
the doc-tests: the crate docs and the code blocks of the README are compiled, so they cannot go
stale. Table-driven and "rejects bad input" cases are merged into one test each, and the Leptonica golden tests are all there. What is not carried over one-to-one: the Go
tests that pin numbers of the real eng model (layer weights, unicharset entries, dawg words,
about 25 in `tessdata` and `dict`), which are replaced by the invariants checked in
`real_model_loads`, the synthetic dawg walks of `dict/tests.rs` and the corpus tests of `recog`.
The tests that need `tests/data/eng.traineddata` (tessdata_best 4.1.0, 15 MB, not in the
repository, as in Go) return early when it is absent; the CI image downloads it with a SHA-256
check (`.github/ci/Dockerfile`), so there they run.

Tests ported from the Go repo are in `src/*/tests.rs`; the Leptonica goldens are in
`tests/data/golden/` (as in Go, regenerated only by hand with the C generator in `golden/gen`; the C
files there are offline golden generators and are never part of the build).

## CI

`.github/workflows/ci.yml` runs `.github/ci/Dockerfile` with BuildKit on GitHub-hosted runners. The only required
status check of a PR is `test` (fmt, clippy `-D warnings`, `cargo test`, an aarch64 release build
of the library); `fuzz-build` also runs on a PR and only builds the fuzz targets, so a break shows
in the PR and not the next night. `arm64`, `parity` and `parity-arm64` run on a push to `main` and
on `workflow_dispatch` (`gh workflow run ci.yml --ref <branch>` runs all of them on a branch), and
must be green on `main`. `arm64` runs the `--lib` tests filtered to `nn::`, `imaging::`, `recog::`
and `tessdata::network` natively on an arm64 runner (`ubuntu-24.04-arm`; there is no SIMD tier, so
one run, and without the model, whose tests skip); `parity` and `parity-arm64` are described
in [parity.md](parity.md).
