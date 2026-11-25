# Performance

How to build and call this crate for speed. The measured numbers are in
[benchmarks.md](benchmarks.md).

## Build profile

A dependent crate's own release profile applies to this code, so the profile that counts is the
consumer's. For a worker that recognizes lines:

```toml
[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"
```

```sh
RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+false-deps-tzcnt,+false-deps-bls" cargo build --release
```

- `lto = "fat"` and `codegen-units = 1` let the matrix kernels of the LSTM and the loops that call
  them inline across modules and across the crate boundary. A dependent that only wants the
  one-unit build of this crate can set `[profile.release.package.cadmus] codegen-units = 1`.
- `panic = "abort"` removes the unwinding tables. A panic on a pool thread is re-raised on the
  calling thread (rayon), and with `abort` the process stops at the panic instead. The crate does
  not panic on any input (clippy denies `unwrap`, `expect` and `panic` in its non-test code), so
  this only matters for a bug.
- `-C target-cpu=x86-64-v4` lets the compiler use AVX-512 (F, BW, CD, DQ, VL) everywhere. **Use it
  only when every machine that runs the binary has AVX-512.** On a CPU without it the program dies
  with `SIGILL` (illegal instruction) at the first such instruction, with no message. Not every
  GitHub-hosted runner has AVX-512, so CI runs the v4 builds only where `/proc/cpuinfo` lists
  `avx512bw` (`avx512f` for the bench tables); the default build needs no flag and runs everywhere.
- `-C target-feature=+false-deps-tzcnt,+false-deps-bls` goes with it. On Zen 4 and Zen 5 `tzcnt`,
  `blsr`, `blsi` and `blsmsk` wait for the last write to their output register, and LLVM breaks
  that wait only under `target-cpu=znver4`/`znver5` tuning, not under `x86-64-v4` (the jpeg-rs
  encoder ran 30 to 60% slower without the two features). This crate has no `tzcnt` loop that we
  know of, so here the flags are most likely a no-op; they are in the recommendation so that every
  crate of a worker is built the same way. rustc warns that it does not know the two features and
  passes them to LLVM, which does.
- The output does not change. The crate has no SIMD tier and no runtime dispatch; its float math
  is written so that it is byte-identical to Go ([float-math.md](float-math.md)), and LLVM never
  fuses a multiply and an add into an FMA unless the source asks for one, whatever `target-cpu`
  says. Where it does ask (`f64::mul_add` in the ported `math.Exp`), a v4 build makes it one FMA
  instruction instead of the libm `fma` call of the default x86_64 build; both round once, so the
  result is the same. The real-file gate ([quality.md](quality.md)) runs an oracle built with these flags and
  holds it to the Go output byte for byte.
- aarch64: do not set any of these flags (they are x86 only). The default aarch64 target already
  has NEON; `lto`, `codegen-units` and `panic` apply as above. When arm64 nodes join, build a
  separate image for them and pick its `target-cpu` for those nodes.

## The v4 bench tables

`bench.yml` also builds the Go oracle with `GOAMD64=v4` and the bench example with the flags
above, and prints two more tables (default parallelism and one thread) next to the default
builds. They are a report: the 1.10x rule applies to the default builds. Numbers:
[benchmarks.md](benchmarks.md).

## PGO and BOLT

Profile-guided optimisation and BOLT apply to the final binary, so the consumer does them (the
worker that links this crate); the library does not change. `pgo.yml` (dispatch only, no secrets)
measures how much they gain on the bench example: it builds the example with `cargo-pgo`, trains
it on the bench inputs (the line corpus and the scan; `bench <model> tests/data 1` and with the
default threads, so both paths are in the PGO profile; the BOLT profile is trained on one thread
only), then times the default, the PGO and the PGO plus BOLT binaries with `tests/parity/bench.sh`
and prints the tables in the job summary. BOLT is optional there: if the package is not available
the job reports PGO only.

Numbers: see the PGO section of [benchmarks.md](benchmarks.md).

## Reusing a Workspace

A worker that recognizes one text line after another should keep a `recog::Workspace` and a
`recog::Line` and call `Recognizer::recognize_into(&mut ws, &img, &mut line, &limits, &cancel)`
instead of `recognize`. `recognize` is `recognize_into` on a fresh `Workspace` and a fresh `Line`;
the two give the same bytes for every workspace, line, thread count and sequence of calls whenever both succeed.

**What is kept.** The `Workspace` holds every buffer one call needs: the scaled and normalized
line, the output and scratch tensors of each layer of the network (Series, Parallel, Reversed,
LSTM, Convolve, FullyConnected, Reconfig and Maxpool alike), the LSTM's state and gate vectors, the
beam search's nodes, dictionary positions, steps and best path, the scored symbols with their
strings, and the character boundaries. The buffers are handed out by the order a call asks for
them, so a repeated call finds each one as large as it needs. `line` is overwritten in place: its
text is cleared and refilled, its `Word` entries (and their strings) are reused, extra words are
truncated and a word is pushed only when the line has more than any before. After an error `line` is
unspecified.

**Zero allocations.** With `Limits::max_threads` 1, a call on a warm workspace and line makes no
allocator call (alloc, realloc or dealloc): after a few calls on the same image, or on one of the same
size whose result needs no more words, no longer strings and no larger beam than the buffers
already hold. The beam's arenas are kept a third larger than the call needed, because the
convolutions' random border taps move the beam a little from call to call. `tests/reuse_alloc.rs`
counts the calls. The exception is a source line more than 1800 pixels high, which the scaler
reduces below a factor of 0.02 with a small work array that it still allocates.

**`max_alloc` is still the peak of each call.** At the start of every call the capacity the
workspace holds is charged to the call's `Budget`, then whatever the call grows on top of it. If
the kept buffers alone do not fit in `max_alloc`, the workspace is emptied and the call runs as on a
fresh one. If the call then fails with `Error::AllocLimit` on a workspace that held buffers, it runs
once more on an empty workspace (with the generator state it started from, which rewinds the
generator that concurrent callers of one `Recognizer` share), so a limit that a fresh
call passes is passed by any workspace. The converse is not promised: a warm call grows less than a
fresh one (no doubling steps), so a limit just under a fresh call's peak may pass when reused. After `Error::AllocLimit`
the workspace holds nothing; after any other error it is usable. Releases happen at the end of the
call (the `Budget` is per call). Because the kept buffers count at their capacity, a call's peak
is the workspace's size, which is the largest line it has served, not the sum of what that line
needs at once; set `max_alloc` no tighter than the call that built the buffers needs.

**Threads.** With `max_threads` above 1 a call may make small allocations, bounded by the thread
count: the pool's job boxes and the list of bands of a `FullyConnected` layer (each extra thread is
charged 4096 bytes to `max_alloc`, as ever). Everything else is reused.

**Memory is held until the workspace is dropped.** It grows to the largest call it has served and
never shrinks, and each worker thread needs its own: a `Workspace` serves one call at a time.
Drop it (or replace it with `Workspace::default()`) to give the memory back.
