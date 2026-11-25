# Resource limits

The model is 15 MB on disk and its weights are held as `f64` in memory; `Limits::default()` is a
1 GiB budget, and `tests/limits_peak.rs` measures what each call needs.

`Limits` is a plain struct with public fields and a `Default`. Every public entry point that
allocates from a size the input controls takes `&Limits` (there is no unlimited call that builds
a bitmap, parses a model or runs the network; the few functions that take none are listed under
Errors below), and the long ones also take `cancel: &AtomicBool`. A
caller in a 2-CPU, 1 GiB pod sets `max_threads: 2, max_alloc: 768 << 20` and no call goes above
either. This example runs without the model, on a synthetic image (it is a doc-test):

The crate-level doc-test in `src/lib.rs` (rustdoc "Resource limits") runs an extended form of this example.

```rust
use cadmus::imaging::{Bitmap, otsu};
use cadmus::{Error, Limits};
use std::sync::atomic::{AtomicBool, Ordering};

fn main() -> Result<(), Error> {
    let limits = Limits { max_threads: 1, max_alloc: 64 << 20, ..Limits::default() };
    let cancel = AtomicBool::new(false);
    let gray: Vec<u8> = (0..512 * 512).map(|i| if i % 512 < 256 { 30 } else { 220 }).collect();
    let img = Bitmap::from_gray8(512, 512, &gray, &limits)?;
    let bin = otsu(&img, &limits, &cancel)?;

    // A budget that is too small is an error, not an allocation.
    let tiny = Limits { max_alloc: 16, ..limits.clone() };
    assert_eq!(otsu(&img, &tiny, &cancel), Err(Error::AllocLimit));

    // A set cancel flag stops the call at its next row.
    cancel.store(true, Ordering::Relaxed);
    assert_eq!(otsu(&img, &limits, &cancel), Err(Error::Cancelled));
    assert_eq!(bin.at(0, 0), 1);
    Ok(())
}
```

| Field | Default | Unit | Over it |
|---|---|---|---|
| `max_alloc` | 1 GiB | bytes, peak of one call, all threads together | `Error::AllocLimit` |
| `max_threads` | `std::thread::available_parallelism()` (1 if unknown) | OS threads, the calling thread included | never an error: the call uses fewer |
| `pool` | `None`: the crate's shared pool | a `ThreadPool` of the caller's, which always wins | not a limit |
| `max_model_bytes` | 64 MiB | bytes, size of a `.traineddata` file | `Error::ModelLimit` |
| `max_pixels` | 1<<27 | pixels of one image plane (`width * height`) | `Error::PixelLimit` |
| `max_width` | 1<<16 | pixels | `Error::PixelLimit` |
| `max_height` | 1<<16 | pixels | `Error::PixelLimit` |

Go had no limits; the defaults follow `docs/guides/memory-limits.md` of the Go repo (64 MiB
model, 1<<27 pixel plane, a 65,536 pixel line width, which is `max_width`) plus the 1 GiB
`max_alloc`. `max_height` (1<<16) is this crate's own choice: the guide has no height cap.

- **`max_alloc` is the enforced peak, not an estimate.** A call makes one `Budget` and passes it
  to every step it runs (`scale_gray` and its sharpening, `normalize`, `Recognizer::new` and the
  network it builds, `recognize` and `recognize_into` and the beam). Each allocation that grows with the input is
  charged before it is made and released when it is freed, so `Budget::peak` is the most the call
  held at once, threads included. This crate has no nested crates, so there is no budget to hand
  down. The image the caller passes in and the model a call returns are not part of it. If the
  allocator refuses one of those buffers, the call returns `Error::OutOfMemory` (the short strings
  of a symbol or a layer name are the exception: they are made without a fallible allocation). `tests/limits_peak.rs`
  measures the real heap peak of the main calls (`recognize` on a fresh workspace, not a warm
  `recognize_into`: `tests/reuse_alloc.rs` checks that one only for Ok or `AllocLimit`) with a counting allocator and checks it against
  `max_alloc`, down to the smallest value each call succeeds with.
- Threads and pools: see [threading.md](threading.md).
- **Cancellation.** The calls that can run long take `cancel: &AtomicBool` and check it per row,
  timestep or layer, on every thread. When it is set they return `Error::Cancelled` at the next
  check and drop what they built. The crate never sets or resets the flag. `Recognizer::new`
  also checks it between the model components.
- **Errors.** Limit errors are their own `Error` variants, so a caller can match on them:
  `AllocLimit`, `ModelLimit`, `PixelLimit`, `OutOfMemory`, `Cancelled` (see the table). Go's
  panics on bad arguments (wrong depth, empty bitmap, bad connectivity) are `Error::Imaging`,
  `Error::Network`, `Error::Tessdata` or `Error::Recog`. Every public function documents the
  limits it enforces and its errors in `# Errors`; the same section is in the crate docs
  (`src/lib.rs`). The functions that take no `Limits` allocate nothing from input-controlled
  sizes and have no `# Errors`: the activation kernels (`tanh`, `logistic`, `softmax_in_place`,
  `tanh_in_place`, `logistic_in_place`, `Matrix::dot_vector`), the `Tensor` time-step copies
  (`write_time_step`, `copy_time_step*`), `Dawg::contains` and `has_prefix`, `Dict::new`, and the
  layer constructors (`Input::new`, `Series::new`, `Parallel::new`, `Reversed::new`,
  `Reconfig::new`, `Reconfig::new_maxpool`, `Convolve::new`, `FullyConnected::new`, `Lstm::new`),
  which copy a name and weights the caller already sized and charged.
- No global thread pool except the lazy shared one ([threading.md](threading.md)), no growing global cache, no environment
  variable changes a limit.
