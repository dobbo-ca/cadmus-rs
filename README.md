# cadmus-rs

A pure Rust OCR engine: reads Tesseract `.traineddata` models directly and runs their LSTM
recognizer. A port of [dobbo-ca/cadmus](https://github.com/dobbo-ca/cadmus) at b7fac51 (Go).
Pure Rust, one dependency (`rayon`, for the worker pool: [docs/dependencies.md](docs/dependencies.md)),
`#![forbid(unsafe_code)]` (rayon's own `unsafe` is the documented exception), no SIMD tier.

```toml
cadmus = { git = "https://github.com/dobbo-ca/cadmus-rs", rev = "<sha>" }
```

## Quick start

Recognize a line of text: a binary PGM crop of one line in, text out (like `tesseract --psm 13`).
The model is `eng.traineddata` from tessdata_best 4.1.0; it is not in the repository. Every call
takes the `Limits` it must stay within; the long ones also take a cancel flag.

```rust,no_run
use cadmus::imaging::Bitmap;
use cadmus::recog::Recognizer;
use cadmus::Limits;
use std::sync::atomic::AtomicBool;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // One thread, at most 256 MiB of working memory per call.
    let limits = Limits { max_threads: 1, max_alloc: 256 << 20, ..Limits::default() };
    let cancel = AtomicBool::new(false); // set it from another thread to stop a call

    let recognizer = Recognizer::new(&std::fs::read("eng.traineddata")?, &limits, &cancel)?;
    let line = Bitmap::from_pgm(&std::fs::read("line.pgm")?, &limits)?;
    let result = recognizer.recognize(&line, &limits, &cancel)?;
    println!("{} ({:.0}%)", result.text, result.confidence);
    Ok(())
}
```

## Resource limits

| Field | Default | Unit | Over it |
|---|---|---|---|
| `max_alloc` | 1 GiB | bytes, peak of one call, all threads together | `Error::AllocLimit` |
| `max_threads` | `std::thread::available_parallelism()` (1 if unknown) | OS threads, the calling thread included | never an error: the call uses fewer |
| `pool` | `None`: the crate's shared pool | a `ThreadPool` of the caller's, which always wins | not a limit |
| `max_model_bytes` | 64 MiB | bytes, size of a `.traineddata` file | `Error::ModelLimit` |
| `max_pixels` | 1<<27 | pixels of one image plane (`width * height`) | `Error::PixelLimit` |
| `max_width` | 1<<16 | pixels | `Error::PixelLimit` |
| `max_height` | 1<<16 | pixels | `Error::PixelLimit` |

Every call that allocates from input sizes takes `&Limits`; the long ones also a cancel flag. Details: [docs/limits.md](docs/limits.md), [docs/threading.md](docs/threading.md).

## Docs

- [docs/status.md](docs/status.md): what is ported, modules and entry points
- [docs/go-differences.md](docs/go-differences.md): differences from the Go library, errors
- [docs/float-math.md](docs/float-math.md): float math and byte-identity with Go
- [docs/testing.md](docs/testing.md): tests and CI
- [docs/parity.md](docs/parity.md): Go parity job
- [docs/fuzzing.md](docs/fuzzing.md): fuzz targets (`cargo fuzz --fuzz-dir tests/fuzz`)
- [docs/benchmarks.md](docs/benchmarks.md): benchmark
- [docs/performance.md](docs/performance.md): build profile, x86-64-v4, PGO and BOLT
- [docs/quality.md](docs/quality.md): real-file OCR gate (OpenSlide labels)
