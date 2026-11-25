//! Pure Rust OCR engine: reads Tesseract's `.traineddata` models directly and runs their LSTM
//! recognizer. A port of dobbo-ca/cadmus (Go).
//!
//! The Go library has only internal packages, so each is a public module here: [`nn`] (the
//! tensor runtime), [`imaging`], [`tessdata`] (the model loader), [`dict`] and [`recog`].
//!
//! # Resource limits
//!
//! Every public entry point that allocates from a size the input controls takes
//! `&`[`Limits`], and the large ones also a cancel flag (`&AtomicBool`). There is no unlimited
//! call that builds a bitmap, parses a model or runs the network; the few functions that take
//! no `Limits` (the activation kernels, the tensor time-step copies, `Dawg` lookups, `Dict::new`
//! and the layer constructors) allocate nothing from input-controlled sizes. A caller in a small
//! pod sets two fields and the call never goes above either:
//!
//! ```
//! use cadmus::imaging::{Bitmap, otsu};
//! use cadmus::{Error, Limits, ThreadPool};
//! use std::sync::atomic::{AtomicBool, Ordering};
//!
//! // One thread, 64 MiB of working memory.
//! let limits = Limits { max_threads: 1, max_alloc: 64 << 20, ..Limits::default() };
//! let cancel = AtomicBool::new(false);
//!
//! // A synthetic 512x512 image: dark ink on the left half, paper on the right.
//! let gray: Vec<u8> = (0..512 * 512).map(|i| if i % 512 < 256 { 30 } else { 220 }).collect();
//! let img = Bitmap::from_gray8(512, 512, &gray, &limits)?;
//! let bin = otsu(&img, &limits, &cancel)?;
//! assert_eq!((bin.at(0, 0), bin.at(511, 0)), (1, 0));
//!
//! // More threads give the same bytes.
//! let two = Limits { max_threads: 2, ..limits.clone() };
//! assert_eq!(otsu(&img, &two, &cancel)?, bin);
//!
//! // A pool the caller keeps gives the same bytes: it replaces the crate's shared pool.
//! let pooled = Limits { pool: Some(ThreadPool::new(2)?), ..two.clone() };
//! assert_eq!(otsu(&img, &pooled, &cancel)?, bin);
//!
//! // A budget that is too small is an error, not an allocation.
//! let tiny = Limits { max_alloc: 16, ..limits.clone() };
//! assert_eq!(otsu(&img, &tiny, &cancel), Err(Error::AllocLimit));
//!
//! // A set cancel flag stops the call at its next row.
//! cancel.store(true, Ordering::Relaxed);
//! assert_eq!(otsu(&img, &limits, &cancel), Err(Error::Cancelled));
//! # Ok::<(), Error>(())
//! ```
//!
//! | `Limits` field | Default | Unit | Over it |
//! |---|---|---|---|
//! | `max_alloc` | 1 GiB | bytes | [`Error::AllocLimit`] |
//! | `max_threads` | `available_parallelism()` | threads | never an error |
//! | `pool` | `None`: the crate's shared pool | a [`ThreadPool`] of the caller's | not a limit |
//! | `max_model_bytes` | 64 MiB | bytes | [`Error::ModelLimit`] |
//! | `max_pixels` | 1<<27 | pixels per plane | [`Error::PixelLimit`] |
//! | `max_width` | 1<<16 | pixels | [`Error::PixelLimit`] |
//! | `max_height` | 1<<16 | pixels | [`Error::PixelLimit`] |
//!
//! - `max_alloc` is the enforced peak working memory of one call, all threads together. A call
//!   makes one [`Budget`], charges every allocation that grows with the input before it is made
//!   and releases it when it is freed, so the peak never goes above `max_alloc`. The input the
//!   caller passes in and the model a call returns are not part of it. This crate has no nested
//!   crates, so there is no budget to hand down. A refusal by the allocator of one of those
//!   buffers is [`Error::OutOfMemory`] (the short strings of a symbol or a layer name are the
//!   exception: they are made without a fallible allocation).
//! - `max_threads` is the most OS threads one call uses, the calling thread included.
//!   `max_threads = 1` runs everything on the calling thread, never touches a pool and spawns
//!   none. Each extra thread is charged 4096 bytes to `max_alloc` from its first stage until the
//!   call returns, so a small budget also means fewer threads. The output is byte-identical for every thread count.
//! - The threads come from a pool, so small inputs use them too. [`Limits::pool`] (a
//!   [`ThreadPool`], made once with `ThreadPool::new(n)` and kept) always wins. Without one, a
//!   call with `max_threads` above 1 uses the crate's shared pool: a `OnceLock` built lazily by
//!   the first such call that has a parallel stage, with `std::thread::available_parallelism()`
//!   threads and never more, and holding no other state. `max_threads` 1 never touches it (it is
//!   not even built). Each call takes at most `max_threads` threads from the pool, the calling
//!   thread included, and no more workers than the pool has; every stage of the call runs on it,
//!   so no thread is started per stage. If the shared pool cannot be built, the call starts one
//!   set of workers of its own at its first parallel stage, which every later stage shares. A
//!   pool's own memory (stacks, queues) is fixed, not sized by an input, and is not part of
//!   `max_alloc`. The pool changes speed only, not the bytes.
//! - Cancellation: the calls that can run long take `cancel: &AtomicBool` and check it per row,
//!   timestep or layer, on every thread. Once it is set they return [`Error::Cancelled`] at the
//!   next check; whatever was built is dropped and its memory released. Nothing resets the flag.
//! - There is no global thread pool (but the lazy shared one above), cache or environment
//!   variable that changes a limit.
#![forbid(unsafe_code)]
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_range_contains
)]
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented
    )
)]

pub mod dict;
mod error;
pub mod imaging;
mod limits;
pub mod nn;
pub mod par;
mod pool;
pub mod recog;
pub mod tessdata;

pub use error::Error;
pub use limits::{Budget, Buf, Hold, Limits};
pub use par::ThreadPool;
#[doc(hidden)]
pub use par::shared_pool_started;

/// Compiles the code blocks of the README as doc-tests, so they cannot go stale.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
