# Dependencies

One crate, `rayon`, with what it pulls in (`rayon-core`, `crossbeam-deque`, `crossbeam-epoch`,
`crossbeam-utils`, `either`; exact versions in `Cargo.lock`). All are pure Rust: no C, no `-sys`
crate, no build script that compiles anything. It is here only for the worker pool: safe Rust
cannot reuse worker threads for jobs that borrow short-lived buffers (`std::thread::scope` must
start and join its threads on every call, tens of microseconds each, and the code that used it
started a scope for every `FullyConnected` layer of a recognition and for every stage of a
multi-stage imaging call), and rayon's scoped pool does. It backs the caller's `Limits::pool`, the
lazy shared pool and a call's own fallback workers (see [limits.md](limits.md)). This crate stays
`#![forbid(unsafe_code)]`; the `unsafe` inside rayon is rayon's own. The crate uses only
`ThreadPool::in_place_scope` and `broadcast` (the latter in tests).
