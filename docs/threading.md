# Threading

From the `Limits` rules ([limits.md](limits.md)):

- **`max_threads = 1` spawns no thread**: everything runs on the calling thread and never touches
  a pool. With more, work that splits into independent rows or bands (see [threading.md](threading.md)) runs on up
  to `max_threads` threads, the caller included, and takes them from the same `Budget`: each
  extra thread is charged 4096 bytes to `max_alloc` from its first stage until the call returns
  (the pool frees a task's box only after the stage has ended), so a small budget also means fewer
  threads.
  The output is byte-identical for every thread count.
- **The threads come from a pool**, so small inputs use them too (a stage is worth a worker from
  about 32 K elements; Go starts a goroutine with no threshold):
  - `Limits::pool` (a `ThreadPool`, made once with `ThreadPool::new(n)` and kept) always wins.
    Clones share the workers, and calls on other threads can share one pool.
  - Without one, a call with `max_threads` above 1 uses the crate's shared pool: a `OnceLock`,
    created lazily by the first such call that has a parallel stage, with
    `std::thread::available_parallelism()` threads and never more. It keeps no other state.
    `max_threads: 1` never touches it (it is not even created) and starts nothing.
  - Each call takes at most `max_threads` threads from the pool, counting the caller, and no more
    workers than the pool has.
  - Every stage of a call (`scale_gray` and its sharpening, the two passes of `open`, each
    `FullyConnected` layer of a recognition) runs on the call's pool, so a call starts no thread
    per stage: the workers are already there.
  - If the shared pool cannot be created, the call starts one set of workers of its own at its
    first parallel stage (at most `max_threads` - 1 of them, and no more than the CPUs), and every
    later stage shares them. `tests/threads.rs` forces this with a hidden `Limits` field.
  - A pool's own memory (thread stacks and queues) is fixed and not sized by any input, so it is
    not part of `max_alloc`; what a call holds for its workers is (4096 bytes per extra thread,
    until the call returns). The pool changes speed only, not the bytes.

## Where threads are used

Threads are used where the work is split into independent rows or bands, and only for planes of
at least about 32K elements per thread (a smaller call uses fewer threads, down to the calling
thread alone): `otsu`, `sauvola`, `rotate`, `skew_angle` (the candidate angles), the morphology
passes, `scale_gray` and its sharpening, and the `FullyConnected` layer (timesteps). Each stage
hands its bands out one at a time, in order, to the threads of the call's pool (see [limits.md](limits.md)), so a stage with fewer free workers than bands still completes, on the threads it has. The
output is byte-identical for every `max_threads` and every pool; each thread's scratch buffers
are charged to the same `Budget`. The LSTM
recurrence, the `Convolve` layer (its off-image taps come from one sequential generator), the
flood fills, the distance function and the beam search run on the calling thread.
