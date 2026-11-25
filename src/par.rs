//! Data parallelism for the calls that split their work over disjoint chunks, in safe Rust: a
//! rayon scope on a [`ThreadPool`] (the caller's from `Limits::pool`, the crate's lazy shared one,
//! or one the call builds for itself) over chunks handed out one at a time, so every thread owns
//! what it works on. The pool is resolved once per call, by its [`Budget`], and every stage of
//! the call runs on it. The threads of a call come out of the [`Budget`], so a call never has
//! more than `Limits::max_threads` running (itself included), and `max_threads = 1` runs on the
//! calling thread, never touches a pool and does not even build the shared one.

use crate::imaging::check_cancel;
use crate::{Budget, Error};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

/// Worker threads that calls share. Make one with [`ThreadPool::new`], keep it, and pass it in
/// `Limits::pool`: each call then runs its parallel stages on these threads instead of the
/// crate's shared pool. `Limits::max_threads` still caps the threads one call uses, the caller
/// included, and a call never takes more workers than the pool has. Clones share the threads.
#[derive(Clone, Debug)]
pub struct ThreadPool(Arc<rayon::ThreadPool>);

impl PartialEq for ThreadPool {
    fn eq(&self, other: &ThreadPool) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ThreadPool {}

impl ThreadPool {
    /// A pool of `threads` workers (at least 1), started now and stopped when the last clone
    /// drops. It enforces no limit: `threads` is not capped and its own memory (stacks, queues)
    /// is not charged to any call; each call on it is bounded by its own `Limits`.
    ///
    /// # Errors
    ///
    /// `Error::OutOfMemory` if the operating system will not start the threads.
    pub fn new(threads: usize) -> Result<ThreadPool, Error> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1))
            .thread_name(|i| format!("cadmus-worker-{i}"))
            .build()
            .map(|p| ThreadPool(Arc::new(p)))
            .map_err(|_| Error::OutOfMemory)
    }

    /// How many worker threads the pool has.
    pub fn threads(&self) -> usize {
        self.0.current_num_threads()
    }
}

/// The crate's shared pool, built on first use by a call with `max_threads` above 1 and no caller
/// pool that has a parallel stage, with `available_parallelism` threads and never more. `None`
/// if the build failed.
static SHARED: OnceLock<Option<ThreadPool>> = OnceLock::new();

pub(crate) fn shared_pool() -> Option<&'static ThreadPool> {
    SHARED
        .get_or_init(|| {
            ThreadPool::new(std::thread::available_parallelism().map_or(1, |n| n.get())).ok()
        })
        .as_ref()
}

/// Whether the shared pool has been built yet (for the tests: `max_threads` 1 never builds it).
#[doc(hidden)]
pub fn shared_pool_started() -> bool {
    SHARED.get().is_some()
}

/// What each extra thread of a call is charged to the budget, from its first stage until the call
/// ends: the heap its task takes in the pool (the boxed closure and its queue slot), which the
/// call would otherwise not account for. The pool frees that box after the stage has returned, so
/// the charge is not given back between stages. A budget with room for fewer threads than
/// `max_threads` gives the call fewer.
pub(crate) const THREAD_BYTES: usize = 4096;

/// A mutex's data even if a thread panicked while holding it: the panic goes on to the caller of
/// the scope and the data is dropped with the call.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Calls `f(index, chunk)` for every `chunk_len`-long chunk of `data` (the last may be shorter;
/// `index` counts from 0), on at most `budget.limits().max_threads` threads including the
/// caller, taken from the call's pool. The chunks are handed out in order, one at a time, to
/// whichever thread is free, so the result of an `f` that only writes its own chunk does not
/// depend on the thread count. A `chunk_len` of 0 is taken as 1.
///
/// The cancel flag is checked before every chunk on every thread. After the first error no new
/// chunk starts; the chunks already started finish, so the error returned is the one of the
/// lowest-numbered failing chunk, which does not depend on how the threads are scheduled.
///
/// # Errors
///
/// `Error::Cancelled` if `cancel` is set, `Error::OutOfMemory` if a pool the call needs cannot
/// be built, else the first error `f` returns.
pub fn for_each_chunk_mut<T, F>(
    budget: &Budget,
    cancel: &AtomicBool,
    data: &mut [T],
    chunk_len: usize,
    f: F,
) -> Result<(), Error>
where
    T: Send,
    F: Fn(usize, &mut [T]) -> Result<(), Error> + Sync,
{
    let chunk_len = chunk_len.max(1);
    let chunks = data.len().div_ceil(chunk_len);
    let run = |i: usize, chunk: &mut [T]| -> Result<(), Error> {
        check_cancel(cancel)?;
        f(i, chunk)
    };
    // No pool is looked at for a single chunk or for `max_threads` 1.
    let pool = if chunks > 1 && budget.limits().max_threads > 1 {
        budget.pool()
    } else {
        None
    };
    let (pool, want) = match pool {
        Some(pool) => {
            // Workers the call already holds a charge for cost nothing more.
            let room = budget.limits().max_alloc.saturating_sub(budget.used()) / THREAD_BYTES
                + budget.workers_held();
            let cap = chunks.saturating_sub(1).min(room).min(pool.threads());
            (pool, budget.acquire_threads(cap))
        }
        None => return serial(data, chunk_len, run),
    };
    // The charge is not given back when the stage ends: see `Budget::hold_workers`.
    if want == 0 || budget.hold_workers(want, THREAD_BYTES).is_err() {
        budget.release_threads(want);
        return serial(data, chunk_len, run);
    }
    let queue = Mutex::new(data.chunks_mut(chunk_len).enumerate());
    let failed = AtomicBool::new(false);
    let first: Mutex<Option<(usize, Error)>> = Mutex::new(None);
    let work = || {
        while !failed.load(Ordering::Relaxed) {
            let Some((i, chunk)) = lock(&queue).next() else {
                return;
            };
            if let Err(e) = run(i, chunk) {
                let mut fst = lock(&first);
                if fst.as_ref().is_none_or(|(j, _)| i < *j) {
                    *fst = Some((i, e));
                }
                failed.store(true, Ordering::Relaxed);
                return;
            }
        }
    };
    // The caller works too; the pool's workers join as they are free, and one that comes after
    // the chunks are gone returns at once.
    let work = &work;
    pool.0.in_place_scope(|sc| {
        for _ in 0..want {
            sc.spawn(move |_| work());
        }
        work();
    });
    budget.release_threads(want);
    match first.into_inner().unwrap_or_else(|e| e.into_inner()) {
        Some((_, e)) => Err(e),
        None => Ok(()),
    }
}

/// The chunks in order on the calling thread.
fn serial<T>(
    data: &mut [T],
    chunk_len: usize,
    run: impl Fn(usize, &mut [T]) -> Result<(), Error>,
) -> Result<(), Error> {
    data.chunks_mut(chunk_len)
        .enumerate()
        .try_for_each(|(i, chunk)| run(i, chunk))
}

/// The least work, counted in elements the call touches, that is worth a thread: handing a task
/// to a pool worker costs about as much as this many simple per-element operations.
const MIN_WORK_PER_THREAD: usize = 1 << 15;

/// How many bands a call with `work` elements of work should split into: one thread per
/// `MIN_WORK_PER_THREAD`, at most `max_threads` and the caller plus the pool's workers, at least
/// 1. A call under the minimum runs on the calling thread alone.
///
/// A call that gives each band its own scratch (`per_band` bytes, charged before the threads are
/// taken) also needs `fixed` bytes besides the bands while they are live. The count is cut to the
/// most bands whose scratch and threads fit in what is left of `max_alloc`, so that a budget that
/// serves one band is not refused because `max_threads` is larger, apart from the threads'
/// `THREAD_BYTES` that stay charged to the end of the call: it just gets fewer bands.
pub(crate) fn band_count(budget: &Budget, work: usize, fixed: usize, per_band: usize) -> usize {
    let mut wanted = (work / MIN_WORK_PER_THREAD).clamp(1, budget.limits().max_threads.max(1));
    if wanted > 1 {
        wanted = wanted.min(budget.pool().map_or(1, |p| p.threads() + 1));
    }
    // `n` bands take `n * per_band` bytes and `n - 1` threads take `THREAD_BYTES` each.
    let room = budget
        .limits()
        .max_alloc
        .saturating_sub(budget.used())
        .saturating_sub(fixed);
    let fit = 1 + room.saturating_sub(per_band) / per_band.saturating_add(THREAD_BYTES);
    wanted.min(fit)
}

/// `for_each_chunk_mut` over rows of `row_len` elements, `f(y, row)`, split into one band of
/// consecutive rows per thread as [`band_count`] says for `work`. The cancel flag is checked
/// before every row.
pub(crate) fn for_each_row_mut<T, F>(
    budget: &Budget,
    cancel: &AtomicBool,
    work: usize,
    data: &mut [T],
    row_len: usize,
    f: F,
) -> Result<(), Error>
where
    T: Send,
    F: Fn(usize, &mut [T]) -> Result<(), Error> + Sync,
{
    let row_len = row_len.max(1);
    let per_band = data
        .len()
        .div_ceil(row_len)
        .div_ceil(band_count(budget, work, 0, 0))
        .max(1);
    for_each_chunk_mut(budget, cancel, data, per_band * row_len, |b, band| {
        for (k, row) in band.chunks_mut(row_len).enumerate() {
            check_cancel(cancel)?;
            f(b * per_band + k, row)?;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Limits;
    use std::collections::HashSet;
    use std::thread::ThreadId;

    /// Workers of the caller's pool in these tests, so that the thread counts do not depend on the
    /// CPUs of the machine.
    const POOL_THREADS: usize = 8;

    fn caller_pool() -> ThreadPool {
        static POOL: OnceLock<ThreadPool> = OnceLock::new();
        POOL.get_or_init(|| ThreadPool::new(POOL_THREADS).unwrap())
            .clone()
    }

    fn budget(max_threads: usize) -> Budget {
        Budget::new(&Limits {
            max_threads,
            pool: Some(caller_pool()),
            ..Limits::default()
        })
    }

    #[test]
    fn the_result_does_not_depend_on_the_thread_count() {
        let cancel = AtomicBool::new(false);
        let want: Vec<u64> = (0..1000u64).map(|i| i * i + 1).collect();
        for threads in [1, 2, 3, 4, 16] {
            let b = budget(threads);
            let mut data = vec![0u64; 1000];
            for_each_chunk_mut(&b, &cancel, &mut data, 7, |i, chunk| {
                for (j, v) in chunk.iter_mut().enumerate() {
                    let k = (i * 7 + j) as u64;
                    *v = k * k + 1;
                }
                Ok(())
            })
            .unwrap();
            assert_eq!(data, want, "{threads} threads");
            assert!(b.peak_threads() <= threads, "{threads} threads");
            assert_eq!(
                b.peak_threads(),
                threads.min(POOL_THREADS + 1),
                "{threads} threads"
            );
        }
    }

    #[test]
    fn one_thread_runs_on_the_caller() {
        let cancel = AtomicBool::new(false);
        let b = budget(1);
        let me = std::thread::current().id();
        let mut data = vec![0u8; 64];
        for_each_chunk_mut(&b, &cancel, &mut data, 1, |_, _| {
            assert_eq!(std::thread::current().id(), me);
            Ok(())
        })
        .unwrap();
        assert_eq!(b.peak_threads(), 1);
    }

    /// The pool a call runs on is the caller's, and no more threads of it than `max_threads` less
    /// the caller run the chunks.
    #[test]
    fn a_caller_pool_is_the_one_used() {
        let cancel = AtomicBool::new(false);
        let pool = ThreadPool::new(3).unwrap();
        let workers: HashSet<ThreadId> = pool
            .0
            .broadcast(|_| std::thread::current().id())
            .into_iter()
            .collect();
        assert_eq!(workers.len(), 3);
        let me = std::thread::current().id();
        for max_threads in [2, 3, 8] {
            let b = Budget::new(&Limits {
                max_threads,
                pool: Some(pool.clone()),
                ..Limits::default()
            });
            let ids = Mutex::new(HashSet::new());
            let mut data = vec![0u8; 256];
            for_each_chunk_mut(&b, &cancel, &mut data, 1, |_, _| {
                // A little work, so that the workers get a chunk too.
                std::thread::sleep(std::time::Duration::from_micros(200));
                lock(&ids).insert(std::thread::current().id());
                Ok(())
            })
            .unwrap();
            let ids = ids.into_inner().unwrap();
            assert!(
                ids.iter().all(|t| *t == me || workers.contains(t)),
                "{max_threads}"
            );
            assert!(
                ids.len() <= max_threads,
                "{max_threads} threads: {}",
                ids.len()
            );
            assert!(ids.len() >= 2, "{max_threads} threads: only {}", ids.len());
            assert!(b.peak_threads() <= max_threads.min(4));
        }
    }

    /// With no pool of the caller's, the shared pool runs the call; with `per_call_pool` the call
    /// builds its own workers. Either way the bytes are the same.
    #[test]
    fn the_shared_pool_and_the_calls_own_workers_give_the_same_bytes() {
        let cancel = AtomicBool::new(false);
        let want: Vec<u64> = (0..500u64).map(|i| 3 * i + 1).collect();
        for per_call_pool in [false, true] {
            let b = Budget::new(&Limits {
                max_threads: 3,
                per_call_pool,
                ..Limits::default()
            });
            let mut data = vec![0u64; 500];
            for_each_chunk_mut(&b, &cancel, &mut data, 5, |i, chunk| {
                for (j, v) in chunk.iter_mut().enumerate() {
                    *v = 3 * (i * 5 + j) as u64 + 1;
                }
                Ok(())
            })
            .unwrap();
            assert_eq!(data, want, "per_call_pool {per_call_pool}");
            assert!(
                (2..=3).contains(&b.peak_threads()),
                "per_call_pool {per_call_pool}"
            );
            assert!(shared_pool_started() || per_call_pool);
        }
    }

    #[test]
    fn threads_go_back_to_the_budget() {
        let cancel = AtomicBool::new(false);
        let b = budget(4);
        let mut data = vec![0u8; 64];
        for _ in 0..3 {
            for_each_chunk_mut(&b, &cancel, &mut data, 8, |_, _| Ok(())).unwrap();
        }
        assert_eq!(b.peak_threads(), 4);
        assert_eq!(b.acquire_threads(10), 3);
    }

    /// A pool worker frees its task's box after the stage has returned, so what a call holds for
    /// its workers is not given back between stages: the next stage's allocations cannot take
    /// those bytes while the box may still be live (a heap peak over `max_alloc`).
    #[test]
    fn the_workers_charge_stays_until_the_call_ends() {
        let cancel = AtomicBool::new(false);
        let b = Budget::new(&Limits {
            max_alloc: 3 * THREAD_BYTES + 100,
            max_threads: 4,
            pool: Some(caller_pool()),
            ..Limits::default()
        });
        let mut data = vec![0u8; 64];
        for _ in 0..3 {
            for_each_chunk_mut(&b, &cancel, &mut data, 8, |_, _| Ok(())).unwrap();
            assert_eq!(b.used(), 3 * THREAD_BYTES);
        }
        // A stage that needs fewer workers charges nothing more, and one that needs more than the
        // call has held is charged only the difference.
        let mut two = [0u8; 2];
        for_each_chunk_mut(&b, &cancel, &mut two, 1, |_, _| Ok(())).unwrap();
        assert_eq!(b.used(), 3 * THREAD_BYTES);
        // Only what is left of `max_alloc` can still be charged.
        assert_eq!(b.charge(101), Err(Error::AllocLimit));
        b.charge(100).unwrap();
    }

    #[test]
    fn an_error_and_the_cancel_flag_stop_the_run() {
        let b = budget(4);
        let mut data = vec![0u8; 64];
        let cancel = AtomicBool::new(false);
        // Two chunks fail: the lower one is the error, as in a serial run.
        let r = for_each_chunk_mut(&b, &cancel, &mut data, 4, |i, _| {
            if i == 5 || i == 9 {
                Err(Error::Imaging(format!("chunk {i}")))
            } else {
                Ok(())
            }
        });
        assert_eq!(r, Err(Error::Imaging("chunk 5".into())));
        cancel.store(true, Ordering::Relaxed);
        let r = for_each_chunk_mut(&b, &cancel, &mut data, 8, |_, _| Ok(()));
        assert_eq!(r, Err(Error::Cancelled));
        // The threads of both runs went back.
        assert_eq!(b.acquire_threads(10), 3);
    }

    #[test]
    fn empty_data_and_zero_chunks_are_fine() {
        let cancel = AtomicBool::new(false);
        let b = budget(4);
        let mut none: [u8; 0] = [];
        for_each_chunk_mut(&b, &cancel, &mut none, 0, |_, _| Ok(())).unwrap();
        let mut one = [1u8];
        for_each_chunk_mut(&b, &cancel, &mut one, 0, |_, c| {
            c[0] = 9;
            Ok(())
        })
        .unwrap();
        assert_eq!(one, [9]);
    }

    #[test]
    fn a_budget_with_room_for_fewer_threads_gets_fewer() {
        let cancel = AtomicBool::new(false);
        for (room, peak) in [(0, 1), (1, 2), (2, 3), (50, 4)] {
            let b = Budget::new(&Limits {
                max_alloc: room * THREAD_BYTES + THREAD_BYTES - 1,
                max_threads: 4,
                pool: Some(caller_pool()),
                ..Limits::default()
            });
            let mut data = vec![0u8; 64];
            for_each_chunk_mut(&b, &cancel, &mut data, 8, |_, c| {
                c.fill(1);
                Ok(())
            })
            .unwrap();
            // The workers' charge is still there: it stays until the call ends.
            let held = (peak - 1) * THREAD_BYTES;
            assert_eq!(
                (data, b.peak_threads(), b.used()),
                (vec![1; 64], peak, held)
            );
            assert_eq!(b.acquire_threads(10), 3);
        }
    }

    #[test]
    fn rows_are_banded_by_work_and_thread_limit() {
        let cancel = AtomicBool::new(false);
        let rows = |threads: usize, work: usize| {
            let b = budget(threads);
            let mut data = vec![0usize; 1000 * 3];
            for_each_row_mut(&b, &cancel, work, &mut data, 3, |y, row| {
                row.fill(y + 1);
                Ok(())
            })
            .unwrap();
            let want: Vec<usize> = (0..1000).flat_map(|y| [y + 1; 3]).collect();
            assert_eq!(data, want, "{threads} threads, work {work}");
            b.peak_threads()
        };
        // Under the minimum, one thread does it all.
        assert_eq!(rows(8, MIN_WORK_PER_THREAD - 1), 1);
        // One thread per minimum, up to the limit.
        assert_eq!(rows(8, 3 * MIN_WORK_PER_THREAD), 3);
        assert_eq!(rows(2, 100 * MIN_WORK_PER_THREAD), 2);
        assert_eq!(rows(1, 100 * MIN_WORK_PER_THREAD), 1);
    }
}
