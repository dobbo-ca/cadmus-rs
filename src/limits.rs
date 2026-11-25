use crate::Error;
use crate::par::{self, ThreadPool};
use std::ops::{Deref, DerefMut};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Resource limits for one call. Go had none; the defaults follow the cadmus memory-limits
/// guide (docs/guides/memory-limits.md): a 64 MiB model, a 1<<27 pixel plane.
///
/// `max_alloc` and `max_threads` are the two controls for a caller in a small pod: with
/// `max_threads = 2` and `max_alloc = 768 << 20` a call never uses more than two threads or
/// 768 MiB of working memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Limits {
    /// The peak working memory of one call, in bytes, all threads together. The default is 1 GiB,
    /// this crate's choice. Every allocation that grows with the input is charged to one
    /// [`Budget`] before it is made and released when it is freed, so the sum of what the call
    /// holds at once never goes above this value, whatever steps the call runs. Past it the call
    /// returns `Error::AllocLimit`. The model a call returns, and the input the caller passed in,
    /// are not part of it.
    pub max_alloc: usize,
    /// The most OS threads one call uses, the calling thread included. 1 runs everything on the
    /// calling thread, never touches a pool and starts no thread. The default is
    /// `std::thread::available_parallelism()`, or 1 when that is unknown. Needing more is never an
    /// error: a call just uses fewer threads (each extra thread is charged 4096 bytes to
    /// `max_alloc` from its first stage until the call returns, so a small budget also limits
    /// it, and a budget at the serial minimum needs those bytes on top). The threads are workers
    /// of a pool: see `pool`.
    pub max_threads: usize,
    /// The worker threads the call's parallel stages run on, besides the calling thread: made
    /// once with [`ThreadPool::new`] and kept, so that calls reuse threads instead of starting
    /// them. It always wins. `None` (the default) means the crate's shared pool: built lazily by
    /// the first call that has a parallel stage and `max_threads` above 1, with
    /// `available_parallelism` threads and never more, and holding no other state. If that
    /// cannot be built, the call builds one pool of its own for all its stages. `max_threads` 1
    /// never touches any pool. The output is the same bytes whichever pool runs the call.
    pub pool: Option<ThreadPool>,
    /// For tests: skip the shared pool and build the call's own workers (see `pool`).
    #[doc(hidden)]
    pub per_call_pool: bool,
    /// Size of a `.traineddata` file, in bytes. Default 64 MiB. Over it: `Error::ModelLimit`.
    pub max_model_bytes: usize,
    /// Pixels (`width * height`) of an image plane. Default 1<<27. Over it: `Error::PixelLimit`.
    pub max_pixels: u64,
    /// Width of an image plane, in pixels. Default 1<<16, the line width of the guide. Over it:
    /// `Error::PixelLimit`.
    pub max_width: usize,
    /// Height of an image plane, in pixels. Default 1<<16, this crate's choice (the guide has no
    /// height cap). Over it: `Error::PixelLimit`.
    pub max_height: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_alloc: 1 << 30,
            max_threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
            pool: None,
            per_call_pool: false,
            max_model_bytes: 64 << 20,
            max_pixels: 1 << 27,
            max_width: 1 << 16,
            max_height: 1 << 16,
        }
    }
}

impl Limits {
    /// Checks a plane's dimensions against the pixel, width and height caps.
    ///
    /// # Errors
    ///
    /// `Error::PixelLimit` if `width` or `height` is over its cap or `width * height` is over
    /// `max_pixels`.
    pub fn check_plane(&self, width: usize, height: usize) -> Result<(), Error> {
        if width > self.max_width || height > self.max_height {
            return Err(Error::PixelLimit);
        }
        match (width as u64).checked_mul(height as u64) {
            Some(n) if n <= self.max_pixels => Ok(()),
            _ => Err(Error::PixelLimit),
        }
    }
}

/// The ledger of one call: the bytes it holds against `Limits::max_alloc`, and the threads it
/// uses against `Limits::max_threads`.
///
/// A public call makes one `Budget` and passes it by reference to every step it runs, on any
/// thread (it is `Sync`). Memory is charged before it is allocated and released when it is
/// freed (a [`Buf`] or [`Hold`] does that on drop), so the ledger holds the live total, and
/// [`Budget::peak`] is the most it ever held. A charge is made once per buffer, never per
/// element, so a hot loop does not touch the atomics.
#[derive(Debug)]
pub struct Budget {
    limits: Limits,
    used: AtomicUsize,
    peak: AtomicUsize,
    threads: AtomicUsize,
    peak_threads: AtomicUsize,
    /// The most workers the call has used at once, each charged `par::THREAD_BYTES` to `used`
    /// until the ledger drops: a worker's job box can outlive its stage, so the charge cannot.
    workers: AtomicUsize,
    /// The workers the call built for itself, when it has no caller pool and the shared one is
    /// not used or could not be built; made at the first parallel stage that needs them.
    own_pool: OnceLock<Option<ThreadPool>>,
}

impl Budget {
    /// A ledger with nothing charged and one thread (the caller) in use.
    pub fn new(limits: &Limits) -> Budget {
        Budget {
            limits: limits.clone(),
            used: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            threads: AtomicUsize::new(1),
            peak_threads: AtomicUsize::new(1),
            workers: AtomicUsize::new(0),
            own_pool: OnceLock::new(),
        }
    }

    /// The limits this ledger was made from.
    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    /// The pool the call's parallel stages run on: the caller's, else the crate's shared one,
    /// else (if that cannot be built, or `Limits::per_call_pool` is set) one built now for the
    /// whole call. `None` for `max_threads` 1, which never touches or builds any pool, and when
    /// no pool can be built: the call then runs on its own thread.
    pub(crate) fn pool(&self) -> Option<&ThreadPool> {
        let l = &self.limits;
        if l.max_threads <= 1 {
            return None;
        }
        if let Some(p) = &l.pool {
            return Some(p);
        }
        if !l.per_call_pool
            && let Some(p) = par::shared_pool()
        {
            return Some(p);
        }
        // The caller is one of the `max_threads` threads, and no more workers than CPUs are made.
        let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
        let workers = l.max_threads.min(cpus).saturating_sub(1);
        if workers == 0 {
            return None;
        }
        self.own_pool
            .get_or_init(|| ThreadPool::new(workers).ok())
            .as_ref()
    }

    /// The workers this call built for itself, if it built any (see [`Budget::pool`]). Never
    /// builds them.
    pub(crate) fn own_pool_built(&self) -> Option<ThreadPool> {
        self.own_pool.get().and_then(Clone::clone)
    }

    /// Charges `bytes`; `Error::AllocLimit` if the ledger would go over `max_alloc`.
    ///
    /// # Errors
    ///
    /// `Error::AllocLimit` if the total would go over `max_alloc`; nothing is charged then.
    pub fn charge(&self, bytes: usize) -> Result<(), Error> {
        let mut cur = self.used.load(Ordering::Relaxed);
        loop {
            let new = cur
                .checked_add(bytes)
                .filter(|&n| n <= self.limits.max_alloc)
                .ok_or(Error::AllocLimit)?;
            match self
                .used
                .compare_exchange_weak(cur, new, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => {
                    self.peak.fetch_max(new, Ordering::Relaxed);
                    return Ok(());
                }
                Err(c) => cur = c,
            }
        }
    }

    /// Returns `bytes` that were freed.
    pub fn release(&self, bytes: usize) {
        let mut cur = self.used.load(Ordering::Relaxed);
        while let Err(c) = self.used.compare_exchange_weak(
            cur,
            cur.saturating_sub(bytes),
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            cur = c;
        }
    }

    /// The bytes charged now.
    pub fn used(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }

    /// The most bytes the ledger has held at once.
    pub fn peak(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }

    /// The most threads the call has used at once, the calling thread included.
    pub fn peak_threads(&self) -> usize {
        self.peak_threads.load(Ordering::Relaxed)
    }

    /// Takes up to `want` threads beyond the ones in use, so that the call never uses more than
    /// `max_threads`, and returns how many it got. The caller hands them back with
    /// `release_threads`.
    pub(crate) fn acquire_threads(&self, want: usize) -> usize {
        let max = self.limits.max_threads.max(1);
        let mut cur = self.threads.load(Ordering::Relaxed);
        loop {
            let got = want.min(max.saturating_sub(cur));
            if got == 0 {
                return 0;
            }
            match self.threads.compare_exchange_weak(
                cur,
                cur + got,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.peak_threads.fetch_max(cur + got, Ordering::Relaxed);
                    return got;
                }
                Err(c) => cur = c,
            }
        }
    }

    pub(crate) fn release_threads(&self, n: usize) {
        self.threads.fetch_sub(n, Ordering::Relaxed);
    }

    /// Makes sure `n` workers are charged, `bytes` each, and keeps the charge until the ledger
    /// drops, even when the stage that used them has ended: a pool worker frees its job box after
    /// the stage's latch is set, so the call may already be allocating again while that box is
    /// live. Only workers past the most the call has used so far are charged.
    ///
    /// # Errors
    ///
    /// `Error::AllocLimit` if the extra workers do not fit; nothing is charged then.
    pub(crate) fn hold_workers(&self, n: usize, bytes: usize) -> Result<(), Error> {
        let mut cur = self.workers.load(Ordering::Relaxed);
        while n > cur {
            self.charge((n - cur) * bytes)?;
            match self
                .workers
                .compare_exchange(cur, n, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => return Ok(()),
                Err(c) => {
                    self.release((n - cur) * bytes);
                    cur = c;
                }
            }
        }
        Ok(())
    }

    /// The workers whose charge the ledger holds (see `hold_workers`).
    pub(crate) fn workers_held(&self) -> usize {
        self.workers.load(Ordering::Relaxed)
    }

    /// Charges `n * size_of::<T>()` and returns `n` copies of `v` without aborting on failure.
    /// The caller releases the charge when it frees the vec; a [`Buf`] does it by itself.
    ///
    /// # Errors
    ///
    /// `Error::AllocLimit` over `max_alloc` (or on overflow), `Error::OutOfMemory` if the
    /// allocator refuses.
    pub fn vec<T: Clone>(&self, n: usize, v: T) -> Result<Vec<T>, Error> {
        let bytes = n
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(Error::AllocLimit)?;
        self.charge(bytes)?;
        let mut out = Vec::new();
        if out.try_reserve_exact(n).is_err() {
            self.release(bytes);
            return Err(Error::OutOfMemory);
        }
        out.resize(n, v);
        Ok(out)
    }

    /// Charges the room for `additional` more elements and reserves exactly that. The charge is
    /// the caller's to release. A vec that grows may move, and its old block is live until the
    /// new one is made, so the old capacity is charged on top while the call runs and given back
    /// when it returns.
    ///
    /// # Errors
    ///
    /// `Error::AllocLimit` over `max_alloc` (or on overflow), `Error::OutOfMemory` if the
    /// allocator refuses.
    pub fn reserve<T>(&self, v: &mut Vec<T>, additional: usize) -> Result<(), Error> {
        let bytes = additional
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(Error::AllocLimit)?;
        let old = v.capacity() * std::mem::size_of::<T>();
        let total = bytes.checked_add(old).ok_or(Error::AllocLimit)?;
        self.charge(total)?;
        if v.try_reserve_exact(additional).is_err() {
            self.release(total);
            return Err(Error::OutOfMemory);
        }
        self.release(old);
        Ok(())
    }

    /// `v.push(item)`, charging the room when the vec has to grow (it doubles, at least 4).
    /// The charge is the caller's to release. While it grows the old block is charged too (see
    /// [`Budget::reserve`]).
    ///
    /// # Errors
    ///
    /// As [`Budget::reserve`], when the vec has to grow.
    pub fn push<T>(&self, v: &mut Vec<T>, item: T) -> Result<(), Error> {
        if v.len() == v.capacity() {
            let additional = v.capacity().max(4);
            self.reserve(v, additional)?;
        }
        v.push(item);
        Ok(())
    }

    /// Charges `bytes` until the returned guard is dropped.
    ///
    /// # Errors
    ///
    /// `Error::AllocLimit` if the total would go over `max_alloc`.
    pub fn hold(&self, bytes: usize) -> Result<Hold<'_>, Error> {
        self.charge(bytes)?;
        Ok(Hold {
            budget: self,
            bytes,
        })
    }

    /// `n` copies of `v` in a vec that is charged until the [`Buf`] is dropped.
    ///
    /// # Errors
    ///
    /// `Error::AllocLimit` over `max_alloc` (or on overflow), `Error::OutOfMemory` if the
    /// allocator refuses.
    pub fn buf<T: Clone>(&self, n: usize, v: T) -> Result<Buf<'_, T>, Error> {
        let bytes = n
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(Error::AllocLimit)?;
        let hold = self.hold(bytes)?;
        let mut out = Vec::new();
        out.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
        out.resize(n, v);
        Ok(Buf {
            v: out,
            _hold: hold,
        })
    }
}

/// A charge on a [`Budget`] that is released when the guard is dropped. Declare it before the
/// memory it covers, so the memory is freed first.
#[derive(Debug)]
pub struct Hold<'a> {
    budget: &'a Budget,
    bytes: usize,
}

impl Drop for Hold<'_> {
    fn drop(&mut self) {
        self.budget.release(self.bytes);
    }
}

/// A vec whose bytes are charged to a [`Budget`] until it is dropped. It must not grow past the
/// length it was made with.
#[derive(Debug)]
pub struct Buf<'a, T> {
    // The vec is the first field, so it is freed before the charge is released.
    v: Vec<T>,
    _hold: Hold<'a>,
}

impl<T> Deref for Buf<'_, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.v
    }
}

impl<T> DerefMut for Buf<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(max_alloc: usize, max_threads: usize) -> Limits {
        Limits {
            max_alloc,
            max_threads,
            ..Limits::default()
        }
    }

    #[test]
    fn default_threads_is_at_least_one() {
        assert!(Limits::default().max_threads >= 1);
    }

    #[test]
    fn charge_release_and_peak() {
        let b = Budget::new(&limits(100, 1));
        b.charge(60).unwrap();
        assert_eq!(b.charge(41), Err(Error::AllocLimit));
        b.release(30);
        b.charge(70).unwrap();
        assert_eq!((b.used(), b.peak()), (100, 100));
        b.release(1000);
        assert_eq!((b.used(), b.peak()), (0, 100));
        assert_eq!(b.charge(usize::MAX), Err(Error::AllocLimit));
    }

    #[test]
    fn buf_and_hold_release_on_drop() {
        let b = Budget::new(&limits(100, 1));
        {
            let mut buf = b.buf(10, 0u64).unwrap();
            buf[3] = 7;
            assert_eq!((buf.len(), b.used()), (10, 80));
            assert_eq!(b.buf(3, 0u64).unwrap_err(), Error::AllocLimit);
            let _h = b.hold(20).unwrap();
            assert_eq!(b.used(), 100);
        }
        assert_eq!((b.used(), b.peak()), (0, 100));
    }

    #[test]
    fn push_and_reserve_charge_the_capacity() {
        let b = Budget::new(&limits(1 << 20, 1));
        let mut v: Vec<u64> = Vec::new();
        for i in 0..100 {
            b.push(&mut v, i).unwrap();
        }
        // The charge covers every byte the vec holds.
        assert!(b.used() >= v.capacity() * 8);
        assert_eq!(b.reserve(&mut v, usize::MAX), Err(Error::AllocLimit));
    }

    // A growing vec may move: the old block and the new one are live together, so the ledger
    // must hold both for the moment, and only the new one after.
    #[test]
    fn growing_charges_the_old_block_while_it_is_copied() {
        let b = Budget::new(&limits(1 << 20, 1));
        let mut v: Vec<u64> = Vec::new();
        b.reserve(&mut v, 100).unwrap();
        assert_eq!((b.used(), b.peak()), (800, 800));
        let cap = v.capacity();
        v.resize(cap, 0);
        b.push(&mut v, 1).unwrap();
        // At the peak the ledger held the old block (800) and the new one (800) on top of the
        // 800 charged first; after, the old block's extra charge is gone.
        assert_eq!((b.peak(), b.used()), (2400, 1600));
        // A limit that holds the result but not the transient is refused.
        let tight = Budget::new(&limits(2000, 1));
        let mut w: Vec<u64> = Vec::new();
        tight.reserve(&mut w, 100).unwrap();
        w.resize(w.capacity(), 0);
        assert_eq!(tight.push(&mut w, 1), Err(Error::AllocLimit));
    }

    #[test]
    fn the_ledger_is_shared_by_threads() {
        let b = Budget::new(&limits(1000, 4));
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    for _ in 0..100 {
                        let _h = b.hold(10).unwrap();
                    }
                });
            }
        });
        assert_eq!(b.used(), 0);
        assert!(b.peak() <= 40);
    }

    #[test]
    fn workers_are_charged_once_and_kept() {
        let b = Budget::new(&limits(10_000, 8));
        b.hold_workers(2, 1000).unwrap();
        assert_eq!((b.used(), b.workers_held()), (2000, 2));
        // Fewer or as many workers cost nothing more; more cost the difference.
        b.hold_workers(1, 1000).unwrap();
        b.hold_workers(2, 1000).unwrap();
        assert_eq!(b.used(), 2000);
        b.hold_workers(5, 1000).unwrap();
        assert_eq!((b.used(), b.workers_held()), (5000, 5));
        // Past `max_alloc` nothing is charged.
        assert_eq!(b.hold_workers(11, 1000), Err(Error::AllocLimit));
        assert_eq!((b.used(), b.workers_held()), (5000, 5));
    }

    #[test]
    fn the_calls_own_pool_has_no_more_workers_than_cpus_less_the_caller() {
        let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
        let b = Budget::new(&Limits {
            per_call_pool: true,
            ..limits(1 << 20, 8)
        });
        // On one CPU the caller is the only thread: no pool, not a pool of one.
        match b.pool() {
            None => assert_eq!(cpus.min(8), 1),
            Some(p) => assert_eq!(p.threads(), cpus.min(8) - 1),
        }
    }

    #[test]
    fn threads_are_capped_by_max_threads() {
        let b = Budget::new(&limits(1, 3));
        assert_eq!(b.acquire_threads(10), 2);
        assert_eq!(b.acquire_threads(1), 0);
        b.release_threads(2);
        assert_eq!(b.acquire_threads(1), 1);
        assert_eq!(b.peak_threads(), 3);
        // A limit of 0 still leaves the calling thread.
        assert_eq!(Budget::new(&limits(1, 0)).acquire_threads(5), 0);
    }
}
