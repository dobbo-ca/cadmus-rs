//! Output is byte-identical for every thread count and every pool: the crate's shared pool, a
//! caller's pool (the same one for all calls, as a server keeps) and workers the call builds for
//! itself, for `max_threads` 1, 2, 3 and the default. Whether a call may use a thread is the
//! ledger's business (see `tests/limits_peak.rs`).

use cadmus::imaging::{self, Bitmap};
use cadmus::{Budget, Error, Limits, ThreadPool, par};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// A noisy gray plane with dark bars, so every operator has work to do and splits into bands.
fn gray(w: usize, h: usize) -> Bitmap {
    let mut x = 0x2545_f491_4f6c_dd1du64;
    let pix: Vec<u8> = (0..w * h)
        .map(|i| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let bar = if (i % w / 9 + i / w / 7).is_multiple_of(5) {
                90
            } else {
                200
            };
            (bar + (x % 40) as usize % 55).min(255) as u8
        })
        .collect();
    Bitmap::from_gray8(w, h, &pix, &Limits::default()).expect("a small image")
}

/// Every operator that runs on more than one thread, as bytes and bits.
fn run(l: &Limits, g: &Bitmap) -> Vec<Vec<u8>> {
    let cancel = AtomicBool::new(false);
    let bin = imaging::otsu(g, l, &cancel).expect("otsu");
    let skew = imaging::skew_angle(&bin, l, &cancel).expect("skew_angle");
    vec![
        vec![imaging::otsu_threshold(g, l, &cancel).expect("otsu_threshold")],
        bin.pix.clone(),
        imaging::sauvola(g, 25, 0.34, l, &cancel)
            .expect("sauvola")
            .pix,
        imaging::scale_gray(g, 0.5, l, &cancel)
            .expect("scale 0.5")
            .pix,
        imaging::scale_gray(g, 1.7, l, &cancel)
            .expect("scale 1.7")
            .pix,
        imaging::scale_gray(g, 2.0, l, &cancel)
            .expect("scale 2")
            .pix,
        imaging::open(&bin, 5, 5, l, &cancel).expect("open").pix,
        imaging::close(&bin, 5, 5, l, &cancel).expect("close").pix,
        imaging::rotate(&bin, 0.03, l, &cancel).expect("rotate").pix,
        skew.to_bits().to_le_bytes().to_vec(),
    ]
}

#[test]
fn every_thread_count_and_pool_gives_the_same_bytes() {
    let g = gray(700, 600);
    let one = Limits {
        max_threads: 1,
        ..Limits::default()
    };
    let want = run(&one, &g);
    let caller = ThreadPool::new(3).expect("a pool of 3 threads");
    let threads = [1, 2, 3, Limits::default().max_threads];
    for t in threads {
        for (name, pool, per_call_pool) in [
            ("shared", None, false),
            ("caller", Some(caller.clone()), false),
            ("per-call", None, true),
        ] {
            let l = Limits {
                max_threads: t,
                pool,
                per_call_pool,
                ..Limits::default()
            };
            assert!(run(&l, &g) == want, "{t} threads, {name} pool");
        }
    }
}

/// Several calls at once on one pool, each with threads of its own, get the bytes a lone call
/// gets: the pool is shared, the work of a call is not.
#[test]
fn concurrent_calls_on_one_pool_agree() {
    let g = gray(700, 600);
    let one = Limits {
        max_threads: 1,
        ..Limits::default()
    };
    let want = run(&one, &g);
    let pool = ThreadPool::new(2).expect("a pool of 2 threads");
    std::thread::scope(|s| {
        for t in 0..4 {
            let (g, want, pool) = (&g, &want, pool.clone());
            s.spawn(move || {
                let l = Limits {
                    max_threads: 3,
                    pool: Some(pool),
                    ..Limits::default()
                };
                assert!(run(&l, g) == *want, "caller {t}");
            });
        }
    });
}

/// A cancel flag that is already set stops the call on every pool.
#[test]
fn a_set_cancel_flag_stops_a_call_on_every_pool() {
    let g = gray(700, 600);
    let cancel = AtomicBool::new(true);
    for (pool, per_call_pool) in [
        (None, false),
        (Some(ThreadPool::new(3).expect("a pool")), false),
        (None, true),
    ] {
        let l = Limits {
            max_threads: 4,
            pool,
            per_call_pool,
            ..Limits::default()
        };
        assert_eq!(
            imaging::otsu(&g, &l, &cancel),
            Err(cadmus::Error::Cancelled)
        );
        assert_eq!(
            imaging::scale_gray(&g, 0.5, &l, &cancel),
            Err(cadmus::Error::Cancelled)
        );
    }
}

/// 600 chunks of about a millisecond each, on `b`'s pool, so a call is long enough to be
/// cancelled while its workers run.
fn slow_stage(b: &Budget, cancel: &AtomicBool) -> Result<(), Error> {
    let mut data = vec![0u8; 600];
    par::for_each_chunk_mut(b, cancel, &mut data, 1, |_, c| {
        std::thread::sleep(Duration::from_millis(1));
        c[0] = 1;
        Ok(())
    })
}

/// The most chunks of a short stage on `b` that run at once.
fn most_at_once(b: &Budget) -> usize {
    let (now, most) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let mut data = vec![0u8; 64];
    par::for_each_chunk_mut(b, &AtomicBool::new(false), &mut data, 1, |_, c| {
        most.fetch_max(now.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(2));
        now.fetch_sub(1, Ordering::SeqCst);
        c[0] = 1;
        Ok(())
    })
    .expect("a stage");
    most.load(Ordering::SeqCst)
}

/// Runs `run` once to time it, then again with `cancel` set from another thread after a tenth of
/// that time: it returns `Cancelled`, in well under a full run.
fn cancel_midway(name: &str, run: impl Fn(&AtomicBool) -> Result<(), Error>) {
    let start = Instant::now();
    run(&AtomicBool::new(false)).expect(name);
    let full = start.elapsed();
    let cancel = AtomicBool::new(false);
    let (r, took) = std::thread::scope(|s| {
        s.spawn(|| {
            std::thread::sleep(full / 10);
            cancel.store(true, Ordering::Relaxed);
        });
        let start = Instant::now();
        (run(&cancel), start.elapsed())
    });
    assert_eq!(r, Err(Error::Cancelled), "{name}");
    assert!(
        took < full / 2,
        "{name}: cancelled after {took:?}, full {full:?}"
    );
}

/// A cancel flag set from another thread while the pool's workers are running stops the call on
/// the shared pool, a caller's and the call's own workers, for 2, 3 and the default `max_threads`;
/// the workers go back to the call's budget.
#[test]
fn cancel_from_another_thread_stops_a_threaded_call() {
    let g = gray(2000, 1500);
    let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
    let caller = ThreadPool::new(3).expect("a pool of 3 threads");
    for t in [2, 3, Limits::default().max_threads] {
        for (pool_name, pool, per_call_pool) in [
            ("shared", None, false),
            ("caller", Some(caller.clone()), false),
            ("per-call", None, true),
        ] {
            let l = Limits {
                max_threads: t,
                pool,
                per_call_pool,
                ..Limits::default()
            };
            let name = format!("{t} threads, {pool_name} pool");
            cancel_midway(&format!("slow stage, {name}"), |c| {
                slow_stage(&Budget::new(&l), c)
            });
            cancel_midway(&format!("sauvola, {name}"), |c| {
                imaging::sauvola(&g, 25, 0.34, &l, c).map(drop)
            });
            // After a cancel the same budget still has its workers to give.
            let b = Budget::new(&l);
            let cancel = AtomicBool::new(false);
            let r = std::thread::scope(|s| {
                s.spawn(|| {
                    std::thread::sleep(Duration::from_millis(20));
                    cancel.store(true, Ordering::Relaxed);
                });
                slow_stage(&b, &cancel)
            });
            assert_eq!(r, Err(Error::Cancelled), "{name}");
            if t >= 2 && (!per_call_pool || cpus >= 2) {
                assert!(
                    most_at_once(&b) >= 2,
                    "{name}: the threads did not come back"
                );
            }
        }
    }
}
