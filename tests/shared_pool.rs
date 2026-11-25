//! The pools: `max_threads` 1 never touches the crate's shared pool, not even to build it, and a
//! caller's pool always wins over it. This test binary has one test and nothing else in it calls
//! the crate, so nothing can have built the shared pool before the checks, and the last step
//! (a call with no pool and `max_threads` above 1) shows that the checks could fail.

use cadmus::imaging::{self, Bitmap};
use cadmus::recog::Recognizer;
use cadmus::{Limits, ThreadPool, shared_pool_started};
use std::sync::atomic::AtomicBool;

/// A noisy gray plane of 524,288 pixels, so every operator splits into bands.
fn gray() -> Bitmap {
    let (w, h) = (1024, 512);
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    let pix: Vec<u8> = (0..w * h)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 32) as u8
        })
        .collect();
    Bitmap::from_gray8(w, h, &pix, &Limits::default()).expect("a small image")
}

/// Every threaded imaging stage, then the recognizer when the model and a line are present.
fn run(l: &Limits, g: &Bitmap, rec: Option<(&Recognizer, &Bitmap)>) {
    let cancel = AtomicBool::new(false);
    let bin = imaging::otsu(g, l, &cancel).expect("otsu");
    imaging::scale_gray(g, 0.5, l, &cancel).expect("scale_gray");
    imaging::scale_gray(g, 1.7, l, &cancel).expect("scale_gray");
    imaging::sauvola(g, 25, 0.34, l, &cancel).expect("sauvola");
    imaging::open(&bin, 5, 5, l, &cancel).expect("open");
    imaging::close(&bin, 5, 5, l, &cancel).expect("close");
    imaging::rotate(&bin, 0.03, l, &cancel).expect("rotate");
    imaging::skew_angle(&bin, l, &cancel).expect("skew_angle");
    if let Some((r, line)) = rec {
        r.recognize(line, l, &cancel).expect("recognize");
    }
}

#[test]
fn one_thread_and_a_caller_pool_leave_the_shared_pool_unbuilt() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let g = gray();
    // The recognizer's fully connected layers split their timesteps too.
    let model = std::fs::read(format!("{dir}/tests/data/eng.traineddata")).ok();
    let line = std::fs::read(format!("{dir}/tests/data/lines/h36/0001.pgm"))
        .ok()
        .and_then(|d| Bitmap::from_pgm(&d, &Limits::default()).ok());
    let one = Limits {
        max_threads: 1,
        ..Limits::default()
    };
    let r = model.and_then(|m| Recognizer::new(&m, &one, &AtomicBool::new(false)).ok());
    let rec = r.as_ref().zip(line.as_ref());

    run(&one, &g, rec);
    assert!(
        !shared_pool_started(),
        "max_threads 1 built the shared pool"
    );

    // A pool of the caller's, at any max_threads, and one thread with it: not the shared pool.
    let pool = ThreadPool::new(2).expect("a pool of 2 threads");
    for max_threads in [1, 2, 4] {
        let l = Limits {
            max_threads,
            pool: Some(pool.clone()),
            ..Limits::default()
        };
        run(&l, &g, rec);
        assert!(
            !shared_pool_started(),
            "a caller pool with max_threads {max_threads} built the shared pool"
        );
    }

    // With no pool and threads to use, the shared pool is built.
    let many = Limits {
        max_threads: 4,
        ..Limits::default()
    };
    run(&many, &g, rec);
    assert!(shared_pool_started(), "the shared pool was never built");
}
