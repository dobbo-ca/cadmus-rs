//! `Recognizer::recognize_into` on a `Workspace` and a `Line` the caller keeps: when both calls
//! succeed the result is the one `recognize` gives, byte for byte, for the sequences of lines,
//! thread counts, lexicon switches and failed calls tried here; a call on a warm workspace makes no
//! allocator call with `max_threads` 1 after a few calls; and `max_alloc` is still the peak of each
//! call, whatever the workspace held: a limit a fresh call passes, a warm one passes (the converse
//! is not promised: a warm call grows less).
//!
//! The counting allocator is per thread (the test harness runs the tests of this file in
//! parallel), so each test counts only its own calls. The model tests skip when
//! `tests/data/eng.traineddata` or the line corpus is absent (CI has them).
#![allow(unsafe_code)]

use cadmus::imaging::Bitmap;
use cadmus::recog::{Line, Recognizer, Workspace};
use cadmus::{Error, Limits};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::AtomicBool;

thread_local! {
    /// Whether this thread's allocator calls are counted now, and how many there were.
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
}

/// Counts every alloc, realloc and dealloc of a thread that has switched `COUNTING` on.
struct Counting;

fn count() {
    // `try_with`: the thread-local may be gone while the thread exits.
    let _ = COUNTING.try_with(|on| {
        if on.get() {
            let _ = CALLS.try_with(|n| n.set(n.get() + 1));
        }
    });
}

// SAFETY: every method calls the System allocator with the arguments it got and returns its
// result unchanged, so the contract of `GlobalAlloc` is the one System keeps; the counting only
// touches a thread-local `Cell` of plain integers, which does not allocate.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: the caller upholds the contract of `alloc`, which is System's.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: the caller upholds the contract of `alloc_zeroed`, which is System's.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        count();
        // SAFETY: `ptr` and `layout` are the pair the caller got from this allocator, which got
        // them from System.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count();
        // SAFETY: as in `dealloc`, and `new_size` is the caller's, valid by the same contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// The allocator calls `f` makes on this thread.
fn calls_in<T>(f: impl FnOnce() -> T) -> (T, usize) {
    CALLS.set(0);
    COUNTING.set(true);
    let r = f();
    COUNTING.set(false);
    (r, CALLS.get())
}

fn dir() -> String {
    env!("CARGO_MANIFEST_DIR").to_string()
}

/// The model's bytes, or `None` (and a message) without the model.
fn model() -> Option<Vec<u8>> {
    let m = std::fs::read(format!("{}/tests/data/eng.traineddata", dir())).ok();
    if m.is_none() {
        eprintln!("eng.traineddata not present; skipping");
    }
    m
}

/// Up to `n` lines of each corpus arm, narrowest first, so that a walk over them shrinks and
/// grows the line.
fn lines(n: usize) -> Vec<Bitmap> {
    let mut out = Vec::new();
    for set in ["native", "h36"] {
        let Ok(entries) = std::fs::read_dir(format!("{}/tests/data/lines/{set}", dir())) else {
            continue;
        };
        let mut names: Vec<String> = entries
            .filter_map(|e| e.ok()?.path().to_str().map(str::to_string))
            .filter(|p| p.ends_with(".pgm"))
            .collect();
        names.sort();
        for p in names.iter().take(n) {
            let data = std::fs::read(p).expect("a line");
            out.push(Bitmap::from_pgm(&data, &Limits::default()).expect("a PGM line"));
        }
    }
    out.sort_by_key(|b| b.width * b.height);
    out
}

fn recognizer(model: &[u8]) -> Recognizer {
    Recognizer::new(model, &Limits::default(), &AtomicBool::new(false)).expect("the model loads")
}

/// Whether two lines are the same, floats to the bit.
fn same(a: &Line, b: &Line) -> bool {
    a.text == b.text
        && a.bounds == b.bounds
        && a.confidence.to_bits() == b.confidence.to_bits()
        && a.words.len() == b.words.len()
        && a.words.iter().zip(&b.words).all(|(x, y)| {
            x.text == y.text
                && x.bounds == y.bounds
                && x.confidence.to_bits() == y.confidence.to_bits()
        })
}

fn empty_line() -> Line {
    Line {
        text: String::new(),
        words: Vec::new(),
        bounds: Default::default(),
        confidence: 0.0,
    }
}

#[test]
fn a_reused_workspace_and_line_give_the_fresh_result() {
    let Some(model) = model() else { return };
    let imgs = lines(3);
    if imgs.len() < 2 {
        eprintln!("the line corpus is not present; skipping");
        return;
    }
    // Growing, shrinking and repeating: narrowest to widest, back, and the same line twice.
    let n = imgs.len();
    let order: Vec<usize> = (0..n).chain((0..n).rev()).chain([0, 0, n - 1]).collect();
    let cancel = AtomicBool::new(false);
    for threads in [1, 4] {
        let limits = Limits {
            max_threads: threads,
            ..Limits::default()
        };
        // Two recognizers of the same model draw the same sequence of random border taps, so the
        // fresh and the reused call of each step see the same generator.
        let (fresh, reused) = (recognizer(&model), recognizer(&model));
        let (mut ws, mut line) = (Workspace::default(), empty_line());
        for (step, &i) in order.iter().enumerate() {
            let want = fresh.recognize(&imgs[i], &limits, &cancel).expect("fresh");
            reused
                .recognize_into(&mut ws, &imgs[i], &mut line, &limits, &cancel)
                .expect("reused");
            assert!(
                same(&want, &line),
                "{threads} threads, step {step} (line {i}): {want:?} vs {line:?}"
            );
        }
    }
}

#[test]
fn a_call_on_a_warm_workspace_allocates_nothing() {
    let Some(model) = model() else { return };
    let imgs = lines(1);
    if imgs.is_empty() {
        eprintln!("the line corpus is not present; skipping");
        return;
    }
    let one = Limits {
        max_threads: 1,
        ..Limits::default()
    };
    let cancel = AtomicBool::new(false);
    for (k, img) in imgs.iter().enumerate() {
        let r = recognizer(&model);
        let (mut ws, mut line) = (Workspace::default(), empty_line());
        // The network's random border taps move the beam a little from call to call, so the
        // workspace is warmed with a few calls before one is counted.
        for _ in 0..3 {
            r.recognize_into(&mut ws, img, &mut line, &one, &cancel)
                .expect("warm-up");
        }
        let (res, calls) = calls_in(|| r.recognize_into(&mut ws, img, &mut line, &one, &cancel));
        res.expect("the counted call");
        assert_eq!(calls, 0, "line {k}: allocator calls in a warm call");
    }
}

/// The smallest `max_alloc` a fresh call passes with, by bisection.
fn smallest(r: &Recognizer, img: &Bitmap) -> usize {
    let cancel = AtomicBool::new(false);
    let (mut lo, mut hi) = (0usize, 256 << 20);
    while hi - lo > 4096 {
        let mid = lo + (hi - lo) / 2;
        let l = Limits {
            max_alloc: mid,
            max_threads: 1,
            ..Limits::default()
        };
        match r.recognize(img, &l, &cancel) {
            Ok(_) => hi = mid,
            Err(Error::AllocLimit) => lo = mid,
            Err(e) => panic!("max_alloc {mid} gave `{e}`"),
        }
    }
    hi
}

#[test]
fn max_alloc_is_the_peak_of_each_call_whatever_the_workspace_held() {
    let Some(model) = model() else { return };
    let imgs = lines(2);
    if imgs.len() < 2 {
        eprintln!("the line corpus is not present; skipping");
        return;
    }
    let (small, big) = (&imgs[0], &imgs[imgs.len() - 1]);
    let cancel = AtomicBool::new(false);
    let r = recognizer(&model);
    // What the small line needs, with some room (the border taps make the need vary a little).
    let min = smallest(&r, small);
    let tight = Limits {
        max_alloc: min + (min >> 3),
        max_threads: 1,
        ..Limits::default()
    };
    let roomy = Limits {
        max_threads: 1,
        ..Limits::default()
    };

    // A workspace that served the big line holds more than the tight limit: the call frees it and
    // runs as a fresh one, which passes.
    let (mut ws, mut line) = (Workspace::default(), empty_line());
    r.recognize_into(&mut ws, big, &mut line, &roomy, &cancel)
        .expect("the big line");
    r.recognize_into(&mut ws, small, &mut line, &tight, &cancel)
        .expect("the small line on a workspace that held the big one");
    let fresh = recognizer(&model);
    // (`fresh` has not drawn what `r` has, so only the shape of the result is compared.)
    let want = fresh
        .recognize(small, &tight, &cancel)
        .expect("a fresh call");
    assert_eq!(line.words.len(), want.words.len());

    // A limit the fresh call fails with fails the same way on a warm workspace, which is then
    // empty and still usable.
    let tiny = Limits {
        max_alloc: min / 2,
        max_threads: 1,
        ..Limits::default()
    };
    assert_eq!(
        fresh.recognize(small, &tiny, &cancel).unwrap_err(),
        Error::AllocLimit
    );
    r.recognize_into(&mut ws, small, &mut line, &roomy, &cancel)
        .expect("a warm workspace");
    assert_eq!(
        r.recognize_into(&mut ws, small, &mut line, &tiny, &cancel),
        Err(Error::AllocLimit)
    );
    // Empty: a warm call that needs memory must make allocator calls again.
    let (res, calls) = calls_in(|| r.recognize_into(&mut ws, small, &mut line, &roomy, &cancel));
    res.expect("the workspace is usable after AllocLimit");
    assert!(calls > 0, "the workspace was not emptied by AllocLimit");
    // And is warm again (after a few calls: see the other test).
    for _ in 0..3 {
        r.recognize_into(&mut ws, small, &mut line, &roomy, &cancel)
            .expect("a warm-up");
    }
    let (res, calls) = calls_in(|| r.recognize_into(&mut ws, small, &mut line, &roomy, &cancel));
    res.expect("a warm call");
    assert_eq!(calls, 0, "allocator calls in a warm call after AllocLimit");
}

/// The warm retry of `recognize_into`: the workspace the call starts with fits `max_alloc`, the
/// call then grows past it and reruns on an empty workspace from the generator state it started
/// with. The result must be the one a fresh call gives, from the same generator state, for a limit
/// the fresh call passes. (Which branch ran is not observable here; the limits sweep the
/// neighbourhood of each.)
#[test]
fn a_limit_a_fresh_call_passes_is_passed_by_a_warm_workspace_with_the_same_result() {
    let Some(model) = model() else { return };
    let imgs = lines(2);
    if imgs.len() < 2 {
        eprintln!("the line corpus is not present; skipping");
        return;
    }
    let (small, big) = (&imgs[0], &imgs[imgs.len() - 1]);
    let cancel = AtomicBool::new(false);
    let roomy = Limits {
        max_threads: 1,
        ..Limits::default()
    };
    let min = smallest(&recognizer(&model), big);
    for slack in [0, min >> 4, min >> 3, min >> 1] {
        let limits = Limits {
            max_alloc: min + slack,
            max_threads: 1,
            ..Limits::default()
        };
        // Two recognizers of one model draw the same border taps; both see the same calls.
        let (fresh, reused) = (recognizer(&model), recognizer(&model));
        let (mut ws, mut line) = (Workspace::default(), empty_line());
        let _ = fresh
            .recognize(small, &roomy, &cancel)
            .expect("fresh warm-up");
        reused
            .recognize_into(&mut ws, small, &mut line, &roomy, &cancel)
            .expect("reused warm-up");
        // The workspace holds the small line's buffers, which fit; the big line grows them.
        let want = fresh.recognize(big, &limits, &cancel);
        let got = reused.recognize_into(&mut ws, big, &mut line, &limits, &cancel);
        if let Ok(w) = want {
            assert_eq!(got, Ok(()), "slack {slack}: a fresh call passed");
            assert!(same(&w, &line), "slack {slack}: {w:?} vs {line:?}");
        }
    }
}

/// Lexicon switches and failed calls between good ones leave the reused workspace and the
/// generator in step with a fresh call per image.
#[test]
fn lexicon_switches_and_failed_calls_between_reused_calls_give_the_fresh_result() {
    let Some(model) = model() else { return };
    let imgs = lines(4);
    if imgs.len() < 2 {
        eprintln!("the line corpus is not present; skipping");
        return;
    }
    let limits = Limits {
        max_threads: 1,
        ..Limits::default()
    };
    let (go, stop) = (AtomicBool::new(false), AtomicBool::new(true));
    let (mut fresh, mut reused) = (recognizer(&model), recognizer(&model));
    let (mut ws, mut line) = (Workspace::default(), empty_line());
    for (step, img) in imgs.iter().chain(imgs.iter().rev()).enumerate() {
        let lexicon = step % 3 != 1;
        fresh.set_lexicon(lexicon);
        reused.set_lexicon(lexicon);
        if step % 2 == 1 {
            let a = fresh.recognize(img, &limits, &stop);
            let b = reused.recognize_into(&mut ws, img, &mut line, &limits, &stop);
            assert_eq!(a.err(), b.err(), "step {step}: a cancelled call");
        }
        let want = fresh.recognize(img, &limits, &go).expect("fresh");
        reused
            .recognize_into(&mut ws, img, &mut line, &limits, &go)
            .expect("reused");
        assert!(same(&want, &line), "step {step}: {want:?} vs {line:?}");
    }
}

/// The warm retry itself (forced with the hidden hook, since a real `AllocLimit` that only the
/// warm attempt hits depends on the buffers' shapes): the first attempt runs and draws the border
/// taps, the retry rewinds the generator and reruns on an empty workspace, and the result and every
/// later call must equal fresh calls of a recognizer in lockstep. With 1 thread and with 4 on a
/// per-call pool (the retry must reuse the first attempt's workers).
#[test]
fn a_forced_warm_retry_gives_the_fresh_result() {
    let Some(model) = model() else { return };
    let imgs = lines(3);
    if imgs.len() < 2 {
        eprintln!("the line corpus is not present; skipping");
        return;
    }
    let cancel = AtomicBool::new(false);
    for threads in [1, 4] {
        let limits = Limits {
            max_threads: threads,
            per_call_pool: threads > 1,
            ..Limits::default()
        };
        let (fresh, reused) = (recognizer(&model), recognizer(&model));
        let (mut ws, mut line) = (Workspace::default(), empty_line());
        for (i, img) in imgs.iter().chain(imgs.iter()).enumerate() {
            if i > 0 {
                ws.force_retry();
            }
            let want = fresh.recognize(img, &limits, &cancel).expect("fresh");
            reused
                .recognize_into(&mut ws, img, &mut line, &limits, &cancel)
                .expect("reused");
            assert!(
                same(&want, &line),
                "call {i}, {threads} threads: {want:?} vs {line:?}"
            );
            assert_eq!(ws.retries(), i as u64, "call {i}: the retry ran");
        }
    }
}
