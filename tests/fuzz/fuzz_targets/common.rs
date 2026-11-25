//! Shared by the targets that need a model. The real one (tessdata_best eng, fetched by
//! .github/ci/Dockerfile) is read once and the `Recognizer` built from it is reused across inputs, so the
//! targets also cover a recognizer that has run before (its `Rand` is carried from one call to the
//! next). It takes about a second to run a line through, so most targets use the mini model
//! instead: the real model's charset, recoder and dawgs around a network of a few hundred
//! weights.
//!
//! This file also holds the writer of crafted `lstm` components that the `network` target and the
//! mini model share.
#![allow(dead_code)]

use cadmus::imaging::Bitmap;
use cadmus::recog::Recognizer;
use cadmus::tessdata::{LayerType, Model, Type, parse_container};
use cadmus::{Error, Limits};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Duration;

/// Counts the heap bytes that are live and their peak, so that `within` can check the budget
/// invariant: a call's heap peak stays inside its `max_alloc` (the sanitizer's malloc limit does
/// not exist on aarch64 and only catches one huge block, not many that add up).
struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(by: usize) {
    let now = LIVE.fetch_add(by, SeqCst) + by;
    PEAK.fetch_max(now, SeqCst);
}

// SAFETY: every method calls the System allocator with the arguments it got and returns its
// result unchanged, so the contract of `GlobalAlloc` is the one System keeps; the counters only
// read the sizes.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds the contract of `alloc`, which is System's.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds the contract of `alloc_zeroed`, which is System's.
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` are the pair the caller got from this allocator, which got
        // them from System.
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), SeqCst);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: as in `dealloc`, and `new_size` is the caller's, valid by the same contract.
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            if new_size > layout.size() {
                // A growing block may move: count the new one in full before the old one goes.
                grew(new_size);
                LIVE.fetch_sub(layout.size(), SeqCst);
            } else {
                LIVE.fetch_sub(layout.size() - new_size, SeqCst);
            }
        }
        p
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Heap that a call may hold beyond `max_alloc`: what the budget does not charge (an error
/// message, a thread's bookkeeping).
const UNCHARGED_SLACK: usize = 64 << 10;

/// Runs `f` and panics if the heap it held at once, counted from what was live when it started,
/// went over `max_alloc` plus `UNCHARGED_SLACK`: an allocation that scales with the input must be
/// charged to the `Budget` before it is made. The shared pool is built first (see
/// `warm_shared_pool`), so no target that measures pays for it inside a measured call.
pub fn within<T>(max_alloc: usize, f: impl FnOnce() -> T) -> T {
    warm_shared_pool();
    let base = LIVE.load(SeqCst);
    PEAK.store(base, SeqCst);
    let r = f();
    let peak = PEAK.load(SeqCst).saturating_sub(base);
    assert!(
        peak <= max_alloc.saturating_add(UNCHARGED_SLACK),
        "heap peak {peak} is over max_alloc {max_alloc}: an allocation was not charged"
    );
    r
}

/// Builds the crate's shared pool, once, before anything is measured, and gives every worker a
/// job so that what a worker allocates on its first one (queue blocks) exists too. This is fixed
/// bookkeeping, a few KiB per thread for `available_parallelism` threads (the aarch64 imaging
/// target built it inside a call with `max_alloc` 64 KiB, whose `skew_angle` splits a 55x62
/// plane, and measured 149,422 bytes), not input-sized, and is not
/// charged to a call's `max_alloc`.
/// Whichever call first needs a pool builds it, and a small input can be that call.
pub fn warm_shared_pool() {
    static WARM: OnceLock<()> = OnceLock::new();
    WARM.get_or_init(|| {
        let limits = Limits::default();
        let b = cadmus::Budget::new(&Limits {
            max_threads: limits.max_threads.max(2),
            ..limits
        });
        let mut data = vec![0u8; 8 * b.limits().max_threads];
        for _ in 0..4 {
            let _ = cadmus::par::for_each_chunk_mut(
                &b,
                &AtomicBool::new(false),
                &mut data,
                1,
                |_, _| {
                    std::thread::sleep(Duration::from_micros(200));
                    Ok(())
                },
            );
        }
    });
}

/// The model bytes. A missing model is a failure, not a skipped target.
pub fn model() -> &'static [u8] {
    static M: OnceLock<Vec<u8>> = OnceLock::new();
    M.get_or_init(|| {
        let p =
            std::env::var("CADMUS_MODEL").unwrap_or_else(|_| "tests/data/eng.traineddata".into());
        std::fs::read(&p).unwrap_or_else(|e| panic!("model {p}: {e}"))
    })
}

pub fn recognizer() -> MutexGuard<'static, Recognizer> {
    static R: OnceLock<Mutex<Recognizer>> = OnceLock::new();
    R.get_or_init(|| {
        Mutex::new(
            Recognizer::new(model(), &Limits::default(), &AtomicBool::new(false))
                .expect("the model loads"),
        )
    })
    .lock()
    .unwrap_or_else(|e| e.into_inner())
}

/// A `w x h` gray bitmap whose pixels are `pix`, repeated (zero if `pix` is empty).
pub fn gray(w: usize, h: usize, pix: &[u8], limits: &Limits) -> Result<Bitmap, Error> {
    let data: Vec<u8> = if pix.is_empty() {
        vec![0; w * h]
    } else {
        pix.iter().copied().cycle().take(w * h).collect()
    };
    Bitmap::from_gray8(w, h, &data, limits)
}

/// An allocator failure is never an acceptable answer to a call that has limits: the budget
/// must refuse first.
pub fn no_oom<T>(r: Result<T, Error>) -> Option<T> {
    match r {
        Ok(v) => Some(v),
        Err(Error::OutOfMemory) => panic!("OutOfMemory instead of a limit error"),
        Err(_) => None,
    }
}

/// The real model's recognizer's twin on the mini model, shared by every input like `recognizer`.
pub fn mini_recognizer() -> MutexGuard<'static, Recognizer> {
    static R: OnceLock<Mutex<Recognizer>> = OnceLock::new();
    R.get_or_init(|| {
        Mutex::new(
            Recognizer::new(mini_model(), &Limits::default(), &AtomicBool::new(false))
                .expect("the mini model loads"),
        )
    })
    .lock()
    .unwrap_or_else(|e| e.into_inner())
}

/// The real model with its `lstm` component swapped for `mini_lstm`, so the unicharset, recoder
/// and dawgs are the real ones and a line costs microseconds. The same output count as the real
/// network, so the decoders see 111 codes.
pub fn mini_model() -> &'static [u8] {
    static M: OnceLock<Vec<u8>> = OnceLock::new();
    M.get_or_init(|| {
        let limits = Limits::default();
        let real = model();
        let outs = Model::load(real, &limits, &AtomicBool::new(false))
            .expect("the model loads")
            .recoder
            .code_range();
        let c = parse_container(real, &limits).expect("the container parses");
        let lstm = mini_lstm(outs);
        let mut parts: Vec<(Type, &[u8])> = vec![(Type::Lstm, &lstm[..])];
        for t in [
            Type::LstmPuncDawg,
            Type::LstmSystemDawg,
            Type::LstmNumberDawg,
            Type::LstmUnicharset,
            Type::LstmRecoder,
            Type::Version,
        ] {
            if let Some(e) = c.entry(t) {
                parts.push((t, e));
            }
        }
        // The native container: an entry count, one i64 offset per type (-1 for none), the data.
        const N: usize = 24;
        let mut offsets = [-1i64; N];
        let mut data = Vec::new();
        for (t, d) in parts {
            offsets[t as usize] = (4 + 8 * N + data.len()) as i64;
            data.extend_from_slice(d);
        }
        let mut out = (N as u32).to_le_bytes().to_vec();
        for o in offsets {
            out.extend(o.to_le_bytes());
        }
        out.extend(data);
        out
    })
}

/// Input of height 8, a Reconfig that folds 2 columns and 8 rows into 16 features, one LSTM of 16
/// states and a softmax of `outs` outputs, with weights from a fixed generator.
fn mini_lstm(outs: usize) -> Vec<u8> {
    let mut s = 12345u64;
    let mut w = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 40) as f64 / (1u64 << 24) as f64 - 0.5
    };
    let input = layer(LayerType::Input, 0, 1, 0, "in", &i32s(&[1, 8, 0, 1, 0]));
    let fold = layer(LayerType::Reconfig, 1, 16, 0, "fold", &i32s(&[2, 8]));
    let mut gates = i32s(&[32]);
    for _ in 0..4 {
        gates.extend(matrix(16, 33, &mut w));
    }
    let lstm = layer(LayerType::Lstm, 16, 16, 4 * 16 * 33, "lstm", &gates);
    let head = layer(
        LayerType::Softmax,
        16,
        outs,
        outs * 17,
        "out",
        &matrix(outs, 17, &mut w),
    );
    let mut b = plumbing(
        LayerType::Series,
        1,
        outs,
        4 * 16 * 33 + outs * 17,
        "root",
        &[input, fold, lstm, head],
    );
    b.extend(trailer(outs - 1, 1));
    b
}

/// A cursor over the fuzz input; a read past the end gives 0.
pub struct Cur<'a>(pub &'a [u8]);

impl Cur<'_> {
    pub fn u8(&mut self) -> u8 {
        let Some((&b, rest)) = self.0.split_first() else {
            return 0;
        };
        self.0 = rest;
        b
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// An activation from one byte: mostly probabilities, with the values that break arithmetic
/// (NaN, infinities, negatives, out of range, huge) at the top of the range.
pub fn act(b: u8) -> f32 {
    match b {
        255 => f32::NAN,
        254 => f32::INFINITY,
        253 => f32::NEG_INFINITY,
        252 => -1.0,
        251 => 1e30,
        250 => -1e30,
        249 => 2.0,
        b => f32::from(b) / 248.0,
    }
}

/// A weight from one byte: small values, and now and then an infinity, a huge value or NaN. A
/// read past the end gives 0.
fn weight(c: &mut Cur) -> f64 {
    if c.is_empty() {
        return 0.0;
    }
    match c.u8() {
        0 => f64::INFINITY,
        1 => f64::NEG_INFINITY,
        2 => 1e300,
        3 => -1e300,
        255 => f64::NAN,
        b => (f64::from(b) - 128.0) / 32.0,
    }
}

fn i32s(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// `WeightMatrix::Serialize` in the float64 format: the mode byte, both dimensions, the unused
/// `empty_` value and the weights.
fn matrix(rows: usize, cols: usize, w: &mut dyn FnMut() -> f64) -> Vec<u8> {
    let mut b = vec![128u8];
    b.extend(i32s(&[rows as i32, cols as i32]));
    b.extend(0f64.to_le_bytes());
    for _ in 0..rows * cols {
        b.extend(w().to_le_bytes());
    }
    b
}

/// `Network::Serialize`: the type tag, the training and backprop bytes, the flags, input, output
/// and weight counts and the name, then the layer's own bytes.
fn layer(typ: LayerType, ni: usize, no: usize, nw: usize, name: &str, body: &[u8]) -> Vec<u8> {
    let mut b = vec![typ as u8, 0, 0];
    b.extend(i32s(&[
        0,
        ni as i32,
        no as i32,
        nw as i32,
        name.len() as i32,
    ]));
    b.extend(name.as_bytes());
    b.extend(body);
    b
}

/// A layer that holds others: the child count, then the children.
fn plumbing(
    typ: LayerType,
    ni: usize,
    no: usize,
    nw: usize,
    name: &str,
    kids: &[Vec<u8>],
) -> Vec<u8> {
    let mut body = (kids.len() as u32).to_le_bytes().to_vec();
    for k in kids {
        body.extend_from_slice(k);
    }
    layer(typ, ni, no, nw, name, &body)
}

/// What follows the root layer: the spec string, TF_COMPRESS_UNICHARSET, the training counters,
/// `null_char` and the three f32 fields.
fn trailer(null_char: usize, sample_iteration: u8) -> Vec<u8> {
    let mut b = 3u32.to_le_bytes().to_vec();
    b.extend(b"[x]");
    b.extend(i32s(&[
        64,
        0,
        i32::from(sample_iteration),
        null_char as i32,
    ]));
    b.extend([0u8; 12]);
    b
}

/// The header's output count, wrong one time in 32.
fn lie(c: &mut Cur, n: usize) -> usize {
    if c.u8() % 32 == 0 { n + 1 } else { n }
}

struct Built {
    bytes: Vec<u8>,
    no: usize,
    weights: usize,
}

fn dense(c: &mut Cur, typ: LayerType, ni: usize, no: usize, name: &str) -> Built {
    let m = matrix(no, ni + 1, &mut || weight(c));
    let weights = no * (ni + 1);
    Built {
        bytes: layer(typ, ni, lie(c, no), weights, name, &m),
        no,
        weights,
    }
}

/// A layer, and what is below it, that takes `ni` features: every layer type the runtime has, with
/// dimensions that are right by construction (the header's output count is sometimes wrong).
fn gen_layer(c: &mut Cur, ni: usize, depth: usize) -> Built {
    use LayerType as T;
    // Nothing that grows the feature count once it is large, and no nesting past 3.
    let kind = if ni > 64 {
        0
    } else {
        c.u8() % if depth >= 3 { 5 } else { 9 }
    };
    let name = format!("l{depth}");
    match kind {
        0 | 1 => {
            let typ = [
                T::Tanh,
                T::Relu,
                T::Linear,
                T::Logistic,
                T::PosClip,
                T::SymClip,
                T::Softmax,
                T::SoftmaxNoCtc,
            ][usize::from(c.u8()) % 8];
            let no = 1 + usize::from(c.u8() % 6);
            dense(c, typ, ni, no, &name)
        }
        2 => {
            let ns = 1 + usize::from(c.u8() % 5);
            let na = ni + ns;
            let typ = if c.u8() & 1 == 0 {
                T::Lstm
            } else {
                T::LstmSummary
            };
            let mut body = i32s(&[na as i32]);
            for _ in 0..4 {
                body.extend(matrix(ns, na + 1, &mut || weight(c)));
            }
            let weights = 4 * ns * (na + 1);
            Built {
                bytes: layer(typ, ni, lie(c, ns), weights, &name, &body),
                no: ns,
                weights,
            }
        }
        3 => {
            let (xs, ys) = (1 + usize::from(c.u8() % 3), 1 + usize::from(c.u8() % 3));
            let (typ, no) = if c.u8() & 1 == 0 {
                (T::Reconfig, ni * xs * ys)
            } else {
                (T::Maxpool, ni)
            };
            let body = i32s(&[xs as i32, ys as i32]);
            Built {
                bytes: layer(typ, ni, lie(c, no), 0, &name, &body),
                no,
                weights: 0,
            }
        }
        4 => {
            let (hx, hy) = (usize::from(c.u8() % 2), usize::from(c.u8() % 2));
            let no = ni * (2 * hx + 1) * (2 * hy + 1);
            let body = i32s(&[hx as i32, hy as i32]);
            Built {
                bytes: layer(T::Convolve, ni, lie(c, no), 0, &name, &body),
                no,
                weights: 0,
            }
        }
        5 | 6 => {
            let n = 1 + usize::from(c.u8() % 3);
            let (mut kids, mut cur, mut weights) = (Vec::new(), ni, 0);
            for _ in 0..n {
                let k = gen_layer(c, cur, depth + 1);
                (cur, weights) = (k.no, weights + k.weights);
                kids.push(k.bytes);
            }
            Built {
                bytes: plumbing(T::Series, ni, lie(c, cur), weights, &name, &kids),
                no: cur,
                weights,
            }
        }
        7 => {
            let n = 1 + usize::from(c.u8() % 3);
            let typ = if c.u8() & 1 == 0 {
                T::Parallel
            } else {
                T::Replicated
            };
            let (mut kids, mut no, mut weights) = (Vec::new(), 0, 0);
            for _ in 0..n {
                let k = gen_layer(c, ni, depth + 1);
                (no, weights) = (no + k.no, weights + k.weights);
                kids.push(k.bytes);
            }
            Built {
                bytes: plumbing(typ, ni, lie(c, no), weights, &name, &kids),
                no,
                weights,
            }
        }
        _ => {
            let typ = [T::XReversed, T::YReversed, T::XyTranspose][usize::from(c.u8()) % 3];
            let k = gen_layer(c, ni, depth + 1);
            Built {
                bytes: plumbing(typ, ni, lie(c, k.no), k.weights, &name, &[k.bytes]),
                no: k.no,
                weights: k.weights,
            }
        }
    }
}

/// A crafted `lstm` component, and the height its Input layer declares.
pub struct Crafted {
    pub bytes: Vec<u8>,
    pub height: usize,
}

/// An `lstm` component built from the input: Series of an Input, a generated layer tree of
/// every type and a softmax head, with weights that include infinities and NaN, and the
/// trailer. Dimensions are consistent except where `lie` says otherwise, so inputs get past the
/// parser and the builder and into the forward pass.
pub fn craft(c: &mut Cur) -> Crafted {
    let height = 1 + usize::from(c.u8() % 12);
    let input = layer(
        LayerType::Input,
        0,
        1,
        0,
        "in",
        &i32s(&[1, height as i32, 0, 1, 0]),
    );
    let body = gen_layer(c, 1, 0);
    let outs = 1 + usize::from(c.u8() % 8);
    let head = dense(c, LayerType::Softmax, body.no, outs, "head");
    let weights = body.weights + head.weights;
    let mut bytes = plumbing(
        LayerType::Series,
        1,
        outs,
        weights,
        "root",
        &[input, body.bytes, head.bytes],
    );
    let null_char = usize::from(c.u8()) % (outs + 1);
    let sample = c.u8();
    bytes.extend(trailer(null_char, sample));
    Crafted { bytes, height }
}
