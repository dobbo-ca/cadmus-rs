#![no_main]
use cadmus::imaging;
use cadmus::nn::{Activation, Ctx, FullyConnected, Layer, Matrix, StrideMap, Tensor};
use cadmus::{Budget, Limits, ThreadPool};
use libfuzzer_sys::fuzz_target;
use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;

mod common;

/// The call under `one` thread and under `many` must give the same result, error included, and
/// the heap peak of each run must stay within its own `max_alloc`, whatever the thread count
/// (the sanitizer's malloc limit would not catch many small blocks, and aarch64 has none).
fn same<T: PartialEq + std::fmt::Debug>(
    one: &Limits,
    many: &Limits,
    name: &str,
    call: impl Fn(&Limits) -> T,
) {
    let run = |l: &Limits| common::within(l.max_alloc, || call(l));
    assert_eq!(run(one), run(many), "{name}");
}

/// The caller's pool of the `mem / 13 % 3 == 1` runs, made once.
fn caller_pool() -> ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| ThreadPool::new(3).expect("a pool of 3 threads"))
        .clone()
}

// data = [w][h][flags][mem][scale][pixels]. A gray plane of (256 + w) x (256 + h) pixels (65,536 to
// 261,121, so over the 32,768 that a thread needs and the operators really split into bands) filled
// with the pixel bytes repeated. Every operator that runs on more than one thread (the imaging
// ones, and the FullyConnected layer, whose timesteps are split) is run with `max_threads` 1 and
// with `max_threads` 2 to 9 (bits 0-2 of `flags`; a row operator has at most one band per 32,768
// pixels, so on these planes values above 8 behave as 8, and the layer, which has more work, uses
// them), and the two results must be equal: output is the same for every thread count. `mem / 13
// % 3` picks the pool of the `max_threads` run: 0 the crate's shared pool, 1 a pool of the
// caller's, 2 workers the call builds for itself. `scale`
// picks the scale factor, which reaches every band path of `scale_gray` (2.0 and 4.0 the
// linear-interpolation kernels, 0.3 the low-factor area map, 0.01 the smoothing path), bits 5-7
// of `flags` the rotation angle. `mem` sets `max_alloc` from 64 KiB
// to 256 MiB (`64 KiB << mem % 13`), so a budget that is only just enough is covered too: whether
// a call fits must not depend on the thread count, errors included. imaging.rs fuzzes these
// operators on planes of at most 64x64, which never split.
fuzz_target!(|data: &[u8]| {
    let Some((&[w, h, flags, mem, scale], pix)) = data.split_first_chunk::<5>() else {
        return;
    };
    let limits = |max_threads| Limits {
        max_alloc: (64 << 10) << (mem % 13),
        max_pixels: 1 << 22,
        max_threads,
        ..Limits::default()
    };
    common::warm_shared_pool();
    let many = Limits {
        pool: (mem / 13 % 3 == 1).then(caller_pool),
        per_call_pool: mem / 13 % 3 == 2,
        ..limits(2 + usize::from(flags & 7))
    };
    let (one, many) = (&limits(1), &many);
    let cancel = &AtomicBool::new(false);
    let (pw, ph) = (256 + usize::from(w), 256 + usize::from(h));
    let Some(g) = common::no_oom(common::gray(pw, ph, pix, one)) else {
        return;
    };
    fully_connected(one, many, w, h, flags, pix);
    same(one, many, "otsu_threshold", |l| {
        imaging::otsu_threshold(&g, l, cancel)
    });
    same(one, many, "otsu", |l| imaging::otsu(&g, l, cancel));
    same(one, many, "sauvola", |l| {
        imaging::sauvola(&g, 25, 0.34, l, cancel)
    });
    let factor = [0.5, 1.5, 2.5, 0.9, 2.0, 4.0, 0.3, 0.01][usize::from(scale) % 8];
    same(one, many, "scale_gray", |l| {
        imaging::scale_gray(&g, factor, l, cancel)
    });
    let Some(bin) = common::no_oom(imaging::otsu(&g, one, cancel)) else {
        return;
    };
    let angle = [0.03, -0.05, 0.1, 0.0][usize::from(flags >> 5) % 4];
    same(one, many, "rotate", |l| {
        imaging::rotate(&bin, angle, l, cancel)
    });
    same(one, many, "skew_angle", |l| {
        imaging::skew_angle(&bin, l, cancel).map(f64::to_bits)
    });
    let (bw, bh) = (
        1 + usize::from(flags >> 2) % 7,
        1 + usize::from(flags >> 4) % 7,
    );
    same(one, many, "dilate", |l| {
        imaging::dilate(&bin, bw, bh, l, cancel)
    });
    same(one, many, "erode", |l| {
        imaging::erode(&bin, bw, bh, l, cancel)
    });
    same(one, many, "open", |l| {
        imaging::open(&bin, bw, bh, l, cancel)
    });
    same(one, many, "close", |l| {
        imaging::close(&bin, bw, bh, l, cancel)
    });
});

/// A FullyConnected layer of 32 to 127 inputs and outputs over 64 to 255 timesteps, weights and
/// activations from the pixel bytes, run under both limits.
fn fully_connected(one: &Limits, many: &Limits, w: u8, h: u8, flags: u8, pix: &[u8]) {
    let (ni, no, steps) = (
        32 + usize::from(w) % 96,
        32 + usize::from(h) % 96,
        64 + usize::from(flags),
    );
    let at = |i: usize| f64::from(pix.get(i % pix.len().max(1)).copied().unwrap_or(0));
    let weights: Vec<f64> = (0..no * (ni + 1))
        .map(|i| (at(i) - 128.0) / 128.0)
        .collect();
    let act = [
        Activation::Tanh,
        Activation::Logistic,
        Activation::Relu,
        Activation::Softmax,
        Activation::Linear,
        Activation::PosClip,
        Activation::SymClip,
    ][usize::from(flags >> 3) % 7];
    let Some(m) = common::no_oom(Matrix::new(no, ni, weights, one)) else {
        return;
    };
    let fc = Layer::FullyConnected(FullyConnected::new("fc", act, m));
    let map = StrideMap {
        height: 1,
        width: steps,
    };
    let Some(mut input) = common::no_oom(Tensor::new(map, ni, &Budget::new(one))) else {
        return;
    };
    for t in 0..steps {
        for (j, v) in input.row_mut(t).iter_mut().enumerate() {
            *v = (at(t * 31 + j) / 64.0 - 2.0) as f32;
        }
    }
    // Only the forward pass is measured: the bits are collected into vecs the budget does not own.
    let run = |l: &Limits| {
        let budget = Budget::new(l);
        common::within(l.max_alloc, || fc.forward(&input, &mut Ctx::new(&budget))).map(|t| {
            (0..steps)
                .flat_map(|s| t.row(s).iter().map(|v| v.to_bits()).collect::<Vec<_>>())
                .collect::<Vec<u32>>()
        })
    };
    assert_eq!(run(one), run(many), "fully_connected");
}
