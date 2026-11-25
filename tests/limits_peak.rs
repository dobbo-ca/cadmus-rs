//! `Limits::max_alloc` is the peak working memory of one call and `Limits::max_threads` the most
//! threads it uses. This test measures both on the real thing: a counting global allocator gives
//! the heap peak of each main call, which must stay within `max_alloc` for several values down to
//! the smallest one the call succeeds with, and a call with less must fail with
//! `Error::AllocLimit`. A thread counter checks `max_threads` the same way. The image and the
//! recognizer calls run with `max_threads` 1, 2 and 4 on an image big enough to split into bands,
//! on the crate's shared pool and, for 2 and 4, on a pool the caller made. Both pools are built
//! before anything is measured: a pool's own memory (stacks, queues) is fixed, not input-sized, and
//! is not part of `max_alloc`; what a call holds for its workers is charged and measured here.
//!
//! The strings the model controls are measured too: `Layer::tree` on a long control-byte name
//! (the quoted text is four times the name), a parse, a build (`Recognizer::new`) and a forward
//! pass that fail with a long name in the file (their errors must not copy the name), and a
//! unicharset line with many fields, each with the heap peak checked against `max_alloc`. The
//! calls that run on the threads (`skew_angle`, `otsu`, `otsu_threshold`), the flood fills, the
//! distance function and the `Grid` are measured like the others; `raster_op` must allocate
//! nothing.
//!
//! The allocator is global, so everything runs in one test function, one call at a time. The
//! model calls skip when `tests/data/eng.traineddata` is absent (CI has it).
#![allow(unsafe_code)]

use cadmus::imaging::{self, Bitmap, Op, Point, Rect};
use cadmus::nn::{StrideMap, Tensor};
use cadmus::recog::Recognizer;
use cadmus::tessdata::{
    Layer, LayerType, Model, Type, parse_container, parse_recognizer, parse_unicharset,
};
use cadmus::{Budget, Error, Limits, ThreadPool, par};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
use std::time::Duration;

/// Counts the bytes that are live and their peak.
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
                // A growing block may move, and then the old block and the new one are live
                // together, so count the new one in full before the old one is given back, as if
                // every growth moved.
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

/// Runs `f` and returns its result and the most heap bytes it held at once, counted from what was
/// live when it started.
fn measure<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = LIVE.load(SeqCst);
    PEAK.store(base, SeqCst);
    let r = f();
    (r, PEAK.load(SeqCst).saturating_sub(base))
}

/// The `max_threads` the calls run with now; 0 is the default.
static THREADS: AtomicUsize = AtomicUsize::new(0);

/// Whether the calls run on `caller_pool()` (else on the crate's shared pool).
static CALLER: AtomicBool = AtomicBool::new(false);

/// A pool of the caller's, made before any measure.
fn caller_pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| ThreadPool::new(3).expect("a pool of 3 threads"))
}

fn limits(max_alloc: usize) -> Limits {
    let mut l = Limits {
        max_alloc,
        ..Limits::default()
    };
    match THREADS.load(SeqCst) {
        0 => {}
        n => l.max_threads = n,
    }
    if CALLER.load(SeqCst) {
        l.pool = Some(caller_pool().clone());
    }
    l
}

/// Runs a parallel stage on the shared pool and on the caller's, so that both exist, with their
/// first queue blocks allocated, before a call is measured.
fn warm_pools() {
    let cancel = AtomicBool::new(false);
    for pool in [None, Some(caller_pool().clone())] {
        let b = Budget::new(&Limits {
            max_threads: 4,
            pool,
            ..Limits::default()
        });
        let mut data = vec![0u8; 64];
        for _ in 0..4 {
            par::for_each_chunk_mut(&b, &cancel, &mut data, 1, |_, c| {
                std::thread::sleep(Duration::from_micros(200));
                c[0] = 1;
                Ok(())
            })
            .expect("a warm-up stage");
        }
    }
}

type Run<'a> = &'a dyn Fn(&Limits) -> Result<(), Error>;

/// The smallest `max_alloc` the call succeeds with, found by bisection below `upper`. Any error
/// other than `AllocLimit` is a failure of the call.
fn smallest(name: &str, run: Run, upper: usize, failures: &mut Vec<String>) -> usize {
    let (mut lo, mut hi) = (0usize, upper);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        match run(&limits(mid)) {
            Ok(()) => hi = mid,
            Err(Error::AllocLimit) => lo = mid,
            Err(e) => {
                failures.push(format!("{name}: max_alloc {mid} gave `{e}`"));
                break;
            }
        }
    }
    hi
}

/// One call: its heap peak must be within `max_alloc` for the default and for the smallest
/// `max_alloc` it works with and a multiple of it, and less than that must be `AllocLimit`.
fn check(name: &str, run: Run, failures: &mut Vec<String>, report: &mut Vec<String>) {
    let pool = if CALLER.load(SeqCst) {
        "caller"
    } else {
        "shared"
    };
    let name = &format!(
        "{name} (max_threads {}, {pool} pool)",
        limits(0).max_threads
    );
    let default = limits(Limits::default().max_alloc);
    let (r, at_default) = measure(|| run(&default));
    if let Err(e) = r {
        failures.push(format!("{name}: failed with the default limits: {e}"));
        return;
    }
    let mut min = smallest(name, run, 2 * at_default + (1 << 20), failures);
    // The recognizer carries its random generator from one call to the next (the convolution
    // fills off-image taps from it), so the beam, and with it the ledger, differs a little from
    // run to run. Where the smallest value found fails now, take the next one up.
    for _ in 0..16 {
        if run(&limits(min)) != Err(Error::AllocLimit) {
            break;
        }
        min += min / 50 + 1;
    }
    let mut line =
        format!("{name}: heap peak {at_default} with the defaults, smallest max_alloc {min}");
    // With threads the budget has room for them only a little above the smallest value: check
    // there too, where a per-thread allocation that is not charged would show.
    let t = limits(0).max_threads;
    let mut points = vec![min, 2 * min, Limits::default().max_alloc];
    if t > 1 {
        points.extend([min + 4096, min + (t - 1) * 4096 + 2048, min + t * 8192]);
    }
    for max_alloc in points {
        let l = limits(max_alloc);
        let (r, peak) = measure(|| run(&l));
        line.push_str(&format!("; {peak} of {max_alloc}"));
        if let Err(e) = r {
            failures.push(format!("{name}: max_alloc {max_alloc} gave `{e}`"));
        } else if peak > max_alloc {
            failures.push(format!(
                "{name}: the heap peak {peak} is over max_alloc {max_alloc}"
            ));
        }
    }
    for max_alloc in [0, 1, min / 2] {
        let r = run(&limits(max_alloc));
        if r != Err(Error::AllocLimit) {
            failures.push(format!(
                "{name}: max_alloc {max_alloc} gave {r:?}, want AllocLimit"
            ));
        }
    }
    report.push(line);
}

/// A gray image of noise over a gradient, with dark bars, so every operator has work to do.
fn gray(w: usize, h: usize) -> Bitmap {
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    let mut pix = Vec::with_capacity(w * h);
    for y in 0..h {
        for col in 0..w {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let bar = if (col / 9 + y / 7) % 5 == 0 { 90 } else { 200 };
            pix.push((bar + (x % 40) as usize % 55 + y / 8).min(255) as u8);
        }
    }
    Bitmap::from_gray8(w, h, &pix, &Limits::default()).expect("a small image")
}

fn imaging_calls(failures: &mut Vec<String>, report: &mut Vec<String>) {
    let cancel = AtomicBool::new(false);
    // 524,288 pixels: over the 32,768 a band needs, so every operator splits into bands.
    let g = gray(1024, 512);
    let bin = imaging::otsu(&g, &Limits::default(), &cancel).expect("otsu");
    for f in [0.5, 1.2, 2.5, 0.1] {
        check(
            &format!("scale_gray {f}"),
            &|l| imaging::scale_gray(&g, f, l, &cancel).map(drop),
            failures,
            report,
        );
    }
    check(
        "sauvola",
        &|l| imaging::sauvola(&g, 25, 0.34, l, &cancel).map(drop),
        failures,
        report,
    );
    check(
        "dilate",
        &|l| imaging::dilate(&bin, 5, 3, l, &cancel).map(drop),
        failures,
        report,
    );
    check(
        "erode",
        &|l| imaging::erode(&bin, 5, 3, l, &cancel).map(drop),
        failures,
        report,
    );
    check(
        "open",
        &|l| imaging::open(&bin, 5, 3, l, &cancel).map(drop),
        failures,
        report,
    );
    check(
        "close",
        &|l| imaging::close(&bin, 5, 3, l, &cancel).map(drop),
        failures,
        report,
    );
    check(
        "rotate",
        &|l| imaging::rotate(&bin, 0.03, l, &cancel).map(drop),
        failures,
        report,
    );
    check(
        "skew_angle",
        &|l| imaging::skew_angle(&bin, l, &cancel).map(drop),
        failures,
        report,
    );
    check(
        "otsu_threshold",
        &|l| imaging::otsu_threshold(&g, l, &cancel).map(drop),
        failures,
        report,
    );
    check(
        "otsu",
        &|l| imaging::otsu(&g, l, &cancel).map(drop),
        failures,
        report,
    );
    check(
        "conn_comp",
        &|l| imaging::conn_comp(&bin, 8, l, &cancel).map(drop),
        failures,
        report,
    );
    let eroded = imaging::erode(&bin, 3, 3, &Limits::default(), &cancel).expect("erode");
    check(
        "seed_fill",
        &|l| imaging::seed_fill(&eroded, &bin, 8, l, &cancel).map(drop),
        failures,
        report,
    );
    check(
        "distance_function",
        &|l| imaging::distance_function(&bin, l, &cancel).map(drop),
        failures,
        report,
    );
    // `raster_op` allocates nothing at all.
    let mut dst = bin.clone();
    let l = limits(1 << 20);
    let src = Point { x: 3, y: 5 };
    let (r, peak) = measure(|| {
        imaging::raster_op(
            &mut dst,
            Rect::new(-10, 4, 900, 400),
            Op::SrcXorDst,
            Some(&bin),
            src,
            &l,
            &cancel,
        )
    });
    if r.is_err() || peak != 0 {
        failures.push(format!("raster_op: {r:?}, heap peak {peak}, want 0"));
    }
}

/// `Grid::insert` and `query`: the vecs double as they grow, and the old block is live while the
/// new one is made. The first grid has many cells and boxes of every size (empty ones too); the
/// second one cell and many empty boxes, which cover no cell but are still filed.
fn grid_calls(failures: &mut Vec<String>, report: &mut Vec<String>) {
    check(
        "Grid insert and query",
        &|l| {
            let mut g = imaging::Grid::new(Rect::new(0, 0, 512, 512), 16, l)?;
            let mut x = 0x2545_f491_4f6c_dd1du64;
            for id in 0..3000i64 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let (x0, y0) = ((x % 500) as isize, ((x >> 16) % 500) as isize);
                let (w, h) = (((x >> 32) % 40) as isize, ((x >> 40) % 40) as isize);
                g.insert(id, Rect::new(x0, y0, x0 + w, y0 + h))?;
            }
            g.query(Rect::new(100, 100, 300, 300)).map(drop)
        },
        failures,
        report,
    );
    check(
        "Grid with many empty boxes",
        &|l| {
            let mut g = imaging::Grid::new(Rect::new(0, 0, 1, 1), 1, l)?;
            for id in 0..70_000i64 {
                g.insert(id, Rect::new(0, 0, 0, 0))?;
            }
            g.query(Rect::new(0, 0, 1, 1)).map(drop)
        },
        failures,
        report,
    );
}

fn model_calls(first: bool, failures: &mut Vec<String>, report: &mut Vec<String>) {
    let dir = env!("CARGO_MANIFEST_DIR");
    let Ok(data) = std::fs::read(format!("{dir}/tests/data/eng.traineddata")) else {
        eprintln!("eng.traineddata not present; skipping the model calls");
        return;
    };
    let cancel = AtomicBool::new(false);
    // Loading uses no threads, so it is measured once.
    if first {
        check(
            "Model::load",
            &|l| Model::load(&data, l, &cancel).map(drop),
            failures,
            report,
        );
        check(
            "Recognizer::new",
            &|l| Recognizer::new(&data, l, &cancel).map(drop),
            failures,
            report,
        );
    }

    let Ok(entries) = std::fs::read_dir(format!("{dir}/tests/data/lines/h36")) else {
        eprintln!("tests/data/lines/h36 not present; skipping recognize");
        return;
    };
    let mut names: Vec<_> = entries
        .filter_map(|e| e.ok()?.path().to_str().map(str::to_string))
        .filter(|p| p.ends_with(".pgm"))
        .collect();
    names.sort();
    let Some(path) = names.first() else {
        return;
    };
    let img = Bitmap::from_pgm(&std::fs::read(path).expect("a line"), &Limits::default())
        .expect("a PGM line");
    let r = Recognizer::new(&data, &Limits::default(), &cancel).expect("the model loads");
    check(
        "recognize",
        &|l| r.recognize(&img, l, &cancel).map(drop),
        failures,
        report,
    );
    check(
        "recognize_text",
        &|l| r.recognize_text(&img, l, &cancel).map(drop),
        failures,
        report,
    );
    // The beam on an output built to make it churn: 200 timesteps, eight codes at about 0.12 each
    // in every row, so each step keeps the beams full and replaces nodes. A natural line does not
    // do that, and the beam's nodes, heaps and positions must stay charged for it too.
    let Ok((out, _)) = r.forward(&img, &Limits::default(), &cancel) else {
        failures.push("forward on the line failed".to_string());
        return;
    };
    let mut churn = Tensor::new(
        StrideMap {
            height: 1,
            width: 200,
        },
        out.features,
        &Budget::new(&Limits::default()),
    )
    .expect("a small tensor");
    let mut x = 0x2545_f491_4f6c_dd1du64;
    let mut row = vec![0.0005f64; out.features];
    for t in 0..200 {
        row.fill(0.0005);
        for _ in 0..8 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            row[(x % out.features as u64) as usize] = 0.12;
        }
        churn.write_time_step(t, &row);
    }
    check(
        "beam_decode on a churning output",
        &|l| r.beam_decode(&churn, l, &cancel).map(drop),
        failures,
        report,
    );
    check(
        "greedy_decode",
        &|l| r.greedy_decode(&out, l, &cancel).map(drop),
        failures,
        report,
    );
    if first {
        long_name_forward(&data, &img, failures, report);
    }
}

fn i32s(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// `Network::Serialize`: the type tag, the training and backprop bytes, the flags, input, output
/// and weight counts and the name, then the layer's own bytes.
fn layer_bytes(typ: LayerType, (ni, no): (usize, usize), name: &str, body: &[u8]) -> Vec<u8> {
    let mut b = vec![typ as u8, 0, 0];
    b.extend(i32s(&[0, ni as i32, no as i32, 0, name.len() as i32]));
    b.extend(name.as_bytes());
    b.extend(body);
    b
}

/// The `lstm` component: the root layer, then the trailer (the spec string, TF_COMPRESS_UNICHARSET,
/// the training counters, `null_char` and the three f32 fields).
fn lstm_component(root: &[u8], null_char: usize) -> Vec<u8> {
    let mut b = root.to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend(b"[x]");
    b.extend(i32s(&[64, 0, 1, null_char as i32]));
    b.extend([0u8; 12]);
    b
}

fn series(no: usize, kids: &[Vec<u8>]) -> Vec<u8> {
    let mut body = (kids.len() as u32).to_le_bytes().to_vec();
    kids.iter().for_each(|k| body.extend_from_slice(k));
    layer_bytes(LayerType::Series, (1, no), "root", &body)
}

/// The native container: an entry count, one i64 offset per type (-1 for none), the data.
fn container(parts: Vec<(Type, &[u8])>) -> Vec<u8> {
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
}

/// A forward pass that fails in a layer with a long name: the real model with its `lstm`
/// component swapped for an Input and a Reconfig that wants more features than the Input makes.
/// The error must not copy the name, and the heap peak must stay within `max_alloc`.
fn long_name_forward(
    real: &[u8],
    img: &Bitmap,
    failures: &mut Vec<String>,
    report: &mut Vec<String>,
) {
    let cancel = AtomicBool::new(false);
    let all = Limits::default();
    let outs = Model::load(real, &all, &cancel)
        .expect("the model loads")
        .recoder
        .code_range();
    let n = 2usize << 20;
    let input = layer_bytes(LayerType::Input, (0, 1), "in", &i32s(&[1, 36, 0, 1, 0]));
    let wide = layer_bytes(
        LayerType::Reconfig,
        (outs, outs),
        &"\u{1}".repeat(n),
        &i32s(&[1, 1]),
    );
    let lstm = lstm_component(&series(outs, &[input, wide]), outs - 1);
    let c = parse_container(real, &all).expect("the container parses");
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
    let model = container(parts);
    let r = match Recognizer::new(&model, &all, &cancel) {
        Ok(r) => r,
        Err(e) => {
            failures.push(format!(
                "forward with a long name: the model failed to load: {e}"
            ));
            return;
        }
    };
    let max_alloc = 8usize << 20;
    let l = limits(max_alloc);
    let (res, peak) = measure(|| r.forward(img, &l, &cancel).map(drop));
    if !matches!(&res, Err(Error::Recog(m)) if m.len() < 1000) {
        failures.push(format!("forward with a long name: got {res:?}"));
    }
    if peak > max_alloc {
        failures.push(format!(
            "forward with a long name: the heap peak {peak} is over max_alloc {max_alloc}"
        ));
    }
    report.push(format!("forward with a long name: {peak} of {max_alloc}"));
}

/// A layer with a long name of control bytes: `tree` quotes each as `\x01`, four bytes, and the
/// text must be the only buffer; and a parse that fails with such a name in the file must not
/// build an error that copies it.
fn string_calls(failures: &mut Vec<String>, report: &mut Vec<String>) {
    let n = 1usize << 20;
    let layer = Layer {
        typ: LayerType::Tanh,
        name: "\u{1}".repeat(n),
        num_inputs: 1,
        num_outputs: 1,
        num_weights: 0,
        flags: 0,
        matrices: Vec::new(),
        children: Vec::new(),
        half_x: 0,
        half_y: 0,
        x_scale: 0,
        y_scale: 0,
        shape: None,
        na: 0,
    };
    check(
        "Layer::tree",
        &|l| layer.tree(l).map(drop),
        failures,
        report,
    );

    let cancel = AtomicBool::new(false);
    let n = 4usize << 20;
    // An unknown type name, 0x10 bytes (six in `{:?}`), and then a Tanh layer with that name
    // whose weights do not fit.
    let mut unknown = vec![0u8];
    unknown.extend_from_slice(&(n as u32).to_le_bytes());
    unknown.resize(5 + n, 0x10);
    let mut tanh = vec![LayerType::Tanh as u8, 2, 0];
    for v in [0i32, 2, 2, 6, n as i32] {
        tanh.extend_from_slice(&v.to_le_bytes());
    }
    tanh.resize(tanh.len() + n, 0x10);
    let root = std::mem::size_of::<Layer>();
    for (what, data, max_alloc, want) in [
        ("unknown type", &unknown, 2 * n, None),
        (
            "name, then weights",
            &tanh,
            n + root + 1,
            Some(Error::AllocLimit),
        ),
    ] {
        let l = limits(max_alloc);
        let (r, peak) = measure(|| parse_recognizer(data, false, &Budget::new(&l), &cancel));
        let ok = match (&r, &want) {
            (Err(Error::Tessdata(m)), None) => m.len() < 1000,
            (Err(e), Some(w)) => e == w,
            _ => false,
        };
        if !ok {
            failures.push(format!(
                "parse with a long name, {what}: got {:?}",
                r.map(drop)
            ));
        }
        if peak > max_alloc {
            failures.push(format!(
                "parse with a long name, {what}: the heap peak {peak} is over max_alloc {max_alloc}"
            ));
        }
        report.push(format!(
            "parse with a long name, {what}: {peak} of {max_alloc}"
        ));
    }

    // A model that parses and then fails to build: a Reconfig with a negative scale and a name of
    // control bytes (six bytes each in `{:?}`). Neither the build error nor the ones above it may
    // copy the name.
    let n = 1usize << 20;
    let root = layer_bytes(
        LayerType::Reconfig,
        (1, 1),
        &"\u{1}".repeat(n),
        &i32s(&[-1, 1]),
    );
    let model = container(vec![(Type::Lstm, &lstm_component(&root, 0)[..])]);
    let max_alloc = 5 * n;
    let l = limits(max_alloc);
    let (r, peak) = measure(|| Recognizer::new(&model, &l, &cancel).map(drop));
    if !matches!(&r, Err(Error::Recog(m)) if m.len() < 1000) {
        failures.push(format!("build with a long name: got {r:?}"));
    }
    if peak > max_alloc {
        failures.push(format!(
            "build with a long name: the heap peak {peak} is over max_alloc {max_alloc}"
        ));
    }
    report.push(format!("build with a long name: {peak} of {max_alloc}"));

    // One unicharset line of 16,385 fields: the list of fields is made at its final size, not
    // grown by doubling (the old and the new block would be live together, over the charge).
    let mut text = String::from("1\n");
    text.push_str(&"a ".repeat(16_384));
    text.push('a');
    let need = {
        let b = Budget::new(&limits(Limits::default().max_alloc));
        if let Err(e) = parse_unicharset(text.as_bytes(), &b) {
            failures.push(format!("unicharset with many fields: {e}"));
        }
        b.peak()
    };
    let l = limits(need);
    let (r, peak) = measure(|| parse_unicharset(text.as_bytes(), &Budget::new(&l)).map(drop));
    if r.is_err() || peak > need {
        failures.push(format!(
            "unicharset with many fields: {r:?}, heap peak {peak} over the ledger's {need}"
        ));
    }
    report.push(format!("unicharset with many fields: {peak} of {need}"));
}

/// The ledger's own peak, with charges and releases racing on several threads.
fn ledger_peak(failures: &mut Vec<String>) {
    for max_alloc in [1000usize, 4096, 1 << 20] {
        let b = Budget::new(&limits(max_alloc));
        let b = &b;
        std::thread::scope(|s| {
            for t in 0..4usize {
                s.spawn(move || {
                    for i in 0..2000usize {
                        let bytes = 1 + (i * 37 + t * 101) % (max_alloc / 3 + 1);
                        if let Ok(hold) = b.hold(bytes) {
                            std::hint::black_box(&hold);
                        }
                    }
                });
            }
        });
        if b.peak() > max_alloc || b.used() != 0 {
            failures.push(format!(
                "ledger: peak {} over max_alloc {max_alloc}, or {} bytes left charged",
                b.peak(),
                b.used()
            ));
        }
    }
}

/// The threads a call really runs at once, for several `max_threads` including 1, on the shared
/// pool and on a caller's of 8.
fn thread_peak(failures: &mut Vec<String>) {
    let cancel = AtomicBool::new(false);
    let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
    let big = ThreadPool::new(8).expect("a pool of 8 threads");
    for (pool_name, pool, workers) in [("shared", None, cpus), ("caller", Some(big), 8)] {
        for max_threads in [1usize, 2, 3, 8] {
            let b = Budget::new(&Limits {
                max_threads,
                pool: pool.clone(),
                ..Limits::default()
            });
            let (running, most) = (AtomicUsize::new(0), AtomicUsize::new(0));
            let mut data = vec![0u8; 64];
            let r = par::for_each_chunk_mut(&b, &cancel, &mut data, 1, |_, chunk| {
                most.fetch_max(running.fetch_add(1, SeqCst) + 1, SeqCst);
                std::thread::sleep(Duration::from_millis(2));
                chunk[0] = 1;
                running.fetch_sub(1, SeqCst);
                Ok(())
            });
            let (seen, ledger) = (most.load(SeqCst), b.peak_threads());
            // The caller and the pool's workers, up to `max_threads`.
            let want = max_threads.min(1 + workers);
            if r.is_err() || data.iter().any(|&v| v != 1) {
                failures.push(format!(
                    "par with {max_threads} threads, {pool_name} pool: {r:?}"
                ));
            }
            if seen > max_threads || ledger != want || (max_threads == 1 && seen != 1) {
                failures.push(format!(
                    "par, {pool_name} pool: {seen} threads ran at once and the ledger counted {ledger}, with max_threads {max_threads} (want {want})"
                ));
            }
        }
    }
}

#[test]
fn the_peaks_stay_within_the_limits() {
    let (mut failures, mut report) = (Vec::new(), Vec::new());
    warm_pools();
    // `max_threads` 1 never touches a pool, so it is run once; 2 and 4 on both pools.
    for (threads, caller) in [
        (1usize, false),
        (2, false),
        (4, false),
        (2, true),
        (4, true),
    ] {
        THREADS.store(threads, SeqCst);
        CALLER.store(caller, SeqCst);
        imaging_calls(&mut failures, &mut report);
        model_calls(threads == 1, &mut failures, &mut report);
        if threads == 1 {
            grid_calls(&mut failures, &mut report);
            string_calls(&mut failures, &mut report);
        }
    }
    THREADS.store(0, SeqCst);
    CALLER.store(false, SeqCst);
    ledger_peak(&mut failures);
    thread_peak(&mut failures);
    for line in &report {
        eprintln!("{line}");
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
