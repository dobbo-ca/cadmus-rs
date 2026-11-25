//! The Rust side of the parity job: `oracle [-model eng.traineddata] [-threads N] [-pool P] <inputs> <out>` runs this
//! crate over every file of `<inputs>` and writes `<name>.out` (a text report) or `<name>.err`
//! (an error class), exactly as `tests/parity/go/oracle` does with the Go library. The two output trees
//! must be equal. See docs/parity.md for the input kinds, the report formats and the error
//! classes. `-threads N` sets `Limits::max_threads` (1 runs everything on one thread; the default
//! is the number of CPUs), and the output must be the same for every N. `-pool P` picks the pool
//! the threads come from: `shared` (the default: no caller pool, so the crate's shared pool),
//! `caller` (one `ThreadPool` made up front and reused for every call, as a server would) or
//! `percall` (workers the call builds for itself); the output must be the same for all three.
#![forbid(unsafe_code)]

use cadmus::imaging::{self, Bitmap, Component, Grid, Op, Point, Rect};
use cadmus::nn;
use cadmus::recog::Recognizer;
use cadmus::tessdata::{self, Dawg, Model, Recoder, Type};
use cadmus::{Budget, Error, Limits, ThreadPool};
use std::fmt::Write as _;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// `-threads N`; 0 keeps the default of `Limits`.
static THREADS: AtomicUsize = AtomicUsize::new(0);

/// `-pool caller`: the pool every call runs on, made once.
static CALLER_POOL: OnceLock<ThreadPool> = OnceLock::new();

/// `-pool percall`: every call builds its own workers.
static PER_CALL: AtomicBool = AtomicBool::new(false);

/// The limits the Go oracle mirrors: planes up to 1<<24 pixels, models up to 64 MiB.
fn limits() -> Limits {
    let mut l = Limits {
        max_pixels: 1 << 24,
        ..Limits::default()
    };
    match THREADS.load(Ordering::Relaxed) {
        0 => {}
        n => l.max_threads = n,
    }
    l.pool = CALLER_POOL.get().cloned();
    l.per_call_pool = PER_CALL.load(Ordering::Relaxed);
    l
}

fn class(e: &Error) -> &'static str {
    if std::env::var_os("ORACLE_MSG").is_some() {
        eprintln!("[rust] {e}");
    }
    match e {
        Error::Tessdata(_) => "tessdata",
        Error::Network(_) => "nn",
        Error::Imaging(_) => "imaging",
        Error::Recog(_) => "recog",
        Error::ModelLimit => "model_limit",
        Error::PixelLimit => "pixel_limit",
        Error::AllocLimit => "alloc_limit",
        Error::OutOfMemory => "out_of_memory",
        Error::Cancelled => "cancelled",
    }
}

/// FNV-1a 64.
struct Fnv(u64);

impl Fnv {
    fn new() -> Fnv {
        Fnv(14695981039346656037)
    }
    fn u8(&mut self, b: u8) {
        self.0 ^= u64::from(b);
        self.0 = self.0.wrapping_mul(1099511628211);
    }
    fn u32(&mut self, v: u32) {
        for b in v.to_le_bytes() {
            self.u8(b);
        }
    }
    fn u64(&mut self, v: u64) {
        self.u32(v as u32);
        self.u32((v >> 32) as u32);
    }
    fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        for b in s.bytes() {
            self.u8(b);
        }
    }
    fn flag(&mut self, v: bool) {
        self.u8(u8::from(v));
    }
}

/// NaN payloads are not compared.
fn f64b(x: f64) -> u64 {
    if x.is_nan() {
        0x7ff8000000000000
    } else {
        x.to_bits()
    }
}

fn f32b(x: f32) -> u32 {
    if x.is_nan() { 0x7fc00000 } else { x.to_bits() }
}

fn hex(s: &str) -> String {
    s.bytes().fold(String::new(), |mut o, b| {
        let _ = write!(o, "{b:02x}");
        o
    })
}

fn hash_bitmap(b: &Bitmap) -> String {
    let mut f = Fnv::new();
    for y in 0..b.height {
        for x in 0..b.width {
            f.u8(b.at(x, y));
        }
    }
    format!("{}x{}x{} {:016x}", b.width, b.height, b.depth, f.0)
}

/// Writes `name: hash` or `name: ERR class` and returns the bitmap.
fn rec(out: &mut String, name: &str, r: Result<Bitmap, Error>) -> Option<Bitmap> {
    match r {
        Ok(x) => {
            let _ = writeln!(out, "{name}: {}", hash_bitmap(&x));
            Some(x)
        }
        Err(e) => {
            let _ = writeln!(out, "{name}: ERR {}", class(&e));
            None
        }
    }
}

fn val(out: &mut String, name: &str, r: Result<String, Error>) {
    match r {
        Ok(s) => {
            let _ = writeln!(out, "{name}: {s}");
        }
        Err(e) => {
            let _ = writeln!(out, "{name}: ERR {}", class(&e));
        }
    }
}

fn comps_hash(cs: &[Component]) -> String {
    let mut f = Fnv::new();
    for c in cs {
        f.u32(c.bounds.min.x as u32);
        f.u32(c.bounds.min.y as u32);
        f.u32(c.bounds.max.x as u32);
        f.u32(c.bounds.max.y as u32);
        f.u32(c.pixels as u32);
    }
    format!("{} {:016x}", cs.len(), f.0)
}

fn rect_s(r: &Rect) -> String {
    format!("({},{})-({},{})", r.min.x, r.min.y, r.max.x, r.max.y)
}

fn rect(x0: isize, y0: isize, x1: isize, y1: isize) -> Rect {
    Rect::new(x0, y0, x1, y1)
}

fn grid_report(g: &Bitmap, comps: &[Component], lim: &Limits) -> Result<String, Error> {
    let (w, h) = (g.width as isize, g.height as isize);
    let mut gr = Grid::new(g.bounds(), 8, lim)?;
    for (i, c) in comps.iter().take(500).enumerate() {
        gr.insert(i as i64, c.bounds)?;
    }
    let mut f = Fnv::new();
    let mut n = 0;
    for q in [
        rect(0, 0, w / 2, h / 2),
        rect(w / 3, h / 3, w, h),
        rect(-5, -5, 3, 3),
        rect(w - 4, h - 4, w + 9, h + 9),
        rect(0, 0, 0, 0),
    ] {
        let ids = gr.query(q)?;
        n += ids.len();
        f.u32(ids.len() as u32);
        for id in ids {
            f.u32(id as u32);
        }
    }
    Ok(format!("{n} {:016x}", f.0))
}

fn imaging_report(out: &mut String, g: &Bitmap, lim: &Limits, cancel: &AtomicBool) {
    let (w, h) = (g.width, g.height);
    val(
        out,
        "otsu_threshold",
        imaging::otsu_threshold(g, lim, cancel).map(|t| t.to_string()),
    );
    let bin = rec(out, "otsu", imaging::otsu(g, lim, cancel));
    rec(
        out,
        "sauvola_15_034",
        imaging::sauvola(g, 15, 0.34, lim, cancel),
    );
    rec(
        out,
        "sauvola_5_02",
        imaging::sauvola(g, 5, 0.2, lim, cancel),
    );
    rec(out, "sauvola_3", imaging::sauvola(g, 3, 0.2, lim, cancel));
    rec(
        out,
        "sauvola_5_neg1e20",
        imaging::sauvola(g, 5, -1e20, lim, cancel),
    );
    rec(
        out,
        "sauvola_5_nan",
        imaging::sauvola(g, 5, f64::NAN, lim, cancel),
    );
    let Some(bin) = bin else { return };
    let mut ang = 0.0f64;
    match imaging::skew_angle(&bin, lim, cancel) {
        Ok(a) => {
            ang = a;
            let _ = writeln!(out, "skew_angle: {:016x}", f64b(a));
        }
        Err(e) => {
            let _ = writeln!(out, "skew_angle: ERR {}", class(&e));
        }
    }
    rec(out, "rotate_skew", imaging::rotate(&bin, ang, lim, cancel));
    rec(out, "rotate_005", imaging::rotate(&bin, 0.05, lim, cancel));
    rec(out, "rotate_gray", imaging::rotate(g, 0.05, lim, cancel));
    for (name, a) in [("nan", f64::NAN), ("inf", f64::INFINITY), ("1e39", 1e39)] {
        rec(
            out,
            &format!("rotate_{name}"),
            imaging::rotate(&bin, a, lim, cancel),
        );
    }
    // An angle in every octant of sin and cos, and 1e6 and 1e9 for the argument reduction.
    for (name, a) in [
        ("0.9", 0.9f64),
        ("2", 2.0),
        ("-2.5", -2.5),
        ("3.5", 3.5),
        ("5", 5.0),
        ("pi2", std::f64::consts::FRAC_PI_2),
        ("1e6", 1e6),
        ("1e9", 1e9),
    ] {
        rec(
            out,
            &format!("rotate_a{name}"),
            imaging::rotate(&bin, a, lim, cancel),
        );
    }
    for (name, f) in [
        ("0.5", 0.5f64),
        ("1", 1.0),
        ("1.7", 1.7),
        ("2.5", 2.5),
        ("0", 0.0),
        ("-1", -1.0),
        ("0.01", 0.01),
        ("1e-20", 1e-20),
        ("1e-50", 1e-50),
    ] {
        if f > 1.0 && (w * h) as f64 * f * f > (1 << 22) as f64 {
            continue;
        }
        rec(
            out,
            &format!("scale_{name}"),
            imaging::scale_gray(g, f, lim, cancel),
        );
    }
    // Go returns a bitmap of negative size for these and the oracle reports it as `imaging`.
    for (name, f) in [("NaN", f64::NAN), ("Inf", f64::INFINITY), ("1e39", 1e39)] {
        rec(
            out,
            &format!("scale_{name}"),
            imaging::scale_gray(g, f, lim, cancel),
        );
    }
    rec(
        out,
        "scale_bin",
        imaging::scale_gray(&bin, 0.5, lim, cancel),
    );
    rec(out, "dilate_3x3", imaging::dilate(&bin, 3, 3, lim, cancel));
    let eroded = rec(out, "erode_3x3", imaging::erode(&bin, 3, 3, lim, cancel));
    rec(out, "open_5x1", imaging::open(&bin, 5, 1, lim, cancel));
    rec(out, "close_5x1", imaging::close(&bin, 5, 1, lim, cancel));
    rec(out, "dilate_gray", imaging::dilate(g, 3, 3, lim, cancel));
    rec(out, "dilate_0x1", imaging::dilate(&bin, 0, 1, lim, cancel));
    let mut comps = Vec::new();
    for c in [4usize, 8, 6] {
        match imaging::conn_comp(&bin, c, lim, cancel) {
            Ok(cs) => {
                let _ = writeln!(out, "conn_comp_{c}: {}", comps_hash(&cs));
                if c == 8 {
                    comps = cs;
                }
            }
            Err(e) => {
                let _ = writeln!(out, "conn_comp_{c}: ERR {}", class(&e));
            }
        }
    }
    val(
        out,
        "conn_comp_gray",
        imaging::conn_comp(g, 4, lim, cancel).map(|cs| comps_hash(&cs)),
    );
    rec(
        out,
        "distance",
        imaging::distance_function(&bin, lim, cancel),
    );
    if let Some(er) = &eroded {
        for c in [4usize, 8, 6] {
            rec(
                out,
                &format!("seed_fill_{c}"),
                imaging::seed_fill(er, &bin, c, lim, cancel),
            );
        }
        let r =
            Bitmap::new(w + 1, h, 1, lim).and_then(|m| imaging::seed_fill(er, &m, 4, lim, cancel));
        rec(out, "seed_fill_mismatch", r);
    }
    let (wi, hi) = (w as isize, h as isize);
    for (name, op) in [
        ("clear", Op::Clear),
        ("notsrc", Op::NotSrc),
        ("xor", Op::SrcXorDst),
        ("and", Op::SrcAndDst),
        ("src", Op::Src),
        ("or", Op::SrcOrDst),
        ("set", Op::Set),
    ] {
        if let Some(er) = &eroded {
            let mut d = bin.clone();
            let r = imaging::raster_op(
                &mut d,
                rect(-3, -2, wi / 2 + 1, hi / 2 + 1),
                op,
                Some(er),
                Point { x: 2, y: 1 },
                lim,
                cancel,
            );
            rec(out, &format!("rop_{name}"), r.map(|()| d));
        }
        let mut d = g.clone();
        let r = imaging::raster_op(
            &mut d,
            rect(1, 1, wi, hi),
            op,
            Some(g),
            Point { x: 0, y: 2 },
            lim,
            cancel,
        );
        rec(out, &format!("rop8_{name}"), r.map(|()| d));
    }
    let mut d = bin.clone();
    let whole = d.bounds();
    let r = imaging::raster_op(
        &mut d,
        whole,
        Op::Src,
        Some(g),
        Point { x: 0, y: 0 },
        lim,
        cancel,
    );
    rec(out, "rop_depth_mismatch", r.map(|()| d));
    val(out, "grid", grid_report(g, &comps, lim));
    val(
        out,
        "grid_bad_cell",
        Grid::new(g.bounds(), 0, lim).map(|_| "ok".to_string()),
    );
}

fn recog_report(
    out: &mut String,
    r: &mut Recognizer,
    g: &Bitmap,
    lim: &Limits,
    cancel: &AtomicBool,
) {
    let fwd = r.forward(g, lim, cancel);
    let t = match fwd {
        Ok((t, norm)) => {
            let mut f = Fnv::new();
            for i in 0..t.map.len() {
                for &v in t.row(i) {
                    f.u32(f32b(v));
                }
            }
            let _ = writeln!(
                out,
                "forward: {} {} {:016x} scale={:016x} black={:08x} contrast={:08x}",
                t.map.len(),
                t.features,
                f.0,
                f64b(norm.scale_factor),
                f32b(norm.black),
                f32b(norm.contrast)
            );
            t
        }
        Err(e) => {
            let _ = writeln!(out, "forward: ERR {}", class(&e));
            return;
        }
    };
    match r.greedy_decode(&t, lim, cancel) {
        Ok(syms) => {
            let mut s = syms.len().to_string();
            for x in &syms {
                let _ = write!(
                    s,
                    " {}:{}:{}-{}",
                    x.unichar_id,
                    hex(&x.text),
                    x.start,
                    x.end
                );
            }
            let _ = writeln!(out, "greedy: {s}");
        }
        Err(e) => {
            let _ = writeln!(out, "greedy: ERR {}", class(&e));
        }
    }
    for lex in [true, false] {
        r.set_lexicon(lex);
        match r.beam_decode(&t, lim, cancel) {
            Ok(sc) => {
                let _ = writeln!(out, "beam_lexicon_{lex}: {}", sc.len());
                for x in &sc {
                    let _ = writeln!(
                        out,
                        "  id={} text={} s={} e={} c={:016x} r={:016x}",
                        x.symbol.unichar_id,
                        hex(&x.symbol.text),
                        x.symbol.start,
                        x.symbol.end,
                        f64b(x.certainty),
                        f64b(x.rating)
                    );
                }
            }
            Err(e) => {
                let _ = writeln!(out, "beam_lexicon_{lex}: ERR {}", class(&e));
            }
        }
        if lex {
            match r.recognize(g, lim, cancel) {
                Ok(l) => {
                    let _ = writeln!(
                        out,
                        "recognize: text={} conf={:016x} bounds={}",
                        hex(&l.text),
                        f64b(l.confidence),
                        rect_s(&l.bounds)
                    );
                    for w in &l.words {
                        let _ = writeln!(
                            out,
                            "  word={} bounds={} conf={:016x}",
                            hex(&w.text),
                            rect_s(&w.bounds),
                            f64b(w.confidence)
                        );
                    }
                    out.push('\n');
                }
                Err(e) => {
                    let _ = writeln!(out, "recognize: ERR {}", class(&e));
                }
            }
        }
    }
    r.set_lexicon(true);
}

fn vectors(data: &[u8]) -> Result<String, String> {
    if data.is_empty() || !data.len().is_multiple_of(8) || data.len() > 8 * 4096 {
        return Err("bad_input".into());
    }
    let (chunks, _) = data.as_chunks::<8>();
    let v: Vec<f64> = chunks.iter().map(|c| f64::from_le_bytes(*c)).collect();
    let h = |v: &[f64]| {
        let mut f = Fnv::new();
        for &x in v {
            f.u64(f64b(x));
        }
        format!("{:016x}", f.0)
    };
    let (mut t, mut l, mut s) = (v.clone(), v.clone(), v.clone());
    nn::tanh_in_place(&mut t);
    nn::logistic_in_place(&mut l);
    nn::softmax_in_place(&mut s);
    let mut out = format!("tanh {}\nlogistic {}\nsoftmax {}\n", h(&t), h(&l), h(&s));
    for &x in v.iter().take(8) {
        let _ = writeln!(
            out,
            "{:016x} tanh={:016x} logistic={:016x}",
            f64b(x),
            f64b(nn::tanh(x)),
            f64b(nn::logistic(x))
        );
    }
    Ok(out)
}

fn unicharset(data: &[u8], lim: &Limits) -> Result<String, Error> {
    let u = tessdata::parse_unicharset(data, &Budget::new(lim))?;
    let mut f = Fnv::new();
    for i in 0..u.size() {
        if let Some(c) = u.char(i) {
            f.str(&c.text);
            f.str(&c.normed);
            f.str(&c.script);
            f.u32(c.properties);
        }
        f.str(u.text(i));
        f.str(u.normed(i));
        f.flag(u.is_space_delimited(i));
    }
    Ok(format!("size {} {:016x}\n", u.size(), f.0))
}

fn recoder(data: &[u8], swap: bool, lim: &Limits) -> Result<String, Error> {
    let rc = Recoder::parse(data, swap, &Budget::new(lim))?;
    let mut f = Fnv::new();
    for i in 0..rc.size() {
        let code = rc.encode(i).unwrap_or(&[]);
        f.u32(code.len() as u32);
        for &c in code {
            f.u32(c as u32);
        }
        if let Some(id) = rc.decode_unichar(code) {
            f.u32(id as u32);
            f.u8(1);
        }
    }
    for c in -1..=(rc.code_range() as i32) {
        f.flag(rc.is_valid_first_code(c));
    }
    Ok(format!(
        "size {} range {} maxlen {} {:016x}\n",
        rc.size(),
        rc.code_range(),
        rc.max_code_len(),
        f.0
    ))
}

/// Looked up in each dawg, as unichar id sequences of the unedited model's unicharset (a word with
/// a character the charset lacks is dropped); the same list as the Go oracle's `probeWords`.
const PROBE_WORDS: [&str; 22] = [
    "the",
    "of",
    "and",
    "to",
    "in",
    "is",
    "that",
    "for",
    "it",
    "as",
    "hello",
    "world",
    "Tesseract",
    "recognition",
    "dictionary",
    "cadmus",
    "12",
    "1999",
    "a",
    "I",
    "zzzz",
    "xqzv",
];

fn word_ids(model: &[u8], lim: &Limits) -> Vec<Vec<usize>> {
    let Ok(c) = tessdata::parse_container(model, lim) else {
        return Vec::new();
    };
    let Ok(u) = tessdata::parse_unicharset(
        c.entry(Type::LstmUnicharset).unwrap_or(&[]),
        &Budget::new(lim),
    ) else {
        return Vec::new();
    };
    let mut ids = std::collections::HashMap::new();
    for id in 0..u.size() {
        ids.entry(u.text(id).to_string()).or_insert(id);
    }
    PROBE_WORDS
        .iter()
        .filter_map(|w| {
            w.chars()
                .map(|c| ids.get(c.to_string().as_str()).copied())
                .collect()
        })
        .collect()
}

fn dawg(data: &[u8], swap: bool, model: &[u8], lim: &Limits) -> Result<String, Error> {
    let d = Dawg::parse(data, swap, &Budget::new(lim))?;
    let mut f = Fnv::new();
    let n = d.num_edges();
    for e in 0..n {
        let e = e as i64;
        f.u32(d.unichar_id(e) as u32);
        f.u64(d.next_node(e) as u64);
        f.flag(d.end_of_word(e));
    }
    let mut probes: Vec<Vec<usize>> = vec![
        vec![1],
        vec![1, 2],
        vec![2, 3, 4],
        vec![5],
        vec![10, 11],
        vec![],
    ];
    for w in word_ids(model, lim) {
        probes.push(w.clone());
        probes.push(w[..w.len() - 1].to_vec());
    }
    for p in &probes {
        f.flag(d.contains(p));
        f.flag(d.has_prefix(p));
    }
    for node in (0..n).step_by(7) {
        for id in 0..8 {
            f.u64(d.edge_char(node as i64, id, false) as u64);
            f.u64(d.edge_char(node as i64, id, true) as u64);
        }
    }
    Ok(format!(
        "edges {n} size {} {:016x}\n",
        d.unicharset_size, f.0
    ))
}

fn summary(out: &mut String, m: &Model, lim: &Limits) {
    let rec = &m.recognizer;
    let _ = writeln!(out, "version {} swapped {}", hex(&m.version), m.swapped);
    let _ = writeln!(
        out,
        "spec {} flags {} iter {} sample {} null {} adam {:08x} lr {:08x} mom {:08x}",
        hex(&rec.network_str),
        rec.training_flags,
        rec.training_iteration,
        rec.sample_iteration,
        rec.null_char,
        f32b(rec.adam_beta),
        f32b(rec.learning_rate),
        f32b(rec.momentum)
    );
    let _ = writeln!(
        out,
        "unicharset {} recoder {} range {} maxlen {}",
        m.unicharset.size(),
        m.recoder.size(),
        m.recoder.code_range(),
        m.recoder.max_code_len()
    );
    for d in [&m.punc_dawg, &m.word_dawg, &m.number_dawg] {
        match d {
            None => out.push_str("dawg -\n"),
            Some(d) => {
                let _ = writeln!(out, "dawg {}", d.num_edges());
            }
        }
    }
    match rec.network.tree(lim) {
        Ok(t) => out.push_str(&t),
        Err(e) => {
            let _ = writeln!(out, "tree: ERR {}", class(&e));
        }
    }
}

fn apply_edits(base: &[u8], lines: &[&str]) -> Vec<u8> {
    let mut d = base.to_vec();
    for l in lines {
        let f: Vec<&str> = l.split_whitespace().collect();
        let n = |i: usize| f.get(i).and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
        match f.first().copied() {
            Some("trunc") if n(1) < d.len() => d.truncate(n(1)),
            Some("flip") if n(1) < d.len() => d[n(1)] ^= 1u8 << n(2),
            Some("set") if n(1) < d.len() => d[n(1)] = n(2) as u8,
            // setf OFFSET HEXBITS: eight bytes, the bits of an f64 (ignored past the end).
            Some("setf") if n(1).saturating_add(8) <= d.len() => {
                let bits = f.get(2).and_then(|s| u64::from_str_radix(s, 16).ok());
                d[n(1)..n(1) + 8].copy_from_slice(&bits.unwrap_or(0).to_le_bytes());
            }
            // data HEX: replace everything with these bytes.
            Some("data") => {
                let h = f.get(1).copied().unwrap_or("").as_bytes();
                d = h
                    .chunks(2)
                    .filter_map(|p| u8::from_str_radix(std::str::from_utf8(p).ok()?, 16).ok())
                    .collect();
            }
            _ => {}
        }
    }
    d
}

/// The fixed synthetic line the mutated models read: a gradient with xorshift noise.
fn noise_line(lim: &Limits) -> Result<Bitmap, Error> {
    const W: usize = 120;
    const H: usize = 36;
    let mut x: u64 = 88172645463325252;
    let mut pix = vec![0u8; W * H];
    for (i, p) in pix.iter_mut().enumerate() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *p = (((i % W) * 2) as u64 + x % 64) as u8 ^ ((i / W * 3) as u8);
    }
    Bitmap::from_gray8(W, H, &pix, lim)
}

fn part_type(p: &str) -> Option<Type> {
    Some(match p {
        "uni" => Type::LstmUnicharset,
        "rec" => Type::LstmRecoder,
        "lstm" => Type::Lstm,
        "dawgp" => Type::LstmPuncDawg,
        "dawgw" => Type::LstmSystemDawg,
        "dawgn" => Type::LstmNumberDawg,
        _ => return None,
    })
}

fn mutant(
    model: Option<&[u8]>,
    recipe: &str,
    lim: &Limits,
    cancel: &AtomicBool,
) -> Result<String, String> {
    let Some(model) = model else {
        return Err("no_model".into());
    };
    let lines: Vec<&str> = recipe.trim().lines().collect();
    let first: Vec<&str> = lines[0].split_whitespace().collect();
    if first.first() == Some(&"part") {
        let swap = first.get(2) == Some(&"swap");
        let t = part_type(first.get(1).copied().unwrap_or("")).ok_or("bad_recipe")?;
        let c = tessdata::parse_container(model, lim).map_err(|e| class(&e))?;
        let d = apply_edits(c.entry(t).unwrap_or(&[]), &lines[1..]);
        let r = match first[1] {
            "uni" => unicharset(&d, lim),
            "rec" => recoder(&d, swap, lim),
            "lstm" => {
                tessdata::parse_recognizer(&d, swap, &Budget::new(lim), cancel).and_then(|rec| {
                    Ok(format!(
                        "spec {} flags {} iter {} sample {} null {} adam {:08x} lr {:08x} mom {:08x}\n{}",
                        hex(&rec.network_str),
                        rec.training_flags,
                        rec.training_iteration,
                        rec.sample_iteration,
                        rec.null_char,
                        f32b(rec.adam_beta),
                        f32b(rec.learning_rate),
                        f32b(rec.momentum),
                        rec.network.tree(lim)?
                    ))
                })
            }
            _ => dawg(&d, swap, model, lim),
        };
        return r.map_err(|e| class(&e).to_string());
    }
    let m = apply_edits(model, &lines);
    if m.len() > lim.max_model_bytes {
        return Err("model_limit".into());
    }
    let mut out = String::new();
    match Model::load(&m, lim, cancel) {
        Ok(model) => {
            out.push_str("load: ok\n");
            summary(&mut out, &model, lim);
        }
        Err(e) => {
            let _ = writeln!(out, "load: ERR {}", class(&e));
        }
    }
    match Recognizer::new(&m, lim, cancel) {
        Ok(mut r) => {
            out.push_str("recognizer: ok\n");
            let g = noise_line(lim).map_err(|e| class(&e))?;
            recog_report(&mut out, &mut r, &g, lim, cancel);
        }
        Err(e) => {
            let _ = writeln!(out, "recognizer: ERR {}", class(&e));
        }
    }
    Ok(out)
}

fn run(
    name: &str,
    data: &[u8],
    rec: Option<&mut Recognizer>,
    model: Option<&[u8]>,
) -> Result<String, String> {
    let lim = limits();
    let cancel = AtomicBool::new(false);
    let ext = name.rsplit('.').next().unwrap_or("");
    match ext {
        "pgm" => {
            let g = Bitmap::from_pgm(data, &lim).map_err(|e| class(&e))?;
            let mut out = format!("dims {} {}\n", g.width, g.height);
            val(
                &mut out,
                "gray8",
                Bitmap::from_gray8(g.width, g.height, &g.pix, &lim).map(|b| hash_bitmap(&b)),
            );
            imaging_report(&mut out, &g, &lim, &cancel);
            if let Some(r) = rec
                && (name.starts_with("line-") || name.starts_with("rnd-"))
            {
                recog_report(&mut out, r, &g, &lim, &cancel);
            }
            Ok(out)
        }
        "rgba" => {
            // [w u32 LE][h u32 LE][w*h*4 bytes of premultiplied RGBA]
            let u = |i: usize| {
                data.get(i..i + 4)
                    .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            };
            let (Some(w), Some(h)) = (u(0), u(4)) else {
                return Err("imaging".into());
            };
            let g = Bitmap::from_rgba8(w as usize, h as usize, &data[8..], &lim)
                .map_err(|e| class(&e))?;
            let mut out = format!("dims {} {}\nrgba: {}\n", g.width, g.height, hash_bitmap(&g));
            imaging_report(&mut out, &g, &lim, &cancel);
            if let Some(r) = rec {
                recog_report(&mut out, r, &g, &lim, &cancel);
            }
            Ok(out)
        }
        "nn" => vectors(data),
        "mut" => mutant(model, &String::from_utf8_lossy(data), &lim, &cancel),
        _ => Err("unknown_input".into()),
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut model_path = None;
    while let Some(flag @ ("-model" | "-threads" | "-pool")) = args.first().map(String::as_str) {
        let flag = flag.to_string();
        args.remove(0);
        if args.is_empty() {
            eprintln!("usage: oracle [-model m] [-threads n] [-pool p] <inputs> <out>");
            std::process::exit(2);
        }
        let v = args.remove(0);
        match flag.as_str() {
            "-model" => model_path = Some(v),
            "-threads" => {
                let n = v.parse::<usize>().expect("-threads takes a number");
                THREADS.store(n, Ordering::Relaxed);
            }
            _ => match v.as_str() {
                "shared" => {}
                "caller" => {
                    let n = std::thread::available_parallelism().map_or(1, |n| n.get());
                    let pool = ThreadPool::new(n).expect("a pool of the CPUs");
                    CALLER_POOL.set(pool).expect("-pool once");
                }
                "percall" => PER_CALL.store(true, Ordering::Relaxed),
                _ => {
                    eprintln!("-pool takes shared, caller or percall");
                    std::process::exit(2);
                }
            },
        }
    }
    let [input, output] = args.as_slice() else {
        eprintln!("usage: oracle [-model m] [-threads n] [-pool p] <inputs> <out>");
        std::process::exit(2);
    };
    let model = model_path.map(|p| std::fs::read(&p).expect("read the model"));
    let mut rec = model
        .as_deref()
        .map(|m| Recognizer::new(m, &limits(), &AtomicBool::new(false)).expect("the model loads"));
    std::fs::create_dir_all(output).expect("create the output dir");
    let mut names: Vec<String> = std::fs::read_dir(input)
        .expect("read the input dir")
        .filter_map(Result::ok)
        .filter(|e| e.path().is_file())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    for (n, name) in names.iter().enumerate() {
        let data = std::fs::read(Path::new(input).join(name)).expect("read an input");
        let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
            run(name, &data, rec.as_mut(), model.as_deref())
        }))
        .unwrap_or_else(|_| Err("panic".into()));
        let (file, text) = match r {
            Ok(s) => (format!("{name}.out"), s),
            Err(c) => (format!("{name}.err"), format!("{c}\n")),
        };
        std::fs::write(Path::new(output).join(file), text).expect("write an output");
        if (n + 1).is_multiple_of(2000) {
            eprintln!("{}/{}", n + 1, names.len());
        }
    }
    eprintln!("{} inputs", names.len());
}
