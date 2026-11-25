//! Cuts text-line crops out of the small untiled pages of whole-slide images (label, macro,
//! thumbnail), for the real-file OCR gate (`docs/quality.md`). The crops are cut once, here, and
//! the Go and Rust oracles then read the same files, so this cutter does not have to match either
//! library: it only has to be deterministic.
//!
//! `labels <out-dir> <slide>...`: every page of every slide that is not tiled, 8-bit grey or RGB and
//! at most `MAX_PAGE_PIXELS` is decoded with tiff-rs (one thread) and turned to grey. Each of its
//! three orientations (as stored, turned 90 and 270 degrees) is thresholded with Otsu, and every run
//! of rows with ink that is `MIN_H..=MAX_H` rows high becomes one crop: the run's rows and ink
//! columns plus a `PAD` pixel margin (a taller run is cut again with a stricter row threshold),
//! from the grey image, as `line-<slide>-p<page>-r<deg>-<n>.pgm`.
//! A log line per slide and page goes to stdout.

use std::ops::ControlFlow;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::{env, fs, process};

/// The largest page decoded: labels and macros are at most a few megapixels.
const MAX_PAGE_PIXELS: usize = 8 << 20;
/// Height of a kept row run, in rows.
const MIN_H: usize = 8;
const MAX_H: usize = 160;
/// Width of a kept run's ink, in columns.
const MIN_W: usize = 16;
/// Margin around a crop, in pixels.
const PAD: usize = 4;
/// Crops per page and orientation.
const MAX_LINES: usize = 40;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let Some((out, slides)) = args.split_first() else {
        usage()
    };
    if slides.is_empty() {
        usage()
    }
    if let Err(e) = fs::create_dir_all(out) {
        fail(&format!("{out}: {e}"))
    }
    let mut total = 0;
    for s in slides {
        total += slide(s, out);
    }
    println!("{total} line crops");
}

fn usage() -> ! {
    eprintln!("usage: labels <out-dir> <slide>...");
    process::exit(2)
}

fn fail(msg: &str) -> ! {
    eprintln!("labels: {msg}");
    process::exit(2)
}

fn slide(path: &str, out: &str) -> usize {
    let name = Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().replace('.', "_"))
        .unwrap_or_default();
    let f = fs::File::open(path).unwrap_or_else(|e| fail(&format!("{path}: {e}")));
    let size = f
        .metadata()
        .unwrap_or_else(|e| fail(&format!("{path}: {e}")))
        .len();
    let lim = tiff::Limits {
        max_threads: 1,
        ..tiff::Limits::default()
    };
    let tf = match tiff::File::open(f, size, &lim) {
        Ok(t) => t,
        Err(e) => {
            println!("{name}: not read: {e}");
            return 0;
        }
    };
    let mut n = 0;
    for i in 0..tf.num_pages() {
        let page = match tf.page(i) {
            Ok(p) => p,
            Err(e) => {
                println!("{name} p{i}: not read: {e}");
                continue;
            }
        };
        let info = page.info();
        let (w, h, spp) = (info.width, info.height, info.samples_per_pixel);
        if page.layout().tiled
            || w == 0
            || h == 0
            || info.bits_per_sample != 8
            || (spp != 1 && spp != 3)
            || !info.extra_samples.is_empty()
            || info.photometric == tiff::PALETTE
            || info.photometric == tiff::CMYK
            || info.row_bytes != (w * spp) as u64
            || w.checked_mul(h).is_none_or(|p| p > MAX_PAGE_PIXELS)
        {
            continue;
        }
        let invert = info.photometric == tiff::WHITE_IS_ZERO;
        let mut gray = Vec::with_capacity(w * h);
        let cancel = AtomicBool::new(false);
        let res = page.rows(&cancel, |_, band| {
            for px in band.chunks_exact(spp) {
                let g = if spp == 3 {
                    luma(px[0], px[1], px[2])
                } else {
                    px[0]
                };
                gray.push(if invert { 255 - g } else { g });
            }
            ControlFlow::Continue(())
        });
        if let Err(e) = res {
            println!("{name} p{i} {w}x{h}: decode failed: {e}");
            continue;
        }
        if gray.len() != w * h {
            println!("{name} p{i} {w}x{h}: {} samples", gray.len());
            continue;
        }
        let mut counts = Vec::new();
        for deg in [0, 90, 270] {
            let (img, iw, ih) = turn(&gray, w, h, deg);
            let lines = cut(&img, iw, ih);
            counts.push(format!("r{deg} {}", lines.len()));
            for (k, &(x0, y0, x1, y1)) in lines.iter().enumerate() {
                let file = format!("{out}/line-{name}-p{i:02}-r{deg:03}-{k:03}.pgm");
                let mut pgm = format!("P5\n{} {}\n255\n", x1 - x0, y1 - y0).into_bytes();
                for y in y0..y1 {
                    pgm.extend_from_slice(&img[y * iw + x0..y * iw + x1]);
                }
                if let Err(e) = fs::write(&file, pgm) {
                    fail(&format!("{file}: {e}"))
                }
                n += 1;
            }
        }
        println!(
            "{name} p{i} {w}x{h} spp {spp} compression {}: {}",
            info.compression,
            counts.join(", ")
        );
    }
    n
}

/// Go's `color.GrayModel` on 8-bit samples.
fn luma(r: u8, g: u8, b: u8) -> u8 {
    let (r, g, b) = (
        u32::from(r) * 0x101,
        u32::from(g) * 0x101,
        u32::from(b) * 0x101,
    );
    ((19595 * r + 38470 * g + 7471 * b + (1 << 15)) >> 24) as u8
}

/// `img` turned clockwise by `deg` (0, 90 or 270), with its width and height.
fn turn(img: &[u8], w: usize, h: usize, deg: u32) -> (Vec<u8>, usize, usize) {
    if deg == 0 {
        return (img.to_vec(), w, h);
    }
    // The turned image is h wide and w high.
    let mut out = vec![0u8; w * h];
    for y in 0..w {
        for x in 0..h {
            let (sx, sy) = if deg == 90 {
                (y, h - 1 - x)
            } else {
                (w - 1 - y, x)
            };
            out[y * h + x] = img[sy * w + sx];
        }
    }
    (out, h, w)
}

/// The boxes `(x0, y0, x1, y1)` of the text-line crops of `img`, top to bottom.
fn cut(img: &[u8], w: usize, h: usize) -> Vec<(usize, usize, usize, usize)> {
    let t = otsu(img);
    // Ink is the minority class: dark text on light, or light text on dark.
    let dark = img.iter().filter(|&&v| v <= t).count();
    let light_ink = dark * 2 > img.len();
    let ink = |v: u8| if light_ink { v > t } else { v <= t };
    let min_ink = (w / 200).max(2);
    let counts: Vec<usize> = img
        .chunks_exact(w)
        .map(|r| r.iter().filter(|&&v| ink(v)).count())
        .collect();
    // A run taller than MAX_H is a photographed page whose every row has a few ink pixels, not a
    // line: it is cut again with a row threshold of 3% of the page width.
    let mut runs = Vec::new();
    for (y0, y1) in row_runs(&counts, min_ink, 0, h) {
        if y1 - y0 > MAX_H {
            runs.extend(row_runs(&counts, (w * 3 / 100).max(min_ink + 1), y0, y1));
        } else {
            runs.push((y0, y1));
        }
    }
    let mut boxes = Vec::new();
    for (y0, y1) in runs {
        if boxes.len() == MAX_LINES {
            break;
        }
        if !(MIN_H..=MAX_H).contains(&(y1 - y0)) {
            continue;
        }
        let cols = (0..w).filter(|&x| (y0..y1).any(|y| ink(img[y * w + x])));
        let (mut x0, mut x1) = (usize::MAX, 0);
        for x in cols {
            x0 = x0.min(x);
            x1 = x + 1;
        }
        if x1 <= x0 || x1 - x0 < MIN_W {
            continue;
        }
        boxes.push((
            x0.saturating_sub(PAD),
            y0.saturating_sub(PAD),
            (x1 + PAD).min(w),
            (y1 + PAD).min(h),
        ));
    }
    boxes
}

/// The runs `(start, end)` of rows in `from..to` with at least `min_ink` ink pixels; gaps of at most
/// 2 rows join two runs.
fn row_runs(counts: &[usize], min_ink: usize, from: usize, to: usize) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut y = from;
    while y < to {
        if counts[y] < min_ink {
            y += 1;
            continue;
        }
        let start = y;
        while y < to && counts[y] >= min_ink {
            y += 1;
        }
        let joined = match runs.last_mut() {
            Some(last) if start - last.1 <= 2 => {
                last.1 = y;
                true
            }
            _ => false,
        };
        if !joined {
            runs.push((start, y));
        }
    }
    runs
}

/// Otsu's threshold: the grey level `t` that splits `v <= t` from `v > t` with the largest
/// between-class variance.
fn otsu(img: &[u8]) -> u8 {
    let mut hist = [0u64; 256];
    for &v in img {
        hist[v as usize] += 1;
    }
    let total = img.len() as f64;
    let sum: f64 = hist
        .iter()
        .enumerate()
        .map(|(i, &c)| i as f64 * c as f64)
        .sum();
    let (mut w0, mut sum0, mut best, mut best_t) = (0.0, 0.0, -1.0, 0u8);
    for (t, &c) in hist.iter().enumerate() {
        w0 += c as f64;
        sum0 += t as f64 * c as f64;
        let w1 = total - w0;
        if w0 == 0.0 || w1 == 0.0 {
            continue;
        }
        let (m0, m1) = (sum0 / w0, (sum - sum0) / w1);
        let between = w0 * w1 * (m0 - m1) * (m0 - m1);
        if between > best {
            best = between;
            best_t = t as u8;
        }
    }
    best_t
}
