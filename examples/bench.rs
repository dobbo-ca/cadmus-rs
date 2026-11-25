//! `bench <eng.traineddata> <testdata> [max_threads|nopool]`: prints `name<TAB>ms` for the recognizer on the line
//! corpus and the imaging operators on the Leptonica scan, as `oracle -bench` does for Go. Each is
//! the fastest of at least 3 runs (and up to 1 s of them) after a warm-up run. `max_threads`
//! defaults to `Limits::default()`'s, run on one `ThreadPool` made up front and reused for every
//! call, as a server would; `nopool`: the default `max_threads` with `Limits::default()`'s no
//! pool, so the crate's shared pool; 1 uses no thread. See `tests/parity/bench.sh`.

#![forbid(unsafe_code)]

use cadmus::imaging::{self, Bitmap};
use cadmus::recog::{Line, Recognizer, Workspace};
use cadmus::{Error, Limits, ThreadPool};
use std::fs;
use std::hint::black_box;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

/// The warm-up run must succeed, so a call that returns an error early cannot show up as a fast
/// time.
fn time_it<T>(name: &str, mut f: impl FnMut() -> Result<T, Error>) {
    black_box(f().unwrap_or_else(|e| panic!("{name}: {e}")));
    let mut best = Duration::MAX;
    let start = Instant::now();
    let mut n = 0;
    while n < 3 || (start.elapsed() < Duration::from_secs(1) && n < 200) {
        let t = Instant::now();
        let _ = black_box(f());
        best = best.min(t.elapsed());
        n += 1;
    }
    println!("{name}\t{:.3}", best.as_nanos() as f64 / 1e6);
}

fn empty_line() -> Line {
    Line {
        text: String::new(),
        words: Vec::new(),
        bounds: Default::default(),
        confidence: 0.0,
    }
}

fn load_lines(dir: &str, lim: &Limits) -> Vec<Bitmap> {
    let mut files: Vec<String> = fs::read_dir(dir)
        .expect("read the line dir")
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".pgm"))
        .collect();
    files.sort();
    files
        .iter()
        .map(|f| {
            let data = fs::read(format!("{dir}/{f}")).expect("read a line");
            Bitmap::from_pgm(&data, lim).expect("a PGM line")
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (model, testdata, threads) = match args.as_slice() {
        [_, m, t] => (m, t, None),
        [_, m, t, n] => (m, t, Some(n.as_str())),
        _ => {
            eprintln!("usage: bench <eng.traineddata> <testdata> [max_threads|nopool]");
            std::process::exit(2);
        }
    };
    let mut lim = Limits::default();
    let mut caller_pool = true;
    match threads {
        None => {}
        Some("nopool") => caller_pool = false,
        Some(n) => lim.max_threads = n.parse().expect("max_threads is a number or nopool"),
    }
    if caller_pool {
        lim.pool = Some(ThreadPool::new(lim.max_threads).expect("a thread pool"));
    }
    let cancel = AtomicBool::new(false);
    let model = fs::read(model).expect("read the model");
    let r = Recognizer::new(&model, &lim, &cancel).expect("the model loads");
    for set in ["native", "h36"] {
        let lines = load_lines(&format!("{testdata}/lines/{set}"), &lim);
        let outs: Vec<_> = lines
            .iter()
            .map(|g| r.forward(g, &lim, &cancel).expect("forward").0)
            .collect();
        time_it(&format!("forward_{set}"), || {
            for g in &lines {
                black_box(r.forward(g, &lim, &cancel))?;
            }
            Ok(())
        });
        time_it(&format!("beam_{set}"), || {
            for o in &outs {
                black_box(r.beam_decode(o, &lim, &cancel))?;
            }
            Ok(())
        });
        time_it(&format!("recognize_{set}"), || {
            for g in &lines {
                black_box(r.recognize(g, &lim, &cancel))?;
            }
            Ok(())
        });
        // One workspace and one line kept across all the calls (including the warm-up run).
        let (mut ws, mut line) = (Workspace::default(), empty_line());
        time_it(&format!("recognize_reuse_{set}"), || {
            for g in &lines {
                r.recognize_into(&mut ws, g, &mut line, &lim, &cancel)?;
                black_box(&line);
            }
            Ok(())
        });
    }
    let raw = fs::read(format!("{testdata}/golden/gray.bin")).expect("read gray.bin");
    let word = |i: usize| u32::from_le_bytes([raw[i], raw[i + 1], raw[i + 2], raw[i + 3]]) as usize;
    let (w, h) = (word(0), word(4));
    let g = Bitmap::from_gray8(w, h, &raw[12..12 + w * h], &lim).expect("the scan");
    let bin = imaging::otsu(&g, &lim, &cancel).expect("otsu");
    let eroded = imaging::erode(&bin, 3, 3, &lim, &cancel).expect("erode");
    time_it("otsu", || imaging::otsu(&g, &lim, &cancel));
    time_it("sauvola_15", || {
        imaging::sauvola(&g, 15, 0.34, &lim, &cancel)
    });
    time_it("skew_angle", || imaging::skew_angle(&bin, &lim, &cancel));
    time_it("rotate", || imaging::rotate(&bin, 0.02, &lim, &cancel));
    time_it("scale_0.5", || imaging::scale_gray(&g, 0.5, &lim, &cancel));
    time_it("scale_1.7", || imaging::scale_gray(&g, 1.7, &lim, &cancel));
    time_it("dilate_3x3", || imaging::dilate(&bin, 3, 3, &lim, &cancel));
    time_it("erode_3x3", || imaging::erode(&bin, 3, 3, &lim, &cancel));
    time_it("open_5x5", || imaging::open(&bin, 5, 5, &lim, &cancel));
    time_it("close_5x5", || imaging::close(&bin, 5, 5, &lim, &cancel));
    time_it("conn_comp_8", || imaging::conn_comp(&bin, 8, &lim, &cancel));
    time_it("distance", || {
        imaging::distance_function(&bin, &lim, &cancel)
    });
    time_it("seed_fill_8", || {
        imaging::seed_fill(&eroded, &bin, 8, &lim, &cancel)
    });
}
