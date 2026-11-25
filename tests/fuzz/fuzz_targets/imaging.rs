#![no_main]
use cadmus::imaging::{self, Bitmap, Op, Point, Rect};
use cadmus::nn::Rand;
use cadmus::recog::normalize;
use cadmus::{Error, Limits};
use libfuzzer_sys::fuzz_target;
use std::sync::atomic::AtomicBool;

mod common;

/// Breaks one field of `b`, which a caller can do because they are public: bits 0-2 of `bad` pick
/// the field (none, stride, pix length, depth, width, height, stride past usize, depth swapped),
/// the rest of the bits the new value.
fn corrupt(mut b: Bitmap, bad: u8) -> Bitmap {
    let v = usize::from(bad >> 3);
    match bad & 7 {
        1 => b.stride /= 2,
        2 => b.pix.truncate(b.pix.len() / 2),
        3 => b.depth = v,
        4 => {
            b.width = if v & 1 != 0 {
                usize::MAX
            } else {
                b.width + 1 + v
            }
        }
        5 => {
            b.height = if v & 1 != 0 {
                usize::MAX
            } else {
                b.height + 1 + v
            }
        }
        6 => b.stride = usize::MAX,
        7 => b.depth = if b.depth == 8 { 1 } else { 8 },
        _ => {}
    }
    b
}

/// Every operator on `a`, with `b` where one takes a second bitmap, whatever the depth or shape of
/// either. Only a panic, or an allocator failure that the budget should have stopped, is a bug.
fn every(
    a: &Bitmap,
    b: &Bitmap,
    r: Rect,
    sp: Point,
    op: Op,
    (input_height, x_scale): (usize, usize),
    lim: &Limits,
    cancel: &AtomicBool,
) {
    common::no_oom(imaging::sauvola(a, 5, 0.3, lim, cancel));
    common::no_oom(imaging::scale_gray(a, 0.5, lim, cancel));
    common::within(lim.max_alloc, || {
        common::no_oom(imaging::rotate(a, 0.1, lim, cancel))
    });
    common::within(lim.max_alloc, || {
        common::no_oom(imaging::otsu_threshold(a, lim, cancel))
    });
    common::within(lim.max_alloc, || {
        common::no_oom(imaging::otsu(a, lim, cancel))
    });
    common::within(lim.max_alloc, || {
        common::no_oom(imaging::skew_angle(a, lim, cancel))
    });
    common::within(lim.max_alloc, || {
        common::no_oom(imaging::dilate(a, 3, 3, lim, cancel))
    });
    common::within(lim.max_alloc, || {
        common::no_oom(imaging::erode(a, 3, 3, lim, cancel))
    });
    common::within(lim.max_alloc, || {
        common::no_oom(imaging::open(a, 3, 3, lim, cancel))
    });
    common::within(lim.max_alloc, || {
        common::no_oom(imaging::close(a, 3, 3, lim, cancel))
    });
    common::no_oom(imaging::distance_function(a, lim, cancel));
    common::no_oom(imaging::conn_comp(a, 8, lim, cancel));
    common::no_oom(imaging::seed_fill(a, b, 4, lim, cancel));
    for src in [Some(b), None] {
        common::no_oom(imaging::raster_op(
            &mut a.clone(),
            r,
            op,
            src,
            sp,
            lim,
            cancel,
        ));
    }
    common::no_oom(normalize(
        a,
        input_height,
        x_scale,
        &Rand::new(7),
        lim,
        cancel,
    ));
}

// data = [flags][w][h][k][win][conn][op][x0][y0][x1][y1][bad][pixels]. Every operator on one image,
// under a budget between 256 bytes and 8 MiB (so every allocation is charged against something
// small) and, when bit 0 of `flags` is set, a cancel flag that is already set. Bits 1-3 of
// `flags` pick the scale factor (including tiny, zero, NaN and infinite ones), bits 4-5 the
// rotation angle (including NaN, infinite and f32-overflowing ones), bits 6-7 the Sauvola `k`.
// `bad` breaks one public field of a copy of each bitmap (see `corrupt`), and every operator also
// gets a gray image where it wants a bilevel one and the other way round, a missing source, and
// `normalize` gets fuzzed `input_height` and `x_scale`.
fuzz_target!(|data: &[u8]| {
    let Some((&[flags, w, h, k, win, conn, op, x0, y0, x1, y1, bad], pix)) =
        data.split_first_chunk::<12>()
    else {
        return;
    };
    let limits = Limits {
        max_alloc: 1 << (8 + k % 16),
        max_pixels: 1 << 20,
        ..Limits::default()
    };
    let lim = &limits;
    let cancel = &AtomicBool::new(flags & 1 != 0);
    let (w, h) = (1 + usize::from(w) % 64, 1 + usize::from(h) % 64);
    let Some(g) = common::no_oom(common::gray(w, h, pix, lim)) else {
        return;
    };
    let w3 = usize::from(win) % 40;
    let ks = [f64::from(k) / 128.0 - 0.5, -1e20, f64::NAN, 1e20];
    common::within(lim.max_alloc, || {
        common::no_oom(imaging::sauvola(
            &g,
            w3,
            ks[usize::from(flags >> 6)],
            lim,
            cancel,
        ))
    });
    let factors = [
        0.1 + f64::from(k) / 40.0,
        0.01,
        1e-20,
        1e-50,
        0.0,
        f64::NAN,
        f64::INFINITY,
        1e39,
    ];
    common::within(lim.max_alloc, || {
        common::no_oom(imaging::scale_gray(
            &g,
            factors[usize::from(flags >> 1) % 8],
            lim,
            cancel,
        ))
    });
    let angles = [0.1, f64::NAN, f64::INFINITY, 1e39];
    common::within(lim.max_alloc, || {
        common::no_oom(imaging::rotate(
            &g,
            angles[usize::from(flags >> 4) % 4],
            lim,
            cancel,
        ))
    });
    let Some(bin) = common::within(lim.max_alloc, || {
        common::no_oom(imaging::otsu(&g, lim, cancel))
    }) else {
        return;
    };
    assert_eq!((bin.width, bin.height, bin.depth), (w, h, 1));
    let ones = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .filter(|&(x, y)| bin.at(x, y) != 0)
        .count();
    common::within(lim.max_alloc, || {
        common::no_oom(imaging::rotate(
            &bin,
            angles[usize::from(flags >> 4) % 4],
            lim,
            cancel,
        ))
    });
    if let Some(a) = common::within(lim.max_alloc, || {
        common::no_oom(imaging::skew_angle(&bin, lim, cancel))
    }) {
        assert!(a.is_finite());
        common::within(lim.max_alloc, || {
            common::no_oom(imaging::rotate(&bin, a, lim, cancel))
        });
    }
    let (bw, bh) = (1 + usize::from(win) % 7, 1 + usize::from(conn) % 7);
    let within = |f: &dyn Fn() -> Result<Bitmap, Error>| common::within(lim.max_alloc, f);
    for out in [
        within(&|| imaging::dilate(&bin, bw, bh, lim, cancel)),
        within(&|| imaging::erode(&bin, bw, bh, lim, cancel)),
        within(&|| imaging::open(&bin, bw, bh, lim, cancel)),
        within(&|| imaging::close(&bin, bw, bh, lim, cancel)),
        imaging::distance_function(&bin, lim, cancel),
    ] {
        if let Some(o) = common::no_oom(out) {
            assert_eq!((o.width, o.height), (w, h));
        }
    }
    let connectivity = if conn & 1 == 0 { 4 } else { 8 };
    if let Some(cs) = common::no_oom(imaging::conn_comp(&bin, connectivity, lim, cancel)) {
        assert_eq!(cs.iter().map(|c| c.pixels).sum::<usize>(), ones);
        for c in &cs {
            assert!(!c.bounds.is_empty() && c.pixels > 0);
            assert!(c.bounds.min.x >= 0 && c.bounds.min.y >= 0);
            assert!(c.bounds.max.x <= w as isize && c.bounds.max.y <= h as isize);
        }
    }
    if let Some(er) = common::no_oom(imaging::erode(&bin, 3, 3, lim, cancel))
        && let Some(f) = common::no_oom(imaging::seed_fill(&er, &bin, connectivity, lim, cancel))
    {
        for y in 0..h {
            for x in 0..w {
                assert!(f.at(x, y) <= bin.at(x, y), "a fill stays inside its mask");
            }
        }
    }
    let ops = [
        Op::Clear,
        Op::NotSrc,
        Op::SrcXorDst,
        Op::SrcAndDst,
        Op::Src,
        Op::SrcOrDst,
        Op::Set,
    ];
    let mut d = bin.clone();
    let r = Rect::new(
        isize::from(x0 as i8),
        isize::from(y0 as i8),
        isize::from(x1 as i8),
        isize::from(y1 as i8),
    );
    let sp = Point {
        x: isize::from(x1 as i8),
        y: isize::from(y0 as i8),
    };
    let op_byte = op;
    let op = ops[usize::from(op) % ops.len()];
    let normalize_args = (
        [usize::from(win), 0, 1, 36, usize::MAX, usize::MAX / 2][usize::from(op_byte) % 6],
        [usize::from(x0), 0, 1, usize::MAX][usize::from(conn) % 4],
    );
    let (bad_g, bad_bin) = (corrupt(g.clone(), bad), corrupt(bin.clone(), bad));
    for (a, b) in [
        (&g, &bin),
        (&bin, &g),
        (&bad_g, &bad_bin),
        (&bad_bin, &bad_g),
    ] {
        every(a, b, r, sp, op, normalize_args, lim, cancel);
    }
    match imaging::raster_op(&mut d, r, op, Some(&bin), sp, lim, cancel) {
        Ok(()) | Err(Error::Imaging(_) | Error::Cancelled) => {}
        Err(e) => panic!("raster_op: {e}"),
    }
    let mut d8 = g.clone();
    match imaging::raster_op(&mut d8, r, op, Some(&g), sp, lim, cancel) {
        Ok(()) | Err(Error::Imaging(_) | Error::Cancelled) => {}
        Err(e) => panic!("raster_op 8bpp: {e}"),
    }
    // The same box and point scaled out to the edges of isize.
    let far = |v: u8| isize::from(v as i8) * (isize::MAX / 128);
    let r = Rect::new(far(x0), far(y0), far(x1), far(y1));
    let sp = Point {
        x: far(x1),
        y: far(y0),
    };
    match imaging::raster_op(&mut d, r, op, Some(&bin), sp, lim, cancel) {
        Ok(()) | Err(Error::Imaging(_) | Error::Cancelled) => {}
        Err(e) => panic!("raster_op: {e}"),
    }
});
