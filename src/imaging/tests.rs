use super::*;
use std::sync::atomic::AtomicBool;

fn lim() -> Limits {
    Limits::default()
}

fn nc() -> AtomicBool {
    AtomicBool::new(false)
}

fn bm(w: usize, h: usize, d: usize) -> Bitmap {
    Bitmap::new(w, h, d, &lim()).unwrap()
}

fn fill_rect(b: &mut Bitmap, x0: usize, y0: usize, w: usize, h: usize) {
    for y in y0..y0 + h {
        for x in x0..x0 + w {
            b.set(x, y, 1);
        }
    }
}

fn golden_bytes(name: &str) -> Vec<u8> {
    let p = format!("{}/tests/data/golden/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&p).unwrap_or_else(|e| panic!("golden {p}: {e}"))
}

fn i32_at(raw: &[u8], i: usize) -> i32 {
    i32::from_le_bytes([raw[4 * i], raw[4 * i + 1], raw[4 * i + 2], raw[4 * i + 3]])
}

/// A dump from tests/data/golden/gen/gen.c: three int32 (width, height, depth), then the rows as
/// 32-bit words, most significant bit first.
fn load_golden(name: &str) -> Bitmap {
    let raw = golden_bytes(name);
    assert!(raw.len() >= 12, "golden {name} is truncated");
    let (w, h, d) = (
        i32_at(&raw, 0) as usize,
        i32_at(&raw, 1) as usize,
        i32_at(&raw, 2) as usize,
    );
    let mut b = bm(w, h, d);
    let wpl = (w * d).div_ceil(32);
    let body = &raw[12..];
    for y in 0..h {
        for x in 0..w {
            let o = (y * wpl + (x * d) / 32) * 4;
            let word = u32::from_le_bytes([body[o], body[o + 1], body[o + 2], body[o + 3]]);
            match d {
                1 => b.set(x, y, ((word >> (31 - (x * d) % 32)) & 1) as u8),
                8 => b.set(x, y, ((word >> (24 - (x * d) % 32)) & 0xff) as u8),
                _ => panic!("unsupported golden depth {d}"),
            }
        }
    }
    b
}

fn count_diff(got: &Bitmap, want: &Bitmap) -> usize {
    assert_eq!((got.width, got.height), (want.width, want.height), "size");
    let mut diff = 0;
    for y in 0..got.height {
        for x in 0..got.width {
            if got.at(x, y) != want.at(x, y) {
                diff += 1;
            }
        }
    }
    diff
}

fn count_fg(b: &Bitmap) -> usize {
    let mut n = 0;
    for y in 0..b.height {
        for x in 0..b.width {
            if b.at(x, y) != 0 {
                n += 1;
            }
        }
    }
    n
}

fn load_components(name: &str) -> Vec<Component> {
    let raw = golden_bytes(name);
    let n = i32_at(&raw, 0) as usize;
    assert_eq!(raw.len(), 4 + 20 * n, "golden {name} size");
    (0..n)
        .map(|i| {
            let f = |k: usize| i32_at(&raw[4 + 20 * i..], k) as isize;
            Component {
                bounds: Rect::new(f(0), f(1), f(0) + f(2), f(1) + f(3)),
                pixels: f(4) as usize,
            }
        })
        .collect()
}

fn load_floats(name: &str, want: usize) -> Vec<f32> {
    let raw = golden_bytes(name);
    let n = i32_at(&raw, 0) as usize;
    assert!(n == want && raw.len() == 4 + 4 * n, "golden {name}");
    (0..n)
        .map(|i| f32::from_bits(i32_at(&raw, 1 + i) as u32))
        .collect()
}

fn same_pixels(name: &str, got: &Bitmap, want: &Bitmap) {
    let diff = count_diff(got, want);
    assert_eq!(diff, 0, "{name} differs from Leptonica in {diff} pixels");
}

#[test]
fn bitmap_basics() {
    let mut b = bm(10, 3, 1);
    assert_eq!(b.stride, 2);
    b.set(9, 2, 1);
    assert_eq!(b.at(9, 2), 1);
    assert_eq!(b.pix[2 * 2 + 1], 0x40);
    b.set(9, 2, 0);
    assert_eq!(b.pix[2 * 2 + 1], 0);
    assert_eq!(b.bounds(), Rect::new(0, 0, 10, 3));
    assert!(Bitmap::new(4, 4, 4, &lim()).is_err());
    let small = Limits {
        max_pixels: 100,
        ..lim()
    };
    assert_eq!(
        Bitmap::new(11, 10, 8, &small).unwrap_err(),
        Error::PixelLimit
    );
    let tight = Limits {
        max_alloc: 99,
        ..lim()
    };
    assert_eq!(
        Bitmap::new(10, 10, 8, &tight).unwrap_err(),
        Error::AllocLimit
    );
}

#[test]
fn from_gray_and_rgba() {
    let g = Bitmap::from_gray8(2, 1, &[7, 9], &lim()).unwrap();
    assert_eq!(g.pix, [7, 9]);
    assert!(Bitmap::from_gray8(2, 2, &[7, 9], &lim()).is_err());
    let c = Bitmap::from_rgba8(2, 1, &[255, 255, 255, 255, 0, 0, 0, 255], &lim()).unwrap();
    assert_eq!(c.pix, [255, 0]);
}

#[test]
fn otsu_matches_leptonica() {
    let gray = load_golden("gray.bin");
    same_pixels(
        "Otsu",
        &otsu(&gray, &lim(), &nc()).unwrap(),
        &load_golden("otsu.bin"),
    );
}

#[test]
fn sauvola_matches_leptonica() {
    let gray = load_golden("gray.bin");
    let got = sauvola(&gray, 17, 0.34375, &lim(), &nc()).unwrap();
    same_pixels("Sauvola", &got, &load_golden("sauvola.bin"));
}

#[test]
fn sauvola_foreground_is_one() {
    let mut b = bm(40, 40, 8);
    b.pix.fill(240);
    for y in 18..21 {
        for x in 18..21 {
            b.set(x, y, 10);
        }
    }
    let got = sauvola(&b, 9, 0.34375, &lim(), &nc()).unwrap();
    assert_eq!(got.at(19, 19), 1);
    assert_eq!(got.at(35, 35), 0);
}

#[test]
fn sauvola_follows_illumination_gradient() {
    let (w, h) = (120usize, 60usize);
    let mut b = bm(w, h, 8);
    for y in 0..h {
        for x in 0..w {
            b.set(x, y, (100 + 150 * x / (w - 1)) as u8);
        }
    }
    let marks = [10usize, 55, 100];
    for mx in marks {
        for y in 19..22 {
            for x in mx - 1..mx + 2 {
                b.set(x, y, 40);
            }
        }
    }
    let got = sauvola(&b, 9, 0.34375, &lim(), &nc()).unwrap();
    for mx in marks {
        assert_eq!(got.at(mx, 20), 1, "mark {mx}");
    }
    for x in [10, 60, 110] {
        assert_eq!(got.at(x, 50), 0, "background {x}");
    }
}

#[test]
fn sauvola_rejects_bad_windows() {
    let b = bm(40, 40, 8);
    for window in [0, 1, 3, 4, 100] {
        assert!(
            sauvola(&b, window, 0.3, &lim(), &nc()).is_err(),
            "window {window}"
        );
    }
}

#[test]
fn otsu_threshold_bimodal_and_foreground() {
    let mut b = bm(10, 10, 8);
    for y in 0..10 {
        for x in 0..10 {
            b.set(x, y, if y < 5 { 20 } else { 200 });
        }
    }
    let t = otsu_threshold(&b, &lim(), &nc()).unwrap();
    assert!((20..=200).contains(&t), "{t}");

    let mut b = bm(10, 10, 8);
    b.pix.fill(240);
    for y in 2..5 {
        for x in 2..5 {
            b.set(x, y, 10);
        }
    }
    let got = otsu(&b, &lim(), &nc()).unwrap();
    assert_eq!(got.at(3, 3), 1);
    assert_eq!(got.at(8, 8), 0);
}

#[test]
fn conn_comp_matches_leptonica() {
    let otsu_in = load_golden("otsu.bin");
    let diag_in = load_golden("conncomp_diag_in.bin");
    for conn in [4, 8] {
        let got = conn_comp(&otsu_in, conn, &lim(), &nc()).unwrap();
        assert_eq!(
            got,
            load_components(&format!("conncomp{conn}.bin")),
            "conn {conn}"
        );
        let got = conn_comp(&diag_in, conn, &lim(), &nc()).unwrap();
        assert_eq!(
            got,
            load_components(&format!("conncomp_diag{conn}.bin")),
            "diag conn {conn}"
        );
    }
}

fn comp(x0: isize, y0: isize, x1: isize, y1: isize, pixels: usize) -> Component {
    Component {
        bounds: Rect::new(x0, y0, x1, y1),
        pixels,
    }
}

#[test]
fn conn_comp_small_cases() {
    let blank = bm(32, 24, 1);
    let mut one = bm(32, 24, 1);
    one.set(7, 11, 1);
    for conn in [4, 8] {
        assert!(conn_comp(&blank, conn, &lim(), &nc()).unwrap().is_empty());
        assert_eq!(
            conn_comp(&one, conn, &lim(), &nc()).unwrap(),
            [comp(7, 11, 8, 12, 1)]
        );
    }

    // A component reaching the image edge keeps its full extent.
    let (w, h) = (16usize, 12usize);
    let mut ring = bm(w, h, 1);
    for x in 0..w {
        ring.set(x, 0, 1);
        ring.set(x, h - 1, 1);
    }
    for y in 0..h {
        ring.set(0, y, 1);
        ring.set(w - 1, y, 1);
    }
    for conn in [4, 8] {
        assert_eq!(
            conn_comp(&ring, conn, &lim(), &nc()).unwrap(),
            [comp(0, 0, w as isize, h as isize, 2 * w + 2 * (h - 2))]
        );
    }

    // Raster order of the first pixel.
    let mut b = bm(20, 20, 1);
    for (x, y) in [(15, 2), (1, 5), (9, 5), (3, 12)] {
        b.set(x, y, 1);
    }
    assert_eq!(
        conn_comp(&b, 8, &lim(), &nc()).unwrap(),
        [
            comp(15, 2, 16, 3, 1),
            comp(1, 5, 2, 6, 1),
            comp(9, 5, 10, 6, 1),
            comp(3, 12, 4, 13, 1)
        ]
    );

    // Arms that meet late.
    let mut b = bm(20, 20, 1);
    for y in 2..=10 {
        b.set(4, y, 1);
        b.set(10, y, 1);
    }
    for x in 4..=10 {
        b.set(x, 10, 1);
    }
    assert_eq!(
        conn_comp(&b, 4, &lim(), &nc()).unwrap(),
        [comp(4, 2, 11, 11, 2 * 8 + 7)]
    );
}

#[test]
fn conn_comp_rejects_bad_input_and_cancel() {
    assert!(conn_comp(&bm(4, 4, 1), 6, &lim(), &nc()).is_err());
    assert!(conn_comp(&bm(4, 4, 8), 8, &lim(), &nc()).is_err());
    let set = AtomicBool::new(true);
    assert_eq!(
        conn_comp(&bm(4, 4, 1), 8, &lim(), &set).unwrap_err(),
        Error::Cancelled
    );
    let tight = Limits {
        max_alloc: 10,
        ..lim()
    };
    assert_eq!(
        conn_comp(&bm(8, 8, 1), 8, &tight, &nc()).unwrap_err(),
        Error::AllocLimit
    );
}

fn ids(g: &Grid, r: Rect) -> Vec<i64> {
    g.query(r).unwrap()
}

fn grid(x0: isize, y0: isize, x1: isize, y1: isize, cell: usize) -> Grid {
    Grid::new(Rect::new(x0, y0, x1, y1), cell, &lim()).unwrap()
}

#[test]
fn grid_queries() {
    let mut g = grid(0, 0, 100, 100, 10);
    g.insert(1, Rect::new(0, 0, 20, 20)).unwrap();
    g.insert(2, Rect::new(15, 15, 25, 25)).unwrap();
    g.insert(3, Rect::new(60, 60, 70, 70)).unwrap();
    assert_eq!(ids(&g, Rect::new(0, 0, 5, 5)), [1]);
    assert_eq!(ids(&g, Rect::new(16, 16, 17, 17)), [1, 2]);
    assert_eq!(ids(&g, Rect::new(62, 62, 63, 63)), [3]);
    assert_eq!(ids(&g, Rect::new(0, 0, 100, 100)), [1, 2, 3]);
    assert!(ids(&g, Rect::new(30, 30, 40, 40)).is_empty());
    assert!(ids(&grid(0, 0, 100, 100, 10), Rect::new(0, 0, 100, 100)).is_empty());
}

#[test]
fn grid_is_exact_not_cell_level() {
    let mut g = grid(0, 0, 100, 100, 50);
    g.insert(1, Rect::new(0, 0, 5, 5)).unwrap();
    g.insert(2, Rect::new(40, 40, 45, 45)).unwrap();
    assert_eq!(ids(&g, Rect::new(0, 0, 6, 6)), [1]);
    assert_eq!(ids(&g, Rect::new(44, 44, 49, 49)), [2]);
}

#[test]
fn grid_spanning_and_abutting() {
    let mut g = grid(0, 0, 100, 100, 10);
    g.insert(7, Rect::new(5, 5, 95, 95)).unwrap();
    for y in (5..95).step_by(10) {
        for x in (5..95).step_by(10) {
            assert_eq!(ids(&g, Rect::new(x, y, x + 1, y + 1)), [7]);
        }
    }
    assert_eq!(ids(&g, Rect::new(0, 0, 100, 100)), [7]);

    let mut g = grid(0, 0, 100, 100, 10);
    g.insert(1, Rect::new(0, 0, 10, 10)).unwrap();
    assert!(ids(&g, Rect::new(10, 0, 20, 10)).is_empty());
    assert_eq!(ids(&g, Rect::new(9, 0, 19, 10)), [1]);
}

#[test]
fn grid_clamps_origin_partial_cells_empty_and_duplicates() {
    let mut g = grid(0, 0, 100, 100, 10);
    g.insert(1, Rect::new(-40, -40, -30, -30)).unwrap();
    g.insert(2, Rect::new(90, 90, 200, 200)).unwrap();
    assert_eq!(ids(&g, Rect::new(-45, -45, -35, -35)), [1]);
    assert_eq!(ids(&g, Rect::new(150, 150, 160, 160)), [2]);
    assert_eq!(ids(&g, Rect::new(95, 95, 96, 96)), [2]);
    assert!(ids(&g, Rect::new(20, 20, 30, 30)).is_empty());

    let mut g = grid(200, 100, 400, 300, 25);
    g.insert(1, Rect::new(210, 110, 230, 130)).unwrap();
    g.insert(2, Rect::new(380, 280, 395, 295)).unwrap();
    assert_eq!(ids(&g, Rect::new(220, 120, 221, 121)), [1]);
    assert_eq!(ids(&g, Rect::new(200, 100, 400, 300)), [1, 2]);
    assert!(ids(&g, Rect::new(300, 200, 310, 210)).is_empty());

    let mut g = grid(0, 0, 25, 25, 10);
    g.insert(1, Rect::new(23, 23, 25, 25)).unwrap();
    assert_eq!(ids(&g, Rect::new(24, 24, 25, 25)), [1]);
    assert_eq!(ids(&g, Rect::new(0, 0, 25, 25)), [1]);

    let mut g = grid(0, 0, 100, 100, 10);
    g.insert(1, Rect::new(10, 10, 10, 20)).unwrap();
    g.insert(2, Rect::new(30, 30, 40, 40)).unwrap();
    assert_eq!(ids(&g, Rect::new(0, 0, 100, 100)), [2]);
    assert!(ids(&g, Rect::new(35, 35, 35, 35)).is_empty());

    let mut g = grid(0, 0, 100, 100, 10);
    g.insert(5, Rect::new(0, 0, 10, 10)).unwrap();
    g.insert(5, Rect::new(80, 80, 90, 90)).unwrap();
    g.insert(2, Rect::new(40, 40, 50, 50)).unwrap();
    assert_eq!(ids(&g, Rect::new(0, 0, 100, 100)), [2, 5]);
}

#[test]
fn grid_rejects_bad_input_and_charges_memory() {
    assert!(Grid::new(Rect::new(0, 0, 10, 10), 0, &lim()).is_err());
    assert!(Grid::new(Rect::new(10, 10, 10, 10), 4, &lim()).is_err());
    let tight = Limits {
        max_alloc: 1000,
        ..lim()
    };
    assert_eq!(
        Grid::new(Rect::new(0, 0, 1 << 20, 1 << 20), 1, &tight).unwrap_err(),
        Error::AllocLimit
    );
    let mut g = Grid::new(
        Rect::new(0, 0, 100, 100),
        1,
        &Limits {
            max_alloc: 100 * 100 * 24 + 1000,
            ..lim()
        },
    )
    .unwrap();
    assert_eq!(
        g.insert(1, Rect::new(0, 0, 100, 100)).unwrap_err(),
        Error::AllocLimit
    );
}

// A query charges its flags and its result to the grid's budget, and gives them back.
#[test]
fn grid_query_is_charged_and_released() {
    use std::mem::size_of;
    // One cell and one box: exactly what `new` and `insert` charge.
    let held = size_of::<Vec<usize>>()
        + 4 * size_of::<usize>()
        + 4 * size_of::<i64>()
        + 4 * size_of::<Rect>();
    let at = |extra: usize| Limits {
        max_alloc: held + extra,
        ..lim()
    };
    let q = Rect::new(0, 0, 10, 10);
    let mut g = Grid::new(Rect::new(0, 0, 10, 10), 10, &at(0)).unwrap();
    g.insert(1, Rect::new(0, 0, 5, 5)).unwrap();
    assert_eq!(g.query(q), Err(Error::AllocLimit));
    // One flag byte and the room for four ids.
    let mut g = Grid::new(Rect::new(0, 0, 10, 10), 10, &at(1 + 4 * size_of::<i64>())).unwrap();
    g.insert(1, Rect::new(0, 0, 5, 5)).unwrap();
    for _ in 0..3 {
        assert_eq!(g.query(q).unwrap(), [1]);
    }
}

// One flood that covers the whole image must still see a cancel that arrives while it runs.
#[test]
fn a_flood_fill_checks_cancel_inside_one_component() {
    let mut b = bm(2048, 2048, 1);
    fill_rect(&mut b, 0, 0, 2048, 2048);
    let flood = |run: &(dyn Fn(&AtomicBool) -> Result<(), Error> + Sync)| {
        let cancel = AtomicBool::new(false);
        std::thread::scope(|s| {
            s.spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(5));
                cancel.store(true, std::sync::atomic::Ordering::Relaxed);
            });
            run(&cancel)
        })
    };
    assert_eq!(
        flood(&|c| conn_comp(&b, 8, &lim(), c).map(drop)),
        Err(Error::Cancelled)
    );
    assert_eq!(
        flood(&|c| seed_fill(&b, &b, 8, &lim(), c).map(drop)),
        Err(Error::Cancelled)
    );
}

#[test]
fn morphology_matches_leptonica() {
    let input = load_golden("otsu.bin");
    type F = fn(&Bitmap, usize, usize, &Limits, &AtomicBool) -> Result<Bitmap, Error>;
    let cases: [(&str, F, usize, usize, &str); 5] = [
        ("dilate", dilate as F, 5, 3, "dilate_5x3.bin"),
        ("dilate even", dilate as F, 4, 2, "dilate_4x2.bin"),
        ("erode", erode as F, 3, 7, "erode_3x7.bin"),
        ("open", open as F, 5, 5, "open_5x5.bin"),
        ("close", close as F, 7, 3, "close_7x3.bin"),
    ];
    for (name, f, w, h, golden) in cases {
        let got = f(&input, w, h, &lim(), &nc()).unwrap();
        same_pixels(name, &got, &load_golden(golden));
    }
}

#[test]
fn dilate_even_brick_is_off_centre() {
    let mut b = bm(12, 12, 1);
    b.set(5, 5, 1);
    let got = dilate(&b, 4, 2, &lim(), &nc()).unwrap();
    for y in 0..12 {
        for x in 0..12 {
            let want = u8::from((3..=6).contains(&x) && (4..=5).contains(&y));
            assert_eq!(got.at(x, y), want, "({x},{y})");
        }
    }
}

#[test]
fn erode_clears_border_under_asymmetric_bc() {
    let n = 20;
    let mut b = bm(n, n, 1);
    fill_rect(&mut b, 0, 0, n, n);
    let got = erode(&b, 3, 3, &lim(), &nc()).unwrap();
    for y in 0..n {
        for x in 0..n {
            let want = u8::from((1..=n - 2).contains(&x) && (1..=n - 2).contains(&y));
            assert_eq!(got.at(x, y), want, "({x},{y})");
        }
    }
}

#[test]
fn open_close_and_unit_brick() {
    let mut b = bm(40, 40, 1);
    fill_rect(&mut b, 10, 10, 9, 9);
    fill_rect(&mut b, 30, 30, 2, 2);
    let got = open(&b, 5, 5, &lim(), &nc()).unwrap();
    for y in 0..40 {
        for x in 0..40 {
            let want = u8::from((10..19).contains(&x) && (10..19).contains(&y));
            assert_eq!(got.at(x, y), want, "open ({x},{y})");
        }
    }

    let mut b = bm(40, 40, 1);
    fill_rect(&mut b, 10, 10, 5, 5);
    fill_rect(&mut b, 17, 10, 5, 5);
    let got = close(&b, 5, 1, &lim(), &nc()).unwrap();
    for y in 0..40 {
        for x in 0..40 {
            let want = u8::from((10..=21).contains(&x) && (10..15).contains(&y));
            assert_eq!(got.at(x, y), want, "close ({x},{y})");
        }
    }

    let mut b = bm(16, 16, 1);
    fill_rect(&mut b, 3, 4, 5, 6);
    for got in [
        dilate(&b, 1, 1, &lim(), &nc()).unwrap(),
        erode(&b, 1, 1, &lim(), &nc()).unwrap(),
        open(&b, 1, 1, &lim(), &nc()).unwrap(),
        close(&b, 1, 1, &lim(), &nc()).unwrap(),
    ] {
        assert_eq!(got, b);
    }
    assert!(dilate(&b, 0, 1, &lim(), &nc()).is_err());
    assert!(dilate(&bm(4, 4, 8), 1, 1, &lim(), &nc()).is_err());
}

#[test]
fn raster_op_matches_leptonica() {
    let rect = Rect::new(-40, -30, 360, 470);
    let origin = Point { x: 60, y: 80 };
    let cases = [
        (Op::Set, "rop_set.bin"),
        (Op::Clear, "rop_clr.bin"),
        (Op::Src, "rop_copy.bin"),
        (Op::NotSrc, "rop_notsrc.bin"),
        (Op::SrcOrDst, "rop_or.bin"),
        (Op::SrcAndDst, "rop_and.bin"),
        (Op::SrcXorDst, "rop_xor.bin"),
    ];
    for (op, golden) in cases {
        let mut dst = load_golden("rop_dst_in.bin");
        let src = load_golden("rop_src_in.bin");
        raster_op(&mut dst, rect, op, Some(&src), origin, &lim(), &nc()).unwrap();
        same_pixels(golden, &dst, &load_golden(golden));
    }

    let mut dst = load_golden("rop_dst_in.bin");
    let src = load_golden("rop_src_in.bin");
    raster_op(
        &mut dst,
        Rect::new(200, 250, 400, 550),
        Op::Src,
        Some(&src),
        Point { x: -60, y: -80 },
        &lim(),
        &nc(),
    )
    .unwrap();
    same_pixels(
        "negative source origin",
        &dst,
        &load_golden("rop_negsrc.bin"),
    );
}

#[test]
fn raster_op_disjoint_rect_is_a_no_op_and_errors() {
    let mut dst = bm(16, 16, 1);
    dst.set(3, 3, 1);
    let mut src = bm(16, 16, 1);
    fill_rect(&mut src, 0, 0, 16, 16);
    let before = dst.clone();
    let zero = Point::default();
    raster_op(
        &mut dst,
        Rect::new(40, 40, 60, 60),
        Op::Src,
        Some(&src),
        zero,
        &lim(),
        &nc(),
    )
    .unwrap();
    raster_op(
        &mut dst,
        Rect::new(-40, -40, -20, -20),
        Op::Set,
        None,
        zero,
        &lim(),
        &nc(),
    )
    .unwrap();
    assert_eq!(dst, before);
    assert!(
        raster_op(
            &mut dst,
            Rect::new(0, 0, 4, 4),
            Op::Src,
            None,
            zero,
            &lim(),
            &nc()
        )
        .is_err()
    );
    let g8 = bm(16, 16, 8);
    assert!(
        raster_op(
            &mut dst,
            Rect::new(0, 0, 4, 4),
            Op::Src,
            Some(&g8),
            zero,
            &lim(),
            &nc()
        )
        .is_err()
    );
}

#[test]
fn seed_fill_matches_leptonica() {
    for conn in [4, 8] {
        let mask = load_golden("seedfill_mask_in.bin");
        let seed = load_golden("seedfill_seed_in.bin");
        let got = seed_fill(&seed, &mask, conn, &lim(), &nc()).unwrap();
        same_pixels(
            "SeedFill",
            &got,
            &load_golden(&format!("seedfill{conn}.bin")),
        );

        let mask = load_golden("conncomp_diag_in.bin");
        let seed = load_golden("seedfill_diag_seed_in.bin");
        let got = seed_fill(&seed, &mask, conn, &lim(), &nc()).unwrap();
        same_pixels(
            "SeedFill diag",
            &got,
            &load_golden(&format!("seedfill_diag{conn}.bin")),
        );
    }
    let mask = load_golden("seedfill_mask_in.bin");
    let filled = count_fg(&load_golden("seedfill4.bin"));
    let total = count_fg(&mask);
    assert!(filled > 0 && filled < total, "{filled} of {total}");
}

#[test]
fn seed_fill_selects_whole_components() {
    let mut mask = bm(30, 30, 1);
    fill_rect(&mut mask, 3, 3, 6, 6);
    fill_rect(&mut mask, 20, 20, 4, 4);
    let mut seed = bm(30, 30, 1);
    seed.set(5, 5, 1);
    seed.set(15, 15, 1);
    let (m0, s0) = (mask.clone(), seed.clone());
    let got = seed_fill(&seed, &mask, 8, &lim(), &nc()).unwrap();
    for y in 0..30 {
        for x in 0..30 {
            let want = u8::from((3..9).contains(&x) && (3..9).contains(&y));
            assert_eq!(got.at(x, y), want, "({x},{y})");
        }
    }
    assert_eq!((mask.clone(), seed), (m0, s0));

    let empty = seed_fill(&bm(30, 30, 1), &mask, 4, &lim(), &nc()).unwrap();
    assert_eq!(count_fg(&empty), 0);
}

#[test]
fn seed_fill_rejects_bad_input() {
    let l = lim();
    let c = nc();
    assert!(seed_fill(&bm(4, 4, 1), &bm(4, 4, 1), 6, &l, &c).is_err());
    assert!(seed_fill(&bm(4, 4, 8), &bm(4, 4, 1), 4, &l, &c).is_err());
    assert!(seed_fill(&bm(4, 4, 1), &bm(4, 4, 8), 4, &l, &c).is_err());
    assert!(seed_fill(&bm(4, 4, 1), &bm(5, 4, 1), 4, &l, &c).is_err());
}

#[test]
fn distance_function_matches_leptonica_and_hand_cases() {
    let got = distance_function(&load_golden("dist_in.bin"), &lim(), &nc()).unwrap();
    assert_eq!(got.depth, 8);
    same_pixels("DistanceFunction", &got, &load_golden("dist4.bin"));

    let (w, h) = (24usize, 26usize);
    let mut b = bm(w, h, 1);
    fill_rect(&mut b, 4, 6, 10, 9);
    let got = distance_function(&b, &lim(), &nc()).unwrap();
    for y in 0..h {
        for x in 0..w {
            let mut want = 0;
            if (4..14).contains(&x) && (6..15).contains(&y) {
                want = 1 + (x - 4).min(13 - x).min((y - 6).min(14 - y));
            }
            assert_eq!(usize::from(got.at(x, y)), want, "({x},{y})");
        }
    }

    let n = 21;
    let mut b = bm(n, n, 1);
    fill_rect(&mut b, 0, 0, n, n);
    let got = distance_function(&b, &lim(), &nc()).unwrap();
    for y in 0..n {
        for x in 0..n {
            let want = 1 + x.min(n - 1 - x).min(y.min(n - 1 - y));
            assert_eq!(usize::from(got.at(x, y)), want, "({x},{y})");
        }
    }

    let n = 600;
    let mut b = bm(n, n, 1);
    fill_rect(&mut b, 0, 0, n, n);
    let got = distance_function(&b, &lim(), &nc()).unwrap();
    for y in 254..=345 {
        for x in 254..=345 {
            assert_eq!(got.at(x, y), 255, "({x},{y})");
        }
    }
    assert!(distance_function(&bm(4, 4, 8), &lim(), &nc()).is_err());
}

fn degrees(radians: f64) -> f64 {
    radians * 180.0 / std::f64::consts::PI
}

#[test]
fn rotate_and_skew_match_leptonica() {
    let meta = load_floats("deskew_meta.bin", 5);
    let straight = load_golden("otsu.bin");
    let want = load_golden("deskew_rot.bin");
    let got = rotate(&straight, f64::from(meta[0]), &lim(), &nc()).unwrap();
    same_pixels("Rotate", &got, &want);

    let got = degrees(skew_angle(&want, &lim(), &nc()).unwrap());
    let leptonica = -f64::from(meta[1]);
    assert!((got - leptonica).abs() <= 0.1, "{got} vs {leptonica}");

    let got = degrees(skew_angle(&straight, &lim(), &nc()).unwrap());
    let leptonica = -f64::from(meta[3]);
    assert!((got - leptonica).abs() <= 0.1, "{got} vs {leptonica}");
    assert!(got.abs() <= 0.05, "{got}");
}

#[test]
fn skew_angle_recovers_a_known_rotation() {
    let straight = load_golden("otsu.bin");
    for want in [-4.5, -2.5, -1.23, 0.7, 2.5, 3.87] {
        let rot = rotate(
            &straight,
            want * std::f64::consts::PI / 180.0,
            &lim(),
            &nc(),
        )
        .unwrap();
        let got = degrees(skew_angle(&rot, &lim(), &nc()).unwrap());
        assert!((got - want).abs() <= 0.05, "{got} vs {want}");
    }
}

#[test]
fn rotate_behaviour() {
    let (w, h) = (200usize, 120usize);
    let mut b = bm(w, h, 1);
    fill_rect(&mut b, 0, 60, w, 1);
    let got = rotate(&b, 5.0 * std::f64::consts::PI / 180.0, &lim(), &nc()).unwrap();
    let col = |x: usize| (0..h).find(|&y| got.at(x, y) != 0);
    let (left, right) = (col(20).unwrap(), col(w - 20).unwrap());
    assert!(right > left, "left {left}, right {right}");

    let mut b = bm(37, 29, 1);
    fill_rect(&mut b, 3, 5, 11, 7);
    b.set(36, 28, 1);
    assert_eq!(rotate(&b, 0.0, &lim(), &nc()).unwrap(), b);

    let n = 101;
    let mut b = bm(n, n, 1);
    fill_rect(&mut b, 0, 0, n, n);
    let got = rotate(&b, 20.0 * std::f64::consts::PI / 180.0, &lim(), &nc()).unwrap();
    assert_eq!((got.width, got.height), (n, n));
    for (x, y) in [(0, 0), (n - 1, 0), (0, n - 1), (n - 1, n - 1)] {
        assert_eq!(got.at(x, y), 0, "corner ({x},{y})");
    }
    assert_eq!(got.at(n / 2, n / 2), 1);

    assert_eq!(skew_angle(&bm(50, 50, 1), &lim(), &nc()).unwrap(), 0.0);
    assert!(rotate(&bm(4, 4, 8), 0.1, &lim(), &nc()).is_err());
    assert!(skew_angle(&bm(4, 4, 8), &lim(), &nc()).is_err());
    let set = AtomicBool::new(true);
    assert_eq!(rotate(&b, 0.1, &lim(), &set).unwrap_err(), Error::Cancelled);
}

/// `rotate` against Go's loop written out pixel by pixel, on random bitmaps with padded strides
/// whose padding bits are set, so the byte-at-a-time copy must not read past the width.
#[test]
fn rotate_matches_go_loop() {
    use super::gomath::{cos, go_int, sin};
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s as usize
    };
    let angles = [
        0.0,
        0.02,
        -0.02,
        0.05,
        1e-9,
        0.3,
        -1.1,
        std::f64::consts::FRAC_PI_2,
        3.0,
        -2.5,
        1e6,
        f64::NAN,
        f64::INFINITY,
    ];
    for case in 0..80 {
        let (w, h) = (1 + next() % 200, 1 + next() % 90);
        let stride = w.div_ceil(8) + next() % 3;
        let pix = (0..stride * h).map(|_| next() as u8).collect();
        let b = Bitmap {
            width: w,
            height: h,
            depth: 1,
            stride,
            pix,
        };
        let angle = angles[case % angles.len()];
        let got = rotate(&b, angle, &lim(), &nc()).unwrap();
        let a = angle as f32;
        let sina = f64::from(sin(f64::from(a)) as f32);
        let cosa = f64::from(cos(f64::from(a)) as f32);
        let (xcen, ycen) = ((w / 2) as i64, (h / 2) as i64);
        let mut want = bm(w, h, 1);
        for i in 0..h {
            for j in 0..w {
                let (xdif, ydif) = ((xcen - j as i64) as f64, (ycen - i as i64) as f64);
                let x = xcen.wrapping_add(go_int(f64::from((-xdif * cosa - ydif * sina) as f32)));
                let y = ycen.wrapping_add(go_int(f64::from((-ydif * cosa + xdif * sina) as f32)));
                if (0..w as i64).contains(&x) && (0..h as i64).contains(&y) {
                    want.set(j, i, b.at(x as usize, y as usize));
                }
            }
        }
        assert_eq!(
            got, want,
            "case {case}: angle {angle}, {w}x{h}, stride {stride}"
        );
    }
}

#[test]
fn go_trig_matches_libm_closely() {
    use super::gomath::{cos, sin, tan};
    for i in -2000..2000 {
        let x = f64::from(i) * 0.013;
        assert!((sin(x) - x.sin()).abs() < 1e-15, "sin {x}");
        assert!((cos(x) - x.cos()).abs() < 1e-15, "cos {x}");
        if x.cos().abs() > 0.1 {
            assert!(
                (tan(x) - x.tan()).abs() < 1e-12 * (1.0 + x.tan().abs()),
                "tan {x}"
            );
        }
    }
    // Payne-Hanek range reduction, above 2^29.
    for x in [1e9, 1.7e12, 3.3e20, 1e100] {
        assert!((sin(x) - x.sin()).abs() < 1e-12, "sin {x}");
        assert!((cos(x) - x.cos()).abs() < 1e-12, "cos {x}");
    }
    assert!(sin(f64::INFINITY).is_nan());
}

#[test]
fn scale_gray_matches_leptonica() {
    let src = load_golden("gray.bin");
    let up = load_golden("scale_up_in.bin");
    let cases: [(&str, &Bitmap, f64); 7] = [
        ("scale_identity", &src, 1.0),
        ("scale_li", &src, 0.80),
        ("scale_areamap2", &src, 0.50),
        ("scale_areamap", &src, 0.35),
        ("scale_smooth", &src, 0.015),
        ("scale_2xli", &up, 2.0),
        ("scale_4xli", &up, 4.0),
    ];
    for (name, s, factor) in cases {
        let got = scale_gray(s, factor, &lim(), &nc()).unwrap();
        let want = load_golden(&format!("{name}.bin"));
        same_pixels(name, &got, &want);
    }
    assert_eq!(scale_gray(&src, 1.0, &lim(), &nc()).unwrap(), src);
}

#[test]
fn scale_gray_rejects_bad_input() {
    let b = bm(8, 8, 8);
    for f in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e39, f64::MAX] {
        assert!(scale_gray(&b, f, &lim(), &nc()).is_err(), "{f}");
    }
    assert!(scale_gray(&bm(0, 8, 8), 0.5, &lim(), &nc()).is_err());
    assert!(scale_gray(&bm(8, 8, 1), 0.5, &lim(), &nc()).is_err());
    assert_eq!(
        scale_gray(&b, 1e9, &lim(), &nc()).unwrap_err(),
        Error::PixelLimit
    );
    // Tiny sources and every kernel must not panic.
    for (w, h) in [(1, 1), (1, 5), (5, 1), (2, 2), (3, 7)] {
        let s = bm(w, h, 8);
        for f in [
            0.015, 0.2, 0.25, 0.35, 0.5, 0.69, 0.7, 0.9, 1.3, 1.5, 2.0, 4.0, 3.3,
        ] {
            let _ = scale_gray(&s, f, &lim(), &nc());
        }
    }
}

#[test]
fn pgm_reader() {
    let l = lim();
    let b = Bitmap::from_pgm(b"P5\n# comment\n3 2\n255\nabcdef", &l).unwrap();
    assert_eq!((b.width, b.height), (3, 2));
    assert_eq!(b.pix, b"abcdef");
    assert!(Bitmap::from_pgm(b"P6\n1 1\n255\nabc", &l).is_err());
    assert!(Bitmap::from_pgm(b"P5\n3 2\n255\nabc", &l).is_err());
    assert!(Bitmap::from_pgm(b"P5\n3 2\n65535\nabcdef", &l).is_err());
    assert!(Bitmap::from_pgm(b"P5\n99999999999 99999999999\n255\n", &l).is_err());
    assert!(Bitmap::from_pgm(b"P5\nx", &l).is_err());
}

#[test]
fn grid_and_raster_op_survive_coordinates_at_the_edge_of_isize() {
    let mut g = Grid::new(Rect::new(1, 1, 100, 100), 10, &lim()).unwrap();
    g.insert(7, Rect::new(1, 1, 5, 5)).unwrap();
    assert_eq!(g.query(Rect::new(isize::MIN, 0, 5, 5)).unwrap(), vec![7]);
    g.insert(8, Rect::new(isize::MIN, 0, 5, 5)).unwrap();
    assert_eq!(
        g.query(Rect::new(0, isize::MIN, isize::MAX, isize::MAX))
            .unwrap(),
        vec![7, 8]
    );

    let mut d = bm(10, 10, 1);
    let s = bm(10, 10, 1);
    let far = Point {
        x: isize::MAX,
        y: 0,
    };
    raster_op(
        &mut d,
        Rect::new(-5, 0, 5, 1),
        Op::Src,
        Some(&s),
        far,
        &lim(),
        &nc(),
    )
    .unwrap();
    let min = Point {
        x: isize::MIN,
        y: isize::MIN,
    };
    let all = Rect::new(isize::MIN, isize::MIN, isize::MAX, isize::MAX);
    raster_op(&mut d, all, Op::Src, Some(&s), min, &lim(), &nc()).unwrap();
    raster_op(&mut d, all, Op::Set, None, min, &lim(), &nc()).unwrap();
}

#[test]
fn operators_reject_a_bitmap_whose_public_fields_disagree() {
    let empty = |depth| Bitmap {
        width: 4,
        height: 4,
        depth,
        stride: 4,
        pix: Vec::new(),
    };
    let (g, b) = (empty(8), empty(1));
    let short_rows = Bitmap {
        width: 100,
        height: 1,
        depth: 8,
        stride: 10,
        pix: vec![0; 10],
    };
    let (l, c) = (lim(), nc());
    for r in [
        otsu_threshold(&g, &lim(), &c).map(|_| ()),
        otsu_threshold(&short_rows, &lim(), &c).map(|_| ()),
        otsu(&g, &l, &c).map(|_| ()),
        sauvola(&g, 5, 0.2, &l, &c).map(|_| ()),
        scale_gray(&g, 0.5, &l, &c).map(|_| ()),
        skew_angle(&b, &l, &c).map(|_| ()),
        rotate(&b, 0.1, &l, &c).map(|_| ()),
        dilate(&b, 3, 3, &l, &c).map(|_| ()),
        conn_comp(&b, 4, &l, &c).map(|_| ()),
        distance_function(&b, &l, &c).map(|_| ()),
        seed_fill(&b, &bm(4, 4, 1), 4, &l, &c).map(|_| ()),
        seed_fill(&bm(4, 4, 1), &b, 4, &l, &c).map(|_| ()),
        raster_op(
            &mut b.clone(),
            b.bounds(),
            Op::Set,
            None,
            Point::default(),
            &lim(),
            &nc(),
        ),
        raster_op(
            &mut bm(4, 4, 1),
            b.bounds(),
            Op::Src,
            Some(&b),
            Point::default(),
            &lim(),
            &nc(),
        ),
    ] {
        assert!(matches!(r, Err(Error::Imaging(_))), "{r:?}");
    }
}

#[test]
fn otsu_threshold_checks_the_cancel_flag() {
    let set = AtomicBool::new(true);
    assert_eq!(
        otsu_threshold(&bm(4, 4, 8), &lim(), &set),
        Err(Error::Cancelled)
    );
}

#[test]
fn brick_operations_charge_every_bitmap_they_hold_at_once() {
    let b = bm(1024, 1024, 1);
    let n = b.pix.len();
    // One thread: the workers of a threaded call stay charged to its end, on top of the bitmaps.
    let cap = |k: usize| Limits {
        max_alloc: k * n,
        max_threads: 1,
        ..lim()
    };
    // Two passes hold the first result while they make the second: one bitmap's budget is not
    // enough, two is.
    assert_eq!(dilate(&b, 3, 3, &cap(1), &nc()), Err(Error::AllocLimit));
    assert_eq!(erode(&b, 3, 3, &cap(1), &nc()), Err(Error::AllocLimit));
    assert!(dilate(&b, 3, 3, &cap(2), &nc()).is_ok());
    assert!(erode(&b, 3, 3, &cap(2), &nc()).is_ok());
    // Open and close also hold their first result: three.
    assert_eq!(open(&b, 3, 3, &cap(2), &nc()), Err(Error::AllocLimit));
    assert_eq!(close(&b, 3, 3, &cap(2), &nc()), Err(Error::AllocLimit));
    assert!(open(&b, 3, 3, &cap(3), &nc()).is_ok());
    assert!(close(&b, 3, 3, &cap(3), &nc()).is_ok());
}

/// Go converts an out-of-range float with `int()`, which gives MinInt64 on amd64; the crate
/// follows it, so each of these differs from what a saturating `as` cast would do.
#[test]
fn scale_smooth_with_a_tiny_factor_takes_a_filter_of_two() {
    let mut src = bm(4, 4, 8);
    src.set(2, 2, 200);
    // 1e-20 is 1/scale > 2^63; 1e-50 underflows f32 to 0 (1/0 = +Inf).
    for f in [1e-20, 1e-50] {
        let got = scale_gray(&src, f, &lim(), &nc()).unwrap();
        assert_eq!((got.width, got.height, got.at(0, 0)), (1, 1, 0), "{f}");
    }
}

#[test]
fn sauvola_with_a_huge_negative_k_clears_every_pixel() {
    let g = Bitmap::from_gray8(5, 5, &[100; 25], &lim()).unwrap();
    let got = sauvola(&g, 5, -1e20, &lim(), &nc()).unwrap();
    assert!((0..5).all(|y| (0..5).all(|x| got.at(x, y) == 0)));
}

#[test]
fn rotate_by_a_non_finite_angle_is_empty() {
    let mut b = bm(3, 3, 1);
    b.set(1, 1, 1);
    for a in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e39] {
        let got = rotate(&b, a, &lim(), &nc()).unwrap();
        assert!((0..3).all(|y| (0..3).all(|x| got.at(x, y) == 0)), "{a}");
    }
}

#[test]
fn rotate_reads_a_padded_stride() {
    let padded = Bitmap {
        width: 8,
        height: 2,
        depth: 1,
        stride: 3,
        pix: vec![0x80, 0, 0, 0x01, 0, 0],
    };
    let mut plain = bm(8, 2, 1);
    plain.set(0, 0, 1);
    plain.set(7, 1, 1);
    for a in [0.0, 0.3] {
        let (p, q) = (
            rotate(&padded, a, &lim(), &nc()).unwrap(),
            rotate(&plain, a, &lim(), &nc()).unwrap(),
        );
        assert_eq!(p, q, "{a}");
    }
}

#[test]
fn a_plane_whose_row_bits_overflow_is_a_pixel_limit() {
    let l = Limits {
        max_width: usize::MAX,
        ..lim()
    };
    assert_eq!(Bitmap::new(1 << 61, 0, 8, &l), Err(Error::PixelLimit));
}

#[test]
fn otsu_threshold_checks_the_plane_against_the_limits() {
    let small = Limits {
        max_pixels: 99,
        ..lim()
    };
    assert_eq!(
        otsu_threshold(&bm(10, 10, 8), &small, &nc()),
        Err(Error::PixelLimit)
    );
}

// Threading: the output is the serial output for every thread count, a call never uses more
// threads than `max_threads`, and the cancel flag stops all of them.

/// A noisy `w x h` plane (a quarter of the bits set at depth 1), from a xorshift.
fn noise(w: usize, h: usize, depth: usize) -> Bitmap {
    let mut b = bm(w, h, depth);
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    for p in b.pix.iter_mut() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *p = if depth == 1 {
            ((x >> 24) as u8) & ((x >> 40) as u8)
        } else {
            (x >> 32) as u8
        };
    }
    b
}

fn budget_of(max_threads: usize) -> Budget {
    Budget::new(&Limits {
        max_threads,
        ..lim()
    })
}

fn thread_counts() -> [usize; 4] {
    [1, 2, 3, Limits::default().max_threads]
}

fn same_output_for_every_thread_count<T: PartialEq + std::fmt::Debug>(
    name: &str,
    f: impl Fn(&Budget, &AtomicBool) -> Result<T, Error>,
) {
    let serial = budget_of(1);
    let want = f(&serial, &nc()).unwrap();
    assert_eq!(
        serial.peak_threads(),
        1,
        "{name}: one thread spawns nothing"
    );
    for n in thread_counts() {
        let b = budget_of(n);
        assert_eq!(f(&b, &nc()).unwrap(), want, "{name}, {n} threads");
        assert!(b.peak_threads() <= n, "{name}, {n} threads");
        if n >= 2 {
            assert!(b.peak_threads() >= 2, "{name}: {n} threads were not used");
        }
        // The cancel flag stops every thread, and they all go back to the budget.
        let stop = AtomicBool::new(true);
        let b = budget_of(n);
        assert_eq!(f(&b, &stop).err(), Some(Error::Cancelled), "{name}, {n}");
        assert_eq!(b.acquire_threads(100), n - 1, "{name}, {n}");
    }
    // The same bytes on a caller's pool and on workers the call builds for itself.
    let caller = crate::ThreadPool::new(2).unwrap();
    for n in [2, 3] {
        for (pool, per_call_pool) in [(Some(caller.clone()), false), (None, true)] {
            let b = Budget::new(&Limits {
                max_threads: n,
                pool,
                per_call_pool,
                ..lim()
            });
            assert_eq!(f(&b, &nc()).unwrap(), want, "{name}, {n} threads");
            assert!((2..=n).contains(&b.peak_threads()), "{name}, {n} threads");
        }
    }
}

#[test]
fn binarization_is_the_same_for_every_thread_count() {
    let g = noise(800, 700, 8);
    same_output_for_every_thread_count("otsu_threshold", |b, c| {
        binarize::otsu_threshold_in(&g, b, c)
    });
    same_output_for_every_thread_count("otsu", |b, c| {
        let out = binarize::otsu_in(&g, b, c)?;
        Ok(out.pix.clone())
    });
    same_output_for_every_thread_count("sauvola", |b, c| {
        let out = binarize::sauvola_in(&g, 15, 0.34, b, c)?;
        Ok(out.pix.clone())
    });
}

#[test]
fn deskew_and_rotate_are_the_same_for_every_thread_count() {
    let bin = noise(640, 480, 1);
    same_output_for_every_thread_count("skew_angle", |b, c| deskew::skew_angle_in(&bin, b, c));
    same_output_for_every_thread_count("rotate", |b, c| {
        Ok(deskew::rotate_in(&bin, 0.02, b, c)?.pix.clone())
    });
}

#[test]
fn morphology_is_the_same_for_every_thread_count() {
    let bin = noise(640, 480, 1);
    for (w, h) in [(3, 3), (1, 7), (7, 1), (5, 4)] {
        same_output_for_every_thread_count(&format!("dilate {w}x{h}"), |b, c| {
            Ok(ops::dilate_in(&bin, w, h, c, b)?.pix.clone())
        });
        same_output_for_every_thread_count(&format!("erode {w}x{h}"), |b, c| {
            Ok(ops::erode_in(&bin, w, h, c, b)?.pix.clone())
        });
    }
}

#[test]
fn scaling_is_the_same_for_every_thread_count() {
    let g = noise(800, 700, 8);
    // Area map, smooth and linear interpolation, each with and without sharpening.
    for f in [0.5, 0.69, 0.01, 1.7, 1.2] {
        same_output_for_every_thread_count(&format!("scale {f}"), |b, c| {
            Ok(scale_gray_in(&g, f, b, c)?.pix.clone())
        });
    }
    let small = noise(400, 300, 8);
    for f in [2.0, 4.0] {
        same_output_for_every_thread_count(&format!("scale {f}"), |b, c| {
            Ok(scale_gray_in(&small, f, b, c)?.pix.clone())
        });
    }
}

#[test]
fn a_small_plane_spawns_no_thread() {
    let b = budget_of(8);
    let g = noise(64, 64, 8);
    binarize::sauvola_in(&g, 15, 0.34, &b, &nc()).unwrap();
    scale_gray_in(&g, 1.7, &b, &nc()).unwrap();
    assert_eq!(b.peak_threads(), 1);
}

fn budget_with(max_alloc: usize, max_threads: usize) -> Budget {
    Budget::new(&Limits {
        max_alloc,
        max_threads,
        ..lim()
    })
}

/// Whether a call fits a given `max_alloc` hardly depends on `max_threads`: where one thread
/// works, any `max_threads` works given the 4096 bytes each extra thread stays charged until the
/// call ends, with fewer threads (the per-band scratch is cut to the budget), and once the budget
/// has room for threads they are used.
fn check_budget_for_one_thread_serves_every_max_threads<T: PartialEq + std::fmt::Debug>(
    name: &str,
    f: impl Fn(&Budget, &AtomicBool) -> Result<T, Error>,
) {
    let (mut lo, mut hi) = (0usize, 64usize << 20);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        match f(&budget_with(mid, 1), &nc()) {
            Ok(_) => hi = mid,
            Err(Error::AllocLimit) => lo = mid,
            Err(e) => panic!("{name}: max_alloc {mid} gave {e}"),
        }
    }
    let want = f(&budget_with(hi, 1), &nc()).unwrap();
    for n in [2, 3, 4, 8] {
        let b = budget_with(hi + (n - 1) * crate::par::THREAD_BYTES, n);
        let got = f(&b, &nc());
        assert_eq!(
            got.as_ref(),
            Ok(&want),
            "{name}: max_alloc {hi} and {n} threads' bytes, {n} threads"
        );
        assert!(b.peak_threads() <= n, "{name}: {n} threads");
        let b = budget_with(hi + n * (1 << 20), n);
        assert_eq!(f(&b, &nc()).as_ref(), Ok(&want), "{name}, {n} threads");
        assert!(b.peak_threads() >= 2, "{name}: {n} threads were not used");
    }
}

#[test]
fn a_budget_for_one_thread_serves_every_max_threads() {
    let g = noise(800, 700, 8);
    check_budget_for_one_thread_serves_every_max_threads("otsu_threshold", |b, c| {
        binarize::otsu_threshold_in(&g, b, c)
    });
    let bin = noise(640, 480, 1);
    check_budget_for_one_thread_serves_every_max_threads("skew_angle", |b, c| {
        deskew::skew_angle_in(&bin, b, c).map(f64::to_bits)
    });
}

/// Go's `Rotate`, pixel by pixel as `deskew.go` writes it.
fn rotate_reference(b: &Bitmap, radians: f64) -> Bitmap {
    use super::gomath::{cos, go_int, sin};
    let angle = radians as f32;
    let sina = f64::from(sin(f64::from(angle)) as f32);
    let cosa = f64::from(cos(f64::from(angle)) as f32);
    let mut out = bm(b.width, b.height, 1);
    let (xcen, ycen) = ((b.width / 2) as isize, (b.height / 2) as isize);
    for i in 0..b.height {
        let ydif = (ycen - i as isize) as f64;
        for j in 0..b.width {
            let xdif = (xcen - j as isize) as f64;
            let x =
                xcen.wrapping_add(go_int(f64::from((-xdif * cosa - ydif * sina) as f32)) as isize);
            if x < 0 || x >= b.width as isize {
                continue;
            }
            let y =
                ycen.wrapping_add(go_int(f64::from((-ydif * cosa + xdif * sina) as f32)) as isize);
            if y < 0 || y >= b.height as isize {
                continue;
            }
            if b.at(x as usize, y as usize) != 0 {
                out.set(j, i, 1);
            }
        }
    }
    out
}

#[test]
fn rotate_matches_the_pixel_by_pixel_reference() {
    for (w, h) in [
        (100, 77),
        (64, 64),
        (1, 1),
        (9, 3),
        (3, 40),
        (17, 5),
        (130, 90),
    ] {
        let mut b = bm(w, h, 1);
        for y in 0..h {
            for x in 0..w {
                if (x * 7 + y * 13 + x * y) % 5 < 2 {
                    b.set(x, y, 1);
                }
            }
        }
        for radians in [
            0.0,
            0.02,
            -0.02,
            0.5,
            1.5,
            std::f64::consts::FRAC_PI_2,
            -std::f64::consts::FRAC_PI_2,
            std::f64::consts::PI,
            -2.0,
            1e300,
            f64::INFINITY,
            f64::NAN,
        ] {
            let got = rotate(&b, radians, &lim(), &nc()).unwrap();
            assert_eq!(got, rotate_reference(&b, radians), "{w}x{h} at {radians}");
        }
    }
}
