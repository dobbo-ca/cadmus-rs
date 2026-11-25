//! Skew detection and nearest-neighbour rotation (`deskew.go`).
//!
//! Angles are in radians, clockwise positive in a coordinate system whose y axis increases
//! downward, as in Leptonica. The search bounds are in degrees.

use super::gomath::{cos, sin, tan};
use super::{Bitmap, check_cancel, need_depth};
use crate::par;
use crate::{Budget, Error, Limits};
use std::f64::consts::PI;
use std::sync::atomic::AtomicBool;

const SKEW_SWEEP_RANGE_DEG: f64 = 5.0;
const SKEW_SWEEP_STEP_DEG: f64 = 0.1;
const SKEW_REFINE_RANGE_DEG: f64 = 0.1;
const SKEW_REFINE_STEP_DEG: f64 = 0.01;
/// `(skewSweepRangeDeg+skewRefineRangeDeg)*math.Pi/180`, which Go folds as an exact constant.
const MAX_SWEEP_RAD: f64 = 0.0890117918517108;

/// Estimates the skew of `b`, in radians: the angle by which its text lines are tilted away from
/// horizontal, clockwise positive. `rotate(b, -skew_angle(b))` straightens the page. (Leptonica's
/// `pixFindSkew` returns the opposite quantity.)
///
/// The estimate maximises the variance of the horizontal projection profile over +-5 degrees in
/// 0.1 degree steps, refined over +-0.1 degrees in 0.01 degree steps. An empty bitmap has no
/// skew and returns 0. `b` must be depth 1.
///
/// Enforces `limits.max_alloc` (every bitmap this call holds is charged), `limits.max_threads` and
/// the plane caps (`max_pixels`, `max_width`, `max_height`).
///
/// # Errors
///
/// `Error::Imaging` for an argument that does not fit (not depth 1), `Error::PixelLimit` over a
/// plane cap, `Error::AllocLimit` over `max_alloc`, `Error::OutOfMemory` if the allocator refuses,
/// `Error::Cancelled` if `cancel` is set (checked per row).
pub fn skew_angle(b: &Bitmap, limits: &Limits, cancel: &AtomicBool) -> Result<f64, Error> {
    skew_angle_in(b, &Budget::new(limits), cancel)
}

/// `skew_angle` on the call's own `budget`. The candidate angles of a sweep are independent, so
/// each thread scores a run of them with its own scratch buffers; the best is then picked in
/// angle order, as the serial loop does.
pub(crate) fn skew_angle_in(
    b: &Bitmap,
    budget: &Budget,
    cancel: &AtomicBool,
) -> Result<f64, Error> {
    need_depth(b, 1, "SkewAngle")?;
    budget.limits().check_plane(b.width, b.height)?;

    // Collect the foreground once, so every candidate angle costs a pass over the ink.
    let mut count = 0usize;
    for y in 0..b.height {
        check_cancel(cancel)?;
        for x in 0..b.width {
            if b.at(x, y) != 0 {
                count += 1;
            }
        }
    }
    if count == 0 {
        return Ok(0.0);
    }
    let _ink_hold = budget.hold(count.checked_mul(8).ok_or(Error::AllocLimit)?)?;
    let mut ink: Vec<(i32, i32)> = Vec::new();
    ink.try_reserve_exact(count)
        .map_err(|_| Error::OutOfMemory)?;
    for y in 0..b.height {
        check_cancel(cancel)?;
        for x in 0..b.width {
            if b.at(x, y) != 0 {
                ink.push((x as i32, y as i32));
            }
        }
    }

    // The profile is padded by the largest shift any candidate can produce, so no ink is ever
    // dropped off the ends.
    let pad = (b.width as f64 / 2.0 * tan(MAX_SWEEP_RAD)) as usize + 1;
    // The refinement sweep has the fewest angles, so no more bands than that are ever used.
    let refine_angles = (2.0 * SKEW_REFINE_RANGE_DEG / SKEW_REFINE_STEP_DEG).round() as usize + 1;
    let plen = b.height + 2 * pad + 1;
    // Each band has its own profile, offsets and fractions, and the sweep holds its scores and
    // one `Band` per band while they are live; the band count is cut to what the budget affords.
    type Band<'a> = (
        usize,
        &'a mut [f64],
        &'a mut [f64],
        &'a mut [usize],
        &'a mut [f64],
    );
    let sweep_angles = (2.0 * SKEW_SWEEP_RANGE_DEG / SKEW_SWEEP_STEP_DEG).round() as usize + 1;
    let band_bytes = plen
        .saturating_add(2 * b.width)
        .saturating_mul(8)
        .saturating_add(std::mem::size_of::<Band>());
    let bands = par::band_count(
        budget,
        ink.len().saturating_mul(refine_angles),
        sweep_angles * std::mem::size_of::<f64>(),
        band_bytes,
    )
    .min(refine_angles);
    let mut profile = budget.buf(bands * plen, 0f64)?;
    let mut lo = budget.buf(bands * b.width, 0usize)?;
    let mut frac = budget.buf(bands * b.width, 0f64)?;
    let xc = b.width as f64 / 2.0;

    let score =
        |deg: f64, profile: &mut [f64], lo: &mut [usize], frac: &mut [f64]| -> Result<f64, Error> {
            check_cancel(cancel)?;
            let t = tan(deg * PI / 180.0);
            for x in 0..b.width {
                let shift = (x as f64 - xc) * t;
                let floor = shift.floor();
                lo[x] = (floor as isize + pad as isize) as usize;
                frac[x] = shift - floor;
            }
            profile.fill(0.0);
            // Each bin gets its additions in ink order, as before; the two bins the ink is
            // adding to are kept in registers until the ink moves on to other bins.
            let (mut cur, mut p0, mut p1) = (0usize, profile[0], profile[1]);
            for &(px, py) in &ink {
                let i = py as usize + lo[px as usize];
                let f = frac[px as usize];
                if i != cur {
                    profile[cur] = p0;
                    profile[cur + 1] = p1;
                    (cur, p0, p1) = (i, profile[i], profile[i + 1]);
                }
                p0 += 1.0 - f;
                p1 += f;
            }
            profile[cur] = p0;
            profile[cur + 1] = p1;
            let mut sum = 0.0f64;
            for n in profile.iter() {
                sum += n * n;
            }
            Ok(sum)
        };

    // The angle being searched for is the one that straightens the page; the skew the page
    // carries is its negative, which is what gets returned.
    let mut best = (0.0f64, f64::NEG_INFINITY);
    let mut sweep = |centre: f64, radius: f64, step: f64, best: &mut (f64, f64)| {
        let count = (2.0 * radius / step).round() as usize + 1;
        let deg = |i: usize| centre - radius + i as f64 * step;
        let mut scores = budget.buf(count, 0f64)?;
        {
            // One run of angles per band, with the band's own scratch buffers.
            let per_band = count.div_ceil(bands);
            let _bands_hold = budget.hold(bands * std::mem::size_of::<Band>())?;
            let mut runs: Vec<Band> = scores
                .chunks_mut(per_band)
                .zip(profile.chunks_mut(plen))
                .zip(lo.chunks_mut(b.width))
                .zip(frac.chunks_mut(b.width))
                .enumerate()
                .map(|(k, (((s, p), l), f))| (k * per_band, s, p, l, f))
                .collect();
            par::for_each_chunk_mut(budget, cancel, &mut runs[..], 1, |_, run| {
                let (first, out, p, l, f) = &mut run[0];
                for (k, s) in out.iter_mut().enumerate() {
                    *s = score(deg(*first + k), &mut p[..], &mut l[..], &mut f[..])?;
                }
                Ok(())
            })?;
        }
        for (i, &s) in scores.iter().enumerate() {
            if s > best.1 {
                *best = (deg(i), s);
            }
        }
        Ok::<(), Error>(())
    };
    sweep(0.0, SKEW_SWEEP_RANGE_DEG, SKEW_SWEEP_STEP_DEG, &mut best)?;
    let centre = best.0;
    sweep(
        centre,
        SKEW_REFINE_RANGE_DEG,
        SKEW_REFINE_STEP_DEG,
        &mut best,
    )?;
    Ok(-best.0 * PI / 180.0)
}

/// `b` rotated about its centre by `radians`, clockwise for a positive angle. The result is the
/// same size as `b`, so the corners are cut off and background is brought in behind them.
///
/// This is a nearest-neighbour rotation with the centre at `(width/2, height/2)` and no
/// interpolation, matching Leptonica's `pixRotateBySampling`. `b` must be depth 1.
///
/// Enforces `limits.max_alloc` (every bitmap this call holds is charged), `limits.max_threads` and
/// the plane caps (`max_pixels`, `max_width`, `max_height`).
///
/// # Errors
///
/// `Error::Imaging` for an argument that does not fit (not depth 1), `Error::PixelLimit` over a
/// plane cap, `Error::AllocLimit` over `max_alloc`, `Error::OutOfMemory` if the allocator refuses,
/// `Error::Cancelled` if `cancel` is set (checked per row).
pub fn rotate(
    b: &Bitmap,
    radians: f64,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<Bitmap, Error> {
    rotate_in(b, radians, &Budget::new(limits), cancel)
}

/// `rotate` on the call's own `budget`; the output rows are independent, so each thread fills a
/// band of them.
pub(crate) fn rotate_in(
    b: &Bitmap,
    radians: f64,
    budget: &Budget,
    cancel: &AtomicBool,
) -> Result<Bitmap, Error> {
    need_depth(b, 1, "Rotate")?;
    // The source coordinate is truncated, not rounded. Leptonica carries it in float32; the
    // products here are taken in f64 and rounded to f32 once, which matches `pixRotateBySampling`
    // pixel for pixel.
    let angle = radians as f32;
    let sina = f64::from(sin(f64::from(angle)) as f32);
    let cosa = f64::from(cos(f64::from(angle)) as f32);

    let mut out = Bitmap::new_in(b.width, b.height, 1, budget)?;
    let (xcen, ycen) = ((b.width / 2) as isize, (b.height / 2) as isize);
    let stride = b.stride;
    let ostride = out.stride;
    if sina.is_nan() || cosa.is_nan() {
        // Every coordinate is NaN, which Go's `int()` maps to MinInt64: out of range.
        return Ok(out);
    }
    // The terms that depend on the column only, and below the ones that depend on the row only,
    // are each the product Go takes per pixel, so the sums are the same f64 operations in the
    // same order.
    let mut xcos = budget.buf(b.width, 0f64)?;
    let mut xsin = budget.buf(b.width, 0f64)?;
    for (j, (c, s)) in xcos.iter_mut().zip(xsin.iter_mut()).enumerate() {
        let xdif = (xcen - j as isize) as f64;
        (*c, *s) = (-xdif * cosa, xdif * sina);
    }
    let (xcos, xsin) = (&xcos[..], &xsin[..]);
    // Every |coordinate| is below width/2 + height/2 + 2 < 2^31 then, so an f32 cast to i32 does
    // not saturate and is Go's `int()`.
    let exact_i32 = b.width <= 1 << 30 && b.height <= 1 << 30;
    par::for_each_row_mut(
        budget,
        cancel,
        b.width * b.height,
        &mut out.pix,
        ostride,
        |i, row| {
            let ydif = (ycen - i as isize) as f64;
            let (ysin, ycos) = (ydif * sina, -ydif * cosa);
            // The f32 value truncated is Go's `int()`; a value too large for an isize saturates
            // here, where Go gives MinInt64, and either is out of range once the centre is added
            // back (a wrapped negative is a huge usize). No value is NaN: the angle is not.
            let source_bit = |xc: f64, xs: f64| -> u8 {
                let x = xcen.wrapping_add(((xc - ysin) as f32) as isize) as usize;
                if x < b.width {
                    let y = ycen.wrapping_add(((ycos + xs) as f32) as isize) as usize;
                    if y < b.height {
                        return (b.pix[y * stride + (x >> 3)] >> (7 - (x & 7))) & 1;
                    }
                }
                0
            };
            // Each source coordinate is monotone in the column (the products, the sums, the f32
            // rounding and the truncation all are), so across one output byte the source row is
            // constant when it is the same at both ends, and the source columns are in range when
            // both ends are. If also each source column is the one before plus 1, the byte is n
            // consecutive bits of one source row. The columns are taken 64 at a time as i32,
            // exact under `exact_i32`.
            let y_at = |j: usize| ycen + (((ycos + xsin[j]) as f32) as i32) as isize;
            let mut tx = [0i32; 64];
            for (c, (bytes, xcs)) in row.chunks_mut(8).zip(xcos.chunks(64)).enumerate() {
                let tx = &mut tx[..xcs.len()];
                for (t, &xc) in tx.iter_mut().zip(xcs) {
                    *t = ((xc - ysin) as f32) as i32;
                }
                let t0 = tx[0];
                let steps = exact_i32
                    && tx
                        .iter()
                        .zip(0..)
                        .fold(0, |a, (&t, k)| a | (t.wrapping_sub(t0) ^ k))
                        == 0;
                // The output is gathered a byte (8 pixels) at a time, so no pixel branches on its
                // value; the last byte of a row may hold fewer, aligned to the high bits.
                for (k, (byte, txs)) in bytes.iter_mut().zip(tx.chunks(8)).enumerate() {
                    let (j, n) = (c * 64 + k * 8, txs.len());
                    let ya = y_at(j);
                    let (x, y) = ((xcen + txs[0] as isize) as usize, ya as usize);
                    *byte = if steps
                        && x < b.width
                        && b.width - x >= n
                        && y < b.height
                        && y_at(j + n - 1) == ya
                    {
                        let at = y * stride + (x >> 3);
                        let next = b.pix.get(at + 1).copied().unwrap_or(0);
                        let pair = (u16::from(b.pix[at]) << 8) | u16::from(next);
                        ((pair << (x & 7)) >> 8) as u8 & (0xff << (8 - n))
                    } else {
                        let mut acc = 0u8;
                        for (&xc, &xs) in xcos[j..j + n].iter().zip(&xsin[j..j + n]) {
                            acc = (acc << 1) | source_bit(xc, xs);
                        }
                        acc << (8 - n)
                    };
                }
            }
            Ok(())
        },
    )?;
    Ok(out)
}
