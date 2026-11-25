//! Otsu and Sauvola binarization (`binarize.go`).

use super::gomath::go_int;
use super::{Bitmap, check_cancel, check_shape};
use crate::par;
use crate::{Budget, Error, Limits};
use std::sync::atomic::AtomicBool;

/// The grayscale level maximising between-class variance, the standard Otsu criterion.
///
/// Enforces `limits.max_alloc` (the per-band histograms are charged), `limits.max_threads` (the
/// rows are counted in bands, one per thread) and the plane caps.
///
/// # Errors
///
/// `Error::PixelLimit` for a plane over the `limits` caps, `Error::Imaging` for an inconsistent
/// bitmap, `Error::Cancelled`, `Error::AllocLimit` over `max_alloc`, `Error::OutOfMemory` if the allocator refuses.
pub fn otsu_threshold(gray: &Bitmap, limits: &Limits, cancel: &AtomicBool) -> Result<u8, Error> {
    otsu_threshold_in(gray, &Budget::new(limits), cancel)
}

/// `otsu_threshold` on the call's own `budget`; the histogram is counted by one band of rows per
/// thread and the counts are added, so the result does not depend on the thread count.
pub(crate) fn otsu_threshold_in(
    gray: &Bitmap,
    budget: &Budget,
    cancel: &AtomicBool,
) -> Result<u8, Error> {
    check_shape(gray)?;
    budget.limits().check_plane(gray.width, gray.height)?;
    let bands = par::band_count(
        budget,
        gray.width * gray.height,
        0,
        std::mem::size_of::<[usize; 256]>(),
    );
    let rows = gray.height.div_ceil(bands).max(1);
    let mut parts = budget.buf(bands, [0usize; 256])?;
    par::for_each_chunk_mut(budget, cancel, &mut parts[..], 1, |b, part| {
        for y in b * rows..((b + 1) * rows).min(gray.height) {
            check_cancel(cancel)?;
            for x in 0..gray.width {
                part[0][usize::from(gray.at(x, y))] += 1;
            }
        }
        Ok(())
    })?;
    let mut hist = [0usize; 256];
    for part in parts.iter() {
        for (h, n) in hist.iter_mut().zip(part) {
            *h += n;
        }
    }
    let total = gray.width * gray.height;
    let mut sum = 0.0f64;
    for (i, n) in hist.iter().enumerate() {
        sum += (i * n) as f64;
    }

    let mut sum_b = 0.0f64;
    let mut w_b = 0usize;
    let mut best = 0.0f64;
    let mut threshold = 0u8;
    for t in 0..256 {
        w_b += hist[t];
        if w_b == 0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0 {
            break;
        }
        sum_b += (t * hist[t]) as f64;
        let m_b = sum_b / w_b as f64;
        let m_f = (sum - sum_b) / w_f as f64;
        let between = w_b as f64 * w_f as f64 * (m_b - m_f) * (m_b - m_f);
        if between > best {
            best = between;
            threshold = t as u8;
        }
    }
    Ok(threshold)
}

/// Binarizes `gray` at its Otsu threshold. In the result, 1 is foreground (ink): a pixel
/// darker than or equal to the threshold becomes 1.
///
/// Enforces `limits.max_alloc` (every bitmap this call holds is charged), `limits.max_threads` and
/// the plane caps (`max_pixels`, `max_width`, `max_height`).
///
/// # Errors
///
/// `Error::Imaging` for an inconsistent bitmap (the depth is not checked, as in Go: a depth-1
/// bitmap is read as 0/1 values), `Error::PixelLimit` over a
/// plane cap, `Error::AllocLimit` over `max_alloc`, `Error::OutOfMemory` if the allocator refuses,
/// `Error::Cancelled` if `cancel` is set (checked per row).
pub fn otsu(gray: &Bitmap, limits: &Limits, cancel: &AtomicBool) -> Result<Bitmap, Error> {
    otsu_in(gray, &Budget::new(limits), cancel)
}

/// `otsu` on the call's own `budget`.
pub(crate) fn otsu_in(
    gray: &Bitmap,
    budget: &Budget,
    cancel: &AtomicBool,
) -> Result<Bitmap, Error> {
    let t = otsu_threshold_in(gray, budget, cancel)?;
    let mut out = Bitmap::new_in(gray.width, gray.height, 1, budget)?;
    let stride = out.stride;
    par::for_each_row_mut(
        budget,
        cancel,
        gray.width * gray.height,
        &mut out.pix,
        stride,
        |y, row| {
            for x in 0..gray.width {
                if gray.at(x, y) <= t {
                    row[x >> 3] |= 0x80 >> (x & 7);
                }
            }
            Ok(())
        },
    )?;
    Ok(out)
}

/// Binarizes `gray` against a threshold computed per pixel from the statistics of the window
/// around it, `t = m * (1 - k * (1 - s/128))`, where `m` and `s` are the mean and standard
/// deviation over that window. A pixel strictly darker than its own threshold becomes 1.
///
/// `window` is the edge length of the square window, rounded down to the nearest odd value; the
/// effective window must be at least 5 and no wider than the bitmap. Windows overhanging an edge
/// are filled by reflecting the image about that edge with the edge pixel duplicated. The mean is
/// truncated to a byte, the mean square rounded to an integer and the threshold truncated, as
/// Leptonica's `pixSauvolaBinarize` does, so the two agree pixel for pixel.
///
/// Enforces `limits.max_alloc` (every bitmap this call holds is charged), `limits.max_threads` and
/// the plane caps (`max_pixels`, `max_width`, `max_height`).
///
/// # Errors
///
/// `Error::Imaging` for an inconsistent bitmap or a window under 5 or wider than the bitmap (the
/// depth is not checked, as in Go: a depth-1 bitmap is read as 0/1 values), `Error::PixelLimit` over a plane cap, `Error::AllocLimit` over `max_alloc`,
/// `Error::OutOfMemory` if the allocator refuses, `Error::Cancelled` if `cancel` is set (checked
/// per row).
pub fn sauvola(
    gray: &Bitmap,
    window: usize,
    k: f64,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<Bitmap, Error> {
    sauvola_in(gray, window, k, &Budget::new(limits), cancel)
}

/// `sauvola` on the call's own `budget`. The integral images are built by rows (each row's
/// running sums, in parallel) and then summed down the columns; the window pass is parallel by
/// rows. Every sum is an integer, so the result does not depend on the thread count.
pub(crate) fn sauvola_in(
    gray: &Bitmap,
    window: usize,
    k: f64,
    budget: &Budget,
    cancel: &AtomicBool,
) -> Result<Bitmap, Error> {
    check_shape(gray)?;
    let half = window.saturating_sub(1) / 2;
    if window < 1 || half < 2 || half + 1 > gray.width || half + 1 > gray.height {
        return Err(Error::Imaging(format!(
            "Sauvola window {window} invalid for a {}x{} bitmap",
            gray.width, gray.height
        )));
    }

    // Integral images over the reflected image, which is inset by `border` on every side.
    // Both are inclusive. The u32 sums wrap as Go's do; a window sum is far below 2^32.
    let border = half + 1;
    let (pw, ph) = (gray.width + 2 * border, gray.height + 2 * border);
    budget.limits().check_plane(pw, ph)?;
    let mut sums = budget.buf(pw * ph, 0u32)?;
    let mut squares = budget.buf(pw * ph, 0f64)?;
    {
        // Each row holds its own running sums for now.
        type Row<'a> = (&'a mut [u32], &'a mut [f64]);
        let _rows_hold = budget.hold(ph.saturating_mul(std::mem::size_of::<Row>()))?;
        let mut rows: Vec<Row> = sums.chunks_mut(pw).zip(squares.chunks_mut(pw)).collect();
        par::for_each_row_mut(budget, cancel, pw * ph, &mut rows, 1, |y, row| {
            let (rs_row, rq_row) = &mut row[0];
            let sy = mirror_index(y as isize - border as isize, gray.height);
            let (mut rs, mut rq) = (0u32, 0f64);
            for x in 0..pw {
                let v =
                    u32::from(gray.at(mirror_index(x as isize - border as isize, gray.width), sy));
                rs = rs.wrapping_add(v);
                rq += f64::from(v) * f64::from(v);
                rs_row[x] = rs;
                rq_row[x] = rq;
            }
            Ok(())
        })?;
    }
    // Cell (x, y) is its row's running sum plus the cell above it. The sums are integers (the
    // f64 ones stay far below 2^53, so they are exact) and the u32 ones wrap, so the order of
    // the additions does not change a value.
    for y in 1..ph {
        check_cancel(cancel)?;
        let (above, cur) = sums.split_at_mut(y * pw);
        for (c, a) in cur[..pw].iter_mut().zip(&above[(y - 1) * pw..]) {
            *c = c.wrapping_add(*a);
        }
        let (above, cur) = squares.split_at_mut(y * pw);
        for (c, a) in cur[..pw].iter_mut().zip(&above[(y - 1) * pw..]) {
            *c += *a;
        }
    }

    // A window of side `incr` centred on pixel (x, y) spans reflected columns x+1..x+incr and
    // rows y+1..y+incr.
    let incr = 2 * half + 1;
    let area = incr * incr;
    let mean_norm = (1.0 / area as f64) as f32;
    let square_norm = 1.0 / area as f64;

    let mut out = Bitmap::new_in(gray.width, gray.height, 1, budget)?;
    let (sums, squares) = (&sums[..], &squares[..]);
    let stride = out.stride;
    par::for_each_row_mut(
        budget,
        cancel,
        gray.width * gray.height,
        &mut out.pix,
        stride,
        |y, row| {
            let (top, bot) = (y * pw, (y + incr) * pw);
            let (st, sb) = (&sums[top..top + pw], &sums[bot..bot + pw]);
            let (qt, qb) = (&squares[top..top + pw], &squares[bot..bot + pw]);
            for x in 0..gray.width {
                let sum = sb[x + incr]
                    .wrapping_sub(sb[x])
                    .wrapping_sub(st[x + incr])
                    .wrapping_add(st[x]);
                let square = qb[x + incr] - qb[x] - qt[x + incr] + qt[x];

                let mean = i64::from((sum as f32 * mean_norm) as u8);
                let mean_square = i64::from(((square_norm * square) + 0.5) as u32);
                // The two are quantised independently, so guard the subtraction.
                let variance = (mean_square - mean * mean).max(0);
                let sd = f64::from((variance as f64).sqrt() as f32);

                let threshold = go_int(mean as f64 * (1.0 - k * (1.0 - sd / 128.0)));
                if i64::from(gray.at(x, y)) < threshold {
                    row[x >> 3] |= 0x80 >> (x & 7);
                }
            }
            Ok(())
        },
    )?;
    Ok(out)
}

/// Maps `t` into `[0, n)` by reflecting about the edges with the edge element duplicated: -1 and
/// 0 both give 0, `n-1` and `n` both give `n-1`.
fn mirror_index(t: isize, n: usize) -> usize {
    let n = n as isize;
    (if t < 0 {
        -1 - t
    } else if t >= n {
        2 * n - 1 - t
    } else {
        t
    }) as usize
}
