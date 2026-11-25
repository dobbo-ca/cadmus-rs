//! A translation of Leptonica's `pixScale` and the 8bpp routines it reaches, from Leptonica
//! 1.87.0 (https://github.com/DanBloomberg/leptonica), licensed under the BSD 2-Clause License:
//! `src/scale1.c` (`pixScale`, `pixScaleGeneral`, `pixScaleGrayLI`, `pixScaleGray2xLI`,
//! `pixScaleGray4xLI`, `pixScaleAreaMap`, `pixScaleAreaMap2`, `pixScaleSmooth` and their
//! low-level kernels), `src/enhance.c` (`pixUnsharpMasking`) and `src/pix2.c`
//! (`pixCopyBorder`). The translation is not verbatim. See NOTICE.
//!
//! For `scalex == scaley == f`, `pixScale` sharpens with fraction 0.2 and width 1 below 0.7 and
//! 0.4 and 2 above, and `pixScaleGeneral` picks: `f == 1.0` a copy; `f < 0.02` `scale_smooth`;
//! `0.02 <= f < 0.7` `scale_area_map`; `f >= 0.7` `scale_gray_li`; then unsharp masking on the
//! reduction branch when `f > 0.2` and on the interpolation branch when `f < 1.4`. The
//! exact-factor kernels (area map at 0.5/0.25/0.125/0.0625, linear interpolation at 2 and 4) are
//! different kernels, not fast paths.
//!
//! Leptonica computes in `l_float32` and truncates toward zero on every cast to `l_int32`, so
//! the arithmetic is f32 with explicit conversions where C converts. Where a literal promotes an
//! expression to double (`scy * (i + 1.0)`, `+ 0.5`) that promotion is reproduced too.

use super::gomath::go_int;
use super::{Bitmap, need_depth};
use crate::nn::Scratch;
use crate::par;
use crate::{Budget, Error, Limits};
use std::sync::atomic::AtomicBool;

/// The call's ledger and cancel flag, and the buffers its bitmaps and work arrays come from when
/// the caller keeps them between calls (`None`: every one is allocated, charged and freed).
struct Cx<'a> {
    budget: &'a Budget,
    cancel: &'a AtomicBool,
    pool: Option<&'a mut Scratch>,
}

impl Cx<'_> {
    fn bitmap(&mut self, w: usize, h: usize, depth: usize) -> Result<Bitmap, Error> {
        let budget = self.budget;
        match self.pool.as_deref_mut() {
            Some(p) => Bitmap::new_with(w, h, depth, budget, |n| p.u8s.take(budget, n, 0u8)),
            None => Bitmap::new_in(w, h, depth, budget),
        }
    }

    fn clone_bitmap(&mut self, src: &Bitmap) -> Result<Bitmap, Error> {
        let budget = self.budget;
        match self.pool.as_deref_mut() {
            Some(p) => {
                let mut pix = p.u8s.take(budget, src.pix.len(), 0u8)?;
                pix.copy_from_slice(&src.pix);
                Ok(Bitmap {
                    width: src.width,
                    height: src.height,
                    depth: src.depth,
                    stride: src.stride,
                    pix,
                })
            }
            None => src.clone_in(budget),
        }
    }

    fn free(&mut self, b: Bitmap) {
        release_bitmap(b, self.budget, self.pool.as_deref_mut());
    }

    /// `n` zeros of work array.
    fn sums(&mut self, n: usize) -> Result<Vec<f32>, Error> {
        match self.pool.as_deref_mut() {
            Some(p) => p.f32s.take(self.budget, n, 0f32),
            None => self.budget.vec(n, 0f32),
        }
    }

    fn free_sums(&mut self, v: Vec<f32>) {
        match self.pool.as_deref_mut() {
            Some(p) => p.f32s.give(v),
            None => {
                let bytes = v.len() * std::mem::size_of::<f32>();
                drop(v);
                self.budget.release(bytes);
            }
        }
    }
}

/// Frees `b` and returns its bytes to `budget`, or gives its pixels back to `pool`.
pub(crate) fn release_bitmap(b: Bitmap, budget: &Budget, pool: Option<&mut Scratch>) {
    match pool {
        Some(p) => p.u8s.give(b.pix),
        None => b.free_in(budget),
    }
}

/// Scales an 8bpp bitmap isotropically, reproducing Leptonica's `pixScale(src, factor, factor)`
/// with its sharpening, so the output dimensions come from Leptonica's own rounding. An empty
/// source, or a factor that is not positive and finite as an f32, is an error (Go panics or misbehaves).
///
/// The call's one [`Budget`] covers the scaled bitmap, the sharpening's work buffers and any
/// intermediate result; the intermediates are released as they are freed.
///
/// Enforces `limits.max_alloc` and `limits.max_threads` (the rows are split into bands, one per
/// thread).
///
/// # Errors
///
/// `Error::Imaging` for an invalid bitmap or factor, `Error::PixelLimit` and `Error::AllocLimit`
/// from `limits`, `Error::Cancelled`, `Error::OutOfMemory` if the allocator refuses.
pub fn scale_gray(
    src: &Bitmap,
    factor: f64,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<Bitmap, Error> {
    scale_gray_in(src, factor, &Budget::new(limits), cancel)
}

/// `scale_gray` on the call's own `budget`; only the result stays charged.
pub(crate) fn scale_gray_in(
    src: &Bitmap,
    factor: f64,
    budget: &Budget,
    cancel: &AtomicBool,
) -> Result<Bitmap, Error> {
    scale_gray_pooled(src, factor, budget, cancel, None)
}

/// `scale_gray_in`, with the bitmaps and work arrays taken from `pool` (and the result one of
/// them: give it back with `release_bitmap`) when there is one.
pub(crate) fn scale_gray_pooled(
    src: &Bitmap,
    factor: f64,
    budget: &Budget,
    cancel: &AtomicBool,
    pool: Option<&mut Scratch>,
) -> Result<Bitmap, Error> {
    need_depth(src, 8, "ScaleGray")?;
    // Go checks only `factor <= 0`; NaN, +Inf and a factor that overflows f32 (1e39) give it a
    // bitmap of negative size, which the crate reports as an error.
    if !(factor > 0.0 && (factor as f32).is_finite()) {
        return Err(Error::Imaging(format!(
            "ScaleGray needs a finite factor > 0, got {factor}"
        )));
    }
    if src.width == 0 || src.height == 0 {
        return Err(Error::Imaging("ScaleGray needs a non-empty bitmap".into()));
    }
    let scale = factor as f32;
    // pixScale halves its default sharpening parameters below 0.7.
    let (sharp_fract, sharp_width) = if f64::from(scale) < 0.7 {
        (0.2f32, 1usize)
    } else {
        (0.4f32, 2usize)
    };
    let mut cx = Cx {
        budget,
        cancel,
        pool,
    };
    scale_general(&mut cx, src, scale, sharp_fract, sharp_width)
}

/// `pixScaleGeneral` restricted to 8bpp and to equal x and y factors.
fn scale_general(
    cx: &mut Cx,
    src: &Bitmap,
    scale: f32,
    sharp_fract: f32,
    sharp_width: usize,
) -> Result<Bitmap, Error> {
    let s = f64::from(scale);
    if s == 1.0 {
        return cx.clone_bitmap(src);
    }
    let (scaled, sharpen) = if s < 0.7 {
        let scaled = if s < 0.02 {
            scale_smooth(cx, src, scale)?
        } else {
            scale_area_map(cx, src, scale)?
        };
        (scaled, s > 0.2)
    } else {
        (scale_gray_li(cx, src, scale)?, s < 1.4)
    };
    if sharpen && sharp_fract > 0.0 && sharp_width > 0 {
        let sharpened = unsharp_masking_gray_2d(cx, &scaled, sharp_width, sharp_fract);
        cx.free(scaled);
        return sharpened;
    }
    Ok(scaled)
}

/// `pixScaleGrayLI`.
fn scale_gray_li(cx: &mut Cx, src: &Bitmap, scale: f32) -> Result<Bitmap, Error> {
    if f64::from(scale) == 2.0 {
        return scale_gray_2x_li(cx, src);
    }
    if f64::from(scale) == 4.0 {
        return scale_gray_4x_li(cx, src);
    }
    let mut dst = new_scaled(cx, scale, src)?;
    scale_gray_li_low(cx, &mut dst, src)?;
    Ok(dst)
}

/// Leptonica's `(l_int32)(scale * (l_float32)n + 0.5)`: an f32 product, then a double addition,
/// then truncation.
fn scaled_dim(scale: f32, n: usize) -> usize {
    (f64::from(scale * n as f32) + 0.5) as usize
}

fn new_scaled(cx: &mut Cx, scale: f32, src: &Bitmap) -> Result<Bitmap, Error> {
    let (w, h) = (scaled_dim(scale, src.width), scaled_dim(scale, src.height));
    cx.bitmap(w, h, 8)
}

/// Each src pixel is divided into 16x16 sub-pixels and each dest pixel is the area-weighted blend
/// of the four src pixels nearest its upper-left corner.
fn scale_gray_li_low(cx: &Cx, dst: &mut Bitmap, src: &Bitmap) -> Result<(), Error> {
    let (ws, hs) = (src.width as isize, src.height as isize);
    let (wd, hd) = (dst.width, dst.height);
    // (scx, scy) map dest coords to src coords, in sixteenths of a pixel.
    let scx = 16.0f32 * ws as f32 / wd as f32;
    let scy = 16.0f32 * hs as f32 / hd as f32;
    let (wm2, hm2) = (ws - 2, hs - 2);

    let stride = dst.stride;
    par::for_each_row_mut(
        cx.budget,
        cx.cancel,
        wd * hd,
        &mut dst.pix,
        stride,
        |i, row| {
            let ypm = (scy * i as f32) as isize;
            let (yp, yf) = (ypm >> 4, ypm & 0x0f);
            for j in 0..wd {
                let xpm = (scx * j as f32) as isize;
                let (xp, xf) = (xpm >> 4, xpm & 0x0f);
                let px = |x: isize, y: isize| isize::from(src.at(x as usize, y as usize));

                let v00val = px(xp, yp);
                let (v01val, v10val, v11val);
                if xp <= wm2 && yp <= hm2 {
                    v10val = px(xp + 1, yp);
                    v01val = px(xp, yp + 1);
                    v11val = px(xp + 1, yp + 1);
                } else if yp > hm2 && xp <= wm2 {
                    // pixels near the bottom
                    v01val = v00val;
                    v10val = px(xp + 1, yp);
                    v11val = v10val;
                } else if xp > wm2 && yp <= hm2 {
                    // pixels near the right side
                    v01val = px(xp, yp + 1);
                    v10val = v00val;
                    v11val = v01val;
                } else {
                    // pixels at the lower-right corner
                    v10val = v00val;
                    v01val = v00val;
                    v11val = v00val;
                }

                let v00 = (16 - xf) * (16 - yf) * v00val;
                let v10 = xf * (16 - yf) * v10val;
                let v01 = (16 - xf) * yf * v01val;
                let v11 = xf * yf * v11val;
                row[j] = ((v00 + v01 + v10 + v11 + 128) / 256) as u8;
            }
            Ok(())
        },
    )
}

/// `pixScaleGray2xLI`: 2x expansion where each src pixel becomes four dest pixels.
fn scale_gray_2x_li(cx: &mut Cx, src: &Bitmap) -> Result<Bitmap, Error> {
    let mut dst = cx.bitmap(2 * src.width, 2 * src.height, 8)?;
    let ds = dst.stride;
    // Each source line makes two dest lines.
    par::for_each_row_mut(
        cx.budget,
        cx.cancel,
        dst.pix.len(),
        &mut dst.pix,
        2 * ds,
        |i, rows| {
            scale_gray_2x_li_line(rows, ds, src, i, i == src.height - 1);
            Ok(())
        },
    )?;
    Ok(dst)
}

/// `rows` holds the two dest lines of source line `is`, `ds` bytes each.
fn scale_gray_2x_li_line(rows: &mut [u8], ds: usize, src: &Bitmap, is: usize, last_line: bool) {
    let wsm = src.width - 1;
    let px = |x: usize, y: usize| usize::from(src.at(x, y));
    if last_line {
        // No lower src line: both dest lines come from this one.
        let mut sval2 = px(0, is);
        for j in 0..wsm {
            let jd = 2 * j;
            let sval1 = sval2;
            sval2 = px(j + 1, is);
            rows[jd] = sval1 as u8;
            rows[ds + jd] = sval1 as u8;
            rows[jd + 1] = ((sval1 + sval2) / 2) as u8;
            rows[ds + jd + 1] = ((sval1 + sval2) / 2) as u8;
        }
        let last = sval2 as u8;
        rows[2 * wsm] = last;
        rows[2 * wsm + 1] = last;
        rows[ds + 2 * wsm] = last;
        rows[ds + 2 * wsm + 1] = last;
        return;
    }

    let mut sval2 = px(0, is);
    let mut sval4 = px(0, is + 1);
    for j in 0..wsm {
        let jd = 2 * j;
        let (sval1, sval3) = (sval2, sval4);
        sval2 = px(j + 1, is);
        sval4 = px(j + 1, is + 1);
        rows[jd] = sval1 as u8;
        rows[jd + 1] = ((sval1 + sval2) / 2) as u8;
        rows[ds + jd] = ((sval1 + sval3) / 2) as u8;
        rows[ds + jd + 1] = ((sval1 + sval2 + sval3 + sval4) / 4) as u8;
    }
    let (sval1, sval3) = (sval2, sval4);
    rows[2 * wsm] = sval1 as u8;
    rows[2 * wsm + 1] = sval1 as u8;
    rows[ds + 2 * wsm] = ((sval1 + sval3) / 2) as u8;
    rows[ds + 2 * wsm + 1] = ((sval1 + sval3) / 2) as u8;
}

/// `pixScaleGray4xLI`: 4x expansion where each src pixel becomes sixteen dest pixels, bilinearly
/// weighted in quarters.
fn scale_gray_4x_li(cx: &mut Cx, src: &Bitmap) -> Result<Bitmap, Error> {
    let mut dst = cx.bitmap(4 * src.width, 4 * src.height, 8)?;
    let ds = dst.stride;
    // Each source line makes four dest lines.
    par::for_each_row_mut(
        cx.budget,
        cx.cancel,
        dst.pix.len(),
        &mut dst.pix,
        4 * ds,
        |i, rows| {
            scale_gray_4x_li_line(rows, ds, src, i, i == src.height - 1);
            Ok(())
        },
    )?;
    Ok(dst)
}

/// `rows` holds the four dest lines of source line `is`, `ds` bytes each.
fn scale_gray_4x_li_line(rows: &mut [u8], ds: usize, src: &Bitmap, is: usize, last_line: bool) {
    let wsm = src.width - 1;
    let wsm4 = 4 * wsm;
    let px = |x: usize, y: usize| usize::from(src.at(x, y));
    if last_line {
        // No lower src line: all four dest lines come from this one.
        let mut s2 = px(0, is);
        for j in 0..wsm {
            let jd = 4 * j;
            let s1 = s2;
            s2 = px(j + 1, is);
            let (s1t, s2t) = (3 * s1, 3 * s2);
            for r in 0..4 {
                rows[r * ds + jd] = s1 as u8;
                rows[r * ds + jd + 1] = ((s1t + s2) / 4) as u8;
                rows[r * ds + jd + 2] = ((s1 + s2) / 2) as u8;
                rows[r * ds + jd + 3] = ((s1 + s2t) / 4) as u8;
            }
        }
        let last = s2 as u8;
        for r in 0..4 {
            for k in 0..4 {
                rows[r * ds + wsm4 + k] = last;
            }
        }
        return;
    }

    let mut s2 = px(0, is);
    let mut s4 = px(0, is + 1);
    for j in 0..wsm {
        let jd = 4 * j;
        let (s1, s3) = (s2, s4);
        s2 = px(j + 1, is);
        s4 = px(j + 1, is + 1);
        let (s1t, s2t, s3t, s4t) = (3 * s1, 3 * s2, 3 * s3, 3 * s4);
        rows[jd] = s1 as u8;
        rows[jd + 1] = ((s1t + s2) / 4) as u8;
        rows[jd + 2] = ((s1 + s2) / 2) as u8;
        rows[jd + 3] = ((s1 + s2t) / 4) as u8;
        rows[ds + jd] = ((s1t + s3) / 4) as u8;
        rows[ds + jd + 1] = ((9 * s1 + s2t + s3t + s4) / 16) as u8;
        rows[ds + jd + 2] = ((s1t + s2t + s3 + s4) / 8) as u8;
        rows[ds + jd + 3] = ((s1t + 9 * s2 + s3 + s4t) / 16) as u8;
        rows[2 * ds + jd] = ((s1 + s3) / 2) as u8;
        rows[2 * ds + jd + 1] = ((s1t + s2 + s3t + s4) / 8) as u8;
        rows[2 * ds + jd + 2] = ((s1 + s2 + s3 + s4) / 4) as u8;
        rows[2 * ds + jd + 3] = ((s1 + s2t + s3 + s4t) / 8) as u8;
        rows[3 * ds + jd] = ((s1 + s3t) / 4) as u8;
        rows[3 * ds + jd + 1] = ((s1t + s2 + 9 * s3 + s4t) / 16) as u8;
        rows[3 * ds + jd + 2] = ((s1 + s2 + s3t + s4t) / 8) as u8;
        rows[3 * ds + jd + 3] = ((s1 + s2t + s3t + 9 * s4) / 16) as u8;
    }
    let (s1, s3) = (s2, s4);
    let (s1t, s3t) = (3 * s1, 3 * s3);
    for k in 0..4 {
        rows[wsm4 + k] = s1 as u8;
        rows[ds + wsm4 + k] = ((s1t + s3) / 4) as u8;
        rows[2 * ds + wsm4 + k] = ((s1 + s3) / 2) as u8;
        rows[3 * ds + wsm4 + k] = ((s1 + s3t) / 4) as u8;
    }
}

/// `pixScaleAreaMap` for 8bpp. The exact 2x, 4x, 8x and 16x reductions are separate kernels:
/// `pixScaleAreaMap2` averages whole 2x2 blocks with a shift.
fn scale_area_map(cx: &mut Cx, src: &Bitmap, scale: f32) -> Result<Bitmap, Error> {
    let sc = f64::from(scale);
    let halvings = if sc == 0.5 {
        1
    } else if sc == 0.25 {
        2
    } else if sc == 0.125 {
        3
    } else if sc == 0.0625 {
        4
    } else {
        0
    };
    if halvings > 0 {
        let mut cur = scale_area_map_2(cx, src)?;
        for _ in 1..halvings {
            let next = scale_area_map_2(cx, &cur)?;
            let old = std::mem::replace(&mut cur, next);
            cx.free(old);
        }
        return Ok(cur);
    }
    let mut dst = new_scaled(cx, scale, src)?;
    scale_gray_area_map_low(cx, &mut dst, src)?;
    Ok(dst)
}

/// `pixScaleAreaMap2` for 8bpp: each dest pixel is the mean of a 2x2 src block, truncated by a
/// right shift.
fn scale_area_map_2(cx: &mut Cx, src: &Bitmap) -> Result<Bitmap, Error> {
    let mut dst = cx.bitmap(src.width / 2, src.height / 2, 8)?;
    let (w, stride) = (dst.width, dst.stride);
    par::for_each_row_mut(
        cx.budget,
        cx.cancel,
        dst.pix.len(),
        &mut dst.pix,
        stride,
        |i, row| {
            area_map_2_row(src, w, i, row);
            Ok(())
        },
    )?;
    Ok(dst)
}

/// Row `i` of `pixScaleAreaMap2`: `w` dest pixels, each the mean of a 2x2 src block.
fn area_map_2_row(src: &Bitmap, w: usize, i: usize, row: &mut [u8]) {
    // The two src rows as slices, so the pairs are read without a bounds check each. `src` is
    // depth 8 and `2w` wide at most, and `2i + 1` is a src row (`scale_area_map_2` halves both).
    let (top, bottom) = (2 * i * src.stride, (2 * i + 1) * src.stride);
    let (Some(top), Some(bottom)) = (
        src.pix.get(top..top + 2 * w),
        src.pix.get(bottom..bottom + 2 * w),
    ) else {
        return;
    };
    let ((top, _), (bottom, _)) = (top.as_chunks::<2>(), bottom.as_chunks::<2>());
    for ((out, t), b) in row.iter_mut().zip(top).zip(bottom) {
        *out = ((usize::from(t[0]) + usize::from(t[1]) + usize::from(b[0]) + usize::from(b[1]))
            >> 2) as u8;
    }
}

/// Every src pixel is subdivided into 256 sub-pixels and weighted by the number of sub-pixels
/// the dest pixel covers.
fn scale_gray_area_map_low(cx: &Cx, dst: &mut Bitmap, src: &Bitmap) -> Result<(), Error> {
    let (ws, hs) = (src.width as isize, src.height as isize);
    let (wd, hd) = (dst.width, dst.height);
    let scx = 16.0f32 * ws as f32 / wd as f32;
    let scy = 16.0f32 * hs as f32 / hd as f32;
    let (wm2, hm2) = (ws - 2, hs - 2);
    let px = |x: isize, y: isize| isize::from(src.at(x as usize, y as usize));

    let stride = dst.stride;
    par::for_each_row_mut(
        cx.budget,
        cx.cancel,
        wd * hd,
        &mut dst.pix,
        stride,
        |i, row| {
            // The lower corner promotes to double in Leptonica, the upper one does not; the
            // asymmetry is observable.
            let yu = (scy * i as f32) as isize;
            let yl = (f64::from(scy) * (i as f64 + 1.0)) as isize;
            let (yup, yuf) = (yu >> 4, yu & 0x0f);
            let (ylp, ylf) = (yl >> 4, yl & 0x0f);
            let dely = ylp - yup;
            for j in 0..wd {
                let xu = (scx * j as f32) as isize;
                let xl = (f64::from(scx) * (j as f64 + 1.0)) as isize;
                let (xup, xuf) = (xu >> 4, xu & 0x0f);
                let (xlp, xlf) = (xl >> 4, xl & 0x0f);
                let delx = xlp - xup;

                // If near the edge, just use a src pixel value.
                if xlp > wm2 || ylp > hm2 {
                    row[j] = src.at(xup as usize, yup as usize);
                    continue;
                }

                // The area summed over, in sub-pixels. Quantization makes it vary.
                let area =
                    ((16 - xuf) + 16 * (delx - 1) + xlf) * ((16 - yuf) + 16 * (dely - 1) + ylf);
                if area == 0 {
                    return Err(Error::Imaging("degenerate area-map window".into()));
                }

                let v00 = (16 - xuf) * (16 - yuf) * px(xup, yup);
                let v10 = xlf * (16 - yuf) * px(xlp, yup);
                let v01 = (16 - xuf) * ylf * px(xup, yup + dely);
                let v11 = xlf * ylf * px(xlp, yup + dely);
                let mut vin = 0isize;
                for k in 1..dely {
                    // full interior src pixels
                    for m in 1..delx {
                        vin += 256 * px(xup + m, yup + k);
                    }
                }
                let mut vmid = 0isize;
                for k in 1..dely {
                    // left side
                    vmid += (16 - xuf) * 16 * px(xup, yup + k);
                }
                for k in 1..dely {
                    // right side
                    vmid += xlf * 16 * px(xlp, yup + k);
                }
                for m in 1..delx {
                    // top side
                    vmid += 16 * (16 - yuf) * px(xup + m, yup);
                }
                for m in 1..delx {
                    // bottom side
                    vmid += 16 * ylf * px(xup + m, yup + dely);
                }
                row[j] = ((v00 + v01 + v10 + v11 + vin + vmid + 128) / area) as u8;
            }
            Ok(())
        },
    )
}

/// `pixScaleSmooth` for 8bpp: a flat filter whose width tracks the reduction ratio, evaluated
/// only at the subsampling locations.
fn scale_smooth(cx: &mut Cx, src: &Bitmap, scale: f32) -> Result<Bitmap, Error> {
    let (ws, hs) = (src.width, src.height);
    // The ideal filter full width is 1/scale, never below 2.
    // Go converts with `int()`, which on amd64 maps an out-of-range value to MinInt64 (so 2).
    let isize_ = go_int((1.0f32 / scale) as f64 + 0.5).clamp(2, 10000) as usize;
    if ws < isize_ || hs < isize_ {
        // Ridiculously small scaling factor: one pixel from the middle.
        let mut dst = cx.bitmap(1, 1, 8)?;
        dst.set(0, 0, src.at(ws / 2, hs / 2));
        return Ok(dst);
    }

    let wd = scaled_dim(scale, ws).max(1);
    let hd = scaled_dim(scale, hs).max(1);
    let mut dst = cx.bitmap(wd, hd, 8)?;
    let norm = 1.0f32 / (isize_ * isize_) as f32;
    let wratio = ws as f32 / wd as f32;
    let hratio = hs as f32 / hd as f32;

    // The UL corner of the src square that each dest pixel averages.
    let mut srow = cx.budget.buf(hd, 0usize)?;
    for (i, r) in srow.iter_mut().enumerate() {
        *r = ((hratio * i as f32) as usize).min(hs - isize_);
    }
    let mut scol = cx.budget.buf(wd, 0usize)?;
    for (j, c) in scol.iter_mut().enumerate() {
        *c = ((wratio * j as f32) as usize).min(ws - isize_);
    }

    let (srow, scol) = (&srow[..], &scol[..]);
    let stride = dst.stride;
    par::for_each_row_mut(
        cx.budget,
        cx.cancel,
        wd * hd * isize_ * isize_,
        &mut dst.pix,
        stride,
        |i, row| {
            for j in 0..wd {
                let mut val = 0usize;
                for m in 0..isize_ {
                    for n in 0..isize_ {
                        val += usize::from(src.at(scol[j] + n, srow[i] + m));
                    }
                }
                row[j] = (val as f32 * norm) as usize as u8;
            }
            Ok(())
        },
    )?;
    Ok(dst)
}

/// `pixUnsharpMaskingGray2D`, the routine `pixUnsharpMasking` reaches for halfwidth 1 and 2. The
/// low pass is computed separably; with `L` the low-pass value, `I` the src pixel and `f` the
/// fraction, the sharpened value is `I + f * (I - L)`.
fn unsharp_masking_gray_2d(
    cx: &mut Cx,
    src: &Bitmap,
    halfwidth: usize,
    fract: f32,
) -> Result<Bitmap, Error> {
    let (w, h) = (src.width, src.height);
    // The pixels the loops below never reach keep their src values.
    let mut dst = copy_border(cx, src, halfwidth)?;

    // Horizontal smoothing, into an intermediate float buffer. The columns the loop skips stay
    // zero and are never read back.
    let mut sums_buf = cx.sums(w * h)?;
    par::for_each_row_mut(
        cx.budget,
        cx.cancel,
        w * h,
        &mut sums_buf[..],
        w,
        |i, row| {
            unsharp_sum_row(src, halfwidth, i, row);
            Ok(())
        },
    )?;
    let sums = &sums_buf[..];

    let full = (2 * halfwidth + 1) as f32;
    let norm = 1.0f32 / (full * full);
    let stride = dst.stride;
    par::for_each_row_mut(
        cx.budget,
        cx.cancel,
        w * h,
        &mut dst.pix,
        stride,
        |i, row| {
            if i >= halfwidth && i + halfwidth < h {
                unsharp_row(src, sums, (halfwidth, norm, fract), i, row);
            }
            Ok(())
        },
    )?;
    cx.free_sums(sums_buf);
    Ok(dst)
}

/// The horizontal smoothing of row `i` of `src`, into `row`.
#[inline(never)]
fn unsharp_sum_row(src: &Bitmap, halfwidth: usize, i: usize, row: &mut [f32]) {
    let w = src.width;
    for j in halfwidth..w.saturating_sub(halfwidth) {
        let mut total = 0usize;
        for k in j - halfwidth..=j + halfwidth {
            total += usize::from(src.at(k, i));
        }
        row[j] = total as f32;
    }
}

/// The vertical smoothing and the sharpening of row `i`, from the horizontal sums.
#[inline(never)]
fn unsharp_row(
    src: &Bitmap,
    sums: &[f32],
    (halfwidth, norm, fract): (usize, f32, f32),
    i: usize,
    row: &mut [u8],
) {
    let w = src.width;
    for j in halfwidth..w.saturating_sub(halfwidth) {
        // Vertical smoothing finishes the low-pass filter. The additions run left to right
        // in f32, as Leptonica's do.
        let mut col = 0.0f32;
        for k in i - halfwidth..=i + halfwidth {
            col += sums[k * w + j];
        }
        let lowpass = norm * col;
        let sval = f32::from(src.at(j, i));
        let highpass = fract * (sval - lowpass);
        let ival = (f64::from(sval + highpass) + 0.5) as isize;
        row[j] = ival.clamp(0, 255) as u8;
    }
}

/// `pixCopyBorder` with a single width: a zeroed bitmap of the same size as `src` whose outermost
/// `width` rows and columns are copied from `src`.
fn copy_border(cx: &mut Cx, src: &Bitmap, width: usize) -> Result<Bitmap, Error> {
    let mut dst = cx.bitmap(src.width, src.height, src.depth)?;
    for y in 0..src.height {
        let in_row = y < width || y + width >= src.height;
        for x in 0..src.width {
            if in_row || x < width || x + width >= src.width {
                dst.set(x, y, src.at(x, y));
            }
        }
    }
    Ok(dst)
}
