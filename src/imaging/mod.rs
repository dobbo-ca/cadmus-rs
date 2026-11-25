//! Image primitives: the [`Bitmap`] substrate plus the binarization, morphology and analysis
//! operators built on it. Coordinates are `(x, y)` with the origin top-left, as in Leptonica.
//! A port of `internal/imaging`; `scalegray` is a translation of Leptonica 1.87.0 (see NOTICE).

mod binarize;
mod deskew;
mod gomath;
mod grid;
mod ops;
mod scalegray;

pub use binarize::{otsu, otsu_threshold, sauvola};
pub use deskew::{rotate, skew_angle};
pub use grid::Grid;
pub use ops::{
    Component, Op, close, conn_comp, dilate, distance_function, erode, open, raster_op, seed_fill,
};
pub use scalegray::scale_gray;
#[cfg(test)]
pub(crate) use scalegray::scale_gray_in;
pub(crate) use scalegray::{release_bitmap, scale_gray_pooled};

use crate::{Budget, Error, Limits};
use std::sync::atomic::{AtomicBool, Ordering};

/// A point; `x` grows right and `y` down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Point {
    pub x: isize,
    pub y: isize,
}

/// A half-open rectangle `[min, max)`, as `image.Rectangle`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rect {
    pub min: Point,
    pub max: Point,
}

impl Rect {
    /// `image.Rect`: the corners are swapped if needed so that `min <= max`.
    pub fn new(x0: isize, y0: isize, x1: isize, y1: isize) -> Rect {
        let (x0, x1) = if x0 > x1 { (x1, x0) } else { (x0, x1) };
        let (y0, y1) = if y0 > y1 { (y1, y0) } else { (y0, y1) };
        Rect {
            min: Point { x: x0, y: y0 },
            max: Point { x: x1, y: y1 },
        }
    }

    pub fn dx(&self) -> isize {
        self.max.x.saturating_sub(self.min.x)
    }

    pub fn dy(&self) -> isize {
        self.max.y.saturating_sub(self.min.y)
    }

    pub fn is_empty(&self) -> bool {
        self.min.x >= self.max.x || self.min.y >= self.max.y
    }

    pub fn overlaps(&self, s: &Rect) -> bool {
        !self.is_empty()
            && !s.is_empty()
            && self.min.x < s.max.x
            && s.min.x < self.max.x
            && self.min.y < s.max.y
            && s.min.y < self.max.y
    }
}

/// `Err(Cancelled)` if the flag is set.
pub(crate) fn check_cancel(cancel: &AtomicBool) -> Result<(), Error> {
    if cancel.load(Ordering::Relaxed) {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}

/// An 8-bit grayscale or 1-bit bilevel image. `depth` is 1 or 8.
///
/// In a depth-1 bitmap, 1 means foreground (ink) and 0 background, as in Leptonica; every
/// operator relies on it. Depth-1 rows are packed MSB-first: pixel `x` of a row is bit
/// `7 - x % 8` of byte `x / 8`. Rows are `stride` bytes apart. Coordinates passed to
/// [`Bitmap::at`] and [`Bitmap::set`] must lie inside the bitmap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bitmap {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    /// Bytes per row.
    pub stride: usize,
    pub pix: Vec<u8>,
}

impl Bitmap {
    /// A zeroed `w x h` bitmap of depth 1 or 8, within `limits`.
    ///
    /// Enforces `max_alloc` and the plane caps (`max_pixels`, `max_width`, `max_height`).
    ///
    /// # Errors
    ///
    /// `Error::Imaging` for a depth other than 1 or 8, `Error::PixelLimit` over a plane cap,
    /// `Error::AllocLimit` over `max_alloc`, `Error::OutOfMemory` if the allocator refuses.
    pub fn new(w: usize, h: usize, depth: usize, limits: &Limits) -> Result<Bitmap, Error> {
        Bitmap::new_in(w, h, depth, &Budget::new(limits))
    }

    /// `new`, charging the pixels against `budget` (and its limits) for as long as the caller
    /// keeps them charged.
    ///
    /// # Errors
    ///
    /// As [`Bitmap::new`], with the budget's limits.
    pub fn new_in(w: usize, h: usize, depth: usize, budget: &Budget) -> Result<Bitmap, Error> {
        Bitmap::new_with(w, h, depth, budget, |n| budget.vec(n, 0u8))
    }

    /// `new_in`, with the zeroed pixels `alloc(n)` makes (and charges) instead of `budget.vec`.
    pub(crate) fn new_with(
        w: usize,
        h: usize,
        depth: usize,
        budget: &Budget,
        alloc: impl FnOnce(usize) -> Result<Vec<u8>, Error>,
    ) -> Result<Bitmap, Error> {
        if depth != 1 && depth != 8 {
            return Err(Error::Imaging(format!("unsupported depth {depth}")));
        }
        budget.limits().check_plane(w, h)?;
        let stride = w.checked_mul(depth).ok_or(Error::PixelLimit)?.div_ceil(8);
        let pix = alloc(stride.checked_mul(h).ok_or(Error::PixelLimit)?)?;
        Ok(Bitmap {
            width: w,
            height: h,
            depth,
            stride,
            pix,
        })
    }

    /// An 8bpp bitmap from `w * h` gray bytes.
    ///
    /// Enforces `max_alloc` and the plane caps.
    ///
    /// # Errors
    ///
    /// `Error::Imaging` if `gray.len() != w * h`, else as [`Bitmap::new`].
    pub fn from_gray8(w: usize, h: usize, gray: &[u8], limits: &Limits) -> Result<Bitmap, Error> {
        let mut b = Bitmap::new(w, h, 8, limits)?;
        if gray.len() != w * h {
            return Err(Error::Imaging(format!(
                "{} gray bytes for a {w}x{h} image",
                gray.len()
            )));
        }
        b.pix.copy_from_slice(gray);
        Ok(b)
    }

    /// An 8bpp bitmap from `w * h` premultiplied RGBA pixels (`image.RGBA`), converted with
    /// `color.GrayModel`.
    ///
    /// Enforces `max_alloc` and the plane caps.
    ///
    /// # Errors
    ///
    /// `Error::Imaging` if `rgba.len() != w * h * 4`, else as [`Bitmap::new`].
    pub fn from_rgba8(w: usize, h: usize, rgba: &[u8], limits: &Limits) -> Result<Bitmap, Error> {
        let mut b = Bitmap::new(w, h, 8, limits)?;
        if rgba.len() != w * h * 4 {
            return Err(Error::Imaging(format!(
                "{} RGBA bytes for a {w}x{h} image",
                rgba.len()
            )));
        }
        let (chunks, _) = rgba.as_chunks::<4>();
        for (d, p) in b.pix.iter_mut().zip(chunks) {
            let r = u32::from(p[0]) * 0x101;
            let g = u32::from(p[1]) * 0x101;
            let bl = u32::from(p[2]) * 0x101;
            *d = ((19595 * r + 38470 * g + 7471 * bl + (1 << 15)) >> 24) as u8;
        }
        Ok(b)
    }

    /// An 8bpp bitmap from a binary PGM (`P5`, maxval 255), the plain format the test corpus and
    /// `cadmusdump` read (there is no PNG decoder in std).
    ///
    /// Enforces `max_alloc` and the plane caps; the header's size is checked against them before
    /// any pixel is allocated.
    ///
    /// # Errors
    ///
    /// `Error::Imaging` for a malformed or truncated PGM, else as [`Bitmap::new`].
    pub fn from_pgm(data: &[u8], limits: &Limits) -> Result<Bitmap, Error> {
        let bad = |m: &str| Error::Imaging(format!("PGM: {m}"));
        let mut pos = 0usize;
        let token = |pos: &mut usize| -> Result<usize, Error> {
            loop {
                while *pos < data.len() && data[*pos].is_ascii_whitespace() {
                    *pos += 1;
                }
                if data.get(*pos) == Some(&b'#') {
                    while *pos < data.len() && data[*pos] != b'\n' {
                        *pos += 1;
                    }
                } else {
                    break;
                }
            }
            let start = *pos;
            while *pos < data.len() && !data[*pos].is_ascii_whitespace() {
                *pos += 1;
            }
            std::str::from_utf8(&data[start..*pos])
                .ok()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| bad("bad header"))
        };
        if data.get(..2) != Some(b"P5") {
            return Err(bad("not a binary PGM (P5) file"));
        }
        pos += 2;
        let (w, h, max) = (token(&mut pos)?, token(&mut pos)?, token(&mut pos)?);
        if max != 255 {
            return Err(bad("only maxval 255 is supported"));
        }
        pos += 1; // the single whitespace byte after the header
        let n = w.checked_mul(h).ok_or_else(|| bad("size overflows"))?;
        let pix = data
            .get(pos..)
            .and_then(|d| d.get(..n))
            .ok_or_else(|| bad("pixel data is truncated"))?;
        Bitmap::from_gray8(w, h, pix, limits)
    }

    /// The value at `(x, y)`: 0 or 1 for depth 1, 0-255 for depth 8.
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> u8 {
        if self.depth == 1 {
            return (self.pix[y * self.stride + x / 8] >> (7 - x % 8)) & 1;
        }
        self.pix[y * self.stride + x]
    }

    /// Writes `v` at `(x, y)`. For depth 1 only the low bit of `v` is used.
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, v: u8) {
        if self.depth == 1 {
            let i = y * self.stride + x / 8;
            let mask = 1u8 << (7 - x % 8);
            if v & 1 != 0 {
                self.pix[i] |= mask;
            } else {
                self.pix[i] &= !mask;
            }
            return;
        }
        self.pix[y * self.stride + x] = v;
    }

    /// The bitmap's extent, rooted at the origin.
    pub fn bounds(&self) -> Rect {
        Rect::new(0, 0, self.width as isize, self.height as isize)
    }

    /// A deep copy, charged against `budget`.
    ///
    /// # Errors
    ///
    /// `Error::AllocLimit` over the budget's `max_alloc`, `Error::OutOfMemory` if the allocator
    /// refuses.
    pub fn clone_in(&self, budget: &Budget) -> Result<Bitmap, Error> {
        let mut out = self.clone_shape(budget)?;
        out.pix.copy_from_slice(&self.pix);
        Ok(out)
    }

    /// A zeroed bitmap of the same shape, charged against `budget`.
    pub(crate) fn clone_shape(&self, budget: &Budget) -> Result<Bitmap, Error> {
        Ok(Bitmap {
            width: self.width,
            height: self.height,
            depth: self.depth,
            stride: self.stride,
            pix: budget.vec(self.pix.len(), 0u8)?,
        })
    }

    /// Frees the bitmap and returns the bytes a `new_in` or `clone_in` charged for it.
    pub(crate) fn free_in(self, budget: &Budget) {
        let bytes = self.pix.len();
        drop(self);
        budget.release(bytes);
    }
}

/// The fields of a [`Bitmap`] are public, so an operator cannot trust them: `Err` unless the
/// depth is 1 or 8 and `pix` holds `height` rows of at least `ceil(width * depth / 8)` bytes.
pub(crate) fn check_shape(b: &Bitmap) -> Result<(), Error> {
    let row = b.width.checked_mul(b.depth).map(|bits| bits.div_ceil(8));
    let ok = (b.depth == 1 || b.depth == 8)
        && row.is_some_and(|row| b.stride >= row)
        && b.stride
            .checked_mul(b.height)
            .is_some_and(|n| b.pix.len() >= n);
    if ok {
        Ok(())
    } else {
        Err(Error::Imaging(format!(
            "inconsistent bitmap: {}x{} depth {} stride {} with {} bytes",
            b.width,
            b.height,
            b.depth,
            b.stride,
            b.pix.len()
        )))
    }
}

pub(crate) fn need_depth(b: &Bitmap, depth: usize, what: &str) -> Result<(), Error> {
    check_shape(b)?;
    if b.depth != depth {
        return Err(Error::Imaging(format!(
            "{what} needs a depth-{depth} bitmap, got depth {}",
            b.depth
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
