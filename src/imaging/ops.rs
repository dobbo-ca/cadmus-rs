//! Raster operations, connected components, seed fill, distance function and morphology
//! (`rasterop.go`, `conncomp.go`, `seedfill.go`, `morph.go`).

use super::{Bitmap, Point, Rect, check_cancel, check_shape, need_depth};
use crate::par;
use crate::{Budget, Error, Limits};
use std::sync::atomic::AtomicBool;

/// A bitwise raster operation combining a source and a destination pixel. The values are
/// Leptonica's op codes, a truth table: bit `2*src + dst` of the code is the result for that
/// combination of input bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Clear = 0x0,
    NotSrc = 0x3,
    SrcXorDst = 0x6,
    SrcAndDst = 0x8,
    Src = 0xc,
    SrcOrDst = 0xe,
    Set = 0xf,
}

impl Op {
    fn uses_src(self) -> bool {
        self != Op::Clear && self != Op::Set
    }

    /// Combines one source and one destination value. The masks are full bytes, so the same
    /// expression serves depth 1 and depth 8.
    fn apply(self, s: u8, d: u8) -> u8 {
        let op = self as u8;
        let mut v = 0u8;
        if op & 0x1 != 0 {
            v |= !s & !d;
        }
        if op & 0x2 != 0 {
            v |= !s & d;
        }
        if op & 0x4 != 0 {
            v |= s & !d;
        }
        if op & 0x8 != 0 {
            v |= s & d;
        }
        v
    }
}

/// Combines the source rectangle rooted at `sp` into the destination rectangle `dr` pixel by
/// pixel under `op`. The source rectangle has the same size as `dr`. `src` is not read for
/// `Set` and `Clear` and may be `None` for those.
///
/// Rectangles are clipped, not rejected, as Leptonica's `pixRasterop` does: both are trimmed
/// until they lie inside their images, keeping them the same size and aligned with each other.
/// If nothing survives, `dst` is untouched. The two bitmaps must have the same depth.
///
/// Enforces the plane caps on `dst`; it allocates nothing, and `max_threads` does not apply.
///
/// # Errors
///
/// `Error::Imaging` for a bitmap whose fields disagree with its pixels, a depth mismatch or a
/// missing source, `Error::PixelLimit` over a plane cap, `Error::Cancelled` if `cancel` is set
/// (checked per row).
pub fn raster_op(
    dst: &mut Bitmap,
    dr: Rect,
    op: Op,
    src: Option<&Bitmap>,
    sp: Point,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<(), Error> {
    check_shape(dst)?;
    limits.check_plane(dst.width, dst.height)?;
    let (src_w, src_h, sp) = if op.uses_src() {
        let s = src.ok_or_else(|| {
            Error::Imaging(format!(
                "RasterOp with op {:#x} needs a source bitmap",
                op as u8
            ))
        })?;
        check_shape(s)?;
        if s.depth != dst.depth {
            return Err(Error::Imaging(format!(
                "RasterOp between depth {} and depth {} bitmaps",
                s.depth, dst.depth
            )));
        }
        (s.width as isize, s.height as isize, sp)
    } else {
        // A source-less op is clipped against the destination alone.
        (dst.width as isize, dst.height as isize, dr.min)
    };

    // The coordinates are the caller's, so the clipping saturates: a value that far out clips the
    // rectangle away whichever way it rounds.
    let (mut dx, mut dy) = (dr.min.x, dr.min.y);
    let (mut sx, mut sy) = (sp.x, sp.y);
    let (mut dw, mut dh) = (dr.dx(), dr.dy());

    if dx < 0 {
        sx = sx.saturating_sub(dx);
        dw = dw.saturating_add(dx);
        dx = 0;
    }
    if sx < 0 {
        dx = dx.saturating_sub(sx);
        dw = dw.saturating_add(sx);
        sx = 0;
    }
    dw = dw
        .min((dst.width as isize).saturating_sub(dx))
        .min(src_w.saturating_sub(sx));

    if dy < 0 {
        sy = sy.saturating_sub(dy);
        dh = dh.saturating_add(dy);
        dy = 0;
    }
    if sy < 0 {
        dy = dy.saturating_sub(sy);
        dh = dh.saturating_add(sy);
        sy = 0;
    }
    dh = dh
        .min((dst.height as isize).saturating_sub(dy))
        .min(src_h.saturating_sub(sy));

    for y in 0..dh.max(0) {
        check_cancel(cancel)?;
        for x in 0..dw.max(0) {
            let s = match (op.uses_src(), src) {
                (true, Some(s)) => s.at((sx + x) as usize, (sy + y) as usize),
                _ => 0,
            };
            let (px, py) = ((dx + x) as usize, (dy + y) as usize);
            let d = dst.at(px, py);
            dst.set(px, py, op.apply(s, d));
        }
    }
    Ok(())
}

/// One connected run of foreground pixels: its bounding box and the number of foreground
/// pixels inside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Component {
    pub bounds: Rect,
    pub pixels: usize,
}

const NEIGHBOURS_4: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
const NEIGHBOURS_8: [(isize, isize); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

fn neighbours(connectivity: usize) -> Result<&'static [(isize, isize)], Error> {
    match connectivity {
        4 => Ok(&NEIGHBOURS_4),
        8 => Ok(&NEIGHBOURS_8),
        c => Err(Error::Imaging(format!(
            "unsupported connectivity {c}, want 4 or 8"
        ))),
    }
}

/// A flood-fill work stack of pixel indices whose growth is charged against the budget until the
/// stack is dropped.
struct Stack<'a> {
    v: Vec<u32>,
    budget: &'a Budget,
    charged: usize,
}

impl<'a> Stack<'a> {
    fn new(w: usize, h: usize, budget: &'a Budget) -> Result<Stack<'a>, Error> {
        if (w as u64) * (h as u64) > u64::from(u32::MAX) {
            return Err(Error::PixelLimit);
        }
        Ok(Stack {
            v: Vec::new(),
            budget,
            charged: 0,
        })
    }

    fn push(&mut self, i: usize) -> Result<(), Error> {
        if self.v.len() == self.v.capacity() {
            let add = self.v.capacity().max(1024);
            self.budget.reserve(&mut self.v, add)?;
            self.charged += add * 4;
        }
        self.v.push(i as u32);
        Ok(())
    }
}

impl Drop for Stack<'_> {
    fn drop(&mut self) {
        // The vec goes first, then its charge.
        drop(std::mem::take(&mut self.v));
        self.budget.release(self.charged);
    }
}

/// The connected foreground components of a depth-1 bitmap. `connectivity` is 4 or 8.
///
/// Components come back in raster order of their first foreground pixel: by row, then by
/// column within that row (Leptonica's `pixConnComp` order). An all-background bitmap yields an
/// empty vec.
///
/// Enforces `max_alloc` (visited map, stack and components). It runs on the calling thread.
///
/// # Errors
///
/// `Error::Imaging` for a depth other than 1 or a connectivity other than 4 or 8,
/// `Error::AllocLimit` over `max_alloc`, `Error::OutOfMemory` if the allocator refuses,
/// `Error::Cancelled` if `cancel` is set.
pub fn conn_comp(
    b: &Bitmap,
    connectivity: usize,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<Vec<Component>, Error> {
    need_depth(b, 1, "ConnComp")?;
    let deltas = neighbours(connectivity)?;
    let budget = Budget::new(limits);
    let (w, h) = (b.width, b.height);
    let mut seen = budget.buf(w * h, false)?;
    let mut stack = Stack::new(w, h, &budget)?;
    let mut comps = Vec::new();
    // The flood below is checked for cancel every `w` pixels, like a row.
    let mut flooded = 0usize;

    for y in 0..h {
        check_cancel(cancel)?;
        for x in 0..w {
            if seen[y * w + x] || b.at(x, y) == 0 {
                continue;
            }
            seen[y * w + x] = true;
            stack.v.clear();
            stack.push(y * w + x)?;
            let (mut min_x, mut max_x, mut min_y, mut max_y) = (x, x, y, y);
            let mut pixels = 0usize;
            while let Some(i) = stack.v.pop() {
                let (px, py) = (i as usize % w, i as usize / w);
                pixels += 1;
                flooded += 1;
                if flooded >= w {
                    flooded = 0;
                    check_cancel(cancel)?;
                }
                min_x = min_x.min(px);
                max_x = max_x.max(px);
                min_y = min_y.min(py);
                max_y = max_y.max(py);
                for &(dx, dy) in deltas {
                    let nx = px as isize + dx;
                    let ny = py as isize + dy;
                    if nx < 0 || nx >= w as isize || ny < 0 || ny >= h as isize {
                        continue;
                    }
                    let (nx, ny) = (nx as usize, ny as usize);
                    if seen[ny * w + nx] || b.at(nx, ny) == 0 {
                        continue;
                    }
                    seen[ny * w + nx] = true;
                    stack.push(ny * w + nx)?;
                }
            }
            budget.push(
                &mut comps,
                Component {
                    bounds: Rect::new(
                        min_x as isize,
                        min_y as isize,
                        max_x as isize + 1,
                        max_y as isize + 1,
                    ),
                    pixels,
                },
            )?;
        }
    }
    Ok(comps)
}

/// The binary reconstruction of `seed` under `mask`: every connected component of `mask` holding
/// at least one seed pixel, in full, and nothing else. `connectivity` is 4 or 8 and describes
/// the mask's components. Both bitmaps must be depth 1 and the same size.
///
/// This is Leptonica's `pixSeedfillBinary`, except that it floods each seeded component
/// directly where Leptonica sweeps up to 40 times, so it reaches the full reconstruction
/// whatever the mask's topology.
///
/// Enforces `max_alloc` and the plane caps. It runs on the calling thread.
///
/// # Errors
///
/// `Error::Imaging` for a depth other than 1, differing sizes or a connectivity other than 4 or 8,
/// `Error::PixelLimit` over a plane cap, `Error::AllocLimit` over `max_alloc`,
/// `Error::OutOfMemory` if the allocator refuses, `Error::Cancelled` if `cancel` is set.
pub fn seed_fill(
    seed: &Bitmap,
    mask: &Bitmap,
    connectivity: usize,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<Bitmap, Error> {
    check_shape(seed)?;
    check_shape(mask)?;
    if seed.depth != 1 || mask.depth != 1 {
        return Err(Error::Imaging(format!(
            "SeedFill needs depth-1 bitmaps, got seed depth {} and mask depth {}",
            seed.depth, mask.depth
        )));
    }
    if seed.width != mask.width || seed.height != mask.height {
        return Err(Error::Imaging(format!(
            "SeedFill seed is {}x{} but mask is {}x{}",
            seed.width, seed.height, mask.width, mask.height
        )));
    }
    let deltas = neighbours(connectivity)?;
    let budget = Budget::new(limits);
    let (w, h) = (mask.width, mask.height);
    let mut out = Bitmap::new_in(w, h, 1, &budget)?;
    let mut stack = Stack::new(w, h, &budget)?;
    // The flood below is checked for cancel every `w` pixels, like a row.
    let mut flooded = 0usize;

    for y in 0..h {
        check_cancel(cancel)?;
        for x in 0..w {
            if seed.at(x, y) == 0 || mask.at(x, y) == 0 || out.at(x, y) != 0 {
                continue;
            }
            out.set(x, y, 1);
            stack.v.clear();
            stack.push(y * w + x)?;
            while let Some(i) = stack.v.pop() {
                let (px, py) = (i as usize % w, i as usize / w);
                flooded += 1;
                if flooded >= w {
                    flooded = 0;
                    check_cancel(cancel)?;
                }
                for &(dx, dy) in deltas {
                    let nx = px as isize + dx;
                    let ny = py as isize + dy;
                    if nx < 0 || nx >= w as isize || ny < 0 || ny >= h as isize {
                        continue;
                    }
                    let (nx, ny) = (nx as usize, ny as usize);
                    if out.at(nx, ny) != 0 || mask.at(nx, ny) == 0 {
                        continue;
                    }
                    out.set(nx, ny, 1);
                    stack.push(ny * w + nx)?;
                }
            }
        }
    }
    Ok(out)
}

/// The city-block distance from every foreground pixel of `b` to the nearest background pixel,
/// as a depth-8 bitmap. Background pixels are 0 and the distance grows from 1 at a component's
/// edge. Everything outside the image counts as background (Leptonica's `L_BOUNDARY_BG`), and
/// distances saturate at 255. This is `pixDistanceFunction(pixs, 4, 8, L_BOUNDARY_BG)`.
///
/// Enforces `max_alloc` and the plane caps. It runs on the calling thread.
///
/// # Errors
///
/// `Error::Imaging` for a depth other than 1, `Error::PixelLimit` over a plane cap,
/// `Error::AllocLimit` over `max_alloc`, `Error::OutOfMemory` if the allocator refuses,
/// `Error::Cancelled` if `cancel` is set.
pub fn distance_function(
    b: &Bitmap,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<Bitmap, Error> {
    need_depth(b, 1, "DistanceFunction")?;
    let mut out = Bitmap::new(b.width, b.height, 8, limits)?;
    for y in 0..b.height {
        check_cancel(cancel)?;
        for x in 0..b.width {
            if b.at(x, y) != 0 {
                out.set(x, y, 1);
            }
        }
    }
    // The one-pixel border keeps the 1 it was seeded with, so both sweeps run over the
    // interior only.
    for y in 1..b.height.saturating_sub(1) {
        check_cancel(cancel)?;
        for x in 1..b.width.saturating_sub(1) {
            if out.at(x, y) == 0 {
                continue;
            }
            let m = usize::from(out.at(x, y - 1).min(out.at(x - 1, y)));
            out.set(x, y, (m.min(254) + 1) as u8);
        }
    }
    for y in (1..b.height.saturating_sub(1)).rev() {
        check_cancel(cancel)?;
        for x in (1..b.width.saturating_sub(1)).rev() {
            let v = usize::from(out.at(x, y));
            if v == 0 {
                continue;
            }
            let m = usize::from(out.at(x, y + 1).min(out.at(x + 1, y)));
            out.set(x, y, (m + 1).min(v) as u8);
        }
    }
    Ok(out)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Axis {
    X,
    Y,
}

fn check_brick(b: &Bitmap, w: usize, h: usize) -> Result<(), Error> {
    need_depth(b, 1, "morphology")?;
    if w < 1 || h < 1 {
        return Err(Error::Imaging(format!(
            "brick {w}x{h} must be at least 1x1"
        )));
    }
    Ok(())
}

/// `b` dilated by a solid `w x h` brick: a pixel of the result is foreground when the brick,
/// placed over it, covers any foreground pixel. The brick's origin sits at `(w/2, h/2)`.
/// Pixels outside the image are background (Leptonica's default asymmetric boundary).
///
/// Enforces `limits.max_alloc` (every bitmap this call holds is charged), `limits.max_threads` and
/// the plane caps (`max_pixels`, `max_width`, `max_height`).
///
/// # Errors
///
/// `Error::Imaging` for an argument that does not fit (depth other than 1, or a brick under 1x1),
/// `Error::PixelLimit` over a plane cap, `Error::AllocLimit` over `max_alloc`,
/// `Error::OutOfMemory` if the allocator refuses, `Error::Cancelled` if `cancel` is set (checked
/// per row).
pub fn dilate(
    b: &Bitmap,
    w: usize,
    h: usize,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<Bitmap, Error> {
    dilate_in(b, w, h, cancel, &Budget::new(limits))
}

/// `dilate`, charging every bitmap it holds at once (the pass-1 result and the output) to `budget`.
/// The pass-1 result is released; the output stays charged.
pub(crate) fn dilate_in(
    b: &Bitmap,
    w: usize,
    h: usize,
    cancel: &AtomicBool,
    budget: &Budget,
) -> Result<Bitmap, Error> {
    check_brick(b, w, h)?;
    let t = brick_pass(b, w - 1 - w / 2, w / 2, 1, Axis::X, cancel, budget)?;
    let out = brick_pass(&t, h - 1 - h / 2, h / 2, 1, Axis::Y, cancel, budget);
    t.free_in(budget);
    out
}

/// `b` eroded by a solid `w x h` brick: a pixel of the result is foreground only when the
/// brick, placed over it, covers foreground everywhere. Pixels outside the image count as
/// background.
///
/// Enforces `limits.max_alloc` (every bitmap this call holds is charged), `limits.max_threads` and
/// the plane caps (`max_pixels`, `max_width`, `max_height`).
///
/// # Errors
///
/// `Error::Imaging` for an argument that does not fit (depth other than 1, or a brick under 1x1),
/// `Error::PixelLimit` over a plane cap, `Error::AllocLimit` over `max_alloc`,
/// `Error::OutOfMemory` if the allocator refuses, `Error::Cancelled` if `cancel` is set (checked
/// per row).
pub fn erode(
    b: &Bitmap,
    w: usize,
    h: usize,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<Bitmap, Error> {
    erode_in(b, w, h, cancel, &Budget::new(limits))
}

/// `erode`, charging every bitmap it holds at once to `budget`. The pass-1 result is released;
/// the output stays charged.
pub(crate) fn erode_in(
    b: &Bitmap,
    w: usize,
    h: usize,
    cancel: &AtomicBool,
    budget: &Budget,
) -> Result<Bitmap, Error> {
    check_brick(b, w, h)?;
    let t = brick_pass(b, w / 2, w - 1 - w / 2, w, Axis::X, cancel, budget)?;
    let out = brick_pass(&t, h / 2, h - 1 - h / 2, h, Axis::Y, cancel, budget);
    t.free_in(budget);
    out
}

/// `b` eroded then dilated by a `w x h` brick.
///
/// Enforces `limits.max_alloc` (every bitmap this call holds is charged), `limits.max_threads` and
/// the plane caps (`max_pixels`, `max_width`, `max_height`).
///
/// # Errors
///
/// `Error::Imaging` for an argument that does not fit (depth other than 1, or a brick under 1x1),
/// `Error::PixelLimit` over a plane cap, `Error::AllocLimit` over `max_alloc`,
/// `Error::OutOfMemory` if the allocator refuses, `Error::Cancelled` if `cancel` is set (checked
/// per row).
pub fn open(
    b: &Bitmap,
    w: usize,
    h: usize,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<Bitmap, Error> {
    let budget = Budget::new(limits);
    let e = erode_in(b, w, h, cancel, &budget)?;
    let out = dilate_in(&e, w, h, cancel, &budget);
    e.free_in(&budget);
    out
}

/// `b` dilated then eroded by a `w x h` brick.
///
/// Enforces `limits.max_alloc` (every bitmap this call holds is charged), `limits.max_threads` and
/// the plane caps (`max_pixels`, `max_width`, `max_height`).
///
/// # Errors
///
/// `Error::Imaging` for an argument that does not fit (depth other than 1, or a brick under 1x1),
/// `Error::PixelLimit` over a plane cap, `Error::AllocLimit` over `max_alloc`,
/// `Error::OutOfMemory` if the allocator refuses, `Error::Cancelled` if `cancel` is set (checked
/// per row).
pub fn close(
    b: &Bitmap,
    w: usize,
    h: usize,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<Bitmap, Error> {
    let budget = Budget::new(limits);
    let d = dilate_in(b, w, h, cancel, &budget)?;
    let out = erode_in(&d, w, h, cancel, &budget);
    d.free_in(&budget);
    out
}

/// One axis of a separable brick operation. A pixel of the result is set when at least
/// `min_hits` foreground pixels lie in the window `[k-before, k+after]` along the axis. Dilation
/// passes 1; erosion passes the window size, and since positions outside the image are never
/// counted, a window overhanging an edge can never reach it.
///
/// The output rows are split into one band per thread; each band slides its windows over its own
/// rows only, starting from the window at its first row.
fn brick_pass(
    src: &Bitmap,
    before: usize,
    after: usize,
    min_hits: usize,
    along: Axis,
    cancel: &AtomicBool,
    budget: &Budget,
) -> Result<Bitmap, Error> {
    let mut out = Bitmap::new_in(src.width, src.height, 1, budget)?;
    let stride = out.stride;
    let bands = par::band_count(budget, src.width * src.height, 0, 0);
    let per_band = src.height.div_ceil(bands).max(1);
    par::for_each_chunk_mut(
        budget,
        cancel,
        &mut out.pix,
        per_band * stride,
        |b, rows| {
            let first = b * per_band;
            let nrows = rows.len() / stride;
            let (n, lines, ks) = match along {
                Axis::X => (src.width, first..first + nrows, 0..src.width),
                Axis::Y => (src.height, 0..src.width, first..first + nrows),
            };
            match along {
                Axis::X => slide(
                    (lines, ks, n),
                    (before, after, min_hits),
                    cancel,
                    |line, k| src.at(k, line),
                    |line, k| rows[(line - first) * stride + (k >> 3)] |= 0x80 >> (k & 7),
                ),
                Axis::Y => slide(
                    (lines, ks, n),
                    (before, after, min_hits),
                    cancel,
                    |line, k| src.at(line, k),
                    |line, k| rows[(k - first) * stride + (line >> 3)] |= 0x80 >> (line & 7),
                ),
            }
        },
    )?;
    Ok(out)
}

/// The sliding window of `brick_pass` along one axis, over the `lines` and the positions `ks`
/// of a band (`n` positions in all): `get(line, k)` reads a source pixel and `put(line, k)` sets
/// a result pixel.
fn slide(
    (lines, ks, n): (std::ops::Range<usize>, std::ops::Range<usize>, usize),
    (before, after, min_hits): (usize, usize, usize),
    cancel: &AtomicBool,
    get: impl Fn(usize, usize) -> u8,
    mut put: impl FnMut(usize, usize),
) -> Result<(), Error> {
    for line in lines {
        check_cancel(cancel)?;
        // The window at the band's first position.
        let mut count = 0usize;
        for k in ks.start.saturating_sub(before)..=(ks.start + after).min(n - 1) {
            if get(line, k) != 0 {
                count += 1;
            }
        }
        for k in ks.clone() {
            if count >= min_hits {
                put(line, k);
            }
            if k >= before && get(line, k - before) != 0 {
                count -= 1;
            }
            let add = k + after + 1;
            if add < n && get(line, add) != 0 {
                count += 1;
            }
        }
    }
    Ok(())
}
