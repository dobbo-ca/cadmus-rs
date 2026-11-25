//! A uniform-cell spatial index over bounding boxes (`grid.go`).

use super::Rect;
use crate::{Budget, Error, Limits};

/// A uniform-cell spatial index over bounding boxes: boxes are filed under the cells they
/// cover, so a query only examines the boxes filed under the cells it touches. Boxes reaching
/// outside `bounds` are clamped into the edge cells, so no intersection is lost.
#[derive(Debug)]
pub struct Grid {
    bounds: Rect,
    cell_size: usize,
    cols: usize,
    rows: usize,
    ids: Vec<i64>,
    rects: Vec<Rect>,
    cells: Vec<Vec<usize>>,
    budget: Budget,
}

impl Grid {
    /// An empty grid covering `bounds` with square cells `cell_size` across. The cells and
    /// everything later inserted are charged against `limits.max_alloc`.
    ///
    /// Enforces `limits.max_alloc`: the grid keeps its own `Budget` made from `limits`. No plane
    /// caps apply and it spawns no thread.
    ///
    /// # Errors
    ///
    /// `Error::Imaging` for a zero `cell_size` or empty `bounds`, `Error::AllocLimit` if the cells
    /// exceed `limits.max_alloc`, `Error::OutOfMemory` if the allocator refuses.
    pub fn new(bounds: Rect, cell_size: usize, limits: &Limits) -> Result<Grid, Error> {
        if cell_size == 0 {
            return Err(Error::Imaging(
                "NewGrid needs a positive cell size, got 0".into(),
            ));
        }
        if bounds.is_empty() {
            return Err(Error::Imaging(format!(
                "NewGrid needs non-empty bounds, got {bounds:?}"
            )));
        }
        let dx = usize::try_from(bounds.dx()).map_err(|_| Error::AllocLimit)?;
        let dy = usize::try_from(bounds.dy()).map_err(|_| Error::AllocLimit)?;
        let cols = dx.div_ceil(cell_size);
        let rows = dy.div_ceil(cell_size);
        let n = cols.checked_mul(rows).ok_or(Error::AllocLimit)?;
        let budget = Budget::new(limits);
        let cells = {
            budget.charge(
                n.checked_mul(std::mem::size_of::<Vec<usize>>())
                    .ok_or(Error::AllocLimit)?,
            )?;
            let mut v: Vec<Vec<usize>> = Vec::new();
            v.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
            v.resize_with(n, Vec::new);
            v
        };
        Ok(Grid {
            bounds,
            cell_size,
            cols,
            rows,
            ids: Vec::new(),
            rects: Vec::new(),
            cells,
            budget,
        })
    }

    /// Files box `r` under `id`. Ids are the caller's labels: they need not be unique, dense or
    /// ordered. An empty `r` covers no cells and can never be returned.
    ///
    /// Enforces the `max_alloc` the grid was made with.
    ///
    /// # Errors
    ///
    /// `Error::AllocLimit` if the grown index would exceed that budget (the grid is unchanged),
    /// `Error::OutOfMemory` if the allocator refuses.
    pub fn insert(&mut self, id: i64, r: Rect) -> Result<(), Error> {
        let e = self.ids.len();
        let cover = self.cell_range(r);
        // Every vec is given its room before anything is pushed, so a refusal leaves the grid as it
        // was. Each growth is charged through the ledger, old block included.
        room(&self.budget, &mut self.ids)?;
        room(&self.budget, &mut self.rects)?;
        if let Some((c0, c1, r0, r1)) = cover {
            for row in r0..=r1 {
                for col in c0..=c1 {
                    room(&self.budget, &mut self.cells[row * self.cols + col])?;
                }
            }
        }
        self.ids.push(id);
        self.rects.push(r);
        if let Some((c0, c1, r0, r1)) = cover {
            for row in r0..=r1 {
                for col in c0..=c1 {
                    self.cells[row * self.cols + col].push(e);
                }
            }
        }
        Ok(())
    }

    /// The ids of every box intersecting `r`, ascending with duplicates removed. Intersection is
    /// exact and half-open: a box that merely abuts `r` does not match.
    ///
    /// Enforces the `max_alloc` the grid was made with: the per-entry flags and the result are
    /// charged to it while the query runs, and given back when it returns (the returned vec is the
    /// caller's).
    ///
    /// # Errors
    ///
    /// `Error::AllocLimit` if the flags and the result do not fit in what is left of that budget,
    /// `Error::OutOfMemory` if the allocator refuses.
    pub fn query(&self, r: Rect) -> Result<Vec<i64>, Error> {
        let mut out = Vec::new();
        let found = self.query_into(r, &mut out);
        self.budget
            .release(out.capacity() * std::mem::size_of::<i64>());
        found?;
        out.sort_unstable();
        out.dedup();
        Ok(out)
    }

    fn query_into(&self, r: Rect, out: &mut Vec<i64>) -> Result<(), Error> {
        let Some((c0, c1, r0, r1)) = self.cell_range(r) else {
            return Ok(());
        };
        // One flag per entry, so a box that covers many cells is pushed once and `out` never
        // outgrows the entries `insert` charged for.
        let mut seen = self.budget.buf(self.ids.len(), false)?;
        for row in r0..=r1 {
            for col in c0..=c1 {
                for &e in &self.cells[row * self.cols + col] {
                    if !seen[e] && self.rects[e].overlaps(&r) {
                        seen[e] = true;
                        self.budget.push(out, self.ids[e])?;
                    }
                }
            }
        }
        Ok(())
    }

    /// The covered cells `(col0, col1, row0, row1)` after clamping `r` to the grid, or `None`
    /// for an empty `r`.
    fn cell_range(&self, r: Rect) -> Option<(usize, usize, usize, usize)> {
        if r.is_empty() {
            return None;
        }
        // Max is exclusive, so the last covered cell is the one holding Max-1.
        // The subtractions saturate: the clamp pins any offset that far out to an edge cell anyway,
        // and `r` is not empty, so `max - 1` cannot underflow.
        let off = |v: isize, origin: isize| v.saturating_sub(origin);
        let col0 = clamp_index(off(r.min.x, self.bounds.min.x), self.cell_size, self.cols);
        let col1 = clamp_index(
            off(r.max.x - 1, self.bounds.min.x),
            self.cell_size,
            self.cols,
        );
        let row0 = clamp_index(off(r.min.y, self.bounds.min.y), self.cell_size, self.rows);
        let row1 = clamp_index(
            off(r.max.y - 1, self.bounds.min.y),
            self.cell_size,
            self.rows,
        );
        Some((col0, col1, row0, row1))
    }
}

/// Makes room for one more element in `v`, charging the growth (and, while it runs, the old
/// block) to `budget`. The charge for the capacity stays until the grid is dropped.
fn room<T>(budget: &Budget, v: &mut Vec<T>) -> Result<(), Error> {
    if v.len() == v.capacity() {
        budget.reserve(v, v.capacity().max(4))?;
    }
    Ok(())
}

/// Maps an offset from the grid origin to a cell index in `[0, n)`, pinning offsets outside the
/// grid to the nearest edge cell.
fn clamp_index(offset: isize, cell_size: usize, n: usize) -> usize {
    if offset < 0 {
        return 0;
    }
    (offset as usize / cell_size).min(n - 1)
}
