//! The tensor runtime of Tesseract's LSTM engine (float path, inference only): a port of the
//! `src/lstm` classes of Tesseract OCR (Apache-2.0, see NOTICE).
//!
//! Every float operation is the scalar f64 one of Tesseract's reference build. Rust never fuses
//! `a * b + c`, so the products need no wrapping as they do in Go on arm64.

mod funcs;
mod layers;
#[allow(clippy::approx_constant, clippy::excessive_precision)]
mod tables;

pub(crate) use funcs::{go_exp, go_log};
pub use funcs::{logistic, logistic_in_place, softmax_in_place, tanh, tanh_in_place};
pub use layers::{
    Activation, Convolve, Ctx, FullyConnected, Input, Layer, Lstm, Parallel, Reconfig,
    ReversalKind, Reversed, Series, Tap,
};

use crate::pool::SlotPool;
use crate::{Budget, Error, Limits};
use std::sync::atomic::{AtomicU64, Ordering};

/// The buffers a forward pass takes its tensors and scratch vectors from when a caller reuses
/// them (`Ctx::pool`), and the scaled image the line normalization makes. It holds no state
/// between calls but its capacity; `Workspace` owns one.
#[derive(Debug, Default)]
pub struct Scratch {
    pub(crate) f32s: SlotPool<f32>,
    pub(crate) f64s: SlotPool<f64>,
    pub(crate) u8s: SlotPool<u8>,
}

impl Scratch {
    /// Marks every buffer free, as at the start of a call.
    pub(crate) fn reset(&mut self) {
        self.f32s.reset();
        self.f64s.reset();
        self.u8s.reset();
    }

    /// The heap the buffers hold, by capacity.
    pub(crate) fn bytes(&self) -> usize {
        self.f32s.bytes() + self.f64s.bytes() + self.u8s.bytes()
    }
}

/// Tesseract's `TRand`: a linear congruential generator with Knuth's constants, used to fill
/// the off-image taps of a convolution.
///
/// It is not a general-purpose RNG. The draw sequence is part of the network's output:
/// `Convolve` consumes it in a position-dependent order at every image border.
#[derive(Debug)]
pub struct Rand {
    seed: AtomicU64,
}

impl Rand {
    /// Seeds the generator and discards one iterate, as `LSTMRecognizer::SetRandomSeed` does
    /// (`set_seed` then `IntRand`). The caller passes `sample_iteration * 0x10000001`.
    pub fn new(seed: u64) -> Rand {
        let r = Rand {
            seed: AtomicU64::new(seed),
        };
        r.int_rand();
        r
    }

    /// The generator state, to restore with `set_seed`.
    pub(crate) fn seed(&self) -> u64 {
        self.seed.load(Ordering::Relaxed)
    }

    /// Puts back a state taken with `seed`.
    pub(crate) fn set_seed(&self, s: u64) {
        self.seed.store(s, Ordering::Relaxed);
    }

    /// Steps the generator and returns a value in `[0, i32::MAX]`.
    pub fn int_rand(&self) -> i32 {
        let s = self
            .seed
            .load(Ordering::Relaxed)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.seed.store(s, Ordering::Relaxed);
        (s >> 33) as i32
    }

    /// A value in `[-rng, rng]`, closed at both ends (`int_rand`'s maximum is exactly
    /// `i32::MAX`, so the division is exactly 1 at the top of the range).
    pub fn signed_rand(&self, rng: f64) -> f64 {
        rng * 2.0 * f64::from(self.int_rand()) / 2147483647.0 - rng
    }
}

/// One of Tesseract's `WeightMatrix` objects on the float path: an `outputs x (inputs+1)`
/// row-major array of f64 weights whose last column is the bias, applied against an implicit 1.
#[derive(Debug, Clone)]
pub struct Matrix {
    pub outputs: usize,
    pub inputs: usize,
    pub w: Vec<f64>,
}

impl Matrix {
    /// A matrix over the weights `w`, which the caller has already allocated. It enforces
    /// `limits.max_alloc` against those weights alone, allocates nothing and spawns no thread.
    ///
    /// # Errors
    ///
    /// `Error::AllocLimit` if the weights are over `limits.max_alloc`, `Error::Network` if
    /// `outputs` is 0, the shape overflows, or the count of weights is not
    /// `outputs * (inputs + 1)`.
    pub fn new(
        outputs: usize,
        inputs: usize,
        w: Vec<f64>,
        limits: &Limits,
    ) -> Result<Matrix, Error> {
        if w.len().saturating_mul(std::mem::size_of::<f64>()) > limits.max_alloc {
            return Err(Error::AllocLimit);
        }
        if outputs == 0 {
            return Err(Error::Network(format!(
                "invalid matrix shape {outputs}x{}",
                inputs.wrapping_add(1)
            )));
        }
        let want = inputs
            .checked_add(1)
            .and_then(|c| c.checked_mul(outputs))
            .ok_or_else(|| Error::Network("matrix shape overflows".into()))?;
        if w.len() != want {
            return Err(Error::Network(format!(
                "matrix {outputs}x{} needs {want} weights, got {}",
                inputs + 1,
                w.len()
            )));
        }
        Ok(Matrix { outputs, inputs, w })
    }

    /// Whether the public `outputs` and `inputs` still describe `w`.
    pub fn is_consistent(&self) -> bool {
        self.inputs
            .checked_add(1)
            .and_then(|c| c.checked_mul(self.outputs))
            == Some(self.w.len())
    }

    /// `WeightMatrix::MatrixDotVector`: `v[i] = sum_j W[i][j]*u[j]` plus the row's bias, added
    /// after the whole dot product exactly as `MatrixDotVectorInternal` does. The first
    /// `min(outputs, v.len())` entries of `v` are written.
    ///
    /// # Errors
    ///
    /// `Error::Network` if the matrix's public fields disagree with its weights or `u` has fewer
    /// than `inputs` entries.
    pub fn dot_vector(&self, u: &[f64], v: &mut [f64]) -> Result<(), Error> {
        if !self.is_consistent() || u.len() < self.inputs {
            return Err(Error::Network(format!(
                "dot product of a {}x{} matrix with {} weights and a vector of {}",
                self.outputs,
                self.inputs.saturating_add(1),
                self.w.len(),
                u.len()
            )));
        }
        let stride = self.inputs + 1;
        for (i, out) in v.iter_mut().enumerate().take(self.outputs) {
            let row = &self.w[i * stride..(i + 1) * stride];
            let mut total = 0.0f64;
            for j in 0..self.inputs {
                total += row[j] * u[j];
            }
            *out = total + row[self.inputs];
        }
        Ok(())
    }
}

/// Maps a 2-D `(y, x)` position of a feature map to the flat timestep index of a [`Tensor`].
///
/// Tesseract's `StrideMap` has a batch dimension; cadmus recognizes one line image at a time,
/// so it is always 1 and elided. Raster order: y outer, x inner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrideMap {
    pub height: usize,
    pub width: usize,
}

impl StrideMap {
    pub fn len(&self) -> usize {
        self.height.saturating_mul(self.width)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The timestep index of `(y, x)`.
    pub fn t(&self, y: usize, x: usize) -> usize {
        y.saturating_mul(self.width).saturating_add(x)
    }

    /// Inverts [`StrideMap::t`]; `None` for a map of width 0.
    pub fn yx(&self, t: usize) -> Option<(usize, usize)> {
        Some((t.checked_div(self.width)?, t % self.width))
    }

    /// The timestep `dy` rows and `dx` columns from `t`, if that position is inside the map
    /// (`StrideMap::Index::AddOffset` on height and width in turn).
    pub fn offset(&self, t: usize, dy: isize, dx: isize) -> Option<usize> {
        let (y, x) = self.yx(t)?;
        let y = y as isize + dy;
        let x = x as isize + dx;
        if y < 0 || y >= self.height as isize || x < 0 || x >= self.width as isize {
            return None;
        }
        Some(self.t(y as usize, x as usize))
    }

    /// `StrideMap::ScaleXY`: both dimensions are integer-divided, so a partial trailing
    /// pooling window is dropped.
    pub fn scale_xy(&self, x_factor: usize, y_factor: usize) -> StrideMap {
        StrideMap {
            height: self.height.checked_div(y_factor).unwrap_or(0),
            width: self.width.checked_div(x_factor).unwrap_or(0),
        }
    }

    pub fn transpose_xy(&self) -> StrideMap {
        StrideMap {
            height: self.width,
            width: self.height,
        }
    }

    /// `StrideMap::ReduceWidthTo1`, used by `NT_LSTM_SUMMARY`.
    pub fn reduce_width_to_1(&self) -> StrideMap {
        StrideMap {
            height: self.height,
            width: 1,
        }
    }
}

impl std::fmt::Display for StrideMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.height, self.width)
    }
}

/// Tesseract's `NetworkIO` on the float path: a `height x width` grid of `features`-long
/// activation vectors addressed by flat timestep.
///
/// The backing store is f32 even though every arithmetic register is f64: `NetworkIO` stores
/// `float`, writes narrow and reads widen, and the LSTM's recurrent output passes through one
/// on every timestep, so the narrowing is part of the result.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub map: StrideMap,
    pub features: usize,
    data: Vec<f32>,
}

impl Tensor {
    /// A zeroed tensor, charged against `budget` before it is allocated.
    ///
    /// # Errors
    ///
    /// `Error::AllocLimit` over the budget's `max_alloc` (or on overflow), `Error::OutOfMemory` if
    /// the allocator refuses.
    pub fn new(map: StrideMap, features: usize, budget: &Budget) -> Result<Tensor, Error> {
        let data = budget.vec(Tensor::len_of(map, features)?, 0.0f32)?;
        Ok(Tensor {
            map,
            features,
            data,
        })
    }

    fn len_of(map: StrideMap, features: usize) -> Result<usize, Error> {
        map.height
            .checked_mul(map.width)
            .and_then(|n| n.checked_mul(features))
            .ok_or(Error::AllocLimit)
    }

    /// `new`, in a buffer of `pool`; only the growth of the pool's capacity is charged.
    pub(crate) fn pooled(
        map: StrideMap,
        features: usize,
        pool: &mut SlotPool<f32>,
        budget: &Budget,
    ) -> Result<Tensor, Error> {
        let data = pool.take(budget, Tensor::len_of(map, features)?, 0.0f32)?;
        Ok(Tensor {
            map,
            features,
            data,
        })
    }

    /// The backing store, to give back to the pool it came from.
    pub(crate) fn into_data(self) -> Vec<f32> {
        self.data
    }

    /// Copies `src`'s values, which has the same shape.
    pub(crate) fn copy_data_from(&mut self, src: &Tensor) {
        if self.data.len() == src.data.len() {
            self.data.copy_from_slice(&src.data);
        }
    }

    /// Whether `map` and `features`, which are public, still describe the backing store.
    pub fn is_consistent(&self) -> bool {
        self.map
            .height
            .checked_mul(self.map.width)
            .and_then(|n| n.checked_mul(self.features))
            == Some(self.data.len())
    }

    /// Bytes of the backing store, for `Budget::release`.
    pub fn bytes(&self) -> usize {
        self.data.len() * 4
    }

    /// The activation vector at timestep `t`; empty if `t` is outside the data.
    pub fn row(&self, t: usize) -> &[f32] {
        self.row_range(t)
            .and_then(|r| self.data.get(r))
            .unwrap_or_default()
    }

    /// `row`, mutable.
    pub fn row_mut(&mut self, t: usize) -> &mut [f32] {
        match self.row_range(t) {
            Some(r) => self.data.get_mut(r).unwrap_or_default(),
            None => Default::default(),
        }
    }

    fn row_range(&self, t: usize) -> Option<std::ops::Range<usize>> {
        let start = t.checked_mul(self.features)?;
        Some(start..start.checked_add(self.features)?)
    }

    /// Widens timestep `t` into `dst`, which has length `features`.
    pub fn read_time_step(&self, t: usize, dst: &mut [f64]) {
        for (d, v) in dst.iter_mut().zip(self.row(t)) {
            *d = f64::from(*v);
        }
    }

    /// Narrows the first `features` values of `src` into timestep `t`.
    pub fn write_time_step(&mut self, t: usize, src: &[f64]) {
        for (d, v) in self.row_mut(t).iter_mut().zip(src) {
            *d = *v as f32;
        }
    }

    /// Narrows the first `n` values of `src` into timestep `t` starting at feature `offset`.
    ///
    /// # Errors
    ///
    /// `Error::Network` if the timestep or the feature range is outside the tensor, or `src` has
    /// fewer than `n` values.
    pub fn write_time_step_part(
        &mut self,
        t: usize,
        offset: usize,
        n: usize,
        src: &[f64],
    ) -> Result<(), Error> {
        let Some(end) = offset.checked_add(n) else {
            return Err(shape_error("write_time_step_part", t));
        };
        let Some(part) = self.row_mut(t).get_mut(offset..end) else {
            return Err(shape_error("write_time_step_part", t));
        };
        if src.len() < n {
            return Err(shape_error("write_time_step_part", t));
        }
        for (d, v) in part.iter_mut().zip(src) {
            *d = *v as f32;
        }
        Ok(())
    }

    /// Copies a whole timestep from `src`.
    ///
    /// # Errors
    ///
    /// `Error::Network` if either timestep is outside its tensor or the feature counts differ.
    pub fn copy_time_step(&mut self, t: usize, src: &Tensor, src_t: usize) -> Result<(), Error> {
        let (dst, from) = (self.row_mut(t), src.row(src_t));
        if dst.len() != from.len() || dst.len() != src.features {
            return Err(shape_error("copy_time_step", t));
        }
        dst.copy_from_slice(from);
        Ok(())
    }

    /// `NetworkIO::CopyTimeStepGeneral`: copies `n` features from `src`'s timestep `src_t`
    /// starting at `src_offset` into timestep `t` starting at `offset`.
    ///
    /// # Errors
    ///
    /// `Error::Network` if either range is outside its tensor.
    pub fn copy_time_step_part(
        &mut self,
        t: usize,
        offset: usize,
        n: usize,
        src: &Tensor,
        src_t: usize,
        src_offset: usize,
    ) -> Result<(), Error> {
        let (Some(end), Some(src_end)) = (offset.checked_add(n), src_offset.checked_add(n)) else {
            return Err(shape_error("copy_time_step_part", t));
        };
        let Some(dst) = self.row_mut(t).get_mut(offset..end) else {
            return Err(shape_error("copy_time_step_part", t));
        };
        let Some(from) = src.row(src_t).get(src_offset..src_end) else {
            return Err(shape_error("copy_time_step_part", t));
        };
        dst.copy_from_slice(from);
        Ok(())
    }

    /// The elementwise maximum of timestep `t` and `src`'s timestep `src_t`, in place.
    pub fn maxpool_time_step(&mut self, t: usize, src: &Tensor, src_t: usize) {
        for (d, v) in self.row_mut(t).iter_mut().zip(src.row(src_t)) {
            if *d < *v {
                *d = *v;
            }
        }
    }
}

fn shape_error(what: &str, t: usize) -> Error {
    Error::Network(format!(
        "{what}: timestep {t} or a feature range is out of bounds"
    ))
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) fn go_log_for_test(x: f64) -> f64 {
    funcs::go_log(x)
}
