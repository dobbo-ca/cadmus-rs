//! The layers of a runnable network graph: `network.cpp`, `plumbing.cpp`, `series.cpp`,
//! `parallel.cpp`, `reversed.cpp`, `input.cpp`, `convolve.cpp`, `reconfig.cpp`, `maxpool.cpp`,
//! `fullyconnected.cpp` and `lstm.cpp` of Tesseract OCR (Apache-2.0, see NOTICE).

use super::{
    Matrix, Rand, Scratch, StrideMap, Tensor, logistic_in_place, softmax_in_place, tanh,
    tanh_in_place,
};
use crate::imaging::check_cancel;
use crate::tessdata::short;
use crate::{Budget, Error, par};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// An observer of layer outputs.
pub type Tap<'a> = &'a mut dyn FnMut(&Layer, &Tensor);

/// What a `forward` call needs: the memory budget, and an optional observer that sees every
/// layer's output (except `Input`'s) as the graph runs. The observer exists so a debugging
/// harness can diff activations against Tesseract's layer by layer.
pub struct Ctx<'a> {
    pub budget: &'a Budget,
    pub tap: Option<Tap<'a>>,
    /// Checked before each layer and at each timestep (or row) of the LSTM, fully connected,
    /// convolve and reconfig layers.
    pub cancel: Option<&'a AtomicBool>,
    /// Where the layers take their output tensors and scratch vectors from and give them back to,
    /// so that a pass on a warm pool allocates nothing; only the growth of its capacity is charged
    /// to `budget`. `None` allocates and frees (charges and releases) every one.
    /// The capacity the pool already holds is not charged here: a caller that keeps a `Scratch`
    /// across calls with a new `Budget` each time must account for it itself (`Workspace` does).
    pub pool: Option<&'a mut Scratch>,
}

impl<'a> Ctx<'a> {
    /// A context charging to `budget`, with no tap and no cancel flag.
    pub fn new(budget: &'a Budget) -> Ctx<'a> {
        Ctx {
            budget,
            tap: None,
            cancel: None,
            pool: None,
        }
    }

    fn check_cancel(&self) -> Result<(), Error> {
        match self.cancel {
            Some(c) if c.load(Ordering::Relaxed) => Err(Error::Cancelled),
            _ => Ok(()),
        }
    }

    /// A zeroed tensor, from the pool if there is one.
    fn tensor(&mut self, map: StrideMap, features: usize) -> Result<Tensor, Error> {
        match self.pool.as_deref_mut() {
            Some(p) => Tensor::pooled(map, features, &mut p.f32s, self.budget),
            None => Tensor::new(map, features, self.budget),
        }
    }

    /// Gives `t` back to the pool, or frees it and returns its bytes to the budget.
    fn free(&mut self, t: Tensor) {
        match self.pool.as_deref_mut() {
            Some(p) => p.f32s.give(t.into_data()),
            None => free(self.budget, t),
        }
    }

    /// `n` zeros, from the pool if there is one.
    fn f64s(&mut self, n: usize) -> Result<Vec<f64>, Error> {
        match self.pool.as_deref_mut() {
            Some(p) => p.f64s.take(self.budget, n, 0.0),
            None => self.budget.vec(n, 0.0f64),
        }
    }

    /// Gives `v` back to the pool, or frees it and returns its bytes to the budget.
    fn free_f64(&mut self, v: Vec<f64>) {
        match self.pool.as_deref_mut() {
            Some(p) => p.f64s.give(v),
            None => {
                let bytes = v.len() * std::mem::size_of::<f64>();
                drop(v);
                self.budget.release(bytes);
            }
        }
    }
}

/// Frees `t` and returns its bytes to `budget`.
fn free(budget: &Budget, t: Tensor) {
    let bytes = t.bytes();
    drop(t);
    budget.release(bytes);
}

/// One node of a runnable network graph. `forward` returns a freshly allocated tensor.
#[derive(Debug)]
pub enum Layer {
    Input(Input),
    Series(Series),
    Parallel(Parallel),
    Reversed(Reversed),
    Reconfig(Reconfig),
    Convolve(Convolve),
    FullyConnected(FullyConnected),
    Lstm(Lstm),
}

impl Layer {
    /// The layer's serialized name. Activation dumps key on it.
    pub fn name(&self) -> &str {
        match self {
            Layer::Input(l) => &l.name,
            Layer::Series(l) => &l.name,
            Layer::Parallel(l) => &l.name,
            Layer::Reversed(l) => &l.name,
            Layer::Reconfig(l) => &l.name,
            Layer::Convolve(l) => &l.name,
            Layer::FullyConnected(l) => &l.name,
            Layer::Lstm(l) => &l.name,
        }
    }

    /// The feature count of the tensor `forward` produces.
    pub fn num_outputs(&self) -> usize {
        match self {
            Layer::Input(l) => l.features,
            Layer::Series(l) => l.stack.last().map_or(0, Layer::num_outputs),
            Layer::Parallel(l) => l
                .stack
                .iter()
                .fold(0usize, |n, c| n.saturating_add(c.num_outputs())),
            Layer::Reversed(l) => l.sub.num_outputs(),
            Layer::Reconfig(l) => l.num_outputs(),
            Layer::Convolve(l) => l.num_outputs(),
            Layer::FullyConnected(l) => l.w.outputs,
            Layer::Lstm(l) => l.ns,
        }
    }

    /// Propagates `input` and returns the layer's output.
    ///
    /// Enforces `max_alloc` through `cx.budget` (outputs and scratch are charged) and
    /// `max_threads` for the layers that split their rows; `cx.cancel` is checked per layer and
    /// per timestep.
    ///
    /// # Errors
    ///
    /// `Error::Network` for an input that does not fit the layer, `Error::AllocLimit` over
    /// `max_alloc`, `Error::OutOfMemory` if the allocator refuses, `Error::Cancelled` if
    /// `cx.cancel` is set.
    pub fn forward(&self, input: &Tensor, cx: &mut Ctx) -> Result<Tensor, Error> {
        cx.check_cancel()?;
        if !input.is_consistent() {
            return Err(Error::Network(format!(
                "{} got a tensor whose map and features disagree with its data",
                short(self.name())
            )));
        }
        let out = match self {
            Layer::Input(l) => return l.forward(input, cx),
            Layer::Series(l) => l.forward(input, cx)?,
            Layer::Parallel(l) => l.forward(input, cx)?,
            Layer::Reversed(l) => l.forward(input, cx)?,
            Layer::Reconfig(l) => l.forward(input, cx)?,
            Layer::Convolve(l) => l.forward(input, cx)?,
            Layer::FullyConnected(l) => l.forward(input, cx)?,
            Layer::Lstm(l) => l.forward(input, cx)?,
        };
        if let Some(tap) = cx.tap.as_deref_mut() {
            tap(self, &out);
        }
        Ok(out)
    }
}

/// `NT_INPUT`. `Input::Forward` is `*output = input`.
#[derive(Debug)]
pub struct Input {
    name: String,
    features: usize,
}

impl Input {
    /// An input layer of `features` values per timestep. Copies `name`, which is the caller's to size, and takes no `Limits`.
    pub fn new(name: &str, features: usize) -> Input {
        Input {
            name: name.to_string(),
            features,
        }
    }

    fn forward(&self, input: &Tensor, cx: &mut Ctx) -> Result<Tensor, Error> {
        let mut out = cx.tensor(input.map, input.features)?;
        out.copy_data_from(input);
        Ok(out)
    }
}

/// `NT_SERIES`: each layer's output is the next layer's input.
#[derive(Debug)]
pub struct Series {
    name: String,
    pub stack: Vec<Layer>,
}

impl Series {
    /// A series of `stack`, each layer's output the next one's input. It copies `name`, which
    /// is the caller's to size, and takes no `Limits`.
    ///
    /// # Errors
    ///
    /// `Error::Network` if `stack` is empty.
    pub fn new(name: &str, stack: Vec<Layer>) -> Result<Series, Error> {
        if stack.is_empty() {
            return Err(Error::Network(format!(
                "series {} has no layers",
                short(name)
            )));
        }
        Ok(Series {
            name: name.to_string(),
            stack,
        })
    }

    fn forward(&self, input: &Tensor, cx: &mut Ctx) -> Result<Tensor, Error> {
        let mut cur: Option<Tensor> = None;
        for sub in &self.stack {
            let out = sub
                .forward(cur.as_ref().unwrap_or(input), cx)
                .map_err(|e| prefix(&self.name, e))?;
            if let Some(old) = cur.replace(out) {
                cx.free(old);
            }
        }
        cur.ok_or_else(|| Error::Network("empty series".into()))
    }
}

/// Adds the enclosing layer's name to a network error, as Go's `fmt.Errorf("%s: %w")`.
fn prefix(name: &str, e: Error) -> Error {
    match e {
        Error::Network(s) => Error::Network(format!("{}: {s}", short(name))),
        e => e,
    }
}

/// `NT_PARALLEL` and `NT_REPLICATED`: every child sees the same input and their outputs are
/// concatenated along the feature axis, in stack order.
#[derive(Debug)]
pub struct Parallel {
    name: String,
    pub stack: Vec<Layer>,
}

impl Parallel {
    /// A parallel layer over `stack`: every child sees the same input and the outputs are
    /// concatenated. It copies `name`, which is the caller's to size, and takes no
    /// `Limits`.
    ///
    /// # Errors
    ///
    /// `Error::Network` if `stack` is empty.
    pub fn new(name: &str, stack: Vec<Layer>) -> Result<Parallel, Error> {
        if stack.is_empty() {
            return Err(Error::Network(format!(
                "parallel {} has no layers",
                short(name)
            )));
        }
        Ok(Parallel {
            name: name.to_string(),
            stack,
        })
    }

    fn forward(&self, input: &Tensor, cx: &mut Ctx) -> Result<Tensor, Error> {
        let n = self
            .stack
            .iter()
            .try_fold(0usize, |n, c| n.checked_add(c.num_outputs()))
            .ok_or(Error::AllocLimit)?;
        // Each child's output is copied into the result as it is made, so only one is live
        // beside it.
        let mut out: Option<Tensor> = None;
        let mut off = 0;
        for (i, sub) in self.stack.iter().enumerate() {
            let part = sub.forward(input, cx).map_err(|e| prefix(&self.name, e))?;
            // An `Input` child returns its input whatever feature count its header declared.
            if part.features != sub.num_outputs() {
                return Err(Error::Network(format!(
                    "parallel {} child {i} produced {} features; want {}",
                    short(&self.name),
                    part.features,
                    sub.num_outputs()
                )));
            }
            if out.is_none() {
                out = Some(cx.tensor(part.map, n)?);
            }
            if let Some(o) = out.as_mut() {
                if part.map != o.map {
                    return Err(Error::Network(format!(
                        "parallel {} child {i} produced map {}; want {}",
                        short(&self.name),
                        part.map,
                        o.map
                    )));
                }
                for t in 0..o.map.len() {
                    o.copy_time_step_part(t, off, part.features, &part, t, 0)?;
                }
            }
            off += part.features;
            cx.free(part);
        }
        out.ok_or_else(|| Error::Network(format!("parallel {} has no layers", short(&self.name))))
    }
}

/// Which of `Reversed`'s three transforms applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReversalKind {
    /// `NT_XREVERSED`: mirror x within each row.
    ReverseX,
    /// `NT_YREVERSED`: mirror y within each column.
    ReverseY,
    /// `NT_XYTRANSPOSE`: swap the two spatial dimensions.
    TransposeXY,
}

/// Tesseract's `Reversed` plumbing: it applies its transform to the input, runs its single
/// child, and applies the same transform to the child's output.
#[derive(Debug)]
pub struct Reversed {
    name: String,
    pub kind: ReversalKind,
    pub sub: Box<Layer>,
}

impl Reversed {
    /// `sub` run on a reversed copy of its input. Copies `name`, which is the caller's to size, and takes no `Limits`.
    pub fn new(name: &str, kind: ReversalKind, sub: Layer) -> Reversed {
        Reversed {
            name: name.to_string(),
            kind,
            sub: Box::new(sub),
        }
    }

    fn forward(&self, input: &Tensor, cx: &mut Ctx) -> Result<Tensor, Error> {
        let rev = reverse_data(input, self.kind, cx)?;
        let out = self
            .sub
            .forward(&rev, cx)
            .map_err(|e| prefix(&self.name, e));
        cx.free(rev);
        let out = out?;
        let back = reverse_data(&out, self.kind, cx);
        cx.free(out);
        back
    }
}

/// `NetworkIO::CopyWithXReversal`, `CopyWithYReversal` and `CopyWithXYTranspose`.
fn reverse_data(src: &Tensor, kind: ReversalKind, cx: &mut Ctx) -> Result<Tensor, Error> {
    let mut m = src.map;
    if kind == ReversalKind::TransposeXY {
        m = m.transpose_xy();
    }
    let mut dst = cx.tensor(m, src.features)?;
    for y in 0..src.map.height {
        for x in 0..src.map.width {
            let dt = match kind {
                ReversalKind::ReverseX => m.t(y, src.map.width - 1 - x),
                ReversalKind::ReverseY => m.t(src.map.height - 1 - y, x),
                ReversalKind::TransposeXY => m.t(x, y),
            };
            dst.copy_time_step(dt, src, src.map.t(y, x))?;
        }
    }
    Ok(dst)
}

/// `NT_RECONFIG` and, with `max` set, `NT_MAXPOOL`. Maxpool takes the elementwise maximum over
/// the window instead of concatenating it, and forces the output depth back to `ni`.
///
/// The output map is floor-divided in both dimensions, so a partial trailing window is dropped.
#[derive(Debug)]
pub struct Reconfig {
    name: String,
    pub x_scale: usize,
    pub y_scale: usize,
    pub ni: usize,
    pub max: bool,
}

impl Reconfig {
    /// `NT_RECONFIG`: folds `x_scale * y_scale` blocks of the input into the features. Copies `name`, which is the caller's to size, and takes no `Limits`.
    pub fn new(name: &str, ni: usize, x_scale: usize, y_scale: usize) -> Reconfig {
        Reconfig {
            name: name.to_string(),
            x_scale,
            y_scale,
            ni,
            max: false,
        }
    }

    /// `NT_MAXPOOL`: the maximum over each `x_scale * y_scale` block. Copies `name`, which is the caller's to size, and takes no `Limits`.
    pub fn new_maxpool(name: &str, ni: usize, x_scale: usize, y_scale: usize) -> Reconfig {
        Reconfig {
            max: true,
            ..Reconfig::new(name, ni, x_scale, y_scale)
        }
    }

    pub fn num_outputs(&self) -> usize {
        if self.max {
            self.ni
        } else {
            self.ni
                .saturating_mul(self.x_scale)
                .saturating_mul(self.y_scale)
        }
    }

    fn forward(&self, input: &Tensor, cx: &mut Ctx) -> Result<Tensor, Error> {
        if input.features != self.ni {
            return Err(Error::Network(format!(
                "{} got {} input features, want {}",
                short(&self.name),
                input.features,
                self.ni
            )));
        }
        if self.x_scale == 0 || self.y_scale == 0 {
            return Err(Error::Network(format!(
                "{} has invalid scale {}x{}",
                short(&self.name),
                self.x_scale,
                self.y_scale
            )));
        }
        let out_map = input.map.scale_xy(self.x_scale, self.y_scale);
        let mut out = cx.tensor(out_map, self.num_outputs())?;
        for oy in 0..out_map.height {
            cx.check_cancel()?;
            for ox in 0..out_map.width {
                let ot = out_map.t(oy, ox);
                let origin = input.map.t(oy * self.y_scale, ox * self.x_scale);
                if self.max {
                    out.copy_time_step(ot, input, origin)?;
                }
                for dx in 0..self.x_scale {
                    for dy in 0..self.y_scale {
                        let src = input.map.t(oy * self.y_scale + dy, ox * self.x_scale + dx);
                        if self.max {
                            out.maxpool_time_step(ot, input, src);
                        } else {
                            out.copy_time_step_part(
                                ot,
                                (dx * self.y_scale + dy) * self.ni,
                                self.ni,
                                input,
                                src,
                                0,
                            )?;
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

/// `NT_CONVOLVE`: a stride-1 gather that stacks the `(2*half_x+1) x (2*half_y+1)` neighbourhood
/// of every position into `(2*half_x+1)*(2*half_y+1)*ni` output features. It holds no weights.
///
/// Feature order is x-major, then y, then channel. Off-image taps are filled from the
/// recognizer's LCG, not with zeros.
#[derive(Debug)]
pub struct Convolve {
    name: String,
    pub half_x: usize,
    pub half_y: usize,
    pub ni: usize,
    pub rand: Arc<Rand>,
}

impl Convolve {
    /// `NT_CONVOLVE`: gathers the `(2 * half_x + 1) * (2 * half_y + 1)` taps around each position,
    /// drawing the off-image ones from `rand`. Copies `name`, which is the caller's to size, and takes no `Limits`.
    pub fn new(name: &str, ni: usize, half_x: usize, half_y: usize, rand: Arc<Rand>) -> Convolve {
        Convolve {
            name: name.to_string(),
            half_x,
            half_y,
            ni,
            rand,
        }
    }

    /// Recomputed as `Convolve::DeSerialize` does, overwriting whatever the header claimed.
    pub fn num_outputs(&self) -> usize {
        self.ni
            .saturating_mul(self.half_x.saturating_mul(2).saturating_add(1))
            .saturating_mul(self.half_y.saturating_mul(2).saturating_add(1))
    }

    fn forward(&self, input: &Tensor, cx: &mut Ctx) -> Result<Tensor, Error> {
        if input.features != self.ni {
            return Err(Error::Network(format!(
                "convolve {} got {} input features, want {}",
                short(&self.name),
                input.features,
                self.ni
            )));
        }
        let y_scale = self.half_y.saturating_mul(2).saturating_add(1);
        let (hx, hy) = (self.half_x as isize, self.half_y as isize);
        let mut out = cx.tensor(input.map, self.num_outputs())?;
        for t in 0..input.map.len() {
            cx.check_cancel()?;
            let row = out.row_mut(t);
            let mut out_ix = 0;
            for dx in -hx..=hx {
                if input.map.offset(t, 0, dx).is_none() {
                    // The whole column of taps is outside the image.
                    self.randomize(&mut row[out_ix..out_ix + y_scale * self.ni]);
                    out_ix += y_scale * self.ni;
                    continue;
                }
                let mut out_iy = out_ix;
                for dy in -hy..=hy {
                    match input.map.offset(t, dy, dx) {
                        None => self.randomize(&mut row[out_iy..out_iy + self.ni]),
                        Some(src) => {
                            row[out_iy..out_iy + self.ni].copy_from_slice(input.row(src));
                        }
                    }
                    out_iy += self.ni;
                }
                out_ix += y_scale * self.ni;
            }
        }
        Ok(out)
    }

    /// `NetworkIO::Randomize` on the float path.
    fn randomize(&self, dst: &mut [f32]) {
        for d in dst {
            *d = self.rand.signed_rand(1.0) as f32;
        }
    }
}

/// The nonlinearity `FullyConnected::ForwardTimeStep` applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// `NT_LINEAR`
    Linear,
    /// `NT_TANH`
    Tanh,
    /// `NT_LOGISTIC`
    Logistic,
    /// `NT_RELU`
    Relu,
    /// `NT_SOFTMAX` and `NT_SOFTMAX_NO_CTC`
    Softmax,
    /// `NT_POSCLIP`
    PosClip,
    /// `NT_SYMCLIP`
    SymClip,
}

/// Tesseract's `FullyConnected`: one weight matrix applied per timestep, followed by the
/// type's nonlinearity. One batch means no padding, so every timestep is valid.
#[derive(Debug)]
pub struct FullyConnected {
    name: String,
    pub act: Activation,
    pub w: Matrix,
}

impl FullyConnected {
    /// A layer applying `w` to each timestep, then `act`. Copies `name`, which is the caller's to size, and takes no `Limits`.
    pub fn new(name: &str, act: Activation, w: Matrix) -> FullyConnected {
        FullyConnected {
            name: name.to_string(),
            act,
            w,
        }
    }

    fn forward(&self, input: &Tensor, cx: &mut Ctx) -> Result<Tensor, Error> {
        if !self.w.is_consistent() {
            return Err(Error::Network(format!(
                "{} has a matrix whose shape disagrees with its weights",
                short(&self.name)
            )));
        }
        if input.features != self.w.inputs {
            return Err(Error::Network(format!(
                "{} got {} input features, want {}",
                short(&self.name),
                input.features,
                self.w.inputs
            )));
        }
        let mut out = cx.tensor(input.map, self.w.outputs)?;
        // The timesteps are independent, so each thread takes a run of them, with its own
        // scratch vectors. The work is steps * outputs * (inputs + 1) multiply-adds.
        let (ni, no) = (self.w.inputs, self.w.outputs);
        let steps = input.map.len();
        let work = steps
            .saturating_mul(no)
            .saturating_mul(ni.saturating_add(1));
        type Band<'a> = (usize, &'a mut [f32], &'a mut [f64], &'a mut [f64]);
        // The scratch of one band, so that the band count fits the budget (see `band_count`).
        let scratch = ni
            .max(1)
            .saturating_add(no.max(1))
            .saturating_mul(std::mem::size_of::<f64>())
            .saturating_add(std::mem::size_of::<Band>());
        let bands = par::band_count(cx.budget, work, 0, scratch);
        let per_band = steps.div_ceil(bands).max(1);
        let mut u = cx.f64s(bands * ni.max(1))?;
        let mut v = cx.f64s(bands * no.max(1))?;
        let never = AtomicBool::new(false);
        let cancel = cx.cancel.unwrap_or(&never);
        // One band: the timesteps `first..` held in `data`.
        let band =
            |first: usize, data: &mut [f32], u: &mut [f64], v: &mut [f64]| -> Result<(), Error> {
                for (k, dst) in data.chunks_mut(no.max(1)).enumerate() {
                    check_cancel(cancel)?;
                    input.read_time_step(first + k, &mut u[..ni]);
                    self.w.dot_vector(&u[..ni], &mut v[..no])?;
                    self.activate(&mut v[..no])?;
                    for (d, x) in dst.iter_mut().zip(v.iter()) {
                        *d = *x as f32;
                    }
                }
                Ok(())
            };
        if bands == 1 {
            // No band list, so that a call on one thread allocates nothing.
            band(0, &mut out.data[..], &mut u[..], &mut v[..])?;
        } else {
            let _bands_hold = cx.budget.hold(bands * std::mem::size_of::<Band>())?;
            let mut runs: Vec<Band> = out
                .data
                .chunks_mut((per_band * no).max(1))
                .zip(u.chunks_mut(ni.max(1)))
                .zip(v.chunks_mut(no.max(1)))
                .enumerate()
                .map(|(k, ((d, u), v))| (k * per_band, d, u, v))
                .collect();
            par::for_each_chunk_mut(cx.budget, cancel, &mut runs[..], 1, |_, run| {
                let (first, data, u, v) = &mut run[0];
                band(*first, data, u, v)
            })?;
            drop(runs);
        }
        cx.free_f64(u);
        cx.free_f64(v);
        Ok(out)
    }

    fn activate(&self, v: &mut [f64]) -> Result<(), Error> {
        match self.act {
            Activation::Linear => {}
            Activation::Tanh => {
                no_nan(v)?;
                tanh_in_place(v);
            }
            Activation::Logistic => {
                no_nan(v)?;
                logistic_in_place(v);
            }
            Activation::Relu => {
                for x in v.iter_mut() {
                    if *x < 0.0 {
                        *x = 0.0;
                    }
                }
            }
            Activation::Softmax => softmax_in_place(v),
            Activation::PosClip => clip_in_place(v, 0.0, 1.0),
            Activation::SymClip => clip_in_place(v, -1.0, 1.0),
        }
        Ok(())
    }
}

/// Go's tabulated tanh and logistic index their table with `int(x)`, which for NaN is a negative
/// index and a runtime panic. The crate fails the forward pass with an error instead.
fn no_nan(v: &[f64]) -> Result<(), Error> {
    if v.iter().any(|x| x.is_nan()) {
        return Err(Error::Network(
            "a NaN reached a tabulated activation".into(),
        ));
    }
    Ok(())
}

fn clip_in_place(v: &mut [f64], lo: f64, hi: f64) {
    for x in v.iter_mut() {
        if *x < lo {
            *x = lo;
        } else if *x > hi {
            *x = hi;
        }
    }
}

/// Gate indices, from `LSTM::WeightType`. GFS, the fifth gate, exists only for 2-D LSTMs and is
/// rejected at construction.
pub const GATE_CI: usize = 0;
pub const GATE_GI: usize = 1;
pub const GATE_GF1: usize = 2;
pub const GATE_GO: usize = 3;
const NUM_GATES: usize = 4;

/// `kStateClip` from `lstm.cpp`.
const STATE_CLIP: f64 = 100.0;

/// `NT_LSTM` and, with `summary` set, `NT_LSTM_SUMMARY`.
///
/// `ni` is the layer input count, `ns` the internal state count (the output count) and `na` the
/// padded gate-input width, so every gate matrix is `ns x (na+1)`. The column layout of a gate
/// row is `[0, ni)` the layer input, `[ni, na)` the recurrent output of the previous timestep,
/// `[na]` the bias. Softmax feedback and the 2-D case are rejected by `new`.
#[derive(Debug)]
pub struct Lstm {
    name: String,
    pub ni: usize,
    pub ns: usize,
    pub na: usize,
    pub summary: bool,
    pub gates: [Matrix; NUM_GATES],
}

impl Lstm {
    /// A 1-D LSTM layer of `ni` inputs over the five `gates` (`CI`, `GI`, `GF1`, `GO`, `GFS`), whose
    /// outputs per gate give the state size `ns`. It checks the gates' shapes only: the matrices
    /// are the caller's, so it copies only `name` (the caller's to size) and takes no `Limits`.
    ///
    /// # Errors
    ///
    /// `Error::Network` if a gate's input count is not `na` or its output count differs from the
    /// `CI` gate's, if the layer is 2-D (`na == ni + 2 * ns`), or if `na != ni + ns` (a
    /// softmax-feedback LSTM).
    pub fn new(
        name: &str,
        ni: usize,
        na: usize,
        summary: bool,
        gates: [Matrix; NUM_GATES],
    ) -> Result<Lstm, Error> {
        for (i, m) in gates.iter().enumerate() {
            if m.inputs != na {
                return Err(Error::Network(format!(
                    "lstm {} gate {i} has {} input columns, want na={na}",
                    short(name),
                    m.inputs
                )));
            }
            if m.outputs != gates[GATE_CI].outputs {
                return Err(Error::Network(format!(
                    "lstm {} gate {i} has {} outputs, want {}",
                    short(name),
                    m.outputs,
                    gates[GATE_CI].outputs
                )));
            }
        }
        let ns = gates[GATE_CI].outputs;
        // `LSTM::DeSerialize` derives is_2d_ as na_ - nf_ == ni_ + 2*ns_. With no softmax
        // feedback nf_ is zero, so a 1-D layer has na == ni + ns exactly.
        if na == ni.saturating_add(ns.saturating_mul(2)) {
            return Err(Error::Network(format!(
                "lstm {} is 2-D (na={na}, ni={ni}, ns={ns}); 2-D LSTMs and their GFS gate are out of scope",
                short(name)
            )));
        }
        if na != ni.saturating_add(ns) {
            return Err(Error::Network(format!(
                "lstm {} has na={na} but ni+ns={}; softmax-feedback LSTMs are out of scope",
                short(name),
                ni.saturating_add(ns)
            )));
        }
        Ok(Lstm {
            name: name.to_string(),
            ni,
            ns,
            na,
            summary,
            gates,
        })
    }

    fn forward(&self, input: &Tensor, cx: &mut Ctx) -> Result<Tensor, Error> {
        let (out, last_state) = self.forward_state(input, cx)?;
        cx.free_f64(last_state);
        Ok(out)
    }

    /// `forward`, and the cell state as computed at each row's final timestep, captured before
    /// the end-of-row reset zeroes it (of the last row; kept so tests can assert the
    /// `kStateClip` behaviour). The state's bytes stay charged to `cx.budget`; the caller
    /// releases them when it drops it.
    ///
    /// # Errors
    ///
    /// As [`Layer::forward`].
    pub fn forward_state(&self, input: &Tensor, cx: &mut Ctx) -> Result<(Tensor, Vec<f64>), Error> {
        if input.features != self.ni {
            return Err(Error::Network(format!(
                "lstm {} got {} input features, want {}",
                short(&self.name),
                input.features,
                self.ni
            )));
        }
        if self
            .gates
            .iter()
            .any(|g| !g.is_consistent() || g.inputs != self.na || g.outputs != self.ns)
        {
            return Err(Error::Network(format!(
                "lstm {} has a gate whose shape disagrees with its weights",
                short(&self.name)
            )));
        }
        let m = input.map;
        let out_map = if self.summary {
            m.reduce_width_to_1()
        } else {
            m
        };
        let mut out = cx.tensor(out_map, self.ns)?;

        // source_ is a NetworkIO, so the recurrent output is narrowed to f32 before it is
        // read back into the matvec. That rounding is inside the recurrence.
        let mut source = cx.tensor(m, self.na)?;

        let mut u = cx.f64s(self.na)?;
        let mut gate = [
            cx.f64s(self.ns)?,
            cx.f64s(self.ns)?,
            cx.f64s(self.ns)?,
            cx.f64s(self.ns)?,
        ];
        let mut state = cx.f64s(self.ns)?;
        let mut output = cx.f64s(self.ns)?;
        let mut last_state = cx.f64s(self.ns)?;

        let mut dest_t = 0;
        for y in 0..m.height {
            for x in 0..m.width {
                cx.check_cancel()?;
                let t = m.t(y, x);
                source.copy_time_step_part(t, 0, self.ni, input, t, 0)?;
                source.write_time_step_part(t, self.ni, self.ns, &output)?;
                source.read_time_step(t, &mut u);

                self.gates[GATE_CI].dot_vector(&u, &mut gate[GATE_CI])?;
                no_nan(&gate[GATE_CI])?;
                tanh_in_place(&mut gate[GATE_CI]);
                for g in [GATE_GI, GATE_GF1, GATE_GO] {
                    self.gates[g].dot_vector(&u, &mut gate[g])?;
                    no_nan(&gate[g])?;
                    logistic_in_place(&mut gate[g]);
                }

                for i in 0..self.ns {
                    // Two statements in Tesseract: MultiplyVectorsInPlace then
                    // MultiplyAccumulate.
                    let keep = state[i] * gate[GATE_GF1][i];
                    let add = gate[GATE_CI][i] * gate[GATE_GI][i];
                    let s = (keep + add).clamp(-STATE_CLIP, STATE_CLIP);
                    state[i] = s;
                    no_nan(&[s])?;
                    output[i] = tanh(s) * gate[GATE_GO][i];
                }

                if self.summary {
                    if x == m.width - 1 {
                        out.write_time_step(dest_t, &output);
                        dest_t += 1;
                    }
                } else {
                    out.write_time_step(t, &output);
                }

                if x == m.width - 1 {
                    // Capture before zeroing: every row is an independent sequence.
                    last_state.copy_from_slice(&state);
                    state.fill(0.0);
                    output.fill(0.0);
                }
            }
        }
        cx.free(source);
        cx.free_f64(u);
        for g in gate {
            cx.free_f64(g);
        }
        cx.free_f64(state);
        cx.free_f64(output);
        Ok((out, last_state))
    }
}

#[cfg(test)]
pub(super) fn reverse_data_for_test(
    src: &Tensor,
    kind: ReversalKind,
    budget: &Budget,
) -> Result<Tensor, Error> {
    reverse_data(src, kind, &mut Ctx::new(budget))
}
