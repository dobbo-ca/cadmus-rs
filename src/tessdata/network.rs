//! The LSTM network graph with its weights, as serialized in the `lstm` component: a port of
//! `Network::DeSerialize` and the `DeSerialize` methods of the layer types (`network.cpp`,
//! `plumbing.cpp`, `lstm.cpp`, `fullyconnected.cpp`, `convolve.cpp`, `maxpool.cpp`,
//! `reconfig.cpp`, `input.cpp`, `lstmrecognizer.cpp`, `weightmatrix.cpp`, `matrix.h`).
//!
//! `Network::Serialize`, every layer, whatever its type: `int8 type_tag` (0, `NT_NONE`, means a
//! type *name* string follows), `int8 training_state` (1, `TS_ENABLED`, means a training dump),
//! `int8 needs_to_backprop`, `int32 network_flags`, `int32 ni`, `int32 no`, `int32 num_weights`,
//! `string name`, then the type-specific body.
//!
//! `GENERIC_2D_ARRAY<T>`: `int32 dim1` and `int32 dim2` (each rejected above `UINT16_MAX`), then
//! ONE padding element `empty_` before the array proper, then `dim1*dim2` values. The lone
//! `empty_` is the easy thing to miss; omitting it desynchronises every subsequent read.
//!
//! `WeightMatrix::DeSerialize`: `uint8 mode` (bit0 int8 weights, bit2 Adam, bit7 new format);
//! without bit7 it is `DeSerializeOld` (float32), rejected; with bit0 int8 + scales, rejected;
//! else a float64 `GENERIC_2D_ARRAY`.
//!
//! The `lstm` component is a serialized `LSTMRecognizer`: the root layer, then (with
//! `include_charsets == false`, the case for every model that ships lstm-unicharset and
//! lstm-recoder as separate components) `string network_str_`, `int32 training_flags_`,
//! `training_iteration_`, `sample_iteration_`, `null_char_`, and `f32 adam_beta_`,
//! `learning_rate_`, `momentum_`.
//!
//! Translated from Tesseract OCR (Apache-2.0, see NOTICE), through the Go port. The translation
//! is not verbatim.

use super::goquote::{quote_into, short};
use super::{MAX_STRING_LEN, Reader, err};
use crate::imaging::check_cancel;
use crate::{Budget, Error, Limits};
use std::fmt::Write;
use std::sync::atomic::AtomicBool;

/// `NetworkType` from src/lstm/network.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    None = 0,
    Input = 1,
    Convolve = 2,
    Maxpool = 3,
    Parallel = 4,
    Replicated = 5,
    ParRlLstm = 6,
    ParUdLstm = 7,
    Par2dLstm = 8,
    Series = 9,
    Reconfig = 10,
    XReversed = 11,
    YReversed = 12,
    XyTranspose = 13,
    Lstm = 14,
    LstmSummary = 15,
    Logistic = 16,
    PosClip = 17,
    SymClip = 18,
    Tanh = 19,
    Relu = 20,
    Linear = 21,
    Softmax = 22,
    SoftmaxNoCtc = 23,
    LstmSoftmax = 24,
    LstmSoftmaxEncoded = 25,
    TensorFlow = 26,
}

const NUM_LAYER_TYPES: usize = 27;

/// `kTypeNames` from src/lstm/network.cpp, in enum order. A layer whose type tag is 0
/// identifies itself by name, and the name must round-trip through this table.
const LAYER_TYPE_NAMES: [&str; NUM_LAYER_TYPES] = [
    "Invalid",
    "Input",
    "Convolve",
    "Maxpool",
    "Parallel",
    "Replicated",
    "ParBidiLSTM",
    "DepParUDLSTM",
    "Par2dLSTM",
    "Series",
    "Reconfig",
    "RTLReversed",
    "TTBReversed",
    "XYTranspose",
    "LSTM",
    "SummLSTM",
    "Logistic",
    "LinLogistic",
    "LinTanh",
    "Tanh",
    "Relu",
    "Linear",
    "Softmax",
    "SoftmaxNoCTC",
    "LSTMSoftmax",
    "LSTMBinarySoftmax",
    "TensorFlow",
];

const ALL_LAYER_TYPES: [LayerType; NUM_LAYER_TYPES] = [
    LayerType::None,
    LayerType::Input,
    LayerType::Convolve,
    LayerType::Maxpool,
    LayerType::Parallel,
    LayerType::Replicated,
    LayerType::ParRlLstm,
    LayerType::ParUdLstm,
    LayerType::Par2dLstm,
    LayerType::Series,
    LayerType::Reconfig,
    LayerType::XReversed,
    LayerType::YReversed,
    LayerType::XyTranspose,
    LayerType::Lstm,
    LayerType::LstmSummary,
    LayerType::Logistic,
    LayerType::PosClip,
    LayerType::SymClip,
    LayerType::Tanh,
    LayerType::Relu,
    LayerType::Linear,
    LayerType::Softmax,
    LayerType::SoftmaxNoCtc,
    LayerType::LstmSoftmax,
    LayerType::LstmSoftmaxEncoded,
    LayerType::TensorFlow,
];

impl LayerType {
    pub fn name(self) -> &'static str {
        LAYER_TYPE_NAMES[self as usize]
    }
}

impl std::fmt::Display for LayerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// `TS_ENABLED`: only a training dump carries the extra per-matrix update arrays.
const TS_ENABLED: i8 = 1;
/// `NF_LAYER_SPECIFIC_LR`.
const NF_LAYER_SPECIFIC_LR: i32 = 64;
/// `Plumbing::DeSerialize`'s guard.
const MAX_STACK_SIZE: u32 = 10000;
/// Not in Tesseract (Go recurses without a bound): a guard so a hostile file cannot overflow the
/// stack. Real networks are about 5 deep.
const MAX_DEPTH: usize = 64;
/// The fewest bytes a serialized layer takes: the type tag, the training state and backprop flags
/// (1 each), and the flags, `ni`, `no`, `num_weights` and name length (4 each).
const MIN_LAYER_BYTES: usize = 23;
/// The `UINT16_MAX` cap in `GENERIC_2D_ARRAY::DeSerializeSize`.
const MAX_MATRIX_DIM: i32 = 65535;
const MODE_INT8_FLAG: u8 = 1;
// The Adam flag (4) is deliberately NOT consulted: whether the Adam arrays follow a matrix is
// decided by the layer's training-state byte, and training dumps are rejected outright. Keying
// off it would desynchronise the stream: every matrix in tessdata_best eng has mode 132.
const MODE_DOUBLE_FLAG: u8 = 128;

/// `TF_INT_MODE` and `TF_COMPRESS_UNICHARSET`, from src/lstm/lstmrecognizer.h:44.
const TF_INT_MODE: i32 = 1;
const TF_COMPRESS_UNICHARSET: i32 = 64;

/// One deserialized weight matrix, row-major with row stride `cols`: element `(row, col)` is
/// `values[row*cols+col]`. `rows` is the output count; `cols` is the input count plus ONE
/// trailing bias column (`MatrixDotVectorInternal` takes the dot product over `dim2-1` elements
/// and adds `w[i][dim2-1]` against an implicit 1.0).
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f64>,
}

impl Matrix {
    /// The weight connecting input `col` to output `row`.
    pub fn at(&self, row: usize, col: usize) -> f64 {
        self.values[row * self.cols + col]
    }

    /// The minimum, maximum and arithmetic mean of the values; zeroes for an empty matrix.
    pub fn stats(&self) -> (f64, f64, f64) {
        if self.values.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let (mut min, mut max) = (self.values[0], self.values[0]);
        let mut sum = 0.0f64;
        for &v in &self.values {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            sum += v;
        }
        (min, max, sum / self.values.len() as f64)
    }
}

/// Tesseract's `StaticShape`, the 4-D tensor description an Input layer carries. Width 0 means
/// "determined at runtime".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputShape {
    pub batch: i32,
    pub height: i32,
    pub width: i32,
    pub depth: i32,
    pub loss_type: i32,
}

/// One node of a deserialized LSTM network graph, with its weight values.
#[derive(Debug, Clone)]
pub struct Layer {
    pub typ: LayerType,
    pub name: String,
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub num_weights: usize,
    pub flags: i32,
    pub matrices: Vec<Matrix>,
    pub children: Vec<Layer>,
    /// Convolve: `half_x_`, `half_y_`.
    pub half_x: i32,
    pub half_y: i32,
    /// Maxpool, Reconfig: `x_scale_`, `y_scale_`.
    pub x_scale: i32,
    pub y_scale: i32,
    /// Input only.
    pub shape: Option<InputShape>,
    /// LSTM family: `na_`.
    pub na: i32,
}

/// A parsed `lstm` component: the network graph plus the `LSTMRecognizer` fields serialized
/// after it.
#[derive(Debug, Clone)]
pub struct Recognizer {
    pub network: Layer,
    /// The spec string the model was built from, e.g. `[1,36,0,1Ct3,3,16Mp3,3Lfys64...O1c1]`. Its
    /// output count is a PLACEHOLDER (`ParseOutput` overrides it with the recoder's code range).
    pub network_str: String,
    pub training_flags: i32,
    pub training_iteration: i32,
    /// Seeds `LSTMRecognizer::SetRandomSeed`, which drives the random edge padding `Convolve`
    /// applies at image borders.
    pub sample_iteration: i32,
    /// The network output index of the CTC blank. It is the authority: it is `code_range-1` in
    /// practice, but the format does not require that and this field does.
    pub null_char: i32,
    pub adam_beta: f32,
    pub learning_rate: f32,
    pub momentum: f32,
}

impl Recognizer {
    /// Whether the model carries int8-quantized weights.
    pub fn is_int_mode(&self) -> bool {
        self.training_flags & TF_INT_MODE != 0
    }
}

/// Deserializes a `.traineddata` lstm component. `swap` comes from `Container::swapped`.
///
/// Requiring the buffer to end exactly where the format says it should is what catches a parser
/// that desynchronised mid-graph and returned a plausible-looking tree.
///
/// Everything the result holds is charged to `budget` (which also supplies the limits), and
/// `cancel` is checked at every layer and every weight matrix.
///
/// # Errors
///
/// `Error::AllocLimit` when the weights and strings do not fit in `max_alloc`,
/// `Error::Cancelled`, else `Error::Tessdata` for a malformed or unsupported component, `Error::OutOfMemory` if the allocator refuses.
pub fn parse_recognizer(
    data: &[u8],
    swap: bool,
    budget: &Budget,
    cancel: &AtomicBool,
) -> Result<Recognizer, Error> {
    let mut r = Reader::new(data);
    r.set_swap(swap);

    // The root layer is the only one that is not a slot of its parent's `children`.
    budget.charge(std::mem::size_of::<Layer>())?;
    let network = parse_layer(&mut r, budget, cancel, 0)?;
    let network_str = r.string(budget).map_err(|e| {
        err(format!(
            "reading network spec string (the graph parse may have desynchronised): {e}"
        ))
    })?;
    let mut ints = [0i32; 4];
    for (dst, name) in ints.iter_mut().zip([
        "training_flags",
        "training_iteration",
        "sample_iteration",
        "null_char",
    ]) {
        *dst = r.int32().map_err(|e| err(format!("reading {name}: {e}")))?;
    }
    // adam_beta_, learning_rate_ and momentum_ are declared `float`: 4 bytes each, not 8.
    let mut floats = [0f32; 3];
    for (dst, name) in floats
        .iter_mut()
        .zip(["adam_beta", "learning_rate", "momentum"])
    {
        *dst = r
            .float32()
            .map_err(|e| err(format!("reading {name}: {e}")))?;
    }
    if r.remaining() != 0 {
        return Err(err(format!(
            "{} bytes left unconsumed after the lstm component; the parse desynchronised",
            r.remaining()
        )));
    }
    let rec = Recognizer {
        network,
        network_str,
        training_flags: ints[0],
        training_iteration: ints[1],
        sample_iteration: ints[2],
        null_char: ints[3],
        adam_beta: floats[0],
        learning_rate: floats[1],
        momentum: floats[2],
    };

    if rec.is_int_mode() {
        return Err(err(format!(
            "training_flags {} has TF_INT_MODE set: this is an int8-quantized model. Cadmus implements the float weight path only (tessdata_best)",
            rec.training_flags
        )));
    }
    if rec.training_flags & TF_COMPRESS_UNICHARSET == 0 {
        return Err(err(format!(
            "training_flags {} has TF_COMPRESS_UNICHARSET clear: the model has no recoder and LSTMRecognizer::LoadRecoder would synthesise a pass-through one. Cadmus requires a recoding model",
            rec.training_flags
        )));
    }
    if rec.null_char < 0 || rec.null_char as usize >= rec.network.num_outputs {
        return Err(err(format!(
            "null_char {} is outside the network's {} outputs",
            rec.null_char, rec.network.num_outputs
        )));
    }
    Ok(rec)
}

/// Reads the common Network header and dispatches on layer type.
pub(super) fn parse_layer(
    r: &mut Reader,
    budget: &Budget,
    cancel: &AtomicBool,
    depth: usize,
) -> Result<Layer, Error> {
    check_cancel(cancel)?;
    if depth > MAX_DEPTH {
        return Err(err(format!(
            "network nested deeper than {MAX_DEPTH} layers"
        )));
    }
    let typ = parse_layer_type(r, budget)?;
    let training = r
        .int8()
        .map_err(|e| err(format!("reading {typ} training state: {e}")))?;
    r.int8()
        .map_err(|e| err(format!("reading {typ} backprop flag: {e}")))?;
    let flags = r
        .int32()
        .map_err(|e| err(format!("reading {typ} flags: {e}")))?;
    let ni = r
        .int32()
        .map_err(|e| err(format!("reading {typ} input count: {e}")))?;
    let no = r
        .int32()
        .map_err(|e| err(format!("reading {typ} output count: {e}")))?;
    let num_weights = r
        .int32()
        .map_err(|e| err(format!("reading {typ} weight count: {e}")))?;
    let name = r
        .string(budget)
        .map_err(|e| err(format!("reading {typ} name: {e}")))?;
    if ni < 0 || no < 0 || num_weights < 0 {
        return Err(err(format!(
            "invalid {typ} layer parameters: ni={ni} no={no} num_weights={num_weights}"
        )));
    }
    // TS_ENABLED means the file is a training dump, whose weight matrices carry an extra updates_
    // array that this parser does not read. No released model has TS_ENABLED.
    if training == TS_ENABLED {
        return Err(err(format!(
            "{typ} layer {} has training state TS_ENABLED: training dumps are not supported",
            short(&name)
        )));
    }

    let mut l = Layer {
        typ,
        name,
        num_inputs: ni as usize,
        num_outputs: no as usize,
        num_weights: num_weights as usize,
        flags,
        matrices: Vec::new(),
        children: Vec::new(),
        half_x: 0,
        half_y: 0,
        x_scale: 0,
        y_scale: 0,
        shape: None,
        na: 0,
    };

    use LayerType as T;
    let res = match typ {
        T::Series
        | T::Parallel
        | T::Replicated
        | T::ParRlLstm
        | T::ParUdLstm
        | T::Par2dLstm
        | T::XReversed
        | T::YReversed
        | T::XyTranspose => parse_plumbing(r, &mut l, budget, cancel, depth),
        T::Softmax
        | T::SoftmaxNoCtc
        | T::Relu
        | T::Tanh
        | T::Linear
        | T::Logistic
        | T::PosClip
        | T::SymClip => budget
            .reserve(&mut l.matrices, 1)
            .and_then(|()| parse_matrix(r, budget, cancel))
            .map(|m| l.matrices.push(m))
            .map_err(|e| wrap("reading weights", e)),
        T::Lstm | T::LstmSummary | T::LstmSoftmax | T::LstmSoftmaxEncoded => {
            parse_lstm(r, &mut l, budget, cancel, depth)
        }
        // half_x_, half_y_ (convolve.cpp:45). Convolve holds no weights: it is a pure im2col
        // gather and recomputes no_ = ni_*(2*half_x+1)*(2*half_y+1).
        T::Convolve => parse_i32_pair(r).map(|(a, b)| {
            l.half_x = a;
            l.half_y = b;
        }),
        // x_scale_, y_scale_ (reconfig.cpp:60). Maxpool serializes identical bytes and then
        // overrides no_ = ni_.
        T::Maxpool | T::Reconfig => parse_i32_pair(r).map(|(a, b)| {
            l.x_scale = a;
            l.y_scale = b;
        }),
        T::Input => parse_input_shape(r).map(|s| l.shape = Some(s)),
        // NT_NONE and NT_TENSORFLOW have no deserializer in Tesseract either.
        T::None | T::TensorFlow => {
            return Err(err(format!("unsupported network layer type {typ}")));
        }
    };
    // The message is built only for a `Tessdata` error, and the name is cut, so a long name in the
    // file costs no uncharged copy (and `AllocLimit` and `Cancelled` pass through untouched).
    res.map_err(|e| match e {
        Error::Tessdata(s) => Error::Tessdata(format!("{typ} layer {}: {s}", short(&l.name))),
        e => e,
    })?;

    // A layer's num_weights is the total element count of its own matrices (bias columns
    // included) plus its children's num_weights. A mismatch means the parse desynchronised.
    let mut total = 0usize;
    for m in &l.matrices {
        total = total.saturating_add(m.rows * m.cols);
    }
    for c in &l.children {
        total = total.saturating_add(c.num_weights);
    }
    if total != l.num_weights {
        return Err(err(format!(
            "{typ} layer {} declares num_weights={} but its matrices and children total {total}",
            short(&l.name),
            l.num_weights
        )));
    }
    Ok(l)
}

fn wrap(what: &str, e: Error) -> Error {
    match e {
        Error::Tessdata(s) => Error::Tessdata(format!("{what}: {s}")),
        e => e,
    }
}

fn parse_i32_pair(r: &mut Reader) -> Result<(i32, i32), Error> {
    let a = r
        .int32()
        .map_err(|e| err(format!("reading first field: {e}")))?;
    let b = r
        .int32()
        .map_err(|e| err(format!("reading second field: {e}")))?;
    Ok((a, b))
}

/// `StaticShape`'s `batch_`, `height_`, `width_`, `depth_` and `loss_type_`, in that order.
fn parse_input_shape(r: &mut Reader) -> Result<InputShape, Error> {
    let mut v = [0i32; 5];
    for (i, dst) in v.iter_mut().enumerate() {
        *dst = r
            .int32()
            .map_err(|e| err(format!("reading input shape field {i}: {e}")))?;
    }
    Ok(InputShape {
        batch: v[0],
        height: v[1],
        width: v[2],
        depth: v[3],
        loss_type: v[4],
    })
}

/// `getNetworkType`: current Tesseract always writes a 0 tag followed by the type name.
fn parse_layer_type(r: &mut Reader, budget: &Budget) -> Result<LayerType, Error> {
    let tag = r
        .int8()
        .map_err(|e| err(format!("reading layer type tag: {e}")))?;
    if tag != 0 {
        if tag < 0 || tag as usize >= NUM_LAYER_TYPES {
            return Err(err(format!("layer type tag {tag} out of range")));
        }
        return Ok(ALL_LAYER_TYPES[tag as usize]);
    }
    let name = r
        .string(budget)
        .map_err(|e| err(format!("reading layer type name: {e}")))?;
    let typ = LAYER_TYPE_NAMES
        .iter()
        .position(|n| *n == name)
        .map(|i| ALL_LAYER_TYPES[i])
        .ok_or_else(|| err(format!("unknown network layer type {}", short(&name))));
    let n = name.len();
    drop(name);
    budget.release(n);
    typ
}

/// `Plumbing::DeSerialize`.
fn parse_plumbing(
    r: &mut Reader,
    l: &mut Layer,
    budget: &Budget,
    cancel: &AtomicBool,
    depth: usize,
) -> Result<(), Error> {
    let size = r
        .uint32()
        .map_err(|e| err(format!("reading stack size: {e}")))?;
    if size > MAX_STACK_SIZE {
        return Err(err(format!(
            "stack size {size} exceeds maximum {MAX_STACK_SIZE}"
        )));
    }
    // Every child takes at least MIN_LAYER_BYTES, so the bytes left bound the room reserved here.
    if size as usize > r.remaining() / MIN_LAYER_BYTES {
        return Err(err(format!(
            "stack size {size} does not fit in the {} bytes that remain",
            r.remaining()
        )));
    }
    budget.reserve(&mut l.children, size as usize)?;
    for i in 0..size {
        let child = parse_layer(r, budget, cancel, depth + 1)
            .map_err(|e| wrap(&format!("child {i}"), e))?;
        l.children.push(child);
    }
    if l.flags & NF_LAYER_SPECIFIC_LR != 0 {
        // Plumbing::learning_rates_ is std::vector<float>: a uint32 count then that many
        // 4-byte values, skipped.
        let n = r.uint32().map_err(|e| {
            err(format!(
                "reading layer-specific learning rates: reading count: {e}"
            ))
        })?;
        if n > MAX_STRING_LEN {
            return Err(err(format!(
                "reading layer-specific learning rates: count {n} exceeds maximum {MAX_STRING_LEN}"
            )));
        }
        r.bytes(4 * n as usize).map_err(|e| {
            err(format!(
                "reading layer-specific learning rates: skipping {n} values: {e}"
            ))
        })?;
    }
    Ok(())
}

/// `LSTM::DeSerialize`.
fn parse_lstm(
    r: &mut Reader,
    l: &mut Layer,
    budget: &Budget,
    cancel: &AtomicBool,
    depth: usize,
) -> Result<(), Error> {
    let na = r.int32().map_err(|e| err(format!("reading na: {e}")))?;
    l.na = na;
    // nf_ is the number of softmax outputs fed back into the input.
    let nf = match l.typ {
        LayerType::LstmSoftmax => l.num_outputs as i64,
        LayerType::LstmSoftmaxEncoded => i64::from(ceil_log2(l.num_outputs as u32)),
        _ => 0,
    };
    // Gate order CI, GI, GF1, GO, then GFS for a 2-D LSTM only.
    budget.reserve(&mut l.matrices, 5)?;
    let mut is_2d = false;
    for w in 0..5 {
        if w == 4 && !is_2d {
            continue;
        }
        let m = parse_matrix(r, budget, cancel)
            .map_err(|e| wrap(&format!("reading gate weights {w}"), e))?;
        if w == 0 {
            // ns_ = gate_weights_[CI].NumOutputs(); the 2-D test is exactly Tesseract's.
            let ns = m.rows as i64;
            is_2d = i64::from(na) - nf == l.num_inputs as i64 + 2 * ns;
        }
        l.matrices.push(m);
    }
    if matches!(
        l.typ,
        LayerType::LstmSoftmax | LayerType::LstmSoftmaxEncoded
    ) {
        budget.reserve(&mut l.children, 1)?;
        let softmax = parse_layer(r, budget, cancel, depth + 1)
            .map_err(|e| wrap("reading built-in softmax", e))?;
        l.children.push(softmax);
    }
    Ok(())
}

/// `WeightMatrix::DeSerialize` for the only configuration cadmus supports: the modern format
/// (`kDoubleFlag`), float64 weights (`kInt8Flag` clear), not a training dump.
fn parse_matrix(r: &mut Reader, budget: &Budget, cancel: &AtomicBool) -> Result<Matrix, Error> {
    check_cancel(cancel)?;
    let mode = r
        .uint8()
        .map_err(|e| err(format!("reading matrix mode: {e}")))?;
    if mode & MODE_DOUBLE_FLAG == 0 {
        return Err(err(format!(
            "matrix mode {mode} has kDoubleFlag clear, selecting WeightMatrix::DeSerializeOld (float32 weights); cadmus supports only the modern float64 format"
        )));
    }
    if mode & MODE_INT8_FLAG != 0 {
        return Err(err(format!(
            "matrix mode {mode} has kInt8Flag set: this is an int8-quantized model. Cadmus implements the float weight path only (tessdata_best)"
        )));
    }
    read_float64_matrix(r, budget)
}

/// `GENERIC_2D_ARRAY<double>::DeSerialize`: `dim1`, `dim2`, the discarded `empty_`, then
/// `dim1*dim2` values.
fn read_float64_matrix(r: &mut Reader, budget: &Budget) -> Result<Matrix, Error> {
    let dim1 = r.int32().map_err(|e| err(format!("reading dim1: {e}")))?;
    let dim2 = r.int32().map_err(|e| err(format!("reading dim2: {e}")))?;
    // Tesseract checks only the UINT16_MAX upper bound; a negative size passes and then
    // misbehaves inside Resize(). Reject it here.
    if !(0..=MAX_MATRIX_DIM).contains(&dim1) || !(0..=MAX_MATRIX_DIM).contains(&dim2) {
        return Err(err(format!("implausible matrix dimensions {dim1}x{dim2}")));
    }
    let n = dim1 as usize * dim2 as usize;
    let need = 8 * (1 + n);
    if r.remaining() < need {
        return Err(err(format!(
            "matrix {dim1}x{dim2} needs {need} bytes, {} remain",
            r.remaining()
        )));
    }
    r.float64()?; // empty_
    let mut values = budget.vec(n, 0f64)?;
    for (i, v) in values.iter_mut().enumerate() {
        *v = r
            .float64()
            .map_err(|e| err(format!("reading element {i} of {dim1}x{dim2}: {e}")))?;
    }
    Ok(Matrix {
        rows: dim1 as usize,
        cols: dim2 as usize,
        values,
    })
}

/// `ceil_log2` from src/lstm/lstm.cpp.
fn ceil_log2(n: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    let l2 = 31 - n.leading_zeros();
    if n == 1 << l2 { l2 } else { l2 + 1 }
}

impl Layer {
    /// An indented one-line-per-layer rendering of the graph. Names are quoted as Go's `%q` does.
    ///
    /// Enforces `limits.max_alloc`: the text is charged to it before it grows. No plane caps
    /// apply and it spawns no thread.
    ///
    /// # Errors
    ///
    /// `Error::AllocLimit` if the text does not fit in `limits.max_alloc`, `Error::OutOfMemory` if
    /// the allocator refuses.
    pub fn tree(&self, limits: &Limits) -> Result<String, Error> {
        let mut out = String::new();
        self.tree_into(&mut out, 0, &Budget::new(limits))?;
        Ok(out)
    }

    fn tree_into(&self, w: &mut String, depth: usize, budget: &Budget) -> Result<(), Error> {
        // The most one layer's text can take: the quoted name is at most 4 bytes per name byte,
        // each count at most 20 bytes, and a matrix line holds three `%f` floats of up to 320
        // bytes. The string grows to hold that before the layer is written, and the growth is
        // charged first.
        let bound =
            2 * depth + 4 * self.name.len() + 1024 + self.matrices.len() * (2 * depth + 1100);
        if w.len() + bound > w.capacity() {
            let cap = (w.len() + bound).max(2 * w.capacity());
            // The old buffer is still live while the new one is made, so the new capacity is
            // charged in full and the old one given back after.
            let old = w.capacity();
            budget.charge(cap)?;
            if w.try_reserve_exact(cap - w.len()).is_err() {
                budget.release(cap);
                return Err(Error::OutOfMemory);
            }
            budget.release(old);
        }
        let pad = "  ".repeat(depth);
        let _ = write!(w, "{pad}{} ", self.typ);
        // The room for the quoted name is in the bound above, so this writes into the charged
        // buffer and allocates nothing.
        quote_into(w, &self.name);
        let _ = write!(w, " ni={} no={}", self.num_inputs, self.num_outputs);
        if self.na != 0 {
            let _ = write!(w, " na={}", self.na);
        }
        let _ = writeln!(w, " weights={}", self.num_weights);

        match self.typ {
            LayerType::Input => {
                if let Some(s) = &self.shape {
                    let _ = writeln!(
                        w,
                        "{pad}    shape batch={} height={} width={} depth={} loss={}",
                        s.batch, s.height, s.width, s.depth, s.loss_type
                    );
                }
            }
            LayerType::Convolve => {
                let _ = writeln!(w, "{pad}    half_x={} half_y={}", self.half_x, self.half_y);
            }
            LayerType::Maxpool | LayerType::Reconfig => {
                let _ = writeln!(
                    w,
                    "{pad}    x_scale={} y_scale={}",
                    self.x_scale, self.y_scale
                );
            }
            _ => {}
        }

        // Gate names label an LSTM layer's matrices in serialization order.
        const GATE_NAMES: [&str; 5] = ["CI", "GI", "GF1", "GO", "GFS"];
        let is_lstm = matches!(
            self.typ,
            LayerType::Lstm
                | LayerType::LstmSummary
                | LayerType::LstmSoftmax
                | LayerType::LstmSoftmaxEncoded
        );
        for (i, m) in self.matrices.iter().enumerate() {
            let label = if is_lstm && i < GATE_NAMES.len() {
                GATE_NAMES[i]
            } else {
                ""
            };
            let (min, max, mean) = m.stats();
            let _ = writeln!(
                w,
                "{pad}    [{i}] {label:<3} {}x{} min={} max={} mean={}",
                m.rows,
                m.cols,
                go_f6(min),
                go_f6(max),
                go_f6(mean)
            );
        }
        for c in &self.children {
            c.tree_into(w, depth + 1, budget)?;
        }
        Ok(())
    }
}

#[cfg(test)]
pub(super) fn ceil_log2_for_test(n: u32) -> u32 {
    ceil_log2(n)
}

/// Go's `%f` of a float: six decimals, and `+Inf` and `-Inf` where Rust prints `inf`.
fn go_f6(v: f64) -> String {
    if v.is_infinite() {
        return if v < 0.0 { "-Inf" } else { "+Inf" }.into();
    }
    format!("{v:.6}")
}
