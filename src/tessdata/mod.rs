//! The `.traineddata` loader: a port of `internal/tessdata`, translated from `tessdatamanager.cpp`,
//! `serialis.cpp`, `lstmrecognizer.cpp`, `dict.cpp` and the network, unicharset, recoder and dawg
//! readers of Tesseract OCR (Apache-2.0, see NOTICE).
//!
//! Every error is [`Error::Tessdata`] (Go's are all `fmt.Errorf` strings of this package) or a
//! limit error. A `Limits` caps the model size, and every allocation sized from the file is
//! charged against `max_alloc` after being checked against the bytes that remain.

mod dawg;
mod goquote;
mod network;
mod recoder;
mod unicharset;

pub use dawg::{Dawg, NO_EDGE, PATTERN_UNICHAR_ID};
pub use goquote::quote;
pub(crate) use goquote::short;
pub use network::{InputShape, Layer, LayerType, Matrix, Recognizer, parse_recognizer};
pub use recoder::Recoder;
pub use unicharset::{
    PROP_ALPHA, PROP_DIGIT, PROP_LOWER, PROP_PUNCT, PROP_UPPER, UNICHAR_BROKEN, UNICHAR_JOINED,
    UNICHAR_SPACE, Unichar, Unicharset, parse_unicharset,
};

use crate::imaging::check_cancel;
use crate::{Budget, Error, Limits};
use std::sync::atomic::AtomicBool;

pub(crate) fn err(msg: impl Into<String>) -> Error {
    Error::Tessdata(msg.into())
}

/// `kMaxNumTessdataEntries`. A count above it in the file's own byte order means the file was
/// written on the other endianness.
const MAX_NUM_ENTRIES: u32 = 1000;

/// `TFile::DeSerialize(std::string&)`'s arbitrary corruption guard.
pub(crate) const MAX_STRING_LEN: u32 = 50_000_000;

/// Reads Tesseract's serialization primitives from an in-memory buffer. Tesseract writes in host
/// byte order and detects a foreign-endian file at the container header, then byte-swaps every
/// subsequent multi-byte read; `set_swap` reproduces that.
#[derive(Debug)]
pub(crate) struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    swap: bool,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Reader<'a> {
        Reader {
            data,
            pos: 0,
            swap: false,
        }
    }

    pub(crate) fn set_swap(&mut self, swap: bool) {
        self.swap = swap;
    }

    pub(crate) fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    pub(crate) fn bytes(&mut self, n: usize) -> Result<&'a [u8], Error> {
        if self.remaining() < n {
            return Err(err(format!(
                "short read at offset {}: want {n} bytes, have {}",
                self.pos,
                self.remaining()
            )));
        }
        let b = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(b)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        let b = self.bytes(N)?;
        let mut a = [0u8; N];
        a.copy_from_slice(b);
        Ok(a)
    }

    pub(crate) fn uint8(&mut self) -> Result<u8, Error> {
        Ok(self.bytes(1)?[0])
    }

    pub(crate) fn int8(&mut self) -> Result<i8, Error> {
        Ok(self.uint8()? as i8)
    }

    pub(crate) fn uint32(&mut self) -> Result<u32, Error> {
        let v = u32::from_le_bytes(self.array()?);
        Ok(if self.swap { v.swap_bytes() } else { v })
    }

    pub(crate) fn int32(&mut self) -> Result<i32, Error> {
        Ok(self.uint32()? as i32)
    }

    /// A 2-byte signed integer. Only the DAWG header uses one (`kDawgMagicNumber`).
    pub(crate) fn int16(&mut self) -> Result<i16, Error> {
        let v = u16::from_le_bytes(self.array()?);
        Ok((if self.swap { v.swap_bytes() } else { v }) as i16)
    }

    /// A 4-byte IEEE-754 value. Tesseract writes `float` rather than `TFloat` for the
    /// recognizer's `adam_beta_`, `learning_rate_` and `momentum_`.
    pub(crate) fn float32(&mut self) -> Result<f32, Error> {
        Ok(f32::from_bits(self.uint32()?))
    }

    /// An 8-byte unsigned integer. DAWG edge records are `uint64`.
    pub(crate) fn uint64(&mut self) -> Result<u64, Error> {
        let v = u64::from_le_bytes(self.array()?);
        Ok(if self.swap { v.swap_bytes() } else { v })
    }

    pub(crate) fn int64(&mut self) -> Result<i64, Error> {
        Ok(self.uint64()? as i64)
    }

    pub(crate) fn float64(&mut self) -> Result<f64, Error> {
        Ok(f64::from_bits(self.uint64()?))
    }

    /// A uint32 length and that many bytes. Go strings are bytes; bytes that are not UTF-8 are
    /// replaced (the strings read here are names and spec strings).
    ///
    /// The copy is charged to `budget` once the bytes are known to be in the buffer, before it is made.
    pub(crate) fn string(&mut self, budget: &Budget) -> Result<String, Error> {
        let n = self.uint32()?;
        if n > MAX_STRING_LEN {
            return Err(err(format!(
                "implausible string length {n} at offset {}",
                self.pos
            )));
        }
        if n == 0 {
            return Ok(String::new());
        }
        let bytes = self.bytes(n as usize)?;
        lossy_string(bytes, budget)
    }
}

/// `String::from_utf8_lossy(bytes)` as an owned string, charged to `budget` for its exact length
/// before it is made: each invalid sequence becomes U+FFFD, which is 3 bytes.
pub(crate) fn lossy_string(bytes: &[u8], budget: &Budget) -> Result<String, Error> {
    let len: usize = bytes
        .utf8_chunks()
        .map(|c| c.valid().len() + if c.invalid().is_empty() { 0 } else { 3 })
        .sum();
    budget.charge(len)?;
    let mut s = String::new();
    if s.try_reserve_exact(len).is_err() {
        budget.release(len);
        return Err(Error::OutOfMemory);
    }
    for c in bytes.utf8_chunks() {
        s.push_str(c.valid());
        if !c.invalid().is_empty() {
            s.push('\u{FFFD}');
        }
    }
    Ok(s)
}

/// The component types of a `.traineddata` container (`TessdataType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Type {
    LangConfig = 0,
    Unicharset = 1,
    Ambigs = 2,
    /// legacy engine
    IntTemp = 3,
    /// legacy engine
    PffmTable = 4,
    /// legacy engine
    NormProto = 5,
    PuncDawg = 6,
    SystemDawg = 7,
    NumberDawg = 8,
    FreqDawg = 9,
    /// legacy engine
    ShapeTable = 13,
    BigramDawg = 14,
    UnambigDawg = 15,
    ParamsModel = 16,
    Lstm = 17,
    LstmPuncDawg = 18,
    LstmSystemDawg = 19,
    LstmNumberDawg = 20,
    LstmUnicharset = 21,
    LstmRecoder = 22,
    Version = 23,
}

const NUM_TYPES: usize = 24;

impl Type {
    const ALL: [Type; 21] = [
        Type::LangConfig,
        Type::Unicharset,
        Type::Ambigs,
        Type::IntTemp,
        Type::PffmTable,
        Type::NormProto,
        Type::PuncDawg,
        Type::SystemDawg,
        Type::NumberDawg,
        Type::FreqDawg,
        Type::ShapeTable,
        Type::BigramDawg,
        Type::UnambigDawg,
        Type::ParamsModel,
        Type::Lstm,
        Type::LstmPuncDawg,
        Type::LstmSystemDawg,
        Type::LstmNumberDawg,
        Type::LstmUnicharset,
        Type::LstmRecoder,
        Type::Version,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Type::LangConfig => "lang_config",
            Type::Unicharset => "unicharset",
            Type::Ambigs => "ambigs",
            Type::IntTemp => "inttemp",
            Type::PffmTable => "pffmtable",
            Type::NormProto => "normproto",
            Type::PuncDawg => "punc_dawg",
            Type::SystemDawg => "system_dawg",
            Type::NumberDawg => "number_dawg",
            Type::FreqDawg => "freq_dawg",
            Type::ShapeTable => "shapetable",
            Type::BigramDawg => "bigram_dawg",
            Type::UnambigDawg => "unambig_dawg",
            Type::ParamsModel => "params_model",
            Type::Lstm => "lstm",
            Type::LstmPuncDawg => "lstm-punc-dawg",
            Type::LstmSystemDawg => "lstm-word-dawg",
            Type::LstmNumberDawg => "lstm-number-dawg",
            Type::LstmUnicharset => "lstm-unicharset",
            Type::LstmRecoder => "lstm-recoder",
            Type::Version => "version",
        }
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A parsed `.traineddata` file. Entry slices alias the input buffer.
#[derive(Debug)]
pub struct Container<'a> {
    entries: [Option<&'a [u8]>; NUM_TYPES],
    swap: bool,
}

/// Parses Tesseract's native `.traineddata` layout. (Tesseract built with libarchive also accepts
/// zip/tar archives of component files; the published models all use the native layout.)
///
/// # Errors
///
/// `Error::ModelLimit` for a file over `limits.max_model_bytes`, `Error::Tessdata` for a malformed
/// header or offset table, `Error::AllocLimit` over `max_alloc`, `Error::OutOfMemory` if the allocator refuses.
pub fn parse_container<'a>(data: &'a [u8], limits: &Limits) -> Result<Container<'a>, Error> {
    parse_container_in(data, &Budget::new(limits))
}

pub(crate) fn parse_container_in<'a>(
    data: &'a [u8],
    budget: &Budget,
) -> Result<Container<'a>, Error> {
    if data.len() > budget.limits().max_model_bytes {
        return Err(Error::ModelLimit);
    }
    let mut r = Reader::new(data);
    let mut n = r
        .uint32()
        .map_err(|e| err(format!("reading entry count: {e}")))?;
    let swap = n > MAX_NUM_ENTRIES;
    if swap {
        n = n.swap_bytes();
        r.set_swap(true);
    }
    if n > MAX_NUM_ENTRIES {
        return Err(err(format!(
            "entry count {n} exceeds maximum {MAX_NUM_ENTRIES} in both byte orders"
        )));
    }
    let _offsets_hold = budget.hold(n as usize * std::mem::size_of::<i64>())?;
    let mut offsets = Vec::new();
    offsets
        .try_reserve_exact(n as usize)
        .map_err(|_| Error::OutOfMemory)?;
    for i in 0..n {
        offsets.push(
            r.int64()
                .map_err(|e| err(format!("reading offset {i}: {e}")))?,
        );
    }

    let size = data.len() as i64;
    let mut entries = [None; NUM_TYPES];
    for i in 0..offsets.len() {
        if offsets[i] < 0 {
            continue;
        }
        if offsets[i] > size {
            return Err(err(format!(
                "entry {i} offset {} past end of {size}-byte file",
                offsets[i]
            )));
        }
        let mut end = size;
        for j in i + 1..offsets.len() {
            if offsets[j] >= 0 {
                if offsets[j] > size {
                    return Err(err(format!(
                        "entry {j} offset {} past end of {size}-byte file",
                        offsets[j]
                    )));
                }
                end = offsets[j];
                break;
            }
        }
        if end < offsets[i] {
            return Err(err(format!("entry {i} has negative length")));
        }
        if i < NUM_TYPES {
            entries[i] = Some(&data[offsets[i] as usize..end as usize]);
        }
    }
    Ok(Container { entries, swap })
}

impl<'a> Container<'a> {
    /// Whether the source file was foreign-endian. Readers created for its entries must be told.
    pub fn swapped(&self) -> bool {
        self.swap
    }

    pub fn entry(&self, t: Type) -> Option<&'a [u8]> {
        self.entries[t as usize]
    }

    /// The component types in the file, in type order.
    pub fn present(&self) -> Vec<Type> {
        Type::ALL
            .iter()
            .copied()
            .filter(|&t| self.entries[t as usize].is_some())
            .collect()
    }

    /// The model's version string, or "" if the file predates it.
    ///
    /// # Errors
    ///
    /// `Error::AllocLimit` if the string does not fit in `limits.max_alloc`, `Error::OutOfMemory` if the allocator refuses.
    pub fn version(&self, limits: &Limits) -> Result<String, Error> {
        self.version_in(&Budget::new(limits))
    }

    fn version_in(&self, budget: &Budget) -> Result<String, Error> {
        match self.entries[Type::Version as usize] {
            None => Ok(String::new()),
            Some(b) => {
                let end = b
                    .iter()
                    .rposition(|&c| c != 0 && c != b'\n')
                    .map_or(0, |i| i + 1);
                lossy_string(&b[..end], budget)
            }
        }
    }
}

/// A fully loaded `.traineddata`: everything the LSTM recognition path reads, with every weight
/// value present.
#[derive(Debug)]
pub struct Model {
    pub version: String,
    pub swapped: bool,
    pub recognizer: Recognizer,
    pub unicharset: Unicharset,
    pub recoder: Recoder,
    /// The three lexicons `Dict::LoadLSTM` reads. `None` when the component is absent; a
    /// lexicon-free model still recognizes text, it just loses the dictionary-weighted beam.
    pub punc_dawg: Option<Dawg>,
    pub word_dawg: Option<Dawg>,
    pub number_dawg: Option<Dawg>,
}

impl Model {
    /// Parses a `.traineddata` file in Tesseract's native container layout and validates the
    /// invariants that bind its components together.
    ///
    /// The LSTM path requires `Lstm`, `LstmUnicharset` and `LstmRecoder`. A model missing either
    /// of the last two is rejected rather than mis-parsed: the embedded-charset layout is a
    /// pre-4.0 arrangement no stock model uses.
    ///
    /// One [`Budget`] covers the whole call; the parsed model is what stays charged when it
    /// returns. The cancel flag is checked between the components and at every layer and matrix.
    ///
    /// Enforces `limits.max_model_bytes` and `limits.max_alloc`; `limits.max_threads` does not apply
    /// (loading runs on the calling thread).
    ///
    /// # Errors
    ///
    /// `Error::ModelLimit` for a file over `limits.max_model_bytes`, `Error::AllocLimit` when the
    /// parsed model would not fit in `limits.max_alloc`, `Error::Cancelled`, else `Error::Tessdata`, `Error::OutOfMemory` if the allocator refuses.
    pub fn load(data: &[u8], limits: &Limits, cancel: &AtomicBool) -> Result<Model, Error> {
        let budget = Budget::new(limits);
        let c = parse_container_in(data, &budget).map_err(|e| wrap("parsing container", e))?;
        let required = |t: Type| {
            c.entry(t)
                .ok_or_else(|| err(format!("model has no {t} component")))
        };

        let recognizer = parse_recognizer(required(Type::Lstm)?, c.swapped(), &budget, cancel)
            .map_err(|e| wrap("lstm component", e))?;
        check_cancel(cancel)?;
        let unicharset = parse_unicharset(required(Type::LstmUnicharset)?, &budget)
            .map_err(|e| wrap("lstm-unicharset component", e))?;
        check_cancel(cancel)?;
        let recoder = Recoder::parse(required(Type::LstmRecoder)?, c.swapped(), &budget)
            .map_err(|e| wrap("lstm-recoder component", e))?;

        let mut dawgs = [None, None, None];
        for (slot, t) in dawgs.iter_mut().zip([
            Type::LstmPuncDawg,
            Type::LstmSystemDawg,
            Type::LstmNumberDawg,
        ]) {
            if let Some(b) = c.entry(t) {
                check_cancel(cancel)?;
                *slot = Some(
                    Dawg::parse(b, c.swapped(), &budget)
                        .map_err(|e| wrap(&format!("{t} component"), e))?,
                );
            }
        }
        let [punc_dawg, word_dawg, number_dawg] = dawgs;

        let m = Model {
            version: c.version_in(&budget)?,
            swapped: c.swapped(),
            recognizer,
            unicharset,
            recoder,
            punc_dawg,
            word_dawg,
            number_dawg,
        };
        m.validate()?;
        Ok(m)
    }

    /// The cross-component invariants. Each is free to check and each catches a whole class of
    /// mis-parse.
    fn validate(&self) -> Result<(), Error> {
        if self.recoder.size() != self.unicharset.size() {
            return Err(err(format!(
                "recoder has {} entries but the unicharset has {}; they are 1:1",
                self.recoder.size(),
                self.unicharset.size()
            )));
        }
        let sm = self
            .softmax()
            .ok_or_else(|| err("network has no softmax output layer"))?;
        // LSTMTrainer::InitNetwork builds the network with recoder_.code_range() as its output
        // count, so these must agree exactly.
        if sm.num_outputs != self.recoder.code_range() {
            return Err(err(format!(
                "softmax layer has {} outputs but the recoder's code range is {}",
                sm.num_outputs,
                self.recoder.code_range()
            )));
        }
        if self.recognizer.network.num_outputs != sm.num_outputs {
            return Err(err(format!(
                "root layer declares {} outputs but the softmax layer has {}",
                self.recognizer.network.num_outputs, sm.num_outputs
            )));
        }
        // UnicharCompress::DefragmentCodeValues deliberately relocates the null code to the top
        // of the range, so null_char_ is the LAST output index.
        if i64::from(self.recognizer.null_char) != self.recoder.code_range() as i64 - 1 {
            return Err(err(format!(
                "null_char is {} but the code range is {}; expected the blank at the last output index",
                self.recognizer.null_char,
                self.recoder.code_range()
            )));
        }
        for d in [&self.punc_dawg, &self.word_dawg, &self.number_dawg]
            .into_iter()
            .flatten()
        {
            if d.unicharset_size != self.unicharset.size() {
                return Err(err(format!(
                    "a dawg was built for a {}-entry unicharset but the model's has {}",
                    d.unicharset_size,
                    self.unicharset.size()
                )));
            }
        }
        Ok(())
    }

    /// The network's unique softmax output layer, or `None` if the graph does not have exactly
    /// one. Its `num_outputs` is the authoritative output count, never the number in the spec
    /// string. A model whose LSTM nests a softmax (`LSTMSoftmax`) has two, is rejected by
    /// `validate`, and that is the correct behaviour: the topology is not implemented.
    pub fn softmax(&self) -> Option<&Layer> {
        fn walk<'a>(l: &'a Layer, n: &mut usize, found: &mut Option<&'a Layer>) {
            if matches!(l.typ, LayerType::Softmax | LayerType::SoftmaxNoCtc) {
                *n += 1;
                *found = Some(l);
            }
            for c in &l.children {
                walk(c, n, found);
            }
        }
        let (mut n, mut found) = (0, None);
        walk(&self.recognizer.network, &mut n, &mut found);
        if n != 1 { None } else { found }
    }

    /// The network's output count, and therefore the width of one timestep of its output.
    pub fn num_outputs(&self) -> usize {
        self.recognizer.network.num_outputs
    }

    /// The network output index of the CTC blank.
    pub fn null_char(&self) -> usize {
        self.recognizer.null_char as usize
    }
}

fn wrap(what: &str, e: Error) -> Error {
    match e {
        Error::Tessdata(s) => Error::Tessdata(format!("{what}: {s}")),
        e => e,
    }
}

#[cfg(test)]
mod tests;
