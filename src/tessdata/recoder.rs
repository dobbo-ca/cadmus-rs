//! `Recoder` (`UnicharCompress`): maps unichar ids to the network's output code sequences and
//! back. From `unicharcompress.cpp` and `unicharcompress.h`.
//!
//! Translated from Tesseract OCR (Apache-2.0, see NOTICE), through the Go port. The translation
//! is not verbatim.

use super::{MAX_STRING_LEN, Reader, UNICHAR_SPACE, err};
use crate::{Budget, Error};
use std::collections::HashMap;

/// `RecodedCharID::kMaxCodeLen`, src/ccutil/unicharcompress.h:37.
const MAX_CODE_LEN: usize = 9;

/// For eng every code has length 1 and the mapping is a near-permutation: the only collapse is
/// unichar ids 1 ("Joined") and 2 ("|Broken|0|1") both encoding to 110, which is also the CTC
/// blank. CJK and Indic models use genuinely multi-code sequences.
#[derive(Debug, Clone)]
pub struct Recoder {
    codes: Vec<Vec<i32>>,
    code_range: usize,
    max_len: usize,
    decoder: HashMap<Vec<i32>, usize>,
    valid_start: Vec<bool>,
}

impl Recoder {
    /// Deserializes a `.traineddata` lstm-recoder component. `swap` comes from
    /// `Container::swapped`.
    ///
    /// Charged to `budget`, so it enforces its `max_alloc`.
    ///
    /// # Errors
    ///
    /// `Error::Tessdata` for a malformed component, `Error::AllocLimit` over `max_alloc`,
    /// `Error::OutOfMemory` if the allocator refuses.
    pub fn parse(data: &[u8], swap: bool, budget: &Budget) -> Result<Recoder, Error> {
        let mut r = Reader::new(data);
        r.set_swap(swap);
        let n = r
            .uint32()
            .map_err(|e| err(format!("reading recoder entry count: {e}")))?;
        // TFile::DeSerialize(std::vector<T>&) rejects counts above 50,000,000.
        if n > MAX_STRING_LEN {
            return Err(err(format!("implausible recoder entry count {n}")));
        }
        // The smallest possible entry is 5 bytes (int8 self_normalized + uint32 length, zero
        // codes), so the buffer itself is the tighter bound.
        let n = n as usize;
        if n > r.remaining() / 5 {
            return Err(err(format!(
                "recoder declares {n} entries but only {} bytes remain",
                r.remaining()
            )));
        }
        let mut codes: Vec<Vec<i32>> = Vec::new();
        budget.reserve(&mut codes, n)?;
        let mut max_len = 0;
        for i in 0..n {
            // self_normalized_ is int8 and never byte-swapped; it carries no information.
            r.int8()
                .map_err(|e| err(format!("recoder entry {i} self_normalized: {e}")))?;
            let length = r
                .uint32()
                .map_err(|e| err(format!("recoder entry {i} length: {e}")))?
                as usize;
            if length > MAX_CODE_LEN {
                return Err(err(format!(
                    "recoder entry {i} length {length} exceeds kMaxCodeLen {MAX_CODE_LEN}"
                )));
            }
            budget.charge(4 * length)?;
            let mut code = Vec::with_capacity(length);
            for j in 0..length {
                let v = r
                    .int32()
                    .map_err(|e| err(format!("recoder entry {i} code {j}: {e}")))?;
                if v < 0 {
                    return Err(err(format!("recoder entry {i} code {j} is negative ({v})")));
                }
                code.push(v);
            }
            max_len = max_len.max(length);
            codes.push(code);
        }
        if r.remaining() != 0 {
            return Err(err(format!(
                "{} bytes left unconsumed after the recoder",
                r.remaining()
            )));
        }

        // UnicharCompress::ComputeCodeRange: max code + 1.
        let code_range = codes
            .iter()
            .flatten()
            .map(|&v| v as usize + 1)
            .max()
            .unwrap_or(0);
        let mut rc = Recoder {
            codes,
            code_range,
            max_len,
            decoder: HashMap::new(),
            valid_start: Vec::new(),
        };
        rc.setup_decoder(budget)?;
        Ok(rc)
    }

    /// `UnicharCompress::SetupDecoder`. It iterates unichar ids in ascending order and
    /// overwrites, so when two ids share a code the HIGHER id wins: in eng, ids 1 and 2 both
    /// encode to [110], and [110] decodes to 2.
    ///
    /// Deliberate divergence for a zero-length entry: Tesseract registers it in `decoder_` and
    /// sets `is_valid_start_[code(0)]` before its `if (len == 0) continue`. It is skipped here:
    /// an empty code sequence cannot be decoded. No shipped model has one.
    fn setup_decoder(&mut self, budget: &Budget) -> Result<(), Error> {
        self.valid_start = budget.vec(self.code_range, false)?;
        // The table is sized once for every entry, so it never grows. Its buckets are a power of
        // two of at least 8/7 of the entries, each one a key, a value and a control byte; the
        // keys are copies of the codes.
        let n = self.codes.len();
        let buckets = n.saturating_mul(16) / 7 + 8;
        let table = buckets
            .saturating_mul(std::mem::size_of::<(Vec<i32>, usize)>() + 1)
            .saturating_add(16);
        let keys: usize = self.codes.iter().map(|c| 4 * c.len()).sum();
        budget.charge(table.saturating_add(keys))?;
        self.decoder
            .try_reserve(n)
            .map_err(|_| Error::OutOfMemory)?;
        for (id, code) in self.codes.iter().enumerate() {
            if code.is_empty() {
                continue;
            }
            self.decoder.insert(code.clone(), id);
            self.valid_start[code[0] as usize] = true;
        }
        // LSTMRecognizer::LoadRecoder's own check: "Space was garbled in recoding!!"
        if self.codes.is_empty()
            || self.codes[UNICHAR_SPACE].is_empty()
            || self.codes[UNICHAR_SPACE][0] != UNICHAR_SPACE as i32
        {
            return Err(err(format!(
                "space was garbled in recoding: unichar {UNICHAR_SPACE} encodes to {:?}",
                self.encode(UNICHAR_SPACE)
            )));
        }
        Ok(())
    }

    /// The number of entries, one per unichar id.
    pub fn size(&self) -> usize {
        self.codes.len()
    }

    /// The number of distinct code values, and therefore the network's output count. It is the
    /// authority: `NetworkBuilder::ParseOutput` overrides the spec string's output count with it.
    pub fn code_range(&self) -> usize {
        self.code_range
    }

    /// The longest code sequence. 1 means the mapping is flat and a decoder can skip the
    /// multi-code lookahead entirely.
    pub fn max_code_len(&self) -> usize {
        self.max_len
    }

    /// The code sequence for a unichar id, or `None` if out of range.
    pub fn encode(&self, unichar_id: usize) -> Option<&[i32]> {
        self.codes.get(unichar_id).map(Vec::as_slice)
    }

    /// The unichar id for a complete code sequence.
    pub fn decode_unichar(&self, code: &[i32]) -> Option<usize> {
        if code.is_empty() || code.len() > MAX_CODE_LEN {
            return None;
        }
        self.decoder.get(code).copied()
    }

    /// Whether a code value can begin a sequence.
    pub fn is_valid_first_code(&self, code: i32) -> bool {
        usize::try_from(code)
            .ok()
            .and_then(|c| self.valid_start.get(c))
            .copied()
            .unwrap_or(false)
    }
}
