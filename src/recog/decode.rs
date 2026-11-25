//! Loading a recognizer and the forward pass with greedy CTC decoding.
//!
//! A translation of `RecodeBeamSearch::ExtractBestPathAsLabels` in `src/lstm/recodebeam.cpp` and
//! `LSTMRecognizer::DecodeLabels` in `src/lstm/lstmrecognizer.cpp`.
//!
//! Translated from Tesseract OCR (Apache-2.0, see NOTICE), through the Go port. The translation
//! is not verbatim.

use super::normalize::normalize_pooled;
use super::{Network, Normalized, err};
use crate::dict::Dict;
use crate::imaging::{Bitmap, check_cancel};
use crate::nn::{Ctx, Scratch, Tap, Tensor};
use crate::tessdata::{self, Dawg, Recoder, Type, Unicharset};
use crate::{Budget, Error, Limits};
use std::sync::atomic::AtomicBool;

/// A recognizer: the network, charset, recoder and, when the model has dawgs, the lexicon.
#[derive(Debug)]
pub struct Recognizer {
    pub net: Network,
    pub charset: Unicharset,
    pub recoder: Recoder,
    punc: Option<Dawg>,
    word: Option<Dawg>,
    number: Option<Dawg>,
    lexicon: bool,
}

/// One decoded symbol: its unichar id and text and the timesteps `[start, end)` it spans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Symbol {
    pub unichar_id: usize,
    pub text: String,
    pub start: usize,
    pub end: usize,
}

impl Recognizer {
    /// Loads a `.traineddata` file. The dictionary starts enabled when the model has any of the
    /// three dawgs; `set_lexicon(false)` turns it off.
    ///
    /// One [`Budget`] covers the call. The parsed network is held while its runnable copy is
    /// built and released before the rest of the model is read, so the peak is the model plus one
    /// copy of the weights. The cancel flag is checked between the components and at every layer.
    ///
    /// Enforces `limits.max_model_bytes` and `limits.max_alloc`; `limits.max_threads` does not apply
    /// (loading runs on the calling thread).
    ///
    /// # Errors
    ///
    /// `Error::ModelLimit` for a file over `limits.max_model_bytes`, `Error::AllocLimit` when the
    /// model does not fit in `limits.max_alloc`, `Error::Cancelled`, else `Error::Recog` for a
    /// model this recognizer cannot run, `Error::OutOfMemory` if the allocator refuses.
    pub fn new(model: &[u8], limits: &Limits, cancel: &AtomicBool) -> Result<Recognizer, Error> {
        let budget = Budget::new(limits);
        let c = tessdata::parse_container_in(model, &budget).map_err(wrap)?;
        let lstm = c
            .entry(Type::Lstm)
            .ok_or_else(|| err("model has no lstm component"))?;
        let before = budget.used();
        let rec = tessdata::parse_recognizer(lstm, c.swapped(), &budget, cancel).map_err(wrap)?;
        let parsed = budget.used().saturating_sub(before);
        let net = Network::build_in(&rec, &budget, cancel)?;
        drop(rec);
        budget.release(parsed);
        check_cancel(cancel)?;

        let cs_entry = c.entry(Type::LstmUnicharset).ok_or_else(|| {
            err("model has no lstm-unicharset component; the embedded-charset layout is not supported")
        })?;
        let charset = tessdata::parse_unicharset(cs_entry, &budget).map_err(wrap)?;
        check_cancel(cancel)?;
        let rc_entry = c
            .entry(Type::LstmRecoder)
            .ok_or_else(|| err("model has no lstm-recoder component"))?;
        let recoder = Recoder::parse(rc_entry, c.swapped(), &budget).map_err(wrap)?;

        if recoder.code_range() != net.num_outputs {
            return Err(err(format!(
                "recoder code range {} does not match the {} network outputs",
                recoder.code_range(),
                net.num_outputs
            )));
        }
        if recoder.size() != charset.size() {
            return Err(err(format!(
                "recoder has {} entries but the unicharset has {}",
                recoder.size(),
                charset.size()
            )));
        }
        if recoder.max_code_len() != 1 {
            return Err(err(format!(
                "recoder has codes up to {} long; single-code recoders only are supported",
                recoder.max_code_len()
            )));
        }
        match recoder.encode(tessdata::UNICHAR_SPACE) {
            Some([0]) => {}
            _ => return Err(err("space was garbled in recoding")),
        }

        let mut dawgs: [Option<Dawg>; 3] = [None, None, None];
        for (slot, t) in dawgs.iter_mut().zip([
            Type::LstmPuncDawg,
            Type::LstmSystemDawg,
            Type::LstmNumberDawg,
        ]) {
            let Some(e) = c.entry(t) else { continue };
            check_cancel(cancel)?;
            let d = Dawg::parse(e, c.swapped(), &budget).map_err(wrap)?;
            if d.unicharset_size != charset.size() {
                return Err(err(format!(
                    "the {t} was built for a {}-entry unicharset but the model's has {}",
                    d.unicharset_size,
                    charset.size()
                )));
            }
            *slot = Some(d);
        }
        let [punc, word, number] = dawgs;
        let lexicon = punc.is_some() || word.is_some() || number.is_some();
        Ok(Recognizer {
            net,
            charset,
            recoder,
            punc,
            word,
            number,
            lexicon,
        })
    }

    /// Enables or disables the dictionary for `beam_decode` and `recognize`. Enabling is a no-op
    /// for a model without dawgs.
    pub fn set_lexicon(&mut self, enabled: bool) {
        self.lexicon = enabled && self.has_dawgs();
    }

    fn has_dawgs(&self) -> bool {
        self.punc.is_some() || self.word.is_some() || self.number.is_some()
    }

    /// The active dictionary, if any.
    pub fn dict(&self) -> Option<Dict<'_>> {
        if self.lexicon {
            Some(Dict::new(
                &self.charset,
                self.punc.as_ref(),
                self.word.as_ref(),
                self.number.as_ref(),
            ))
        } else {
            None
        }
    }

    /// Prepares `img` (8bpp grey) and runs the network, returning the softmax output, one row per
    /// timestep, and the normalization.
    ///
    /// Enforces `limits.max_alloc` and `limits.max_threads` (the fully connected layers split their
    /// timesteps into bands, one per thread).
    ///
    /// # Errors
    ///
    /// `Error::PixelLimit` and `Error::AllocLimit` from `limits`, `Error::Cancelled`, else
    /// `Error::Recog` for an image or network that does not fit, `Error::OutOfMemory` if the allocator refuses.
    pub fn forward(
        &self,
        img: &Bitmap,
        limits: &Limits,
        cancel: &AtomicBool,
    ) -> Result<(Tensor, Normalized), Error> {
        self.forward_tap(img, limits, cancel, None)
    }

    /// `forward`, with `tap` called with every layer's output (but the `Input`'s) as the graph
    /// runs, for diffing activations against Tesseract's.
    ///
    /// As [`Recognizer::forward`]: it enforces `max_alloc`, `max_threads` and the plane caps of
    /// `limits`.
    ///
    /// # Errors
    ///
    /// As [`Recognizer::forward`], `Error::OutOfMemory` if the allocator refuses.
    pub fn forward_tap(
        &self,
        img: &Bitmap,
        limits: &Limits,
        cancel: &AtomicBool,
        tap: Option<Tap<'_>>,
    ) -> Result<(Tensor, Normalized), Error> {
        self.forward_in(img, &Budget::new(limits), cancel, tap, None)
    }

    /// `forward_tap` on the call's own `budget`: the normalized input and the activations of the
    /// network are charged to it, and the returned tensors stay charged. With a `pool` they are
    /// its buffers (the caller gives them back) and only the growth of its capacity is charged.
    pub(crate) fn forward_in(
        &self,
        img: &Bitmap,
        budget: &Budget,
        cancel: &AtomicBool,
        tap: Option<Tap<'_>>,
        mut pool: Option<&mut Scratch>,
    ) -> Result<(Tensor, Normalized), Error> {
        let norm = normalize_pooled(
            img,
            self.net.input_height,
            self.net.x_scale,
            &self.net.rand,
            budget,
            cancel,
            pool.as_deref_mut(),
        )?;
        // Shorten the observer's lifetime to the local ones of the context.
        let tap: Option<Tap<'_>> = match tap {
            Some(t) => Some(&mut *t),
            None => None,
        };
        let mut cx = Ctx {
            budget,
            tap,
            cancel: Some(cancel),
            pool,
        };
        let out = self
            .net
            .root
            .forward(&norm.input, &mut cx)
            .map_err(|e| match e {
                Error::Network(s) => err(format!("forward: {s}")),
                e => e,
            })?;
        if out.features != self.net.num_outputs {
            return Err(err(format!(
                "network produced {} outputs, want {}",
                out.features, self.net.num_outputs
            )));
        }
        Ok((out, norm))
    }

    /// Best-path CTC decoding: the argmax of each timestep with repeats collapsed and the blank
    /// dropped.
    ///
    /// The symbols and their text are charged to `limits.max_alloc` as they are made, and the
    /// cancel flag is checked at every timestep.
    ///
    /// # Errors
    ///
    /// `Error::AllocLimit`, `Error::Cancelled`, `Error::Recog` for an output that does not fit
    /// the network or a code the charset cannot decode, `Error::OutOfMemory` if the allocator refuses.
    pub fn greedy_decode(
        &self,
        out: &Tensor,
        limits: &Limits,
        cancel: &AtomicBool,
    ) -> Result<Vec<Symbol>, Error> {
        self.greedy_decode_in(out, &Budget::new(limits), cancel)
    }

    /// `greedy_decode` on the call's own `budget`; the symbols stay charged.
    pub(crate) fn greedy_decode_in(
        &self,
        out: &Tensor,
        budget: &Budget,
        cancel: &AtomicBool,
    ) -> Result<Vec<Symbol>, Error> {
        if out.features == 0 || out.features != self.net.num_outputs || !out.is_consistent() {
            return Err(err(format!(
                "output has {} features, want {}",
                out.features, self.net.num_outputs
            )));
        }
        let mut syms: Vec<Symbol> = Vec::new();
        let mut prev: Option<usize> = None;
        for t in 0..out.map.len() {
            check_cancel(cancel)?;
            let code = argmax(out.row(t));
            if prev != Some(code) && code != self.net.null_char {
                let id = self
                    .recoder
                    .decode_unichar(&[code as i32])
                    .filter(|&id| id < self.charset.size())
                    .ok_or_else(|| {
                        err(format!(
                            "code {code} at t={t} does not decode to a unichar id inside the charset"
                        ))
                    })?;
                let text = self.charset.text(id);
                budget.charge(text.len())?;
                budget.push(
                    &mut syms,
                    Symbol {
                        unichar_id: id,
                        text: text.to_string(),
                        start: t,
                        end: t + 1,
                    },
                )?;
            } else if prev == Some(code)
                && code != self.net.null_char
                && let Some(last) = syms.last_mut()
            {
                last.end = t + 1;
            }
            prev = Some(code);
        }
        Ok(syms)
    }

    /// The text of the best path through `img`.
    ///
    /// Enforces `max_alloc`, `max_threads` and the plane caps of `limits`.
    ///
    /// # Errors
    ///
    /// As [`Recognizer::forward`], `Error::OutOfMemory` if the allocator refuses.
    pub fn recognize_text(
        &self,
        img: &Bitmap,
        limits: &Limits,
        cancel: &AtomicBool,
    ) -> Result<String, Error> {
        let budget = Budget::new(limits);
        let (out, _) = self.forward_in(img, &budget, cancel, None, None)?;
        let before = budget.used();
        let syms = self.greedy_decode_in(&out, &budget, cancel)?;
        let held = budget.used().saturating_sub(before);
        let text = join_texts(syms.iter().map(|s| s.text.as_str()), &budget)?;
        drop(syms);
        budget.release(held);
        Ok(text)
    }
}

/// The concatenation of `texts`, charged to `budget` for its exact length before it is made.
pub(super) fn join_texts<'a>(
    texts: impl Iterator<Item = &'a str> + Clone,
    budget: &Budget,
) -> Result<String, Error> {
    let len: usize = texts.clone().map(str::len).sum();
    budget.charge(len)?;
    let mut s = String::new();
    if s.try_reserve_exact(len).is_err() {
        budget.release(len);
        return Err(Error::OutOfMemory);
    }
    for t in texts {
        s.push_str(t);
    }
    Ok(s)
}

fn wrap(e: Error) -> Error {
    match e {
        Error::Tessdata(s) => err(s),
        e => e,
    }
}

/// The index of the first maximum of `row` (0 for an empty row).
pub(super) fn argmax(row: &[f32]) -> usize {
    let Some(&first) = row.first() else {
        return 0;
    };
    let (mut best, mut best_i) = (first, 0);
    for (i, &v) in row.iter().enumerate().skip(1) {
        if v > best {
            best = v;
            best_i = i;
        }
    }
    best_i
}

#[cfg(test)]
impl Recognizer {
    /// A recognizer without dawgs, from parts, for tests.
    pub(super) fn for_test(net: Network, charset: Unicharset, recoder: Recoder) -> Recognizer {
        Recognizer {
            net,
            charset,
            recoder,
            punc: None,
            word: None,
            number: None,
            lexicon: false,
        }
    }
}
