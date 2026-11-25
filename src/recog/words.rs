//! Words and lines: confidence, word splitting and bounding boxes of a beam-decoded line.
//!
//! A translation of `RecodeBeamSearch::ExtractBestPathAsWords` and `::InitializeWord` in
//! `src/lstm/recodebeam.cpp`, `Tesseract::SearchWords` in `src/ccmain/linerec.cpp` and
//! `LTRResultIterator::Confidence` in `src/ccmain/ltrresultiterator.cpp`.
//!
//! Translated from Tesseract OCR (Apache-2.0, see NOTICE), through the Go port. The translation
//! is not verbatim.

use super::certainty::char_boundaries_into;
use super::{CERTAINTY_SCALE, Recognizer, Scored, Workspace};
use crate::imaging::{Bitmap, Rect};
use crate::pool::{headroom, set_joined};
use crate::tessdata::UNICHAR_SPACE;
use crate::{Budget, Error, Limits};
use std::sync::atomic::AtomicBool;

#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub text: String,
    pub bounds: Rect,
    /// Percent, 0 to 100.
    pub confidence: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Line {
    pub text: String,
    pub words: Vec<Word>,
    pub bounds: Rect,
    /// The least word confidence, percent; 0 for a line without words.
    pub confidence: f64,
}

/// Tesseract's certainty to a 0 to 100 confidence: `100 + 5 * certainty_scale * certainty`,
/// clamped.
pub fn confidence_percent(certainty: f64) -> f64 {
    let c = 100.0 + 5.0 * CERTAINTY_SCALE * certainty;
    c.clamp(0.0, 100.0)
}

pub(super) struct WordGroup {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) space_cert: f64,
}

/// The words of a run of symbols, split at spaces, one at a time. A word's `space_cert` is the
/// worse of the certainties of the spaces on either side.
pub(super) struct WordGroups<'a> {
    syms: &'a [Scored],
    i: usize,
    prev_space_cert: f64,
}

pub(super) fn word_groups(syms: &[Scored]) -> WordGroups<'_> {
    WordGroups {
        syms,
        i: 0,
        prev_space_cert: 0.0,
    }
}

impl Iterator for WordGroups<'_> {
    type Item = WordGroup;

    fn next(&mut self) -> Option<WordGroup> {
        let syms = self.syms;
        while self.i < syms.len() && syms[self.i].symbol.unichar_id == UNICHAR_SPACE {
            self.prev_space_cert = syms[self.i].certainty;
            self.i += 1;
        }
        if self.i >= syms.len() {
            return None;
        }
        let start = self.i;
        while self.i < syms.len() && syms[self.i].symbol.unichar_id != UNICHAR_SPACE {
            self.i += 1;
        }
        let mut space_cert = self.prev_space_cert;
        if self.i < syms.len() && syms[self.i].certainty < space_cert {
            space_cert = syms[self.i].certainty;
        }
        Some(WordGroup {
            start,
            end: self.i,
            space_cert,
        })
    }
}

/// `word_groups` as a vec.
#[cfg(test)]
pub(super) fn split_words(syms: &[Scored]) -> Vec<WordGroup> {
    word_groups(syms).collect()
}

pub(super) fn word_bounds(
    bounds: &[usize],
    start: usize,
    end: usize,
    scale: f64,
    line_box: Rect,
) -> Rect {
    let left = line_box.min.x + (bounds[start] as f64 * scale).floor() as isize;
    let mut right = line_box.min.x + (bounds[end] as f64 * scale).ceil() as isize;
    if right <= left {
        right = left + 1;
    }
    Rect::new(left, line_box.min.y, right, line_box.max.y)
}

impl Recognizer {
    /// Recognizes a line image (8bpp grey) with the beam search and returns its text, words and
    /// confidences. Word boxes span the line box vertically and map timesteps back to source x.
    ///
    /// One [`Budget`] covers the whole call: the scaled image, the network's activations, the
    /// beam, and every string and vec of the result. This is [`Recognizer::recognize_into`] on a
    /// fresh [`Workspace`] and a fresh [`Line`]; a caller that recognizes many lines should keep
    /// both and call that instead.
    ///
    /// Enforces `limits.max_alloc` and `limits.max_threads` (the forward pass and the scaling split
    /// their work over up to that many threads).
    ///
    /// # Errors
    ///
    /// `Error::PixelLimit` and `Error::AllocLimit` from `limits`, `Error::Cancelled`, else
    /// `Error::Recog` for an image or model that does not fit, `Error::OutOfMemory` if the allocator refuses.
    pub fn recognize(
        &self,
        img: &Bitmap,
        limits: &Limits,
        cancel: &AtomicBool,
    ) -> Result<Line, Error> {
        let mut line = Line {
            text: String::new(),
            words: Vec::new(),
            bounds: Rect::default(),
            confidence: 0.0,
        };
        self.recognize_into(&mut Workspace::default(), img, &mut line, limits, cancel)?;
        Ok(line)
    }

    /// `recognize`, into a `Line` and a [`Workspace`] the caller keeps from one call to the next.
    ///
    /// `line` is overwritten in place with what `recognize` would return: its text is cleared and
    /// refilled and its words reuse the `Word` entries (and their strings) it has, truncated or
    /// pushed to the count this line needs. After an error `line` is unspecified. With
    /// `limits.max_threads` 1, a call that has a warm `ws` and `line` (a few earlier calls on the same
    /// image, or on one of the same size that finds no more words and no longer strings, so that
    /// the beam's random border taps have not moved the need past the kept buffers) makes no
    /// allocator call at all; with more threads it makes small allocations bounded by the thread
    /// count (the pool's job boxes and the list of bands). When both calls succeed the result is the
    /// same as `recognize`'s, byte for byte, for every workspace, line, thread count and sequence of
    /// calls in which both sides succeed (a call that fails part way has drawn only part of the
    /// random border taps, so after one side alone fails the two generators are out of step). Under a `max_alloc` just below a fresh call's peak the warm call may succeed where
    /// `recognize` fails with `Error::AllocLimit` (it makes no doubling steps and reuses `line`).
    ///
    /// `max_alloc` is still the peak of each call: the capacity `ws` holds is charged to the call's
    /// [`Budget`] before anything grows, together with what the call grows. If that does not fit,
    /// `ws` is emptied and the call runs as on a fresh one. After `Error::AllocLimit` `ws` holds
    /// nothing; after any other error it is usable. `ws` keeps its memory until it is dropped.
    ///
    /// The retry after `Error::AllocLimit` rewinds the recognizer's random generator to its state at
    /// the start of the call, so concurrent calls on one `Recognizer` may repeat each other's draws.
    /// Give each worker thread its own `Recognizer` when the draws must not depend on that.
    ///
    /// Enforces `limits.max_alloc` and `limits.max_threads` as `recognize` does.
    ///
    /// # Errors
    ///
    /// As [`Recognizer::recognize`].
    pub fn recognize_into(
        &self,
        ws: &mut Workspace,
        img: &Bitmap,
        line: &mut Line,
        limits: &Limits,
        cancel: &AtomicBool,
    ) -> Result<(), Error> {
        let budget = Budget::new(limits);
        let mut warm = ws.bytes() > 0;
        let forced = std::mem::take(&mut ws.force_retry);
        let seed = self.net.rand.seed();
        // The kept buffers are held at their capacity for the whole call.
        if budget.charge(ws.bytes()).is_err() {
            *ws = Workspace::default();
            warm = false;
        }
        let mut r = self.recognize_with(ws, img, line, &budget, cancel);
        if forced && warm && r.is_ok() {
            r = Err(Error::AllocLimit);
        }
        if matches!(r, Err(Error::AllocLimit)) {
            let retries = ws.retries;
            *ws = Workspace::default();
            if warm {
                ws.retries = retries.saturating_add(1);
                // The kept buffers may be what did not fit: run as on a fresh workspace, from
                // the generator state this call started with. The rerun gets a new ledger but the
                // workers the first attempt built for itself, if any (rayon does not join a
                // dropped pool's threads, so a second pool could briefly double them).
                let again = Limits {
                    pool: limits.pool.clone().or_else(|| budget.own_pool_built()),
                    ..limits.clone()
                };
                drop(budget);
                self.net.rand.set_seed(seed);
                r = self.recognize_with(ws, img, line, &Budget::new(&again), cancel);
                if matches!(r, Err(Error::AllocLimit)) {
                    *ws = Workspace::default();
                }
            }
        }
        r
    }

    fn recognize_with(
        &self,
        ws: &mut Workspace,
        img: &Bitmap,
        line: &mut Line,
        budget: &Budget,
        cancel: &AtomicBool,
    ) -> Result<(), Error> {
        ws.scratch.reset();
        let (out, norm) = self.forward_in(img, budget, cancel, None, Some(&mut ws.scratch))?;
        let width = out.map.len();
        let scale_factor = norm.scale_factor;
        let decoded = self.beam_decode_into(&out, budget, cancel, &mut ws.beam, &mut ws.scored);
        ws.scratch.f32s.give(out.into_data());
        ws.scratch.f32s.give(norm.input.into_data());
        decoded?;
        let scored = ws.scored.as_slice();
        char_boundaries_into(scored, width, &mut ws.bounds, budget)?;
        let line_box = img.bounds();

        line.bounds = line_box;
        line.confidence = 100.0;
        set_joined(
            budget,
            &mut line.text,
            scored.iter().map(|s| s.symbol.text.as_str()),
        )?;
        let mut n = 0;
        for g in word_groups(scored) {
            let mut cert = 0.0f64;
            let texts = scored[g.start..g.end]
                .iter()
                .map(|s| s.symbol.text.as_str());
            for s in &scored[g.start..g.end] {
                if s.certainty < cert {
                    cert = s.certainty;
                }
            }
            if g.space_cert < cert {
                cert = g.space_cert;
            }
            let c = confidence_percent(cert);
            let bounds = word_bounds(&ws.bounds, g.start, g.end, scale_factor, line_box);
            match line.words.get_mut(n) {
                Some(w) => {
                    set_joined(budget, &mut w.text, texts)?;
                    w.bounds = bounds;
                    w.confidence = c;
                }
                None => {
                    let mut text = String::new();
                    set_joined(budget, &mut text, texts)?;
                    budget.push(
                        &mut line.words,
                        Word {
                            text,
                            bounds,
                            confidence: c,
                        },
                    )?;
                }
            }
            n += 1;
            if c < line.confidence {
                line.confidence = c;
            }
        }
        line.words.truncate(n);
        if n == 0 {
            line.confidence = 0.0;
        }
        // Last, so that these best-effort charges cannot starve a charge that the call needs.
        ws.beam.headroom(budget);
        ws.scored.headroom(budget);
        headroom(budget, &mut ws.bounds);
        Ok(())
    }
}
