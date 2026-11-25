//! The buffers one `recognize` call needs, kept for the next call.

use super::Scored;
use super::beam::BeamBufs;
use crate::nn::Scratch;
use crate::pool::set_str;
use crate::{Budget, Error};

/// Everything one `Recognizer::recognize_into` call needs, kept from one call to the next so that
/// a call on a warm workspace allocates nothing (with `Limits::max_threads` 1; see
/// [`Recognizer::recognize_into`](super::Recognizer::recognize_into)).
///
/// It holds the scaled and normalized line, the output and scratch tensors of every layer of the
/// network, the LSTM's state and gate vectors, the beam search's nodes, positions, steps and path,
/// the scored symbols and their strings, and the character boundaries. A workspace is not tied to
/// a recognizer or an image size: it grows to the largest call it has served and keeps that
/// memory until it is dropped. One workspace serves one call at a time; give each worker thread
/// its own.
#[derive(Debug, Default)]
pub struct Workspace {
    pub(super) scratch: Scratch,
    pub(super) beam: BeamBufs,
    pub(super) scored: ScoredBuf,
    pub(super) bounds: Vec<usize>,
    /// Test hook: the next warm call treats its first attempt as `Error::AllocLimit` and retries.
    pub(super) force_retry: bool,
    /// The warm retries this workspace has run (test hook).
    pub(super) retries: u64,
}

impl Workspace {
    /// Test hook, not part of the API: the next call on a warm workspace runs, then retries as if
    /// it had hit `Error::AllocLimit`, so the retry path can be tested.
    #[doc(hidden)]
    pub fn force_retry(&mut self) {
        self.force_retry = true;
    }

    /// Test hook, not part of the API: how many warm retries this workspace has run.
    #[doc(hidden)]
    pub fn retries(&self) -> u64 {
        self.retries
    }

    /// The heap the workspace holds, by capacity: what a call charges to its `Budget` before it
    /// grows anything.
    pub(super) fn bytes(&self) -> usize {
        self.scratch.bytes()
            + self.beam.bytes()
            + self.scored.bytes()
            + self.bounds.capacity() * std::mem::size_of::<usize>()
    }
}

/// The symbols a beam search found: a vec of which only the first `len` are live, so that the
/// strings of the rest keep their capacity for the next call.
#[derive(Debug, Default)]
pub(super) struct ScoredBuf {
    items: Vec<Scored>,
    len: usize,
}

impl ScoredBuf {
    pub(super) fn clear(&mut self) {
        self.len = 0;
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(super) fn as_slice(&self) -> &[Scored] {
        &self.items[..self.len]
    }

    pub(super) fn last_mut(&mut self) -> Option<&mut Scored> {
        self.items[..self.len].last_mut()
    }

    /// Appends a symbol: `span` is its `(start, end)` timesteps, `score` its `(certainty,
    /// rating)`. Only growth is charged to `budget`.
    pub(super) fn push(
        &mut self,
        budget: &Budget,
        unichar_id: usize,
        text: &str,
        span: (usize, usize),
        score: (f64, f64),
    ) -> Result<(), Error> {
        if self.len == self.items.len() {
            budget.push(
                &mut self.items,
                Scored {
                    symbol: super::Symbol {
                        unichar_id,
                        text: String::new(),
                        start: span.0,
                        end: span.1,
                    },
                    certainty: score.0,
                    rating: score.1,
                },
            )?;
        }
        let item = &mut self.items[self.len];
        item.symbol.unichar_id = unichar_id;
        item.symbol.start = span.0;
        item.symbol.end = span.1;
        item.certainty = score.0;
        item.rating = score.1;
        set_str(budget, &mut item.symbol.text, text)?;
        self.len += 1;
        Ok(())
    }

    /// `pool::headroom` for the vec of symbols.
    pub(super) fn headroom(&mut self, budget: &Budget) {
        crate::pool::headroom(budget, &mut self.items);
    }

    /// The live symbols as a fresh vec, charged to `budget`.
    pub(super) fn to_vec(&self, budget: &Budget) -> Result<Vec<Scored>, Error> {
        let mut v = Vec::new();
        budget.reserve(&mut v, self.len)?;
        for s in self.as_slice() {
            budget.charge(s.symbol.text.len())?;
            v.push(s.clone());
        }
        Ok(v)
    }

    /// The heap held, by capacity.
    pub(super) fn bytes(&self) -> usize {
        self.items.capacity() * std::mem::size_of::<Scored>()
            + self
                .items
                .iter()
                .map(|s| s.symbol.text.capacity())
                .sum::<usize>()
    }
}
