//! The LSTM lexicon: a port of `Dict::def_letter_is_okay`, `Dict::default_dawgs`,
//! `Dict::GetStartingNode` and `Dict::char_for_dawg` from `src/dict/dict.cpp` and `dict.h` of
//! Tesseract OCR (Apache-2.0, see NOTICE).

use crate::tessdata::{Dawg, NO_EDGE, PATTERN_UNICHAR_ID, PROP_DIGIT, Unicharset};
use crate::{Budget, Error, Limits};

/// Which lexicon a dawg is, which decides how its unichar ids are folded before lookup and which
/// other dawgs may follow it.
///
/// `DAWG_TYPE_PATTERN` has no value on purpose: pattern dawgs come from `user-patterns` through a
/// separate API and no `.traineddata` LSTM component is one. `Dict::new` takes exactly three dawgs
/// and assigns these types itself, so an unhandled type is unreachable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DawgType {
    Punctuation,
    Word,
    Number,
}

/// Tesseract's `PermuterType`, restricted to the values the LSTM path can produce. Larger is
/// better; `letter_is_okay` takes the maximum over the surviving positions and then applies the
/// carry rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PermuterType(pub i32);

pub const NO_PERM: PermuterType = PermuterType(0);
pub const PUNC_PERM: PermuterType = PermuterType(1);
pub const NUMBER_PERM: PermuterType = PermuterType(6);
pub const SYSTEM_DAWG_PERM: PermuterType = PermuterType(8);
/// What the beam stamps on a non-dictionary hypothesis.
pub const TOP_CHOICE_PERM: PermuterType = PermuterType(2);

/// `DawgPosition`: a point in the lexicon frontier. A `dawg_index` or `punc_index` of -1 means
/// "not in that dawg", and a ref of `NO_EDGE` means "at the start of it".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    pub dawg_index: i32,
    pub dawg_ref: i64,
    pub punc_index: i32,
    pub punc_ref: i64,
    pub back_to_punc: bool,
}

const PUNC_IDX: usize = 0;
const WORD_IDX: usize = 1;
const NUMBER_IDX: usize = 2;

/// The LSTM lexicon: the punctuation dawg wrapping the word and number dawgs, exactly as
/// `Dict::LoadLSTM` assembles it.
#[derive(Debug, Clone, Copy)]
pub struct Dict<'a> {
    charset: &'a Unicharset,
    /// Index 0 punctuation, 1 word, 2 number.
    dawgs: [Option<&'a Dawg>; 3],
    types: [DawgType; 3],
    perms: [PermuterType; 3],
}

impl<'a> Dict<'a> {
    /// A dictionary view over `charset` and up to three dawgs (punctuation, word, number). It
    /// allocates nothing and takes no `Limits`.
    pub fn new(
        charset: &'a Unicharset,
        punc: Option<&'a Dawg>,
        word: Option<&'a Dawg>,
        number: Option<&'a Dawg>,
    ) -> Dict<'a> {
        Dict {
            charset,
            dawgs: [punc, word, number],
            types: [DawgType::Punctuation, DawgType::Word, DawgType::Number],
            perms: [PUNC_PERM, SYSTEM_DAWG_PERM, NUMBER_PERM],
        }
    }

    /// `Dict::default_dawgs`. The punctuation dawg seeds a position on its own; the word and
    /// number dawgs are subsumed by it whenever the punctuation dawg has a `kPatternUnicharID`
    /// edge out of its root, which every stock eng model does.
    ///
    /// # Errors
    ///
    /// `Error::AllocLimit` if the positions do not fit in `limits.max_alloc`,
    /// `Error::OutOfMemory` if the allocator refuses.
    pub fn start(&self, limits: &Limits) -> Result<Vec<Position>, Error> {
        self.start_in(&Budget::new(limits))
    }

    /// `start`, charging the positions to the caller's `budget` as the vec grows (the charge is
    /// the caller's to release).
    pub(crate) fn start_in(&self, budget: &Budget) -> Result<Vec<Position>, Error> {
        let mut out = Vec::new();
        self.start_into(&mut out, budget)?;
        Ok(out)
    }

    /// `start_in`, appending to `out` (the caller's, which keeps its capacity from one call to the
    /// next); the growth is charged to `budget`.
    pub(crate) fn start_into(&self, out: &mut Vec<Position>, budget: &Budget) -> Result<(), Error> {
        let punc_available = self.dawgs[PUNC_IDX]
            .is_some_and(|p| p.edge_char(0, PATTERN_UNICHAR_ID, true) != NO_EDGE);

        for (i, dw) in self.dawgs.iter().enumerate() {
            if dw.is_none() {
                continue;
            }
            if i == PUNC_IDX {
                budget.push(
                    out,
                    Position {
                        dawg_index: -1,
                        dawg_ref: NO_EDGE,
                        punc_index: i as i32,
                        punc_ref: NO_EDGE,
                        back_to_punc: false,
                    },
                )?;
                continue;
            }
            // Word and number are successors of punctuation, so they are only seeded
            // independently when punctuation cannot reach them.
            if !punc_available {
                budget.push(
                    out,
                    Position {
                        dawg_index: i as i32,
                        dawg_ref: NO_EDGE,
                        punc_index: -1,
                        punc_ref: NO_EDGE,
                        back_to_punc: false,
                    },
                )?;
            }
        }
        Ok(())
    }

    /// `Dict::def_letter_is_okay`.
    ///
    /// `prev` is the caller's carried permuter (`dawg_args->permuter`): pass `NO_PERM` for the
    /// first letter of a word and the previous return value thereafter. Returns the permuter,
    /// the surviving positions and whether the word may end here; `(NO_PERM, [], false)` when no
    /// position survives.
    ///
    /// # Errors
    ///
    /// `Error::AllocLimit` if the surviving positions (at most three per active one) do not fit in
    /// `limits.max_alloc`, `Error::OutOfMemory` if the allocator refuses.
    pub fn letter_is_okay(
        &self,
        prev: PermuterType,
        active: &[Position],
        unichar_id: usize,
        word_end: bool,
        limits: &Limits,
    ) -> Result<(PermuterType, Vec<Position>, bool), Error> {
        self.letter_is_okay_in(prev, active, unichar_id, word_end, &Budget::new(limits))
    }

    /// `letter_is_okay`, charging the surviving positions to the caller's `budget` as the vec
    /// grows (the charge is the caller's to release).
    pub(crate) fn letter_is_okay_in(
        &self,
        prev: PermuterType,
        active: &[Position],
        unichar_id: usize,
        word_end: bool,
        budget: &Budget,
    ) -> Result<(PermuterType, Vec<Position>, bool), Error> {
        let mut updated = Vec::new();
        let (perm, valid_end) =
            self.letter_is_okay_into(prev, active, unichar_id, word_end, &mut updated, budget)?;
        if perm == NO_PERM {
            budget.release(updated.capacity() * std::mem::size_of::<Position>());
            return Ok((NO_PERM, Vec::new(), false));
        }
        Ok((perm, updated, valid_end))
    }

    /// `letter_is_okay_in`, appending the surviving positions to `updated` (the caller's, which
    /// keeps its capacity from one call to the next) and returning the permuter and whether the
    /// word may end here. Nothing is appended when the permuter is `NO_PERM`. The growth is
    /// charged to `budget`.
    pub(crate) fn letter_is_okay_into(
        &self,
        prev: PermuterType,
        active: &[Position],
        unichar_id: usize,
        word_end: bool,
        updated: &mut Vec<Position>,
        budget: &Budget,
    ) -> Result<(PermuterType, bool), Error> {
        // A word may never contain the pattern wildcard; accepting it would let a pattern dawg
        // match arbitrary text.
        if unichar_id == PATTERN_UNICHAR_ID || unichar_id >= self.charset.size() {
            return Ok((NO_PERM, false));
        }

        let base = updated.len();
        let mut curr = NO_PERM;
        let mut valid_end = false;
        let mut add = |p: Position| -> Result<(), Error> {
            if !updated[base..].contains(&p) {
                budget.push(updated, p)?;
            }
            Ok(())
        };

        for pos in active {
            let punc = usize::try_from(pos.punc_index)
                .ok()
                .and_then(|i| self.dawgs.get(i).copied().flatten());
            let dw = usize::try_from(pos.dawg_index)
                .ok()
                .and_then(|i| self.dawgs.get(i).copied().flatten());
            if punc.is_none() && dw.is_none() {
                continue;
            }

            let Some(dw) = dw else {
                // In the punctuation dawg with no core dawg chosen yet.
                let Some(punc) = punc else { continue };
                let punc_node = starting_node(punc, pos.punc_ref);
                let trans = punc.edge_char(punc_node, PATTERN_UNICHAR_ID, word_end);
                if trans != NO_EDGE {
                    for si in self.successors(pos.punc_index as usize) {
                        let Some(sd) = self.dawgs[si] else { continue };
                        let ch = self.char_for_dawg(unichar_id, si);
                        let e = sd.edge_char(0, ch, word_end);
                        if e != NO_EDGE {
                            add(Position {
                                dawg_index: si as i32,
                                dawg_ref: e,
                                punc_index: pos.punc_index,
                                punc_ref: trans,
                                back_to_punc: false,
                            })?;
                            curr = curr.max(self.perms[si]);
                            if sd.end_of_word(e) && punc.end_of_word(trans) {
                                valid_end = true;
                            }
                        }
                    }
                }
                let e = punc.edge_char(punc_node, unichar_id, word_end);
                if e != NO_EDGE {
                    add(Position {
                        dawg_index: -1,
                        dawg_ref: NO_EDGE,
                        punc_index: pos.punc_index,
                        punc_ref: e,
                        back_to_punc: false,
                    })?;
                    curr = curr.max(PUNC_PERM);
                    if punc.end_of_word(e) {
                        valid_end = true;
                    }
                }
                continue;
            };

            if let Some(punc) = punc
                && pos.dawg_ref != NO_EDGE
                && dw.end_of_word(pos.dawg_ref)
            {
                // The core word can end here; see whether punctuation continues.
                let punc_node = starting_node(punc, pos.punc_ref);
                let e = punc.edge_char(punc_node, unichar_id, word_end);
                if e != NO_EDGE {
                    add(Position {
                        dawg_index: pos.dawg_index,
                        dawg_ref: pos.dawg_ref,
                        punc_index: pos.punc_index,
                        punc_ref: e,
                        back_to_punc: true,
                    })?;
                    curr = curr.max(self.perms[pos.dawg_index as usize]);
                    if punc.end_of_word(e) {
                        valid_end = true;
                    }
                }
            }

            if pos.back_to_punc {
                continue;
            }

            // DAWG_TYPE_PATTERN would be handled here, before the edge lookup; there is no such
            // type.
            let node = starting_node(dw, pos.dawg_ref);
            let mut e = NO_EDGE;
            if node != NO_EDGE {
                e = dw.edge_char(
                    node,
                    self.char_for_dawg(unichar_id, pos.dawg_index as usize),
                    word_end,
                );
            }
            if e == NO_EDGE {
                continue;
            }
            if word_end
                && punc.is_some_and(|p| pos.punc_ref != NO_EDGE && !p.end_of_word(pos.punc_ref))
            {
                // The punctuation constraint is not satisfied at the end of a word.
                continue;
            }
            curr = curr.max(self.perms[pos.dawg_index as usize]);
            if dw.end_of_word(e)
                && punc.is_none_or(|p| pos.punc_ref == NO_EDGE || p.end_of_word(pos.punc_ref))
            {
                valid_end = true;
            }
            add(Position {
                dawg_index: pos.dawg_index,
                dawg_ref: e,
                punc_index: pos.punc_index,
                punc_ref: pos.punc_ref,
                back_to_punc: false,
            })?;
        }

        // The permuter carry, from the tail of def_letter_is_okay:
        //
        //   if (dawg_args->permuter == NO_PERM || curr_perm == NO_PERM ||
        //       (curr_perm != PUNC_PERM && dawg_args->permuter != COMPOUND_PERM)) {
        //     dawg_args->permuter = curr_perm;
        //   }
        //
        // COMPOUND_PERM is unreachable on the LSTM path, so that conjunct is always true. When
        // this letter's own best permuter is PUNC_PERM and a core word was already established,
        // the OLD permuter is preserved: "(the)" reports SYSTEM_DAWG_PERM at the closing paren.
        let out = if prev == NO_PERM || curr == NO_PERM || curr != PUNC_PERM {
            curr
        } else {
            prev
        };
        if out == NO_PERM {
            updated.truncate(base);
            return Ok((NO_PERM, false));
        }
        Ok((out, valid_end))
    }

    /// `kDawgSuccessors`: punctuation may be followed by the word and number dawgs, and those two
    /// only by punctuation.
    fn successors(&self, index: usize) -> impl Iterator<Item = usize> + '_ {
        [WORD_IDX, NUMBER_IDX]
            .into_iter()
            .filter(move |&i| index == PUNC_IDX && self.dawgs[i].is_some())
    }

    /// `Dict::char_for_dawg`: inside the number dawg every digit folds to the pattern wildcard,
    /// so "2024" and "1999" share one path.
    fn char_for_dawg(&self, unichar_id: usize, dawg_index: usize) -> usize {
        if self.types.get(dawg_index) != Some(&DawgType::Number) {
            return unichar_id;
        }
        match self.charset.char(unichar_id) {
            Some(c) if c.properties & PROP_DIGIT != 0 => PATTERN_UNICHAR_ID,
            _ => unichar_id,
        }
    }
}

/// `Dict::GetStartingNode`: `NO_EDGE` -> 0 (beginning to explore the dawg); else
/// `next_node(edge)`, with 0 remapped to `NO_EDGE` (end of word).
///
/// The `next_node == 0` remapping is load-bearing: a squished dawg writes 0 to mean "every word
/// through this edge terminates here", never "return to the root". Returning 0 would re-enter
/// the root and let the lexicon accept arbitrary continuations of any complete word.
fn starting_node(dw: &Dawg, r: i64) -> i64 {
    if r == NO_EDGE {
        return 0;
    }
    let node = dw.next_node(r);
    if node == 0 { NO_EDGE } else { node }
}

#[cfg(test)]
mod tests;
