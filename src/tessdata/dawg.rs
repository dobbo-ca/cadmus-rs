//! `Dawg` (`SquishedDawg`): a directed acyclic word graph, as stored in the lstm-punc-dawg,
//! lstm-word-dawg and lstm-number-dawg components. From `dawg.cpp` and `dawg.h`.
//!
//! Translated from Tesseract OCR (Apache-2.0, see NOTICE), through the Go port. The translation
//! is not verbatim.

use super::{Reader, err};
use crate::{Budget, Error};

/// `kDawgMagicNumber`, src/dict/dawg.h:113.
const DAWG_MAGIC: i16 = 42;

// Edge-record flags, all shifted left by `flag_start_bit`. src/dict/dawg.h:77.
/// Last edge of this node's run.
const MARKER_FLAG: u64 = 1;
/// Set for a backward edge.
const DIRECTION_FLAG: u64 = 2;
const WORD_END_FLAG: u64 = 4;
const NUM_FLAG_BITS: u32 = 3;

/// `read_squished_dawg`'s resource-exhaustion guard.
const MAX_DAWG_EDGES: u32 = 50_000_000;

/// Tesseract's `NO_EDGE`.
pub const NO_EDGE: i64 = -1;

/// `Dawg::kPatternUnicharID`: inside a punctuation or number dawg, unichar id 0 on an edge is a
/// wildcard, not the space character.
pub const PATTERN_UNICHAR_ID: usize = 0;

/// A node is identified by the index of its first edge; its edges are the consecutive run ending
/// at the edge whose MARKER flag is set. Node 0 is the root, so a `next_node` of 0 means "no
/// successor" rather than "back to the root".
#[derive(Debug, Clone)]
pub struct Dawg {
    pub unicharset_size: usize,
    edges: Vec<u64>,
    letter_mask: u64,
    flag_start_bit: u32,
    next_node_start_bit: u32,
}

impl Dawg {
    /// Deserializes one lstm-*-dawg component. `swap` comes from `Container::swapped`.
    ///
    /// Charged to `budget`, so it enforces its `max_alloc`.
    ///
    /// # Errors
    ///
    /// `Error::Tessdata` for a malformed component, `Error::AllocLimit` over `max_alloc`,
    /// `Error::OutOfMemory` if the allocator refuses.
    pub fn parse(data: &[u8], swap: bool, budget: &Budget) -> Result<Dawg, Error> {
        let mut r = Reader::new(data);
        r.set_swap(swap);
        let magic = r
            .int16()
            .map_err(|e| err(format!("reading dawg magic: {e}")))?;
        if magic != DAWG_MAGIC {
            return Err(err(format!("bad dawg magic {magic}, want {DAWG_MAGIC}")));
        }
        let size = r
            .uint32()
            .map_err(|e| err(format!("reading dawg unicharset size: {e}")))?;
        if size == 0 || size > i32::MAX as u32 {
            return Err(err(format!("bad dawg unicharset size {size}")));
        }
        let num_edges = r
            .uint32()
            .map_err(|e| err(format!("reading dawg edge count: {e}")))?;
        if num_edges == 0 {
            return Err(err("dawg has 0 edges"));
        }
        if num_edges > MAX_DAWG_EDGES {
            return Err(err(format!(
                "dawg edge count {num_edges} exceeds the {MAX_DAWG_EDGES} hard limit"
            )));
        }
        if num_edges as usize > r.remaining() / 8 {
            return Err(err(format!(
                "dawg declares {num_edges} edges but only {} bytes remain",
                r.remaining()
            )));
        }
        let mut edges = budget.vec(num_edges as usize, 0u64)?;
        for (i, e) in edges.iter_mut().enumerate() {
            *e = r
                .uint64()
                .map_err(|e| err(format!("reading dawg edge {i}: {e}")))?;
        }
        if r.remaining() != 0 {
            return Err(err(format!(
                "{} bytes left unconsumed after the dawg",
                r.remaining()
            )));
        }
        // Dawg::init: flag_start_bit is the BIT LENGTH of unicharset_size (Tesseract's CeilLog2 is
        // misnamed: CeilLog2(64) == 7).
        let flag_start_bit = 32 - size.leading_zeros();
        let d = Dawg {
            unicharset_size: size as usize,
            edges,
            letter_mask: (1u64 << flag_start_bit) - 1,
            flag_start_bit,
            next_node_start_bit: flag_start_bit + NUM_FLAG_BITS,
        };
        d.validate()?;
        Ok(d)
    }

    /// The invariants `SquishedDawg::write_squished_dawg` guarantees: forward edges only, and
    /// every `next_node` indexes the written array.
    fn validate(&self) -> Result<(), Error> {
        for (i, &e) in self.edges.iter().enumerate() {
            if e & (DIRECTION_FLAG << self.flag_start_bit) != 0 {
                return Err(err(format!(
                    "dawg edge {i} is a backward edge; only written (forward-only) dawgs are supported"
                )));
            }
            let id = (e & self.letter_mask) as usize;
            if id >= self.unicharset_size {
                return Err(err(format!(
                    "dawg edge {i} has unichar id {id} outside the {}-entry unicharset",
                    self.unicharset_size
                )));
            }
            let next = e >> self.next_node_start_bit;
            if next >= self.edges.len() as u64 {
                return Err(err(format!(
                    "dawg edge {i} points at node {next} beyond the {} edges in the file",
                    self.edges.len()
                )));
            }
        }
        Ok(())
    }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Builds a dawg from edge records, for tests.
    #[cfg(test)]
    pub(super) fn from_edges(unicharset_size: usize, edges: Vec<u64>) -> Dawg {
        let flag_start_bit = usize::BITS - unicharset_size.leading_zeros();
        Dawg {
            unicharset_size,
            edges,
            letter_mask: (1u64 << flag_start_bit) - 1,
            flag_start_bit,
            next_node_start_bit: flag_start_bit + NUM_FLAG_BITS,
        }
    }

    #[cfg(test)]
    pub(super) fn flag_start_bit(&self) -> u32 {
        self.flag_start_bit
    }

    #[cfg(test)]
    pub(super) fn edge_record(&self, i: usize) -> u64 {
        self.edges[i]
    }

    #[cfg(test)]
    pub(super) fn set_edge(&mut self, i: usize, v: u64) {
        self.edges[i] = v;
    }

    #[cfg(test)]
    pub(super) fn empty_slot_value(&self) -> u64 {
        self.empty_slot()
    }

    fn next_node_of(&self, edge: usize) -> usize {
        (self.edges[edge] >> self.next_node_start_bit) as usize
    }

    /// `SquishedDawg::edge_char_of` by linear scan from `node`, for the word-dawg lookups.
    ///
    /// Tesseract binary-searches node 0 and linear-scans every other node. The two agree
    /// whenever the root's unichar ids are distinct, which holds for every shipped dawg (eng's
    /// root has 67 edges, sorted, with no repeats; `node0_is_sorted_and_duplicate_free` checks
    /// it on a real model). If a model ever fails that, this must grow the binary search.
    fn edge_of(&self, node: usize, id: usize, word_end: bool) -> Option<usize> {
        for e in node..self.edges.len() {
            let rec = self.edges[e];
            if (rec & self.letter_mask) as usize == id
                && (!word_end || rec & (WORD_END_FLAG << self.flag_start_bit) != 0)
            {
                return Some(e);
            }
            if rec & (MARKER_FLAG << self.flag_start_bit) != 0 {
                break;
            }
        }
        None
    }

    /// Whether the dawg holds the complete word spelled by `ids` (`Dawg::word_in_dawg`).
    ///
    /// WORD DAWGS ONLY: every unichar id on an edge is matched literally, which is wrong for
    /// the punctuation and number dawgs, where `PATTERN_UNICHAR_ID` is a wildcard.
    pub fn contains(&self, ids: &[usize]) -> bool {
        self.prefix_in_dawg(ids, true)
    }

    /// Whether `ids` is a prefix of at least one word in the dawg (`prefix_in_dawg` with
    /// `requires_complete = false`). Word dawgs only.
    pub fn has_prefix(&self, ids: &[usize]) -> bool {
        self.prefix_in_dawg(ids, false)
    }

    fn prefix_in_dawg(&self, ids: &[usize], requires_complete: bool) -> bool {
        let Some((last, init)) = ids.split_last() else {
            return !requires_complete;
        };
        let mut node = 0;
        for &id in init {
            let Some(e) = self.edge_of(node, id, false) else {
                return false;
            };
            // next_node == 0 means every word through this edge terminates here.
            node = self.next_node_of(e);
            if node == 0 {
                return false;
            }
        }
        self.edge_of(node, *last, requires_complete).is_some()
    }

    /// The record of `edge`. An edge that is not in the table (such as `NO_EDGE`, or a caller's
    /// made-up reference) reads as the zero record: character 0, next node 0, not a word end.
    fn record(&self, edge: i64) -> u64 {
        usize::try_from(edge)
            .ok()
            .and_then(|e| self.edges.get(e))
            .copied()
            .unwrap_or(0)
    }

    /// `unichar_id_from_edge_rec`: the character this edge consumes.
    pub fn unichar_id(&self, edge: i64) -> usize {
        (self.record(edge) & self.letter_mask) as usize
    }

    /// `next_node_from_edge_rec`.
    pub fn next_node(&self, edge: i64) -> i64 {
        (self.record(edge) >> self.next_node_start_bit) as i64
    }

    /// `end_of_word_from_edge_rec`: this edge terminates a word.
    pub fn end_of_word(&self, edge: i64) -> bool {
        self.record(edge) & (WORD_END_FLAG << self.flag_start_bit) != 0
    }

    /// `marker_flag_from_edge_rec`: this edge closes its node's run.
    fn last_edge(&self, edge: i64) -> bool {
        self.record(edge) & (MARKER_FLAG << self.flag_start_bit) != 0
    }

    /// `edge_occupied`'s sentinel: an edge record equal to `next_node_mask_` is a hole left by
    /// the squishing pass, not an edge.
    fn empty_slot(&self) -> u64 {
        u64::MAX << self.next_node_start_bit
    }

    /// `SquishedDawg::edge_char_of`: the edge leaving `node` on `unichar_id`, or `NO_EDGE`.
    ///
    /// Node 0 is scanned linearly rather than binary-searched (see `edge_of`), and the direction
    /// bit is not tested, as `edge_char_of` does not test it either (`parse` rejects backward
    /// edges). The empty-slot guard is reproduced, because an empty slot's unichar id is 0, the
    /// wildcard a punctuation-dawg probe asks for.
    pub fn edge_char(&self, node: i64, unichar_id: usize, word_end: bool) -> i64 {
        if node < 0 || node >= self.edges.len() as i64 {
            return NO_EDGE;
        }
        if self.edges[node as usize] == self.empty_slot() {
            return NO_EDGE;
        }
        let mut e = node;
        while e < self.edges.len() as i64 {
            if self.unichar_id(e) == unichar_id && (!word_end || self.end_of_word(e)) {
                return e;
            }
            if self.last_edge(e) {
                break;
            }
            e += 1;
        }
        NO_EDGE
    }
}
