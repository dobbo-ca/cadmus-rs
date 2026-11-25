//! The recoded beam search: a port of the single-code, no-extended-script part of
//! `RecodeBeamSearch` (`src/lstm/recodebeam.cpp`, Apache-2.0, see NOTICE).
//!
//! Nodes live in an arena and refer to their predecessor by index. `push_node` overwrites a node
//! in place when a better one with the same identity arrives, as the C++ `*m = *n` does, so every
//! path through the old node sees the new contents. A node's dictionary positions are a range of a
//! second arena, so that the whole search runs in vecs that a `Workspace` keeps from one call to
//! the next.

use super::workspace::ScoredBuf;
use super::{
    CERT_OFFSET, CERTAINTY_SCALE, DICT_RATIO, MIN_CERTAINTY, Recognizer, Scored, err,
    prob_to_certainty,
};
use crate::dict::{Dict, NO_PERM, PermuterType, Position, TOP_CHOICE_PERM};
use crate::imaging::check_cancel;
use crate::nn::Tensor;
use crate::pool::{extend_charged, grow, refill};
use crate::tessdata::{UNICHAR_SPACE, Unicharset};
use crate::{Budget, Error, Limits};
use std::sync::atomic::AtomicBool;

const BEAM_WIDTH: usize = 5;
const WORST_DICT_CERT: f64 = -25.0 / CERTAINTY_SCALE;

// NodeContinuation
const NC_ANYTHING: usize = 0;
const NC_ONLY_DUP: usize = 1;
const NC_NO_DUP: usize = 2;
const NC_COUNT: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TopNState {
    Top2,
    TopN,
    AlsoRan,
}

const TOP_N_STATES: [TopNState; 3] = [TopNState::Top2, TopNState::TopN, TopNState::AlsoRan];

#[derive(Debug, Clone, Copy)]
struct Node {
    code: usize,
    /// -1 for the blank (`INVALID_UNICHAR_ID`).
    unichar_id: isize,
    permuter: PermuterType,
    start_of_dawg: bool,
    // Set as in the C++ (`start_of_word`) but never read by this port's decoder.
    #[allow(dead_code)]
    start_of_word: bool,
    end_of_word: bool,
    duplicate: bool,
    certainty: f64,
    score: f64,
    prev: Option<usize>,
    /// The start and length of the node's positions in the position arena.
    dawgs: (usize, usize),
    code_hash: u64,
}

/// No positions.
const NO_POSITIONS: (usize, usize) = (0, 0);

/// A beam: at most `BEAM_WIDTH` nodes, plus the one that is pushed before the worst is dropped.
#[derive(Debug, Clone, Copy, Default)]
struct Heap {
    items: [usize; BEAM_WIDTH + 1],
    len: usize,
}

impl Heap {
    fn len(&self) -> usize {
        self.len
    }

    fn as_slice(&self) -> &[usize] {
        &self.items[..self.len]
    }

    fn push(&mut self, n: usize) {
        if let Some(slot) = self.items.get_mut(self.len) {
            *slot = n;
            self.len += 1;
        }
    }

    fn swap_remove(&mut self, i: usize) {
        self.len -= 1;
        self.items[i] = self.items[self.len];
    }
}

#[derive(Debug, Clone, Default)]
struct BeamStep {
    beams: [Heap; 2 * NC_COUNT],
    initial: [Option<Node>; NC_COUNT],
}

/// The vecs of a beam search, kept by a `Workspace` between calls.
#[derive(Debug, Default)]
pub(super) struct BeamBufs {
    nodes: Vec<Node>,
    positions: Vec<Position>,
    active: Vec<Position>,
    steps: Vec<BeamStep>,
    path: Vec<usize>,
    final_codes: Vec<usize>,
    unichars: Vec<isize>,
    top_n_flags: Vec<TopNState>,
}

impl BeamBufs {
    /// `pool::headroom` for the arenas that grow with the line.
    pub(super) fn headroom(&mut self, budget: &Budget) {
        crate::pool::headroom(budget, &mut self.nodes);
        crate::pool::headroom(budget, &mut self.positions);
    }

    /// The heap the vecs hold, by capacity.
    pub(super) fn bytes(&self) -> usize {
        self.nodes.capacity() * std::mem::size_of::<Node>()
            + (self.positions.capacity() + self.active.capacity()) * std::mem::size_of::<Position>()
            + self.steps.capacity() * std::mem::size_of::<BeamStep>()
            + (self.path.capacity() + self.final_codes.capacity()) * std::mem::size_of::<usize>()
            + self.unichars.capacity() * std::mem::size_of::<isize>()
            + self.top_n_flags.capacity() * std::mem::size_of::<TopNState>()
    }
}

fn beam_index(is_dawg: bool, cont: usize) -> usize {
    if is_dawg { NC_COUNT + cont } else { cont }
}

fn continuation_of(index: usize) -> usize {
    index % NC_COUNT
}

fn is_dawg_index(index: usize) -> bool {
    index >= NC_COUNT
}

pub(super) struct Beam<'a> {
    /// Every growth of the arenas is charged to this before it is made.
    budget: &'a Budget,
    cs: &'a Unicharset,
    dict: Option<Dict<'a>>,
    null_char: usize,
    code_range: usize,
    final_codes: Vec<usize>,
    unichars: Vec<isize>,
    space_delimited: bool,
    nodes: Vec<Node>,
    /// The positions of every node, by range.
    positions: Vec<Position>,
    /// The positions of the node a letter continues, copied out of `positions` (which the new
    /// ones are appended to).
    active: Vec<Position>,
    top_n_flags: Vec<TopNState>,
    top_code: isize,
    second_code: isize,
}

impl<'a> Beam<'a> {
    /// A search over `num_outputs` codes in the vecs of `bufs`, which it takes (`finish` gives
    /// them back).
    pub(super) fn new(
        r: &'a Recognizer,
        num_outputs: usize,
        budget: &'a Budget,
        bufs: &mut BeamBufs,
    ) -> Result<Beam<'a>, Error> {
        let null_char = r.net.null_char;
        if null_char >= num_outputs {
            return Err(err(format!(
                "null char {null_char} is outside the {num_outputs} network outputs"
            )));
        }
        // The per-output tables are charged for what they hold: 8 bytes of unichar id and 1 of
        // flag; `final_codes` has at most one entry per unichar. The flags double as the set of
        // codes seen while `final_codes` is made.
        let mut b = Beam {
            budget,
            cs: &r.charset,
            dict: r.dict(),
            null_char,
            code_range: r.recoder.code_range(),
            final_codes: std::mem::take(&mut bufs.final_codes),
            unichars: std::mem::take(&mut bufs.unichars),
            space_delimited: true,
            nodes: std::mem::take(&mut bufs.nodes),
            positions: std::mem::take(&mut bufs.positions),
            active: std::mem::take(&mut bufs.active),
            top_n_flags: std::mem::take(&mut bufs.top_n_flags),
            top_code: -1,
            second_code: -1,
        };
        refill(budget, &mut b.unichars, num_outputs, -1isize)?;
        refill(budget, &mut b.top_n_flags, num_outputs, TopNState::AlsoRan)?;
        grow(
            budget,
            &mut b.final_codes,
            r.recoder.size().min(num_outputs),
        )?;
        b.nodes.clear();
        b.positions.clear();
        if b.dict.is_some() {
            // Room for the positions of a node, so that one with more does not grow the vec.
            grow(budget, &mut b.active, 64)?;
        }
        for id in 0..r.recoder.size() {
            let codes = r.recoder.encode(id).unwrap_or(&[]);
            if codes.len() != 1 {
                return Err(err(format!(
                    "unichar {id} encodes to {} codes; the beam supports single-code recoders only",
                    codes.len()
                )));
            }
            let c = codes[0] as usize;
            if c >= num_outputs {
                return Err(err(format!(
                    "unichar {id} encodes to code {c}, outside the {num_outputs} network outputs"
                )));
            }
            if b.top_n_flags[c] == TopNState::Top2 {
                continue;
            }
            b.top_n_flags[c] = TopNState::Top2;
            b.final_codes.push(c);
        }
        b.top_n_flags.fill(TopNState::AlsoRan);
        for &c in &b.final_codes {
            if c == null_char {
                continue;
            }
            let id = r
                .recoder
                .decode_unichar(&[c as i32])
                .filter(|&id| id < r.charset.size())
                .ok_or_else(|| {
                    err(format!(
                        "code {c} does not decode to a unichar id inside the charset"
                    ))
                })?;
            b.unichars[c] = id as isize;
        }
        if b.dict.is_some() {
            for id in 0..r.charset.size() {
                if !r.charset.is_space_delimited(id) {
                    b.space_delimited = false;
                    break;
                }
            }
        }
        Ok(b)
    }

    /// Gives the vecs back to `bufs`.
    pub(super) fn finish(self, bufs: &mut BeamBufs) {
        bufs.final_codes = self.final_codes;
        bufs.unichars = self.unichars;
        bufs.nodes = self.nodes;
        bufs.positions = self.positions;
        bufs.active = self.active;
        bufs.top_n_flags = self.top_n_flags;
    }

    /// `ComputeTopN`: the top `BEAM_WIDTH` codes of a row, the top two flagged `Top2`.
    fn compute_top_n(&mut self, row: &[f32]) {
        self.top_n_flags.fill(TopNState::AlsoRan);
        self.top_code = -1;
        self.second_code = -1;
        let mut top = [0usize; BEAM_WIDTH];
        let mut n = 0;
        for i in 0..row.len() {
            let v = row[i];
            let mut pos = 0;
            while pos < n && v <= row[top[pos]] {
                pos += 1;
            }
            if pos >= BEAM_WIDTH {
                continue;
            }
            if n < BEAM_WIDTH {
                n += 1;
            }
            top.copy_within(pos..n - 1, pos + 1);
            top[pos] = i;
        }
        let top = &top[..n];
        for (i, &c) in top.iter().enumerate() {
            self.top_n_flags[c] = if i < 2 {
                TopNState::Top2
            } else {
                TopNState::TopN
            };
        }
        if let Some(&c) = top.first() {
            self.top_code = c as isize;
        }
        if let Some(&c) = top.get(1) {
            self.second_code = c as isize;
        }
        self.top_n_flags[self.null_char] = TopNState::Top2;
    }

    fn decode_step(
        &mut self,
        row: &[f32],
        t: usize,
        steps: &[BeamStep],
    ) -> Result<BeamStep, Error> {
        let mut step = BeamStep::default();
        if t == 0 {
            self.continue_context(
                None,
                beam_index(false, NC_ANYTHING),
                row,
                TopNState::Top2,
                &mut step,
            )?;
            if self.dict.is_some() {
                self.continue_context(
                    None,
                    beam_index(true, NC_ANYTHING),
                    row,
                    TopNState::Top2,
                    &mut step,
                )?;
            }
            return Ok(step);
        }
        let prev = &steps[t - 1];
        let mut total = 0;
        for &tn in &TOP_N_STATES {
            if total != 0 {
                break;
            }
            for index in 0..prev.beams.len() {
                let h = &prev.beams[index];
                for i in (0..h.len()).rev() {
                    self.continue_context(Some(h.items[i]), index, row, tn, &mut step)?;
                }
            }
            for index in 0..step.beams.len() {
                if continuation_of(index) == NC_ANYTHING {
                    total += step.beams[index].len();
                }
            }
        }
        for c in 0..NC_COUNT {
            if let Some(n) = step.initial[c].take() {
                let idx = beam_index(true, c);
                push_node(self.budget, &mut self.nodes, &mut step.beams[idx], n)?;
            }
        }
        Ok(step)
    }

    fn continue_context(
        &mut self,
        prev: Option<usize>,
        index: usize,
        row: &[f32],
        flag: TopNState,
        step: &mut BeamStep,
    ) -> Result<(), Error> {
        let use_dawgs = is_dawg_index(index);
        let prev_cont = continuation_of(index);
        if let Some(p) = prev {
            let (p_code, p_uni) = (self.nodes[p].code, self.nodes[p].unichar_id);
            if self.top_n_flags[p_code] == flag {
                if prev_cont != NC_NO_DUP {
                    let cert = prob_to_certainty(f64::from(row[p_code])) + CERT_OFFSET;
                    self.push_dup_or_no_dawg(
                        true,
                        p_code,
                        p_uni,
                        cert,
                        use_dawgs,
                        NC_ANYTHING,
                        prev,
                        step,
                    )?;
                }
                if prev_cont == NC_ANYTHING && flag == TopNState::Top2 && p_code != self.null_char {
                    let cert =
                        prob_to_certainty(f64::from(row[p_code]) + f64::from(row[self.null_char]))
                            + CERT_OFFSET;
                    self.push_dup_or_no_dawg(
                        true, p_code, p_uni, cert, use_dawgs, NC_NO_DUP, prev, step,
                    )?;
                }
            }
            if prev_cont == NC_ONLY_DUP {
                return Ok(());
            }
        }
        for ci in 0..self.final_codes.len() {
            let code = self.final_codes[ci];
            if self.top_n_flags[code] != flag {
                continue;
            }
            let p_code = prev.map(|p| self.nodes[p].code);
            if p_code == Some(code) {
                continue;
            }
            let mut cert = prob_to_certainty(f64::from(row[code])) + CERT_OFFSET;
            if cert < MIN_CERTAINTY && code != self.null_char {
                continue;
            }
            let id = self.unichars[code];
            self.continue_unichar(code, id, cert, use_dawgs, NC_ANYTHING, prev, step)?;
            if flag == TopNState::Top2 && code != self.null_char {
                let mut prob = f64::from(row[code]) + f64::from(row[self.null_char]);
                if let Some(pc) = p_code {
                    let (c, tc, sc) = (code as isize, self.top_code, self.second_code);
                    let pc_i = pc as isize;
                    if prev_cont == NC_ANYTHING
                        && pc != self.null_char
                        && ((pc_i == tc && c == sc) || (c == tc && pc_i == sc))
                    {
                        prob += f64::from(row[pc]);
                    }
                }
                cert = prob_to_certainty(prob) + CERT_OFFSET;
                self.continue_unichar(code, id, cert, use_dawgs, NC_ONLY_DUP, prev, step)?;
            }
        }
        Ok(())
    }

    fn continue_unichar(
        &mut self,
        code: usize,
        unichar_id: isize,
        cert: f64,
        use_dawgs: bool,
        cont: usize,
        prev: Option<usize>,
        step: &mut BeamStep,
    ) -> Result<(), Error> {
        if use_dawgs {
            if cert > WORST_DICT_CERT {
                self.continue_dawg(code, unichar_id, cert, cont, prev, step)?;
            }
            return Ok(());
        }
        self.push_heap(
            beam_index(false, cont),
            code,
            unichar_id,
            TOP_CHOICE_PERM,
            (false, false, false, false),
            cert * DICT_RATIO,
            prev,
            NO_POSITIONS,
            step,
        )?;
        if self.dict.is_none() {
            return Ok(());
        }
        // unichar_id may be -1 (the blank): an invalid id is "space delimited" (-1 as usize is
        // out of range for the charset).
        let space_delimited = self.cs.is_space_delimited(unichar_id as usize);
        if (unichar_id == UNICHAR_SPACE as isize && cert > WORST_DICT_CERT) || !space_delimited {
            let mut dawg_cert = cert;
            let mut permuter = TOP_CHOICE_PERM;
            if unichar_id == UNICHAR_SPACE as isize {
                permuter = NO_PERM;
            } else {
                dawg_cert *= DICT_RATIO;
            }
            self.push_initial_dawg(
                code, unichar_id, permuter, false, false, dawg_cert, cont, prev, step,
            )?;
        }
        Ok(())
    }

    fn continue_dawg(
        &mut self,
        code: usize,
        unichar_id: isize,
        cert: f64,
        cont: usize,
        prev: Option<usize>,
        step: &mut BeamStep,
    ) -> Result<(), Error> {
        let dawg_idx = beam_index(true, cont);
        let no_dawg_idx = beam_index(false, cont);
        if unichar_id < 0 {
            self.push_heap(
                dawg_idx,
                code,
                unichar_id,
                NO_PERM,
                (false, false, false, false),
                cert,
                prev,
                NO_POSITIONS,
                step,
            )?;
            return Ok(());
        }
        let mut score = cert;
        if let Some(p) = prev {
            score += self.nodes[p].score;
        }
        if step.beams[dawg_idx].len() >= BEAM_WIDTH
            && score <= worst_score(&self.nodes, &step.beams[dawg_idx])
            && step.beams[no_dawg_idx].len() >= BEAM_WIDTH
            && score <= worst_score(&self.nodes, &step.beams[no_dawg_idx])
        {
            return Ok(());
        }
        let uid = unichar_id as usize;
        // The nearest previous node that is a real, non-duplicate unichar.
        let mut uni_prev = prev;
        while let Some(u) = uni_prev {
            if self.nodes[u].unichar_id < 0 || self.nodes[u].duplicate {
                uni_prev = self.nodes[u].prev;
            } else {
                break;
            }
        }
        if uid == UNICHAR_SPACE {
            if let Some(u) = uni_prev
                && self.nodes[u].end_of_word
            {
                let perm = self.nodes[u].permuter;
                self.push_initial_dawg(
                    code, unichar_id, perm, false, false, cert, cont, prev, step,
                )?;
                self.push_heap(
                    no_dawg_idx,
                    code,
                    unichar_id,
                    perm,
                    (false, false, false, false),
                    cert,
                    prev,
                    NO_POSITIONS,
                    step,
                )?;
            }
            return Ok(());
        }
        if let Some(u) = uni_prev {
            let un = &self.nodes[u];
            if un.start_of_dawg
                && un.unichar_id != UNICHAR_SPACE as isize
                && self.cs.is_space_delimited(un.unichar_id as usize)
                && self.cs.is_space_delimited(uid)
            {
                return Ok(()); // Can't break words between space delimited chars.
            }
        }
        let Some(dict) = self.dict else { return Ok(()) };
        let budget = self.budget;
        // The active positions are copied out of the arena, which the new ones are appended to.
        self.active.clear();
        let word_start = match uni_prev {
            None => {
                dict.start_into(&mut self.active, budget)?;
                true
            }
            Some(u) if self.nodes[u].dawgs.1 != 0 => {
                let (at, len) = self.nodes[u].dawgs;
                if let Some(src) = self.positions.get(at..at + len) {
                    extend_charged(budget, &mut self.active, src)?;
                }
                self.nodes[u].start_of_dawg
            }
            Some(_) => return Ok(()), // Can't continue if not a dict word.
        };
        let mark = self.positions.len();
        let (permuter, valid_end) = dict.letter_is_okay_into(
            NO_PERM,
            &self.active,
            uid,
            false,
            &mut self.positions,
            budget,
        )?;
        if permuter == NO_PERM {
            return Ok(());
        }
        let updated = (mark, self.positions.len() - mark);
        if !self.push_heap(
            dawg_idx,
            code,
            unichar_id,
            permuter,
            (false, word_start, valid_end, false),
            cert,
            prev,
            updated,
            step,
        )? {
            // The node was not kept, and nothing was added after its positions.
            self.positions.truncate(mark);
        }
        if valid_end && !self.space_delimited {
            self.push_initial_dawg(
                code, unichar_id, permuter, word_start, true, cert, cont, prev, step,
            )?;
            self.push_heap(
                no_dawg_idx,
                code,
                unichar_id,
                permuter,
                (false, word_start, true, false),
                cert,
                prev,
                NO_POSITIONS,
                step,
            )?;
        }
        Ok(())
    }

    fn push_initial_dawg(
        &mut self,
        code: usize,
        unichar_id: isize,
        permuter: PermuterType,
        start: bool,
        end: bool,
        cert: f64,
        cont: usize,
        prev: Option<usize>,
        step: &mut BeamStep,
    ) -> Result<(), Error> {
        let mut score = cert;
        if let Some(p) = prev {
            score += self.nodes[p].score;
        }
        if let Some(best) = &step.initial[cont]
            && score <= best.score
        {
            return Ok(());
        }
        let dawgs = match self.dict {
            Some(d) => {
                let at = self.positions.len();
                d.start_into(&mut self.positions, self.budget)?;
                (at, self.positions.len() - at)
            }
            None => NO_POSITIONS,
        };
        let code_hash = self.code_hash_of(code, false, prev);
        // A node it replaces leaves its positions in the arena.
        step.initial[cont] = Some(Node {
            code,
            unichar_id,
            permuter,
            start_of_dawg: true,
            start_of_word: start,
            end_of_word: end,
            duplicate: false,
            certainty: cert,
            score,
            prev,
            dawgs,
            code_hash,
        });
        Ok(())
    }

    fn push_dup_or_no_dawg(
        &mut self,
        dup: bool,
        code: usize,
        unichar_id: isize,
        cert: f64,
        use_dawgs: bool,
        cont: usize,
        prev: Option<usize>,
        step: &mut BeamStep,
    ) -> Result<(), Error> {
        if use_dawgs {
            if cert > WORST_DICT_CERT {
                let permuter = prev.map_or(NO_PERM, |p| self.nodes[p].permuter);
                self.push_heap(
                    beam_index(true, cont),
                    code,
                    unichar_id,
                    permuter,
                    (false, false, false, dup),
                    cert,
                    prev,
                    NO_POSITIONS,
                    step,
                )?;
            }
            return Ok(());
        }
        let cert = cert * DICT_RATIO;
        if cert >= MIN_CERTAINTY || code == self.null_char {
            let permuter = prev.map_or(TOP_CHOICE_PERM, |p| self.nodes[p].permuter);
            self.push_heap(
                beam_index(false, cont),
                code,
                unichar_id,
                permuter,
                (false, false, false, dup),
                cert,
                prev,
                NO_POSITIONS,
                step,
            )?;
        }
        Ok(())
    }

    /// `flags` is `(dawg_start, word_start, end, dup)`. Whether the node was kept.
    fn push_heap(
        &mut self,
        heap_idx: usize,
        code: usize,
        unichar_id: isize,
        permuter: PermuterType,
        flags: (bool, bool, bool, bool),
        cert: f64,
        prev: Option<usize>,
        dawgs: (usize, usize),
        step: &mut BeamStep,
    ) -> Result<bool, Error> {
        let mut score = cert;
        if let Some(p) = prev {
            score += self.nodes[p].score;
        }
        let heap = &mut step.beams[heap_idx];
        if heap.len() >= BEAM_WIDTH && score <= worst_score(&self.nodes, heap) {
            return Ok(false);
        }
        let (dawg_start, word_start, end, dup) = flags;
        let node = Node {
            code,
            unichar_id,
            permuter,
            start_of_dawg: dawg_start,
            start_of_word: word_start,
            end_of_word: end,
            duplicate: dup,
            certainty: cert,
            score,
            prev,
            dawgs,
            code_hash: self.code_hash_of(code, dup, prev),
        };
        push_node(self.budget, &mut self.nodes, heap, node)
    }

    /// `RecodeBeamSearch::ComputeCodeHash`.
    fn code_hash_of(&self, code: usize, dup: bool, prev: Option<usize>) -> u64 {
        let mut hash = prev.map_or(0, |p| self.nodes[p].code_hash);
        if !dup && code != self.null_char {
            let n = self.code_range as u64;
            let carry = ((hash >> 32).wrapping_mul(n)) >> 32;
            hash = hash.wrapping_mul(n);
            hash = hash.wrapping_add(carry);
            hash = hash.wrapping_add(code as u64);
        }
        hash
    }

    /// The best-scoring final node and the path to it, oldest first.
    fn extract_best_path(&self, last: &BeamStep, path: &mut Vec<usize>) -> Result<(), Error> {
        let mut best: Option<usize> = None;
        for c in 0..NC_COUNT {
            if c == NC_ONLY_DUP {
                continue;
            }
            for is_dawg in [false, true] {
                for &n in last.beams[beam_index(is_dawg, c)].as_slice() {
                    if is_dawg {
                        let mut d = Some(n);
                        while let Some(i) = d {
                            if self.nodes[i].unichar_id < 0 || self.nodes[i].duplicate {
                                d = self.nodes[i].prev;
                            } else {
                                break;
                            }
                        }
                        match d {
                            None => continue,
                            Some(i) => {
                                if !self.nodes[i].end_of_word
                                    && self.nodes[i].unichar_id != UNICHAR_SPACE as isize
                                {
                                    continue;
                                }
                            }
                        }
                    }
                    if best.is_none_or(|b| self.nodes[n].score > self.nodes[b].score) {
                        best = Some(n);
                    }
                }
            }
        }
        // Counted first so the path is made once, at its final size.
        let len = std::iter::successors(best, |&i| self.nodes[i].prev).count();
        grow(self.budget, path, len)?;
        let mut n = best;
        while let Some(i) = n {
            path.push(i);
            n = self.nodes[i].prev;
        }
        path.reverse();
        Ok(())
    }
}

/// `RecodeBeamSearch::PushHeap`'s core: insert `n` into a heap of at most `BEAM_WIDTH`, replacing
/// a node of the same identity if `n` scores better, else dropping the worst. Whether `n` was
/// kept.
///
/// The arena is charged for its growth before it grows.
fn push_node(
    budget: &Budget,
    nodes: &mut Vec<Node>,
    heap: &mut Heap,
    n: Node,
) -> Result<bool, Error> {
    if heap.len() >= BEAM_WIDTH && n.score <= worst_score(nodes, heap) {
        return Ok(false);
    }
    for &m in heap.as_slice() {
        let mn = &nodes[m];
        if mn.code == n.code
            && mn.code_hash == n.code_hash
            && mn.permuter == n.permuter
            && mn.start_of_dawg == n.start_of_dawg
        {
            if n.score > mn.score {
                nodes[m] = n;
                return Ok(true);
            }
            return Ok(false);
        }
    }
    let id = nodes.len();
    budget.push(nodes, n)?;
    heap.push(id);
    if heap.len() > BEAM_WIDTH {
        let mut worst = 0;
        for i in 1..heap.len() {
            if nodes[heap.items[i]].score < nodes[heap.items[worst]].score {
                worst = i;
            }
        }
        heap.swap_remove(worst);
    }
    Ok(true)
}

fn worst_score(nodes: &[Node], heap: &Heap) -> f64 {
    let items = heap.as_slice();
    let mut w = nodes[items[0]].score;
    for &n in &items[1..] {
        if nodes[n].score < w {
            w = nodes[n].score;
        }
    }
    w
}

impl Recognizer {
    /// Decodes the network output with the recoded beam search (width 5), with the dictionary if
    /// it is enabled, and returns the symbols with their certainty and rating.
    ///
    /// The nodes, steps and every symbol and its text are charged to `limits.max_alloc` as they
    /// are made, and the cancel flag is checked at every timestep.
    ///
    /// # Errors
    ///
    /// `Error::AllocLimit`, `Error::Cancelled`, else `Error::Recog` for an output that does not
    /// fit the network, `Error::OutOfMemory` if the allocator refuses.
    pub fn beam_decode(
        &self,
        out: &Tensor,
        limits: &Limits,
        cancel: &AtomicBool,
    ) -> Result<Vec<Scored>, Error> {
        self.beam_decode_in(out, &Budget::new(limits), cancel)
    }

    /// `beam_decode` on the call's own `budget`; the result stays charged.
    pub(crate) fn beam_decode_in(
        &self,
        out: &Tensor,
        budget: &Budget,
        cancel: &AtomicBool,
    ) -> Result<Vec<Scored>, Error> {
        let (mut bufs, mut scored) = (BeamBufs::default(), ScoredBuf::default());
        self.beam_decode_into(out, budget, cancel, &mut bufs, &mut scored)?;
        scored.to_vec(budget)
    }

    /// The search itself, in the vecs of `bufs` and with the symbols in `scored` (both the
    /// caller's, which keeps their capacity from one call to the next). Everything the search
    /// grows is charged to `budget` before it grows: the steps, the node arena and the
    /// dictionary positions.
    pub(super) fn beam_decode_into(
        &self,
        out: &Tensor,
        budget: &Budget,
        cancel: &AtomicBool,
        bufs: &mut BeamBufs,
        scored: &mut ScoredBuf,
    ) -> Result<(), Error> {
        if out.features != self.net.num_outputs || !out.is_consistent() {
            return Err(err(format!(
                "output has {} features, want {}",
                out.features, self.net.num_outputs
            )));
        }
        scored.clear();
        let width = out.map.len();
        if width == 0 {
            return Ok(());
        }
        let mut b = Beam::new(self, out.features, budget, bufs)?;
        let mut steps = std::mem::take(&mut bufs.steps);
        grow(budget, &mut steps, width)?;
        for t in 0..width {
            check_cancel(cancel)?;
            let row = out.row(t);
            b.compute_top_n(row);
            let step = b.decode_step(row, t, &steps)?;
            steps.push(step);
        }
        let mut path = std::mem::take(&mut bufs.path);
        match steps.last() {
            Some(last) => b.extract_best_path(last, &mut path)?,
            None => path.clear(),
        }
        self.path_as_scored(&b.nodes, &path, budget, scored)?;
        bufs.steps = steps;
        bufs.path = path;
        b.finish(bufs);
        Ok(())
    }

    fn path_as_scored(
        &self,
        nodes: &[Node],
        path: &[usize],
        budget: &Budget,
        out: &mut ScoredBuf,
    ) -> Result<(), Error> {
        let width = path.len();
        let mut t = 0;
        while t < width {
            let mut certainty = 0.0f64;
            let mut rating = 0.0f64;
            while t < width && nodes[path[t]].unichar_id < 0 {
                let c = nodes[path[t]].certainty;
                t += 1;
                if c < certainty {
                    certainty = c;
                }
                rating -= c;
            }
            if t >= width {
                if let Some(last) = out.last_mut() {
                    if certainty < last.certainty {
                        last.certainty = certainty;
                    }
                    last.rating += rating;
                }
                continue;
            }
            let id = nodes[path[t]].unichar_id as usize;
            if id == UNICHAR_SPACE && !out.is_empty() && nodes[path[t]].permuter != NO_PERM {
                if let Some(last) = out.last_mut() {
                    if certainty < last.certainty {
                        last.certainty = certainty;
                    }
                    last.rating += rating;
                }
                certainty = 0.0;
                rating = 0.0;
            }
            let start = t;
            loop {
                let c = nodes[path[t]].certainty;
                t += 1;
                if c < certainty || (id == UNICHAR_SPACE && nodes[path[t - 1]].permuter == NO_PERM)
                {
                    certainty = c;
                }
                rating -= c;
                if t >= width || !nodes[path[t]].duplicate {
                    break;
                }
            }
            out.push(
                budget,
                id,
                self.charset.text(id),
                (start, t),
                (certainty, rating),
            )?;
        }
        Ok(())
    }
}
