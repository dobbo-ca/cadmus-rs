use super::*;
use crate::tessdata::{Dawg, parse_unicharset};
use crate::{Budget, Limits};

fn bud() -> Budget {
    Budget::new(&Limits::default())
}

// The unlimited forms the tests compare against.
impl Dict<'_> {
    fn start_raw(&self) -> Vec<Position> {
        self.start_in(&bud()).unwrap()
    }

    fn letter_is_okay_raw(
        &self,
        prev: PermuterType,
        active: &[Position],
        unichar_id: usize,
        word_end: bool,
    ) -> (PermuterType, Vec<Position>, bool) {
        self.letter_is_okay_in(prev, active, unichar_id, word_end, &bud())
            .unwrap()
    }
}

// A unicharset of 5: 0 space, 1 Joined, 2 Broken, 3 "1" (a digit), 4 "a".
fn charset() -> Unicharset {
    parse_unicharset(
        b"5\nNULL 0 Common 0\nJoined 7 0,255,0,255,0,0,0,0,0,0 Latin 1 0 1 Joined\n|Broken|0|1 f 0,255,0,255,0,0,0,0,0,0 Common 2 0 2 |Broken|0|1\n1 8 0,255,0,255,0,0,0,0,0,0 Common 3 0 3 1\na 3 0,255,0,255,0,0,0,0,0,0 Latin 4 0 4 a\n",
        &bud(),
    )
    .unwrap()
}

// Edge records for a unicharset of 5: the flag bits start at bit 3 (its bit length).
fn edge(id: u64, last: bool, eow: bool, next: u64) -> u64 {
    id | u64::from(last) << 3 | u64::from(eow) << 5 | next << 6
}

fn dawg(edges: &[u64]) -> Dawg {
    let mut b = 42i16.to_le_bytes().to_vec();
    b.extend_from_slice(&5u32.to_le_bytes());
    b.extend_from_slice(&(edges.len() as u32).to_le_bytes());
    for e in edges {
        b.extend_from_slice(&e.to_le_bytes());
    }
    Dawg::parse(&b, false, &bud()).unwrap()
}

#[test]
fn start_seeds_word_and_number_without_punctuation() {
    let cs = charset();
    let word = dawg(&[edge(4, true, true, 0)]);
    let number = dawg(&[edge(0, true, true, 0)]);
    let d = Dict::new(&cs, None, Some(&word), Some(&number));
    let start = d.start_raw();
    assert_eq!(start.len(), 2);
    assert_eq!((start[0].dawg_index, start[0].punc_index), (1, -1));
    assert_eq!((start[1].dawg_index, start[1].punc_index), (2, -1));
    assert!(
        start
            .iter()
            .all(|p| p.dawg_ref == NO_EDGE && p.punc_ref == NO_EDGE)
    );
    assert!(Dict::new(&cs, None, None, None).start_raw().is_empty());
}

#[test]
fn start_with_punctuation_seeds_only_it() {
    let cs = charset();
    // The punctuation root has a pattern edge, so it reaches the word dawg itself.
    let punc = dawg(&[edge(0, true, true, 0)]);
    let word = dawg(&[edge(4, true, true, 0)]);
    let d = Dict::new(&cs, Some(&punc), Some(&word), None);
    let start = d.start_raw();
    assert_eq!(start.len(), 1);
    assert_eq!((start[0].dawg_index, start[0].punc_index), (-1, 0));
}

#[test]
fn word_dawg_walk() {
    let cs = charset();
    // Words: "a" and "a1" (the digit stands in for a second letter).
    let word = dawg(&[edge(4, true, true, 1), edge(3, true, true, 0)]);
    let d = Dict::new(&cs, None, Some(&word), None);

    let (perm, pos, end) = d.letter_is_okay_raw(NO_PERM, &d.start_raw(), 4, false);
    assert_eq!(perm, SYSTEM_DAWG_PERM);
    assert!(end);
    assert_eq!(pos.len(), 1);

    let (perm, pos, end) = d.letter_is_okay_raw(perm, &pos, 3, false);
    assert_eq!(perm, SYSTEM_DAWG_PERM);
    assert!(end);

    // The word ends at the second edge (next_node 0): nothing continues it.
    let (perm, pos, end) = d.letter_is_okay_raw(perm, &pos, 3, false);
    assert_eq!((perm, pos.len(), end), (NO_PERM, 0, false));

    // 'a' must be a word end to match a complete word: the first edge is flagged, so it is.
    let (perm, _, end) = d.letter_is_okay_raw(NO_PERM, &d.start_raw(), 4, true);
    assert_eq!((perm, end), (SYSTEM_DAWG_PERM, true));
    // The digit is not a word start.
    assert_eq!(
        d.letter_is_okay_raw(NO_PERM, &d.start_raw(), 3, false).0,
        NO_PERM
    );
}

#[test]
fn pattern_and_out_of_range_ids_are_rejected() {
    let cs = charset();
    let word = dawg(&[edge(4, true, true, 0)]);
    let d = Dict::new(&cs, None, Some(&word), None);
    for id in [PATTERN_UNICHAR_ID, 5, 99] {
        assert_eq!(
            d.letter_is_okay_raw(NO_PERM, &d.start_raw(), id, false),
            (NO_PERM, vec![], false)
        );
    }
}

#[test]
fn number_dawg_folds_digits_to_the_wildcard() {
    let cs = charset();
    let number = dawg(&[edge(0, true, true, 0)]);
    let d = Dict::new(&cs, None, None, Some(&number));
    let (perm, pos, end) = d.letter_is_okay_raw(NO_PERM, &d.start_raw(), 3, false);
    assert_eq!((perm, end), (NUMBER_PERM, true));
    assert_eq!(pos.len(), 1);
    // A letter does not fold, and the number dawg has only the wildcard edge.
    assert_eq!(
        d.letter_is_okay_raw(NO_PERM, &d.start_raw(), 4, false).0,
        NO_PERM
    );
}

#[test]
fn punctuation_wraps_a_word() {
    let cs = charset();
    // Punctuation: node 0 is the run of edges 0 and 1: a wildcard edge leading to node 2 and the
    // digit as punctuation. Node 2 is edge 2: the digit as trailing punctuation, ending the word.
    // Word dawg: "a".
    let punc = dawg(&[
        edge(0, false, true, 2), // node 0: wildcard, continues at node 2
        edge(3, true, true, 0),  // node 0: the digit as punctuation, ends a word
        edge(3, true, true, 0),  // node 2: the digit as trailing punctuation
    ]);
    let word = dawg(&[edge(4, true, true, 0)]);
    let d = Dict::new(&cs, Some(&punc), Some(&word), None);
    let start = d.start_raw();
    assert_eq!(start.len(), 1);
    // 'a' through the wildcard into the word dawg.
    let (perm, pos, end) = d.letter_is_okay_raw(NO_PERM, &start, 4, false);
    assert_eq!(perm, SYSTEM_DAWG_PERM);
    assert!(
        end,
        "a bare word is a valid end: the wildcard edge ends a word"
    );
    assert!(!pos.is_empty());
    // Then the trailing punctuation: the carry keeps the word's permuter.
    let (perm2, _, end2) = d.letter_is_okay_raw(perm, &pos, 3, true);
    assert_eq!(perm2, SYSTEM_DAWG_PERM);
    assert!(end2);
}

#[test]
fn letter_is_okay_survives_positions_that_point_outside_the_dawg() {
    let cs = charset();
    let word = dawg(&[edge(4, true, true, 0)]);
    let d = Dict::new(&cs, None, Some(&word), None);
    for dawg_ref in [1i64 << 40, -7, i64::MAX, i64::MIN] {
        let pos = Position {
            dawg_index: 1,
            dawg_ref,
            punc_index: -1,
            punc_ref: -1,
            back_to_punc: false,
        };
        for word_end in [false, true] {
            let _ = d.letter_is_okay_raw(NO_PERM, &[pos], 4, word_end);
        }
    }
}

// The public entry points take the limits; they agree with the unchecked ones and refuse what does
// not fit.
#[test]
fn public_entry_points_check_the_limits() {
    let cs = charset();
    let word = dawg(&[edge(4, true, true, 1), edge(3, true, true, 0)]);
    let d = Dict::new(&cs, None, Some(&word), None);
    let lim = Limits::default();
    let start = d.start(&lim).unwrap();
    assert_eq!(start, d.start_raw());
    assert_eq!(
        d.letter_is_okay(NO_PERM, &start, 4, false, &lim).unwrap(),
        d.letter_is_okay_raw(NO_PERM, &start, 4, false)
    );
    let none = Limits {
        max_alloc: 0,
        ..Limits::default()
    };
    assert_eq!(d.start(&none), Err(crate::Error::AllocLimit));
    assert_eq!(
        d.letter_is_okay(NO_PERM, &start, 4, false, &none),
        Err(crate::Error::AllocLimit)
    );
}
