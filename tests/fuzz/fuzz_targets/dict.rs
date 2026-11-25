#![no_main]
use cadmus::dict::{Dict, NO_PERM, PermuterType, Position};
use cadmus::tessdata::{Dawg, NO_EDGE, parse_unicharset};
use cadmus::{Budget, Limits};
use libfuzzer_sys::fuzz_target;

mod common;

fn take<'a>(rest: &mut &'a [u8], n: usize) -> &'a [u8] {
    let (a, b) = rest.split_at(n.min(rest.len()));
    *rest = b;
    a
}

fn dawg(part: &[u8], budget: &Budget) -> Option<Dawg> {
    if part.is_empty() {
        return None;
    }
    common::no_oom(Dawg::parse(part, part.first() == Some(&0), budget))
}

// data = [unicharset, punctuation, word, number: u16 LE lengths] then those four byte strings (an
// empty or unparsable dawg is left out) then 6-byte steps [kind][id][a][b][c][d]. The lexicon is
// built from the hostile parts the dawg and unicharset targets make, and walked: each step tries
// unichar `id` (bit 0 of `kind`: at the end of a word) from the positions the previous step
// produced, or, with bit 1, from one position made of a, b, c and d (indexes and edge references
// that need not exist; bit 2 sets `back_to_punc`). Bit 3 replaces the carried permuter.
fuzz_target!(|data: &[u8]| {
    let Some((hdr, mut rest)) = data.split_first_chunk::<8>() else {
        return;
    };
    let len = |i: usize| usize::from(u16::from_le_bytes([hdr[2 * i], hdr[2 * i + 1]]));
    let limits = Limits {
        max_alloc: 16 << 20,
        ..Limits::default()
    };
    let budget = Budget::new(&limits);
    let Some(charset) = common::no_oom(parse_unicharset(take(&mut rest, len(0)), &budget)) else {
        return;
    };
    let punc = dawg(take(&mut rest, len(1)), &budget);
    let word = dawg(take(&mut rest, len(2)), &budget);
    let number = dawg(take(&mut rest, len(3)), &budget);
    let d = Dict::new(&charset, punc.as_ref(), word.as_ref(), number.as_ref());
    let Some(start) = common::no_oom(d.start(&limits)) else {
        return;
    };
    let edge = |v: u8| match v {
        0 => NO_EDGE,
        1 => i64::MAX,
        2 => i64::MIN,
        3 => -1,
        v => i64::from(v) - 4,
    };
    let (mut pos, mut prev) = (start.clone(), NO_PERM);
    for rec in rest.chunks_exact(6) {
        let &[kind, id, a, b, c, e] = rec else {
            continue;
        };
        if kind & 2 != 0 {
            pos = vec![Position {
                dawg_index: i32::from(a as i8),
                dawg_ref: edge(c),
                punc_index: i32::from(b as i8),
                punc_ref: edge(e),
                back_to_punc: kind & 4 != 0,
            }];
        }
        if kind & 8 != 0 {
            prev = PermuterType(i32::from(a as i8));
        }
        let id = usize::from(id) % (charset.size() + 2);
        let Some((p, next, _word_may_end)) =
            common::no_oom(d.letter_is_okay(prev, &pos, id, kind & 1 != 0, &limits))
        else {
            return;
        };
        if next.is_empty() {
            (pos, prev) = (start.clone(), NO_PERM);
        } else {
            (pos, prev) = (next, p);
        }
    }
});
