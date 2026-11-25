#![no_main]
use cadmus::nn::{Ctx, StrideMap, Tensor};
use cadmus::recog::{Scored, Symbol, char_boundaries, score_symbols};
use cadmus::{Budget, Limits};
use libfuzzer_sys::fuzz_target;
use std::sync::atomic::AtomicBool;

mod common;

// The decoders take a Tensor the caller built, and `map` and `features` are public, so they can
// disagree with the data. data = [h][w][features][declared features][n][n symbols: start, end
// pairs][activations]. The feature counts index `FEATURES`: 0, 1, the eng model's 111 outputs and
// one more. The activations fill the tensor row by row from the bytes that are left (common::act:
// probabilities, and NaN, infinities, negatives, out-of-range and huge values), zero once they
// run out.
const FEATURES: [usize; 4] = [0, 1, 111, 112];

fuzz_target!(|data: &[u8]| {
    let Some((&[h, w, f, declared, n], rest)) = data.split_first_chunk::<5>() else {
        return;
    };
    let (sym_bytes, acts) = rest.split_at((usize::from(n) % 17 * 2).min(rest.len()));
    let limits = Limits {
        max_alloc: 16 << 20,
        ..Limits::default()
    };
    let map = StrideMap {
        height: usize::from(h % 3),
        width: usize::from(w % 8),
    };
    let Some(mut t) = common::no_oom(Tensor::new(
        map,
        FEATURES[usize::from(f) % 4],
        &Budget::new(&limits),
    )) else {
        return;
    };
    let mut c = common::Cur(acts);
    for ts in 0..t.map.len() {
        for v in t.row_mut(ts) {
            *v = common::act(c.u8());
        }
    }
    t.features = FEATURES[usize::from(declared) % 4];
    let syms: Vec<Symbol> = sym_bytes
        .chunks_exact(2)
        .map(|p| Symbol {
            unichar_id: 0,
            text: String::new(),
            start: usize::from(p[0]),
            end: usize::from(p[1]),
        })
        .collect();
    let cancel = AtomicBool::new(false);
    let r = common::mini_recognizer();
    common::within(limits.max_alloc, || {
        common::no_oom(r.greedy_decode(&t, &limits, &cancel))
    });
    common::within(limits.max_alloc, || {
        common::no_oom(r.beam_decode(&t, &limits, &cancel))
    });
    let _ = score_symbols(&t, &syms, 2.25, &limits);
    // Layer::forward is public too, and the recognizer's network is reachable.
    let budget = Budget::new(&limits);
    common::no_oom(r.net.root.forward(&t, &mut Ctx::new(&budget)));
    // Symbols with indices out to the ends of usize.
    let far = |v: usize| v.wrapping_mul(usize::MAX / 255);
    let scored: Vec<Scored> = syms
        .iter()
        .map(|s| Scored {
            symbol: Symbol {
                start: far(s.start),
                end: far(s.end),
                ..s.clone()
            },
            certainty: 0.0,
            rating: 0.0,
        })
        .collect();
    let _ = char_boundaries(&scored, 8, &limits);
});
