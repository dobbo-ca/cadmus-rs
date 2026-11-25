#![no_main]
use cadmus::nn::{Ctx, StrideMap, Tensor};
use cadmus::recog::Network;
use cadmus::tessdata::parse_recognizer;
use cadmus::{Budget, Limits};
use libfuzzer_sys::fuzz_target;
use std::sync::atomic::AtomicBool;

mod common;

// An lstm component written from the input (common::craft): a layer tree of every type, its
// dimensions (sometimes wrong in the header) and its weights, which include infinities, huge
// values and NaN. It is parsed, built into a network and run on a tensor whose activations are
// the rest of the input (NaN, infinities and out-of-range values among them), so the layer
// arithmetic and the budget accounting are reached with shapes the real model never has.
fuzz_target!(|data: &[u8]| {
    let limits = Limits {
        max_alloc: 16 << 20,
        max_threads: 2,
        ..Limits::default()
    };
    let cancel = AtomicBool::new(false);
    let mut c = common::Cur(data);
    let comp = common::craft(&mut c);
    let budget = Budget::new(&limits);
    let Some(rec) = common::no_oom(parse_recognizer(&comp.bytes, false, &budget, &cancel)) else {
        return;
    };
    let Some(net) = common::no_oom(Network::build(&rec, &limits, &cancel)) else {
        return;
    };
    let map = StrideMap {
        height: comp.height,
        width: 1 + usize::from(c.u8() % 12),
    };
    // One time in 16 the tensor has more features than the Input layer declares.
    let features = if c.u8() % 16 == 0 { 2 } else { 1 };
    let Some(mut t) = common::no_oom(Tensor::new(map, features, &budget)) else {
        return;
    };
    for ts in 0..t.map.len() {
        for v in t.row_mut(ts) {
            *v = common::act(c.u8());
        }
    }
    let mut cx = Ctx::new(&budget);
    if let Some(out) = common::no_oom(net.root.forward(&t, &mut cx)) {
        assert!(out.is_consistent());
    }
});
