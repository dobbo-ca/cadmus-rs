#![no_main]
use cadmus::tessdata::parse_recognizer;
use cadmus::{Budget, Limits};
use libfuzzer_sys::fuzz_target;
use std::sync::atomic::AtomicBool;

mod common;

// The lstm component: the layer tree, the weight matrices and the trailer fields. The first byte
// is a selector that is not part of the component: its low bit picks the byte-swapped reader. (A
// real component starts with a 0 tag and then the type name, so the component's own first byte
// cannot be the selector: every real seed would be read swapped and rejected at its first length.)
fuzz_target!(|input: &[u8]| {
    let Some((&selector, data)) = input.split_first() else {
        return;
    };
    let limits = Limits {
        max_alloc: 64 << 20,
        ..Limits::default()
    };
    let swap = selector & 1 == 1;
    let cancel = AtomicBool::new(false);
    if let Some(r) = common::no_oom(parse_recognizer(data, swap, &Budget::new(&limits), &cancel)) {
        let _ = common::no_oom(r.network.tree(&limits));
    }
});
