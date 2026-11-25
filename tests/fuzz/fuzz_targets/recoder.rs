#![no_main]
use cadmus::tessdata::Recoder;
use cadmus::{Budget, Limits};
use libfuzzer_sys::fuzz_target;

mod common;

// A first byte of 0 is how a foreign-endian component starts (its count is big-endian), so the
// swapped path is reached from the real components used as seeds.
fuzz_target!(|data: &[u8]| {
    let limits = Limits {
        max_alloc: 8 << 20,
        ..Limits::default()
    };
    let swap = data.first() == Some(&0);
    if let Some(rc) = common::no_oom(Recoder::parse(data, swap, &Budget::new(&limits))) {
        for i in 0..rc.size() {
            let code = rc.encode(i).expect("an id inside the recoder");
            assert!(code.len() <= rc.max_code_len());
            let _ = rc.decode_unichar(code);
        }
        assert!(rc.encode(rc.size()).is_none());
        for c in -1..=(rc.code_range() as i32) {
            let _ = rc.is_valid_first_code(c);
        }
    }
});
