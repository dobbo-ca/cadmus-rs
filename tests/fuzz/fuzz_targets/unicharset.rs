#![no_main]
use cadmus::tessdata;
use cadmus::{Budget, Limits};
use libfuzzer_sys::fuzz_target;

mod common;

fuzz_target!(|data: &[u8]| {
    let limits = Limits {
        max_alloc: 8 << 20,
        ..Limits::default()
    };
    if let Some(u) = common::no_oom(tessdata::parse_unicharset(data, &Budget::new(&limits))) {
        for i in 0..u.size() {
            assert!(u.char(i).is_some());
            let _ = (u.text(i), u.normed(i), u.is_space_delimited(i));
        }
        assert!(u.char(u.size()).is_none());
    }
    // `quote` on any UTF-8 text: it writes at most the 4 * len + 2 bytes it reserved and charged.
    if let Ok(s) = std::str::from_utf8(data)
        && let Some(q) = common::within(limits.max_alloc, || {
            common::no_oom(tessdata::quote(s, &limits))
        })
    {
        assert!(q.len() <= 4 * s.len() + 2);
        assert!(q.starts_with('"') && q.ends_with('"') && q.len() >= 2);
    }
});
