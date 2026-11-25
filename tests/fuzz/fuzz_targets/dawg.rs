#![no_main]
use cadmus::tessdata::{Dawg, NO_EDGE};
use cadmus::{Budget, Limits};
use libfuzzer_sys::fuzz_target;

mod common;

// A first byte of 0 is how a foreign-endian dawg starts (its magic is big-endian). The last bytes
// of the input are also the unichar ids the dawg is probed with.
fuzz_target!(|data: &[u8]| {
    let limits = Limits {
        max_alloc: 16 << 20,
        ..Limits::default()
    };
    let swap = data.first() == Some(&0);
    let Some(d) = common::no_oom(Dawg::parse(data, swap, &Budget::new(&limits))) else {
        return;
    };
    let n = d.num_edges();
    for e in 0..n.min(10_000) {
        let e = e as i64;
        let _ = (d.unichar_id(e), d.next_node(e), d.end_of_word(e));
    }
    // Edge references that are not in the table must read as no edge, not panic.
    for e in [NO_EDGE, n as i64, i64::MAX, i64::MIN] {
        let _ = (d.unichar_id(e), d.next_node(e), d.end_of_word(e));
    }
    let ids: Vec<usize> = data.iter().rev().take(8).map(|&b| usize::from(b)).collect();
    for k in 0..=ids.len() {
        let w = &ids[..k];
        let has = d.has_prefix(w);
        assert!(!d.contains(w) || has, "a word is a prefix of itself");
    }
    for node in (0..n.min(64))
        .map(|i| i as i64)
        .chain([NO_EDGE, n as i64, i64::MAX])
    {
        for &id in ids.iter().take(4) {
            let _ = d.edge_char(node, id, false);
            let _ = d.edge_char(node, id, true);
        }
    }
});
