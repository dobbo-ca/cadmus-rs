#![no_main]
use cadmus::recog::Recognizer;
use cadmus::tessdata::{self, Model};
use cadmus::{Error, Limits};
use libfuzzer_sys::fuzz_target;
use std::sync::atomic::AtomicBool;

mod common;

// The container, the model loader and the recognizer loader on arbitrary bytes. The limits are
// small, so a size field that is not checked against them is an allocation the fuzzer's malloc
// limit catches.
fuzz_target!(|data: &[u8]| {
    let limits = Limits {
        max_alloc: 64 << 20,
        max_model_bytes: 1 << 20,
        ..Limits::default()
    };
    let cancel = AtomicBool::new(false);
    if let Ok(c) = tessdata::parse_container(data, &limits) {
        for t in c.present() {
            let e = c.entry(t).expect("a present entry");
            assert!(e.len() <= data.len());
        }
        let _ = c.version(&limits);
    }
    common::no_oom(Model::load(data, &limits, &cancel));
    common::no_oom(Recognizer::new(data, &limits, &cancel));
    let tiny = Limits {
        max_model_bytes: 16,
        ..limits
    };
    if data.len() > 16 {
        assert_eq!(Model::load(data, &tiny, &cancel).err(), Some(Error::ModelLimit));
    }
});
