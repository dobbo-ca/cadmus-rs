#![no_main]
use cadmus::Limits;
use cadmus::imaging::Bitmap;
use libfuzzer_sys::fuzz_target;

mod common;

fuzz_target!(|data: &[u8]| {
    let limits = Limits {
        max_pixels: 1 << 22,
        max_alloc: 64 << 20,
        ..Limits::default()
    };
    // data = [w][h][rgba...] for from_rgba8, with the pixels repeated to fill w * h.
    if let [w, h, rest @ ..] = data {
        let (w, h) = (usize::from(*w), usize::from(*h));
        let rgba: Vec<u8> = rest.iter().copied().cycle().take(w * h * 4).collect();
        common::no_oom(Bitmap::from_rgba8(w, h, &rgba, &limits));
    }
    if let Some(b) = common::no_oom(Bitmap::from_pgm(data, &limits)) {
        assert_eq!((b.depth, b.stride), (8, b.width));
        assert_eq!(b.pix.len(), b.width * b.height);
        assert!(data.len() >= b.pix.len());
    }
});
