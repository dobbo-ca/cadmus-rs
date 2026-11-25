#![no_main]
use cadmus::Limits;
use cadmus::recog::Recognizer;
use cadmus::tessdata::Model;
use libfuzzer_sys::fuzz_target;
use std::sync::atomic::AtomicBool;

mod common;

// data = [keep: u8][edit: offset u32 LE, value u8, mode u8]... The mini model (the real model's
// charset, recoder and dawgs around a small network), cut to keep/256 of its length when keep is odd, then each edit writes `value` at its offset (inside the first 512
// bytes when mode is even, anywhere otherwise). The edited model is loaded as a Model and as a
// Recognizer; one that loads reads a line, so weights that are NaN or huge go through the network.
fuzz_target!(|data: &[u8]| {
    let Some((&keep, edits)) = data.split_first() else {
        return;
    };
    let mut m = common::mini_model().to_vec();
    if keep & 1 != 0 {
        m.truncate(m.len() / 256 * usize::from(keep));
    }
    for e in edits.chunks_exact(6) {
        let off = u32::from_le_bytes([e[0], e[1], e[2], e[3]]) as usize;
        let off = if e[5] & 1 == 0 {
            off % 512
        } else {
            off % m.len().max(1)
        };
        if let Some(b) = m.get_mut(off) {
            *b = e[4];
        }
    }
    // max_alloc under the fuzz malloc limit (512 MiB), so that no charged allocation can trip it:
    // with the default 1 GiB, an edited input height of 3457 charged and made a 663,080,256-byte
    // softmax output (seeds/model_mutate/oom-input-height-3457).
    //
    // max_pixels 1 << 16 bounds the time. The beam search runs serially, once per output
    // timestep (about 100 us each in CI), and the 40x20 line is scaled to the edited input height
    // h, so a line costs about h * h / 8 steps after a scale of 2 * h * h pixels. An edited
    // height of 1288 is 207,000 steps (19 to 21 s against the 20 s timeout, on the thread scopes
    // and the shared pool alike: seeds/model_mutate/slow-input-height-1288), and one of 6402 is
    // an 82M-pixel scale (14 s to over 20 s: seeds/model_mutate/slow-input-height-6402). The cap
    // makes the scaler refuse past h = 181 (about 4,000 steps, under half a second); it also ends
    // the 3457 input above in `PixelLimit` before the allocation.
    let limits = Limits {
        max_alloc: 256 << 20,
        max_pixels: 1 << 16,
        ..Limits::default()
    };
    let cancel = AtomicBool::new(false);
    common::no_oom(Model::load(&m, &limits, &cancel));
    if let Some(r) = common::no_oom(Recognizer::new(&m, &limits, &cancel))
        && let Some(g) = common::no_oom(common::gray(40, 20, &m[..m.len().min(64)], &limits))
    {
        let _ = common::no_oom(r.recognize(&g, &limits, &cancel));
    }
});
