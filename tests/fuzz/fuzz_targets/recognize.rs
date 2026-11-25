#![no_main]
use cadmus::Limits;
use cadmus::dict::NO_PERM;
use cadmus::imaging::Rect;
use cadmus::nn::{Layer, Rand, Tensor};
use cadmus::recog::{Line, Recognizer, Workspace};
use libfuzzer_sys::fuzz_target;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

mod common;

// data = [mode][w][h][pixels]. The recognizer is the one shared by every input (so its random
// generator and lexicon switch carry over): the mini model's, which runs a line in microseconds,
// or, when bits 3-7 of the mode are all set (1 input in 32, so that the second a real line takes
// does not dominate the run), the real model's. mode % 4: 0 forward (with a tap on every layer),
// greedy decode and `recognize_text`, 1 recognize with the lexicon, 2 recognize without it, 3 walks
// the lexicon with the pixel bytes as unichar ids. Bit 2 of the mode sets the cancel flag. Modes 1
// and 2 also run `recognize_into` twice on one `Workspace` and `Line`, and the three results must be
// equal (on the mini model, whose generator is reseeded before each call: the real model's
// convolutions draw from copies of it that cannot be reseeded).

/// Whether two lines are the same, floats to the bit.
fn same_line(a: &Line, b: &Line) -> bool {
    a.text == b.text
        && a.bounds == b.bounds
        && a.confidence.to_bits() == b.confidence.to_bits()
        && a.words.len() == b.words.len()
        && a.words.iter().zip(&b.words).all(|(x, y)| {
            x.text == y.text
                && x.bounds == y.bounds
                && x.confidence.to_bits() == y.confidence.to_bits()
        })
}

fn reseed(r: &mut Recognizer, on: bool) {
    if on {
        r.net.rand = Arc::new(Rand::new(7));
    }
}

fuzz_target!(|data: &[u8]| {
    let Some((&[mode, w, h], pix)) = data.split_first_chunk::<3>() else {
        return;
    };
    let limits = Limits {
        max_alloc: 256 << 20,
        max_pixels: 1 << 20,
        ..Limits::default()
    };
    let cancel = AtomicBool::new(mode & 4 != 0);
    let mut r = if mode >> 3 == 0x1f {
        common::recognizer()
    } else {
        common::mini_recognizer()
    };
    if mode % 4 == 3 {
        r.set_lexicon(true);
        let Some(d) = r.dict() else { return };
        let Some(start) = common::no_oom(d.start(&limits)) else {
            return;
        };
        let (mut pos, mut prev) = (start.clone(), NO_PERM);
        for pair in pix.chunks_exact(2) {
            let id = usize::from(pair[0]) % (r.charset.size() + 2);
            let Some((p, next, _ok)) =
                common::no_oom(d.letter_is_okay(prev, &pos, id, pair[1] & 1 != 0, &limits))
            else {
                return;
            };
            if next.is_empty() {
                (pos, prev) = (start.clone(), NO_PERM);
            } else {
                (pos, prev) = (next, p);
            }
        }
        return;
    }
    let (w, h) = (1 + usize::from(w) % 64, 2 + usize::from(h) % 46);
    let Some(g) = common::no_oom(common::gray(w, h, pix, &limits)) else {
        return;
    };
    r.set_lexicon(mode % 4 != 2);
    if mode % 4 == 0 {
        let mut tap = |_: &Layer, t: &Tensor| assert!(t.is_consistent());
        if let Some((out, _)) = common::no_oom(r.forward_tap(&g, &limits, &cancel, Some(&mut tap)))
        {
            common::no_oom(r.greedy_decode(&out, &limits, &cancel));
        }
        common::no_oom(r.recognize_text(&g, &limits, &cancel));
    } else {
        let same_rand = mode >> 3 != 0x1f;
        let mut ws = Workspace::default();
        let mut line = Line {
            text: String::new(),
            words: Vec::new(),
            bounds: Rect::default(),
            confidence: 0.0,
        };
        reseed(&mut r, same_rand);
        let want = r.recognize(&g, &limits, &cancel);
        for _ in 0..2 {
            reseed(&mut r, same_rand);
            let got = r.recognize_into(&mut ws, &g, &mut line, &limits, &cancel);
            if same_rand {
                match (&want, &got) {
                    (Ok(w), Ok(())) => assert!(same_line(w, &line), "{w:?} vs {line:?}"),
                    (Err(a), Err(b)) => assert_eq!(a, b),
                    _ => panic!("recognize gave {want:?}, recognize_into {got:?}"),
                }
            }
            common::no_oom(got);
        }
        if same_rand {
            // A fuzzed max_alloc on the warm workspace: a limit a fresh call passes is passed by
            // the warm one (it may empty the workspace and rerun), with the same result.
            let tight = Limits {
                max_alloc: 1 << (10 + usize::from(mode >> 3) % 19),
                max_pixels: 1 << 20,
                ..Limits::default()
            };
            let fresh = r.recognize(&g, &tight, &cancel);
            let got = r.recognize_into(&mut ws, &g, &mut line, &tight, &cancel);
            if let Ok(w) = &fresh {
                assert_eq!(
                    got,
                    Ok(()),
                    "recognize passed max_alloc {}",
                    tight.max_alloc
                );
                assert!(same_line(w, &line), "{w:?} vs {line:?}");
            }
            common::no_oom(fresh);
            common::no_oom(got);
        }
        common::no_oom(want);
    }
});
