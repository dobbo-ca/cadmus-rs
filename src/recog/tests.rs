use super::words::{split_words, word_bounds};
use super::*;
use crate::imaging::{Bitmap, Rect};
use crate::nn::{Input, Layer, Rand, StrideMap, Tensor};
use crate::tessdata::{Recoder, parse_unicharset};
use crate::{Budget, Limits};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

fn lim() -> Limits {
    Limits::default()
}

fn nc() -> AtomicBool {
    AtomicBool::new(false)
}

fn bud() -> Budget {
    Budget::new(&lim())
}

fn gray(w: usize, h: usize, v: u8) -> Bitmap {
    Bitmap::from_gray8(w, h, &vec![v; w * h], &lim()).unwrap()
}

fn norm(img: &Bitmap) -> Result<Normalized, crate::Error> {
    normalize(img, 36, 3, &Rand::new(1), &lim(), &nc())
}

#[test]
fn prob_to_certainty_floor() {
    assert_eq!(prob_to_certainty(1.0), 0.0);
    assert!((prob_to_certainty(0.5) - 0.5f64.ln()).abs() < 1e-15);
    assert_eq!(prob_to_certainty(0.5), crate::nn::go_log_for_test(0.5));
    assert_eq!(prob_to_certainty(0.0), MIN_CERTAINTY);
    assert_eq!(prob_to_certainty(MIN_CERTAINTY.exp() / 2.0), MIN_CERTAINTY);
}

fn probs(codes: &[usize], p: &[f64], n: usize) -> Tensor {
    let mut out = Tensor::new(
        StrideMap {
            height: 1,
            width: codes.len(),
        },
        n,
        &bud(),
    )
    .unwrap();
    let mut row = vec![0.0; n];
    for (t, &c) in codes.iter().enumerate() {
        let rest = (1.0 - p[t]) / (n - 1) as f64;
        row.fill(rest);
        row[c] = p[t];
        out.write_time_step(t, &row);
    }
    out
}

fn sym(id: usize, text: &str, start: usize, end: usize) -> Symbol {
    Symbol {
        unichar_id: id,
        text: text.to_string(),
        start,
        end,
    }
}

#[test]
fn score_symbols_charges_preceding_blanks() {
    let out = probs(&[9, 1, 1, 9], &[0.5, 0.9, 0.8, 0.99], 10);
    let scored = score_symbols(&out, &[sym(1, "A", 1, 3)], 1.0, &lim()).unwrap();
    assert_eq!(scored.len(), 1);
    let cert = |p: f64| prob_to_certainty(f64::from(p as f32)) + CERT_OFFSET;
    let want = cert(0.5).min(cert(0.9)).min(cert(0.8)).min(cert(0.99));
    assert!((scored[0].certainty - want).abs() < 1e-12);
    let rating = -(cert(0.5) + cert(0.9) + cert(0.8) + cert(0.99));
    assert!((scored[0].rating - rating).abs() < 1e-12);
}

#[test]
fn score_symbols_applies_dict_ratio() {
    let out = probs(&[1], &[0.9], 10);
    let s = [sym(1, "", 0, 1)];
    let plain = score_symbols(&out, &s, 1.0, &lim()).unwrap();
    let scaled = score_symbols(&out, &s, DICT_RATIO, &lim()).unwrap();
    assert!((scaled[0].certainty - plain[0].certainty * DICT_RATIO).abs() < 1e-12);
}

fn scored(start: usize, end: usize) -> Scored {
    Scored {
        symbol: sym(0, "", start, end),
        certainty: 0.0,
        rating: 0.0,
    }
}

#[test]
fn char_boundaries_cases() {
    let cb = |syms: &[Scored], width: usize| char_boundaries(syms, width, &lim()).unwrap();
    assert_eq!(cb(&[scored(1, 3), scored(6, 7)], 10), [0, 4, 10]);
    assert_eq!(cb(&[], 10), [10]);
    assert_eq!(cb(&[scored(4, 5)], 10), [0, 10]);
    // Indices at the ends of the range must not overflow.
    let top = isize::MAX as usize;
    assert_eq!(
        cb(&[scored(0, usize::MAX), scored(top, 0)], 1),
        [
            0,
            (usize::MAX as i128 + (top as i128 - usize::MAX as i128) / 2) as usize,
            1
        ]
    );
}

#[test]
fn confidence_percent_clamps() {
    assert!((confidence_percent(-0.095) - 96.675).abs() < 0.01);
    assert_eq!(confidence_percent(0.0), 100.0);
    assert_eq!(confidence_percent(-10.0), 0.0);
}

#[test]
fn split_words_at_spaces() {
    let mk = |id: usize, text: &str, c: f64| Scored {
        symbol: sym(id, text, 0, 0),
        certainty: c,
        rating: 0.0,
    };
    let syms = [
        mk(3, "h", -0.2),
        mk(4, "i", -0.1),
        mk(0, " ", -0.9),
        mk(5, "t", -0.3),
        mk(6, "o", -0.4),
    ];
    let groups = split_words(&syms);
    assert_eq!(groups.len(), 2);
    assert_eq!((groups[0].start, groups[0].end), (0, 2));
    assert_eq!((groups[1].start, groups[1].end), (3, 5));
    assert!((groups[0].space_cert - -0.9).abs() < 1e-12);
}

#[test]
fn word_bounds_scale_from_timesteps() {
    let got = word_bounds(&[0, 4, 10], 0, 1, 6.0, Rect::new(100, 50, 400, 90));
    assert_eq!(got, Rect::new(100, 50, 124, 90));
    // A zero-width word is one pixel wide.
    let got = word_bounds(&[4, 4], 0, 1, 6.0, Rect::new(0, 0, 100, 10));
    assert_eq!(got.dx(), 1);
}

#[test]
fn normalize_flat_image_uses_defaults() {
    let n = norm(&gray(40, 36, 128)).unwrap();
    assert_eq!(n.black, 1.0);
    assert!((f64::from(n.contrast) - 127.5).abs() < 1e-6);
    let (pixel, black, contrast) = (128f32, 1f32, 127.5f32);
    let want = f64::from((pixel - black) / contrast - 1.0);
    assert!(want < 0.0);
    let mut got = [0.0];
    n.input.read_time_step(0, &mut got);
    assert_eq!(got[0], f64::from(want as f32));
}

#[test]
fn normalize_polarity() {
    let mut img = gray(40, 36, 240);
    for y in 0..36 {
        for x in (0..40).step_by(4) {
            img.set(x, y, 10);
        }
    }
    let n = norm(&img).unwrap();
    let (mut dark, mut light) = ([0.0], [0.0]);
    n.input.read_time_step(n.input.map.t(0, 0), &mut dark);
    n.input.read_time_step(n.input.map.t(0, 1), &mut light);
    assert!(dark[0] < light[0], "polarity is inverted");
}

#[test]
fn normalize_scale_and_shape() {
    let n = norm(&gray(97, 36, 0)).unwrap();
    assert_eq!(
        n.input.map,
        StrideMap {
            height: 36,
            width: 97
        }
    );
    assert!((n.scale_factor - 3.0).abs() < 1e-9);
    let n = norm(&gray(200, 72, 0)).unwrap();
    assert!((n.scale_factor - 6.0).abs() < 1e-6);
    // An off-by-one scaled height still gives the network's input height.
    let n = norm(&gray(120, 37, 0)).unwrap();
    assert_eq!(n.input.map.height, 36);
}

#[test]
fn normalize_rejects_bad_images() {
    assert!(norm(&gray(2, 36, 0)).is_err());
    assert!(norm(&gray(5, 72, 0)).is_err());
    let bilevel = Bitmap::new(40, 36, 1, &lim()).unwrap();
    assert!(norm(&bilevel).is_err());
    assert!(norm(&gray(0, 36, 0)).is_err());
    let set = AtomicBool::new(true);
    assert_eq!(
        normalize(&gray(40, 36, 0), 36, 3, &Rand::new(1), &lim(), &set).unwrap_err(),
        crate::Error::Cancelled
    );
}

// What `tests/fuzz/fuzz_targets/model_mutate.rs` relies on: with `max_pixels` 1 << 16, a 40x20 line is
// refused when the input height is edited past 181, before the tensor, the network or the beam.
#[test]
fn normalize_pixel_cap_bounds_input_height() {
    let limits = Limits {
        max_pixels: 1 << 16,
        ..lim()
    };
    let img = gray(40, 20, 0);
    let run = |h| normalize(&img, h, 2, &Rand::new(1), &limits, &nc());
    assert!(run(181).is_ok());
    for h in [182, 1288, 6402] {
        assert_eq!(run(h).unwrap_err(), crate::Error::PixelLimit);
    }
}

// The charset: 0 space, 1 Joined, 2 Broken, 3 'A', 4 'B'. The codes: space 0, Joined and Broken
// both 4 (the blank, as in eng, where they collide on 110), A 1, B 2. Code range 5, blank 4.
fn synthetic() -> Recognizer {
    synthetic_with_a("A")
}

// `synthetic`, with the text of unichar 3 (the one code 1 decodes to) set to `a`.
fn synthetic_with_a(a: &str) -> Recognizer {
    let budget = bud();
    let charset = parse_unicharset(
        format!("5\nNULL 0 Common 0\nJoined 7 0,255,0,255,0,0,0,0,0,0 Latin 1 0 1 Joined\n|Broken|0|1 f 0,255,0,255,0,0,0,0,0,0 Common 2 0 2 |Broken|0|1\n{a} 5 0,255,0,255,0,0,0,0,0,0 Latin 3 0 3 {a}\nB 5 0,255,0,255,0,0,0,0,0,0 Latin 4 0 4 B\n").as_bytes(),
        &budget,
    )
    .unwrap();
    let codes: [&[i32]; 5] = [&[0], &[4], &[4], &[1], &[2]];
    let mut raw = (codes.len() as u32).to_le_bytes().to_vec();
    for c in codes {
        raw.push(1);
        raw.extend_from_slice(&(c.len() as u32).to_le_bytes());
        for v in c {
            raw.extend_from_slice(&v.to_le_bytes());
        }
    }
    let recoder = Recoder::parse(&raw, false, &budget).unwrap();
    let net = Network {
        root: Layer::Input(Input::new("in", 5)),
        input_height: 36,
        num_outputs: 5,
        null_char: 4,
        x_scale: 3,
        rand: Arc::new(Rand::new(1)),
    };
    Recognizer::for_test(net, charset, recoder)
}

fn one_hot(codes: &[usize]) -> Tensor {
    let mut out = Tensor::new(
        StrideMap {
            height: 1,
            width: codes.len(),
        },
        5,
        &bud(),
    )
    .unwrap();
    let mut row = [0.0f64; 5];
    for (t, &c) in codes.iter().enumerate() {
        row.fill(0.001);
        row[c] = 0.9;
        out.write_time_step(t, &row);
    }
    out
}

#[test]
fn greedy_decode_collapses_repeats_and_drops_blanks() {
    let r = synthetic();
    let (a, b, null) = (1, 2, 4);
    let syms = r
        .greedy_decode(&one_hot(&[a, a, null, a, null, null, b]), &lim(), &nc())
        .unwrap();
    let text: String = syms.iter().map(|s| s.text.as_str()).collect();
    assert_eq!(text, "AAB");
    assert_eq!((syms[0].start, syms[0].end), (0, 2));
    assert_eq!((syms[1].start, syms[1].end), (3, 4));
    assert_eq!((syms[2].start, syms[2].end), (6, 7));
    assert!(
        r.greedy_decode(&one_hot(&[null, null]), &lim(), &nc())
            .unwrap()
            .is_empty()
    );
}

#[test]
fn beam_agrees_with_greedy_on_unambiguous_input() {
    let r = synthetic();
    for codes in [
        &[1usize, 1, 4, 1, 4, 4, 2][..],
        &[4, 4, 4],
        &[2],
        &[0, 1, 4, 2, 2, 0],
        &[1, 2, 1, 2, 1, 2],
    ] {
        let out = one_hot(codes);
        let greedy: String = r
            .greedy_decode(&out, &lim(), &nc())
            .unwrap()
            .iter()
            .map(|s| s.text.as_str())
            .collect();
        let beam: String = r
            .beam_decode(&out, &lim(), &nc())
            .unwrap()
            .iter()
            .map(|s| s.symbol.text.as_str())
            .collect();
        assert_eq!(beam, greedy, "codes {codes:?}");
    }
    assert!(
        r.beam_decode(&one_hot(&[]), &lim(), &nc())
            .unwrap()
            .is_empty()
    );
}

#[test]
fn beam_merges_transition_scores() {
    let r = synthetic();
    let (code, null, n) = (1usize, 4usize, 5usize);
    let mut out = Tensor::new(
        StrideMap {
            height: 1,
            width: 3,
        },
        n,
        &bud(),
    )
    .unwrap();
    let mut row = vec![0.05 / (n - 2) as f64; n];
    row[code] = 0.55;
    row[null] = 0.40;
    out.write_time_step(0, &row);
    row.fill(0.01 / (n - 2) as f64);
    row[code] = 0.98;
    row[null] = 0.01;
    out.write_time_step(1, &row);
    row.fill(0.02 / (n - 1) as f64);
    row[null] = 0.98;
    out.write_time_step(2, &row);
    let beam = r.beam_decode(&out, &lim(), &nc()).unwrap();
    assert_eq!(beam.len(), 1);
    assert_eq!(beam[0].symbol.text, "A");
    let cert = |p: f64| (prob_to_certainty(p) + CERT_OFFSET) * DICT_RATIO;
    let unmerged = cert(f64::from(0.55f32));
    let merged = cert(f64::from(0.55f32) + f64::from(0.40f32));
    assert!(
        beam[0].certainty > unmerged + 1e-9,
        "the P(code)+P(blank) merge is missing"
    );
    assert!(
        beam[0].certainty <= merged + 1e-9,
        "a timestep is double-counted"
    );
}

#[test]
fn beam_rejects_wrong_width_and_cancel() {
    let r = synthetic();
    let bad = Tensor::new(
        StrideMap {
            height: 1,
            width: 1,
        },
        4,
        &bud(),
    )
    .unwrap();
    assert!(r.beam_decode(&bad, &lim(), &nc()).is_err());
    let set = AtomicBool::new(true);
    assert_eq!(
        r.beam_decode(&one_hot(&[1]), &lim(), &set).unwrap_err(),
        crate::Error::Cancelled
    );
    let tight = Limits {
        max_alloc: 100,
        ..lim()
    };
    assert_eq!(
        r.beam_decode(&one_hot(&[1]), &tight, &nc()).unwrap_err(),
        crate::Error::AllocLimit
    );
}

#[test]
fn forward_and_recognize_on_a_synthetic_network() {
    // The synthetic network is a bare Input layer of 5 features, but the line image has 1
    // feature per timestep: the forward pass reports it.
    let r = synthetic();
    let img = gray(40, 36, 128);
    let e = r.forward(&img, &lim(), &nc()).unwrap_err();
    assert!(
        e.to_string().contains("network produced 1 outputs, want 5"),
        "{e}"
    );
    assert!(r.recognize_text(&img, &lim(), &nc()).is_err());
}

fn real_model() -> Option<Vec<u8>> {
    std::fs::read(format!(
        "{}/tests/data/eng.traineddata",
        env!("CARGO_MANIFEST_DIR")
    ))
    .ok()
}

#[test]
fn real_model_recognizes_a_blank_line_without_panicking() {
    let Some(raw) = real_model() else {
        eprintln!("eng.traineddata not present; skipping");
        return;
    };
    let mut r = Recognizer::new(&raw, &lim(), &nc()).unwrap();
    assert_eq!(r.net.input_height, 36);
    assert_eq!(r.net.x_scale, 3);
    let img = gray(200, 36, 255);
    let line = r.recognize(&img, &lim(), &nc()).unwrap();
    assert!(line.words.is_empty() || line.confidence >= 0.0);
    r.set_lexicon(false);
    assert!(r.dict().is_none());
    r.recognize_text(&img, &lim(), &nc()).unwrap();
}

#[test]
fn model_size_limit() {
    let small = Limits {
        max_model_bytes: 4,
        ..lim()
    };
    assert_eq!(
        Recognizer::new(&[0; 8], &small, &nc()).unwrap_err(),
        crate::Error::ModelLimit
    );
    assert!(Recognizer::new(&[0; 8], &lim(), &nc()).is_err());
}

struct CorpusLine {
    name: String,
    image: Bitmap,
    oracle: String,
}

/// One arm of `tests/data/lines` ("h36": already 36 px tall; "native"): the line images as PGM
/// (converted from the Go repo's PNGs) with the `tesseract --psm 13` oracle text.
fn load_corpus(arm: &str) -> Vec<CorpusLine> {
    let dir = format!("{}/tests/data/lines/{arm}", env!("CARGO_MANIFEST_DIR"));
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| {
            let n = e.ok()?.file_name().into_string().ok()?;
            n.strip_suffix(".pgm").map(str::to_string)
        })
        .collect();
    names.sort();
    names
        .into_iter()
        .map(|n| {
            let image = Bitmap::from_pgm(&std::fs::read(format!("{dir}/{n}.pgm")).unwrap(), &lim())
                .unwrap();
            let oracle = std::fs::read_to_string(format!("{dir}/{n}.psm13.txt")).unwrap();
            CorpusLine {
                name: n,
                image,
                oracle: oracle.trim_end_matches('\n').to_string(),
            }
        })
        .collect()
}

/// The Levenshtein character error rate of `got` against `want`.
fn cer(got: &str, want: &str) -> f64 {
    let (a, b): (Vec<char>, Vec<char>) = (want.chars().collect(), got.chars().collect());
    if a.is_empty() {
        return if b.is_empty() { 0.0 } else { 1.0 };
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0; b.len() + 1];
    for i in 1..=a.len() {
        curr[0] = i;
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            curr[j] = (curr[j - 1] + 1).min(prev[j] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()] as f64 / a.len() as f64
}

#[test]
fn cer_cases() {
    assert_eq!(cer("", ""), 0.0);
    assert_eq!(cer("a", ""), 1.0);
    assert_eq!(cer("abc", "abc"), 0.0);
    assert!((cer("abd", "abc") - 1.0 / 3.0).abs() < 1e-12);
    assert!((cer("ac", "abc") - 1.0 / 3.0).abs() < 1e-12);
}

#[test]
fn corpus_is_present_and_consistent() {
    for arm in ["h36", "native"] {
        let lines = load_corpus(arm);
        assert!(lines.len() >= 10, "arm {arm} has {} lines", lines.len());
        for l in &lines {
            assert!(
                !l.oracle.is_empty(),
                "{arm}/{}: oracle output is empty",
                l.name
            );
            if arm == "h36" {
                assert_eq!(l.image.height, 36, "h36/{}", l.name);
            }
        }
    }
}

fn real_recognizer() -> Option<Recognizer> {
    let raw = real_model()?;
    Some(Recognizer::new(&raw, &lim(), &nc()).unwrap())
}

#[test]
fn real_model_greedy_matches_oracle_h36() {
    let Some(r) = real_recognizer() else {
        eprintln!("eng.traineddata not present; skipping");
        return;
    };
    let lines = load_corpus("h36");
    let (mut exact, mut total) = (0, 0.0);
    for l in &lines {
        let got = r.recognize_text(&l.image, &lim(), &nc()).unwrap();
        if got == l.oracle {
            exact += 1;
        } else {
            eprintln!(
                "{}\n  oracle: {:?}\n  cadmus: {got:?}\n  cer: {:.4}",
                l.name,
                l.oracle,
                cer(&got, &l.oracle)
            );
        }
        total += cer(&got, &l.oracle);
    }
    let mean = total / lines.len() as f64;
    eprintln!(
        "h36 arm: {exact}/{} exact, mean CER vs oracle {mean:.4}",
        lines.len()
    );
    assert!(mean <= 0.02, "mean CER vs oracle = {mean:.4}");
}

#[test]
fn real_model_beam_matches_oracle() {
    let Some(mut r) = real_recognizer() else {
        eprintln!("eng.traineddata not present; skipping");
        return;
    };
    for arm in ["h36", "native"] {
        let lines = load_corpus(arm);
        let (mut exact, mut total) = (0, 0.0);
        for l in &lines {
            let line = r.recognize(&l.image, &lim(), &nc()).unwrap();
            if line.text == l.oracle {
                exact += 1;
            } else {
                eprintln!(
                    "{arm}/{}\n  oracle: {:?}\n  cadmus: {:?}\n  cer: {:.4}",
                    l.name,
                    l.oracle,
                    line.text,
                    cer(&line.text, &l.oracle)
                );
            }
            total += cer(&line.text, &l.oracle);
        }
        let mean = total / lines.len() as f64;
        eprintln!(
            "{arm} arm: {exact}/{} exact, mean CER vs oracle {mean:.4}",
            lines.len()
        );
        assert!(mean <= 0.01, "{arm} arm: mean CER vs oracle = {mean:.4}");
    }

    // The dictionary-free beam agrees with greedy on the h36 arm.
    r.set_lexicon(false);
    for l in load_corpus("h36") {
        let (out, _) = r.forward(&l.image, &lim(), &nc()).unwrap();
        let greedy: String = r
            .greedy_decode(&out, &lim(), &nc())
            .unwrap()
            .iter()
            .map(|s| s.text.as_str())
            .collect();
        let beam: String = r
            .beam_decode(&out, &lim(), &nc())
            .unwrap()
            .iter()
            .map(|s| s.symbol.text.as_str())
            .collect();
        assert_eq!(beam, greedy, "{}", l.name);
    }
}

#[test]
fn real_model_words_have_geometry() {
    let Some(r) = real_recognizer() else {
        eprintln!("eng.traineddata not present; skipping");
        return;
    };
    let l = &load_corpus("h36")[0];
    let line = r.recognize(&l.image, &lim(), &nc()).unwrap();
    assert!(!line.words.is_empty());
    let mut prev_right = 0;
    for w in &line.words {
        assert_eq!(w.bounds.min.y, line.bounds.min.y);
        assert_eq!(w.bounds.max.y, line.bounds.max.y);
        assert!(
            w.bounds.min.x >= prev_right - 1,
            "words are in reading order"
        );
        assert!((0.0..=100.0).contains(&w.confidence));
        prev_right = w.bounds.max.x;
    }
    let joined: Vec<&str> = line.words.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(
        joined.join(" ").split_whitespace().collect::<Vec<_>>(),
        line.text.split_whitespace().collect::<Vec<_>>()
    );
}

#[test]
fn decoders_reject_a_tensor_whose_public_fields_disagree_with_its_data() {
    let r = synthetic();
    let map = StrideMap {
        height: 1,
        width: 1,
    };
    let zero = Tensor::new(map, 0, &bud()).unwrap();
    // `features` set after the fact: the data is still empty.
    let mut lied = Tensor::new(map, 0, &bud()).unwrap();
    lied.features = 5;
    for t in [&zero, &lied] {
        assert!(r.greedy_decode(t, &lim(), &nc()).is_err());
        assert!(r.beam_decode(t, &lim(), &nc()).is_err());
    }
    assert!(
        score_symbols(&lied, &[sym(1, "A", 0, 1)], 1.0, &lim())
            .unwrap()
            .is_empty()
    );
    // No features: every timestep is the zero-probability certainty, and nothing panics.
    assert_eq!(
        score_symbols(&zero, &[sym(1, "A", 0, 1)], 1.0, &lim())
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn score_symbols_clamps_a_symbol_that_ends_past_the_tensor() {
    let out = one_hot(&[1, 1]);
    let scored = score_symbols(
        &out,
        &[sym(1, "A", 0, 99), sym(2, "B", 99, 100)],
        1.0,
        &lim(),
    )
    .unwrap();
    assert_eq!(scored.len(), 2);
}

#[test]
fn decoded_text_is_charged_to_the_budget() {
    let long = "A".repeat(10_000);
    let r = synthetic_with_a(&long);
    // 500 'A' symbols of 10,000 bytes each: 5 MB of text out of a 1000-step output.
    let codes: Vec<usize> = (0..1000).map(|i| 1 + i % 2).collect();
    let out = one_hot(&codes);
    let tight = Limits {
        max_alloc: 1 << 20,
        ..lim()
    };
    assert_eq!(
        r.greedy_decode(&out, &tight, &nc()).unwrap_err(),
        crate::Error::AllocLimit
    );
    let syms = r.greedy_decode(&out, &lim(), &nc()).unwrap();
    assert_eq!(syms.len(), 1000);
    assert_eq!(syms[0].text.len(), 10_000);
    // The scored copy and the joined text are charged too.
    let few = Limits {
        max_alloc: 200_000,
        ..lim()
    };
    assert_eq!(
        score_symbols(&out, &syms, 1.0, &few).unwrap_err(),
        crate::Error::AllocLimit
    );
    let b = Budget::new(&few);
    assert_eq!(
        decode::join_texts(syms.iter().map(|s| s.text.as_str()), &b),
        Err(crate::Error::AllocLimit)
    );
    assert_eq!(
        char_boundaries(&scored_n(30_000), 1, &few),
        Err(crate::Error::AllocLimit)
    );
}

fn scored_n(n: usize) -> Vec<Scored> {
    vec![scored(0, 1); n]
}

#[test]
fn the_beam_is_charged_for_what_it_holds() {
    let r = synthetic();
    let b = bud();
    let mut bufs = beam::BeamBufs::default();
    let beam = beam::Beam::new(&r, 5, &b, &mut bufs).unwrap();
    // 5 outputs: 8 bytes of unichar id and 1 of flag each, and the room for 5 final codes.
    assert_eq!(b.used(), 5 * 8 + 5 + 5 * 8);
    drop(beam);
}
