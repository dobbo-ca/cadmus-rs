//! cadmusdump: prints what is in a `.traineddata` model, or dumps every layer's activations for a
//! line image. A port of `cmd/cadmusdump`.
//!
//! Usage: `cadmusdump [-unicharset] [-recoder] [-dawgs] <model.traineddata>` and
//! `cadmusdump -activations <model.traineddata> <line.pgm>`.
//!
//! The Go tool reads a PNG. There is no PNG decoder in std and this crate has no image-decoding
//! dependency (only rayon, for the pool), so the line image is a binary PGM (`P5`, 8 bit grey).

#![forbid(unsafe_code)]
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented
    )
)]

use cadmus::imaging::Bitmap;
use cadmus::nn::{Ctx, Layer, Tensor};
use cadmus::recog::{Network, normalize};
use cadmus::tessdata::{self, Model, Type, quote};
use cadmus::{Budget, Limits};
use std::fmt::Write as _;
use std::io::{Read, Write};
use std::sync::atomic::AtomicBool;

#[derive(Default, Clone, Copy)]
struct Options {
    unicharset: bool,
    recoder: bool,
    dawgs: bool,
}

type Res<T> = Result<T, String>;

fn e<E: std::fmt::Display>(what: &str) -> impl FnOnce(E) -> String + '_ {
    move |err| format!("{what}: {err}")
}

/// Go's `%.9g` (`prec = Some(9)`) or `%g` of a float32 (`prec = None`, the shortest digits that
/// round-trip), which Rust's formatting has no equivalent of.
fn format_g(v: f64, prec: Option<usize>) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v < 0.0 {
            "-Inf".into()
        } else {
            "+Inf".into()
        };
    }
    let neg = v.is_sign_negative();
    let a = v.abs();
    if a == 0.0 {
        return if neg { "-0".into() } else { "0".into() };
    }
    let s = match prec {
        Some(p) => format!("{:.*e}", p - 1, a),
        None => format!("{:e}", a as f32),
    };
    let (mant, exp) = s.split_once('e').unwrap_or((&s, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let mut digits: String = mant.chars().filter(|c| *c != '.').collect();
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    let eprec = prec.unwrap_or(6) as i32;
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if exp < -4 || exp >= eprec {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(out, "e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    } else if exp >= 0 {
        let int_len = exp as usize + 1;
        if digits.len() <= int_len {
            out.push_str(&digits);
            out.push_str(&"0".repeat(int_len - digits.len()));
        } else {
            out.push_str(&digits[..int_len]);
            out.push('.');
            out.push_str(&digits[int_len..]);
        }
    } else {
        out.push_str("0.");
        out.push_str(&"0".repeat((-exp - 1) as usize));
        out.push_str(&digits);
    }
    out
}

/// Reads `path` whole, but refuses a file over `max` bytes without reading past it.
fn read_capped(path: &str, max: usize) -> std::io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    std::fs::File::open(path)?
        .take(max as u64 + 1)
        .read_to_end(&mut buf)?;
    if buf.len() > max {
        return Err(std::io::Error::other(format!("over {max} bytes")));
    }
    Ok(buf)
}

fn dump(path: &str, opt: Options, w: &mut impl Write) -> Res<()> {
    let limits = Limits::default();
    let raw = read_capped(path, limits.max_model_bytes).map_err(e(&format!("reading {path}")))?;
    let c = tessdata::parse_container(&raw, &limits).map_err(e("parsing container"))?;
    let version = c.version(&limits).map_err(e("reading the version"))?;
    let io = |r: std::io::Result<()>| r.map_err(|x| x.to_string());
    io(write!(
        w,
        "{path}\nversion: {version}\nbyte-swapped: {}\n\ncomponents:\n",
        c.swapped()
    ))?;
    for t in c.present() {
        let n = c.entry(t).map_or(0, <[u8]>::len);
        io(writeln!(w, "  {:<20} {n:>9} bytes", t.name()))?;
    }
    if c.entry(Type::Lstm).is_none() {
        io(writeln!(w, "\nno lstm component (legacy-only model)"))?;
        return Ok(());
    }
    let m = Model::load(&raw, &limits, &AtomicBool::new(false)).map_err(e("loading model"))?;
    let rec = &m.recognizer;
    io(write!(
        w,
        "\nrecognizer:\n  spec           {}\n  training_flags {}\n  iteration      {}\n  sample_iter    {}\n  null_char      {}\n  adam_beta      {}\n  learning_rate  {}\n  momentum       {}\n\nnetwork:\n",
        rec.network_str,
        rec.training_flags,
        rec.training_iteration,
        rec.sample_iteration,
        rec.null_char,
        format_g(f64::from(rec.adam_beta), None),
        format_g(f64::from(rec.learning_rate), None),
        format_g(f64::from(rec.momentum), None),
    ))?;
    io(write!(
        w,
        "{}",
        rec.network
            .tree(&limits)
            .map_err(e("rendering the network"))?
    ))?;
    if opt.unicharset {
        let u = &m.unicharset;
        io(writeln!(w, "\nunicharset: {} entries", u.size()))?;
        for id in 0..u.size() {
            let Some(c) = u.char(id) else { continue };
            io(write!(
                w,
                "  {id:>3} {:<14} props={:x} script={:<8}",
                quote(&c.text, &limits).map_err(e("quoting a unicharset entry"))?,
                c.properties,
                c.script
            ))?;
            if c.normed != c.text {
                io(write!(
                    w,
                    " normed={}",
                    quote(&c.normed, &limits).map_err(e("quoting a unicharset entry"))?
                ))?;
            }
            io(writeln!(w))?;
        }
    }
    if opt.recoder {
        let rc = &m.recoder;
        io(writeln!(
            w,
            "\nrecoder: {} entries, code range {}, max code length {}",
            rc.size(),
            rc.code_range(),
            rc.max_code_len()
        ))?;
        for id in 0..rc.size() {
            let code = rc.encode(id).unwrap_or(&[]);
            // Go prints the slice with %v: space-separated.
            let codes: Vec<String> = code.iter().map(ToString::to_string).collect();
            io(write!(
                w,
                "  unichar {id:>3} {:<14} -> code [{}]",
                quote(m.unicharset.text(id), &limits).map_err(e("quoting a unicharset entry"))?,
                codes.join(" ")
            ))?;
            if code.len() == 1 && code[0] as usize == m.null_char() {
                io(write!(w, "  BLANK"))?;
            }
            io(writeln!(w))?;
        }
    }
    if opt.dawgs {
        io(writeln!(w, "\ndawgs:"))?;
        for (name, d) in [
            ("punc", &m.punc_dawg),
            ("word", &m.word_dawg),
            ("number", &m.number_dawg),
        ] {
            match d {
                None => io(writeln!(w, "  {name:<8} absent"))?,
                Some(d) => io(writeln!(
                    w,
                    "  {name:<8} {:>9} edges, unicharset size {}",
                    d.num_edges(),
                    d.unicharset_size
                ))?,
            }
        }
    }
    Ok(())
}

fn dump_activations(model_path: &str, image_path: &str, w: &mut impl Write) -> Res<()> {
    let limits = Limits::default();
    let raw = read_capped(model_path, limits.max_model_bytes)
        .map_err(e(&format!("reading {model_path}")))?;
    let budget = Budget::new(&limits);
    let cancel = AtomicBool::new(false);
    let c = tessdata::parse_container(&raw, &limits).map_err(e("parsing container"))?;
    let lstm = c
        .entry(Type::Lstm)
        .ok_or_else(|| format!("{model_path} has no lstm component"))?;
    let rec = tessdata::parse_recognizer(lstm, c.swapped(), &budget, &cancel)
        .map_err(e("parsing recognizer"))?;
    // A PGM is the pixels plus a short header.
    let img_cap = usize::try_from(limits.max_pixels)
        .unwrap_or(usize::MAX)
        .saturating_add(1 << 16);
    let img_raw = read_capped(image_path, img_cap).map_err(e(&format!("opening {image_path}")))?;
    let img = Bitmap::from_pgm(&img_raw, &limits).map_err(e(&format!("decoding {image_path}")))?;
    let net = Network::build(&rec, &limits, &cancel).map_err(e("building network"))?;
    let norm = normalize(
        &img,
        net.input_height,
        net.x_scale,
        &net.rand,
        &limits,
        &cancel,
    )
    .map_err(e("normalizing"))?;
    write_block(w, "Normalized", &norm.input)?;

    let mut dump_err: Option<String> = None;
    let mut tap = |l: &Layer, out: &Tensor| {
        if dump_err.is_none() {
            dump_err = write_block(w, l.name(), out).err();
        }
    };
    let fwd_budget = Budget::new(&limits);
    let mut cx = Ctx {
        budget: &fwd_budget,
        tap: Some(&mut tap),
        cancel: Some(&cancel),
        pool: None,
    };
    net.root
        .forward(&norm.input, &mut cx)
        .map_err(e("forward"))?;
    match dump_err {
        Some(x) => Err(x),
        None => Ok(()),
    }
}

fn write_block(w: &mut impl Write, name: &str, x: &Tensor) -> Res<()> {
    let io = |r: std::io::Result<()>| r.map_err(|x| x.to_string());
    io(writeln!(w, "Output:{name}"))?;
    for feat in 0..x.features {
        let mut line = String::new();
        for t in 0..x.map.len() {
            line.push(' ');
            line.push_str(&format_g(f64::from(x.row(t)[feat]), Some(9)));
        }
        line.push('\n');
        io(w.write_all(line.as_bytes()))?;
    }
    Ok(())
}

fn main() {
    let mut opt = Options::default();
    let mut activations = false;
    let mut args = Vec::new();
    for a in std::env::args().skip(1) {
        match a.trim_start_matches('-') {
            "unicharset" if a.starts_with('-') => opt.unicharset = true,
            "recoder" if a.starts_with('-') => opt.recoder = true,
            "dawgs" if a.starts_with('-') => opt.dawgs = true,
            "activations" if a.starts_with('-') => activations = true,
            _ => args.push(a),
        }
    }
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    let (res, code) = if activations {
        if args.len() != 2 {
            eprintln!("usage: cadmusdump -activations <model.traineddata> <line.pgm>");
            std::process::exit(2);
        }
        (dump_activations(&args[0], &args[1], &mut out), 1)
    } else {
        if args.len() != 1 {
            eprintln!("usage: cadmusdump [-unicharset] [-recoder] [-dawgs] <model.traineddata>");
            std::process::exit(2);
        }
        (dump(&args[0], opt, &mut out), 1)
    };
    let _ = out.flush();
    if let Err(err) = res {
        eprintln!("cadmusdump: {err}");
        std::process::exit(code);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_path() -> Option<String> {
        let p = format!("{}/tests/data/eng.traineddata", env!("CARGO_MANIFEST_DIR"));
        std::path::Path::new(&p).exists().then_some(p)
    }

    #[test]
    fn go_percent_g() {
        assert_eq!(format_g(0.123456789012, Some(9)), "0.123456789");
        assert_eq!(format_g(1e-5, Some(9)), "1e-05");
        assert_eq!(format_g(0.0001, Some(9)), "0.0001");
        assert_eq!(format_g(-1.5, Some(9)), "-1.5");
        assert_eq!(format_g(123456789.0, Some(9)), "123456789");
        assert_eq!(format_g(1234567890.0, Some(9)), "1.23456789e+09");
        assert_eq!(format_g(100.0, Some(9)), "100");
        assert_eq!(format_g(0.0, Some(9)), "0");
        assert_eq!(format_g(f64::from(0.999f32), None), "0.999");
        assert_eq!(format_g(f64::from(0.001f32), None), "0.001");
        assert_eq!(format_g(f64::from(1e-5f32), None), "1e-05");
        assert_eq!(format_g(f64::from(0.5f32), None), "0.5");
        assert_eq!(format_g(1000000.0, None), "1e+06");
        assert_eq!(format_g(100000.0, None), "100000");
        // A float32 widened to f64 prints with 9 digits, as Go prints a float32.
        assert_eq!(format_g(f64::from(0.1f32), Some(9)), "0.100000001");
    }

    #[test]
    fn go_percent_q() {
        let q = |s: &str| quote(s, &Limits::default()).unwrap();
        assert_eq!(q("a"), "\"a\"");
        assert_eq!(q(" "), "\" \"");
        assert_eq!(q("\u{2019}"), "\"\u{2019}\"");
        assert_eq!(q("\"\\\n"), "\"\\\"\\\\\\n\"");
        // strconv.Quote: \a \b \f \v, \x for other ASCII controls, \u and \U for the rest that
        // IsPrint rejects (C1 controls, no-break space, zero-width joiner, unassigned).
        assert_eq!(
            q("\u{7}\u{8}\u{c}\u{b}\u{1}\u{7f}"),
            "\"\\a\\b\\f\\v\\x01\\x7f\""
        );
        assert_eq!(q("\u{85}"), "\"\\u0085\"");
        assert_eq!(q("\u{a0}\u{200d}"), "\"\\u00a0\\u200d\"");
        assert_eq!(q("\u{e0001}"), "\"\\U000e0001\"");
        assert_eq!(q("'\u{301}\u{1f600}"), "\"'\u{301}\u{1f600}\"");
        // Default-ignorable letters and marks are printable to Go: a variation selector, the
        // Hangul filler, a combining grapheme joiner. The format characters are not.
        assert_eq!(q("\u{2764}\u{fe0f}"), "\"\u{2764}\u{fe0f}\"");
        assert_eq!(q("\u{3164}\u{ffa0}\u{34f}"), "\"\u{3164}\u{ffa0}\u{34f}\"");
        assert_eq!(q("\u{e0100}"), "\"\u{e0100}\"");
        assert_eq!(q("\u{ad}\u{feff}"), "\"\\u00ad\\ufeff\"");
        // The most it writes is four bytes per byte plus the quotes, and that is charged.
        let tight = Limits {
            max_alloc: 9,
            ..Limits::default()
        };
        assert_eq!(quote("\u{1}\u{1}", &tight), Err(cadmus::Error::AllocLimit));
        assert_eq!(
            quote(
                "\u{1}\u{1}",
                &Limits {
                    max_alloc: 10,
                    ..tight
                }
            )
            .unwrap(),
            "\"\\x01\\x01\""
        );
    }

    #[test]
    fn missing_file_errors() {
        let mut sink = Vec::new();
        assert!(
            dump(
                "/nonexistent/model.traineddata",
                Options::default(),
                &mut sink
            )
            .is_err()
        );
        assert!(dump_activations("/nonexistent/m", "/nonexistent/i", &mut sink).is_err());
    }

    /// The assertions of `cmd/cadmusdump/main_test.go`, as written.
    #[test]
    fn go_dump_assertions() {
        let Some(model) = model_path() else {
            eprintln!("eng.traineddata not present; skipping");
            return;
        };
        let run = |opt: Options| {
            let mut out = Vec::new();
            dump(&model, opt, &mut out).unwrap();
            String::from_utf8(out).unwrap()
        };
        let out = run(Options::default());
        for want in [
            "lstm-unicharset",
            "lstm-recoder",
            "null_char      110",
            "sample_iter    814136",
            "[1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]",
            "Series \"Series\" ni=36 no=111 weights=1461007",
            "Softmax \"Output\" ni=512 no=111 weights=56943",
            "111x513 min=-29.279424 max=35.155323",
            "LSTM \"Lfx512\" ni=96 no=512 na=608",
            "[2] GF1 512x609",
            "half_x=1 half_y=1",
            "x_scale=3 y_scale=3",
            "shape batch=1 height=36 width=0 depth=1 loss=0",
            "\n\nnetwork:\n",
        ] {
            assert!(out.contains(want), "dump output missing {want:?}\n{out}");
        }
        for not_want in ["unicharset:", "recoder:", "dawgs:"] {
            assert!(!out.contains(not_want), "{not_want} without its flag");
        }

        let out = run(Options {
            unicharset: true,
            ..Options::default()
        });
        for want in [
            "unicharset: 112 entries",
            "    0 \" \"",
            "    2 \"|Broken|0|1\"",
            "props=f",
            "normed=\"'\"",
        ] {
            assert!(out.contains(want), "unicharset section missing {want:?}");
        }

        let out = run(Options {
            recoder: true,
            ..Options::default()
        });
        for want in [
            "recoder: 112 entries, code range 111, max code length 1",
            "-> code [110]",
            "BLANK",
        ] {
            assert!(out.contains(want), "recoder section missing {want:?}");
        }

        let out = run(Options {
            dawgs: true,
            ..Options::default()
        });
        for want in ["dawgs:", "word", "461848 edges"] {
            assert!(out.contains(want), "dawg section missing {want:?}");
        }
    }

    #[test]
    fn oversized_input_is_refused_before_it_is_read_whole() {
        let dir = std::env::temp_dir().join(format!("cadmusdump-cap-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("big");
        std::fs::write(&f, [0u8; 100]).unwrap();
        let p = f.to_str().unwrap();
        assert!(read_capped(p, 100).is_ok());
        assert!(read_capped(p, 99).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn default_sections_and_activation_blocks() {
        let Some(model) = model_path() else {
            eprintln!("eng.traineddata not present; skipping");
            return;
        };
        let mut out = Vec::new();
        dump(
            &model,
            Options {
                unicharset: true,
                recoder: true,
                dawgs: true,
            },
            &mut out,
        )
        .unwrap();
        let text = String::from_utf8(out).unwrap();
        for want in [
            "components:",
            "recognizer:",
            "network:",
            "unicharset:",
            "recoder:",
            "dawgs:",
            "BLANK",
        ] {
            assert!(text.contains(want), "missing {want}");
        }

        let dir = std::env::temp_dir().join(format!("cadmusdump-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (w, h) = (60usize, 36usize);
        let mut pix = vec![235u8; w * h];
        for x0 in (2..w - 2).step_by(7) {
            for y in h / 4..3 * h / 4 {
                pix[y * w + x0] = 30;
                pix[y * w + x0 + 1] = 30;
            }
        }
        let mut pgm = format!("P5\n{w} {h}\n255\n").into_bytes();
        pgm.extend(pix);
        let line = dir.join("line.pgm");
        std::fs::write(&line, pgm).unwrap();
        let mut out = Vec::new();
        dump_activations(&model, line.to_str().unwrap(), &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        let names: Vec<&str> = text
            .lines()
            .filter_map(|l| l.strip_prefix("Output:"))
            .collect();
        assert_eq!(names.first(), Some(&"Normalized"));
        assert!(!names.contains(&"Input"));
        assert_eq!(names.last(), Some(&"Series"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
