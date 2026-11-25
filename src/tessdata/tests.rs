use super::network::parse_layer;
use super::*;

fn lim() -> Limits {
    Limits::default()
}

fn bud() -> Budget {
    Budget::new(&lim())
}

fn nc() -> AtomicBool {
    AtomicBool::new(false)
}

/// The real `eng.traineddata` (tessdata_best 4.1.0), if the fixture was fetched. Not in the
/// repository (15 MB): the tests that need it return early without it, as the Go tests skip.
pub(crate) fn real_model() -> Option<Vec<u8>> {
    std::fs::read(format!(
        "{}/tests/data/eng.traineddata",
        env!("CARGO_MANIFEST_DIR")
    ))
    .ok()
}

fn is_err_containing(e: Error, s: &str) -> bool {
    e.to_string().contains(s)
}

#[test]
fn reader_primitives_little_endian() {
    let mut r = Reader::new(&[1, 0, 0, 0, 0xfe, 0xff, 0xff, 0xff, 7]);
    assert_eq!(r.uint32().unwrap(), 1);
    assert_eq!(r.int32().unwrap(), -2);
    assert_eq!(r.int8().unwrap(), 7);
    assert_eq!(r.remaining(), 0);
}

#[test]
fn reader_swap_reverses_multi_byte_reads() {
    let mut r = Reader::new(&[0, 0, 0, 1]);
    r.set_swap(true);
    assert_eq!(r.uint32().unwrap(), 1);
    let mut r = Reader::new(&[0x00, 0x2a]);
    r.set_swap(true);
    assert_eq!(r.int16().unwrap(), 42);
    let mut r = Reader::new(&[0, 0, 0, 0, 0, 0x13, 0x1a, 0x09]);
    r.set_swap(true);
    assert_eq!(r.uint64().unwrap(), 0x131a09);
    let mut r = Reader::new(&[0x3a, 0x83, 0x12, 0x6f]);
    r.set_swap(true);
    assert_eq!(r.float32().unwrap(), 0.001f32);
}

#[test]
fn reader_other_widths() {
    let mut r = Reader::new(&[0x2a, 0x00, 0xff, 0xff]);
    assert_eq!(r.int16().unwrap(), 42);
    assert_eq!(r.int16().unwrap(), -1);
    assert_eq!(r.remaining(), 0);
    let mut r = Reader::new(&[0x09, 0x1a, 0x13, 0, 0, 0, 0, 0]);
    assert_eq!(r.uint64().unwrap(), 0x131a09);
    let mut r = Reader::new(&[0x6f, 0x12, 0x83, 0x3a]);
    assert_eq!(r.float32().unwrap(), 0.001f32);
    let bytes = 1.5f64.to_le_bytes();
    let mut r = Reader::new(&bytes);
    assert_eq!(r.float64().unwrap(), 1.5);
}

#[test]
fn reader_string_and_truncation() {
    let mut r = Reader::new(&[3, 0, 0, 0, b'a', b'b', b'c']);
    let b = Budget::new(&Limits::default());
    assert_eq!(r.string(&b).unwrap(), "abc");
    assert!(Reader::new(&[0xff, 0xff, 0xff, 0xff]).string(&b).is_err());
    // Within the guard but beyond the buffer: an error, and no allocation of the claimed size.
    assert!(
        Reader::new(&[0xff, 0xff, 0xff, 0x01, 1])
            .string(&b)
            .is_err()
    );
    // The copy is charged before it is made.
    let tiny = Budget::new(&Limits {
        max_alloc: 2,
        ..Limits::default()
    });
    assert_eq!(
        Reader::new(&[3, 0, 0, 0, b'a', b'b', b'c']).string(&tiny),
        Err(Error::AllocLimit)
    );
    assert!(Reader::new(&[1, 2]).uint32().is_err());
    assert!(Reader::new(&[0x2a]).int16().is_err());
}

fn build_container(entries: &[(Type, &[u8])]) -> Vec<u8> {
    let header = 4 + 8 * NUM_TYPES;
    let mut offsets = vec![-1i64; NUM_TYPES];
    let mut body = Vec::new();
    for (t, payload) in entries {
        offsets[*t as usize] = (header + body.len()) as i64;
        body.extend_from_slice(payload);
    }
    let mut out = (NUM_TYPES as u32).to_le_bytes().to_vec();
    for off in offsets {
        out.extend_from_slice(&off.to_le_bytes());
    }
    out.extend(body);
    out
}

#[test]
fn container_round_trip() {
    let raw = build_container(&[
        (Type::Lstm, b"network-bytes"),
        (Type::LstmUnicharset, b"charset"),
        (Type::Version, b"5.0.0-test\n\0"),
    ]);
    let c = parse_container(&raw, &lim()).unwrap();
    assert_eq!(c.entry(Type::Lstm).unwrap(), b"network-bytes");
    assert_eq!(c.entry(Type::LstmUnicharset).unwrap(), b"charset");
    assert_eq!(c.version(&lim()).unwrap(), "5.0.0-test");
    assert!(c.entry(Type::Ambigs).is_none());
    assert_eq!(
        c.present(),
        [Type::Lstm, Type::LstmUnicharset, Type::Version]
    );
    assert!(!c.swapped());
}

#[test]
fn container_last_entry_runs_to_eof_and_no_version() {
    let raw = build_container(&[(Type::Version, b"tail")]);
    assert_eq!(
        parse_container(&raw, &lim())
            .unwrap()
            .entry(Type::Version)
            .unwrap(),
        b"tail"
    );
    let raw = build_container(&[(Type::Lstm, b"x")]);
    assert_eq!(
        parse_container(&raw, &lim())
            .unwrap()
            .version(&lim())
            .unwrap(),
        ""
    );
}

#[test]
fn container_rejects_bad_headers() {
    assert!(parse_container(&2000u32.to_le_bytes(), &lim()).is_err());
    assert!(parse_container(&[1, 2], &lim()).is_err());
    // An offset past the end of the file.
    let mut raw = build_container(&[(Type::Lstm, b"x")]);
    let off = 4 + 8 * Type::Lstm as usize;
    raw[off..off + 8].copy_from_slice(&1_000_000i64.to_le_bytes());
    assert!(parse_container(&raw, &lim()).is_err());
}

#[test]
fn container_foreign_endian() {
    // 24 entries written big-endian.
    let mut raw = 24u32.to_be_bytes().to_vec();
    for i in 0..24usize {
        let off: i64 = if i == Type::Version as usize {
            4 + 8 * 24
        } else {
            -1
        };
        raw.extend_from_slice(&off.to_be_bytes());
    }
    raw.extend_from_slice(b"v1");
    let c = parse_container(&raw, &lim()).unwrap();
    assert!(c.swapped());
    assert_eq!(c.version(&lim()).unwrap(), "v1");
}

#[test]
fn model_rejects_missing_components_garbage_and_size() {
    let only = build_container(&[(Type::Version, b"x")]);
    assert!(Model::load(&only, &lim(), &nc()).is_err());
    assert!(Model::load(&[0xde, 0xad, 0xbe, 0xef], &lim(), &nc()).is_err());
    let small = Limits {
        max_model_bytes: 3,
        ..lim()
    };
    assert_eq!(
        Model::load(&only, &small, &nc()).unwrap_err(),
        Error::ModelLimit
    );
}

#[test]
fn real_model_loads() {
    let Some(raw) = real_model() else {
        eprintln!("eng.traineddata not present; skipping");
        return;
    };
    let m = Model::load(&raw, &lim(), &nc()).unwrap();
    assert!(m.softmax().is_some());
    assert_eq!(m.num_outputs(), m.recoder.code_range());
    assert_eq!(m.null_char(), m.recoder.code_range() - 1);
    assert!(m.word_dawg.is_some());
    assert_eq!(m.unicharset.size(), m.recoder.size());
    assert!(!m.recognizer.network.tree(&lim()).unwrap().is_empty());
}

fn make_edge(id: u64, last: bool, eow: bool, next: u64) -> u64 {
    let mut e = id;
    if last {
        e |= 1 << 7;
    }
    if eow {
        e |= 4 << 7;
    }
    e | next << 10
}

fn build_dawg(size: u32, edges: &[u64]) -> Vec<u8> {
    let mut b = 42i16.to_le_bytes().to_vec();
    b.extend_from_slice(&size.to_le_bytes());
    b.extend_from_slice(&(edges.len() as u32).to_le_bytes());
    for e in edges {
        b.extend_from_slice(&e.to_le_bytes());
    }
    b
}

fn parse_dawg(size: u32, edges: &[u64]) -> Result<Dawg, Error> {
    Dawg::parse(&build_dawg(size, edges), false, &bud())
}

#[test]
fn dawg_flag_start_bit_is_bit_length() {
    for (size, want) in [(1, 1), (63, 6), (64, 7), (112, 7), (128, 8)] {
        assert_eq!(
            Dawg::from_edges(size, vec![]).flag_start_bit(),
            want,
            "size {size}"
        );
    }
}

#[test]
fn dawg_tiny_graph() {
    let edges = [
        make_edge(1, true, false, 2),
        make_edge(9, true, false, 0),
        make_edge(2, true, true, 0),
    ];
    let d = parse_dawg(112, &edges).unwrap();
    assert!(d.contains(&[1, 2]));
    assert!(!d.contains(&[1]), "'a' alone is not word-end");
    assert!(d.has_prefix(&[1]));
    assert!(!d.contains(&[1, 3]));
    assert!(d.has_prefix(&[]));
    assert!(!d.contains(&[]));
    assert_eq!(d.num_edges(), 3);
}

#[test]
fn dawg_parse_errors() {
    let good = [make_edge(1, true, true, 0)];
    let mut bad = build_dawg(112, &good);
    bad[0] = 43;
    assert!(Dawg::parse(&bad, false, &bud()).is_err(), "magic");
    assert!(parse_dawg(112, &[]).is_err(), "0 edges");
    assert!(parse_dawg(0, &good).is_err(), "size 0");
    let mut trailing = build_dawg(112, &good);
    trailing.push(0xde);
    assert!(Dawg::parse(&trailing, false, &bud()).is_err(), "trailing");
    let t = build_dawg(112, &good);
    assert!(
        Dawg::parse(&t[..t.len() - 2], false, &bud()).is_err(),
        "truncated"
    );
    assert!(
        parse_dawg(112, &[make_edge(1, true, true, 0) | 2 << 7]).is_err(),
        "backward"
    );
    assert!(
        parse_dawg(112, &[make_edge(1, true, true, 99)]).is_err(),
        "next_node"
    );
    assert!(
        parse_dawg(1, &[make_edge(1, true, true, 0)]).is_err(),
        "unichar id"
    );
    // A declared edge count the buffer cannot hold is rejected before any allocation.
    let mut huge = build_dawg(112, &good);
    huge[6..10].copy_from_slice(&40_000_000u32.to_le_bytes());
    assert!(Dawg::parse(&huge, false, &bud()).is_err(), "huge count");
}

fn synthetic_dawg() -> Dawg {
    parse_dawg(
        112,
        &[make_edge(1, true, false, 1), make_edge(2, true, true, 0)],
    )
    .unwrap()
}

#[test]
fn dawg_edge_accessors() {
    let d = synthetic_dawg();
    let e = d.edge_char(0, 1, false);
    assert_eq!(e, 0);
    assert_eq!(d.unichar_id(e), 1);
    assert_eq!(d.next_node(e), 1);
    assert!(!d.end_of_word(e));
    let e2 = d.edge_char(1, 2, true);
    assert_eq!(e2, 1);
    assert!(d.end_of_word(e2));
    assert_eq!(d.edge_char(0, 99, false), NO_EDGE);
    assert_eq!(d.edge_char(-1, 1, false), NO_EDGE);
    assert_eq!(d.edge_char(d.num_edges() as i64, 1, false), NO_EDGE);
    assert_eq!(d.edge_record(0) & 0x7f, 1);
}

#[test]
fn edge_char_skips_an_empty_slot() {
    let mut d = synthetic_dawg();
    let slot = d.empty_slot_value();
    d.set_edge(0, slot);
    assert_eq!(d.edge_char(0, PATTERN_UNICHAR_ID, false), NO_EDGE);
}

fn build_recoder(codes: &[&[i32]]) -> Vec<u8> {
    let mut b = (codes.len() as u32).to_le_bytes().to_vec();
    for code in codes {
        b.push(1);
        b.extend_from_slice(&(code.len() as u32).to_le_bytes());
        for v in *code {
            b.extend_from_slice(&v.to_le_bytes());
        }
    }
    b
}

fn parse_recoder(codes: &[&[i32]]) -> Result<Recoder, Error> {
    Recoder::parse(&build_recoder(codes), false, &bud())
}

#[test]
fn recoder_higher_id_wins_on_collision() {
    let rc = parse_recoder(&[&[0], &[5], &[5]]).unwrap();
    assert_eq!(rc.decode_unichar(&[5]), Some(2));
    assert_eq!(rc.code_range(), 6);
    assert_eq!(rc.size(), 3);
    assert_eq!(rc.encode(1), Some(&[5][..]));
    assert_eq!(rc.encode(9), None);
}

#[test]
fn recoder_multi_code() {
    let rc = parse_recoder(&[&[0], &[1, 2, 3]]).unwrap();
    assert_eq!(rc.max_code_len(), 3);
    assert_eq!(rc.decode_unichar(&[1, 2, 3]), Some(1));
    assert_eq!(rc.decode_unichar(&[1, 2]), None);
    assert_eq!(rc.decode_unichar(&[]), None);
    assert!(rc.is_valid_first_code(1));
    assert!(!rc.is_valid_first_code(2));
    assert!(!rc.is_valid_first_code(-1));
}

#[test]
fn recoder_errors() {
    let mut too_long = 1u32.to_le_bytes().to_vec();
    too_long.push(1);
    too_long.extend_from_slice(&10u32.to_le_bytes());
    assert!(Recoder::parse(&too_long, false, &bud()).is_err());
    let mut trailing = build_recoder(&[&[0]]);
    trailing.extend([0xde, 0xad]);
    assert!(Recoder::parse(&trailing, false, &bud()).is_err());
    assert!(parse_recoder(&[&[7], &[0]]).is_err(), "garbled space");
    assert!(parse_recoder(&[&[0], &[-1]]).is_err(), "negative code");
    assert!(Recoder::parse(&[1], false, &bud()).is_err());
    // A count the buffer cannot hold.
    assert!(Recoder::parse(&40_000_000u32.to_le_bytes(), false, &bud()).is_err());
}

fn unicharset(s: &str) -> Result<Unicharset, Error> {
    parse_unicharset(s.as_bytes(), &bud())
}

#[test]
fn unicharset_shapes() {
    let u = unicharset(
        "2\nNULL 0 Common 0\né 3 0,255,0,255,0,0,0,0,0,0 Latin 111 0 111 e\t# é [e9 ]a\n",
    )
    .unwrap();
    assert_eq!(u.size(), 2);
    assert_eq!((u.text(0), u.normed(0)), (" ", " "));
    assert_eq!((u.text(1), u.normed(1)), ("é", "e"));
    let c = u.char(1).unwrap();
    assert_eq!(c.properties, PROP_ALPHA | PROP_LOWER);
    assert_eq!(c.script, "Latin");
    assert!(u.char(2).is_none());
    assert_eq!(u.text(9), "");
    assert!(u.is_space_delimited(1));
    assert!(u.is_space_delimited(99));
}

#[test]
fn unicharset_errors() {
    for (name, input) in [
        ("empty", ""),
        ("bad count", "abc\nNULL 0 Common 0\n"),
        ("count too high", "3\nNULL 0 Common 0\n"),
        ("non-hex property", "1\nA zz Latin 0\n"),
        ("signed property", "1\nA +3 Latin 0\n"),
        ("too few fields", "1\nA 3\n"),
        ("zero count", "0\n"),
        ("huge count", "99999999999\nNULL 0 Common 0\n"),
    ] {
        assert!(unicharset(input).is_err(), "{name}");
    }
    assert!(
        parse_unicharset(&[b'1', b'\n', 0xff], &bud()).is_err(),
        "not UTF-8"
    );
}

#[test]
fn unicharset_ignores_trailing_lines_and_script_delimiters() {
    let u = unicharset(
        "2\nNULL 0 Common 0\nJoined 7 0,255,0,255,0,0,0,0,0,0 Han 1 0 1 Joined\nextra line\n",
    )
    .unwrap();
    assert_eq!(u.size(), 2);
    assert!(u.is_space_delimited(0));
    assert!(!u.is_space_delimited(1));
}

fn layer_header(
    typ: LayerType,
    training: i8,
    flags: i32,
    ni: i32,
    no: i32,
    nw: i32,
    name: &str,
) -> Vec<u8> {
    let mut b = vec![typ as u8, training as u8, 0];
    for v in [flags, ni, no, nw, name.len() as i32] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.extend_from_slice(name.as_bytes());
    b
}

fn float_matrix(mode: u8, rows: i32, cols: i32, values: &[f64]) -> Vec<u8> {
    let mut b = vec![mode];
    b.extend_from_slice(&rows.to_le_bytes());
    b.extend_from_slice(&cols.to_le_bytes());
    b.extend_from_slice(&0f64.to_le_bytes());
    for v in values {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

const VALS: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

fn tanh_layer(training: i8, nw: i32, mode: u8) -> Vec<u8> {
    let mut d = layer_header(LayerType::Tanh, training, 0, 2, 2, nw, "T");
    d.extend(float_matrix(mode, 2, 3, &VALS));
    d
}

fn parse_one(d: &[u8]) -> Result<Layer, Error> {
    let mut r = Reader::new(d);
    let l = parse_layer(&mut r, &bud(), &nc(), 0)?;
    assert_eq!(r.remaining(), 0);
    Ok(l)
}

#[test]
fn matrix_reads_values_row_major() {
    let l = parse_one(&tanh_layer(2, 6, 132)).unwrap();
    assert_eq!(l.matrices.len(), 1);
    let m = &l.matrices[0];
    assert_eq!((m.rows, m.cols), (2, 3));
    for row in 0..2 {
        for col in 0..3 {
            assert_eq!(m.at(row, col), VALS[row * 3 + col]);
        }
    }
    assert_eq!(m.stats(), (1.0, 6.0, 3.5));
    assert_eq!(
        Matrix {
            rows: 0,
            cols: 0,
            values: vec![]
        }
        .stats(),
        (0.0, 0.0, 0.0)
    );
    assert!(
        l.tree(&lim())
            .unwrap()
            .contains("Tanh \"T\" ni=2 no=2 weights=6")
    );
}

// The tree quotes a name as Go's `%q` does: control characters as \xNN, other non-printable
// characters as \uNNNN, and a combining mark as itself.
#[test]
fn tree_quotes_names_like_go() {
    let name = "\u{1}\u{7f}\u{a0}\u{200b}\"\\a\u{301}";
    let mut d = layer_header(LayerType::Tanh, 2, 0, 2, 2, 6, name);
    d.extend(float_matrix(132, 2, 3, &VALS));
    let tree = parse_one(&d).unwrap().tree(&lim()).unwrap();
    assert!(
        tree.starts_with("Tanh \"\\x01\\x7f\\u00a0\\u200b\\\"\\\\a\u{301}\" ni=2 no=2 weights=6\n"),
        "{tree}"
    );
    let tight = Limits {
        max_alloc: 10,
        ..lim()
    };
    assert_eq!(
        parse_one(&d).unwrap().tree(&tight).unwrap_err(),
        Error::AllocLimit
    );
}

// strconv.IsPrint is true for the default-ignorable letters and marks (a variation selector, the
// Hangul fillers), which Rust's `escape_debug` escapes; it is false for the format characters.
#[test]
fn tree_prints_default_ignorable_marks_like_go() {
    let mut d = layer_header(
        LayerType::Tanh,
        2,
        0,
        2,
        2,
        6,
        "\u{2764}\u{fe0f}\u{3164}\u{ffa0}\u{ad}\u{e0100}",
    );
    d.extend(float_matrix(132, 2, 3, &VALS));
    let tree = parse_one(&d).unwrap().tree(&lim()).unwrap();
    assert!(
        tree.starts_with(
            "Tanh \"\u{2764}\u{fe0f}\u{3164}\u{ffa0}\\u00ad\u{e0100}\" ni=2 no=2 weights=6\n"
        ),
        "{tree}"
    );
}

// An error that names a layer cuts the name: a long name in the file must not become a larger
// uncharged message (a control byte is four bytes in `%q` and six in `{:?}`).
#[test]
fn errors_cut_long_names() {
    let name = "\u{10}".repeat(100_000);
    let check = |e: Error| {
        let msg = e.to_string();
        assert!(msg.len() < 600, "{} bytes", msg.len());
        assert!(msg.contains("... (100000 bytes)"), "{msg}");
    };
    check(parse_one(&layer_header(LayerType::Tanh, 1, 0, 2, 2, 6, &name)).unwrap_err());
    let mut d = layer_header(LayerType::Tanh, 2, 0, 2, 2, 99, &name);
    d.extend(float_matrix(132, 2, 3, &VALS));
    check(parse_one(&d).unwrap_err());
    let mut d = layer_header(LayerType::Tanh, 2, 0, 2, 2, 6, &name);
    d.extend(float_matrix(133, 2, 3, &VALS));
    check(parse_one(&d).unwrap_err());
    let mut d = vec![0u8];
    d.extend_from_slice(&(name.len() as u32).to_le_bytes());
    d.extend_from_slice(name.as_bytes());
    check(parse_one(&d).unwrap_err());
    // A child's limit error passes through with no message built for it.
    let mut d = layer_header(LayerType::Tanh, 2, 0, 2, 2, 6, &name);
    d.extend(float_matrix(132, 2, 3, &VALS));
    let tight = Limits {
        max_alloc: name.len() + 8,
        ..lim()
    };
    let mut r = Reader::new(&d);
    let e = parse_layer(&mut r, &Budget::new(&tight), &nc(), 0).unwrap_err();
    assert_eq!(e, Error::AllocLimit);
}

// Invalid UTF-8 is replaced one U+FFFD per maximal ill-formed subpart, as `String::from_utf8_lossy`
// does, which the Go oracle models.
#[test]
fn lossy_string_replaces_each_maximal_subpart() {
    for (bytes, want) in [
        (&[0xFF, 0xFE, b'A'][..], "\u{FFFD}\u{FFFD}A"),
        (&[0xE2, 0x82, b'A'][..], "\u{FFFD}A"),
        (&[0xE2, 0x82][..], "\u{FFFD}"),
        (&[0xF0, 0x9F, 0x98][..], "\u{FFFD}"),
        (&[0xC0, 0x80][..], "\u{FFFD}\u{FFFD}"),
        (&[0xED, 0xA0, 0x80][..], "\u{FFFD}\u{FFFD}\u{FFFD}"),
    ] {
        assert_eq!(lossy_string(bytes, &bud()).unwrap(), want, "{bytes:x?}");
    }
}

#[test]
fn layer_rejects_unsupported_matrix_and_training_dumps() {
    let e = parse_one(&tanh_layer(2, 6, 133)).unwrap_err();
    assert!(is_err_containing(e, "int8"));
    assert!(parse_one(&tanh_layer(2, 6, 4)).is_err(), "legacy format");
    let e = parse_one(&tanh_layer(1, 6, 132)).unwrap_err();
    assert!(is_err_containing(e, "TS_ENABLED"));
    assert!(parse_one(&tanh_layer(2, 99, 132)).is_err(), "num_weights");
    let mut neg = layer_header(LayerType::Tanh, 2, 0, 2, 2, 6, "T");
    neg.push(132);
    neg.extend_from_slice(&(-1i32).to_le_bytes());
    neg.extend_from_slice(&3i32.to_le_bytes());
    assert!(parse_one(&neg).is_err(), "negative dimension");
    let mut big = layer_header(LayerType::Tanh, 2, 0, 2, 2, 6, "T");
    big.extend(float_matrix(132, 65535, 65535, &[]));
    assert!(
        parse_one(&big).is_err(),
        "dimensions larger than the buffer"
    );
}

#[test]
fn network_rejects_absurd_and_deep_stacks() {
    let mut b = vec![LayerType::Series as u8, 0, 0];
    b.extend([0u8; 4]);
    b.extend([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]);
    b.extend([0u8; 4]);
    b.extend([0x9f, 0x86, 0x01, 0x00]);
    assert!(parse_recognizer(&b, false, &bud(), &nc()).is_err());

    // Series nested far deeper than any real network.
    let mut deep = Vec::new();
    for _ in 0..200 {
        deep.extend(layer_header(LayerType::Series, 0, 0, 1, 1, 0, ""));
        deep.extend(1u32.to_le_bytes());
    }
    let e = parse_recognizer(&deep, false, &bud(), &nc()).unwrap_err();
    assert!(is_err_containing(e, "deeper"));
}

fn trailer(spec: &str, flags: i32, train_iter: i32, sample_iter: i32, null_char: i32) -> Vec<u8> {
    let mut b = (spec.len() as u32).to_le_bytes().to_vec();
    b.extend_from_slice(spec.as_bytes());
    for v in [flags, train_iter, sample_iter, null_char] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for v in [0.999f32, 0.001, 0.5] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

fn tiny_recognizer(flags: i32) -> Vec<u8> {
    let mut d = tanh_layer(2, 6, 132);
    d.extend(trailer("[tiny]", flags, 1, 2, 1));
    d
}

#[test]
fn recognizer_trailer_and_rejections() {
    let rec = parse_recognizer(&tiny_recognizer(64), false, &bud(), &nc()).unwrap();
    assert_eq!(rec.network_str, "[tiny]");
    assert_eq!(rec.null_char, 1);
    assert_eq!((rec.training_iteration, rec.sample_iteration), (1, 2));
    assert_eq!(rec.adam_beta, 0.999);
    assert_eq!(rec.learning_rate, 0.001);
    assert_eq!(rec.momentum, 0.5);
    assert!(!rec.is_int_mode());

    let e = parse_recognizer(&tiny_recognizer(65), false, &bud(), &nc()).unwrap_err();
    assert!(is_err_containing(e, "TF_INT_MODE"));
    assert!(parse_recognizer(&tiny_recognizer(0), false, &bud(), &nc()).is_err());
    let mut trailing = tiny_recognizer(64);
    trailing.extend([0xde, 0xad]);
    assert!(parse_recognizer(&trailing, false, &bud(), &nc()).is_err());
    let mut d = tanh_layer(2, 6, 132);
    d.extend(trailer("[tiny]", 64, 1, 2, 2));
    assert!(
        parse_recognizer(&d, false, &bud(), &nc()).is_err(),
        "null_char outside outputs"
    );
}

#[test]
fn matrices_are_charged_against_the_budget() {
    let tight = Limits {
        max_alloc: 100,
        ..lim()
    };
    let d = tiny_recognizer(64);
    assert_eq!(
        parse_recognizer(&d, false, &Budget::new(&tight), &nc()).unwrap_err(),
        Error::AllocLimit
    );
}

#[test]
fn ceil_log2_is_the_lstm_one() {
    assert_eq!(network::ceil_log2_for_test(64), 6);
    assert_eq!(network::ceil_log2_for_test(65), 7);
    assert_eq!(network::ceil_log2_for_test(1), 0);
    assert_eq!(network::ceil_log2_for_test(0), 0);
}

#[test]
fn dawg_accessors_read_an_edge_outside_the_table_as_the_zero_record() {
    let d = synthetic_dawg();
    for e in [NO_EDGE, d.num_edges() as i64, i64::MAX, i64::MIN] {
        assert_eq!(
            (d.unichar_id(e), d.next_node(e), d.end_of_word(e)),
            (0, 0, false),
            "edge {e}"
        );
    }
}

// The first line has no length cap, so an error that quoted it whole would be an uncharged copy
// as large as the input.
#[test]
fn a_long_bad_count_line_is_cut_in_the_error() {
    let line = "\u{1}".repeat(1 << 20);
    let e = unicharset(&line).unwrap_err();
    assert!(e.to_string().len() < 1000, "{} bytes", e.to_string().len());
    let e = unicharset(&format!("1\nA {}\n", "x".repeat(60_000))).unwrap_err();
    assert!(e.to_string().len() < 1000, "{} bytes", e.to_string().len());
}

#[test]
fn unicharset_count_above_what_the_data_can_hold_is_refused_early() {
    let e = parse_unicharset(b"1000000\n", &bud()).unwrap_err();
    assert!(is_err_containing(e, "declares 1000000 entries"));
}

// Each invalid byte becomes U+FFFD, three bytes, so the charge is for the string that is made.
#[test]
fn a_lossy_string_is_charged_for_its_replacement_characters() {
    let mut data = 10u32.to_le_bytes().to_vec();
    data.extend([0xffu8; 10]);
    let cap = |n| {
        Budget::new(&Limits {
            max_alloc: n,
            ..lim()
        })
    };
    assert_eq!(Reader::new(&data).string(&cap(29)), Err(Error::AllocLimit));
    let b = cap(30);
    let s = Reader::new(&data).string(&b).unwrap();
    assert_eq!((s.len(), b.used()), (30, 30));
}

#[test]
fn container_entry_points_take_the_limits() {
    let raw = build_container(&[(Type::Version, b"v1")]);
    let small = Limits {
        max_model_bytes: 3,
        ..lim()
    };
    assert_eq!(
        parse_container(&raw, &small).unwrap_err(),
        Error::ModelLimit
    );
    let none = Limits {
        max_alloc: 1,
        ..lim()
    };
    let c = parse_container(&raw, &lim()).unwrap();
    assert_eq!(c.version(&none), Err(Error::AllocLimit));
}
