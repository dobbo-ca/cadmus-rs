use super::*;
use crate::Limits;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

fn bud() -> Budget {
    Budget::new(&Limits::default())
}

fn sm(height: usize, width: usize) -> StrideMap {
    StrideMap { height, width }
}

fn tensor(h: usize, w: usize, f: usize) -> Tensor {
    Tensor::new(sm(h, w), f, &bud()).unwrap()
}

fn read(t: &Tensor, step: usize) -> Vec<f64> {
    let mut v = vec![0.0; t.features];
    t.read_time_step(step, &mut v);
    v
}

fn run(l: &Layer, input: &Tensor) -> Tensor {
    let b = bud();
    l.forward(input, &mut Ctx::new(&b)).unwrap()
}

#[test]
fn matrix_dot_vector_adds_the_bias_column() {
    let m = Matrix::new(
        2,
        3,
        vec![1.0, 2.0, 3.0, 10.0, 0.0, 0.0, 1.0, -5.0],
        &Limits::default(),
    )
    .unwrap();
    let mut v = vec![0.0; 2];
    m.dot_vector(&[1.0, 1.0, 1.0], &mut v).unwrap();
    assert_eq!(v, [16.0, -4.0]);
}

#[test]
fn new_matrix_rejects_wrong_weight_count() {
    assert!(Matrix::new(2, 3, vec![0.0; 7], &Limits::default()).is_err());
    assert!(Matrix::new(0, 3, vec![], &Limits::default()).is_err());
}

// Rust never fuses a multiply and an add: the fused result would be exactly 0 here.
#[test]
fn matrix_dot_vector_does_not_fuse_multiply_add() {
    let e27 = 2f64.powi(-27);
    let e26 = 2f64.powi(-26);
    let m = Matrix::new(1, 2, vec![1.0, 1.0 + e27, 0.0], &Limits::default()).unwrap();
    let mut v = vec![0.0; 1];
    m.dot_vector(&[-(1.0 + e26), 1.0 + e27], &mut v).unwrap();
    let unfused = (1.0 * -(1.0 + e26)) + ((1.0 + e27) * (1.0 + e27));
    assert_eq!(v[0], unfused);
    let fused = (1.0 + e27).mul_add(1.0 + e27, 1.0 * -(1.0 + e26));
    assert_ne!(fused, unfused);
}

#[test]
fn stride_map_raster_order() {
    let m = sm(3, 4);
    assert_eq!(m.len(), 12);
    assert_eq!(m.t(0, 1), 1);
    assert_eq!(m.t(1, 0), 4);
    for t in 0..m.len() {
        let (y, x) = m.yx(t).unwrap();
        assert_eq!(m.t(y, x), t);
    }
}

#[test]
fn stride_map_offset() {
    let m = sm(3, 4);
    assert_eq!(m.offset(m.t(1, 1), -1, -1), Some(m.t(0, 0)));
    assert_eq!(m.offset(m.t(0, 0), -1, 0), None);
    assert_eq!(m.offset(m.t(0, 3), 0, 1), None);
    assert_eq!(m.offset(m.t(2, 0), 1, 0), None);
}

#[test]
fn stride_map_scale_truncates_and_transposes() {
    assert_eq!(sm(36, 100).scale_xy(3, 3), sm(12, 33));
    let m = sm(12, 33);
    assert_eq!(m.transpose_xy(), sm(33, 12));
    assert_eq!(m.transpose_xy().transpose_xy(), m);
    assert_eq!(m.reduce_width_to_1(), sm(12, 1));
}

#[test]
fn tensor_rounds_through_f32() {
    let mut x = tensor(1, 1, 1);
    let v = 1.0 + 1e-12;
    x.write_time_step(0, &[v]);
    let got = read(&x, 0)[0];
    assert_ne!(got, v);
    assert_eq!(got, f64::from(v as f32));
}

#[test]
fn tensor_write_and_copy_parts() {
    let mut x = tensor(1, 2, 5);
    x.write_time_step(0, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    x.write_time_step_part(0, 2, 2, &[9.0, 8.0]).unwrap();
    assert_eq!(read(&x, 0), [1.0, 2.0, 9.0, 8.0, 5.0]);
    assert_eq!(read(&x, 1), [0.0; 5]);

    let mut src = tensor(1, 1, 3);
    src.write_time_step(0, &[7.0, 8.0, 9.0]);
    let mut dst = tensor(1, 1, 6);
    dst.copy_time_step_part(0, 3, 3, &src, 0, 0).unwrap();
    assert_eq!(read(&dst, 0), [0.0, 0.0, 0.0, 7.0, 8.0, 9.0]);
}

#[test]
fn tensor_maxpool_time_step() {
    let mut a = tensor(1, 1, 3);
    a.write_time_step(0, &[1.0, 5.0, -3.0]);
    let mut b = tensor(1, 1, 3);
    b.write_time_step(0, &[4.0, 2.0, -1.0]);
    a.maxpool_time_step(0, &b, 0);
    assert_eq!(read(&a, 0), [4.0, 5.0, -1.0]);
}

#[test]
fn tensor_is_charged_against_the_budget() {
    let b = Budget::new(&Limits {
        max_alloc: 100,
        ..Limits::default()
    });
    assert_eq!(Tensor::new(sm(4, 4), 2, &b).unwrap_err(), Error::AllocLimit);
    assert!(Tensor::new(sm(2, 2), 2, &b).is_ok());
    assert_eq!(
        Tensor::new(sm(usize::MAX, 2), 2, &b).unwrap_err(),
        Error::AllocLimit
    );
}

#[test]
fn tables_are_complete() {
    use super::tables::{LOGISTIC_TABLE, TANH_TABLE};
    assert_eq!(TANH_TABLE[0], 0.0);
    assert_eq!(LOGISTIC_TABLE[0], 0.5);
    for i in 1..4096 {
        assert!(TANH_TABLE[i] > TANH_TABLE[i - 1], "tanh at {i}");
        assert!(LOGISTIC_TABLE[i] > LOGISTIC_TABLE[i - 1], "logistic at {i}");
    }
}

#[test]
fn tanh_hits_table_stops_exactly() {
    use super::tables::{LOGISTIC_TABLE, TANH_TABLE};
    for i in [1usize, 7, 255, 1000, 4094] {
        let x = i as f64 / 256.0;
        assert_eq!(tanh(x), TANH_TABLE[i]);
        assert_eq!(logistic(x), LOGISTIC_TABLE[i]);
    }
}

#[test]
fn activation_symmetry_and_saturation() {
    assert_eq!(tanh(0.0), 0.0);
    assert_eq!(logistic(0.0), 0.5);
    let sat = 4095.0 / 256.0;
    assert_eq!(tanh(sat), 1.0);
    assert_eq!(tanh(-sat), -1.0);
    assert_eq!(logistic(sat), 1.0);
    assert_eq!(logistic(-sat), 0.0);
    assert_eq!(logistic(-1.5), 1.0 - logistic(1.5));
}

#[test]
fn tanh_is_the_interpolated_table_not_libm() {
    let mut max_diff = 0f64;
    for i in 0..200000 {
        let x = f64::from(i) * 8.0 / 200000.0;
        max_diff = max_diff.max((tanh(x) - x.tanh()).abs());
    }
    assert!(
        max_diff >= 1e-7,
        "max diff {max_diff}: tables look regenerated"
    );
    assert!(
        max_diff <= 2e-6,
        "max diff {max_diff}: beyond the interpolation bound"
    );
}

#[test]
fn softmax_in_place_normalises() {
    let mut v = [1.0, 2.0, 3.0];
    softmax_in_place(&mut v);
    assert!((v.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(v[0] < v[1] && v[1] < v[2]);
    let mut w = [1001.0, 1002.0, 1003.0];
    softmax_in_place(&mut w);
    for i in 0..3 {
        assert!((v[i] - w[i]).abs() < 1e-15, "{v:?} vs {w:?}");
    }
    let mut z = [0.0, -1000.0];
    softmax_in_place(&mut z);
    assert!(z[1] > 0.0, "the -86 clip is missing");
    let mut e: [f64; 0] = [];
    softmax_in_place(&mut e);
}

// The amd64 FMA exp stays within 1 ulp of libm over the softmax domain.
#[test]
fn softmax_exp_is_close_to_libm() {
    for i in 0..=8600 {
        let x = -f64::from(i) / 100.0;
        let mut v = [x, 0.0];
        softmax_in_place(&mut v);
        let e = x.exp();
        let want = e / (e + 1.0);
        assert!(
            (v[0] - want).abs() <= want * 1e-14,
            "x={x}: {} vs {want}",
            v[0]
        );
    }
}

#[test]
fn maxpool_takes_the_block_maximum() {
    let mut input = tensor(3, 6, 2);
    let n = input.map.len();
    for t in 0..n {
        input.write_time_step(t, &[t as f64, (n - t) as f64]);
    }
    let mp = Layer::Reconfig(Reconfig::new_maxpool("Maxpool", 2, 3, 3));
    assert_eq!(mp.num_outputs(), 2);
    let out = run(&mp, &input);
    assert_eq!(out.map, sm(1, 2));
    let got = read(&out, 0);
    assert_eq!(got[0], 14.0);
    assert_eq!(got[1], n as f64);
}

#[test]
fn maxpool_drops_partial_windows() {
    let out = run(
        &Layer::Reconfig(Reconfig::new_maxpool("Maxpool", 1, 3, 3)),
        &tensor(36, 100, 1),
    );
    assert_eq!(out.map, sm(12, 33));
}

#[test]
fn reconfig_stacks_the_block() {
    let mut input = tensor(2, 2, 1);
    for t in 0..4 {
        input.write_time_step(t, &[t as f64 + 1.0]);
    }
    let rc = Layer::Reconfig(Reconfig::new("Reconfig", 1, 2, 2));
    assert_eq!(rc.num_outputs(), 4);
    assert_eq!(read(&run(&rc, &input), 0), [1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn reconfig_rejects_bad_input() {
    let b = bud();
    let rc = Layer::Reconfig(Reconfig::new("Reconfig", 1, 2, 2));
    assert!(rc.forward(&tensor(2, 2, 3), &mut Ctx::new(&b)).is_err());
    let zero = Layer::Reconfig(Reconfig::new("Reconfig", 1, 0, 2));
    assert!(zero.forward(&tensor(2, 2, 1), &mut Ctx::new(&b)).is_err());
}

fn linear(name: &str, ni: usize, no: usize, act: Activation, rows: Vec<f64>) -> Layer {
    Layer::FullyConnected(FullyConnected::new(
        name,
        act,
        Matrix::new(no, ni, rows, &Limits::default()).unwrap(),
    ))
}

#[test]
fn fully_connected_applies_matrix_then_activation() {
    let fc = linear("ConvNL", 1, 1, Activation::Tanh, vec![2.0, 0.0]);
    assert_eq!(fc.num_outputs(), 1);
    let mut input = tensor(1, 2, 1);
    input.write_time_step(0, &[0.25]);
    input.write_time_step(1, &[-0.25]);
    let out = run(&fc, &input);
    assert_eq!(read(&out, 0)[0], f64::from(tanh(0.5) as f32));
    assert_eq!(read(&out, 1)[0], f64::from(tanh(-0.5) as f32));
    assert_eq!(out.map, input.map);
}

#[test]
fn fully_connected_softmax_normalises_per_timestep() {
    let fc = linear(
        "Output",
        1,
        3,
        Activation::Softmax,
        vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0],
    );
    let mut input = tensor(1, 1, 1);
    input.write_time_step(0, &[1.0]);
    let sum: f64 = read(&run(&fc, &input), 0).iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn fully_connected_clips_and_relu() {
    let mut input = tensor(1, 3, 1);
    for (t, v) in [-2.0, 0.5, 3.0].iter().enumerate() {
        input.write_time_step(t, &[*v]);
    }
    let w = vec![1.0, 0.0];
    let relu = run(&linear("r", 1, 1, Activation::Relu, w.clone()), &input);
    assert_eq!(
        [read(&relu, 0)[0], read(&relu, 1)[0], read(&relu, 2)[0]],
        [0.0, 0.5, 3.0]
    );
    let pos = run(&linear("p", 1, 1, Activation::PosClip, w.clone()), &input);
    assert_eq!([read(&pos, 0)[0], read(&pos, 2)[0]], [0.0, 1.0]);
    let sym = run(&linear("s", 1, 1, Activation::SymClip, w), &input);
    assert_eq!([read(&sym, 0)[0], read(&sym, 2)[0]], [-1.0, 1.0]);
}

#[test]
fn a_nan_fails_the_forward_pass_only_at_a_tabulated_activation() {
    // Go indexes its tanh and logistic tables with int(NaN) and panics; softmax gives output.
    let mut input = tensor(1, 1, 1);
    input.write_time_step(0, &[1.0]);
    let b = bud();
    for (act, ok) in [
        (Activation::Tanh, false),
        (Activation::Logistic, false),
        (Activation::Softmax, true),
        (Activation::Linear, true),
    ] {
        let fc = linear("n", 1, 1, act, vec![f64::NAN, 0.0]);
        let r = fc.forward(&input, &mut Ctx::new(&b));
        assert_eq!(r.is_ok(), ok, "{act:?}");
        if !ok {
            assert!(matches!(r, Err(Error::Network(_))));
        }
    }
}

#[test]
fn fully_connected_is_the_same_for_every_thread_count() {
    let (ni, no, steps) = (64, 64, 100);
    let rows: Vec<f64> = (0..no * (ni + 1))
        .map(|i| ((i * 7919 % 1000) as f64 - 500.0) / 1000.0)
        .collect();
    let fc = linear("fc", ni, no, Activation::Tanh, rows);
    let mut input = tensor(1, steps, ni);
    for t in 0..steps {
        let u: Vec<f64> = (0..ni)
            .map(|j| ((t * 31 + j * 17) % 101) as f64 / 50.0 - 1.0)
            .collect();
        input.write_time_step(t, &u);
    }
    let budget_of = |max_threads: usize| {
        Budget::new(&Limits {
            max_threads,
            ..Limits::default()
        })
    };
    let serial = budget_of(1);
    let want = fc.forward(&input, &mut Ctx::new(&serial)).unwrap();
    assert_eq!(serial.peak_threads(), 1);
    for n in [1, 2, 3, Limits::default().max_threads] {
        let b = budget_of(n);
        let got = fc.forward(&input, &mut Ctx::new(&b)).unwrap();
        assert_eq!(got.data, want.data, "{n} threads");
        assert!(b.peak_threads() <= n, "{n} threads");
        if n >= 2 {
            assert!(b.peak_threads() >= 2, "{n} threads were not used");
        }
        // The cancel flag stops every thread, and they all go back to the budget.
        let stop = AtomicBool::new(true);
        let b = budget_of(n);
        let mut cx = Ctx::new(&b);
        cx.cancel = Some(&stop);
        assert_eq!(fc.forward(&input, &mut cx).err(), Some(Error::Cancelled));
        assert_eq!(b.acquire_threads(100), n - 1);
    }
}

// The band scratch is cut to the budget: the smallest `max_alloc` that one thread works with
// works with any `max_threads` (on one band), and with room for the extra bands and threads
// they are used.
#[test]
fn fully_connected_with_a_budget_for_one_band_works_at_any_max_threads() {
    let (ni, no, steps) = (64, 64, 100);
    let rows: Vec<f64> = (0..no * (ni + 1))
        .map(|i| ((i * 7919 % 1000) as f64 - 500.0) / 1000.0)
        .collect();
    let fc = linear("fc", ni, no, Activation::Tanh, rows);
    let mut input = tensor(1, steps, ni);
    for t in 0..steps {
        let u: Vec<f64> = (0..ni)
            .map(|j| ((t * 31 + j * 17) % 101) as f64 / 50.0 - 1.0)
            .collect();
        input.write_time_step(t, &u);
    }
    // A pool with room for every `max_threads` below, whatever the CPUs of the machine.
    let pool = crate::ThreadPool::new(4).unwrap();
    let run = |max_alloc: usize, max_threads: usize| {
        let b = Budget::new(&Limits {
            max_alloc,
            max_threads,
            pool: Some(pool.clone()),
            ..Limits::default()
        });
        let r = fc.forward(&input, &mut Ctx::new(&b)).map(|t| t.data);
        (r, b.peak_threads())
    };
    let (mut lo, mut hi) = (0usize, 16usize << 20);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        match run(mid, 1).0 {
            Ok(_) => hi = mid,
            Err(Error::AllocLimit) => lo = mid,
            Err(e) => panic!("max_alloc {mid} gave {e}"),
        }
    }
    let want = run(hi, 1).0.unwrap();
    for n in [2usize, 4] {
        assert_eq!(
            run(hi, n),
            (Ok(want.clone()), 1),
            "max_alloc {hi}, {n} threads"
        );
        // The scratch of a band is (ni + no) * 8 + 56 bytes, a thread 4096. One band holds no band
        // list, so `hi` has no 56 bytes for the first band's entry: add them.
        let (r, threads) = run(hi + (n - 1) * (8 * (ni + no) + 56 + 4096) + 56, n);
        assert_eq!((r, threads), (Ok(want.clone()), n));
    }
}

#[test]
fn fully_connected_rejects_wrong_input_width() {
    let fc = linear("x", 3, 1, Activation::Linear, vec![1.0, 1.0, 1.0, 0.0]);
    let b = bud();
    assert!(fc.forward(&tensor(1, 1, 2), &mut Ctx::new(&b)).is_err());
}

#[test]
fn rand_matches_the_tesseract_lcg() {
    const MUL: u64 = 6364136223846793005;
    const INC: u64 = 1442695040888963407;
    let first = 5u64.wrapping_mul(MUL).wrapping_add(INC);
    let second = first.wrapping_mul(MUL).wrapping_add(INC);
    let r = Rand::new(5);
    assert_eq!(r.int_rand(), (second >> 33) as i32);
    assert!(r.int_rand() >= 0);
}

#[test]
fn signed_rand_range() {
    let r = Rand::new(814136 * 0x10000001);
    for _ in 0..1000 {
        let v = r.signed_rand(1.0);
        assert!((-1.0..=1.0).contains(&v));
    }
}

#[test]
fn convolve_gathers_and_randomizes_edges() {
    let mut input = tensor(1, 1, 1);
    input.write_time_step(0, &[0.5]);
    let c = Layer::Convolve(Convolve::new("Convolve", 1, 1, 1, Arc::new(Rand::new(1))));
    assert_eq!(c.num_outputs(), 9);
    let out = run(&c, &input);
    assert_eq!(out.map, input.map);
    let got = read(&out, 0);
    assert_eq!(got[4], 0.5);
    for (i, v) in got.iter().enumerate() {
        if i != 4 {
            assert_ne!(*v, 0.0, "feature {i}: off-image taps must be randomized");
        }
    }
}

#[test]
fn convolve_feature_layout() {
    let mut input = tensor(3, 3, 1);
    for t in 0..9 {
        input.write_time_step(t, &[t as f64 + 1.0]);
    }
    let c = Layer::Convolve(Convolve::new("Convolve", 1, 1, 1, Arc::new(Rand::new(1))));
    let out = run(&c, &input);
    let got = read(&out, input.map.t(1, 1));
    for dx in -1i32..=1 {
        for dy in -1i32..=1 {
            let f = ((dx + 1) * 3 + (dy + 1)) as usize;
            let want = input.map.t((1 + dy) as usize, (1 + dx) as usize) as f64 + 1.0;
            assert_eq!(got[f], want, "tap dx={dx} dy={dy}");
        }
    }
}

fn stamp(id: f64, ni: usize, features: usize) -> Layer {
    let mut w = Vec::new();
    for _ in 0..features {
        w.extend(vec![0.0; ni]);
        w.push(id);
    }
    linear("stamp", ni, features, Activation::Linear, w)
}

#[test]
fn input_is_identity() {
    let mut input = tensor(2, 2, 1);
    input.write_time_step(3, &[7.0]);
    let out = run(&Layer::Input(Input::new("Input", 1)), &input);
    assert_eq!(read(&out, 3)[0], 7.0);
}

#[test]
fn series_runs_in_order() {
    let s = Layer::Series(Series::new("Series", vec![stamp(1.0, 1, 2), stamp(2.0, 2, 3)]).unwrap());
    let out = run(&s, &tensor(1, 1, 1));
    assert_eq!(out.features, 3);
    assert_eq!(read(&out, 0), [2.0; 3]);
    assert!(Series::new("empty", vec![]).is_err());
}

#[test]
fn parallel_concatenates_features() {
    let p = Layer::Parallel(
        Parallel::new("Parallel", vec![stamp(1.0, 1, 2), stamp(2.0, 1, 3)]).unwrap(),
    );
    assert_eq!(p.num_outputs(), 5);
    assert_eq!(
        read(&run(&p, &tensor(1, 1, 1)), 0),
        [1.0, 1.0, 2.0, 2.0, 2.0]
    );
    assert!(Parallel::new("empty", vec![]).is_err());
}

#[test]
fn reversed_is_a_no_op_around_identity() {
    for kind in [
        ReversalKind::ReverseX,
        ReversalKind::ReverseY,
        ReversalKind::TransposeXY,
    ] {
        let mut input = tensor(3, 4, 1);
        for t in 0..12 {
            input.write_time_step(t, &[t as f64]);
        }
        let l = Layer::Reversed(Reversed::new(
            "rev",
            kind,
            Layer::Input(Input::new("id", 1)),
        ));
        let out = run(&l, &input);
        assert_eq!(out.map, input.map, "{kind:?}");
        for t in 0..12 {
            assert_eq!(read(&out, t)[0], t as f64, "{kind:?} t={t}");
        }
    }
}

#[test]
fn reverse_data_transforms() {
    let mut src = tensor(2, 3, 1);
    for t in 0..6 {
        src.write_time_step(t, &[t as f64]);
    }
    let b = bud();
    let x = super::layers::reverse_data_for_test(&src, ReversalKind::ReverseX, &b).unwrap();
    assert_eq!(read(&x, x.map.t(0, 0))[0], 2.0);
    let tr = super::layers::reverse_data_for_test(&src, ReversalKind::TransposeXY, &b).unwrap();
    assert_eq!(tr.map, sm(3, 2));
    assert_eq!(read(&tr, tr.map.t(2, 1))[0], src.map.t(1, 2) as f64);
}

#[test]
fn tap_sees_every_layer_but_input() {
    let s = Layer::Series(
        Series::new(
            "Series",
            vec![Layer::Input(Input::new("in", 1)), stamp(1.0, 1, 2)],
        )
        .unwrap(),
    );
    let mut seen = Vec::new();
    let b = bud();
    let mut f = |l: &Layer, _: &Tensor| seen.push(l.name().to_string());
    let mut cx = Ctx {
        budget: &b,
        tap: Some(&mut f),
        cancel: None,
        pool: None,
    };
    s.forward(&tensor(1, 1, 1), &mut cx).unwrap();
    assert_eq!(seen, ["stamp", "Series"]);
}

fn gates(input: [f64; 4], rec: [f64; 4], bias: [f64; 4]) -> [Matrix; 4] {
    std::array::from_fn(|g| {
        Matrix::new(1, 2, vec![input[g], rec[g], bias[g]], &Limits::default()).unwrap()
    })
}

fn gates_na(na: usize) -> [Matrix; 4] {
    std::array::from_fn(|_| Matrix::new(1, na, vec![0.0; na + 1], &Limits::default()).unwrap())
}

fn run_lstm(l: &Lstm, input: &Tensor) -> (Tensor, Vec<f64>) {
    let b = bud();
    l.forward_state(input, &mut Ctx::new(&b)).unwrap()
}

#[test]
fn lstm_cell_matches_the_equations() {
    let g = gates([1.0; 4], [0.0; 4], [0.0; 4]);
    let l = Lstm::new("L", 1, 2, false, g).unwrap();
    let mut input = tensor(1, 2, 1);
    input.write_time_step(0, &[0.5]);
    input.write_time_step(1, &[0.25]);
    let (out, _) = run_lstm(&l, &input);
    let mut state = 0.0;
    for (t, x) in [0.5, 0.25].into_iter().enumerate() {
        let (ci, gi, gf, go) = (tanh(x), logistic(x), logistic(x), logistic(x));
        state = state * gf + ci * gi;
        let want = f64::from((tanh(state) * go) as f32);
        assert!((read(&out, t)[0] - want).abs() < 1e-9, "t={t}");
    }
}

#[test]
fn lstm_clips_the_state_every_timestep() {
    let g = gates([100.0; 4], [0.0; 4], [0.0; 4]);
    let l = Lstm::new("L", 1, 2, false, g).unwrap();
    let mut input = tensor(1, 400, 1);
    for t in 0..400 {
        input.write_time_step(t, &[1.0]);
    }
    let (out, last) = run_lstm(&l, &input);
    assert_eq!(read(&out, 399)[0], 1.0);
    assert_eq!(last[0], 100.0);
}

#[test]
fn lstm_resets_state_at_end_of_row() {
    let g = gates([1.0; 4], [0.5; 4], [0.0; 4]);
    let l = Lstm::new("L", 1, 2, false, g).unwrap();
    let mut input = tensor(2, 3, 1);
    for y in 0..2 {
        for x in 0..3 {
            input.write_time_step(input.map.t(y, x), &[x as f64 * 0.3]);
        }
    }
    let (out, _) = run_lstm(&l, &input);
    for x in 0..3 {
        assert_eq!(
            read(&out, out.map.t(0, x)),
            read(&out, out.map.t(1, x)),
            "x={x}"
        );
    }
}

#[test]
fn lstm_summary_emits_only_the_last_step() {
    let plain = Lstm::new("L", 1, 2, false, gates([1.0; 4], [0.0; 4], [0.0; 4])).unwrap();
    let summ = Lstm::new("Lfys", 1, 2, true, gates([1.0; 4], [0.0; 4], [0.0; 4])).unwrap();
    let mut input = tensor(2, 4, 1);
    for t in 0..8 {
        input.write_time_step(t, &[0.1 * ((t % 4) + 1) as f64]);
    }
    let (full, _) = run_lstm(&plain, &input);
    let (got, _) = run_lstm(&summ, &input);
    assert_eq!(got.map, sm(2, 1));
    for y in 0..2 {
        assert_eq!(
            read(&full, full.map.t(y, 3)),
            read(&got, got.map.t(y, 0)),
            "row {y}"
        );
    }
}

#[test]
fn new_lstm_rejects_unsupported_shapes() {
    let e = Lstm::new("L", 1, 3, false, gates_na(3)).unwrap_err();
    assert!(e.to_string().contains("2-D"), "{e}");
    assert!(Lstm::new("L", 1, 5, false, gates_na(5)).is_err());
    assert!(Lstm::new("L", 1, 2, false, gates_na(3)).is_err());
}

#[test]
fn cancel_stops_a_forward_pass() {
    let l = linear("x", 1, 1, Activation::Linear, vec![1.0, 0.0]);
    let flag = std::sync::atomic::AtomicBool::new(true);
    let b = bud();
    let mut cx = Ctx::new(&b);
    cx.cancel = Some(&flag);
    assert_eq!(
        l.forward(&tensor(1, 1, 1), &mut cx).unwrap_err(),
        Error::Cancelled
    );
}

#[test]
fn parallel_rejects_a_child_whose_features_differ_from_its_declared_outputs() {
    // An Input child returns its input, whatever feature count its header declared.
    let p = |declared| {
        Layer::Parallel(Parallel::new("p", vec![Layer::Input(Input::new("z", declared))]).unwrap())
    };
    let input = tensor(1, 3, 1);
    let b = bud();
    assert!(p(1).forward(&input, &mut Ctx::new(&b)).is_ok());
    for declared in [0, 2] {
        let b = bud();
        let e = p(declared).forward(&input, &mut Ctx::new(&b)).unwrap_err();
        assert!(matches!(e, Error::Network(_)), "{e:?}");
    }
}

#[test]
fn tensor_is_consistent_until_a_public_field_is_changed() {
    let mut t = tensor(2, 3, 4);
    assert!(t.is_consistent());
    t.features = 5;
    assert!(!t.is_consistent());
    t.features = 4;
    t.map.width = usize::MAX;
    assert!(!t.is_consistent());
}

#[test]
fn forward_rejects_a_tensor_or_matrix_that_disagrees_with_its_data() {
    let fc = linear("f", 1, 1, Activation::Linear, vec![1.0, 0.0]);
    let mut t = tensor(1, 1, 1);
    t.map.width = 2;
    let b = bud();
    assert!(fc.forward(&t, &mut Ctx::new(&b)).is_err());

    let mut m = Matrix::new(1, 1, vec![1.0, 0.0], &Limits::default()).unwrap();
    m.inputs = 3;
    assert!(!m.is_consistent());
    let fc = Layer::FullyConnected(FullyConnected::new("f", Activation::Linear, m));
    let t = tensor(1, 1, 3);
    assert!(fc.forward(&t, &mut Ctx::new(&b)).is_err());
}

// A Parallel whose children's sizes do not fit used to panic (overflow checks on) in the sum.
#[test]
fn parallel_sizes_saturate_instead_of_overflowing() {
    let big = usize::MAX / 2 + 1;
    let par = Layer::Parallel(
        Parallel::new(
            "p",
            vec![
                Layer::Reconfig(Reconfig::new("r1", big, 1, 1)),
                Layer::Reconfig(Reconfig::new("r2", big, 1, 1)),
            ],
        )
        .unwrap(),
    );
    assert_eq!(par.num_outputs(), usize::MAX);
    let b = bud();
    assert!(par.forward(&tensor(1, 1, 1), &mut Ctx::new(&b)).is_err());
}

#[test]
fn public_kernels_return_errors_on_bad_arguments() {
    let lim = Limits::default();
    let m = Matrix::new(1, 5, vec![0.0; 6], &lim).unwrap();
    assert!(m.dot_vector(&[0.0], &mut [0.0]).is_err());
    let mut lied = m.clone();
    lied.outputs = 9;
    assert!(lied.dot_vector(&[0.0; 5], &mut [0.0; 9]).is_err());
    let small = Limits {
        max_alloc: 47,
        ..lim.clone()
    };
    assert_eq!(
        Matrix::new(1, 5, vec![0.0; 6], &small).unwrap_err(),
        Error::AllocLimit
    );

    let zero = sm(1, 0);
    assert_eq!(zero.yx(0), None);
    assert_eq!(zero.offset(0, 0, 0), None);
    assert_eq!(sm(3, 4).scale_xy(0, 0), sm(0, 0));
    assert_eq!(sm(usize::MAX, 2).len(), usize::MAX);

    let mut t = tensor(2, 2, 3);
    assert!(t.row(99).is_empty());
    assert!(t.row_mut(usize::MAX).is_empty());
    let src = tensor(1, 1, 3);
    assert!(t.copy_time_step(99, &src, 0).is_err());
    assert!(t.copy_time_step(0, &src, 7).is_err());
    assert!(t.copy_time_step_part(0, 2, 5, &src, 0, 0).is_err());
    assert!(t.copy_time_step_part(0, usize::MAX, 2, &src, 0, 0).is_err());
    assert!(t.write_time_step_part(0, 1, 3, &[0.0; 3]).is_err());
    assert!(t.write_time_step_part(0, 0, 3, &[0.0; 2]).is_err());
    assert!(t.write_time_step_part(0, 0, 3, &[1.0, 2.0, 3.0]).is_ok());
    assert_eq!(t.row(0), [1.0, 2.0, 3.0]);
}

/// The `model_mutate` OOM: an edited input height of 3457 made a 1,493,424-step softmax output,
/// 663,080,256 bytes. It is charged, so a `max_alloc` below it refuses before the allocation.
#[test]
fn a_softmax_output_over_max_alloc_is_refused_before_it_is_allocated() {
    let lim = Limits {
        max_alloc: 256 << 20,
        ..Limits::default()
    };
    let b = Budget::new(&lim);
    let m = Matrix::new(111, 16, vec![0.0; 111 * 17], &lim).unwrap();
    let fc = Layer::FullyConnected(FullyConnected::new("out", Activation::Softmax, m));
    let input = Tensor::new(sm(1, 432 * 3457), 16, &b).unwrap();
    let e = fc.forward(&input, &mut Ctx::new(&b)).unwrap_err();
    assert_eq!(e, Error::AllocLimit);
}
