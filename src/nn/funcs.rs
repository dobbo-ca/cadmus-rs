//! `src/lstm/functions.h`: the tabulated gate activations and the softmax, plus the Go
//! `math.Exp` and `math.Log` the recognizer's certainty computations use.
//!
//! Translated from Tesseract OCR (Apache-2.0, see NOTICE), through the Go port. The translation
//! is not verbatim.
#![allow(clippy::excessive_precision)]

use super::tables::{LOGISTIC_TABLE, TANH_TABLE};

/// `kTableSize` and `kScaleFactor`. Table index `i` is `x = i/256`; indices 0..4094 are usable
/// because the interpolation reads `index+1`.
const TABLE_SIZE: usize = 4096;
const TABLE_SCALE: f64 = 256.0;

/// `kMaxSoftmaxActivation`: the limit on the negative range of the exp input, which
/// guarantees a non-zero probability.
const MAX_SOFTMAX_ACTIVATION: f64 = 86.0;

/// Tesseract's tabulated hyperbolic tangent: linear interpolation between 4096 samples at 1/256
/// spacing, mirrored for negative inputs and saturating to exactly 1 at `|x| >= 4095/256`.
///
/// A NaN gives NaN. Go indexes its table with `int(NaN)` there, a negative index, and panics; the
/// recognizer never lets a NaN reach this (a layer returns `Error::Network` first).
pub fn tanh(x: f64) -> f64 {
    if x < 0.0 {
        return -tanh(-x);
    }
    let x = x * TABLE_SCALE;
    if x >= (TABLE_SIZE - 1) as f64 {
        return 1.0;
    }
    let i = x as usize;
    let (t0, t1) = (TANH_TABLE[i], TANH_TABLE[i + 1]);
    t0 + (t1 - t0) * (x - i as f64)
}

/// Tesseract's tabulated logistic sigmoid. The negative branch is the complement,
/// `1 - Logistic(-x)`, so it returns exactly 0 at `x <= -4095/256`.
///
/// A NaN gives NaN, where Go panics (see [`tanh`]).
pub fn logistic(x: f64) -> f64 {
    if x < 0.0 {
        return 1.0 - logistic(-x);
    }
    let x = x * TABLE_SCALE;
    if x >= (TABLE_SIZE - 1) as f64 {
        return 1.0;
    }
    let i = x as usize;
    let (l0, l1) = (LOGISTIC_TABLE[i], LOGISTIC_TABLE[i + 1]);
    l0 + (l1 - l0) * (x - i as f64)
}

/// `FuncInplace<GFunc>`. A NaN element stays NaN (see [`tanh`]).
pub fn tanh_in_place(v: &mut [f64]) {
    for x in v {
        *x = tanh(*x);
    }
}

/// `FuncInplace<FFunc>`. A NaN element stays NaN (see [`logistic`]).
pub fn logistic_in_place(v: &mut [f64]) {
    for x in v {
        *x = logistic(*x);
    }
}

/// `SoftmaxInPlace`: not tabulated; it subtracts the maximum, clips to `[-86, 0]` and calls
/// exp directly, with Go's `math.Exp` algorithm (below) so the result equals the Go library's.
pub fn softmax_in_place(v: &mut [f64]) {
    if v.is_empty() {
        return;
    }
    let mut max_out = v[0];
    for &x in &v[1..] {
        if x > max_out {
            max_out = x;
        }
    }
    let mut total = 0.0f64;
    for x in v.iter_mut() {
        let p = (*x - max_out).clamp(-MAX_SOFTMAX_ACTIVATION, 0.0);
        let e = go_exp(p);
        total += e;
        *x = e;
    }
    if total > 0.0 {
        for x in v.iter_mut() {
            *x /= total;
        }
    }
}

// go_exp is Go's amd64 `math.Exp` (src/math/exp_amd64.s, Copyright 2010 The Go Authors,
// BSD-3-Clause; the method is Naoki Shibata's, from SLEEF, public domain), the path it takes on
// a CPU with FMA, written with `mul_add` for each fused instruction so the bits are the same.
// Go's own `math.Exp` differs between amd64 without FMA, arm64 and the portable code in the last
// bit; this is the amd64 FMA result, the one the parity job's Go oracle produces. See NOTICE.
// Its argument here is in [-86, 0], so the overflow case is not needed.
pub(crate) fn go_exp(x: f64) -> f64 {
    const LOG2E: f64 = std::f64::consts::LOG2_E;
    const LN2U: f64 = 0.693147180559663;
    const LN2L: f64 = 2.8235290563031577e-13;

    if x.is_nan() {
        return x;
    }
    // CVTSD2SL rounds to nearest even.
    let k = (LOG2E * x).round_ties_even() as i32;
    let kf = f64::from(k);
    let mut x0 = (-LN2U).mul_add(kf, x);
    x0 = (-LN2L).mul_add(kf, x0);
    x0 *= 0.0625;
    let mut x1 = 2.48015873015873e-05f64;
    for c in [
        0.0001984126984126984,
        0.001388888888888889,
        0.008333333333333333,
        0.041666666666666664,
        0.16666666666666666,
        0.5,
        1.0,
    ] {
        x1 = x1.mul_add(x0, c);
    }
    x0 *= x1;
    for _ in 0..3 {
        x1 = x0 + 2.0;
        x0 *= x1;
    }
    x1 = x0 + 2.0;
    x0 = x0.mul_add(x1, 1.0);
    let bx = k + 0x3ff;
    if bx <= 0 || bx >= 0x7ff {
        // Not reachable for x in [-86, 0]; keep the result finite and non-negative.
        return 0.0;
    }
    x0 * f64::from_bits((bx as u64) << 52)
}

/// Go's `math.Log` for a normal positive `x` (src/math/log_amd64.s, Copyright 2010 The Go
/// Authors, BSD-3-Clause; the FreeBSD e_log.c algorithm, Copyright (C) 1993 Sun Microsystems),
/// which has the operations of the portable `log` in the same order and no fused operations. See
/// NOTICE. The special cases are those of Go: NaN and +Inf return `x`, a negative argument NaN,
/// zero -Inf; a subnormal argument is normalized as the portable code does.
pub(crate) fn go_log(x: f64) -> f64 {
    const LN2_HI: f64 = 6.93147180369123816490e-01;
    const LN2_LO: f64 = 1.90821492927058770002e-10;
    const L1: f64 = 6.666666666666735130e-01;
    const L2: f64 = 3.999999999940941908e-01;
    const L3: f64 = 2.857142874366239149e-01;
    const L4: f64 = 2.222219843214978396e-01;
    const L5: f64 = 1.818357216161805012e-01;
    const L6: f64 = 1.531383769920937332e-01;
    const L7: f64 = 1.479819860511658591e-01;

    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    // frexp: x == f1 * 2**ki with f1 in [0.5, 1).
    let (x, extra) = if x < f64::MIN_POSITIVE {
        (x * (1u64 << 52) as f64, -52)
    } else {
        (x, 0)
    };
    let bits = x.to_bits();
    let mut ki = ((bits >> 52) & 0x7ff) as i64 - 0x3fe + extra;
    let mut f1 = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3fe0_0000_0000_0000);
    if f1 < std::f64::consts::FRAC_1_SQRT_2 {
        f1 *= 2.0;
        ki -= 1;
    }
    let f = f1 - 1.0;
    let k = ki as f64;

    let s = f / (2.0 + f);
    let s2 = s * s;
    let s4 = s2 * s2;
    let t1 = s2 * (L1 + s4 * (L3 + s4 * (L5 + s4 * L7)));
    let t2 = s4 * (L2 + s4 * (L4 + s4 * L6));
    let r = t1 + t2;
    let hfsq = 0.5 * f * f;
    k * LN2_HI - ((hfsq - (s * (hfsq + r) + k * LN2_LO)) - f)
}
