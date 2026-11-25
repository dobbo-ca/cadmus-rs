//! Certainty and rating of decoded symbols (`RecodeBeamSearch` and `LSTMRecognizer` helpers).
//!
//! A translation of `NetworkIO::ProbToCertainty` in `src/lstm/networkio.cpp` and of
//! `RecodeBeamSearch::ExtractPathAsUnicharIds` and `::calculateCharBoundaries` in
//! `src/lstm/recodebeam.cpp`.
//!
//! Translated from Tesseract OCR (Apache-2.0, see NOTICE), through the Go port. The translation
//! is not verbatim.

use super::{Symbol, argmax};
use crate::nn::{Tensor, go_exp, go_log};
use crate::pool::grow;
use crate::{Budget, Error, Limits};
use std::sync::OnceLock;

/// `kMinCertainty`, the certainty of a zero-probability symbol.
pub const MIN_CERTAINTY: f64 = -20.0;
/// `kCertOffset`, added to every log-probability.
pub const CERT_OFFSET: f64 = -0.085;
/// `kDictRatio`: the certainty of a non-dictionary path is scaled by it.
pub const DICT_RATIO: f64 = 2.25;
/// `kCertaintyScale`: certainty to confidence.
pub const CERTAINTY_SCALE: f64 = 7.0;

/// `exp(MIN_CERTAINTY)`, computed once with Go's `math.Exp`.
fn min_prob() -> f64 {
    static MIN_PROB: OnceLock<f64> = OnceLock::new();
    *MIN_PROB.get_or_init(|| go_exp(MIN_CERTAINTY))
}

/// `ProbToCertainty`: `ln(p)`, floored at `MIN_CERTAINTY`.
pub fn prob_to_certainty(p: f64) -> f64 {
    if p > min_prob() {
        go_log(p)
    } else {
        MIN_CERTAINTY
    }
}

/// A decoded symbol with the certainty (the worst timestep) and rating (the summed cost) of the
/// timesteps it spans.
#[derive(Debug, Clone, PartialEq)]
pub struct Scored {
    pub symbol: Symbol,
    pub certainty: f64,
    pub rating: f64,
}

/// Scores greedy-decoded symbols against the network output, as `LSTMRecognizer::LSTMRecognizeWord`
/// does for the timesteps each symbol owns, scaled by `dict_ratio`. A tensor whose public fields
/// disagree with its data scores nothing, and a symbol that ends past the tensor is clamped to it.
///
/// The scored symbols and their text are charged to `limits.max_alloc`.
///
/// # Errors
///
/// `Error::AllocLimit` if the result does not fit, `Error::OutOfMemory` if the allocator refuses.
pub fn score_symbols(
    out: &Tensor,
    syms: &[Symbol],
    dict_ratio: f64,
    limits: &Limits,
) -> Result<Vec<Scored>, Error> {
    score_symbols_in(out, syms, dict_ratio, &Budget::new(limits))
}

/// `score_symbols` on the call's own `budget`; the result stays charged.
pub(crate) fn score_symbols_in(
    out: &Tensor,
    syms: &[Symbol],
    dict_ratio: f64,
    budget: &Budget,
) -> Result<Vec<Scored>, Error> {
    if !out.is_consistent() {
        return Ok(Vec::new());
    }
    let width = out.map.len();
    let cert = |t: usize| {
        let row = out.row(t);
        let p = row.get(argmax(row)).map_or(0.0, |&v| f64::from(v));
        (prob_to_certainty(p) + CERT_OFFSET) * dict_ratio
    };
    let mut scored = Vec::new();
    budget.reserve(&mut scored, syms.len())?;
    let mut prev_end = 0;
    for (i, s) in syms.iter().enumerate() {
        let end = if i == syms.len() - 1 {
            width
        } else {
            s.end.min(width)
        };
        let mut c = 0.0f64;
        let mut r = 0.0f64;
        for t in prev_end..end {
            let v = cert(t);
            if v < c {
                c = v;
            }
            r -= v;
        }
        budget.charge(s.text.len())?;
        scored.push(Scored {
            symbol: s.clone(),
            certainty: c,
            rating: r,
        });
        prev_end = s.end;
    }
    Ok(scored)
}

/// `LSTMRecognizer::LabelsToCharBoundaries` for the single-code case: boundary `i` is the midpoint
/// of the gap before symbol `i`, bracketed by 0 and `width`.
///
/// # Errors
///
/// `Error::AllocLimit` if the boundaries do not fit in `limits.max_alloc`, `Error::OutOfMemory` if the allocator refuses.
pub fn char_boundaries(
    syms: &[Scored],
    width: usize,
    limits: &Limits,
) -> Result<Vec<usize>, Error> {
    char_boundaries_in(syms, width, &Budget::new(limits))
}

/// `char_boundaries` on the call's own `budget`; the result stays charged.
pub(crate) fn char_boundaries_in(
    syms: &[Scored],
    width: usize,
    budget: &Budget,
) -> Result<Vec<usize>, Error> {
    let mut bounds = Vec::new();
    char_boundaries_into(syms, width, &mut bounds, budget)?;
    Ok(bounds)
}

/// `char_boundaries_in`, replacing the contents of `bounds` (the caller's, which keeps its
/// capacity from one call to the next); only growth is charged.
pub(crate) fn char_boundaries_into(
    syms: &[Scored],
    width: usize,
    bounds: &mut Vec<usize>,
    budget: &Budget,
) -> Result<(), Error> {
    grow(budget, bounds, syms.len().saturating_add(1))?;
    if syms.is_empty() {
        bounds.push(width);
        return Ok(());
    }
    bounds.push(0);
    for i in 1..syms.len() {
        let end = syms[i - 1].symbol.end;
        // i128, so symbols the caller built from arbitrary indices cannot overflow.
        let gap = syms[i].symbol.start as i128 - end as i128;
        bounds.push((end as i128 + gap / 2) as usize);
    }
    bounds.push(width);
    Ok(())
}
