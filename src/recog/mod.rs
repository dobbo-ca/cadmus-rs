//! The recognizer: a runnable network from a loaded model, line normalization, greedy and beam
//! decoding, and words. A port of `internal/recog`; the pieces are translated from
//! `src/lstm/networkio.cpp`, `input.cpp`, `lstmrecognizer.cpp`, `recodebeam.cpp` and
//! `src/ccstruct/imagedata.cpp` of Tesseract OCR (Apache-2.0, see NOTICE).

mod beam;
mod build;
mod certainty;
mod decode;
mod normalize;
mod words;
mod workspace;

pub use build::Network;
pub use certainty::{
    CERT_OFFSET, CERTAINTY_SCALE, DICT_RATIO, MIN_CERTAINTY, Scored, char_boundaries,
    prob_to_certainty, score_symbols,
};
pub use decode::{Recognizer, Symbol};
pub use normalize::{Normalized, normalize};
pub use words::{Line, Word, confidence_percent};
pub use workspace::Workspace;

use crate::Error;
use decode::argmax;

pub(crate) fn err(msg: impl Into<String>) -> Error {
    Error::Recog(msg.into())
}

#[cfg(test)]
mod tests;
