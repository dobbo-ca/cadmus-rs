use std::fmt;

/// One error enum for the crate. Go's errors are all `fmt.Errorf` strings per package, so the
/// variants are the Go error classes: the package that failed, plus the limit variants Go
/// does not have. The text is a description for people, not a stable token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// `internal/tessdata`: a malformed or unsupported `.traineddata` component.
    Tessdata(String),
    /// `internal/nn`: a layer or tensor shape that does not fit.
    Network(String),
    /// `internal/imaging`: an invalid image argument.
    Imaging(String),
    /// `internal/recog`: a recognizer built or driven with invalid arguments.
    Recog(String),
    /// A model larger than `Limits::max_model_bytes`.
    ModelLimit,
    /// A plane over `max_pixels`, `max_width` or `max_height`.
    PixelLimit,
    /// Working memory over `max_alloc`.
    AllocLimit,
    /// The allocator refused.
    OutOfMemory,
    /// The cancel flag was set.
    Cancelled,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Tessdata(s) => write!(f, "tessdata: {s}"),
            Error::Network(s) => write!(f, "nn: {s}"),
            Error::Imaging(s) => write!(f, "imaging: {s}"),
            Error::Recog(s) => write!(f, "recog: {s}"),
            Error::ModelLimit => f.write_str("model larger than Limits::max_model_bytes"),
            Error::PixelLimit => f.write_str("plane larger than the Limits pixel caps"),
            Error::AllocLimit => f.write_str("working memory over Limits::max_alloc"),
            Error::OutOfMemory => f.write_str("out of memory"),
            Error::Cancelled => f.write_str("cancelled"),
        }
    }
}

impl std::error::Error for Error {}
