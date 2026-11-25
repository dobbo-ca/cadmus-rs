//! `Unicharset`: the text format read by `UNICHARSET::load_via_fgets`, reduced to the columns the
//! LSTM recognition path reads. From `unicharset.cpp`.
//!
//! Translated from Tesseract OCR (Apache-2.0, see NOTICE), through the Go port. The translation
//! is not verbatim.

use super::{err, short};
use crate::{Budget, Error};

/// Reserved unichar ids. `kSpecialUnicharCodes` is `{" ", "Joined", "|Broken|0|1"}` and every
/// unicharset with special codes starts with exactly those three.
pub const UNICHAR_SPACE: usize = 0;
pub const UNICHAR_JOINED: usize = 1;
pub const UNICHAR_BROKEN: usize = 2;

// Character property bits. Stored HEXADECIMAL in the file.
pub const PROP_ALPHA: u32 = 0x1;
pub const PROP_LOWER: u32 = 0x2;
pub const PROP_UPPER: u32 = 0x4;
pub const PROP_DIGIT: u32 = 0x8;
pub const PROP_PUNCT: u32 = 0x10;

/// A corruption guard, not a Tesseract constant. The largest shipped unicharset (chi_sim) is a
/// few thousand entries.
const MAX_UNICHARSET_SIZE: usize = 1_000_000;

/// One unicharset entry. Not retained: the metrics tuple, other_case, direction and mirror, which
/// only the legacy classifier, the wordrec language model and x-height fixing read.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Unichar {
    /// The raw file token, the UTF-8 that `id_to_unichar` returns.
    pub text: String,
    /// `get_normed_unichar`'s result, which `LSTMRecognizer::DecodeLabel` uses. It differs from
    /// `text` for 6 eng entries (curly quotes, em dash, trade mark). Defaults to `text`.
    pub normed: String,
    /// Drives `is_space_delimited`, i.e. word splitting.
    pub script: String,
    pub properties: u32,
}

#[derive(Debug, Clone)]
pub struct Unicharset {
    chars: Vec<Unichar>,
}

/// Parses the unicharset text format. The text must be UTF-8 (Go strings are bytes; invalid
/// UTF-8 is an error here) and a line may be at most 64 KiB, as `bufio.Scanner` allows.
///
/// Charged to `budget`, so it enforces its `max_alloc`.
///
/// # Errors
///
/// `Error::Tessdata` for text that is not UTF-8 or is malformed, `Error::AllocLimit` over
/// `max_alloc`, `Error::OutOfMemory` if the allocator refuses.
pub fn parse_unicharset(data: &[u8], budget: &Budget) -> Result<Unicharset, Error> {
    let text = std::str::from_utf8(data).map_err(|_| err("unicharset is not UTF-8"))?;
    let mut lines = text.split('\n').map(|l| l.strip_suffix('\r').unwrap_or(l));
    let first = lines.next().unwrap_or("");
    if data.is_empty() {
        return Err(err("unicharset is empty"));
    }
    let n: i64 = first.trim().parse().map_err(|_| {
        err(format!(
            "unicharset entry count {} is not a number",
            short(first)
        ))
    })?;
    // Deliberate divergence: load_via_fgets accepts a count of 0 and returns true. An empty LSTM
    // unicharset cannot yield a usable model, so reject it here.
    if n <= 0 || n as usize > MAX_UNICHARSET_SIZE {
        return Err(err(format!("implausible unicharset size {n}")));
    }
    let n = n as usize;
    // Every entry is at least one line of 4 bytes, so a declared count above that cannot be
    // satisfied; refuse it before charging and reserving for it.
    if n > data.len() / 4 {
        return Err(err(format!(
            "unicharset declares {n} entries but has only {} bytes",
            data.len()
        )));
    }
    budget.charge(n.saturating_mul(std::mem::size_of::<Unichar>() + 16))?;
    let mut chars = Vec::new();
    chars.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
    for id in 0..n {
        let Some(line) = lines.next() else {
            return Err(err(format!(
                "unicharset declares {n} entries but ends after {id}"
            )));
        };
        if line.len() > 64 * 1024 {
            return Err(err("unicharset: reading unicharset: token too long"));
        }
        // The entry holds up to three copies of the line's text, and the parse a list of its
        // fields (16 bytes each, at most one per two bytes) while it runs.
        budget.charge(line.len().saturating_mul(3))?;
        let _fields = budget.hold(line.len().saturating_add(1).saturating_mul(16))?;
        chars.push(
            parse_unichar_line(line).map_err(|e| err(format!("unicharset entry {id}: {e}")))?,
        );
    }
    // Anything after the declared count is ignored, matching load_via_fgets.
    Ok(Unicharset { chars })
}

/// One entry line. The C++ loader's six progressive extractions accept: unichar, HEX properties,
/// an optional comma-separated metrics tuple, a script name, then up to four further fields
/// (other_case, direction, mirror, normed).
fn parse_unichar_line(line: &str) -> Result<Unichar, String> {
    // Everything from the first tab is a human-readable debug comment that load_via_fgets never
    // reaches.
    let line = line.split('\t').next().unwrap_or("");
    // Counted first so the list is allocated once at its final size: a list that doubles as it
    // grows holds the old and the new block together, which the caller's charge does not cover.
    let mut f: Vec<&str> = Vec::with_capacity(line.split_whitespace().count());
    f.extend(line.split_whitespace());
    if f.len() < 3 {
        return Err(format!(
            "want at least 3 fields, got {} in {}",
            f.len(),
            short(line)
        ));
    }
    let mut text = f[0].to_string();
    // The literal token "NULL" in column 1 spells the space character.
    if text == "NULL" {
        text = " ".to_string();
    }
    let properties = if f[1].starts_with('+') {
        Err("sign".to_string())
    } else {
        u32::from_str_radix(f[1], 16).map_err(|e| e.to_string())
    }
    .map_err(|e| format!("properties {} is not hexadecimal: {e}", short(f[1])))?;
    // Field 3 is the metrics tuple when it contains commas, else the script.
    let si = if f[2].contains(',') { 3 } else { 2 };
    let Some(script) = f.get(si) else {
        return Err(format!("no script field in {}", short(line)));
    };
    // After the script come other_case, direction, mirror, normed, in that order. normed is
    // present only when all four are.
    let rest = &f[si + 1..];
    let normed = if rest.len() == 4 {
        rest[3].to_string()
    } else {
        text.clone()
    };
    Ok(Unichar {
        text,
        normed,
        script: (*script).to_string(),
        properties,
    })
}

impl Unicharset {
    pub fn size(&self) -> usize {
        self.chars.len()
    }

    pub fn char(&self, id: usize) -> Option<&Unichar> {
        self.chars.get(id)
    }

    /// `id_to_unichar`: the raw file representation ("" for an invalid id).
    pub fn text(&self, id: usize) -> &str {
        self.chars.get(id).map_or("", |c| &c.text)
    }

    /// `get_normed_unichar` ("" for an invalid id).
    pub fn normed(&self, id: usize) -> &str {
        self.chars.get(id).map_or("", |c| &c.normed)
    }

    /// `UNICHARSET::IsSpaceDelimited`: every script except Han, Thai, Hangul, Hiragana and
    /// Katakana writes words with spaces between them.
    ///
    /// Deliberate divergence: Tesseract compares resolved script *ids*, and an unseen script name
    /// resolves to 0 (the null script), so in a unicharset with no Han entry a null-script
    /// character is reported as NOT space delimited. Comparing names avoids that; eng has no
    /// null-script entry, so the two agree on the model at hand.
    pub fn is_space_delimited(&self, id: usize) -> bool {
        match self.chars.get(id) {
            None => true, // INVALID_UNICHAR_ID
            Some(c) => !matches!(
                c.script.as_str(),
                "Han" | "Thai" | "Hangul" | "Hiragana" | "Katakana"
            ),
        }
    }
}
