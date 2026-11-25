//! Go's `%q` for strings (`strconv.Quote`), shared by the model dump and `Layer::tree` so both
//! print what the Go tool prints.

use crate::{Budget, Error, Limits};
use std::fmt::Write as _;

/// `strconv.IsPrint` for a char that is not ASCII-special: letters, marks, numbers, punctuation,
/// symbols and the space. Rust's `escape_debug` escapes the same classes (controls, format
/// characters, separators other than the space, unassigned), but it also escapes a leading
/// grapheme extender (so the char is probed in second place, behind an `a`) and every
/// default-ignorable code point, which Go prints when it is a letter or a mark. Those are
/// listed here (Unicode 17, the version of both Go 1.27 and Rust 1.99).
fn is_print(c: char) -> bool {
    if matches!(
        c as u32,
        0x034F
            | 0x115F..=0x1160
            | 0x17B4..=0x17B5
            | 0x180B..=0x180D
            | 0x180F
            | 0x3164
            | 0xFE00..=0xFE0F
            | 0xFFA0
            | 0xE0100..=0xE01EF
    ) {
        return true;
    }
    let probe = format!("a{c}");
    probe.escape_debug().eq(probe.chars())
}

/// Go's `%q` of a string: double-quoted, with the escapes of `strconv.Quote`. The most it can
/// write is `4 * s.len() + 2` bytes (a control byte becomes `\xNN`), and the caller has room for it.
pub(crate) fn quote_into(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if c.is_ascii() && !c.is_ascii_control() => out.push(c),
            '\x07' => out.push_str("\\a"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x0b' => out.push_str("\\v"),
            c if c.is_ascii() => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c if is_print(c) => out.push(c),
            c if (c as u32) < 0x10000 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => {
                let _ = write!(out, "\\U{:08x}", c as u32);
            }
        }
    }
    out.push('"');
}

/// Go's `%q` of a string: double-quoted, with the escapes of `strconv.Quote`.
///
/// Enforces `limits.max_alloc`: the result takes at most `4 * s.len() + 2` bytes, which are
/// charged before it is built. No plane caps apply and it spawns no thread.
///
/// # Errors
///
/// `Error::AllocLimit` if the result does not fit in `limits.max_alloc`, `Error::OutOfMemory` if
/// the allocator refuses.
pub fn quote(s: &str, limits: &Limits) -> Result<String, Error> {
    let bound = s
        .len()
        .checked_mul(4)
        .and_then(|n| n.checked_add(2))
        .ok_or(Error::AllocLimit)?;
    Budget::new(limits).charge(bound)?;
    let mut out = String::new();
    out.try_reserve_exact(bound)
        .map_err(|_| Error::OutOfMemory)?;
    quote_into(&mut out, s);
    Ok(out)
}

/// A model string cut for an error message: its first 64 chars quoted, then its byte length if
/// it was cut. Errors stay small however long the name in the file is.
pub(crate) fn short(s: &str) -> String {
    const KEEP: usize = 64;
    let end = s.char_indices().nth(KEEP).map_or(s.len(), |(i, _)| i);
    let mut out = String::with_capacity(4 * end + 2 + 24);
    quote_into(&mut out, &s[..end]);
    if end < s.len() {
        let _ = write!(out, "... ({} bytes)", s.len());
    }
    out
}
