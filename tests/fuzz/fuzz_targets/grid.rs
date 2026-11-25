#![no_main]
use cadmus::Limits;
use cadmus::imaging::{Grid, Rect};
use libfuzzer_sys::fuzz_target;

mod common;

// data = [cell][bounds w: u16][bounds h: u16][bounds origin x: i16][origin y: i16] (-32768 and 32767
// are isize::MIN and isize::MAX) then records of [op][x0 i16][y0 i16][x1 i16][y1 i16]:
// an odd op inserts the box, an even op queries with it. Bit 2 of the op maps the coordinates
// -32768 and 32767 to isize::MIN and isize::MAX. One grid is driven through the whole sequence,
// and every query is checked against a scan of the inserted boxes.
fuzz_target!(|data: &[u8]| {
    let Some((&[cell, w0, w1, h0, h1, x0, x1, y0, y1], ops)) = data.split_first_chunk::<9>() else {
        return;
    };
    let limits = Limits {
        max_alloc: 1 << 20,
        ..Limits::default()
    };
    let origin = |lo, hi| match i16::from_le_bytes([lo, hi]) {
        i16::MIN => isize::MIN,
        i16::MAX => isize::MAX,
        x => isize::from(x),
    };
    let (ox, oy) = (origin(x0, x1), origin(y0, y1));
    let bounds = Rect::new(
        ox,
        oy,
        ox.saturating_add(usize::from(u16::from_le_bytes([w0, w1])) as isize),
        oy.saturating_add(usize::from(u16::from_le_bytes([h0, h1])) as isize),
    );
    let Some(mut grid) = common::no_oom(Grid::new(bounds, usize::from(cell) % 65, &limits)) else {
        return;
    };
    let mut boxes: Vec<(i64, Rect)> = Vec::new();
    for rec in ops.chunks_exact(9) {
        let v = |i: usize| match i16::from_le_bytes([rec[i], rec[i + 1]]) {
            i16::MIN if rec[0] & 4 != 0 => isize::MIN,
            i16::MAX if rec[0] & 4 != 0 => isize::MAX,
            x => isize::from(x),
        };
        let r = Rect::new(v(1), v(3), v(5), v(7));
        if rec[0] & 1 == 1 {
            let id = boxes.len() as i64 % 7;
            if grid.insert(id, r).is_err() {
                return;
            }
            boxes.push((id, r));
        } else {
            let mut want: Vec<i64> = boxes
                .iter()
                .filter(|(_, b)| b.overlaps(&r))
                .map(|&(id, _)| id)
                .collect();
            want.sort_unstable();
            want.dedup();
            if let Some(got) = common::no_oom(grid.query(r)) {
                assert_eq!(got, want);
            }
        }
    }
});
