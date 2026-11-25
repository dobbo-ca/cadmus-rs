//! `NetworkIO::FromPixes`, `NetworkIO::Copy2DImage`, `NetworkIO::SetPixel` and `ComputeBlackWhite`
//! in `src/lstm/networkio.cpp`, `Input::PreparePixInput` in `src/lstm/input.cpp`,
//! `ImageData::PreScale` in `src/ccstruct/imagedata.cpp` and `STATS::ile` in
//! `src/ccstruct/statistc.cpp` of Tesseract OCR (Apache-2.0, see NOTICE).

use super::err;
use crate::imaging::{Bitmap, release_bitmap, scale_gray_pooled};
use crate::nn::{Rand, Scratch, StrideMap, Tensor};
use crate::{Budget, Error, Limits};
use std::sync::atomic::AtomicBool;

/// A line image prepared for the network.
#[derive(Debug)]
pub struct Normalized {
    pub input: Tensor,
    /// Maps network timesteps back to source-image x: the source x of timestep `t` is
    /// `t * scale_factor` (`xScale / imFactor`, `LSTMRecognizer::GetScaledX` times the x scale).
    pub scale_factor: f64,
    pub black: f32,
    pub contrast: f32,
}

/// Prepares a line image for the network: scale to `input_height` as `ImageData::PreScale` does,
/// then `Input::PreparePixInput`'s contrast normalization, filling the rows the scaler left short
/// from `rnd` (`NetworkIO::Randomize`). `img` must be 8bpp grey; build it with
/// `Bitmap::from_gray8` or `Bitmap::from_rgba8`.
///
/// One [`Budget`] covers the scaled copy and the input tensor; only the tensor stays charged.
///
/// Enforces `limits.max_alloc` and `limits.max_threads` (through `scale_gray`).
///
/// # Errors
///
/// `Error::Recog` for a bitmap that is not 8bpp or is empty, or a scaled line below the network's
/// minimum; `Error::PixelLimit`, `Error::AllocLimit` from `limits`; `Error::Cancelled`, `Error::OutOfMemory` if the allocator refuses.
pub fn normalize(
    img: &Bitmap,
    input_height: usize,
    x_scale: usize,
    rnd: &Rand,
    limits: &Limits,
    cancel: &AtomicBool,
) -> Result<Normalized, Error> {
    normalize_in(
        img,
        input_height,
        x_scale,
        rnd,
        &Budget::new(limits),
        cancel,
    )
}

/// `normalize` on the call's own `budget`.
pub(crate) fn normalize_in(
    img: &Bitmap,
    input_height: usize,
    x_scale: usize,
    rnd: &Rand,
    budget: &Budget,
    cancel: &AtomicBool,
) -> Result<Normalized, Error> {
    normalize_pooled(img, input_height, x_scale, rnd, budget, cancel, None)
}

/// `normalize_in`, with the scaled copy and the input tensor taken from `pool` when there is one
/// (the caller gives the tensor back).
pub(crate) fn normalize_pooled(
    img: &Bitmap,
    input_height: usize,
    x_scale: usize,
    rnd: &Rand,
    budget: &Budget,
    cancel: &AtomicBool,
    mut pool: Option<&mut Scratch>,
) -> Result<Normalized, Error> {
    if img.depth != 8 {
        return Err(err(format!(
            "line image has depth {}; 8bpp grey is required",
            img.depth
        )));
    }
    if img.height == 0 || img.width == 0 {
        return Err(err(format!(
            "empty line image {}x{}",
            img.width, img.height
        )));
    }
    let im_factor = f64::from(input_height as f32 / img.height as f32);
    let scaled = scale_gray_pooled(img, im_factor, budget, cancel, pool.as_deref_mut())?;
    if scaled.width < x_scale || scaled.height < x_scale {
        return Err(err(format!(
            "scaled line is {}x{}, below the network's minimum of {x_scale} in either dimension",
            scaled.width, scaled.height
        )));
    }

    let (black, white) = compute_black_white(&scaled);
    let mut contrast = (white - black) / 2.0;
    if contrast <= 0.0 {
        contrast = 1.0;
    }

    let map = StrideMap {
        height: input_height,
        width: scaled.width,
    };
    let mut input = match pool.as_deref_mut() {
        Some(p) => Tensor::pooled(map, 1, &mut p.f32s, budget)?,
        None => Tensor::new(map, 1, budget)?,
    };
    let mut v = [0f64; 1];
    for y in 0..input_height {
        crate::imaging::check_cancel(cancel)?;
        if y < scaled.height {
            for x in 0..input.map.width {
                let fp = (f32::from(scaled.at(x, y)) - black) / contrast - 1.0;
                v[0] = f64::from(fp);
                input.write_time_step(input.map.t(y, x), &v);
            }
        } else {
            for x in 0..input.map.width {
                v[0] = rnd.signed_rand(1.0);
                input.write_time_step(input.map.t(y, x), &v);
            }
        }
    }

    release_bitmap(scaled, budget, pool);
    Ok(Normalized {
        input,
        scale_factor: x_scale as f64 / im_factor,
        black,
        contrast,
    })
}

/// `ComputeBlackWhite`: the 25th percentile of the local minima and the 75th percentile of the
/// local maxima along the middle row.
fn compute_black_white(b: &Bitmap) -> (f32, f32) {
    let mut mins = Stats::default();
    let mut maxes = Stats::default();
    if b.width >= 3 {
        let y = b.height / 2;
        let mut prev = i32::from(b.at(0, y));
        let mut curr = i32::from(b.at(1, y));
        for x in 1..b.width - 1 {
            let next = i32::from(b.at(x + 1, y));
            if (curr < prev && curr <= next) || (curr <= prev && curr < next) {
                mins.add(curr, 1);
            }
            if (curr > prev && curr >= next) || (curr >= prev && curr > next) {
                maxes.add(curr, 1);
            }
            prev = curr;
            curr = next;
        }
    }
    if mins.total == 0 {
        mins.add(0, 1);
    }
    if maxes.total == 0 {
        maxes.add(255, 1);
    }
    (mins.ile(0.25) as f32, maxes.ile(0.75) as f32)
}

/// A 256-bucket histogram with `STATS::ile`.
struct Stats {
    buckets: [usize; 256],
    total: usize,
}

impl Default for Stats {
    fn default() -> Stats {
        Stats {
            buckets: [0; 256],
            total: 0,
        }
    }
}

impl Stats {
    fn add(&mut self, value: i32, count: usize) {
        self.buckets[value.clamp(0, 255) as usize] += count;
        self.total += count;
    }

    /// `STATS::ile`: the value below which a fraction of the samples lie, interpolated within
    /// the bucket.
    fn ile(&self, frac: f64) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let total = self.total as f64;
        let mut target = frac * total;
        if target < 1.0 {
            target = 1.0;
        } else if target > total {
            target = total;
        }
        let (mut sum, mut index) = (0usize, 0usize);
        while index < self.buckets.len() && (sum as f64) < target {
            sum += self.buckets[index];
            index += 1;
        }
        if index > 0 {
            return index as f64 - (sum as f64 - target) / self.buckets[index - 1] as f64;
        }
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::Stats;

    // Go TestILEInterpolatesAcrossBuckets: two samples at 10 and two at 20.
    #[test]
    fn ile_interpolates_across_buckets() {
        let mut s = Stats::default();
        s.add(10, 2);
        s.add(20, 2);
        assert_eq!(s.ile(0.25), 10.5);
        assert_eq!(s.ile(0.75), 20.5);
        assert_eq!(Stats::default().ile(0.5), 0.0);
        // A fraction of 0 clips the target up to 1.
        assert_eq!(s.ile(0.0), 10.5);
    }

    // Go TestILEEmptyHistogramDefaults: the one-sample defaults give 1 and 256, not 0 and 255.
    #[test]
    fn ile_of_the_default_samples() {
        let (mut mins, mut maxes) = (Stats::default(), Stats::default());
        mins.add(0, 1);
        maxes.add(255, 1);
        assert_eq!(mins.ile(0.25), 1.0);
        assert_eq!(maxes.ile(0.75), 256.0);
    }
}
