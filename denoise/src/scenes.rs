//! Scene cuts found without denoising, for `analyze`.
//!
//! The denoiser finds cuts in its first stage's result, which costs half a denoise. This
//! compares frames averaged over 8x8 blocks instead, which averages most grain away, and
//! scores them as ffmpeg's scdet does: a cut is a jump in the mean difference between
//! frames, not a high difference alone, so fast motion is not taken for cuts.

use crate::dct::N;
use crate::plane::Plane;
use rayon::prelude::*;

/// The score, in per cent of the code value range, from which a frame starts a scene. The
/// same as scdet's default.
pub const CUT_SCORE: f32 = 10.0;

pub struct CutDetector {
  maxval: f32,
  previous: Option<Plane>,
  /// The mean difference of the last pair, in per cent.
  previous_difference: f32,
}

impl CutDetector {
  pub fn new(maxval: f32) -> Self {
    Self {
      maxval,
      previous: None,
      previous_difference: 0.0,
    }
  }

  /// Whether this luma plane starts a new scene. The first frame does.
  pub fn is_cut(&mut self, luma: &Plane) -> bool {
    let small = luma.downscale((luma.width / N).max(1), (luma.height / N).max(1));
    let Some(previous) = self.previous.replace(small) else {
      return true;
    };
    let current = self.previous.as_ref().expect("just stored");
    if previous.width != current.width || previous.height != current.height {
      return true;
    }
    let sum: f64 = previous.data.iter().zip(&current.data).map(|(a, b)| (a - b).abs() as f64).sum();
    let difference = (sum / current.data.len() as f64 / self.maxval as f64 * 100.0) as f32;
    let score = difference.min((difference - self.previous_difference).abs());
    self.previous_difference = difference;
    score >= CUT_SCORE
  }
}

/// A 3x3 box blur, which cuts white grain to a third of its deviation and keeps most of the
/// picture, so that blocks can be matched between frames on the picture.
pub fn blur(plane: &Plane) -> Plane {
  blur_into(plane, Vec::new())
}

/// Rows of `blur` made on one thread at a time.
const BLUR_ROWS: usize = 8;

/// `blur`, written into `data`, a buffer to reuse.
pub fn blur_into(plane: &Plane, mut data: Vec<f32>) -> Plane {
  let (w, h) = (plane.width, plane.height);
  data.resize(w * h, 0.0);
  let mut out = Plane { width: w, height: h, data };
  out.data.par_chunks_mut(w * BLUR_ROWS).enumerate().for_each(|(band, rows)| {
    // The rows blurred across, for this band's rows and the one on each side.
    let y0 = band * BLUR_ROWS;
    let (first, last) = (y0.saturating_sub(1), (y0 + rows.len() / w).min(h - 1));
    let mut across = vec![0.0f32; (last + 1 - first) * w];
    for (y, row) in (first..=last).zip(across.chunks_mut(w)) {
      let src = plane.row(0, y, w);
      // The edges repeat their sample. The rest runs without the clamps, which vectorizes.
      let at = |x: usize| (src[x.saturating_sub(1)] + src[x] + src[(x + 1).min(w - 1)]) / 3.0;
      row[0] = at(0);
      if w > 1 {
        row[w - 1] = at(w - 1);
      }
      if w > 2 {
        for ((out, s), (l, r)) in row[1..w - 1].iter_mut().zip(&src[1..w - 1]).zip(src[..w - 2].iter().zip(&src[2..])) {
          *out = (l + s + r) / 3.0;
        }
      }
    }
    let at = |y: usize| &across[(y - first) * w..(y - first + 1) * w];
    for (y, row) in (y0..).zip(rows.chunks_mut(w)) {
      let (up, here, down) = (at(y.saturating_sub(1)), at(y), at((y + 1).min(h - 1)));
      for (out, ((u, c), d)) in row.iter_mut().zip(up.iter().zip(here).zip(down)) {
        *out = (u + c + d) / 3.0;
      }
    }
  });
  out
}

#[cfg(test)]
mod tests {
  use super::*;

  fn flat(value: f32, shift: usize) -> Plane {
    let (w, h) = (128, 64);
    let mut plane = Plane::new(w, h);
    for y in 0..h {
      for x in 0..w {
        // A gradient with a pattern, moved `shift` samples to the right.
        let xs = x + 1000 - shift;
        plane.data[y * w + x] = value + (xs % 40) as f32 * 2.0 + ((xs / 7 + y / 5) % 3) as f32 * 5.0;
      }
    }
    plane
  }

  #[test]
  fn a_new_picture_is_a_cut_and_motion_is_not() {
    let mut detector = CutDetector::new(255.0);
    assert!(detector.is_cut(&flat(40.0, 0)));
    for shift in 1..6 {
      assert!(!detector.is_cut(&flat(40.0, shift * 3)), "shift {shift}");
    }
    assert!(detector.is_cut(&flat(160.0, 0)));
    assert!(!detector.is_cut(&flat(160.0, 1)));
  }
}
