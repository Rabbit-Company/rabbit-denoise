//! Collaborative filtering of one plane, in the style of V-BM3D.
//!
//! Every reference patch on a grid is grouped with similar patches from its own frame and
//! the frames around it. The group is transformed with an 8x8 DCT and a DCT along the
//! group, shrunk by the grain variance of each coefficient and averaged back.
//!
//! - `Stage::Hard` matches on the noisy picture and keeps a coefficient only where it
//!   clears `lambda` grain deviations. The result is the basic estimate.
//! - `Stage::Wiener` matches on the basic estimate and scales every noisy coefficient by
//!   basic^2 / (basic^2 + grain variance).

use crate::dct::{AREA, N, Transforms};
use crate::motion::MotionField;
use crate::noise::NoiseModel;
use crate::plane::Plane;
use rayon::prelude::*;

/// Candidates kept per frame, best first. The GPU keeps the same list.
pub const TOP_K: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
  Hard,
  Wiener,
}

#[derive(Clone, Debug)]
pub struct Params {
  /// Distance between reference patches.
  pub step: usize,
  /// Search radius around the reference patch in its own frame.
  pub search_radius: usize,
  /// Search radius around each followed match in the next frame.
  pub predict_radius: usize,
  /// How many of the best matches in a frame are followed into the next.
  pub predict_count: usize,
  /// Frames on each side of the current one.
  pub temporal_radius: usize,
  pub max_group: usize,
  /// Matches taken into the group from each frame, besides the reference.
  pub per_frame: usize,
  /// Follow motion with a block motion field between consecutive frames.
  pub motion: bool,
  /// How far the motion field looks, in samples.
  pub motion_range: usize,
  /// Same frame matches closer than this to the reference are skipped, as they share
  /// most of its grain.
  pub min_shift: usize,
  /// Hard threshold in grain deviations.
  pub lambda: f32,
  /// A match may differ from the reference by at most `distance_factor` times the
  /// difference grain alone makes, plus `distance_floor` on the 8 bit scale.
  pub distance_factor: f32,
  pub distance_floor: f32,
}

impl Params {
  pub fn hard(temporal_radius: usize) -> Self {
    Self {
      step: 3,
      // Within 0.1 dB of 7 and 15 per cent faster on the GPU.
      search_radius: 5,
      // Around the track and the field's prediction. Without the field it is 3.
      predict_radius: 1,
      predict_count: 1,
      temporal_radius,
      max_group: 16,
      per_frame: 2,
      // 0.15 to 0.25 dB better than tracking alone, and 16 to 20 per cent faster.
      motion: true,
      motion_range: 16,
      min_shift: 4,
      lambda: 2.7,
      distance_factor: 2.0,
      distance_floor: 4.0,
    }
  }

  pub fn wiener(temporal_radius: usize) -> Self {
    Self {
      lambda: 0.0,
      // Matching on the basic estimate, where little grain is left.
      distance_factor: 0.5,
      distance_floor: 2.0,
      ..Self::hard(temporal_radius)
    }
  }
}

/// The frames a stage sees: the current one and up to `temporal_radius` on each side.
pub struct Window<'a> {
  pub noisy: Vec<&'a Plane>,
  /// What patches are matched on: the noisy planes for the hard stage, the basic
  /// estimate for the Wiener stage.
  pub guide: Vec<&'a Plane>,
  pub center: usize,
  /// The plane's index, and every frame's key so the GPU uploads each one only once.
  pub plane: usize,
  pub noisy_keys: Vec<FrameKey>,
  pub guide_keys: Vec<FrameKey>,
  /// For chroma with `Params::motion`, the luma planes the motion is measured on.
  pub motion_guide: Option<Vec<&'a Plane>>,
}

/// A frame's index and, for one that is made more than once, its version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameKey {
  pub index: usize,
  pub version: u32,
}

#[derive(Clone, Copy)]
struct Match {
  distance: f32,
  frame: u8,
  x: u16,
  y: u16,
}

pub(crate) fn positions(len: usize, step: usize) -> Vec<usize> {
  let last = len - N;
  let mut out: Vec<usize> = (0..=last).step_by(step).collect();
  if *out.last().unwrap() != last {
    out.push(last);
  }
  out
}

fn load(plane: &Plane, x: usize, y: usize, out: &mut [f32; AREA]) {
  for row in 0..N {
    out[row * N..(row + 1) * N].copy_from_slice(plane.row(x, y + row, N));
  }
}

/// Sum of squared differences, or None once it passes `limit`.
#[inline]
fn distance(reference: &[f32; AREA], plane: &Plane, x: usize, y: usize, limit: f32) -> Option<f32> {
  let mut sum = 0.0f32;
  for row in 0..N {
    let candidate = plane.row(x, y + row, N);
    let r = &reference[row * N..(row + 1) * N];
    let mut s = 0.0f32;
    for i in 0..N {
      let d = r[i] - candidate[i];
      s += d * d;
    }
    sum += s;
    if sum > limit {
      return None;
    }
  }
  Some(sum)
}

pub(crate) fn kaiser() -> [f32; AREA] {
  fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term) = (1.0, 1.0);
    for k in 1..30 {
      term *= (x / 2.0) * (x / 2.0) / (k * k) as f64;
      sum += term;
    }
    sum
  }
  let beta = 2.0;
  let w1: Vec<f64> = (0..N)
    .map(|i| {
      let r = 2.0 * i as f64 / (N - 1) as f64 - 1.0;
      bessel_i0(beta * (1.0 - r * r).sqrt()) / bessel_i0(beta)
    })
    .collect();
  std::array::from_fn(|i| (w1[i / N] * w1[i % N]) as f32)
}

struct Search<'a> {
  window: &'a Window<'a>,
  params: &'a Params,
  /// With `params.motion`, the fields from frame i to i + 1 and to i - 1.
  forward: Vec<Option<MotionField>>,
  backward: Vec<Option<MotionField>>,
}

impl Search<'_> {
  fn frame_search(
    &self,
    reference: &[f32; AREA],
    limit: f32,
    frame: usize,
    centers: &[(usize, usize)],
    radius: usize,
    found: &mut Vec<Match>,
  ) {
    let plane = self.window.guide[frame];
    let mut spots: Vec<(usize, usize)> = Vec::new();
    for &(cx, cy) in centers {
      let x0 = cx.saturating_sub(radius);
      let x1 = (cx + radius).min(plane.width - N);
      let y0 = cy.saturating_sub(radius);
      let y1 = (cy + radius).min(plane.height - N);
      for y in y0..=y1 {
        for x in x0..=x1 {
          spots.push((x, y));
        }
      }
    }
    // Row by row, as one window is searched and as the GPU orders ties.
    if centers.len() > 1 {
      spots.sort_unstable_by_key(|&(x, y)| (y, x));
      spots.dedup();
    }
    for (x, y) in spots {
      if let Some(d) = distance(reference, plane, x, y, limit) {
        found.push(Match {
          distance: d,
          frame: frame as u8,
          x: x as u16,
          y: y as u16,
        });
      }
    }
  }

  /// The group for the reference patch at (x, y): the reference, then the best `per_frame`
  /// matches of every frame, following the best matches from frame to frame. `limit` is
  /// the largest sum of squared differences.
  fn group(&self, reference: &[f32; AREA], limit: f32, x: usize, y: usize, out: &mut Vec<Match>) {
    let c = self.window.center;
    let p = self.params;
    out.clear();
    out.push(Match {
      distance: 0.0,
      frame: c as u8,
      x: x as u16,
      y: y as u16,
    });
    let mut here = Vec::new();
    self.frame_search(reference, limit, c, &[(x, y)], p.search_radius, &mut here);
    here.retain(|m| (m.x as usize).abs_diff(x) >= p.min_shift || (m.y as usize).abs_diff(y) >= p.min_shift);
    here.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    here.truncate(TOP_K);

    // Matches within a frame must also be `min_shift` apart from each other.
    let distinct = |list: &[Match], taken: &mut Vec<Match>| {
      let start = taken.len();
      for m in list {
        if taken.len() - start == p.per_frame {
          break;
        }
        let apart = taken[start..]
          .iter()
          .all(|t| t.x.abs_diff(m.x) as usize >= p.min_shift || t.y.abs_diff(m.y) as usize >= p.min_shift);
        if apart {
          taken.push(*m);
        }
      }
    };
    let mut chosen: Vec<Match> = Vec::new();
    distinct(&here, &mut chosen);
    for direction in [-1i32, 1] {
      // The track starts at the reference's own position.
      let mut centers = vec![(x, y)];
      for k in 1..=p.temporal_radius {
        let frame = c as i32 + direction * k as i32;
        if frame < 0 || frame as usize >= self.window.guide.len() {
          break;
        }
        let mut found = Vec::new();
        if p.motion {
          // Where the field moves the last match from the frame before.
          let from = (frame - direction) as usize;
          let fields = if direction > 0 { &self.forward } else { &self.backward };
          let plane = self.window.guide[frame as usize];
          if let Some(field) = &fields[from] {
            // A field measured on luma for a smaller chroma plane: scale in and out.
            let (sx, sy) = match &self.window.motion_guide {
              Some(guide) => (guide[0].width / plane.width.max(1), guide[0].height / plane.height.max(1)),
              None => (1, 1),
            };
            let (cx, cy) = centers[0];
            let (dx, dy) = field.at(cx * sx, cy * sy);
            let px = (cx as i32 + dx / sx as i32).clamp(0, (plane.width - N) as i32) as usize;
            let py = (cy as i32 + dy / sy as i32).clamp(0, (plane.height - N) as i32) as usize;
            // Around where the field moves it, and around where it was.
            centers = vec![(cx, cy), (px, py)];
          }
        }
        self.frame_search(reference, limit, frame as usize, &centers, p.predict_radius, &mut found);
        if found.is_empty() {
          break;
        }
        found.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        found.truncate(TOP_K);
        centers = found.iter().take(p.predict_count).map(|m| (m.x as usize, m.y as usize)).collect();
        distinct(&found, &mut chosen);
      }
    }
    chosen.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    out.extend(chosen.into_iter().take(p.max_group - 1));
  }
}

/// Filters the window's current frame and returns it.
pub fn filter(stage: Stage, window: &Window, noise: &NoiseModel, params: &Params, transforms: &Transforms, maxval: f32) -> Plane {
  let current = window.noisy[window.center];
  let (width, height) = (current.width, current.height);
  if width < N || height < N {
    return current.clone();
  }
  let scale = maxval / 255.0;
  let frames = window.guide.len();
  let field = |from: usize, to: Option<usize>| -> Option<MotionField> {
    let to = to.filter(|&t| t < frames)?;
    let planes = window.motion_guide.as_ref().unwrap_or(&window.guide);
    params
      .motion
      .then(|| MotionField::estimate(planes[from], planes[to], params.motion_range))
  };
  let search = Search {
    window,
    params,
    forward: (0..frames).map(|i| field(i, Some(i + 1))).collect(),
    backward: (0..frames).map(|i| field(i, i.checked_sub(1))).collect(),
  };
  let floor = params.distance_floor * scale * scale;
  let window_weights = kaiser();
  let xs = positions(width, params.step);
  let ys = positions(height, params.step);

  // Bands of reference rows add into their own strips, summed afterwards.
  let rows_per_band = 8;
  let bands: Vec<&[usize]> = ys.chunks(rows_per_band).collect();
  let strips: Vec<(usize, Vec<f32>, Vec<f32>)> = bands
    .par_iter()
    .map(|band_ys| {
      let top = band_ys[0].saturating_sub(params.search_radius);
      let bottom = (band_ys[band_ys.len() - 1] + params.search_radius + N).min(height);
      let mut numerator = vec![0.0f32; (bottom - top) * width];
      let mut denominator = vec![0.0f32; (bottom - top) * width];
      let mut group = Vec::with_capacity(params.max_group);
      let mut reference = [0.0f32; AREA];
      let mut patch = [0.0f32; AREA];
      let mut variance = [0.0f32; AREA];
      let mut noisy: Vec<[f32; AREA]> = Vec::with_capacity(params.max_group);
      let mut basic: Vec<[f32; AREA]> = Vec::with_capacity(params.max_group);
      for &y in band_ys.iter() {
        for &x in &xs {
          load(window.guide[window.center], x, y, &mut reference);
          let mean = reference.iter().sum::<f32>() / AREA as f32;
          noise.variance(mean, &mut variance);
          // Orthonormal DCT, so the mean over coefficients is the variance per sample.
          let per_sample = variance.iter().sum::<f32>() / AREA as f32;
          let limit = (params.distance_factor * 2.0 * per_sample + floor) * AREA as f32;
          search.group(&reference, limit, x, y, &mut group);

          noisy.clear();
          for m in &group {
            load(window.noisy[m.frame as usize], m.x as usize, m.y as usize, &mut patch);
            let mut spectrum = [0.0f32; AREA];
            transforms.forward(&patch, &mut spectrum);
            noisy.push(spectrum);
          }
          transforms.group_forward(&mut noisy);

          // Variance of the grain left in the estimate, for the aggregation weight.
          let mut left = 0.0f32;
          match stage {
            Stage::Hard => {
              for (k, member) in noisy.iter_mut().enumerate() {
                for f in 0..AREA {
                  if k == 0 && f == 0 {
                    left += variance[f];
                    continue;
                  }
                  let threshold = params.lambda * params.lambda * variance[f];
                  if member[f] * member[f] < threshold {
                    member[f] = 0.0;
                  } else {
                    left += variance[f];
                  }
                }
              }
            }
            Stage::Wiener => {
              basic.clear();
              for m in &group {
                load(window.guide[m.frame as usize], m.x as usize, m.y as usize, &mut patch);
                let mut spectrum = [0.0f32; AREA];
                transforms.forward(&patch, &mut spectrum);
                basic.push(spectrum);
              }
              transforms.group_forward(&mut basic);
              for (member, estimate) in noisy.iter_mut().zip(&basic) {
                for f in 0..AREA {
                  let b2 = estimate[f] * estimate[f];
                  let gain = if b2 + variance[f] > 0.0 { b2 / (b2 + variance[f]) } else { 1.0 };
                  member[f] *= gain;
                  left += gain * gain * variance[f];
                }
              }
            }
          }
          let weight = if left > 0.0 { 1.0 / left } else { 1.0 };

          transforms.group_inverse(&mut noisy);
          for (m, spectrum) in group.iter().zip(&noisy) {
            if m.frame as usize != window.center {
              continue;
            }
            transforms.inverse(spectrum, &mut patch);
            let (px, py) = (m.x as usize, m.y as usize - top);
            for row in 0..N {
              let base = (py + row) * width + px;
              for col in 0..N {
                let w = weight * window_weights[row * N + col];
                numerator[base + col] += w * patch[row * N + col];
                denominator[base + col] += w;
              }
            }
          }
        }
      }
      (top, numerator, denominator)
    })
    .collect();

  let mut numerator = vec![0.0f32; width * height];
  let mut denominator = vec![0.0f32; width * height];
  for (top, n, d) in strips {
    let offset = top * width;
    for (i, (nv, dv)) in n.iter().zip(&d).enumerate() {
      numerator[offset + i] += nv;
      denominator[offset + i] += dv;
    }
  }
  let mut out = Plane::new(width, height);
  for i in 0..width * height {
    out.data[i] = if denominator[i] > 0.0 {
      numerator[i] / denominator[i]
    } else {
      current.data[i]
    };
  }
  out
}
