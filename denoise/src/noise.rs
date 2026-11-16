//! The grain to remove, as a variance for every 8x8 DCT frequency and every brightness.
//!
//! The model keeps a full 8x8 spectrum for each of `BANDS` brightness bands, the same two
//! things AV1 describes with its AR filter and scaling function. Per band, the median
//! squared coefficient of the flatter half of the 8x8 blocks estimates the grain variance.
//! Blocks with more energy than grain can have are left out. Bands without blocks take
//! their neighbours' grain.

use crate::dct::{AREA, N, Transforms};
use crate::plane::Plane;
use rayon::prelude::*;

pub const BANDS: usize = 16;
// Median of a chi-square variable with one degree of freedom.
const MEDIAN_CHI2: f32 = 0.454_936_4;
// The most a frequency's grain deviation may exceed the band's high frequency grain.
// AV1 cannot draw grain that peaks higher, so removing more would only lose picture.
const AV1_PEAK: f32 = 6.0;
// The residual and pair measurements take every BLOCK_STRIDE-th block in each direction.
// That still leaves about 8000 blocks in a 1080p frame.
pub const BLOCK_STRIDE: usize = 2;
// How far the pair measurement looks for each block in the second frame.
pub const PAIR_RADIUS: usize = 6;
// Fewer kept blocks than this and a band borrows from its neighbours.
const MIN_BLOCKS: u32 = 48;
// Blocks this flat are black bars or clipped highlights, which carry no grain.
const MIN_BLOCK_VARIANCE: f32 = 0.05;
// Most energy a block may have, over the flattest tenth's, and still count as grain.
const PLAUSIBLE_RATIO: f32 = 20.0;

/// A block of the first frame of a pair and its best match in the second.
#[derive(Clone, Copy)]
pub struct PairBlock {
  /// The DCT of the halved difference of the noisy blocks.
  pub spectrum: [f32; AREA],
  /// The squared differences of the matched blocks, summed.
  pub distance: f32,
  /// The mean of the first frame's basic estimate over the block.
  pub mean: f32,
}

/// A block of what the basic estimate removed.
#[derive(Clone, Copy)]
pub struct ResidualBlock {
  /// The DCT of the noisy block less the basic one.
  pub spectrum: [f32; AREA],
  /// The energy of the basic block's DCT without its DC coefficient.
  pub kept: f32,
  /// The basic block's DC coefficient.
  pub dc: f32,
}

/// A frame's measured blocks for each plane, where they were measured, before their
/// medians are taken.
pub struct Measures {
  pub pairs: Option<Vec<Vec<PairBlock>>>,
  pub residual: Option<Vec<Vec<ResidualBlock>>>,
}

/// What one frame's plane says about its grain.
#[derive(Clone)]
pub struct NoiseStats {
  variance: Vec<[f32; AREA]>,
  blocks: Vec<u32>,
}

impl NoiseStats {
  pub fn measure(plane: &Plane, maxval: f32, transforms: &Transforms) -> Self {
    let (bw, bh) = (plane.width / N, plane.height / N);
    let load = |bx: usize, by: usize| -> [f32; AREA] {
      let mut patch = [0.0f32; AREA];
      for y in 0..N {
        patch[y * N..(y + 1) * N].copy_from_slice(plane.row(bx * N, by * N + y, N));
      }
      patch
    };
    // Every block's mean and AC energy from plain sums. Only kept blocks are transformed.
    let (means, energy): (Vec<f32>, Vec<f32>) = (0..bh)
      .into_par_iter()
      .flat_map_iter(|by| {
        (0..bw).map(move |bx| {
          let patch = load(bx, by);
          let mean = patch.iter().sum::<f32>() / AREA as f32;
          let ac = patch.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / AREA as f32;
          (mean, ac)
        })
      })
      .unzip();

    // How much each block's mean differs from its neighbours': gradients and edges.
    let structure = |bx: usize, by: usize| {
      let m = means[by * bw + bx];
      let mut sum = 0.0;
      let mut count = 0.0;
      for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
        let (x, y) = (bx as i32 + dx, by as i32 + dy);
        if x >= 0 && y >= 0 && (x as usize) < bw && (y as usize) < bh {
          let d = m - means[y as usize * bw + x as usize];
          sum += d * d;
          count += 1.0;
        }
      }
      if count > 0.0 { sum / count } else { 0.0 }
    };

    let mut live: Vec<f32> = energy.iter().copied().filter(|&e| e >= MIN_BLOCK_VARIANCE).collect();
    if live.is_empty() {
      return Self {
        variance: vec![[0.0; AREA]; BANDS],
        blocks: vec![0; BANDS],
      };
    }
    let tenth = live.len() / 10;
    let (_, floor, _) = live.select_nth_unstable_by(tenth, f32::total_cmp);
    let ceiling = *floor * PLAUSIBLE_RATIO;

    let mut by_band: Vec<Vec<(f32, usize)>> = vec![Vec::new(); BANDS];
    for by in 0..bh {
      for bx in 0..bw {
        let i = by * bw + bx;
        if energy[i] < MIN_BLOCK_VARIANCE || energy[i] > ceiling {
          continue;
        }
        by_band[band(means[i], maxval)].push((structure(bx, by), i));
      }
    }

    let chosen: Vec<Vec<usize>> = by_band
      .iter_mut()
      .map(|members| {
        members.sort_by(|a, b| a.0.total_cmp(&b.0));
        members[..members.len() / 2].iter().map(|&(_, i)| i).collect()
      })
      .collect();
    // The spectra of the kept blocks only, and the lists renumbered to them.
    let kept: Vec<usize> = chosen.iter().flatten().copied().collect();
    let spectra: Vec<[f32; AREA]> = kept
      .par_iter()
      .map(|&i| {
        let mut spectrum = [0.0f32; AREA];
        transforms.forward(&load(i % bw, i / bw), &mut spectrum);
        spectrum
      })
      .collect();
    let mut next = 0;
    let selected: Vec<Vec<usize>> = chosen
      .iter()
      .map(|band| {
        let renumbered = (next..next + band.len()).collect();
        next += band.len();
        renumbered
      })
      .collect();
    // The DC coefficient holds the picture, so its grain comes from the frequencies next to it.
    Self::from_blocks(&spectra, &selected, false)
  }

  /// Measures the grain in what the basic estimate removed, in the blocks where the detail
  /// it kept is weaker than what it removed.
  pub fn measure_residual(noisy: &Plane, basic: &Plane, maxval: f32, transforms: &Transforms) -> Self {
    Self::from_residual(&Self::residual_blocks(noisy, basic, transforms), maxval)
  }

  /// Every BLOCK_STRIDE-th block of the residual, for `from_residual`.
  pub fn residual_blocks(noisy: &Plane, basic: &Plane, transforms: &Transforms) -> Vec<ResidualBlock> {
    let (bw, bh) = (noisy.width / N, noisy.height / N);
    (0..bh.div_ceil(BLOCK_STRIDE))
      .into_par_iter()
      .map(|k| k * BLOCK_STRIDE)
      .flat_map_iter(|by| {
        (0..bw).step_by(BLOCK_STRIDE).map(move |bx| {
          let (mut patch, mut spectrum, mut basic_spectrum) = ([0.0f32; AREA], [0.0f32; AREA], [0.0f32; AREA]);
          for y in 0..N {
            patch[y * N..(y + 1) * N].copy_from_slice(basic.row(bx * N, by * N + y, N));
          }
          transforms.forward(&patch, &mut basic_spectrum);
          for y in 0..N {
            let (n, b) = (noisy.row(bx * N, by * N + y, N), basic.row(bx * N, by * N + y, N));
            for x in 0..N {
              patch[y * N + x] = n[x] - b[x];
            }
          }
          transforms.forward(&patch, &mut spectrum);
          ResidualBlock {
            spectrum,
            kept: basic_spectrum[1..].iter().map(|c| c * c).sum(),
            dc: basic_spectrum[0],
          }
        })
      })
      .collect()
  }

  /// The residual measurement from every BLOCK_STRIDE-th block, as the GPU makes them too.
  pub fn from_residual(blocks: &[ResidualBlock], maxval: f32) -> Self {
    let mut selected: Vec<Vec<usize>> = vec![Vec::new(); BANDS];
    for (i, block) in blocks.iter().enumerate() {
      let removed: f32 = block.spectrum[1..].iter().map(|c| c * c).sum();
      if removed >= MIN_BLOCK_VARIANCE * AREA as f32 && block.kept < removed {
        selected[band(block.dc / N as f32, maxval)].push(i);
      }
    }
    let spectra: Vec<[f32; AREA]> = blocks.iter().map(|b| b.spectrum).collect();
    Self::from_blocks(&spectra, &selected, true)
  }

  /// Measures the grain from two consecutive frames. Where a block of frame a is found
  /// again in frame b, the difference of the noisy blocks holds two grain samples and no
  /// picture. Blocks are matched on the basic estimates. A pair across a scene cut
  /// measures nothing.
  ///
  /// With `matching`, blocks are matched on those planes instead and the caller handles
  /// cuts. Chroma uses the luma basic estimate scaled to its size.
  pub fn measure_temporal(
    noisy_a: &Plane,
    basic_a: &Plane,
    noisy_b: &Plane,
    basic_b: &Plane,
    matching: Option<(&Plane, &Plane)>,
    maxval: f32,
    transforms: &Transforms,
  ) -> Self {
    let blocks = Self::pair_blocks(noisy_a, basic_a, noisy_b, basic_b, matching, transforms);
    Self::from_pairs(&blocks, matching.is_none(), maxval)
  }

  /// Every BLOCK_STRIDE-th block of frame a with its match in frame b, for `from_pairs`.
  pub fn pair_blocks(
    noisy_a: &Plane,
    basic_a: &Plane,
    noisy_b: &Plane,
    basic_b: &Plane,
    matching: Option<(&Plane, &Plane)>,
    transforms: &Transforms,
  ) -> Vec<PairBlock> {
    let (width, height) = (noisy_a.width, noisy_a.height);
    let (bw, bh) = (width / N, height / N);
    let load = |plane: &Plane, x: usize, y: usize| -> [f32; AREA] {
      let mut out = [0.0; AREA];
      for row in 0..N {
        out[row * N..(row + 1) * N].copy_from_slice(plane.row(x, y + row, N));
      }
      out
    };
    (0..bh.div_ceil(BLOCK_STRIDE))
      .into_par_iter()
      .map(|k| k * BLOCK_STRIDE)
      .flat_map_iter(|by| {
        let mut out = Vec::with_capacity(bw);
        for bx in (0..bw).step_by(BLOCK_STRIDE) {
          let (x, y) = (bx * N, by * N);
          let (match_a, match_b) = matching.unwrap_or((basic_a, basic_b));
          let reference = load(match_a, x, y);
          let (mut best, mut at) = (f32::MAX, (x, y));
          for my in y.saturating_sub(PAIR_RADIUS)..=(y + PAIR_RADIUS).min(height - N) {
            for mx in x.saturating_sub(PAIR_RADIUS)..=(x + PAIR_RADIUS).min(width - N) {
              let mut sum = 0.0;
              for row in 0..N {
                let candidate = match_b.row(mx, my + row, N);
                let d: [f32; N] = std::array::from_fn(|i| reference[row * N + i] - candidate[i]);
                // Added in pairs four apart, so that the compiler can use vectors.
                sum +=
                  ((d[0] * d[0] + d[4] * d[4]) + (d[1] * d[1] + d[5] * d[5])) + ((d[2] * d[2] + d[6] * d[6]) + (d[3] * d[3] + d[7] * d[7]));
                if sum >= best {
                  break;
                }
              }
              if sum < best {
                best = sum;
                at = (mx, my);
              }
            }
          }
          let (a, b) = (load(noisy_a, x, y), load(noisy_b, at.0, at.1));
          let difference: [f32; AREA] = std::array::from_fn(|i| (a[i] - b[i]) * std::f32::consts::FRAC_1_SQRT_2);
          let mut spectrum = [0.0f32; AREA];
          transforms.forward(&difference, &mut spectrum);
          out.push(PairBlock {
            spectrum,
            distance: best,
            mean: load(basic_a, x, y).iter().sum::<f32>() / AREA as f32,
          });
        }
        out
      })
      .collect()
  }

  /// The pair measurement from every BLOCK_STRIDE-th block of the first frame, as the GPU
  /// makes them too. With `cuts`, a pair across a scene cut measures nothing.
  pub fn from_pairs(blocks: &[PairBlock], cuts: bool, maxval: f32) -> Self {
    const WORST_RATIO: f32 = 4.0;
    const CUT_RATIO: f32 = 0.5;
    // Per block: its band, the halved difference spectrum, and the basic and noisy distances.
    let found: Vec<(usize, [f32; AREA], f32, f32)> = blocks
      .iter()
      .filter(|b| b.spectrum[1..].iter().map(|c| c * c).sum::<f32>() / AREA as f32 >= MIN_BLOCK_VARIANCE)
      .map(|b| {
        // The halved difference's energy is half the noisy blocks' distance.
        let noisy_distance = b.spectrum.iter().map(|c| c * c).sum::<f32>() * 2.0 / AREA as f32;
        (band(b.mean, maxval), b.spectrum, b.distance / AREA as f32, noisy_distance)
      })
      .collect();
    if found.is_empty() {
      return Self {
        variance: vec![[0.0; AREA]; BANDS],
        blocks: vec![0; BANDS],
      };
    }
    let mut distances: Vec<f32> = found.iter().map(|f| f.2).collect();
    let mid = distances.len() / 2;
    let (_, typical, _) = distances.select_nth_unstable_by(mid, f32::total_cmp);
    let typical = *typical;
    let worst = typical * WORST_RATIO + 1e-3;
    let spectra: Vec<[f32; AREA]> = found.iter().map(|f| f.1).collect();
    let mut selected: Vec<Vec<usize>> = vec![Vec::new(); BANDS];
    for (i, f) in found.iter().enumerate() {
      if f.2 <= worst {
        selected[f.0].push(i);
      }
    }
    // Across a scene cut the basic and noisy distances come close, as both are mostly picture.
    let mut noisy_distances: Vec<f32> = found.iter().map(|f| f.3).collect();
    let (_, noisy_typical, _) = noisy_distances.select_nth_unstable_by(mid, f32::total_cmp);
    if cuts && typical > *noisy_typical * CUT_RATIO {
      return Self {
        variance: vec![[0.0; AREA]; BANDS],
        blocks: vec![0; BANDS],
      };
    }
    Self::from_blocks(&spectra, &selected, true)
  }

  /// Whether nothing was measured, as at a scene cut.
  pub fn is_empty(&self) -> bool {
    self.blocks.iter().all(|&n| n == 0)
  }

  /// Per band and frequency, the median squared coefficient over the selected blocks,
  /// scaled to a variance. With `with_dc` false the DC variance is extrapolated.
  fn from_blocks(spectra: &[[f32; AREA]], selected: &[Vec<usize>], with_dc: bool) -> Self {
    let mut variance = vec![[0.0f32; AREA]; BANDS];
    let mut blocks = vec![0u32; BANDS];
    let mut values = Vec::new();
    for (b, kept) in selected.iter().enumerate() {
      if kept.is_empty() {
        continue;
      }
      blocks[b] = kept.len() as u32;
      for f in if with_dc { 0 } else { 1 }..AREA {
        values.clear();
        values.extend(kept.iter().map(|&i| spectra[i][f] * spectra[i][f]));
        let mid = values.len() / 2;
        let (_, median, _) = values.select_nth_unstable_by(mid, f32::total_cmp);
        variance[b][f] = *median / MEDIAN_CHI2;
      }
      if !with_dc {
        variance[b][0] = (variance[b][1] + variance[b][N]) * 0.5;
      }
    }
    Self { variance, blocks }
  }
}

/// Caps every frequency of a spectrum at AV1_PEAK times the deviation of its highest
/// frequencies, those whose horizontal and vertical frequencies add up to 8 or more.
fn limit_to_av1(spectrum: &mut [f32; AREA]) {
  let (mut sum, mut count) = (0.0f32, 0.0f32);
  for v in 0..N {
    for u in 0..N {
      if u + v >= N {
        sum += spectrum[v * N + u];
        count += 1.0;
      }
    }
  }
  let ceiling = sum / count * AV1_PEAK * AV1_PEAK;
  for value in spectrum.iter_mut() {
    *value = value.min(ceiling);
  }
}

fn band(mean: f32, maxval: f32) -> usize {
  ((mean / (maxval + 1.0) * BANDS as f32).max(0.0) as usize).min(BANDS - 1)
}

/// Grain variance for every frequency and brightness, from one or more frames.
#[derive(Clone)]
pub struct NoiseModel {
  variance: Vec<[f32; AREA]>,
  maxval: f32,
  empty: bool,
}

impl NoiseModel {
  /// Averages the frames' estimates, weighted by each band's block count. Bands with too
  /// few blocks are interpolated from their neighbours.
  pub fn combine(stats: &[&NoiseStats], maxval: f32, scale: f32) -> Self {
    let mut variance = vec![[0.0f32; AREA]; BANDS];
    let mut weight = [0u32; BANDS];
    for s in stats {
      for b in 0..BANDS {
        weight[b] += s.blocks[b];
        for f in 0..AREA {
          variance[b][f] += s.variance[b][f] * s.blocks[b] as f32;
        }
      }
    }
    let valid: Vec<usize> = (0..BANDS).filter(|&b| weight[b] >= MIN_BLOCKS).collect();
    for &b in &valid {
      for v in variance[b].iter_mut() {
        *v = *v / weight[b] as f32 * scale * scale;
      }
    }
    let mut filled = vec![[0.0f32; AREA]; BANDS];
    for (b, out) in filled.iter_mut().enumerate() {
      let below = valid.iter().rev().find(|&&v| v <= b);
      let above = valid.iter().find(|&&v| v >= b);
      *out = match (below, above) {
        (Some(&lo), Some(&hi)) if lo == hi => variance[lo],
        (Some(&lo), Some(&hi)) => {
          let t = (b - lo) as f32 / (hi - lo) as f32;
          std::array::from_fn(|f| variance[lo][f] * (1.0 - t) + variance[hi][f] * t)
        }
        (Some(&v), None) | (None, Some(&v)) => variance[v],
        (None, None) => [0.0; AREA],
      };
    }
    for spectrum in filled.iter_mut() {
      limit_to_av1(spectrum);
    }
    Self {
      variance: filled,
      maxval,
      empty: valid.is_empty(),
    }
  }

  /// The smaller grain of the two models at every frequency and band, or this model when
  /// the other measured nothing.
  pub fn min(&self, other: &NoiseModel) -> NoiseModel {
    if other.empty {
      return self.clone();
    }
    let variance = self
      .variance
      .iter()
      .zip(&other.variance)
      .map(|(a, b)| std::array::from_fn(|f| a[f].min(b[f])))
      .collect();
    NoiseModel {
      variance,
      maxval: self.maxval,
      empty: self.empty,
    }
  }

  /// Whether no band had enough blocks to measure.
  pub fn is_empty(&self) -> bool {
    self.empty
  }

  /// This model, or `fallback` when no band had enough blocks to measure.
  pub fn or(self, fallback: &NoiseModel) -> NoiseModel {
    if self.empty { fallback.clone() } else { self }
  }

  /// Grain variance of every DCT coefficient of a patch with this mean, interpolated
  /// between the band centres.
  pub fn variance(&self, mean: f32, out: &mut [f32; AREA]) {
    let position = (mean / (self.maxval + 1.0) * BANDS as f32 - 0.5).clamp(0.0, (BANDS - 1) as f32);
    let lo = position.floor() as usize;
    let hi = (lo + 1).min(BANDS - 1);
    let t = position - lo as f32;
    for (f, v) in out.iter_mut().enumerate() {
      *v = self.variance[lo][f] * (1.0 - t) + self.variance[hi][f] * t;
    }
  }

  /// Mean grain variance per pixel in each band, for diagnostics.
  pub fn band_variance(&self) -> Vec<f32> {
    self.variance.iter().map(|v| v.iter().sum::<f32>() / AREA as f32).collect()
  }

  /// The 8x8 spectrum of one band, for diagnostics.
  pub fn spectrum(&self, band: usize) -> &[f32; AREA] {
    &self.variance[band]
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  // A small deterministic generator, so the tests need no dependency.
  struct Rng(u64);
  impl Rng {
    fn uniform(&mut self) -> f32 {
      self.0 ^= self.0 << 13;
      self.0 ^= self.0 >> 7;
      self.0 ^= self.0 << 17;
      (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
    fn gauss(&mut self) -> f32 {
      let (u, v) = (self.uniform().max(1e-7), self.uniform());
      (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos()
    }
  }

  #[test]
  fn white_grain_on_a_gradient_is_measured() {
    // Large enough that every band keeps about 1000 blocks.
    let (w, h) = (1024, 512);
    let mut plane = Plane::new(w, h);
    let mut rng = Rng(12345);
    for y in 0..h {
      for x in 0..w {
        plane.data[y * w + x] = 200.0 + x as f32 * 0.5 + 6.0 * rng.gauss();
      }
    }
    let t = Transforms::new(1);
    let stats = NoiseStats::measure(&plane, 1023.0, &t);
    let model = NoiseModel::combine(&[&stats], 1023.0, 1.0);
    let mut spectrum = [0.0; AREA];
    model.variance(400.0, &mut spectrum);
    // High frequencies see only the grain. The gradient adds to the lowest horizontal ones.
    for f in [9, 27, 36, 63] {
      assert!((spectrum[f] / 36.0 - 1.0).abs() < 0.15, "frequency {f}: {}", spectrum[f]);
    }
  }

  #[test]
  fn black_bars_do_not_count_as_grain_free_picture() {
    let (w, h) = (256, 256);
    let mut plane = Plane::new(w, h);
    let mut rng = Rng(7);
    for y in 0..h {
      for x in 0..w {
        // Left half: a bar at 64. Right half: picture at 70 with grain.
        plane.data[y * w + x] = if x < w / 2 { 64.0 } else { 70.0 + 4.0 * rng.gauss() };
      }
    }
    let t = Transforms::new(1);
    let stats = NoiseStats::measure(&plane, 1023.0, &t);
    let model = NoiseModel::combine(&[&stats], 1023.0, 1.0);
    let mut spectrum = [0.0; AREA];
    model.variance(66.0, &mut spectrum);
    assert!((spectrum[36] / 16.0 - 1.0).abs() < 0.2, "{}", spectrum[36]);
  }
}
