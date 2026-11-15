//! How much picture structure is in what a denoiser removed, without a clean reference.
//!
//! `texture_ratio` is the residual variance in the most textured quarter of the luma blocks
//! over the flattest quarter, per brightness band. True grain gives about 1.16 and a blur
//! 1.5 to 1.9. It spots gross smearing on real film but does not rank denoisers.

use super::{BATCH, Source};
use anyhow::{Result, bail};
use rayon::prelude::*;
use serde_json::json;

const BLOCK: usize = 16;
const BANDS: usize = 16;
const MIN_BAND_BLOCKS: usize = 64;

struct BlockSample {
  band: u8,
  texture: f32,
  variance: f32,
}

// 2x2 means of a plane, so (w / 2) x (h / 2).
fn halve(plane: &[u16], width: usize, height: usize) -> Vec<f32> {
  let (hw, hh) = (width / 2, height / 2);
  let mut out = vec![0.0; hw * hh];
  for y in 0..hh {
    for x in 0..hw {
      let at = |dx, dy| plane[(2 * y + dy) * width + 2 * x + dx] as f32;
      out[y * hw + x] = (at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1)) * 0.25;
    }
  }
  out
}

// Squared residual summed over the frame, and the luma blocks.
fn frame_stats(source: &[u16], denoised: &[u16], width: usize, height: usize) -> (f64, Vec<BlockSample>) {
  // Texture is the Laplacian of the denoised picture's 2x2 means, 0 at the border.
  let d = halve(denoised, width, height);
  let (w, h) = (width / 2, height / 2);
  let mut edge = vec![0.0f32; w * h];
  for y in 1..h - 1 {
    for x in 1..w - 1 {
      let i = y * w + x;
      edge[i] = 4.0 * d[i] - d[i - 1] - d[i + 1] - d[i - w] - d[i + w];
    }
  }
  let mut squared = 0.0;

  // Blocks measured on the full size residual, so fine grain counts fully.
  let mut blocks = Vec::new();
  for by in 0..height / BLOCK {
    for bx in 0..width / BLOCK {
      let (mut sum_d, mut sum_r, mut sum_r2) = (0.0f64, 0.0f64, 0.0f64);
      for y in by * BLOCK..(by + 1) * BLOCK {
        for x in bx * BLOCK..(bx + 1) * BLOCK {
          let i = y * width + x;
          let r = source[i] as f64 - denoised[i] as f64;
          sum_d += denoised[i] as f64;
          sum_r += r;
          sum_r2 += r * r;
        }
      }
      let n = (BLOCK * BLOCK) as f64;
      let mut texture = 0.0f64;
      let half = BLOCK / 2;
      for y in by * half..(by + 1) * half {
        for x in bx * half..(bx + 1) * half {
          texture += (edge[y * w + x] as f64).powi(2);
        }
      }
      squared += sum_r2;
      let mean = sum_d / n;
      blocks.push(BlockSample {
        band: ((mean / 1024.0 * BANDS as f64) as usize).min(BANDS - 1) as u8,
        texture: (texture / (half * half) as f64) as f32,
        variance: (sum_r2 / n - (sum_r / n).powi(2)) as f32,
      });
    }
  }
  (squared, blocks)
}

fn texture_ratio(mut blocks: Vec<BlockSample>) -> Option<f64> {
  blocks.sort_by(|a, b| a.band.cmp(&b.band).then(a.texture.total_cmp(&b.texture)));
  let (mut log_sum, mut weight) = (0.0, 0.0);
  for band in blocks.chunk_by(|a, b| a.band == b.band) {
    let quarter = band.len() / 4;
    if quarter < MIN_BAND_BLOCKS {
      continue;
    }
    let mean = |part: &[BlockSample]| part.iter().map(|b| b.variance as f64).sum::<f64>() / part.len() as f64;
    let (flat, textured) = (mean(&band[..quarter]), mean(&band[band.len() - quarter..]));
    if flat <= 0.0 || textured <= 0.0 {
      continue;
    }
    // Geometric mean over bands, weighted by how many blocks each holds.
    log_sum += (textured / flat).ln() * band.len() as f64;
    weight += band.len() as f64;
  }
  (weight > 0.0).then(|| (log_sum / weight).exp())
}

pub fn residual(source: &str, denoised: &str, crop: Option<&str>) -> Result<serde_json::Value> {
  let mut a = Source::open(source, crop)?;
  let mut b = Source::open(denoised, crop)?;
  let (width, height) = a.size();
  if b.size() != (width, height) {
    bail!("the videos differ in size");
  }
  let mut squared = 0.0;
  let mut blocks = Vec::new();
  let mut frames = 0;
  loop {
    let mut batch = Vec::with_capacity(BATCH);
    while batch.len() < BATCH {
      let (Some(s), Some(d)) = (a.next(), b.next()) else { break };
      batch.push((s, d));
    }
    if batch.is_empty() {
      break;
    }
    frames += batch.len();
    let results: Vec<_> = batch
      .par_iter()
      .map(|(s, d)| frame_stats(&s.planes[0], &d.planes[0], width, height))
      .collect();
    for (frame_squared, frame_blocks) in results {
      squared += frame_squared;
      blocks.extend(frame_blocks);
    }
  }
  if frames == 0 {
    bail!("no frames could be read");
  }
  if a.next().is_some() || b.next().is_some() {
    bail!("the videos differ in length");
  }
  Ok(json!({
    "frames": frames,
    "residual_rms": (squared / (frames * (width / BLOCK) * (height / BLOCK) * BLOCK * BLOCK) as f64).sqrt(),
    "texture_ratio": texture_ratio(blocks),
  }))
}
