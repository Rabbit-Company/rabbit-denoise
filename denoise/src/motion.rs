//! Block motion between two frames, for following patches into neighbouring frames.
//!
//! Every 16x16 block gets the move that brings it closest to the second frame, tried on
//! both frames averaged 2x2 to take out most of the grain. Moves are even.

use crate::plane::Plane;
use rayon::prelude::*;

const BLOCK: usize = 16;

pub struct MotionField {
  blocks_x: usize,
  blocks_y: usize,
  vectors: Vec<(i32, i32)>,
}

impl MotionField {
  /// The motion from `from` to `to`, up to `range` samples in each direction.
  pub fn estimate(from: &Plane, to: &Plane, range: usize) -> Self {
    let (hw, hh) = (from.width / 2, from.height / 2);
    let a = from.downscale(hw, hh);
    let b = to.downscale(hw, hh);
    let half = BLOCK / 2;
    let (blocks_x, blocks_y) = ((hw / half).max(1), (hh / half).max(1));
    let r = (range / 2) as i32;
    let vectors = (0..blocks_y)
      .into_par_iter()
      .flat_map_iter(|by| {
        let (a, b) = (&a, &b);
        (0..blocks_x).map(move |bx| {
          let (x0, y0) = ((bx * half).min(hw.saturating_sub(half)), (by * half).min(hh.saturating_sub(half)));
          let mut best = (f32::MAX, 0i32, (0i32, 0i32));
          for dy in -r..=r {
            for dx in -r..=r {
              let (x, y) = (x0 as i32 + dx, y0 as i32 + dy);
              if x < 0 || y < 0 || x as usize + half > hw || y as usize + half > hh {
                continue;
              }
              let mut sum = 0.0f32;
              for row in 0..half {
                let pa = a.row(x0, y0 + row, half);
                let pb = b.row(x as usize, y as usize + row, half);
                for i in 0..half {
                  let d = pa[i] - pb[i];
                  sum += d * d;
                }
              }
              // Among equal matches, the smallest move.
              let length = dx.abs() + dy.abs();
              if sum < best.0 || (sum == best.0 && length < best.1) {
                best = (sum, length, (dx * 2, dy * 2));
              }
            }
          }
          best.2
        })
      })
      .collect();
    Self {
      blocks_x,
      blocks_y,
      vectors,
    }
  }

  /// The move of the block nearest to the 8x8 patch at (x, y).
  pub fn at(&self, x: usize, y: usize) -> (i32, i32) {
    let bx = ((x + 4) / BLOCK).min(self.blocks_x - 1);
    let by = ((y + 4) / BLOCK).min(self.blocks_y - 1);
    self.vectors[by * self.blocks_x + bx]
  }
}
