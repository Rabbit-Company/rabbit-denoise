//! Orthonormal DCT-II of 8x8 patches and of groups of any length.

pub const N: usize = 8;
pub const AREA: usize = N * N;

/// Basis of the orthonormal DCT-II of length `len`: `basis[k * len + i]`.
pub fn basis(len: usize) -> Vec<f32> {
  let mut out = vec![0.0f32; len * len];
  for k in 0..len {
    let scale = if k == 0 {
      (1.0 / len as f64).sqrt()
    } else {
      (2.0 / len as f64).sqrt()
    };
    for i in 0..len {
      out[k * len + i] = (scale * (std::f64::consts::PI * (2 * i + 1) as f64 * k as f64 / (2 * len) as f64).cos()) as f32;
    }
  }
  out
}

/// The 8x8 basis, and the group bases for every length up to `max_group`.
pub struct Transforms {
  patch: [f32; AREA],
  groups: Vec<Vec<f32>>,
}

impl Transforms {
  pub fn new(max_group: usize) -> Self {
    let mut patch = [0.0; AREA];
    patch.copy_from_slice(&basis(N));
    Self {
      patch,
      groups: (0..=max_group).map(basis).collect(),
    }
  }

  /// 2D DCT of a patch stored row by row. The coefficient for vertical frequency v and
  /// horizontal frequency u lands at `v * 8 + u`.
  pub fn forward(&self, input: &[f32; AREA], out: &mut [f32; AREA]) {
    let mut rows = [0.0f32; AREA];
    // Along each row.
    for y in 0..N {
      for u in 0..N {
        let mut sum = 0.0;
        for x in 0..N {
          sum += self.patch[u * N + x] * input[y * N + x];
        }
        rows[y * N + u] = sum;
      }
    }
    // Down each column.
    for v in 0..N {
      for u in 0..N {
        let mut sum = 0.0;
        for y in 0..N {
          sum += self.patch[v * N + y] * rows[y * N + u];
        }
        out[v * N + u] = sum;
      }
    }
  }

  pub fn inverse(&self, input: &[f32; AREA], out: &mut [f32; AREA]) {
    let mut cols = [0.0f32; AREA];
    for y in 0..N {
      for u in 0..N {
        let mut sum = 0.0;
        for v in 0..N {
          sum += self.patch[v * N + y] * input[v * N + u];
        }
        cols[y * N + u] = sum;
      }
    }
    for y in 0..N {
      for x in 0..N {
        let mut sum = 0.0;
        for u in 0..N {
          sum += self.patch[u * N + x] * cols[y * N + u];
        }
        out[y * N + x] = sum;
      }
    }
  }

  /// DCT along the group, for every patch coefficient, in place. `group[i]` is the 2D
  /// spectrum of member i.
  pub fn group_forward(&self, group: &mut [[f32; AREA]]) {
    self.group_apply(group, false);
  }

  pub fn group_inverse(&self, group: &mut [[f32; AREA]]) {
    self.group_apply(group, true);
  }

  fn group_apply(&self, group: &mut [[f32; AREA]], inverse: bool) {
    let len = group.len();
    if len <= 1 {
      return;
    }
    let basis = &self.groups[len];
    let mut column = vec![0.0f32; len];
    for f in 0..AREA {
      for (k, out) in column.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (i, member) in group.iter().enumerate() {
          let b = if inverse { basis[i * len + k] } else { basis[k * len + i] };
          sum += b * member[f];
        }
        *out = sum;
      }
      for (member, value) in group.iter_mut().zip(&column) {
        member[f] = *value;
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn patch_transform_round_trips() {
    let t = Transforms::new(4);
    let mut input = [0.0; AREA];
    for (i, v) in input.iter_mut().enumerate() {
      *v = ((i * 37) % 101) as f32 - 50.0;
    }
    let (mut spectrum, mut back) = ([0.0; AREA], [0.0; AREA]);
    t.forward(&input, &mut spectrum);
    t.inverse(&spectrum, &mut back);
    for (a, b) in input.iter().zip(&back) {
      assert!((a - b).abs() < 1e-3);
    }
    // A flat patch has only a DC coefficient, 8 times its value.
    let flat = [3.0; AREA];
    t.forward(&flat, &mut spectrum);
    assert!((spectrum[0] - 24.0).abs() < 1e-4);
    assert!(spectrum[1..].iter().all(|c| c.abs() < 1e-4));
  }

  #[test]
  fn group_transform_round_trips_and_keeps_energy() {
    let t = Transforms::new(5);
    let mut group: Vec<[f32; AREA]> = (0..5).map(|m| std::array::from_fn(|f| ((m * 7 + f * 3) % 11) as f32)).collect();
    let original = group.clone();
    let energy = |g: &[[f32; AREA]]| g.iter().flatten().map(|v| v * v).sum::<f32>();
    t.group_forward(&mut group);
    assert!((energy(&group) - energy(&original)).abs() / energy(&original) < 1e-5);
    t.group_inverse(&mut group);
    for (a, b) in group.iter().flatten().zip(original.iter().flatten()) {
      assert!((a - b).abs() < 1e-3);
    }
  }
}
