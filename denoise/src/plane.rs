//! Pictures as floating point planes, in the source's own code values.

#[derive(Clone)]
pub struct Plane {
  pub width: usize,
  pub height: usize,
  pub data: Vec<f32>,
}

impl Plane {
  pub fn new(width: usize, height: usize) -> Self {
    Self {
      width,
      height,
      data: vec![0.0; width * height],
    }
  }

  /// This plane averaged down to `width` x `height`.
  pub fn downscale(&self, width: usize, height: usize) -> Plane {
    let (fx, fy) = (self.width.div_ceil(width), self.height.div_ceil(height));
    let mut out = Plane::new(width, height);
    for y in 0..height {
      for x in 0..width {
        let (mut sum, mut count) = (0.0, 0.0);
        for sy in y * fy..((y + 1) * fy).min(self.height) {
          for sx in x * fx..((x + 1) * fx).min(self.width) {
            sum += self.at(sx, sy);
            count += 1.0;
          }
        }
        out.data[y * width + x] = if count > 0.0 { sum / count } else { 0.0 };
      }
    }
    out
  }

  #[inline]
  pub fn at(&self, x: usize, y: usize) -> f32 {
    self.data[y * self.width + x]
  }

  #[inline]
  pub fn row(&self, x: usize, y: usize, len: usize) -> &[f32] {
    let start = y * self.width + x;
    &self.data[start..start + len]
  }
}

/// One picture: luma alone, or luma and two chroma planes.
#[derive(Clone)]
pub struct Frame {
  pub planes: Vec<Plane>,
}
