//! The GPU filter against the CPU reference on a small synthetic clip. Skipped without a GPU.

use rabbit_denoise::{Denoiser, Device, Frame, Plane, Settings};

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

fn clip(frames: usize) -> Vec<Frame> {
  let mut rng = Rng(99);
  (0..frames)
    .map(|t| {
      let plane = |width: usize, height: usize, rng: &mut Rng| {
        let mut white = vec![0.0f32; width * height];
        for v in white.iter_mut() {
          *v = rng.gauss();
        }
        let mut data = vec![0.0f32; width * height];
        for y in 0..height {
          for x in 0..width {
            // Grain that spreads over neighbouring samples.
            let at = |dx: usize, dy: usize| white[(y + dy).min(height - 1) * width + (x + dx).min(width - 1)];
            let grain = (at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1)) * 4.0;
            let picture = 300.0 + (x + 2 * t) as f32 * 2.0 + if (x / 6 + y / 6) % 2 == 0 { 40.0 } else { 0.0 };
            data[y * width + x] = (picture + grain).clamp(0.0, 1023.0).round();
          }
        }
        Plane { width, height, data }
      };
      Frame {
        planes: vec![plane(160, 96, &mut rng), plane(80, 48, &mut rng), plane(80, 48, &mut rng)],
      }
    })
    .collect()
}

fn run(device: Device, input: &[Frame]) -> Option<Vec<Frame>> {
  let mut settings = Settings::new(2);
  settings.device = device;
  let mut denoiser = match Denoiser::new(settings, 10) {
    Ok(d) => d,
    Err(e) if device == Device::Gpu => {
      eprintln!("skipped: {e:#}");
      return None;
    }
    Err(e) => panic!("{e:#}"),
  };
  let mut out = Vec::new();
  for frame in input {
    out.extend(denoiser.push(frame.clone()).unwrap());
  }
  out.extend(denoiser.finish().unwrap());
  Some(out)
}

#[test]
fn gpu_matches_cpu() {
  // Long enough for the GPU's rings of 16 frames to wrap around more than once.
  let input = clip(40);
  let cpu = run(Device::Cpu, &input).unwrap();
  let Some(gpu) = run(Device::Gpu, &input) else { return };
  assert_eq!(cpu.len(), gpu.len());
  let (mut error, mut count) = (0.0f64, 0usize);
  for (a, b) in cpu.iter().zip(&gpu) {
    for (pa, pb) in a.planes.iter().zip(&b.planes) {
      for (x, y) in pa.data.iter().zip(&pb.data) {
        error += ((x - y) as f64).powi(2);
        count += 1;
      }
    }
  }
  let psnr = 10.0 * (1023.0f64 * 1023.0 * count as f64 / error.max(1e-12)).log10();
  assert!(psnr > 60.0, "GPU and CPU differ: {psnr:.1} dB");
}
