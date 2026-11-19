//! The GPU filter and grain measurement against the CPU reference on a small synthetic
//! clip. Skipped without a GPU.

use rabbit_denoise::bm3d::FrameKey;
use rabbit_denoise::dct::Transforms;
use rabbit_denoise::gpu::{Gpu, MeasureFrame};
use rabbit_denoise::noise::NoiseStats;
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

/// Denoises the frames of each run, skipping those between, as `--min-grain` does.
fn run_runs(device: Device, input: &[Frame], runs: &[(usize, usize)]) -> Option<Vec<Vec<Frame>>> {
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
  let mut next = 0;
  for (i, &(first, end)) in runs.iter().enumerate() {
    if i > 0 {
      denoiser.resume(first - next);
    }
    let mut frames = Vec::new();
    for frame in &input[first..end] {
      frames.extend(denoiser.push(frame.clone()).unwrap());
    }
    frames.extend(denoiser.finish().unwrap());
    out.push(frames);
    next = end;
  }
  Some(out)
}

fn psnr(a: &[Frame], b: &[Frame]) -> f64 {
  assert_eq!(a.len(), b.len());
  let (mut error, mut count) = (0.0f64, 0usize);
  for (fa, fb) in a.iter().zip(b) {
    for (pa, pb) in fa.planes.iter().zip(&fb.planes) {
      for (x, y) in pa.data.iter().zip(&pb.data) {
        error += ((x - y) as f64).powi(2);
        count += 1;
      }
    }
  }
  10.0 * (1023.0f64 * 1023.0 * count as f64 / error.max(1e-12)).log10()
}

#[test]
fn a_resumed_run_is_denoised_as_if_the_video_began_there() {
  // The second run starts past the GPU's rings of 16 frames, and its frames would land in
  // the slots of the first run's frames.
  let input = clip(48);
  let runs = [(0, 12), (18, 30), (36, 48)];
  let alone: Vec<Vec<Frame>> = runs.iter().map(|&(a, b)| run(Device::Cpu, &input[a..b]).unwrap()).collect();
  let cpu = run_runs(Device::Cpu, &input, &runs).unwrap();
  for (resumed, fresh) in cpu.iter().zip(&alone) {
    assert_eq!(resumed.len(), fresh.len());
    for (a, b) in resumed.iter().zip(fresh) {
      for (pa, pb) in a.planes.iter().zip(&b.planes) {
        assert_eq!(pa.data, pb.data);
      }
    }
  }
  let Some(gpu) = run_runs(Device::Gpu, &input, &runs) else { return };
  for (resumed, fresh) in gpu.iter().zip(&alone) {
    let db = psnr(resumed, fresh);
    assert!(db > 60.0, "a resumed GPU run differs from a fresh one: {db:.1} dB");
  }
}

#[test]
fn gpu_measures_grain_as_cpu() {
  let input = clip(4);
  // The first stage's results, which the measurements compare the frames with.
  let mut settings = Settings::new(2);
  settings.device = Device::Cpu;
  settings.basic_only = true;
  let mut denoiser = Denoiser::new(settings, 10).unwrap();
  let mut basic = Vec::new();
  for frame in &input {
    basic.extend(denoiser.push(frame.clone()).unwrap());
  }
  basic.extend(denoiser.finish().unwrap());
  let gpu = match Gpu::new(false) {
    Ok(gpu) => gpu,
    Err(e) => {
      eprintln!("skipped: {e:#}");
      return;
    }
  };
  let frame = |i: usize| MeasureFrame {
    noisy: &input[i].planes,
    basic: &basic[i].planes,
    noisy_key: FrameKey { index: i, version: 0 },
    basic_key: FrameKey { index: i, version: 1 },
  };
  let measured = gpu.measure(&frame(2), Some(&frame(1)), true, true, [12, 12]).unwrap();
  let (pairs, residual) = (measured.pairs.unwrap(), measured.residual.unwrap());
  let transforms = Transforms::new(16);
  let close = |what: &str, a: &[f32], b: &[f32]| {
    for (x, y) in a.iter().zip(b) {
      assert!((x - y).abs() <= 1e-3 * x.abs().max(1.0), "{what}: CPU {x}, GPU {y}");
    }
  };
  for p in 0..3 {
    let (a, b) = (&input[1].planes[p], &input[2].planes[p]);
    let (a_basic, b_basic) = (&basic[1].planes[p], &basic[2].planes[p]);
    // Chroma is matched on luma at its size.
    let scaled = (p > 0).then(|| {
      (
        basic[1].planes[0].downscale(a.width, a.height),
        basic[2].planes[0].downscale(a.width, a.height),
      )
    });
    let cpu = NoiseStats::pair_blocks(a, a_basic, b, b_basic, scaled.as_ref().map(|(x, y)| (x, y)), &transforms);
    assert_eq!(cpu.len(), pairs[p].len());
    for (i, (c, g)) in cpu.iter().zip(&pairs[p]).enumerate() {
      let what = format!("plane {p}, pair block {i}");
      close(&what, &c.spectrum, &g.spectrum);
      close(&what, &[c.distance, c.mean], &[g.distance, g.mean]);
    }
    let cpu = NoiseStats::residual_blocks(b, b_basic, &transforms);
    assert_eq!(cpu.len(), residual[p].len());
    for (i, (c, g)) in cpu.iter().zip(&residual[p]).enumerate() {
      let what = format!("plane {p}, residual block {i}");
      close(&what, &c.spectrum, &g.spectrum);
      close(&what, &[c.kept, c.dc], &[g.kept, g.dc]);
    }
  }
}
