//! rdbench compares a denoised video with a reference.
//!
//!     rdbench compare --reference REF --distorted OUT [--noisy IN] [--crop W:H:X:Y] [--ssim-every N]
//!
//! Every input is decoded by ffmpeg to 10 bit 4:2:0, so 8 and 10 bit videos compare on one
//! scale. The result is one JSON object on standard output.
//!
//! With --noisy, REF is the clean video, IN is REF with known grain added and OUT is IN after
//! denoising. Then the grain the denoiser removed (IN - OUT) is split into the part that
//! follows the true grain (IN - REF) and the rest:
//!
//! - `grain_removed` is the share of the true grain that was removed. 1 is ideal, below 1
//!   leaves grain behind, above 1 overshoots.
//! - `collateral_psnr` is the PSNR of the rest, the picture removed that was not grain:
//!   lost detail, smearing and artefacts. Higher is better.

mod residual;

use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use serde_json::json;
use ssimulacra2::{ColorPrimaries, Frame as VFrame, MatrixCoefficients, TransferCharacteristic, Yuv, YuvConfig};
use std::process::{Child, ChildStdout, Command, Stdio};
use v_frame::pixel::ChromaSampling;

const BIT_DEPTH: u32 = 10;
const PEAK: f64 = ((1 << BIT_DEPTH) - 1) as f64;
// Frames read at once, and threads scoring them. 8 on 4 threads stays near 1 GB.
const BATCH: usize = 8;
const THREADS: usize = 4;

struct Picture {
  // Y, U, V samples.
  planes: [Vec<u16>; 3],
}

struct Source {
  child: Child,
  decoder: y4m::Decoder<ChildStdout>,
}

impl Source {
  fn open(path: &str, crop: Option<&str>) -> Result<Self> {
    let mut command = Command::new("ffmpeg");
    command.args(["-v", "error", "-nostdin", "-i", path, "-map", "0:v:0"]);
    if let Some(crop) = crop {
      command.args(["-vf", &format!("crop={crop}")]);
    }
    command.args(["-pix_fmt", "yuv420p10le", "-strict", "-1", "-f", "yuv4mpegpipe", "-"]);
    let mut child = command
      .stdout(Stdio::piped())
      .stdin(Stdio::null())
      .spawn()
      .context("could not start ffmpeg")?;
    let stdout = child.stdout.take().unwrap();
    let decoder = y4m::decode(stdout).map_err(|e| anyhow::anyhow!("{path}: no y4m from ffmpeg: {e:?}"))?;
    Ok(Self { child, decoder })
  }

  fn size(&self) -> (usize, usize) {
    (self.decoder.get_width(), self.decoder.get_height())
  }

  fn next(&mut self) -> Option<Picture> {
    let frame = self.decoder.read_frame().ok()?;
    let samples = |bytes: &[u8]| bytes.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect::<Vec<_>>();
    Some(Picture {
      planes: [
        samples(frame.get_y_plane()),
        samples(frame.get_u_plane()),
        samples(frame.get_v_plane()),
      ],
    })
  }
}

impl Drop for Source {
  fn drop(&mut self) {
    let _ = self.child.kill();
    let _ = self.child.wait();
  }
}

#[derive(Clone, Copy, Default)]
struct PlaneSums {
  // Squared error of the distorted picture against the reference.
  error: f64,
  // Squared error of the noisy picture against the reference.
  noisy_error: f64,
  // grain = noisy - reference, removed = noisy - distorted.
  grain_grain: f64,
  grain_removed: f64,
  removed_removed: f64,
  samples: f64,
}

impl PlaneSums {
  fn add(&mut self, other: &PlaneSums) {
    self.error += other.error;
    self.noisy_error += other.noisy_error;
    self.grain_grain += other.grain_grain;
    self.grain_removed += other.grain_removed;
    self.removed_removed += other.removed_removed;
    self.samples += other.samples;
  }
}

// Identical pictures get MAX_PSNR, since JSON has no infinity.
const MAX_PSNR: f64 = 100.0;

fn psnr(squared_error: f64, samples: f64) -> f64 {
  if squared_error <= 0.0 {
    return MAX_PSNR;
  }
  (10.0 * (PEAK * PEAK * samples / squared_error).log10()).min(MAX_PSNR)
}

fn plane_sums(reference: &[u16], distorted: &[u16], noisy: Option<&[u16]>) -> PlaneSums {
  let mut sums = PlaneSums {
    samples: reference.len() as f64,
    ..Default::default()
  };
  for (i, (&r, &d)) in reference.iter().zip(distorted).enumerate() {
    let (r, d) = (r as f64, d as f64);
    sums.error += (d - r) * (d - r);
    if let Some(noisy) = noisy {
      let n = noisy[i] as f64;
      let (grain, removed) = (n - r, n - d);
      sums.noisy_error += grain * grain;
      sums.grain_grain += grain * grain;
      sums.grain_removed += grain * removed;
      sums.removed_removed += removed * removed;
    }
  }
  sums
}

fn yuv(picture: &Picture, width: usize, height: usize) -> Result<Yuv<u16>> {
  let mut frame = VFrame::new_with_padding(width, height, ChromaSampling::Cs420, 0);
  let widths = [width, width.div_ceil(2), width.div_ceil(2)];
  for (plane, (samples, w)) in frame.planes.iter_mut().zip(picture.planes.iter().zip(widths)) {
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    plane.copy_from_raw_u8(&bytes, w * 2, 2);
  }
  let config = YuvConfig {
    bit_depth: BIT_DEPTH as u8,
    subsampling_x: 1,
    subsampling_y: 1,
    full_range: false,
    matrix_coefficients: MatrixCoefficients::BT709,
    transfer_characteristics: TransferCharacteristic::BT1886,
    color_primaries: ColorPrimaries::BT709,
  };
  Yuv::new(frame, config).map_err(|e| anyhow::anyhow!("bad picture for SSIMULACRA2: {e:?}"))
}

fn ssimulacra2(reference: &Picture, distorted: &Picture, width: usize, height: usize) -> Result<f64> {
  let score = ssimulacra2::compute_frame_ssimulacra2(yuv(reference, width, height)?, yuv(distorted, width, height)?)
    .map_err(|e| anyhow::anyhow!("SSIMULACRA2 failed: {e:?}"))?;
  Ok(score)
}

struct Options {
  reference: String,
  distorted: String,
  noisy: Option<String>,
  crop: Option<String>,
  ssim_every: usize,
}

fn parse(args: &[String]) -> Result<Options> {
  let mut options = Options {
    reference: String::new(),
    distorted: String::new(),
    noisy: None,
    crop: None,
    ssim_every: 1,
  };
  let mut iter = args.iter();
  while let Some(arg) = iter.next() {
    let mut value = || iter.next().cloned().with_context(|| format!("{arg} needs a value"));
    match arg.as_str() {
      "--reference" => options.reference = value()?,
      "--distorted" => options.distorted = value()?,
      "--noisy" => options.noisy = Some(value()?),
      "--crop" => options.crop = Some(value()?),
      "--ssim-every" => options.ssim_every = value()?.parse::<usize>()?.max(1),
      other => bail!("unknown option {other}"),
    }
  }
  if options.reference.is_empty() || options.distorted.is_empty() {
    bail!("--reference and --distorted are required");
  }
  Ok(options)
}

fn compare(options: &Options) -> Result<serde_json::Value> {
  let crop = options.crop.as_deref();
  let mut reference = Source::open(&options.reference, crop)?;
  let mut distorted = Source::open(&options.distorted, crop)?;
  let mut noisy = options.noisy.as_deref().map(|path| Source::open(path, crop)).transpose()?;
  let (width, height) = reference.size();
  if distorted.size() != (width, height) || noisy.as_ref().is_some_and(|n| n.size() != (width, height)) {
    bail!("the videos differ in size");
  }

  let mut sums = [PlaneSums::default(); 3];
  let mut scores = Vec::new();
  let mut frames = 0usize;
  loop {
    let mut batch = Vec::with_capacity(BATCH);
    while batch.len() < BATCH {
      let (Some(r), Some(d)) = (reference.next(), distorted.next()) else {
        break;
      };
      let n = match noisy.as_mut() {
        Some(source) => Some(source.next().context("the noisy video ended early")?),
        None => None,
      };
      batch.push((frames + batch.len(), r, d, n));
    }
    if batch.is_empty() {
      break;
    }
    frames += batch.len();
    let results: Vec<([PlaneSums; 3], Option<f64>)> = batch
      .par_iter()
      .map(|(index, r, d, n)| {
        let planes = std::array::from_fn(|p| plane_sums(&r.planes[p], &d.planes[p], n.as_ref().map(|n| n.planes[p].as_slice())));
        let score = (index % options.ssim_every == 0)
          .then(|| ssimulacra2(r, d, width, height))
          .transpose()?;
        Ok((planes, score))
      })
      .collect::<Result<_>>()?;
    for (planes, score) in results {
      for (total, plane) in sums.iter_mut().zip(&planes) {
        total.add(plane);
      }
      scores.extend(score);
    }
  }
  if frames == 0 {
    bail!("no frames could be read");
  }
  // A shorter distorted video would silently score only the frames it has.
  if reference.next().is_some() || distorted.next().is_some() {
    bail!("the videos differ in length");
  }

  let mut sorted = scores.clone();
  sorted.sort_by(f64::total_cmp);
  let mean = scores.iter().sum::<f64>() / scores.len() as f64;
  let low = sorted[(sorted.len() as f64 * 0.05) as usize];
  let names = ["y", "u", "v"];
  let per_plane =
    |f: &dyn Fn(&PlaneSums) -> f64| serde_json::Value::Object(names.iter().zip(&sums).map(|(n, s)| (n.to_string(), json!(f(s)))).collect());

  let mut result = json!({
    "frames": frames,
    "width": width,
    "height": height,
    "psnr": per_plane(&|s| psnr(s.error, s.samples)),
    "ssimulacra2": { "mean": mean, "p5": low, "frames": scores.len() },
  });
  if noisy.is_some() {
    result["psnr_noisy"] = per_plane(&|s| psnr(s.noisy_error, s.samples));
    // A plane without grain has no share to remove, and everything removed from it is collateral.
    result["grain_removed"] = per_plane(&|s| {
      if s.grain_grain > 0.0 {
        s.grain_removed / s.grain_grain
      } else {
        f64::NAN
      }
    });
    // What is left of the removed picture after taking out its projection on the grain.
    result["collateral_psnr"] = per_plane(&|s| {
      let along_grain = if s.grain_grain > 0.0 {
        s.grain_removed * s.grain_removed / s.grain_grain
      } else {
        0.0
      };
      psnr((s.removed_removed - along_grain).max(0.0), s.samples)
    });
  }
  Ok(result)
}

// A reader that stops early, such as `head`, is not an error.
fn print_json(value: &serde_json::Value) -> Result<()> {
  use std::io::Write;
  match writeln!(std::io::stdout().lock(), "{value}") {
    Err(error) if error.kind() != std::io::ErrorKind::BrokenPipe => Err(error.into()),
    _ => Ok(()),
  }
}

fn main() -> Result<()> {
  // Fewer threads than cores keep memory low.
  let _ = rayon::ThreadPoolBuilder::new().num_threads(THREADS).build_global();
  let args: Vec<String> = std::env::args().skip(1).collect();
  match args.first().map(String::as_str) {
    Some("compare") => {
      let options = parse(&args[1..])?;
      let result = compare(&options)?;
      print_json(&result)?;
      Ok(())
    }
    Some("residual") => {
      let options = parse(&args[1..])?;
      let result = residual::residual(&options.reference, &options.distorted, options.crop.as_deref())?;
      print_json(&result)?;
      Ok(())
    }
    _ => {
      eprintln!("usage: rdbench residual --reference SOURCE --distorted DENOISED [--crop W:H:X:Y]");
      eprintln!("usage: rdbench compare --reference REF --distorted OUT [--noisy IN] [--crop W:H:X:Y] [--ssim-every N]");
      std::process::exit(2);
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn removing_exactly_the_grain_scores_perfectly() {
    let reference = vec![500u16, 510, 520, 530];
    let grain = [3i32, -4, 2, -1];
    let noisy: Vec<u16> = reference.iter().zip(grain).map(|(&r, g)| (r as i32 + g) as u16).collect();
    let sums = plane_sums(&reference, &reference, Some(&noisy));
    assert_eq!(sums.grain_removed / sums.grain_grain, 1.0);
    assert_eq!(
      sums.removed_removed - sums.grain_removed * sums.grain_removed / sums.grain_grain,
      0.0
    );
  }

  #[test]
  fn half_the_grain_left_scores_one_half() {
    let reference = vec![500u16; 4];
    let noisy = vec![504u16, 496, 504, 496];
    let distorted = vec![502u16, 498, 502, 498];
    let sums = plane_sums(&reference, &distorted, Some(&noisy));
    assert_eq!(sums.grain_removed / sums.grain_grain, 0.5);
  }
}
