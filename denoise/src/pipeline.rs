//! Streams frames through both stages.
//!
//! A frame leaves about 2R frames after it came in, R being the temporal radius.

use crate::bm3d::{self, FrameKey, Params, Stage, Window};
use crate::dct::Transforms;
use crate::gpu::{self, Gpu};
use crate::noise::{NoiseModel, NoiseStats};
use crate::plane::Frame;
use crate::plane::Plane;
use anyhow::Result;
use rayon::prelude::*;
use std::collections::BTreeMap;

/// Both stages remove grain best at this multiple of the measured grain deviation, the
/// best PSNR on the known grain clips.
const CALIBRATION: f32 = 1.2;

#[derive(Clone, Debug)]
pub struct Settings {
  pub hard: Params,
  pub wiener: Params,
  /// Multiplies the measured grain deviation.
  pub sigma_scale: f32,
  /// Skip the Wiener stage and return the basic estimate.
  pub basic_only: bool,
  /// Keep every returned frame's noise models for `take_models`.
  pub keep_models: bool,
  /// Multiplies the measured grain deviation of the chroma planes, on top of `sigma_scale`.
  pub chroma_scale: f32,
  /// Cap the grain measured between frames by the grain measured in single frames.
  pub single_frame_cap: bool,
  /// Match chroma blocks on the luma basic estimate when measuring grain between frames.
  pub chroma_luma_matching: bool,
  /// Measure where the time goes, for `timing_report`.
  pub timing: bool,
  pub device: Device,
}

/// Where the filtering runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
  /// The GPU when one opens, the CPU otherwise.
  Auto,
  /// The GPU, or an error.
  Gpu,
  /// The CPU, the reference the GPU is tested against.
  Cpu,
}

impl Settings {
  pub fn new(temporal_radius: usize) -> Self {
    Self {
      hard: Params::hard(temporal_radius),
      wiener: Params::wiener(temporal_radius),
      sigma_scale: 1.0,
      basic_only: false,
      keep_models: false,
      chroma_scale: 1.0,
      single_frame_cap: true,
      chroma_luma_matching: true,
      timing: false,
      device: Device::Auto,
    }
  }

  fn radius(&self) -> usize {
    self.hard.temporal_radius.max(self.wiener.temporal_radius)
  }
}

/// Planes being filtered: on their way through the GPU, or already done on the CPU.
enum Submitted {
  Gpu(gpu::Submission),
  Done(Vec<Plane>),
}

/// The noise models both stages used for one frame, one per plane.
pub struct FrameModels {
  pub index: usize,
  pub hard: Vec<NoiseModel>,
  pub wiener: Vec<NoiseModel>,
}

struct Noisy {
  frame: Frame,
  /// The grain measured on the noisy frame alone, which picture texture inflates.
  stats: Vec<NoiseStats>,
  /// The grain measured in what the basic estimate removed, once it exists.
  residual: Option<Vec<NoiseStats>>,
  /// The grain measured between this frame and the one before it.
  temporal: Option<Vec<NoiseStats>>,
}

/// Measures a frame's grain on its own, on any thread, for `Denoiser::push_measured`.
#[derive(Clone)]
pub struct Measurer {
  maxval: f32,
  transforms: std::sync::Arc<Transforms>,
}

impl Measurer {
  pub fn measure(&self, frame: Frame) -> Measured {
    let start = std::time::Instant::now();
    let stats = frame
      .planes
      .iter()
      .map(|p| NoiseStats::measure(p, self.maxval, &self.transforms))
      .collect();
    Measured {
      frame,
      stats,
      seconds: start.elapsed().as_secs_f64(),
    }
  }
}

/// A frame and its grain measured on its own, from `Measurer::measure`.
pub struct Measured {
  frame: Frame,
  stats: Vec<NoiseStats>,
  seconds: f64,
}

pub struct Denoiser {
  settings: Settings,
  maxval: f32,
  transforms: std::sync::Arc<Transforms>,
  noisy: BTreeMap<usize, Noisy>,
  basic: BTreeMap<usize, Frame>,
  models: BTreeMap<usize, Vec<NoiseModel>>,
  wiener_models: BTreeMap<usize, Vec<NoiseModel>>,
  /// Basic estimates made before any pair of frames had been measured.
  bootstrapped: Vec<usize>,
  /// How many times each kept basic estimate has been made.
  basic_versions: BTreeMap<usize, u32>,
  /// Noise models of returned frames, with `keep_models`.
  returned_models: Vec<FrameModels>,
  gpu: Option<Gpu>,
  /// Seconds spent on single frames, residuals, pairs and filter calls.
  seconds: std::cell::Cell<[f64; 4]>,
  /// Why the GPU is not used, with Device::Auto.
  gpu_note: Option<String>,
  read: usize,
  next_basic: usize,
  next_final: usize,
  /// A second stage left on the GPU between calls, so it always has work queued.
  in_flight: Option<(usize, Submitted)>,
  ended: bool,
}

impl Denoiser {
  /// Opens the GPU unless `settings.device` is Device::Cpu. With Device::Gpu, failing to
  /// open it is an error.
  pub fn new(settings: Settings, bit_depth: usize) -> Result<Self> {
    let max_group = settings.hard.max_group.max(settings.wiener.max_group);
    let supported = gpu::supports(&settings.hard) && gpu::supports(&settings.wiener);
    let (gpu, gpu_note) = match settings.device {
      Device::Cpu => (None, None),
      _ if !supported => {
        let note = "these settings are larger than the GPU filter supports".to_string();
        if settings.device == Device::Gpu {
          anyhow::bail!("{note}");
        }
        (None, Some(note))
      }
      Device::Gpu => (Some(Gpu::new(settings.timing)?), None),
      Device::Auto => match Gpu::new(settings.timing) {
        Ok(gpu) => (Some(gpu), None),
        Err(e) => (None, Some(format!("{e:#}"))),
      },
    };
    Ok(Self {
      settings,
      maxval: ((1u32 << bit_depth) - 1) as f32,
      transforms: std::sync::Arc::new(Transforms::new(max_group)),
      noisy: BTreeMap::new(),
      basic: BTreeMap::new(),
      models: BTreeMap::new(),
      wiener_models: BTreeMap::new(),
      bootstrapped: Vec::new(),
      basic_versions: BTreeMap::new(),
      returned_models: Vec::new(),
      gpu,
      gpu_note,
      seconds: std::cell::Cell::new([0.0; 4]),
      read: 0,
      next_basic: 0,
      next_final: 0,
      in_flight: None,
      ended: false,
    })
  }

  /// The GPU in use, or None on the CPU.
  pub fn gpu_name(&self) -> Option<&str> {
    self.gpu.as_ref().map(Gpu::name)
  }

  /// Why Device::Auto runs on the CPU.
  pub fn gpu_note(&self) -> Option<&str> {
    self.gpu_note.as_deref()
  }

  /// Starts filtering every plane of a frame for one stage. The CPU filters at once.
  fn submit_planes(&self, stage: Stage, windows: &[Window], models: &[NoiseModel], params: &Params) -> Result<Submitted> {
    let start = std::time::Instant::now();
    let out = match &self.gpu {
      Some(gpu) => {
        let jobs: Vec<gpu::Job> = windows
          .iter()
          .zip(models)
          .map(|(window, noise)| gpu::Job {
            stage,
            window,
            noise,
            params,
          })
          .collect();
        gpu.submit(&jobs, self.maxval).map(Submitted::Gpu)
      }
      None => Ok(Submitted::Done(
        windows
          .iter()
          .zip(models)
          .map(|(window, model)| bm3d::filter(stage, window, model, params, &self.transforms, self.maxval))
          .collect(),
      )),
    };
    self.add_seconds(3, start);
    out
  }

  fn collect(&self, submitted: Submitted) -> Result<Vec<Plane>> {
    let start = std::time::Instant::now();
    let out = match submitted {
      Submitted::Gpu(submission) => self.gpu.as_ref().expect("a GPU submission without a GPU").collect(submission),
      Submitted::Done(planes) => Ok(planes),
    };
    self.add_seconds(3, start);
    out
  }

  fn add_seconds(&self, slot: usize, start: std::time::Instant) {
    let mut seconds = self.seconds.get();
    seconds[slot] += start.elapsed().as_secs_f64();
    self.seconds.set(seconds);
  }

  /// Where the time went, with `timing`.
  pub fn timing_report(&self) -> String {
    let [single, residual, pairs, filter] = self.seconds.get();
    let mut out = format!(
      "Grain measured on single frames {single:.2} s, on residuals {residual:.2} s, between frames {pairs:.2} s\nFilter calls {filter:.2} s"
    );
    if let Some(stages) = self.gpu.as_ref().and_then(Gpu::kernel_seconds) {
      let busy: f64 = stages.iter().flatten().sum();
      out += &format!("\nGPU busy {busy:.2} s");
      for (name, [motion, find, groups, normalize]) in ["Hard threshold", "Wiener"].iter().zip(stages) {
        out += &format!(
          "\n{name} stage: motion fields {motion:.2} s, find_groups {find:.2} s, filter_groups {groups:.2} s, normalize {normalize:.2} s"
        );
      }
    }
    out
  }

  /// Takes the next frame and returns the frames that are finished, in order.
  pub fn push(&mut self, frame: Frame) -> Result<Vec<Frame>> {
    let measured = self.measurer().measure(frame);
    self.push_measured(measured)
  }

  /// What measures frames for `push_measured` on another thread.
  pub fn measurer(&self) -> Measurer {
    Measurer {
      maxval: self.maxval,
      transforms: self.transforms.clone(),
    }
  }

  /// Like `push`, for a frame measured by this denoiser's `measurer`.
  pub fn push_measured(&mut self, measured: Measured) -> Result<Vec<Frame>> {
    let Measured { frame, stats, seconds } = measured;
    let mut all = self.seconds.get();
    all[0] += seconds;
    self.seconds.set(all);
    self.noisy.insert(
      self.read,
      Noisy {
        frame,
        stats,
        residual: None,
        temporal: None,
      },
    );
    self.read += 1;
    self.advance()
  }

  /// Ends the stream and returns the remaining frames.
  pub fn finish(&mut self) -> Result<Vec<Frame>> {
    self.ended = true;
    self.advance()
  }

  /// The noise models of the frames returned since the last call, with `keep_models`.
  pub fn take_models(&mut self) -> Vec<FrameModels> {
    std::mem::take(&mut self.returned_models)
  }

  /// The multiple of the measured grain deviation a plane is denoised for.
  fn scale(&self, plane: usize) -> f32 {
    let chroma = if plane > 0 { self.settings.chroma_scale } else { 1.0 };
    self.settings.sigma_scale * CALIBRATION * chroma
  }

  fn span(&self, index: usize, radius: usize, available: usize) -> (usize, usize) {
    (index.saturating_sub(radius), (index + radius).min(available - 1))
  }

  /// Whether the next frame's second stage has every basic estimate of its window.
  fn final_ready(&self) -> bool {
    let radius = self.settings.radius();
    let basic_done = self.next_basic == self.read && self.ended;
    self.next_final < self.next_basic && (self.next_final + radius < self.next_basic || basic_done)
  }

  // Keeps the GPU fed. A first stage is queued behind the second stage in flight, the next
  // second stage behind it, and the CPU measures grain while the GPU works.
  fn advance(&mut self) -> Result<Vec<Frame>> {
    let hard_radius = self.settings.hard.temporal_radius;
    let radius = self.settings.radius();
    let mut out = Vec::new();
    loop {
      let mut progressed = false;
      if self.next_basic < self.read && (self.next_basic + radius < self.read || self.ended) {
        let n = self.next_basic;
        // Only pairs before this frame are measured yet.
        let (models, temporal) = self.hard_models(n, n.saturating_sub(hard_radius), n);
        let basic = self.submit_basic(n, &models)?;
        self.finish_in_flight(&mut out)?;
        // Not while a frame waits to be redone, which could change its window.
        if self.bootstrapped.is_empty() && self.final_ready() {
          let index = self.next_final;
          self.in_flight = Some((index, self.submit_final(index)?));
        }
        self.finish_basic(n, models, basic)?;
        self.next_basic += 1;
        self.measure_residual(n);
        self.measure_pair(n);
        // Frames done again change basic estimates the second stage may read.
        if !temporal || !self.bootstrapped.is_empty() || self.noisy[&n].temporal.as_ref().is_some_and(|t| t[0].is_empty()) {
          self.finish_in_flight(&mut out)?;
        }
        self.redo_after(n, temporal)?;
        progressed = true;
      }
      if !progressed && self.in_flight.is_some() && self.ended {
        self.finish_in_flight(&mut out)?;
        progressed = true;
      }
      // At the end of the stream there is no first stage left to queue behind, and while a
      // frame waits to be redone the queued path is closed. Frames must not pile up then.
      let waiting = !self.bootstrapped.is_empty();
      if !progressed && (self.ended || waiting) && self.in_flight.is_none() && self.final_ready() {
        let index = self.next_final;
        let submitted = self.submit_final(index)?;
        let frame = self.finish_final(submitted)?;
        out.push(self.returned(index, frame));
        progressed = true;
      }
      if !progressed {
        return Ok(out);
      }
    }
  }

  /// Collects the second stage on the GPU, if any.
  fn finish_in_flight(&mut self, out: &mut Vec<Frame>) -> Result<()> {
    if let Some((index, submitted)) = self.in_flight.take() {
      let frame = self.finish_final(submitted)?;
      out.push(self.returned(index, frame));
    }
    Ok(())
  }

  /// Books a finished frame as returned and lets go of what no later frame needs.
  fn returned(&mut self, index: usize, frame: Frame) -> Frame {
    debug_assert_eq!(index, self.next_final);
    let radius = self.settings.radius();
    if self.settings.keep_models {
      self.returned_models.push(FrameModels {
        index,
        hard: self.models[&index].clone(),
        wiener: self.wiener_models[&index].clone(),
      });
    }
    self.next_final += 1;
    // Keep at least the frame just returned, as the next pair needs it.
    let keep_from = self.next_final.saturating_sub(radius.max(1));
    self.noisy.retain(|&i, _| i >= keep_from);
    self.basic.retain(|&i, _| i >= keep_from);
    self.basic_versions.retain(|&i, _| i >= keep_from);
    self.models.retain(|&i, _| i + radius >= self.next_final);
    self.wiener_models.retain(|&i, _| i + radius >= self.next_final);
    frame
  }

  /// Stage 1's noise models: from the pairs `first` to `last`, or from single frames when
  /// no pair is measured yet. The flag tells which.
  fn hard_models(&self, index: usize, first: usize, last: usize) -> (Vec<NoiseModel>, bool) {
    let radius = self.settings.hard.temporal_radius;
    let (w_first, w_last) = self.span(index, radius, self.read);
    let planes = self.noisy[&index].frame.planes.len();
    let mut temporal = true;
    let models = (0..planes)
      .map(|p| {
        let pairs = self.scene_pairs(index, first, last, p);
        let model = NoiseModel::combine(&pairs, self.maxval, self.scale(p));
        let (lo, hi) = self.scene_span(index, w_first, w_last);
        let single = self.single_frame_model(lo, hi, p);
        if model.is_empty() {
          temporal = false;
          single
        } else {
          self.cap(model, &single)
        }
      })
      .collect();
    (models, temporal)
  }

  /// The frames from `first` to `last` in the same scene as `index`.
  fn scene_span(&self, index: usize, first: usize, last: usize) -> (usize, usize) {
    let cut = |i: usize| {
      self
        .noisy
        .get(&i)
        .and_then(|n| n.temporal.as_ref())
        .is_some_and(|t| t[0].is_empty())
    };
    // Pair i joins frames i - 1 and i, so a cut at pair i starts a scene at frame i.
    let lo = (first + 1..=index).rev().find(|&i| cut(i)).unwrap_or(first);
    let hi = (index + 1..=last).find(|&i| cut(i)).map_or(last, |i| i - 1);
    (lo, hi)
  }

  /// The grain measured on the noisy frames from `first` to `last` alone.
  fn single_frame_model(&self, first: usize, last: usize, plane: usize) -> NoiseModel {
    let stats: Vec<&NoiseStats> = (first..=last).filter_map(|i| self.noisy.get(&i).map(|n| &n.stats[plane])).collect();
    NoiseModel::combine(&stats, self.maxval, self.scale(plane))
  }

  /// The grain measured between frames, capped by the grain measured in single frames.
  /// Motion reads high in the first, texture in the second.
  fn cap(&self, temporal: NoiseModel, single: &NoiseModel) -> NoiseModel {
    if self.settings.single_frame_cap {
      temporal.min(single)
    } else {
      temporal
    }
  }

  /// The measured pairs from `first` to `last` in the same scene as `index`. Pair i joins
  /// frames i - 1 and i.
  fn scene_pairs(&self, index: usize, first: usize, last: usize, plane: usize) -> Vec<&NoiseStats> {
    let pair = |i: usize| self.noisy.get(&i).and_then(|n| n.temporal.as_ref());
    let mut out = Vec::new();
    for i in (first..=index).rev() {
      match pair(i) {
        Some(t) if t[0].is_empty() => break,
        Some(t) => out.push(&t[plane]),
        None => {}
      }
    }
    for i in index + 1..=last {
      match pair(i) {
        Some(t) if t[0].is_empty() => break,
        Some(t) => out.push(&t[plane]),
        None => {}
      }
    }
    out
  }

  /// Runs stage 1 on a frame with these models and keeps the result.
  fn run_basic(&mut self, index: usize, models: Vec<NoiseModel>) -> Result<()> {
    let submitted = self.submit_basic(index, &models)?;
    self.finish_basic(index, models, submitted)
  }

  /// Keeps a frame's stage 1 result once it is done.
  fn finish_basic(&mut self, index: usize, models: Vec<NoiseModel>, submitted: Submitted) -> Result<()> {
    let planes = self.collect(submitted)?;
    self.models.insert(index, models);
    self.basic.insert(index, Frame { planes });
    // A redone basic estimate is a new version, so the GPU uploads it again.
    *self.basic_versions.entry(index).or_insert(0) += 1;
    Ok(())
  }

  /// Starts stage 1 on a frame with these models.
  fn submit_basic(&self, index: usize, models: &[NoiseModel]) -> Result<Submitted> {
    let radius = self.settings.hard.temporal_radius;
    let (first, last) = self.span(index, radius, self.read);
    let windows: Vec<Window> = (0..models.len())
      .map(|p| {
        let noisy: Vec<_> = (first..=last).map(|i| &self.noisy[&i].frame.planes[p]).collect();
        let keys: Vec<FrameKey> = (first..=last).map(|i| FrameKey { index: i, version: 0 }).collect();
        Window {
          guide: noisy.clone(),
          noisy,
          center: index - first,
          plane: p,
          noisy_keys: keys.clone(),
          guide_keys: keys,
          motion_guide: (p > 0).then(|| (first..=last).map(|i| &self.noisy[&i].frame.planes[0]).collect()),
        }
      })
      .collect();
    self.submit_planes(Stage::Hard, &windows, models, &self.settings.hard)
  }

  /// Measures the grain in what a frame's basic estimate removed.
  fn measure_residual(&mut self, index: usize) {
    let start = std::time::Instant::now();
    let residual = self.noisy[&index]
      .frame
      .planes
      .par_iter()
      .zip(&self.basic[&index].planes)
      .map(|(n, b)| NoiseStats::measure_residual(n, b, self.maxval, &self.transforms))
      .collect();
    self.add_seconds(1, start);
    self.noisy.get_mut(&index).unwrap().residual = Some(residual);
  }

  /// Measures the grain between a frame and the one before it.
  fn measure_pair(&mut self, index: usize) {
    let Some(previous) = index.checked_sub(1) else { return };
    let (Some(a), Some(a_basic), Some(b), Some(b_basic)) = (
      self.noisy.get(&previous),
      self.basic.get(&previous),
      self.noisy.get(&index),
      self.basic.get(&index),
    ) else {
      return;
    };
    let start = std::time::Instant::now();
    // Chroma is matched on the luma basic estimates at its size, scaled once for each size.
    let planes = b.frame.planes.len();
    let size = |p: usize| (b.frame.planes[p].width, b.frame.planes[p].height);
    let mut sizes: Vec<(usize, usize)> = (1..planes).map(size).collect();
    sizes.dedup();
    if !self.settings.chroma_luma_matching {
      sizes.clear();
    }
    let scaled: Vec<(Plane, Plane)> = sizes
      .par_iter()
      .map(|&(w, h)| rayon::join(|| a_basic.planes[0].downscale(w, h), || b_basic.planes[0].downscale(w, h)))
      .collect();
    // The planes are measured at once, as each alone leaves threads idle.
    let temporal = (0..planes)
      .into_par_iter()
      .map(|p| {
        let matching = sizes
          .iter()
          .position(|&s| p > 0 && s == size(p))
          .map(|i| (&scaled[i].0, &scaled[i].1));
        NoiseStats::measure_temporal(
          &a.frame.planes[p],
          &a_basic.planes[p],
          &b.frame.planes[p],
          &b_basic.planes[p],
          matching,
          self.maxval,
          &self.transforms,
        )
      })
      .collect();
    self.add_seconds(2, start);
    self.noisy.get_mut(&index).unwrap().temporal = Some(temporal);
  }

  /// Redoes frames made before their scene had a measured pair, once it has one.
  fn redo_after(&mut self, index: usize, temporal: bool) -> Result<()> {
    let radius = self.settings.hard.temporal_radius;
    let cut = self.noisy[&index].temporal.as_ref().is_some_and(|t| t[0].is_empty());
    if !temporal || cut {
      self.bootstrapped.push(index);
    }
    let measured = self.noisy[&index].temporal.as_ref().is_some_and(|t| !t[0].is_empty());
    if measured && !self.bootstrapped.is_empty() {
      let redo: Vec<usize> = std::mem::take(&mut self.bootstrapped)
        .into_iter()
        .filter(|&j| j >= self.next_final)
        .collect();
      for &j in &redo {
        let (models, _) = self.hard_models(j, j.saturating_sub(radius), index);
        self.run_basic(j, models)?;
        self.measure_residual(j);
      }
      for &j in &redo {
        self.measure_pair(j);
        self.measure_pair(j + 1);
      }
    }
    Ok(())
  }

  /// Starts a frame's second stage.
  fn submit_final(&mut self, index: usize) -> Result<Submitted> {
    if self.settings.basic_only {
      self.wiener_models.insert(index, self.models[&index].clone());
      return Ok(Submitted::Done(self.basic[&index].planes.clone()));
    }
    let radius = self.settings.wiener.temporal_radius;
    let (first, last) = self.span(index, radius, self.next_basic);
    // From the pairs, else from the residuals, else the hard stage's model.
    let models: Vec<NoiseModel> = self.models[&index]
      .iter()
      .enumerate()
      .map(|(p, fallback)| {
        let temporal = self.scene_pairs(index, first, last, p);
        let residual: Vec<&NoiseStats> = (first..=last)
          .filter_map(|i| self.noisy[&i].residual.as_ref().map(|r| &r[p]))
          .collect();
        let residual = NoiseModel::combine(&residual, self.maxval, self.scale(p)).or(fallback);
        let temporal = NoiseModel::combine(&temporal, self.maxval, self.scale(p));
        if temporal.is_empty() {
          residual
        } else {
          let (lo, hi) = self.scene_span(index, first, last);
          self.cap(temporal, &self.single_frame_model(lo, hi, p))
        }
      })
      .collect();
    let windows: Vec<Window> = (0..models.len())
      .map(|p| Window {
        noisy: (first..=last).map(|i| &self.noisy[&i].frame.planes[p]).collect(),
        guide: (first..=last).map(|i| &self.basic[&i].planes[p]).collect(),
        center: index - first,
        plane: p,
        noisy_keys: (first..=last).map(|i| FrameKey { index: i, version: 0 }).collect(),
        guide_keys: (first..=last)
          .map(|i| FrameKey {
            index: i,
            version: self.basic_versions[&i],
          })
          .collect(),
        motion_guide: (p > 0).then(|| (first..=last).map(|i| &self.basic[&i].planes[0]).collect()),
      })
      .collect();
    let submitted = self.submit_planes(Stage::Wiener, &windows, &models, &self.settings.wiener)?;
    drop(windows);
    self.wiener_models.insert(index, models);
    Ok(submitted)
  }

  /// Waits for a frame's second stage.
  fn finish_final(&self, submitted: Submitted) -> Result<Frame> {
    Ok(Frame {
      planes: self.collect(submitted)?,
    })
  }
}
