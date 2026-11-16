//! The collaborative filter of `bm3d` on the GPU, through wgpu: Vulkan, Metal or DX12.
//!
//! `Gpu::filter` finds the same groups as `bm3d::filter`, and the results are about 70 dB
//! apart. Frames stay on the GPU in rings of slots and are uploaded once.

use crate::bm3d::{FrameKey, Params, Stage, TOP_K, Window};
use crate::dct::{AREA, N, basis};
use crate::noise::{BANDS, Measures, NoiseModel};
use crate::plane::Plane;
use anyhow::{Context, Result, bail};
use std::cell::RefCell;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

mod measure;
pub use measure::MeasureFrame;

const MAX_GROUP: usize = 16;
// Per reference in the groups buffer: the count, then up to MAX_GROUP matches.
const GROUP_STRIDE: usize = 17;
// Reference patches per workgroup side in find_groups.
const TILE_REFS: usize = 4;
const DISPATCH_X: u32 = 4096;
const WORKGROUP: u32 = 64;
// The largest number of frames a window may hold, and so a ring.
const MAX_FRAMES: usize = 63;
// The most planes one submission holds.
const MAX_JOBS: usize = 3;
// Timestamps per plane: around the motion fields and each of the three filter passes.
const STAMPS: u32 = 8;
// The largest region side find_groups searches.
const MAX_TILE_SIDE: usize = 40;

/// The accumulator with float atomics.
const FLOAT_ACCUMULATOR: &str = "
@group(0) @binding(4) var<storage, read_write> accum: array<atomic<f32>>;
fn atomic_add(i: u32, v: f32) { atomicAdd(&accum[i], v); }
fn accum_load(i: u32) -> f32 { return atomicLoad(&accum[i]); }
";

/// The accumulator without them: a compare and swap loop on the bits.
const BITS_ACCUMULATOR: &str = "
@group(0) @binding(4) var<storage, read_write> accum: array<atomic<u32>>;
fn atomic_add(i: u32, v: f32) {
  var old = atomicLoad(&accum[i]);
  loop {
    let result = atomicCompareExchangeWeak(&accum[i], old, bitcast<u32>(bitcast<f32>(old) + v));
    if (result.exchanged) {
      break;
    }
    old = result.old_value;
  }
}
fn accum_load(i: u32) -> f32 { return bitcast<f32>(atomicLoad(&accum[i])); }
";

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
  width: u32,
  height: u32,
  frames: u32,
  center: u32,
  refs_x: u32,
  refs_y: u32,
  search_radius: u32,
  predict_radius: u32,
  temporal_radius: u32,
  per_frame: u32,
  min_shift: u32,
  max_group: u32,
  stage: u32,
  lambda: f32,
  distance_factor: f32,
  distance_floor: f32,
  maxval: f32,
  plane_size: u32,
  dispatch_x: u32,
  slots: u32,
  motion: u32,
  field_blocks_x: u32,
  field_blocks_y: u32,
  motion_scale_x: u32,
  motion_scale_y: u32,
  half_width: u32,
  half_height: u32,
  half_range: u32,
  field_map: u32,
  half_jobs: u32,
  half_count: u32,
  field_jobs: u32,
  field_count: u32,
}

/// The luma planes averaged 2x2 and the motion fields between them, kept between calls.
/// The field from a frame's slot s goes to slot 2s backward and 2s + 1 forward.
struct MotionCache {
  half_planes: wgpu::Buffer,
  fields: wgpu::Buffer,
  half_keys: Vec<Option<FrameKey>>,
  field_keys: Vec<Option<(FrameKey, FrameKey)>>,
  /// Blocks across and down, the luma size and the slots the buffers were made for.
  shape: (usize, usize, usize, usize, usize),
}

/// What a frame's luma plane leaves for its chroma planes.
#[derive(Clone, Default)]
struct FieldUse {
  kind: Option<Kind>,
  blocks_x: usize,
  blocks_y: usize,
  luma_width: usize,
  luma_height: usize,
  map: Vec<f32>,
}

/// Frames of one plane kept on the GPU, each in slot index modulo the slot count.
struct Ring {
  buffer: wgpu::Buffer,
  plane_size: usize,
  keys: Vec<Option<FrameKey>>,
}

/// The buffers a plane's calls reuse.
struct Work {
  plane_size: usize,
  refs: usize,
  groups: wgpu::Buffer,
  accum: wgpu::Buffer,
  output: wgpu::Buffer,
  readback: wgpu::Buffer,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Kind {
  Noisy,
  Basic,
}

/// GPU time per entry point, from timestamps around each pass.
struct Timing {
  /// STAMPS for each of MAX_JOBS planes, of the hard threshold stage, then the Wiener stage.
  queries: wgpu::QuerySet,
  /// One of each for each stage, as one submission of each may be outstanding.
  resolved: [wgpu::Buffer; 2],
  readback: [wgpu::Buffer; 2],
  period: f64,
  /// Nanoseconds per entry point, for the hard threshold stage, then the Wiener stage.
  totals: RefCell<[[f64; 4]; 2]>,
}

/// The entry points, built for one side of the region find_groups searches.
struct Pipelines {
  half_plane: wgpu::ComputePipeline,
  motion: wgpu::ComputePipeline,
  find_groups: wgpu::ComputePipeline,
  filter: wgpu::ComputePipeline,
  normalize: wgpu::ComputePipeline,
}

pub struct Gpu {
  device: wgpu::Device,
  queue: wgpu::Queue,
  layout: wgpu::BindGroupLayout,
  pipeline_layout: wgpu::PipelineLayout,
  float_atomics: bool,
  /// Built on first use for each region side.
  pipelines: RefCell<HashMap<usize, std::rc::Rc<Pipelines>>>,
  /// Kaiser window, 8x8 DCT and the group DCT of every length.
  tables: Vec<f32>,
  name: String,
  rings: RefCell<HashMap<(usize, Kind), Ring>>,
  /// For each plane and stage (false for the hard threshold stage, true for Wiener).
  work: RefCell<HashMap<(usize, bool), Work>>,
  timing: Option<Timing>,
  /// The motion caches, one for each kind of guide.
  motion: RefCell<HashMap<Kind, MotionCache>>,
  /// The fields the last luma plane read.
  field_use: RefCell<FieldUse>,
  /// A buffer bound where the stage has no fields.
  empty: wgpu::Buffer,
  /// Whether a submission of each stage is still outstanding.
  outstanding: std::cell::Cell<[bool; 2]>,
  /// The grain measurement's entry points, built on first use, and each plane's buffers.
  measuring: RefCell<Option<measure::Measuring>>,
  measure_work: RefCell<HashMap<usize, measure::MeasureWork>>,
}

/// Whether the GPU filter can run these parameters. The shader runs without bounds checks
/// in release builds, so these limits keep its reads in range.
pub fn supports(params: &Params) -> bool {
  params.motion_range <= 32
    && params.max_group <= MAX_GROUP
    && params.max_group >= 2
    && params.per_frame <= TOP_K
    && params.predict_count == 1
    && params.temporal_radius <= 31
    && params.search_radius <= 7
    && params.predict_radius <= 7
    && params.step >= 1
    && params.step <= 6
}

impl Gpu {
  /// Opens the first GPU wgpu finds, preferring a discrete one.
  pub fn new(timing: bool) -> Result<Self> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
      power_preference: wgpu::PowerPreference::HighPerformance,
      force_fallback_adapter: false,
      compatible_surface: None,
      ..Default::default()
    }))
    .context("no GPU found")?;
    let info = adapter.get_info();
    if info.device_type == wgpu::DeviceType::Cpu {
      bail!("only a software GPU ({}) was found", info.name);
    }
    let available = adapter.limits();
    // find_groups keeps two regions and its lanes' positions in workgroup memory, and
    // filter_groups the sums of a region and one patch's DCT.
    let find_memory = 2 * MAX_TILE_SIDE * MAX_TILE_SIDE * 4 + 512 * 4 + 2 * 16 * 4 + 16;
    let filter_memory = 2 * MAX_TILE_SIDE * MAX_TILE_SIDE * 4 + (3 * 64 + 2 * 17) * 4 + 16;
    let workgroup_memory = find_memory.max(filter_memory) as u32;
    if available.max_compute_workgroup_storage_size < workgroup_memory {
      bail!("{} has too little workgroup memory for the GPU filter", info.name);
    }
    // Nine storage buffers.
    if available.max_storage_buffers_per_shader_stage < 9 {
      bail!("{} has too few storage buffers for the GPU filter", info.name);
    }
    let limits = wgpu::Limits {
      max_storage_buffers_per_shader_stage: 9,
      max_compute_workgroup_storage_size: workgroup_memory
        .next_power_of_two()
        .min(available.max_compute_workgroup_storage_size),
      max_storage_buffer_binding_size: available.max_storage_buffer_binding_size,
      max_buffer_size: available.max_buffer_size,
      ..wgpu::Limits::default()
    };
    let float_atomics = adapter.features().contains(wgpu::Features::SHADER_FLOAT32_ATOMIC);
    let timestamps = timing && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    let mut features = wgpu::Features::empty();
    if float_atomics {
      features |= wgpu::Features::SHADER_FLOAT32_ATOMIC;
    }
    if timestamps {
      features |= wgpu::Features::TIMESTAMP_QUERY;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
      label: Some("rabbit-denoise"),
      required_limits: limits,
      required_features: features,
      ..Default::default()
    }))
    .context("could not open the GPU")?;

    let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
      binding,
      visibility: wgpu::ShaderStages::COMPUTE,
      ty: wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
      },
      count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      label: Some("bm3d"),
      entries: &[
        wgpu::BindGroupLayoutEntry {
          binding: 0,
          visibility: wgpu::ShaderStages::COMPUTE,
          ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
          },
          count: None,
        },
        storage(1, true),
        storage(2, true),
        storage(3, true),
        storage(4, false),
        storage(5, true),
        storage(6, false),
        storage(7, false),
        storage(8, false),
        storage(9, false),
      ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("bm3d"),
      bind_group_layouts: &[Some(&layout)],
      immediate_size: 0,
    });
    // Kaiser window, 8x8 DCT, then the group DCT of every length at 128 + n * 256.
    let mut tables = crate::bm3d::kaiser().to_vec();
    tables.extend(basis(N));
    for n in 0..=MAX_GROUP {
      let mut block = basis(n);
      block.resize(256, 0.0);
      tables.extend(block);
    }
    let timing = timestamps.then(|| Timing {
      queries: device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("timing"),
        ty: wgpu::QueryType::Timestamp,
        count: 2 * STAMPS * MAX_JOBS as u32,
      }),
      resolved: [(); 2].map(|_| {
        device.create_buffer(&wgpu::BufferDescriptor {
          label: Some("timing"),
          size: 8 * STAMPS as u64 * MAX_JOBS as u64,
          usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
          mapped_at_creation: false,
        })
      }),
      readback: [(); 2].map(|_| {
        device.create_buffer(&wgpu::BufferDescriptor {
          label: Some("timing"),
          size: 8 * STAMPS as u64 * MAX_JOBS as u64,
          usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
          mapped_at_creation: false,
        })
      }),
      period: queue.get_timestamp_period() as f64,
      totals: RefCell::new([[0.0; 4]; 2]),
    });
    let empty = device.create_buffer(&wgpu::BufferDescriptor {
      label: Some("empty"),
      size: 16,
      usage: wgpu::BufferUsages::STORAGE,
      mapped_at_creation: false,
    });
    Ok(Self {
      timing,
      device,
      queue,
      layout,
      pipeline_layout,
      float_atomics,
      pipelines: RefCell::new(HashMap::new()),
      motion: RefCell::new(HashMap::new()),
      field_use: RefCell::new(FieldUse::default()),
      empty,
      outstanding: std::cell::Cell::new([false; 2]),
      measuring: RefCell::new(None),
      measure_work: RefCell::new(HashMap::new()),
      tables,
      name: format!("{} ({:?})", info.name, info.backend),
      rings: RefCell::new(HashMap::new()),
      work: RefCell::new(HashMap::new()),
    })
  }

  /// The adapter's name and backend.
  pub fn name(&self) -> &str {
    &self.name
  }

  /// Seconds per entry point and stage, with timing.
  pub fn kernel_seconds(&self) -> Option<[[f64; 4]; 2]> {
    self
      .timing
      .as_ref()
      .map(|t| t.totals.borrow().map(|stage| stage.map(|ns| ns / 1e9)))
  }

  /// The entry points for a region side, built the first time it is asked for.
  fn pipelines(&self, side: usize) -> Result<std::rc::Rc<Pipelines>> {
    if let Some(p) = self.pipelines.borrow().get(&side) {
      return Ok(p.clone());
    }
    // Shader and pipeline errors come back as errors instead of ending the program.
    let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let source = wgpu::ShaderModuleDescriptor {
      label: Some("bm3d"),
      source: wgpu::ShaderSource::Wgsl(
        include_str!("bm3d.wgsl")
          .replace(
            "//#ACCUMULATOR",
            if self.float_atomics { FLOAT_ACCUMULATOR } else { BITS_ACCUMULATOR },
          )
          .replace(
            "//#TILE_SIDE",
            // The other tile also holds the 16 references' 4 lanes' 8 best distances.
            &format!(
              "const TILE_SIDE: u32 = {side}u;\nconst OTHER_AREA: u32 = {}u;",
              (side * side).max(512)
            ),
          )
          .into(),
      ),
    };
    // Bounds checks cost about a third of the time, so release builds leave them out.
    let checks = wgpu::ShaderRuntimeChecks {
      bounds_checks: cfg!(debug_assertions),
      force_loop_bounding: cfg!(debug_assertions),
      ..Default::default()
    };
    // SAFETY: indices stay in range for every size `supports` and `filter` allow, and
    // every loop ends. Debug builds keep the checks.
    let module = unsafe { self.device.create_shader_module_trusted(source, checks) };
    let pipeline = |entry: &str| {
      self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(&self.pipeline_layout),
        module: &module,
        entry_point: Some(entry),
        // Every workgroup value is written before it is read.
        compilation_options: wgpu::PipelineCompilationOptions {
          zero_initialize_workgroup_memory: false,
          ..Default::default()
        },
        cache: None,
      })
    };
    let built = std::rc::Rc::new(Pipelines {
      half_plane: pipeline("half_plane"),
      motion: pipeline("motion"),
      find_groups: pipeline("find_groups"),
      filter: pipeline("filter_groups"),
      normalize: pipeline("normalize"),
    });
    if let Some(error) = pollster::block_on(scope.pop()) {
      bail!("the GPU could not build the filter: {error}");
    }
    self.pipelines.borrow_mut().insert(side, built.clone());
    Ok(built)
  }

  /// Puts the window's planes into their ring and returns each one's slot.
  fn place(&self, kind: Kind, plane: usize, planes: &[&Plane], keys: &[FrameKey], slots: usize) -> Vec<usize> {
    let plane_size = planes[0].width * planes[0].height;
    let mut rings = self.rings.borrow_mut();
    let ring = rings.entry((plane, kind)).or_insert_with(|| Ring {
      buffer: self.buffer("ring", 0, wgpu::BufferUsages::STORAGE),
      plane_size: 0,
      keys: Vec::new(),
    });
    if ring.plane_size != plane_size || ring.keys.len() != slots {
      ring.buffer = self.buffer(
        "ring",
        slots * plane_size * 4,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
      );
      ring.plane_size = plane_size;
      ring.keys = vec![None; slots];
    }
    planes
      .iter()
      .zip(keys)
      .map(|(p, key)| {
        let slot = key.index % slots;
        if ring.keys[slot] != Some(*key) {
          self
            .queue
            .write_buffer(&ring.buffer, (slot * plane_size * 4) as u64, bytemuck::cast_slice(&p.data));
          ring.keys[slot] = Some(*key);
        }
        slot
      })
      .collect()
  }

  fn buffer(&self, label: &str, bytes: usize, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    self.device.create_buffer(&wgpu::BufferDescriptor {
      label: Some(label),
      size: bytes.max(4) as u64,
      usage,
      mapped_at_creation: false,
    })
  }

  fn work_buffers(&self, plane_size: usize, refs: usize) -> Work {
    Work {
      plane_size,
      refs,
      groups: self.buffer("groups", refs * GROUP_STRIDE * 4, wgpu::BufferUsages::STORAGE),
      accum: self.buffer("accum", plane_size * 8, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
      output: self.buffer("output", plane_size * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
      readback: self.buffer(
        "readback",
        plane_size * 4,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
      ),
    }
  }

  /// Filters the window's current frame, as `bm3d::filter` does.
  pub fn filter(&self, stage: Stage, window: &Window, noise: &NoiseModel, params: &Params, maxval: f32) -> Result<Plane> {
    Ok(
      self
        .filter_planes(
          &[Job {
            stage,
            window,
            noise,
            params,
          }],
          maxval,
        )?
        .remove(0),
    )
  }

  /// Filters several planes of one stage in one submission.
  pub fn filter_planes(&self, jobs: &[Job], maxval: f32) -> Result<Vec<Plane>> {
    let submission = self.submit(jobs, maxval, None)?;
    self.collect(submission)
  }

  /// Starts filtering several planes of one stage and returns at once. One submission of
  /// each stage may be outstanding. With `measure`, the grain of the hard threshold stage's
  /// result is measured in the same submission, for `collect_measured`.
  pub fn submit(&self, jobs: &[Job], maxval: f32, measure: Option<&MeasureAfter>) -> Result<Submission> {
    let wiener = jobs.first().is_some_and(|j| j.stage == Stage::Wiener);
    if jobs.iter().any(|j| (j.stage == Stage::Wiener) != wiener) {
      bail!("one submission holds planes of one stage");
    }
    let mut outstanding = self.outstanding.get();
    if outstanding[wiener as usize] {
      bail!("a GPU submission of this stage is still outstanding");
    }
    if wiener && measure.is_some() {
      bail!("only the hard threshold stage's result is measured");
    }
    let result = self.record(jobs, maxval, wiener, measure);
    if result.is_ok() {
      outstanding[wiener as usize] = true;
      self.outstanding.set(outstanding);
    }
    result
  }

  fn record(&self, jobs: &[Job], maxval: f32, wiener: bool, measure: Option<&MeasureAfter>) -> Result<Submission> {
    if jobs.len() > MAX_JOBS {
      bail!("at most {MAX_JOBS} planes go to the GPU at once");
    }
    // Running out of GPU memory comes back as an error.
    let scope = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let mut encoder = self.device.create_command_encoder(&Default::default());
    let mut pending = Vec::with_capacity(jobs.len());
    for (j, job) in jobs.iter().enumerate() {
      pending.push(self.encode(&mut encoder, job, maxval, wiener as usize * MAX_JOBS + j)?);
    }
    let measured = match measure {
      Some(m) => Some(self.encode_measure_after(&mut encoder, jobs, &pending, m)?),
      None => None,
    };
    if let Some(t) = &self.timing {
      let first = STAMPS * (wiener as usize * MAX_JOBS) as u32;
      let count = STAMPS * jobs.len() as u32;
      let stage = wiener as usize;
      encoder.resolve_query_set(&t.queries, first..first + count, &t.resolved[stage], 0);
      encoder.copy_buffer_to_buffer(&t.resolved[stage], 0, &t.readback[stage], 0, Some(8 * count as u64));
    }
    let index = self.queue.submit([encoder.finish()]);
    if let Some(error) = pollster::block_on(scope.pop()) {
      bail!("the GPU ran out of memory: {error}");
    }
    Ok(Submission {
      pending,
      jobs: jobs.len(),
      wiener,
      index,
      measured,
    })
  }

  /// Waits for a submission and reads its planes back.
  pub fn collect(&self, submission: Submission) -> Result<Vec<Plane>> {
    Ok(self.collect_measured(submission)?.0)
  }

  /// Waits for a submission and reads its planes back, and the grain measured on them.
  pub fn collect_measured(&self, submission: Submission) -> Result<(Vec<Plane>, Option<Measures>)> {
    let Submission {
      pending,
      jobs,
      wiener,
      index,
      measured,
    } = submission;
    let mut outstanding = self.outstanding.get();
    outstanding[wiener as usize] = false;
    self.outstanding.set(outstanding);
    let works = self.work.borrow();
    let (sender, receiver) = std::sync::mpsc::channel();
    let mut mapped = 0;
    for p in &pending {
      if let Pending::Submitted { plane, .. } = p {
        let sender = sender.clone();
        works[&(*plane, wiener)]
          .readback
          .slice(..)
          .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
          });
        mapped += 1;
      }
    }
    if let Some(m) = &measured {
      mapped += self.map_measure(m, &sender);
    }
    if let Some(t) = &self.timing {
      t.readback[wiener as usize].slice(..).map_async(wgpu::MapMode::Read, |_| {});
    }
    // Only this submission: one queued after it may still be running.
    self
      .device
      .poll(wgpu::PollType::Wait {
        submission_index: Some(index),
        timeout: None,
      })
      .context("the GPU stopped responding")?;
    for _ in 0..mapped {
      receiver
        .recv()
        .context("the GPU stopped responding")?
        .context("could not read the result back")?;
    }
    let mut out = Vec::with_capacity(jobs);
    for p in pending {
      match p {
        Pending::Done(plane) => out.push(plane),
        Pending::Submitted { plane, width, height } => {
          let readback = &works[&(plane, wiener)].readback;
          let view = readback
            .slice(..)
            .get_mapped_range()
            .map_err(|e| anyhow::anyhow!("could not read the result back: {e:?}"))?;
          let data = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
          drop(view);
          readback.unmap();
          out.push(Plane { width, height, data });
        }
      }
    }
    if let Some(t) = &self.timing {
      let readback = &t.readback[wiener as usize];
      let view = readback
        .slice(..)
        .get_mapped_range()
        .map_err(|e| anyhow::anyhow!("could not read the timing back: {e:?}"))?;
      let stamps: Vec<u64> = bytemuck::cast_slice::<u8, u64>(&view).to_vec();
      drop(view);
      readback.unmap();
      let mut totals = t.totals.borrow_mut();
      for j in 0..jobs {
        for i in 0..4 {
          let at = STAMPS as usize * j + 2 * i;
          totals[wiener as usize][i] += stamps[at + 1].saturating_sub(stamps[at]) as f64 * t.period;
        }
      }
    }
    let measures = measured.map(|m| self.read_measure(&m)).transpose()?;
    Ok((out, measures))
  }

  /// Records the measurement of the hard threshold stage's result. The result goes into the
  /// basic ring on the GPU, where the Wiener stage finds it.
  fn encode_measure_after(
    &self,
    encoder: &mut wgpu::CommandEncoder,
    jobs: &[Job],
    pending: &[Pending],
    m: &MeasureAfter,
  ) -> Result<measure::PendingMeasure> {
    let mut planes = Vec::with_capacity(jobs.len());
    for (job, done) in jobs.iter().zip(pending) {
      let p = job.window.plane;
      let noisy = &m.noisy[p];
      let size = noisy.width * noisy.height;
      let basic = match done {
        Pending::Submitted { .. } => {
          let slot = self.reserve(Kind::Basic, p, size, m.key, m.slots[1]);
          let rings = self.rings.borrow();
          let works = self.work.borrow();
          encoder.copy_buffer_to_buffer(
            &works[&(p, false)].output,
            0,
            &rings[&(p, Kind::Basic)].buffer,
            (slot * size * 4) as u64,
            Some((size * 4) as u64),
          );
          slot
        }
        // Too small to filter, so the result is the noisy plane.
        Pending::Done(plane) => self.slot_of(Kind::Basic, p, plane, m.key, m.slots[1]),
      };
      let noisy_key = FrameKey {
        index: m.key.index,
        version: 0,
      };
      let previous = m.previous.as_ref().map(|f| {
        [
          self.slot_of(Kind::Noisy, p, &f.noisy[p], f.noisy_key, m.slots[0]),
          self.slot_of(Kind::Basic, p, &f.basic[p], f.basic_key, m.slots[1]),
        ]
      });
      planes.push(measure::MeasurePlane {
        width: noisy.width,
        height: noisy.height,
        current: [self.slot_of(Kind::Noisy, p, noisy, noisy_key, m.slots[0]), basic],
        previous,
      });
    }
    self.encode_measure(encoder, &planes, true, m.luma_matching)
  }

  /// A slot for a frame's plane that the GPU fills itself. A ring that does not exist yet
  /// gets `slots` slots.
  fn reserve(&self, kind: Kind, plane: usize, size: usize, key: FrameKey, slots: usize) -> usize {
    let mut rings = self.rings.borrow_mut();
    let ring = rings.entry((plane, kind)).or_insert_with(|| Ring {
      buffer: self.buffer("ring", 0, wgpu::BufferUsages::STORAGE),
      plane_size: 0,
      keys: Vec::new(),
    });
    if ring.plane_size != size {
      ring.buffer = self.buffer("ring", slots * size * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
      ring.plane_size = size;
      ring.keys = vec![None; slots];
    }
    let slot = key.index % ring.keys.len();
    ring.keys[slot] = Some(key);
    slot
  }

  /// Records one plane's work into the encoder.
  fn encode(&self, encoder: &mut wgpu::CommandEncoder, job: &Job, maxval: f32, slot: usize) -> Result<Pending> {
    let Job {
      stage,
      window,
      noise,
      params,
    } = *job;
    let current = window.noisy[window.center];
    let (width, height) = (current.width, current.height);
    if width < N || height < N {
      return Ok(Pending::Done(current.clone()));
    }
    let frames = window.noisy.len();
    if width >= 1 << 13 || height >= 1 << 13 || frames > MAX_FRAMES || window.center >= frames {
      bail!("{width}x{height} with {frames} frames is too large for the GPU filter");
    }
    if !supports(params) {
      bail!("these settings are larger than the GPU filter supports");
    }
    let same_size = |planes: &[&Plane]| {
      planes
        .iter()
        .all(|p| p.width == width && p.height == height && p.data.len() == width * height)
    };
    if window.guide.len() != frames
      || window.noisy_keys.len() != frames
      || window.guide_keys.len() != frames
      || !same_size(&window.noisy)
      || !same_size(&window.guide)
    {
      bail!("the window's planes do not match");
    }
    let plane_size = width * height;
    // The region 4x4 reference patches search.
    let side = (TILE_REFS - 1) * params.step + N + 2 * params.search_radius;
    debug_assert!(side <= MAX_TILE_SIDE);
    let pipelines = self.pipelines(side)?;
    let xs = crate::bm3d::positions(width, params.step);
    let ys = crate::bm3d::positions(height, params.step);
    let refs = xs.len() * ys.len();
    let scale = maxval / 255.0;

    // Large enough never to overwrite a frame a later window still needs.
    let slots = (4 * params.temporal_radius + 4).max(frames);
    let noisy_slots = self.place(Kind::Noisy, window.plane, &window.noisy, &window.noisy_keys, slots);
    let guide_slots = if stage == Stage::Hard {
      noisy_slots.clone()
    } else {
      self.place(Kind::Basic, window.plane, &window.guide, &window.guide_keys, slots)
    };

    let mut model = Vec::with_capacity(BANDS * AREA + BANDS);
    for band in 0..BANDS {
      model.extend_from_slice(noise.spectrum(band));
    }
    // Each band's mean variance per sample, which find_groups interpolates for its limit.
    for band in 0..BANDS {
      model.push(noise.spectrum(band).iter().sum::<f32>() / AREA as f32);
    }
    let init = |label: &str, contents: &[u8], usage| {
      self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage,
      })
    };
    let model = init("model", bytemuck::cast_slice(&model), wgpu::BufferUsages::STORAGE);
    let mut tables = self.tables.clone();
    tables.extend(xs.iter().chain(&ys).map(|&v| v as f32));
    let slots_at = tables.len();
    tables.extend(noisy_slots.iter().chain(&guide_slots).map(|&s| s as f32));
    // Luma measures the fields not measured yet, and chroma reads them.
    let motion = params.motion && frames > 1;
    let kind = if stage == Stage::Hard { Kind::Noisy } else { Kind::Basic };
    let (mut half_jobs, mut field_jobs) = (Vec::new(), Vec::new());
    if motion && window.plane == 0 {
      let (hw, hh) = (width / 2, height / 2);
      let (blocks_x, blocks_y) = ((hw / N).max(1), (hh / N).max(1));
      let shape = (blocks_x, blocks_y, width, height, slots);
      let mut caches = self.motion.borrow_mut();
      let cache = caches.entry(kind).or_insert_with(|| MotionCache {
        half_planes: self.buffer("half planes", 0, wgpu::BufferUsages::STORAGE),
        fields: self.buffer("fields", 0, wgpu::BufferUsages::STORAGE),
        half_keys: Vec::new(),
        field_keys: Vec::new(),
        shape: (0, 0, 0, 0, 0),
      });
      if cache.shape != shape {
        cache.half_planes = self.buffer("half planes", slots * hw * hh * 4, wgpu::BufferUsages::STORAGE);
        cache.fields = self.buffer("fields", slots * 2 * blocks_x * blocks_y * 4, wgpu::BufferUsages::STORAGE);
        cache.half_keys = vec![None; slots];
        cache.field_keys = vec![None; 2 * slots];
        cache.shape = shape;
      }
      for (f, &slot) in guide_slots.iter().enumerate() {
        if cache.half_keys[slot] != Some(window.guide_keys[f]) {
          cache.half_keys[slot] = Some(window.guide_keys[f]);
          half_jobs.push(slot);
        }
      }
      // Step k goes from frame c - k + 1 to c - k before the centre, and from c + k - 1 to
      // c + k after it. Steps past the window are never read.
      let c = window.center;
      let mut map = vec![0.0; 2 * params.temporal_radius];
      for k in 1..=params.temporal_radius {
        for (after, step) in [(false, k - 1), (true, params.temporal_radius + k - 1)] {
          let (source, target) = match after {
            false if k <= c => (c + 1 - k, c - k),
            true if c + k < frames => (c + k - 1, c + k),
            _ => continue,
          };
          let slot = 2 * guide_slots[source] + after as usize;
          let keys = (window.guide_keys[source], window.guide_keys[target]);
          if cache.field_keys[slot] != Some(keys) {
            cache.field_keys[slot] = Some(keys);
            field_jobs.push([slot, guide_slots[source], guide_slots[target]]);
          }
          map[step] = slot as f32;
        }
      }
      *self.field_use.borrow_mut() = FieldUse {
        kind: Some(kind),
        blocks_x,
        blocks_y,
        luma_width: width,
        luma_height: height,
        map,
      };
    }
    let field_use = self.field_use.borrow().clone();
    if motion && (field_use.kind != Some(kind) || field_use.luma_width < width || field_use.luma_height < height) {
      bail!("chroma planes follow the luma plane's motion, which went first");
    }
    let field_map = tables.len();
    tables.extend(&field_use.map);
    let half_at = tables.len();
    tables.extend(half_jobs.iter().map(|&s| s as f32));
    let fields_at = tables.len();
    tables.extend(field_jobs.iter().flatten().map(|&s| s as f32));
    let tables = init("tables", bytemuck::cast_slice(&tables), wgpu::BufferUsages::STORAGE);
    let (blocks_x, blocks_y, luma_width, luma_height) =
      (field_use.blocks_x, field_use.blocks_y, field_use.luma_width, field_use.luma_height);
    let uniforms = Uniforms {
      field_map: field_map as u32,
      half_jobs: half_at as u32,
      half_count: half_jobs.len() as u32,
      field_jobs: fields_at as u32,
      field_count: field_jobs.len() as u32,
      motion: motion as u32,
      field_blocks_x: blocks_x as u32,
      field_blocks_y: blocks_y as u32,
      motion_scale_x: if motion { (luma_width / width) as u32 } else { 1 },
      motion_scale_y: if motion { (luma_height / height) as u32 } else { 1 },
      half_width: (width / 2) as u32,
      half_height: (height / 2) as u32,
      half_range: (params.motion_range / 2) as u32,
      width: width as u32,
      height: height as u32,
      frames: frames as u32,
      center: window.center as u32,
      refs_x: xs.len() as u32,
      refs_y: ys.len() as u32,
      search_radius: params.search_radius as u32,
      predict_radius: params.predict_radius as u32,
      temporal_radius: params.temporal_radius as u32,
      per_frame: params.per_frame as u32,
      min_shift: params.min_shift as u32,
      max_group: params.max_group as u32,
      stage: (stage == Stage::Wiener) as u32,
      lambda: params.lambda,
      distance_factor: params.distance_factor,
      distance_floor: params.distance_floor * scale * scale,
      maxval,
      plane_size: plane_size as u32,
      dispatch_x: DISPATCH_X,
      slots: slots_at as u32,
    };
    let uniforms = init("uniforms", bytemuck::bytes_of(&uniforms), wgpu::BufferUsages::UNIFORM);

    let mut works = self.work.borrow_mut();
    let work = works
      .entry((window.plane, stage == Stage::Wiener))
      .or_insert_with(|| self.work_buffers(0, 0));
    if work.plane_size != plane_size || work.refs != refs {
      *work = self.work_buffers(plane_size, refs);
    }
    let rings = self.rings.borrow();
    let noisy = &rings[&(window.plane, Kind::Noisy)].buffer;
    let guide = if stage == Stage::Hard {
      noisy
    } else {
      &rings[&(window.plane, Kind::Basic)].buffer
    };
    let caches = self.motion.borrow();
    let (fields, half_planes) = match caches.get(&kind) {
      Some(cache) if motion => (&cache.fields, &cache.half_planes),
      _ => (&self.empty, &self.empty),
    };
    let entries: Vec<wgpu::BindGroupEntry> = [
      &uniforms,
      noisy,
      guide,
      &model,
      &work.accum,
      &tables,
      &work.output,
      &work.groups,
      fields,
      half_planes,
    ]
    .iter()
    .enumerate()
    .map(|(i, buffer)| wgpu::BindGroupEntry {
      binding: i as u32,
      resource: buffer.as_entire_binding(),
    })
    .collect();
    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("bm3d"),
      layout: &self.layout,
      entries: &entries,
    });

    let grid = |workgroups: u32| (workgroups.min(DISPATCH_X), workgroups.div_ceil(DISPATCH_X));
    encoder.clear_buffer(&work.accum, 0, None);
    let tiles = xs.len().div_ceil(TILE_REFS) * ys.len().div_ceil(TILE_REFS);
    if motion && window.plane == 0 {
      let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: self.timing.as_ref().map(|t| wgpu::ComputePassTimestampWrites {
          query_set: &t.queries,
          beginning_of_pass_write_index: Some(STAMPS * slot as u32),
          end_of_pass_write_index: Some(STAMPS * slot as u32 + 1),
        }),
      });
      pass.set_bind_group(0, &bind_group, &[]);
      let (x, y) = grid(((half_jobs.len() * (width / 2) * (height / 2)) as u32).div_ceil(WORKGROUP));
      pass.set_pipeline(&pipelines.half_plane);
      pass.dispatch_workgroups(x, y, 1);
      // One workgroup per block of each field to measure.
      let (x, y) = grid((blocks_x * blocks_y * field_jobs.len()) as u32);
      pass.set_pipeline(&pipelines.motion);
      pass.dispatch_workgroups(x, y, 1);
    }
    let passes: [(&wgpu::ComputePipeline, u32); 3] = [
      (&pipelines.find_groups, tiles as u32),
      (&pipelines.filter, tiles as u32),
      (&pipelines.normalize, (plane_size as u32).div_ceil(WORKGROUP)),
    ];
    for (i, (pipeline, workgroups)) in passes.into_iter().enumerate() {
      let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: self.timing.as_ref().map(|t| wgpu::ComputePassTimestampWrites {
          query_set: &t.queries,
          beginning_of_pass_write_index: Some((STAMPS as usize * slot + 2 + 2 * i) as u32),
          end_of_pass_write_index: Some((STAMPS as usize * slot + 3 + 2 * i) as u32),
        }),
      });
      pass.set_bind_group(0, &bind_group, &[]);
      let (x, y) = grid(workgroups);
      pass.set_pipeline(pipeline);
      pass.dispatch_workgroups(x, y, 1);
    }
    encoder.copy_buffer_to_buffer(&work.output, 0, &work.readback, 0, None);
    Ok(Pending::Submitted {
      plane: window.plane,
      width,
      height,
    })
  }
}

/// Planes on their way through the GPU, from `Gpu::submit` to `Gpu::collect`.
pub struct Submission {
  pending: Vec<Pending>,
  jobs: usize,
  /// Whether its planes are in the Wiener stage.
  wiener: bool,
  index: wgpu::SubmissionIndex,
  /// The grain measured on the result, if asked for.
  measured: Option<measure::PendingMeasure>,
}

/// What `Gpu::submit` measures on the hard threshold stage's result: the residual of the
/// frame, and the pair it ends.
pub struct MeasureAfter<'a> {
  /// The frame's index and the version its new basic estimate gets.
  pub key: FrameKey,
  /// The frame's noisy planes.
  pub noisy: &'a [Plane],
  /// The frame before, done already.
  pub previous: Option<MeasureFrame<'a>>,
  /// Whether chroma is matched on the luma basic estimates.
  pub luma_matching: bool,
  /// The slots of rings made here, noisy then basic.
  pub slots: [usize; 2],
}

/// One plane to filter.
#[derive(Clone, Copy)]
pub struct Job<'a> {
  pub stage: Stage,
  pub window: &'a Window<'a>,
  pub noise: &'a NoiseModel,
  pub params: &'a Params,
}

enum Pending {
  /// Too small to filter: returned as it is.
  Done(Plane),
  /// Recorded, with its result in the plane's readback buffer.
  Submitted { plane: usize, width: usize, height: usize },
}
