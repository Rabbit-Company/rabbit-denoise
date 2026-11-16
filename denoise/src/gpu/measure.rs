//! The grain measurements' blocks on the GPU, from the frames in the rings. The CPU takes
//! the medians of what comes back, as for its own blocks.

use super::{DISPATCH_X, Gpu, Kind, WORKGROUP};
use crate::bm3d::FrameKey;
use crate::dct::{AREA, N, basis};
use crate::noise::{BLOCK_STRIDE, Measures, PairBlock, ResidualBlock};
use crate::plane::Plane;
use anyhow::{Context, Result, bail};
use std::sync::mpsc::Sender;
use wgpu::util::DeviceExt;

// Per block in the output: 64 coefficients, then two numbers.
const BLOCK: usize = AREA + 2;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
  width: u32,
  height: u32,
  blocks_x: u32,
  blocks_y: u32,
  noisy_a: u32,
  basic_a: u32,
  noisy_b: u32,
  basic_b: u32,
  scaled: u32,
  luma_width: u32,
  luma_height: u32,
  luma_a: u32,
  luma_b: u32,
  pair_out: u32,
  residual_out: u32,
  dispatch_x: u32,
}

/// The entry points and what they share, built on first use.
pub(super) struct Measuring {
  layout: wgpu::BindGroupLayout,
  scale: wgpu::ComputePipeline,
  pairs: wgpu::ComputePipeline,
  residual: wgpu::ComputePipeline,
  basis: wgpu::Buffer,
}

/// The buffers a plane's measurements reuse.
pub(super) struct MeasureWork {
  /// Pair blocks, then residual blocks.
  out: wgpu::Buffer,
  readback: wgpu::Buffer,
  /// Both frames' luma scaled to this plane's size.
  scaled: wgpu::Buffer,
  /// The plane's width and height, and whether it is matched on scaled luma.
  shape: (usize, usize, bool),
}

/// One frame for `Gpu::measure`: its planes and keys, noisy and basic.
pub struct MeasureFrame<'a> {
  pub noisy: &'a [Plane],
  pub basic: &'a [Plane],
  pub noisy_key: FrameKey,
  pub basic_key: FrameKey,
}

/// A plane to measure: its size and its frames' slots, noisy then basic, in the rings.
pub(super) struct MeasurePlane {
  pub(super) width: usize,
  pub(super) height: usize,
  pub(super) current: [usize; 2],
  pub(super) previous: Option<[usize; 2]>,
}

/// A recorded measurement: each plane's block count, and what was measured.
pub(super) struct PendingMeasure {
  counts: Vec<usize>,
  pairs: bool,
  residual: bool,
}

impl Gpu {
  fn measuring(&self) -> Result<std::cell::Ref<'_, Measuring>> {
    if self.measuring.borrow().is_none() {
      let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
      let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
          ty,
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      };
      let read = wgpu::BufferBindingType::Storage { read_only: true };
      let write = wgpu::BufferBindingType::Storage { read_only: false };
      let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("measure"),
        entries: &[
          entry(0, wgpu::BufferBindingType::Uniform),
          entry(1, read),
          entry(2, read),
          entry(3, read),
          entry(4, read),
          entry(5, write),
          entry(6, write),
        ],
      });
      let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("measure"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
      });
      let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("measure"),
        source: wgpu::ShaderSource::Wgsl(include_str!("measure.wgsl").into()),
      });
      let pipeline = |entry: &str| {
        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
          label: Some(entry),
          layout: Some(&pipeline_layout),
          module: &module,
          entry_point: Some(entry),
          compilation_options: Default::default(),
          cache: None,
        })
      };
      let built = Measuring {
        scale: pipeline("scale"),
        pairs: pipeline("pair_blocks"),
        residual: pipeline("residual_blocks"),
        basis: self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
          label: Some("basis"),
          contents: bytemuck::cast_slice(&basis(N)),
          usage: wgpu::BufferUsages::STORAGE,
        }),
        layout,
      };
      if let Some(error) = pollster::block_on(scope.pop()) {
        bail!("the GPU could not build the grain measurement: {error}");
      }
      *self.measuring.borrow_mut() = Some(built);
    }
    Ok(std::cell::Ref::map(self.measuring.borrow(), |m| m.as_ref().unwrap()))
  }

  /// A plane of a frame's slot in its ring, uploaded unless it is there. A ring that does
  /// not exist yet gets `slots` slots.
  pub(super) fn slot_of(&self, kind: Kind, plane: usize, data: &Plane, key: FrameKey, slots: usize) -> usize {
    let size = data.width * data.height;
    let slots = match self.rings.borrow().get(&(plane, kind)) {
      Some(ring) if ring.plane_size == size => ring.keys.len(),
      _ => slots,
    };
    self.place(kind, plane, &[data], &[key], slots)[0]
  }

  /// The blocks of the residual of `current`, with `residual`, and of the pair `previous`
  /// and `current`, with `previous`, as NoiseStats takes them. Chroma is matched on the luma
  /// basic estimates with `luma_matching`. Rings made here get `slots` slots, noisy then
  /// basic, as the filter would make them.
  pub fn measure(
    &self,
    current: &MeasureFrame,
    previous: Option<&MeasureFrame>,
    residual: bool,
    luma_matching: bool,
    slots: [usize; 2],
  ) -> Result<Measures> {
    let scope = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    // Every plane's frames into the rings first, as chroma reads the luma ones.
    let frame_slots = |f: &MeasureFrame, p: usize| {
      [
        self.slot_of(Kind::Noisy, p, &f.noisy[p], f.noisy_key, slots[0]),
        self.slot_of(Kind::Basic, p, &f.basic[p], f.basic_key, slots[1]),
      ]
    };
    let planes: Vec<MeasurePlane> = (0..current.noisy.len())
      .map(|p| MeasurePlane {
        width: current.noisy[p].width,
        height: current.noisy[p].height,
        current: frame_slots(current, p),
        previous: previous.map(|f| frame_slots(f, p)),
      })
      .collect();
    let mut encoder = self.device.create_command_encoder(&Default::default());
    let pending = self.encode_measure(&mut encoder, &planes, residual, luma_matching)?;
    let index = self.queue.submit([encoder.finish()]);
    if let Some(error) = pollster::block_on(scope.pop()) {
      bail!("the GPU ran out of memory: {error}");
    }
    let (sender, receiver) = std::sync::mpsc::channel();
    let mapped = self.map_measure(&pending, &sender);
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
        .context("could not read the grain measurement back")?;
    }
    self.read_measure(&pending)
  }

  /// Records the measurement of planes whose frames are in the rings.
  pub(super) fn encode_measure(
    &self,
    encoder: &mut wgpu::CommandEncoder,
    planes: &[MeasurePlane],
    residual: bool,
    luma_matching: bool,
  ) -> Result<PendingMeasure> {
    let measuring = self.measuring()?;
    let rings = self.rings.borrow();
    let mut works = self.measure_work.borrow_mut();
    let pairs = planes.first().is_some_and(|p| p.previous.is_some());
    let mut counts = Vec::with_capacity(planes.len());
    for (p, plane) in planes.iter().enumerate() {
      let (width, height) = (plane.width, plane.height);
      let (blocks_x, blocks_y) = ((width / N).div_ceil(BLOCK_STRIDE), (height / N).div_ceil(BLOCK_STRIDE));
      let blocks = blocks_x * blocks_y;
      counts.push(blocks);
      if blocks == 0 {
        continue;
      }
      let plane_size = width * height;
      let scaled = luma_matching && p > 0;
      let shape = (width, height, scaled);
      let work = works.entry(p).or_insert_with(|| self.measure_buffers((0, 0, false)));
      if work.shape != shape {
        *work = self.measure_buffers(shape);
      }
      let b = plane.current;
      let a = plane.previous.unwrap_or(b);
      let luma = &planes[0];
      let luma_size = luma.width * luma.height;
      let uniforms = Uniforms {
        width: width as u32,
        height: height as u32,
        blocks_x: blocks_x as u32,
        blocks_y: blocks_y as u32,
        noisy_a: (a[0] * plane_size) as u32,
        basic_a: (a[1] * plane_size) as u32,
        noisy_b: (b[0] * plane_size) as u32,
        basic_b: (b[1] * plane_size) as u32,
        scaled: scaled as u32,
        luma_width: luma.width as u32,
        luma_height: luma.height as u32,
        luma_a: (luma.previous.unwrap_or(luma.current)[1] * luma_size) as u32,
        luma_b: (luma.current[1] * luma_size) as u32,
        pair_out: 0,
        residual_out: (blocks * BLOCK) as u32,
        dispatch_x: DISPATCH_X,
      };
      let uniforms = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("measure"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
      });
      let buffers = [
        &uniforms,
        &rings[&(p, Kind::Noisy)].buffer,
        &rings[&(p, Kind::Basic)].buffer,
        &rings[&(0, Kind::Basic)].buffer,
        &measuring.basis,
        &work.scaled,
        &work.out,
      ];
      let entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
        .enumerate()
        .map(|(i, buffer)| wgpu::BindGroupEntry {
          binding: i as u32,
          resource: buffer.as_entire_binding(),
        })
        .collect();
      let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("measure"),
        layout: &measuring.layout,
        entries: &entries,
      });
      let grid = |workgroups: u32| (workgroups.min(DISPATCH_X), workgroups.div_ceil(DISPATCH_X));
      {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_bind_group(0, &bind_group, &[]);
        if pairs {
          if scaled {
            let (x, y) = grid((2 * plane_size as u32).div_ceil(WORKGROUP));
            pass.set_pipeline(&measuring.scale);
            pass.dispatch_workgroups(x, y, 1);
          }
          let (x, y) = grid(blocks as u32);
          pass.set_pipeline(&measuring.pairs);
          pass.dispatch_workgroups(x, y, 1);
        }
        if residual {
          let (x, y) = grid(blocks as u32);
          pass.set_pipeline(&measuring.residual);
          pass.dispatch_workgroups(x, y, 1);
        }
      }
      encoder.copy_buffer_to_buffer(&work.out, 0, &work.readback, 0, Some((2 * blocks * BLOCK * 4) as u64));
    }
    Ok(PendingMeasure { counts, pairs, residual })
  }

  /// Starts reading a measurement back once its submission is done. Returns how many
  /// results `sender` gets.
  pub(super) fn map_measure(&self, pending: &PendingMeasure, sender: &Sender<Result<(), wgpu::BufferAsyncError>>) -> usize {
    let works = self.measure_work.borrow();
    let mut mapped = 0;
    for (p, &blocks) in pending.counts.iter().enumerate() {
      if blocks > 0 {
        let sender = sender.clone();
        works[&p]
          .readback
          .slice(..(2 * blocks * BLOCK * 4) as u64)
          .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
          });
        mapped += 1;
      }
    }
    mapped
  }

  /// The blocks of a measurement read back.
  pub(super) fn read_measure(&self, pending: &PendingMeasure) -> Result<Measures> {
    let works = self.measure_work.borrow();
    let (mut all_pairs, mut all_residual) = (Vec::new(), Vec::new());
    for (p, &blocks) in pending.counts.iter().enumerate() {
      let (mut pairs, mut residual) = (Vec::new(), Vec::new());
      if blocks > 0 {
        let readback = &works[&p].readback;
        let view = readback
          .slice(..(2 * blocks * BLOCK * 4) as u64)
          .get_mapped_range()
          .map_err(|e| anyhow::anyhow!("could not read the grain measurement back: {e:?}"))?;
        let data = bytemuck::cast_slice::<u8, f32>(&view);
        let spectrum = |block: &[f32]| -> [f32; AREA] { block[..AREA].try_into().unwrap() };
        let (pair_data, residual_data) = data.split_at(blocks * BLOCK);
        if pending.pairs {
          pairs = pair_data
            .as_chunks::<BLOCK>()
            .0
            .iter()
            .map(|b| PairBlock {
              spectrum: spectrum(b),
              distance: b[AREA],
              mean: b[AREA + 1],
            })
            .collect();
        }
        if pending.residual {
          residual = residual_data
            .as_chunks::<BLOCK>()
            .0
            .iter()
            .map(|b| ResidualBlock {
              spectrum: spectrum(b),
              kept: b[AREA],
              dc: b[AREA + 1],
            })
            .collect();
        }
        drop(view);
        readback.unmap();
      }
      all_pairs.push(pairs);
      all_residual.push(residual);
    }
    Ok(Measures {
      pairs: pending.pairs.then_some(all_pairs),
      residual: pending.residual.then_some(all_residual),
    })
  }

  fn measure_buffers(&self, shape: (usize, usize, bool)) -> MeasureWork {
    let (width, height, scaled) = shape;
    let blocks = (width / N).div_ceil(BLOCK_STRIDE) * (height / N).div_ceil(BLOCK_STRIDE);
    let bytes = 2 * blocks * BLOCK * 4;
    MeasureWork {
      out: self.buffer("measure", bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
      readback: self.buffer("measure", bytes, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
      scaled: self.buffer(
        "scaled",
        if scaled { 2 * width * height * 4 } else { 0 },
        wgpu::BufferUsages::STORAGE,
      ),
      shape,
    }
  }
}
