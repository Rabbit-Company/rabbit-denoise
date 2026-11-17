// The grain measurements of noise.rs on the GPU: the blocks of a frame pair and of a
// residual, for the CPU to take the medians of.
//
// - scale: the luma basic estimates averaged down to this plane's size, which chroma is
//   matched on.
// - pair_blocks: one workgroup per block of the first frame finds its best match in the
//   second, then takes the DCT of the halved noisy difference.
// - residual_blocks: one workgroup per block takes the DCT of the noisy block less the basic
//   one, and of the basic one.

const N: u32 = 8u;
const AREA: u32 = 64u;
// noise.rs PAIR_RADIUS, BLOCK_STRIDE and GRID_OFFSET.
const RADIUS: u32 = 6u;
const STRIDE: u32 = 2u;
const OFFSET: u32 = 3u;
// Per block in out: 64 coefficients, then two numbers.
const BLOCK: u32 = 66u;

struct Params {
  width: u32,
  height: u32,
  // Blocks across and down, every STRIDE-th block.
  blocks_x: u32,
  blocks_y: u32,
  // Where the planes start: the noisy and basic estimates of the first frame, then the second.
  noisy_a: u32,
  basic_a: u32,
  noisy_b: u32,
  basic_b: u32,
  // 1 to match on the luma basic estimates scaled to this plane's size.
  scaled: u32,
  luma_width: u32,
  luma_height: u32,
  luma_a: u32,
  luma_b: u32,
  // Where the pair blocks and the residual blocks go in out.
  pair_out: u32,
  residual_out: u32,
  // Workgroups along x in every dispatch, the rest along y.
  dispatch_x: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> noisy: array<f32>;
@group(0) @binding(2) var<storage, read> basic: array<f32>;
// The luma plane's basic estimates.
@group(0) @binding(3) var<storage, read> luma: array<f32>;
// The 8x8 DCT basis.
@group(0) @binding(4) var<storage, read> dct_basis: array<f32>;
// The first frame's scaled luma, then the second's.
@group(0) @binding(5) var<storage, read_write> scaled: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

var<workgroup> reference: array<f32, 64>;
// The area the moves reach: 8 + 2 * RADIUS on each side.
var<workgroup> area: array<f32, 400>;
var<workgroup> best_sum: array<f32, 64>;
var<workgroup> best_rank: array<u32, 64>;
var<workgroup> tile: array<f32, 64>;
var<workgroup> rows: array<f32, 64>;

// Coefficient t of the DCT of the patch whose pixel t is `value`, added up in the order
// Transforms::forward does. Every thread calls it.
fn dct(t: u32, value: f32) -> f32 {
  let row = t / N;
  let col = t % N;
  workgroupBarrier();
  tile[t] = value;
  workgroupBarrier();
  var s = 0.0;
  for (var x = 0u; x < N; x++) {
    s += dct_basis[col * N + x] * tile[row * N + x];
  }
  rows[t] = s;
  workgroupBarrier();
  var o = 0.0;
  for (var y = 0u; y < N; y++) {
    o += dct_basis[row * N + y] * rows[y * N + col];
  }
  return o;
}

// The sample a block is matched on, in the first frame (0) or the second (1).
fn matched(frame: u32, at: u32) -> f32 {
  if (params.scaled == 1u) {
    return scaled[frame * params.width * params.height + at];
  }
  return basic[select(params.basic_a, params.basic_b, frame == 1u) + at];
}

fn block_index(wg: vec3<u32>) -> u32 {
  return wg.y * params.dispatch_x + wg.x;
}

// ---- scale ----

// As Plane::downscale does, adding up in the same order.
@compute @workgroup_size(64)
fn scale(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) local: u32) {
  let i = block_index(wg) * 64u + local;
  let size = params.width * params.height;
  if (i >= 2u * size) {
    return;
  }
  let base = select(params.luma_a, params.luma_b, i >= size);
  let x = (i % size) % params.width;
  let y = (i % size) / params.width;
  let fx = (params.luma_width + params.width - 1u) / params.width;
  let fy = (params.luma_height + params.height - 1u) / params.height;
  var sum = 0.0;
  var count = 0.0;
  for (var sy = y * fy; sy < min((y + 1u) * fy, params.luma_height); sy++) {
    for (var sx = x * fx; sx < min((x + 1u) * fx, params.luma_width); sx++) {
      sum += luma[base + sy * params.luma_width + sx];
      count += 1.0;
    }
  }
  scaled[i] = select(0.0, sum / count, count > 0.0);
}

// ---- pair_blocks ----

// As NoiseStats::measure_temporal finds each block's match: the smallest distance, the
// first in search order among equals.
@compute @workgroup_size(64)
fn pair_blocks(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let i = block_index(wg);
  if (i >= params.blocks_x * params.blocks_y) {
    return;
  }
  let x = OFFSET + (i % params.blocks_x) * STRIDE * N;
  let y = OFFSET + (i / params.blocks_x) * STRIDE * N;
  let ax = select(0u, x - RADIUS, x >= RADIUS);
  let ay = select(0u, y - RADIUS, y >= RADIUS);
  let x1 = min(x + RADIUS, params.width - N);
  let y1 = min(y + RADIUS, params.height - N);
  let aw = x1 + N - ax;
  let ah = y1 + N - ay;
  reference[t] = matched(0u, (y + t / N) * params.width + x + t % N);
  for (var j = t; j < aw * ah; j += 64u) {
    area[j] = matched(1u, (ay + j / aw) * params.width + ax + j % aw);
  }
  workgroupBarrier();
  let across = x1 - ax + 1u;
  let count = across * (y1 - ay + 1u);
  var best = 3.0e38;
  var rank = 0xffffffffu;
  for (var m = t; m < count; m += 64u) {
    let base = (m / across) * aw + m % across;
    var sum = 0.0;
    for (var row = 0u; row < N; row++) {
      let r = row * N;
      let c = base + row * aw;
      let d0 = reference[r] - area[c];
      let d1 = reference[r + 1u] - area[c + 1u];
      let d2 = reference[r + 2u] - area[c + 2u];
      let d3 = reference[r + 3u] - area[c + 3u];
      let d4 = reference[r + 4u] - area[c + 4u];
      let d5 = reference[r + 5u] - area[c + 5u];
      let d6 = reference[r + 6u] - area[c + 6u];
      let d7 = reference[r + 7u] - area[c + 7u];
      // In the order noise.rs adds them.
      sum += ((d0 * d0 + d4 * d4) + (d1 * d1 + d5 * d5)) + ((d2 * d2 + d6 * d6) + (d3 * d3 + d7 * d7));
    }
    if (sum < best) {
      best = sum;
      rank = m;
    }
  }
  best_sum[t] = best;
  best_rank[t] = rank;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (t < stride) {
      let other = t + stride;
      if (best_sum[other] < best_sum[t] || (best_sum[other] == best_sum[t] && best_rank[other] < best_rank[t])) {
        best_sum[t] = best_sum[other];
        best_rank[t] = best_rank[other];
      }
    }
    workgroupBarrier();
  }
  let m = best_rank[0];
  let mx = ax + m % across;
  let my = ay + m / across;
  let a = noisy[params.noisy_a + (y + t / N) * params.width + x + t % N];
  let b = noisy[params.noisy_b + (my + t / N) * params.width + mx + t % N];
  let coefficient = dct(t, (a - b) * 0.70710677);
  let at = params.pair_out + i * BLOCK;
  out[at + t] = coefficient;
  // The mean of the first frame's basic block, added up in order.
  tile[t] = basic[params.basic_a + (y + t / N) * params.width + x + t % N];
  workgroupBarrier();
  if (t == 0u) {
    var sum = 0.0;
    for (var p = 0u; p < AREA; p++) {
      sum += tile[p];
    }
    out[at + 64u] = best_sum[0];
    out[at + 65u] = sum / f32(AREA);
  }
}

// ---- residual_blocks ----

// As NoiseStats::measure_residual takes each block, of the second frame.
@compute @workgroup_size(64)
fn residual_blocks(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let i = block_index(wg);
  if (i >= params.blocks_x * params.blocks_y) {
    return;
  }
  let x = OFFSET + (i % params.blocks_x) * STRIDE * N;
  let y = OFFSET + (i / params.blocks_x) * STRIDE * N;
  let p = (y + t / N) * params.width + x + t % N;
  let b = basic[params.basic_b + p];
  let at = params.residual_out + i * BLOCK;
  out[at + t] = dct(t, noisy[params.noisy_b + p] - b);
  let kept = dct(t, b);
  workgroupBarrier();
  tile[t] = kept * kept;
  workgroupBarrier();
  if (t == 0u) {
    var sum = 0.0;
    for (var f = 1u; f < AREA; f++) {
      sum += tile[f];
    }
    out[at + 64u] = sum;
    out[at + 65u] = kept;
  }
}