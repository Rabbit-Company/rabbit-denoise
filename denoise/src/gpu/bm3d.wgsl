// The collaborative filter of bm3d.rs on the GPU, finding the same groups.
//
// - find_groups: 64 threads take 4x4 reference patches, 4 threads each, searching a region
//   held in workgroup memory.
// - filter_groups: 64 threads take the groups of 4x4 reference patches in turn, one thread
//   per coefficient, and add the current frame's patches into a region in workgroup memory,
//   which then goes into the accumulators.
// - normalize: one thread per pixel divides the sums.

const N: u32 = 8u;
const AREA: u32 = 64u;
const TOP_K: u32 = 8u;
const MAX_GROUP: u32 = 16u;
// Per reference in the groups buffer: the count, then up to MAX_GROUP packed matches.
const GROUP_STRIDE: u32 = 17u;
const BANDS: u32 = 16u;
const FAR: f32 = 3.0e38;
// Reference patches per workgroup side in find_groups and threads per reference.
// gpu/mod.rs sets the region side for the step and search radius.
const TILE_REFS: u32 = 4u;
const LANES: u32 = 4u;
// How far past its references filter_groups sums in workgroup memory. Members further out
// go to the accumulators. A smaller region lets more workgroups run at once.
const FILTER_MARGIN: u32 = 2u;
//#TILE_SIDE
// Offsets into the tables buffer: Kaiser window, 8x8 DCT basis, then the group DCT basis
// of length n at GROUP_BASIS + n * 256, then the reference positions.
const KAISER: u32 = 0u;
const DCT: u32 = 64u;
const GROUP_BASIS: u32 = 128u;
const POSITIONS: u32 = 128u + 17u * 256u;

struct Params {
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
  // 0 for the hard threshold stage, 1 for the Wiener stage.
  stage: u32,
  lambda: f32,
  distance_factor: f32,
  // On the plane's own scale, per sample.
  distance_floor: f32,
  maxval: f32,
  plane_size: u32,
  // Workgroups along x in every dispatch, the rest along y.
  dispatch_x: u32,
  // Where in tables each frame's slot in the noisy ring starts, then in the guide ring.
  slots: u32,
  // 1 to follow motion with block motion fields, measured on luma.
  motion: u32,
  // The fields' 16x16 blocks across and down, on the luma plane.
  field_blocks_x: u32,
  field_blocks_y: u32,
  // How many luma samples one sample of this plane covers across and down.
  motion_scale_x: u32,
  motion_scale_y: u32,
  // The luma plane averaged 2x2 that the fields are measured on, and the move range on it.
  half_width: u32,
  half_height: u32,
  half_range: u32,
  // Offsets into tables: each step's field slot, backward first, the half planes to
  // measure, and the fields to measure as (slot, source slot, target slot).
  field_map: u32,
  half_jobs: u32,
  half_count: u32,
  field_jobs: u32,
  field_count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> noisy: array<f32>;
@group(0) @binding(2) var<storage, read> guide: array<f32>;
// BANDS spectra of 64 grain variances, then each band's mean variance per sample.
@group(0) @binding(3) var<storage, read> model: array<f32>;
// Numerators, then denominators. gpu/mod.rs puts in float atomics or a compare and swap loop.
//#ACCUMULATOR
@group(0) @binding(5) var<storage, read> tables: array<f32>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;
@group(0) @binding(7) var<storage, read_write> groups: array<u32>;
// Every block's move per slot, as (dx + 512) | (dy + 512) << 16 in luma samples.
@group(0) @binding(8) var<storage, read_write> fields: array<u32>;
// The same fields for find_groups, which only reads them. Planes that share no buffer they
// write run at the same time.
@group(0) @binding(10) var<storage, read> fields_in: array<u32>;
// Luma planes averaged 2x2, one per guide slot.
@group(0) @binding(9) var<storage, read_write> half_planes: array<f32>;

fn variance_at(mean: f32, f: u32) -> f32 {
  let position = clamp(mean / (params.maxval + 1.0) * f32(BANDS) - 0.5, 0.0, f32(BANDS - 1u));
  let lo = floor(position);
  let hi = min(lo + 1.0, f32(BANDS - 1u));
  let t = position - lo;
  return model[u32(lo) * AREA + f] * (1.0 - t) + model[u32(hi) * AREA + f] * t;
}

// Where a window frame's plane starts in the noisy and the guide ring.
fn noisy_base(frame: u32) -> u32 {
  return u32(tables[params.slots + frame]) * params.plane_size;
}

fn guide_base(frame: u32) -> u32 {
  return u32(tables[params.slots + params.frames + frame]) * params.plane_size;
}

// The grain variance per sample at a brightness, from the band means after the spectra.
fn mean_variance_at(mean: f32) -> f32 {
  let position = clamp(mean / (params.maxval + 1.0) * f32(BANDS) - 0.5, 0.0, f32(BANDS - 1u));
  let lo = floor(position);
  let hi = min(lo + 1.0, f32(BANDS - 1u));
  let t = position - lo;
  return model[BANDS * AREA + u32(lo)] * (1.0 - t) + model[BANDS * AREA + u32(hi)] * t;
}

fn pack(frame: u32, x: u32, y: u32) -> u32 {
  return x | (y << 13u) | (frame << 26u);
}

fn unpack_x(p: u32) -> u32 { return p & 0x1fffu; }
fn unpack_y(p: u32) -> u32 { return (p >> 13u) & 0x1fffu; }
fn unpack_frame(p: u32) -> u32 { return p >> 26u; }

fn apart(a: u32, b: u32, shift: u32) -> bool {
  let ax = unpack_x(a);
  let bx = unpack_x(b);
  let ay = unpack_y(a);
  let by = unpack_y(b);
  let dx = select(bx - ax, ax - bx, ax >= bx);
  let dy = select(by - ay, ay - by, ay >= by);
  return dx >= shift || dy >= shift;
}

// ---- find_groups ----

// The searched region of the current frame, and of the neighbouring frame being searched.
var<workgroup> current_tile: array<f32, TILE_SIDE * TILE_SIDE>;
// Also holds the lanes' best distances between a search and its merge.
var<workgroup> other_tile: array<f32, OTHER_AREA>;
// The region: x0, y0, width, height.
var<workgroup> region: vec4<u32>;
// Every lane's best TOP_K candidates of the frame being searched, for each reference.
var<workgroup> lane_position: array<u32, 512>;
// Where each reference's track goes next, and whether it goes on.
var<workgroup> track: array<u32, 16>;
var<workgroup> track_alive: array<u32, 16>;

// This lane's best TOP_K candidates in registers, d0 the best, unused ones FAR.
var<private> d0: f32;
var<private> q0: u32;
var<private> d1: f32;
var<private> q1: u32;
var<private> d2: f32;
var<private> q2: u32;
var<private> d3: f32;
var<private> q3: u32;
var<private> d4: f32;
var<private> q4: u32;
var<private> d5: f32;
var<private> q5: u32;
var<private> d6: f32;
var<private> q6: u32;
var<private> d7: f32;
var<private> q7: u32;
// Lane 0's group after the reference, in registers, e0 the best, unused ones FAR.
var<private> e0: f32;
var<private> g0: u32;
var<private> e1: f32;
var<private> g1: u32;
var<private> e2: f32;
var<private> g2: u32;
var<private> e3: f32;
var<private> g3: u32;
var<private> e4: f32;
var<private> g4: u32;
var<private> e5: f32;
var<private> g5: u32;
var<private> e6: f32;
var<private> g6: u32;
var<private> e7: f32;
var<private> g7: u32;
var<private> e8: f32;
var<private> g8: u32;
var<private> e9: f32;
var<private> g9: u32;
var<private> e10: f32;
var<private> g10: u32;
var<private> e11: f32;
var<private> g11: u32;
var<private> e12: f32;
var<private> g12: u32;
var<private> e13: f32;
var<private> g13: u32;
var<private> e14: f32;
var<private> g14: u32;
// The reference patch, r0 and r1 its first row, and where it sits in current_tile.
var<private> ref_base: u32;

// Whether candidate a comes before b: smaller distance, then earlier in search order.
fn before(da: f32, a: u32, db: f32, b: u32) -> bool {
  if (da != db) {
    return da < db;
  }
  let ya = unpack_y(a);
  let yb = unpack_y(b);
  return ya < yb || (ya == yb && unpack_x(a) < unpack_x(b));
}

// Keeps this lane's TOP_K best. Among equals the new one goes after.
fn offer(d: f32, position: u32) {
  if (d >= d7) {
    return;
  }
  d7 = d;
  q7 = position;
  if (d7 < d6) {
    let td = d6;
    let tq = q6;
    d6 = d7;
    q6 = q7;
    d7 = td;
    q7 = tq;
  }
  if (d6 < d5) {
    let td = d5;
    let tq = q5;
    d5 = d6;
    q5 = q6;
    d6 = td;
    q6 = tq;
  }
  if (d5 < d4) {
    let td = d4;
    let tq = q4;
    d4 = d5;
    q4 = q5;
    d5 = td;
    q5 = tq;
  }
  if (d4 < d3) {
    let td = d3;
    let tq = q3;
    d3 = d4;
    q3 = q4;
    d4 = td;
    q4 = tq;
  }
  if (d3 < d2) {
    let td = d2;
    let tq = q2;
    d2 = d3;
    q2 = q3;
    d3 = td;
    q3 = tq;
  }
  if (d2 < d1) {
    let td = d1;
    let tq = q1;
    d1 = d2;
    q1 = q2;
    d2 = td;
    q2 = tq;
  }
  if (d1 < d0) {
    let td = d0;
    let tq = q0;
    d0 = d1;
    q0 = q1;
    d1 = td;
    q1 = tq;
  }
}

fn clear_top() {
  d0 = FAR;
  d1 = FAR;
  d2 = FAR;
  d3 = FAR;
  d4 = FAR;
  d5 = FAR;
  d6 = FAR;
  d7 = FAR;
}

// Keeps the best MAX_GROUP - 1 matches, the earlier first among equals.
fn choose(d: f32, position: u32) {
  if (d >= e14) {
    return;
  }
  e14 = d;
  g14 = position;
  if (e14 < e13) {
    let td = e13;
    let tg = g13;
    e13 = e14;
    g13 = g14;
    e14 = td;
    g14 = tg;
  }
  if (e13 < e12) {
    let td = e12;
    let tg = g12;
    e12 = e13;
    g12 = g13;
    e13 = td;
    g13 = tg;
  }
  if (e12 < e11) {
    let td = e11;
    let tg = g11;
    e11 = e12;
    g11 = g12;
    e12 = td;
    g12 = tg;
  }
  if (e11 < e10) {
    let td = e10;
    let tg = g10;
    e10 = e11;
    g10 = g11;
    e11 = td;
    g11 = tg;
  }
  if (e10 < e9) {
    let td = e9;
    let tg = g9;
    e9 = e10;
    g9 = g10;
    e10 = td;
    g10 = tg;
  }
  if (e9 < e8) {
    let td = e8;
    let tg = g8;
    e8 = e9;
    g8 = g9;
    e9 = td;
    g9 = tg;
  }
  if (e8 < e7) {
    let td = e7;
    let tg = g7;
    e7 = e8;
    g7 = g8;
    e8 = td;
    g8 = tg;
  }
  if (e7 < e6) {
    let td = e6;
    let tg = g6;
    e6 = e7;
    g6 = g7;
    e7 = td;
    g7 = tg;
  }
  if (e6 < e5) {
    let td = e5;
    let tg = g5;
    e5 = e6;
    g5 = g6;
    e6 = td;
    g6 = tg;
  }
  if (e5 < e4) {
    let td = e4;
    let tg = g4;
    e4 = e5;
    g4 = g5;
    e5 = td;
    g5 = tg;
  }
  if (e4 < e3) {
    let td = e3;
    let tg = g3;
    e3 = e4;
    g3 = g4;
    e4 = td;
    g4 = tg;
  }
  if (e3 < e2) {
    let td = e2;
    let tg = g2;
    e2 = e3;
    g2 = g3;
    e3 = td;
    g3 = tg;
  }
  if (e2 < e1) {
    let td = e1;
    let tg = g1;
    e1 = e2;
    g1 = g2;
    e2 = td;
    g2 = tg;
  }
  if (e1 < e0) {
    let td = e0;
    let tg = g0;
    e0 = e1;
    g0 = g1;
    e1 = td;
    g1 = tg;
  }
}

fn clear_chosen() {
  e0 = FAR;
  e1 = FAR;
  e2 = FAR;
  e3 = FAR;
  e4 = FAR;
  e5 = FAR;
  e6 = FAR;
  e7 = FAR;
  e8 = FAR;
  e9 = FAR;
  e10 = FAR;
  e11 = FAR;
  e12 = FAR;
  e13 = FAR;
  e14 = FAR;
}

// Squared differences between the reference and the patch at `base`, rows `w` apart.
fn distance_current(base: u32, w: u32) -> f32 {
  var acc = vec4<f32>(0.0);
  let at0 = base + 0u * w;
  let a0 = vec4<f32>(current_tile[ref_base + 0u * region.z], current_tile[ref_base + 0u * region.z + 1u], current_tile[ref_base + 0u * region.z + 2u], current_tile[ref_base + 0u * region.z + 3u]) - vec4<f32>(current_tile[at0], current_tile[at0 + 1u], current_tile[at0 + 2u], current_tile[at0 + 3u]);
  let b0 = vec4<f32>(current_tile[ref_base + 0u * region.z + 4u], current_tile[ref_base + 0u * region.z + 5u], current_tile[ref_base + 0u * region.z + 6u], current_tile[ref_base + 0u * region.z + 7u]) - vec4<f32>(current_tile[at0 + 4u], current_tile[at0 + 5u], current_tile[at0 + 6u], current_tile[at0 + 7u]);
  acc += a0 * a0 + b0 * b0;
  let at1 = base + 1u * w;
  let a1 = vec4<f32>(current_tile[ref_base + 1u * region.z], current_tile[ref_base + 1u * region.z + 1u], current_tile[ref_base + 1u * region.z + 2u], current_tile[ref_base + 1u * region.z + 3u]) - vec4<f32>(current_tile[at1], current_tile[at1 + 1u], current_tile[at1 + 2u], current_tile[at1 + 3u]);
  let b1 = vec4<f32>(current_tile[ref_base + 1u * region.z + 4u], current_tile[ref_base + 1u * region.z + 5u], current_tile[ref_base + 1u * region.z + 6u], current_tile[ref_base + 1u * region.z + 7u]) - vec4<f32>(current_tile[at1 + 4u], current_tile[at1 + 5u], current_tile[at1 + 6u], current_tile[at1 + 7u]);
  acc += a1 * a1 + b1 * b1;
  let at2 = base + 2u * w;
  let a2 = vec4<f32>(current_tile[ref_base + 2u * region.z], current_tile[ref_base + 2u * region.z + 1u], current_tile[ref_base + 2u * region.z + 2u], current_tile[ref_base + 2u * region.z + 3u]) - vec4<f32>(current_tile[at2], current_tile[at2 + 1u], current_tile[at2 + 2u], current_tile[at2 + 3u]);
  let b2 = vec4<f32>(current_tile[ref_base + 2u * region.z + 4u], current_tile[ref_base + 2u * region.z + 5u], current_tile[ref_base + 2u * region.z + 6u], current_tile[ref_base + 2u * region.z + 7u]) - vec4<f32>(current_tile[at2 + 4u], current_tile[at2 + 5u], current_tile[at2 + 6u], current_tile[at2 + 7u]);
  acc += a2 * a2 + b2 * b2;
  let at3 = base + 3u * w;
  let a3 = vec4<f32>(current_tile[ref_base + 3u * region.z], current_tile[ref_base + 3u * region.z + 1u], current_tile[ref_base + 3u * region.z + 2u], current_tile[ref_base + 3u * region.z + 3u]) - vec4<f32>(current_tile[at3], current_tile[at3 + 1u], current_tile[at3 + 2u], current_tile[at3 + 3u]);
  let b3 = vec4<f32>(current_tile[ref_base + 3u * region.z + 4u], current_tile[ref_base + 3u * region.z + 5u], current_tile[ref_base + 3u * region.z + 6u], current_tile[ref_base + 3u * region.z + 7u]) - vec4<f32>(current_tile[at3 + 4u], current_tile[at3 + 5u], current_tile[at3 + 6u], current_tile[at3 + 7u]);
  acc += a3 * a3 + b3 * b3;
  let at4 = base + 4u * w;
  let a4 = vec4<f32>(current_tile[ref_base + 4u * region.z], current_tile[ref_base + 4u * region.z + 1u], current_tile[ref_base + 4u * region.z + 2u], current_tile[ref_base + 4u * region.z + 3u]) - vec4<f32>(current_tile[at4], current_tile[at4 + 1u], current_tile[at4 + 2u], current_tile[at4 + 3u]);
  let b4 = vec4<f32>(current_tile[ref_base + 4u * region.z + 4u], current_tile[ref_base + 4u * region.z + 5u], current_tile[ref_base + 4u * region.z + 6u], current_tile[ref_base + 4u * region.z + 7u]) - vec4<f32>(current_tile[at4 + 4u], current_tile[at4 + 5u], current_tile[at4 + 6u], current_tile[at4 + 7u]);
  acc += a4 * a4 + b4 * b4;
  let at5 = base + 5u * w;
  let a5 = vec4<f32>(current_tile[ref_base + 5u * region.z], current_tile[ref_base + 5u * region.z + 1u], current_tile[ref_base + 5u * region.z + 2u], current_tile[ref_base + 5u * region.z + 3u]) - vec4<f32>(current_tile[at5], current_tile[at5 + 1u], current_tile[at5 + 2u], current_tile[at5 + 3u]);
  let b5 = vec4<f32>(current_tile[ref_base + 5u * region.z + 4u], current_tile[ref_base + 5u * region.z + 5u], current_tile[ref_base + 5u * region.z + 6u], current_tile[ref_base + 5u * region.z + 7u]) - vec4<f32>(current_tile[at5 + 4u], current_tile[at5 + 5u], current_tile[at5 + 6u], current_tile[at5 + 7u]);
  acc += a5 * a5 + b5 * b5;
  let at6 = base + 6u * w;
  let a6 = vec4<f32>(current_tile[ref_base + 6u * region.z], current_tile[ref_base + 6u * region.z + 1u], current_tile[ref_base + 6u * region.z + 2u], current_tile[ref_base + 6u * region.z + 3u]) - vec4<f32>(current_tile[at6], current_tile[at6 + 1u], current_tile[at6 + 2u], current_tile[at6 + 3u]);
  let b6 = vec4<f32>(current_tile[ref_base + 6u * region.z + 4u], current_tile[ref_base + 6u * region.z + 5u], current_tile[ref_base + 6u * region.z + 6u], current_tile[ref_base + 6u * region.z + 7u]) - vec4<f32>(current_tile[at6 + 4u], current_tile[at6 + 5u], current_tile[at6 + 6u], current_tile[at6 + 7u]);
  acc += a6 * a6 + b6 * b6;
  let at7 = base + 7u * w;
  let a7 = vec4<f32>(current_tile[ref_base + 7u * region.z], current_tile[ref_base + 7u * region.z + 1u], current_tile[ref_base + 7u * region.z + 2u], current_tile[ref_base + 7u * region.z + 3u]) - vec4<f32>(current_tile[at7], current_tile[at7 + 1u], current_tile[at7 + 2u], current_tile[at7 + 3u]);
  let b7 = vec4<f32>(current_tile[ref_base + 7u * region.z + 4u], current_tile[ref_base + 7u * region.z + 5u], current_tile[ref_base + 7u * region.z + 6u], current_tile[ref_base + 7u * region.z + 7u]) - vec4<f32>(current_tile[at7 + 4u], current_tile[at7 + 5u], current_tile[at7 + 6u], current_tile[at7 + 7u]);
  acc += a7 * a7 + b7 * b7;
  return acc.x + acc.y + acc.z + acc.w;
}

fn distance_other(base: u32, w: u32) -> f32 {
  var acc = vec4<f32>(0.0);
  let at0 = base + 0u * w;
  let a0 = vec4<f32>(current_tile[ref_base + 0u * region.z], current_tile[ref_base + 0u * region.z + 1u], current_tile[ref_base + 0u * region.z + 2u], current_tile[ref_base + 0u * region.z + 3u]) - vec4<f32>(other_tile[at0], other_tile[at0 + 1u], other_tile[at0 + 2u], other_tile[at0 + 3u]);
  let b0 = vec4<f32>(current_tile[ref_base + 0u * region.z + 4u], current_tile[ref_base + 0u * region.z + 5u], current_tile[ref_base + 0u * region.z + 6u], current_tile[ref_base + 0u * region.z + 7u]) - vec4<f32>(other_tile[at0 + 4u], other_tile[at0 + 5u], other_tile[at0 + 6u], other_tile[at0 + 7u]);
  acc += a0 * a0 + b0 * b0;
  let at1 = base + 1u * w;
  let a1 = vec4<f32>(current_tile[ref_base + 1u * region.z], current_tile[ref_base + 1u * region.z + 1u], current_tile[ref_base + 1u * region.z + 2u], current_tile[ref_base + 1u * region.z + 3u]) - vec4<f32>(other_tile[at1], other_tile[at1 + 1u], other_tile[at1 + 2u], other_tile[at1 + 3u]);
  let b1 = vec4<f32>(current_tile[ref_base + 1u * region.z + 4u], current_tile[ref_base + 1u * region.z + 5u], current_tile[ref_base + 1u * region.z + 6u], current_tile[ref_base + 1u * region.z + 7u]) - vec4<f32>(other_tile[at1 + 4u], other_tile[at1 + 5u], other_tile[at1 + 6u], other_tile[at1 + 7u]);
  acc += a1 * a1 + b1 * b1;
  let at2 = base + 2u * w;
  let a2 = vec4<f32>(current_tile[ref_base + 2u * region.z], current_tile[ref_base + 2u * region.z + 1u], current_tile[ref_base + 2u * region.z + 2u], current_tile[ref_base + 2u * region.z + 3u]) - vec4<f32>(other_tile[at2], other_tile[at2 + 1u], other_tile[at2 + 2u], other_tile[at2 + 3u]);
  let b2 = vec4<f32>(current_tile[ref_base + 2u * region.z + 4u], current_tile[ref_base + 2u * region.z + 5u], current_tile[ref_base + 2u * region.z + 6u], current_tile[ref_base + 2u * region.z + 7u]) - vec4<f32>(other_tile[at2 + 4u], other_tile[at2 + 5u], other_tile[at2 + 6u], other_tile[at2 + 7u]);
  acc += a2 * a2 + b2 * b2;
  let at3 = base + 3u * w;
  let a3 = vec4<f32>(current_tile[ref_base + 3u * region.z], current_tile[ref_base + 3u * region.z + 1u], current_tile[ref_base + 3u * region.z + 2u], current_tile[ref_base + 3u * region.z + 3u]) - vec4<f32>(other_tile[at3], other_tile[at3 + 1u], other_tile[at3 + 2u], other_tile[at3 + 3u]);
  let b3 = vec4<f32>(current_tile[ref_base + 3u * region.z + 4u], current_tile[ref_base + 3u * region.z + 5u], current_tile[ref_base + 3u * region.z + 6u], current_tile[ref_base + 3u * region.z + 7u]) - vec4<f32>(other_tile[at3 + 4u], other_tile[at3 + 5u], other_tile[at3 + 6u], other_tile[at3 + 7u]);
  acc += a3 * a3 + b3 * b3;
  let at4 = base + 4u * w;
  let a4 = vec4<f32>(current_tile[ref_base + 4u * region.z], current_tile[ref_base + 4u * region.z + 1u], current_tile[ref_base + 4u * region.z + 2u], current_tile[ref_base + 4u * region.z + 3u]) - vec4<f32>(other_tile[at4], other_tile[at4 + 1u], other_tile[at4 + 2u], other_tile[at4 + 3u]);
  let b4 = vec4<f32>(current_tile[ref_base + 4u * region.z + 4u], current_tile[ref_base + 4u * region.z + 5u], current_tile[ref_base + 4u * region.z + 6u], current_tile[ref_base + 4u * region.z + 7u]) - vec4<f32>(other_tile[at4 + 4u], other_tile[at4 + 5u], other_tile[at4 + 6u], other_tile[at4 + 7u]);
  acc += a4 * a4 + b4 * b4;
  let at5 = base + 5u * w;
  let a5 = vec4<f32>(current_tile[ref_base + 5u * region.z], current_tile[ref_base + 5u * region.z + 1u], current_tile[ref_base + 5u * region.z + 2u], current_tile[ref_base + 5u * region.z + 3u]) - vec4<f32>(other_tile[at5], other_tile[at5 + 1u], other_tile[at5 + 2u], other_tile[at5 + 3u]);
  let b5 = vec4<f32>(current_tile[ref_base + 5u * region.z + 4u], current_tile[ref_base + 5u * region.z + 5u], current_tile[ref_base + 5u * region.z + 6u], current_tile[ref_base + 5u * region.z + 7u]) - vec4<f32>(other_tile[at5 + 4u], other_tile[at5 + 5u], other_tile[at5 + 6u], other_tile[at5 + 7u]);
  acc += a5 * a5 + b5 * b5;
  let at6 = base + 6u * w;
  let a6 = vec4<f32>(current_tile[ref_base + 6u * region.z], current_tile[ref_base + 6u * region.z + 1u], current_tile[ref_base + 6u * region.z + 2u], current_tile[ref_base + 6u * region.z + 3u]) - vec4<f32>(other_tile[at6], other_tile[at6 + 1u], other_tile[at6 + 2u], other_tile[at6 + 3u]);
  let b6 = vec4<f32>(current_tile[ref_base + 6u * region.z + 4u], current_tile[ref_base + 6u * region.z + 5u], current_tile[ref_base + 6u * region.z + 6u], current_tile[ref_base + 6u * region.z + 7u]) - vec4<f32>(other_tile[at6 + 4u], other_tile[at6 + 5u], other_tile[at6 + 6u], other_tile[at6 + 7u]);
  acc += a6 * a6 + b6 * b6;
  let at7 = base + 7u * w;
  let a7 = vec4<f32>(current_tile[ref_base + 7u * region.z], current_tile[ref_base + 7u * region.z + 1u], current_tile[ref_base + 7u * region.z + 2u], current_tile[ref_base + 7u * region.z + 3u]) - vec4<f32>(other_tile[at7], other_tile[at7 + 1u], other_tile[at7 + 2u], other_tile[at7 + 3u]);
  let b7 = vec4<f32>(current_tile[ref_base + 7u * region.z + 4u], current_tile[ref_base + 7u * region.z + 5u], current_tile[ref_base + 7u * region.z + 6u], current_tile[ref_base + 7u * region.z + 7u]) - vec4<f32>(other_tile[at7 + 4u], other_tile[at7 + 5u], other_tile[at7 + 6u], other_tile[at7 + 7u]);
  acc += a7 * a7 + b7 * b7;
  return acc.x + acc.y + acc.z + acc.w;
}

fn distance_plane(base: u32, w: u32) -> f32 {
  var acc = vec4<f32>(0.0);
  let at0 = base + 0u * w;
  let a0 = vec4<f32>(current_tile[ref_base + 0u * region.z], current_tile[ref_base + 0u * region.z + 1u], current_tile[ref_base + 0u * region.z + 2u], current_tile[ref_base + 0u * region.z + 3u]) - vec4<f32>(guide[at0], guide[at0 + 1u], guide[at0 + 2u], guide[at0 + 3u]);
  let b0 = vec4<f32>(current_tile[ref_base + 0u * region.z + 4u], current_tile[ref_base + 0u * region.z + 5u], current_tile[ref_base + 0u * region.z + 6u], current_tile[ref_base + 0u * region.z + 7u]) - vec4<f32>(guide[at0 + 4u], guide[at0 + 5u], guide[at0 + 6u], guide[at0 + 7u]);
  acc += a0 * a0 + b0 * b0;
  let at1 = base + 1u * w;
  let a1 = vec4<f32>(current_tile[ref_base + 1u * region.z], current_tile[ref_base + 1u * region.z + 1u], current_tile[ref_base + 1u * region.z + 2u], current_tile[ref_base + 1u * region.z + 3u]) - vec4<f32>(guide[at1], guide[at1 + 1u], guide[at1 + 2u], guide[at1 + 3u]);
  let b1 = vec4<f32>(current_tile[ref_base + 1u * region.z + 4u], current_tile[ref_base + 1u * region.z + 5u], current_tile[ref_base + 1u * region.z + 6u], current_tile[ref_base + 1u * region.z + 7u]) - vec4<f32>(guide[at1 + 4u], guide[at1 + 5u], guide[at1 + 6u], guide[at1 + 7u]);
  acc += a1 * a1 + b1 * b1;
  let at2 = base + 2u * w;
  let a2 = vec4<f32>(current_tile[ref_base + 2u * region.z], current_tile[ref_base + 2u * region.z + 1u], current_tile[ref_base + 2u * region.z + 2u], current_tile[ref_base + 2u * region.z + 3u]) - vec4<f32>(guide[at2], guide[at2 + 1u], guide[at2 + 2u], guide[at2 + 3u]);
  let b2 = vec4<f32>(current_tile[ref_base + 2u * region.z + 4u], current_tile[ref_base + 2u * region.z + 5u], current_tile[ref_base + 2u * region.z + 6u], current_tile[ref_base + 2u * region.z + 7u]) - vec4<f32>(guide[at2 + 4u], guide[at2 + 5u], guide[at2 + 6u], guide[at2 + 7u]);
  acc += a2 * a2 + b2 * b2;
  let at3 = base + 3u * w;
  let a3 = vec4<f32>(current_tile[ref_base + 3u * region.z], current_tile[ref_base + 3u * region.z + 1u], current_tile[ref_base + 3u * region.z + 2u], current_tile[ref_base + 3u * region.z + 3u]) - vec4<f32>(guide[at3], guide[at3 + 1u], guide[at3 + 2u], guide[at3 + 3u]);
  let b3 = vec4<f32>(current_tile[ref_base + 3u * region.z + 4u], current_tile[ref_base + 3u * region.z + 5u], current_tile[ref_base + 3u * region.z + 6u], current_tile[ref_base + 3u * region.z + 7u]) - vec4<f32>(guide[at3 + 4u], guide[at3 + 5u], guide[at3 + 6u], guide[at3 + 7u]);
  acc += a3 * a3 + b3 * b3;
  let at4 = base + 4u * w;
  let a4 = vec4<f32>(current_tile[ref_base + 4u * region.z], current_tile[ref_base + 4u * region.z + 1u], current_tile[ref_base + 4u * region.z + 2u], current_tile[ref_base + 4u * region.z + 3u]) - vec4<f32>(guide[at4], guide[at4 + 1u], guide[at4 + 2u], guide[at4 + 3u]);
  let b4 = vec4<f32>(current_tile[ref_base + 4u * region.z + 4u], current_tile[ref_base + 4u * region.z + 5u], current_tile[ref_base + 4u * region.z + 6u], current_tile[ref_base + 4u * region.z + 7u]) - vec4<f32>(guide[at4 + 4u], guide[at4 + 5u], guide[at4 + 6u], guide[at4 + 7u]);
  acc += a4 * a4 + b4 * b4;
  let at5 = base + 5u * w;
  let a5 = vec4<f32>(current_tile[ref_base + 5u * region.z], current_tile[ref_base + 5u * region.z + 1u], current_tile[ref_base + 5u * region.z + 2u], current_tile[ref_base + 5u * region.z + 3u]) - vec4<f32>(guide[at5], guide[at5 + 1u], guide[at5 + 2u], guide[at5 + 3u]);
  let b5 = vec4<f32>(current_tile[ref_base + 5u * region.z + 4u], current_tile[ref_base + 5u * region.z + 5u], current_tile[ref_base + 5u * region.z + 6u], current_tile[ref_base + 5u * region.z + 7u]) - vec4<f32>(guide[at5 + 4u], guide[at5 + 5u], guide[at5 + 6u], guide[at5 + 7u]);
  acc += a5 * a5 + b5 * b5;
  let at6 = base + 6u * w;
  let a6 = vec4<f32>(current_tile[ref_base + 6u * region.z], current_tile[ref_base + 6u * region.z + 1u], current_tile[ref_base + 6u * region.z + 2u], current_tile[ref_base + 6u * region.z + 3u]) - vec4<f32>(guide[at6], guide[at6 + 1u], guide[at6 + 2u], guide[at6 + 3u]);
  let b6 = vec4<f32>(current_tile[ref_base + 6u * region.z + 4u], current_tile[ref_base + 6u * region.z + 5u], current_tile[ref_base + 6u * region.z + 6u], current_tile[ref_base + 6u * region.z + 7u]) - vec4<f32>(guide[at6 + 4u], guide[at6 + 5u], guide[at6 + 6u], guide[at6 + 7u]);
  acc += a6 * a6 + b6 * b6;
  let at7 = base + 7u * w;
  let a7 = vec4<f32>(current_tile[ref_base + 7u * region.z], current_tile[ref_base + 7u * region.z + 1u], current_tile[ref_base + 7u * region.z + 2u], current_tile[ref_base + 7u * region.z + 3u]) - vec4<f32>(guide[at7], guide[at7 + 1u], guide[at7 + 2u], guide[at7 + 3u]);
  let b7 = vec4<f32>(current_tile[ref_base + 7u * region.z + 4u], current_tile[ref_base + 7u * region.z + 5u], current_tile[ref_base + 7u * region.z + 6u], current_tile[ref_base + 7u * region.z + 7u]) - vec4<f32>(guide[at7 + 4u], guide[at7 + 5u], guide[at7 + 6u], guide[at7 + 7u]);
  acc += a7 * a7 + b7 * b7;
  return acc.x + acc.y + acc.z + acc.w;
}

// Squared differences between the reference and the patch at (x, y), or a negative number
// past the limit. Patches outside the region come from the plane.
fn distance(other: bool, frame: u32, x: u32, y: u32, limit: f32) -> f32 {
  let inside = x >= region.x && y >= region.y && x + N <= region.x + region.z && y + N <= region.y + region.w;
  var sum: f32;
  if (inside) {
    let base = (y - region.y) * region.z + x - region.x;
    if (other) {
      sum = distance_other(base, region.z);
    } else {
      sum = distance_current(base, region.z);
    }
  } else {
    sum = distance_plane(guide_base(frame) + y * params.width + x, params.width);
  }
  return select(-1.0, sum, sum <= limit);
}

// This lane's part of a search: every LANES-th column of the window, in search order.
fn search(other: bool, lane: u32, frame: u32, cx: u32, cy: u32, radius: u32, limit: f32, rx: u32, ry: u32, skip_near: bool) {
  clear_top();
  let x0 = select(0u, cx - radius, cx >= radius);
  let y0 = select(0u, cy - radius, cy >= radius);
  let x1 = min(cx + radius, params.width - N);
  let y1 = min(cy + radius, params.height - N);
  // Lane l takes the columns l, l + LANES, ... row by row.
  for (var y = y0; y <= y1; y++) {
    for (var x = x0 + lane; x <= x1; x += LANES) {
    if (skip_near) {
      let dx = select(rx - x, x - rx, x >= rx);
      let dy = select(ry - y, y - ry, y >= ry);
      if (dx < params.min_shift && dy < params.min_shift) {
        continue;
      }
    }
    let d = distance(other, frame, x, y, limit);
    if (d >= 0.0) {
      offer(d, pack(frame, x, y));
    }
    }
  }
}

// Lane 0 merges the lanes' lists, takes up to per_frame distinct matches into the group and
// returns how many there were. The best is the next centre.
fn merge_and_take(slot: u32) -> u32 {
  // Where each lane's list is read next, and the matches taken from this frame.
  var h0 = 0u;
  var h1 = 0u;
  var h2 = 0u;
  var h3 = 0u;
  var t0 = 0u;
  var t1 = 0u;
  var t2 = 0u;
  var t3 = 0u;
  var t4 = 0u;
  var t5 = 0u;
  var t6 = 0u;
  var t7 = 0u;
  var count = 0u;
  var held = 0u;
  let base = slot * LANES * TOP_K;
  for (var k = 0u; k < TOP_K; k++) {
    var best = LANES;
    var best_d = FAR;
    var best_p = 0u;
    {
      let at = base + 0u * TOP_K + min(h0, TOP_K - 1u);
      let d = select(FAR, other_tile[at], h0 < TOP_K);
      if (d < FAR && (best == LANES || before(d, lane_position[at], best_d, best_p))) {
        best = 0u;
        best_d = d;
        best_p = lane_position[at];
      }
    }
    {
      let at = base + 1u * TOP_K + min(h1, TOP_K - 1u);
      let d = select(FAR, other_tile[at], h1 < TOP_K);
      if (d < FAR && (best == LANES || before(d, lane_position[at], best_d, best_p))) {
        best = 1u;
        best_d = d;
        best_p = lane_position[at];
      }
    }
    {
      let at = base + 2u * TOP_K + min(h2, TOP_K - 1u);
      let d = select(FAR, other_tile[at], h2 < TOP_K);
      if (d < FAR && (best == LANES || before(d, lane_position[at], best_d, best_p))) {
        best = 2u;
        best_d = d;
        best_p = lane_position[at];
      }
    }
    {
      let at = base + 3u * TOP_K + min(h3, TOP_K - 1u);
      let d = select(FAR, other_tile[at], h3 < TOP_K);
      if (d < FAR && (best == LANES || before(d, lane_position[at], best_d, best_p))) {
        best = 3u;
        best_d = d;
        best_p = lane_position[at];
      }
    }
    if (best == LANES) {
      break;
    }
    if (best == 0u) { h0 += 1u; }
    if (best == 1u) { h1 += 1u; }
    if (best == 2u) { h2 += 1u; }
    if (best == 3u) { h3 += 1u; }
    held += 1u;
    if (k == 0u) {
      track[slot] = best_p;
    }
    if (count == params.per_frame) {
      continue;
    }
    var ok = true;
    if (count > 0u && !apart(best_p, t0, params.min_shift)) { ok = false; }
    if (count > 1u && !apart(best_p, t1, params.min_shift)) { ok = false; }
    if (count > 2u && !apart(best_p, t2, params.min_shift)) { ok = false; }
    if (count > 3u && !apart(best_p, t3, params.min_shift)) { ok = false; }
    if (count > 4u && !apart(best_p, t4, params.min_shift)) { ok = false; }
    if (count > 5u && !apart(best_p, t5, params.min_shift)) { ok = false; }
    if (count > 6u && !apart(best_p, t6, params.min_shift)) { ok = false; }
    if (count > 7u && !apart(best_p, t7, params.min_shift)) { ok = false; }
    if (ok) {
      if (count == 0u) { t0 = best_p; }
      if (count == 1u) { t1 = best_p; }
      if (count == 2u) { t2 = best_p; }
      if (count == 3u) { t3 = best_p; }
      if (count == 4u) { t4 = best_p; }
      if (count == 5u) { t5 = best_p; }
      if (count == 6u) { t6 = best_p; }
      if (count == 7u) { t7 = best_p; }
      count += 1u;
      choose(best_d, best_p);
    }
  }
  return held;
}

fn publish(slot: u32, lane: u32) {
  let at = (slot * LANES + lane) * TOP_K;
  other_tile[at + 0u] = d0;
  lane_position[at + 0u] = q0;
  other_tile[at + 1u] = d1;
  lane_position[at + 1u] = q1;
  other_tile[at + 2u] = d2;
  lane_position[at + 2u] = q2;
  other_tile[at + 3u] = d3;
  lane_position[at + 3u] = q3;
  other_tile[at + 4u] = d4;
  lane_position[at + 4u] = q4;
  other_tile[at + 5u] = d5;
  lane_position[at + 5u] = q5;
  other_tile[at + 6u] = d6;
  lane_position[at + 6u] = q6;
  other_tile[at + 7u] = d7;
  lane_position[at + 7u] = q7;
}

fn load_tile(other: bool, frame: u32, t: u32) {
  let count = region.z * region.w;
  for (var i = t; i < count; i += 64u) {
    let v = guide[guide_base(frame) + (region.y + i / region.z) * params.width + region.x + i % region.z];
    if (other) {
      other_tile[i] = v;
    } else {
      current_tile[i] = v;
    }
  }
}

@compute @workgroup_size(64)
fn find_groups(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let tiles_x = (params.refs_x + TILE_REFS - 1u) / TILE_REFS;
  let tiles_y = (params.refs_y + TILE_REFS - 1u) / TILE_REFS;
  let tile = wg.y * params.dispatch_x + wg.x;
  if (tile >= tiles_x * tiles_y) {
    return;
  }
  let i0 = (tile % tiles_x) * TILE_REFS;
  let j0 = (tile / tiles_x) * TILE_REFS;
  let i1 = min(i0 + TILE_REFS, params.refs_x) - 1u;
  let j1 = min(j0 + TILE_REFS, params.refs_y) - 1u;
  // Thread t works on reference slot t / LANES as lane t % LANES.
  let slot = t / LANES;
  let lane = t % LANES;
  let i = i0 + slot % TILE_REFS;
  let j = j0 + slot / TILE_REFS;
  let in_tile = i <= i1 && j <= j1;
  let c = params.center;

  // The region every reference of the tile searches in its own frame.
  if (t == 0u) {
    let r = params.search_radius;
    let first_x = u32(tables[POSITIONS + i0]);
    let last_x = u32(tables[POSITIONS + i1]);
    let first_y = u32(tables[POSITIONS + params.refs_x + j0]);
    let last_y = u32(tables[POSITIONS + params.refs_x + j1]);
    let x0 = select(0u, first_x - r, first_x >= r);
    let y0 = select(0u, first_y - r, first_y >= r);
    let x1 = min(last_x + r + N, params.width);
    let y1 = min(last_y + r + N, params.height);
    region = vec4<u32>(x0, y0, x1 - x0, y1 - y0);
  }
  workgroupBarrier();
  load_tile(false, c, t);
  workgroupBarrier();

  var rx = 0u;
  var ry = 0u;
  var limit = 0.0;
  clear_chosen();
  clear_top();
  if (in_tile) {
    rx = u32(tables[POSITIONS + i]);
    ry = u32(tables[POSITIONS + params.refs_x + j]);
    ref_base = (ry - region.y) * region.z + rx - region.x;
    var mean_sum = 0.0;
    for (var p = 0u; p < AREA; p++) {
      mean_sum += current_tile[ref_base + (p / N) * region.z + p % N];
    }
    var mean = mean_sum;
    mean /= f32(AREA);
    let per_sample = mean_variance_at(mean);
    limit = (params.distance_factor * 2.0 * per_sample + params.distance_floor) * f32(AREA);
    search(false, lane, c, rx, ry, params.search_radius, limit, rx, ry, true);
  }
  publish(slot, lane);
  workgroupBarrier();
  if (in_tile && lane == 0u) {
    merge_and_take(slot);
  }

  // Follow the best match frame by frame. Every thread helps load each frame.
  for (var side = 0u; side < 2u; side++) {
    workgroupBarrier();
    if (lane == 0u) {
      track[slot] = pack(c, rx, ry);
      track_alive[slot] = u32(in_tile);
    }
    for (var k = 1u; k <= params.temporal_radius; k++) {
      var frame: u32;
      if (side == 0u) {
        if (k > c) { break; }
        frame = c - k;
      } else {
        frame = c + k;
        if (frame >= params.frames) { break; }
      }
      workgroupBarrier();
      if (params.motion == 0u) {
        load_tile(true, frame, t);
      }
      workgroupBarrier();
      let alive = track_alive[slot] != 0u;
      clear_top();
      if (alive) {
        let center = track[slot];
        if (params.motion == 1u) {
          let field = select(params.temporal_radius, 0u, side == 0u) + k - 1u;
          search_moved(lane, frame, unpack_x(center), unpack_y(center), field, limit);
        } else {
          search(true, lane, frame, unpack_x(center), unpack_y(center), params.predict_radius, limit, 0u, 0u, false);
        }
      }
      // The lists overwrite the frame, so every lane must be done searching it.
      workgroupBarrier();
      publish(slot, lane);
      workgroupBarrier();
      if (alive && lane == 0u) {
        if (merge_and_take(slot) == 0u) {
          track_alive[slot] = 0u;
        }
      }
    }
  }

  if (in_tile && lane == 0u) {
    let r = j * params.refs_x + i;
    let capacity = params.max_group - 1u;
    var n = 0u;
    if (e0 < FAR && 0u < capacity) {
      groups[r * GROUP_STRIDE + 2u] = g0;
      n += 1u;
    }
    if (e1 < FAR && 1u < capacity) {
      groups[r * GROUP_STRIDE + 3u] = g1;
      n += 1u;
    }
    if (e2 < FAR && 2u < capacity) {
      groups[r * GROUP_STRIDE + 4u] = g2;
      n += 1u;
    }
    if (e3 < FAR && 3u < capacity) {
      groups[r * GROUP_STRIDE + 5u] = g3;
      n += 1u;
    }
    if (e4 < FAR && 4u < capacity) {
      groups[r * GROUP_STRIDE + 6u] = g4;
      n += 1u;
    }
    if (e5 < FAR && 5u < capacity) {
      groups[r * GROUP_STRIDE + 7u] = g5;
      n += 1u;
    }
    if (e6 < FAR && 6u < capacity) {
      groups[r * GROUP_STRIDE + 8u] = g6;
      n += 1u;
    }
    if (e7 < FAR && 7u < capacity) {
      groups[r * GROUP_STRIDE + 9u] = g7;
      n += 1u;
    }
    if (e8 < FAR && 8u < capacity) {
      groups[r * GROUP_STRIDE + 10u] = g8;
      n += 1u;
    }
    if (e9 < FAR && 9u < capacity) {
      groups[r * GROUP_STRIDE + 11u] = g9;
      n += 1u;
    }
    if (e10 < FAR && 10u < capacity) {
      groups[r * GROUP_STRIDE + 12u] = g10;
      n += 1u;
    }
    if (e11 < FAR && 11u < capacity) {
      groups[r * GROUP_STRIDE + 13u] = g11;
      n += 1u;
    }
    if (e12 < FAR && 12u < capacity) {
      groups[r * GROUP_STRIDE + 14u] = g12;
      n += 1u;
    }
    if (e13 < FAR && 13u < capacity) {
      groups[r * GROUP_STRIDE + 15u] = g13;
      n += 1u;
    }
    if (e14 < FAR && 14u < capacity) {
      groups[r * GROUP_STRIDE + 16u] = g14;
      n += 1u;
    }
    groups[r * GROUP_STRIDE] = n + 1u;
    groups[r * GROUP_STRIDE + 1u] = pack(c, rx, ry);
  }
}

// The candidates around the track and around the field's prediction, read from the plane.
fn search_moved(lane: u32, frame: u32, cx: u32, cy: u32, field: u32, limit: f32) {
  let blocks = params.field_blocks_x * params.field_blocks_y;
  let bx = min((cx * params.motion_scale_x + 4u) / 16u, params.field_blocks_x - 1u);
  let by = min((cy * params.motion_scale_y + 4u) / 16u, params.field_blocks_y - 1u);
  let slot = u32(tables[params.field_map + field]);
  let packed = fields_in[slot * blocks + by * params.field_blocks_x + bx];
  let dx = (i32(packed & 0xffffu) - 512) / i32(params.motion_scale_x);
  let dy = (i32(packed >> 16u) - 512) / i32(params.motion_scale_y);
  let px = u32(clamp(i32(cx) + dx, 0, i32(params.width - N)));
  let py = u32(clamp(i32(cy) + dy, 0, i32(params.height - N)));
  let a = window_bounds(cx, cy, params.predict_radius);
  let b = window_bounds(px, py, params.predict_radius);
  let side = 2u * params.predict_radius + 1u;
  let count = 2u * side * side;
  let base = guide_base(frame);
  for (var i = lane; i < count; i += LANES) {
    let w = select(b, a, i < side * side);
    let j = i % (side * side);
    let x = w.x + j % side;
    let y = w.y + j / side;
    if (x > w.z || y > w.w) {
      continue;
    }
    // The second window skips what the first has.
    if (i >= side * side && x >= a.x && x <= a.z && y >= a.y && y <= a.w) {
      continue;
    }
    let d = distance_plane(base + y * params.width + x, params.width);
    if (d <= limit) {
      offer(d, pack(frame, x, y));
    }
  }
}

// The window of `radius` about (cx, cy) inside the plane: left, top, right, bottom.
fn window_bounds(cx: u32, cy: u32, radius: u32) -> vec4<u32> {
  return vec4<u32>(
    select(0u, cx - radius, cx >= radius),
    select(0u, cy - radius, cy >= radius),
    min(cx + radius, params.width - N),
    min(cy + radius, params.height - N),
  );
}

// ---- half_plane ----

// The luma planes averaged 2x2 not measured yet, as Plane::downscale does.
@compute @workgroup_size(64)
fn half_plane(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) local: u32) {
  let i = (wg.y * params.dispatch_x + wg.x) * 64u + local;
  let area = params.half_width * params.half_height;
  if (i >= area * params.half_count) {
    return;
  }
  let slot = u32(tables[params.half_jobs + i / area]);
  let x = (i % area) % params.half_width;
  let y = (i % area) / params.half_width;
  let fx = (params.width + params.half_width - 1u) / params.half_width;
  let fy = (params.height + params.half_height - 1u) / params.half_height;
  var sum = 0.0;
  var count = 0.0;
  for (var sy = y * fy; sy < min((y + 1u) * fy, params.height); sy++) {
    for (var sx = x * fx; sx < min((x + 1u) * fx, params.width); sx++) {
      sum += guide[slot * params.plane_size + sy * params.width + sx];
      count += 1.0;
    }
  }
  half_planes[slot * area + i % area] = select(0.0, sum / count, count > 0.0);
}

// ---- motion ----

// One block, the area its moves reach, and each thread's best move.
var<workgroup> motion_template: array<f32, 64>;
var<workgroup> motion_area: array<f32, 1600>;
var<workgroup> motion_sum: array<f32, 64>;
var<workgroup> motion_rank: array<u32, 64>;

// Every 16x16 block's best move, as MotionField::estimate finds it. One workgroup per block.
@compute @workgroup_size(64)
fn motion(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let i = wg.y * params.dispatch_x + wg.x;
  let blocks = params.field_blocks_x * params.field_blocks_y;
  if (i >= blocks * params.field_count) {
    return;
  }
  let job = params.field_jobs + 3u * (i / blocks);
  let block = i % blocks;
  let out = u32(tables[job]) * blocks + block;
  let source_frame = u32(tables[job + 1u]);
  let target_frame = u32(tables[job + 2u]);
  let hw = params.half_width;
  let hh = params.half_height;
  let area = hw * hh;
  let x0 = min((block % params.field_blocks_x) * N, hw - N);
  let y0 = min((block / params.field_blocks_x) * N, hh - N);
  let r = params.half_range;
  // The area the moves reach, clipped to the plane.
  let ax = select(0u, x0 - r, x0 >= r);
  let ay = select(0u, y0 - r, y0 >= r);
  let aw = min(x0 + N + r, hw) - ax;
  let ah = min(y0 + N + r, hh) - ay;
  motion_template[t] = half_planes[source_frame * area + (y0 + t / N) * hw + x0 + t % N];
  for (var j = t; j < aw * ah; j += 64u) {
    motion_area[j] = half_planes[target_frame * area + (ay + j / aw) * hw + ax + j % aw];
  }
  workgroupBarrier();
  let side = 2u * r + 1u;
  var best = 3.0e38;
  var best_rank = 0xffffffffu;
  for (var m = t; m < side * side; m += 64u) {
    let dx = i32(m % side) - i32(r);
    let dy = i32(m / side) - i32(r);
    let x = i32(x0) + dx;
    let y = i32(y0) + dy;
    if (x < 0 || y < 0 || u32(x) + N > hw || u32(y) + N > hh) {
      continue;
    }
    let base = (u32(y) - ay) * aw + u32(x) - ax;
    var sum = 0.0;
    for (var row = 0u; row < N; row++) {
      for (var col = 0u; col < N; col++) {
        let d = motion_template[row * N + col] - motion_area[base + row * aw + col];
        sum += d * d;
      }
    }
    // Ties go to the shortest move, then the first scanned.
    let rank = u32(abs(dx) + abs(dy)) * 4096u + m;
    if (sum < best || (sum == best && rank < best_rank)) {
      best = sum;
      best_rank = rank;
    }
  }
  motion_sum[t] = best;
  motion_rank[t] = best_rank;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (t < stride) {
      let other = t + stride;
      if (motion_sum[other] < motion_sum[t] || (motion_sum[other] == motion_sum[t] && motion_rank[other] < motion_rank[t])) {
        motion_sum[t] = motion_sum[other];
        motion_rank[t] = motion_rank[other];
      }
    }
    workgroupBarrier();
  }
  if (t == 0u) {
    let m = motion_rank[0] % 4096u;
    let dx = (i32(m % side) - i32(r)) * 2;
    let dy = (i32(m / side) - i32(r)) * 2;
    fields[out] = u32(dx + 512) | (u32(dy + 512) << 16u);
  }
}

// ---- filter_groups ----

// One patch on its way through the 8x8 DCT.
var<workgroup> tile: array<f32, 64>;
var<workgroup> rows: array<f32, 64>;
var<workgroup> reduce: array<f32, 64>;
var<workgroup> members: array<u32, 16>;
var<workgroup> member_count: u32;
// Where the current frame members are in the group.
var<workgroup> kept: array<u32, 16>;
var<workgroup> kept_count: u32;
// Numerators, then denominators, of the region around the tile's references.
// Each pixel goes to the accumulators once, as atomics cost most of the time on some GPUs.
var<workgroup> sums: array<f32, 2u * FILTER_SIDE * FILTER_SIDE>;

// The DCT basis values thread t's row and column need, loaded once by load_basis: forward,
// then inverse.
var<private> basis_col: array<f32, 8>;
var<private> basis_row: array<f32, 8>;
var<private> inverse_col: array<f32, 8>;
var<private> inverse_row: array<f32, 8>;

fn load_basis(t: u32) {
  let row = t / N;
  let col = t % N;
  for (var i = 0u; i < N; i++) {
    basis_col[i] = tables[DCT + col * N + i];
    basis_row[i] = tables[DCT + row * N + i];
    inverse_col[i] = tables[DCT + i * N + col];
    inverse_row[i] = tables[DCT + i * N + row];
  }
}

// Coefficient t of the 8x8 DCT of the patch whose pixel t is `value`. Every thread calls it.
fn dct(t: u32, value: f32) -> f32 {
  let row = t / N;
  let col = t % N;
  workgroupBarrier();
  tile[t] = value;
  workgroupBarrier();
  var s = 0.0;
  for (var x = 0u; x < N; x++) {
    s += basis_col[x] * tile[row * N + x];
  }
  rows[t] = s;
  workgroupBarrier();
  var out = 0.0;
  for (var y = 0u; y < N; y++) {
    out += basis_row[y] * rows[y * N + col];
  }
  return out;
}

// Pixel t of the patch whose coefficient t is `value`. Every thread calls it.
fn inverse_dct(t: u32, value: f32) -> f32 {
  let row = t / N;
  let col = t % N;
  workgroupBarrier();
  tile[t] = value;
  workgroupBarrier();
  var s = 0.0;
  for (var u = 0u; u < N; u++) {
    s += inverse_col[u] * tile[row * N + u];
  }
  rows[t] = s;
  workgroupBarrier();
  var out = 0.0;
  for (var v = 0u; v < N; v++) {
    out += inverse_row[v] * rows[v * N + col];
  }
  return out;
}

fn sum_over_threads(t: u32, value: f32) -> f32 {
  workgroupBarrier();
  reduce[t] = value;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (t < stride) {
      reduce[t] += reduce[t + stride];
    }
    workgroupBarrier();
  }
  return reduce[0];
}

// One workgroup per 4x4 block of references, as in find_groups.
@compute @workgroup_size(64)
fn filter_groups(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let tiles_x = (params.refs_x + TILE_REFS - 1u) / TILE_REFS;
  let tiles_y = (params.refs_y + TILE_REFS - 1u) / TILE_REFS;
  let tile = wg.y * params.dispatch_x + wg.x;
  if (tile >= tiles_x * tiles_y) {
    return;
  }
  let i0 = (tile % tiles_x) * TILE_REFS;
  let j0 = (tile / tiles_x) * TILE_REFS;
  let i1 = min(i0 + TILE_REFS, params.refs_x) - 1u;
  let j1 = min(j0 + TILE_REFS, params.refs_y) - 1u;
  // The references' patches and FILTER_MARGIN around them, where most current frame members are.
  if (t == 0u) {
    let r = min(params.search_radius, FILTER_MARGIN);
    let first_x = u32(tables[POSITIONS + i0]);
    let last_x = u32(tables[POSITIONS + i1]);
    let first_y = u32(tables[POSITIONS + params.refs_x + j0]);
    let last_y = u32(tables[POSITIONS + params.refs_x + j1]);
    let x0 = select(0u, first_x - r, first_x >= r);
    let y0 = select(0u, first_y - r, first_y >= r);
    let x1 = min(last_x + r + N, params.width);
    let y1 = min(last_y + r + N, params.height);
    region = vec4<u32>(x0, y0, x1 - x0, y1 - y0);
  }
  workgroupBarrier();
  load_basis(t);
  let area = region.z * region.w;
  for (var i = t; i < area; i += 64u) {
    sums[i] = 0.0;
    sums[FILTER_SIDE * FILTER_SIDE + i] = 0.0;
  }
  for (var j = j0; j <= j1; j++) {
    for (var i = i0; i <= i1; i++) {
      filter_group(j * params.refs_x + i, t);
    }
  }
  workgroupBarrier();
  for (var i = t; i < area; i += 64u) {
    let weight = sums[FILTER_SIDE * FILTER_SIDE + i];
    if (weight > 0.0) {
      let pixel = (region.y + i / region.z) * params.width + region.x + i % region.z;
      atomic_add(pixel, sums[i]);
      atomic_add(params.plane_size + pixel, weight);
    }
  }
}

// Filters group r and adds its current frame members into sums. Every thread calls it.
fn filter_group(r: u32, t: u32) {
  let row = t / N;
  let col = t % N;
  let c = params.center;
  let wiener = params.stage == 1u;
  // The last group may still be reading the members.
  workgroupBarrier();
  if (t == 0u) {
    member_count = groups[r * GROUP_STRIDE];
  }
  if (t < MAX_GROUP) {
    members[t] = groups[r * GROUP_STRIDE + 1u + t];
  }
  let n = workgroupUniformLoad(&member_count);

  // The reference's mean on the plane matched on, as bm3d.rs takes it.
  let reference = members[0];
  let mean = sum_over_threads(t, guide[guide_base(c) + (unpack_y(reference) + row) * params.width + unpack_x(reference) + col]) / f32(AREA);
  let variance = variance_at(mean, t);

  // Coefficient t of every member's 8x8 DCT, noisy and, for Wiener, basic.
  var column: array<f32, 16>;
  var estimate_column: array<f32, 16>;
  // Load every member's sample first so the loads overlap. Fixed loop bounds keep the
  // arrays in registers.
  var sample: array<f32, 16>;
  var estimate_sample: array<f32, 16>;
  for (var m = 0u; m < MAX_GROUP; m++) {
    if (m < n) {
      let p = members[m];
      let at = (unpack_y(p) + row) * params.width + unpack_x(p) + col;
      sample[m] = noisy[noisy_base(unpack_frame(p)) + at];
      if (wiener) {
        estimate_sample[m] = guide[guide_base(unpack_frame(p)) + at];
      }
    }
  }
  for (var m = 0u; m < MAX_GROUP; m++) {
    if (m < n) {
      column[m] = dct(t, sample[m]);
      if (wiener) {
        estimate_column[m] = dct(t, estimate_sample[m]);
      }
    }
  }

  // Along the group.
  let basis = GROUP_BASIS + n * 256u;
  var spectrum: array<f32, 16>;
  for (var k = 0u; k < n; k++) {
    var s = 0.0;
    for (var i = 0u; i < n; i++) {
      s += tables[basis + k * n + i] * column[i];
    }
    spectrum[k] = s;
  }
  var left = 0.0;
  if (wiener) {
    for (var k = 0u; k < n; k++) {
      var s = 0.0;
      for (var i = 0u; i < n; i++) {
        s += tables[basis + k * n + i] * estimate_column[i];
      }
      let b2 = s * s;
      let gain = select(1.0, b2 / (b2 + variance), b2 + variance > 0.0);
      spectrum[k] *= gain;
      left += gain * gain * variance;
    }
  } else {
    let threshold = params.lambda * params.lambda * variance;
    for (var k = 0u; k < n; k++) {
      if (k == 0u && t == 0u) {
        left += variance;
      } else if (spectrum[k] * spectrum[k] < threshold) {
        spectrum[k] = 0.0;
      } else {
        left += variance;
      }
    }
  }
  // Only the members in the current frame go back.
  if (t == 0u) {
    var count = 0u;
    for (var m = 0u; m < n; m++) {
      if (unpack_frame(members[m]) == c) {
        kept[count] = m;
        count++;
      }
    }
    kept_count = count;
  }
  let total = sum_over_threads(t, left);
  let weight = select(1.0, 1.0 / total, total > 0.0);

  // Back along the group and through the inverse 2D DCT, as pixel (row, col), added into
  // the region. The inverse DCT's barriers keep overlapping patches apart.
  let w = weight * tables[KAISER + t];
  let count = workgroupUniformLoad(&kept_count);
  for (var j = 0u; j < count; j++) {
    let m = kept[j];
    var coefficient = 0.0;
    for (var k = 0u; k < n; k++) {
      coefficient += tables[basis + k * n + m] * spectrum[k];
    }
    let s = inverse_dct(t, coefficient);
    let p = members[m];
    let x = unpack_x(p) + col;
    let y = unpack_y(p) + row;
    if (x >= region.x && y >= region.y && x < region.x + region.z && y < region.y + region.w) {
      let at = (y - region.y) * region.z + x - region.x;
      sums[at] += w * s;
      sums[FILTER_SIDE * FILTER_SIDE + at] += w;
    } else {
      atomic_add(y * params.width + x, w * s);
      atomic_add(params.plane_size + y * params.width + x, w);
    }
  }
}

// ---- normalize ----

@compute @workgroup_size(64)
fn normalize(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) local: u32) {
  let i = (wg.y * params.dispatch_x + wg.x) * 64u + local;
  if (i >= params.plane_size) {
    return;
  }
  let numerator = accum_load(i);
  let denominator = accum_load(params.plane_size + i);
  output[i] = select(noisy[noisy_base(params.center) + i], numerator / denominator, denominator > 0.0);
}
