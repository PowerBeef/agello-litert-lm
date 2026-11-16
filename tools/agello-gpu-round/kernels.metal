// The GPU-resident speculative round's own kernels (stage 1 of Agello's
// build/drafts/gpu-resident-round-design-2026-11-16.md). Each must give, bit for bit, what the host
// path gives today, so they're compiled with fast math off (MTLMathModeSafe): no reassociation of
// the two multiplies, no contraction.
//
// - agello_embed_gather: the CPU embedders (tf_lite_embedder, tf_lite_per_layer_embedder) as
//   gathers from the INT2 tables the model file maps (tools/model-surgery/embedder_tables.py).
// - agello_argmax_f32 / _f16: the greedy sampler's pick, the lowest index among equal maxima
//   (runtime/components/sampling_cpu_util.cc, MaxElement).
// - agello_concat: the drafter's input, embedding then activations
//   (LlmLiteRtMtpDrafter::RunDraftingLoop).
#include <metal_stdlib>
using namespace metal;

struct GatherTable {
  // Byte offsets into `base`: the packed rows (cols / 4 bytes each, 2-bit fields, low bits first)
  // and the rows' float32 scales.
  uint rows_offset;
  uint scales_offset;
};

struct GatherShape {
  uint rows;       // 262,144
  uint cols;       // per table: 2,560 or 256
  uint tokens;     // ids to gather
  uint tables;     // 1 or 42; outputs are tables × cols per token, in table order
  uint has_mul;
  float mul;       // sqrt(2560) for tf_lite_embedder
};

// One thread per (byte of a row, token, table): four outputs.
kernel void agello_embed_gather(
    device const int* ids [[buffer(0)]],
    device const uchar* base [[buffer(1)]],
    constant GatherTable* tables [[buffer(2)]],
    constant GatherShape& shape [[buffer(3)]],
    device float* out [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
  const uint bytes_per_row = shape.cols / 4;
  if (gid.x >= bytes_per_row || gid.y >= shape.tokens || gid.z >= shape.tables) return;
  const int id = ids[gid.y];
  // The graph's bounds check: an id outside [0, rows) reads row 0.
  const uint row = (id >= 0 && uint(id) < shape.rows) ? uint(id) : 0u;
  const GatherTable table = tables[gid.z];
  const uchar packed = base[table.rows_offset + row * bytes_per_row + gid.x];
  const float scale = ((device const float*)(base + table.scales_offset))[row];
  device float* o = out + (gid.y * shape.tables + gid.z) * shape.cols + gid.x * 4;
  for (uint k = 0; k < 4; ++k) {
    int q = int((packed >> (2 * k)) & 3);
    q = q >= 2 ? q - 4 : q;
    float v = float(q) * scale;
    if (shape.has_mul) v = v * shape.mul;
    o[k] = v;
  }
}

struct Best {
  float value;
  uint index;
};

static inline Best better(Best a, Best b) {
  // The CPU rule: a strictly greater value wins; among equal values, the lower index.
  if (b.value > a.value || (b.value == a.value && b.index < a.index)) return b;
  return a;
}

// One threadgroup per row of logits (a drafter step, or each of verify's positions).
template <typename T>
static inline void argmax_row(
    device const T* logits, uint vocab, device int* out, uint row, uint lane, uint lanes,
    threadgroup Best* shared) {
  device const T* x = logits + row * vocab;
  Best best = {-INFINITY, 0xffffffffu};
  bool any = false;
  for (uint i = lane; i < vocab; i += lanes) {
    Best candidate = {float(x[i]), i};
    best = any ? better(best, candidate) : candidate;
    any = true;
  }
  shared[lane] = best;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = lanes / 2; stride > 0; stride /= 2) {
    if (lane < stride) shared[lane] = better(shared[lane], shared[lane + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0) out[row] = int(shared[0].index);
}

kernel void agello_argmax_f32(
    device const float* logits [[buffer(0)]], constant uint& vocab [[buffer(1)]],
    device int* out [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]], uint lanes [[threads_per_threadgroup]],
    threadgroup Best* shared [[threadgroup(0)]]) {
  argmax_row(logits, vocab, out, row, lane, lanes, shared);
}

kernel void agello_argmax_f16(
    device const half* logits [[buffer(0)]], constant uint& vocab [[buffer(1)]],
    device int* out [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]], uint lanes [[threads_per_threadgroup]],
    threadgroup Best* shared [[threadgroup(0)]]) {
  argmax_row(logits, vocab, out, row, lane, lanes, shared);
}

// out = first ‖ second, float32.
kernel void agello_concat(
    device const float* first [[buffer(0)]], device const float* second [[buffer(1)]],
    constant uint2& sizes [[buffer(2)]], device float* out [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if (i < sizes.x) out[i] = first[i];
  else if (i < sizes.x + sizes.y) out[i] = second[i - sizes.x];
}
