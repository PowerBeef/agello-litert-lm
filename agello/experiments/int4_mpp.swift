// D1 prototype: int8 activations x 4-bit weights on the GPU's tensor hardware (MPP matmul2d).
// Compares, for C[m][n] = sum_k A[m][k] * (W[k][n] - 8):
//   int8      today's path: weights already converted to int8 [K][N]
//   staged    4-bit weights in ML Drift's decode layout (I4O4 blocks of 4 inputs x 4 outputs,
//             offset by 8), copied per K step into a threadgroup int4 tile with ^0x8888
//   direct    4-bit weights in a plain [K][N] layout (two's complement), read from device memory
import Foundation
import Metal

let source = """
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

#define M_TILE MT_VALUE
#define SG SG_VALUE
#define N_TILE 128
#define KT KT_VALUE

kernel void mm_int8(device int8_t* A [[buffer(0)]], device int8_t* B [[buffer(1)]],
                    device int* C [[buffer(2)]], constant int4& dims [[buffer(3)]],
                    uint2 tgid [[threadgroup_position_in_grid]]) {
  const int M = dims.x, N = dims.y, K = dims.z;
  auto a = tensor(A, dextents<int, 2>(K, M));
  auto b = tensor(B, dextents<int, 2>(N, K));
  auto c = tensor(C, dextents<int, 2>(N, M));
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(M_TILE, N_TILE, static_cast<int>(dynamic_extent), false, false, false);
  mpp::tensor_ops::matmul2d<desc, execution_simdgroups<SG>> op;
  auto ta = a.slice(0, tgid.y * M_TILE);
  auto tb = b.slice(tgid.x * N_TILE, 0);
  auto tc = c.slice(tgid.x * N_TILE, tgid.y * M_TILE);
  op.run(ta, tb, tc);
}

kernel void mm_int4_direct(device int8_t* A [[buffer(0)]], device uchar* B [[buffer(1)]],
                           device int* C [[buffer(2)]], constant int4& dims [[buffer(3)]],
                           uint2 tgid [[threadgroup_position_in_grid]]) {
  const int M = dims.x, N = dims.y, K = dims.z;
  auto a = tensor(A, dextents<int, 2>(K, M));
  auto b = tensor<device int4b_format, dextents<int, 2>, tensor_inline>(B, dextents<int, 2>(N, K));
  auto c = tensor(C, dextents<int, 2>(N, M));
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(M_TILE, N_TILE, static_cast<int>(dynamic_extent), false, false, false);
  mpp::tensor_ops::matmul2d<desc, execution_simdgroups<SG>> op;
  auto ta = a.slice(0, tgid.y * M_TILE);
  auto tb = b.slice(tgid.x * N_TILE, 0);
  auto tc = c.slice(tgid.x * N_TILE, tgid.y * M_TILE);
  op.run(ta, tb, tc);
}

kernel void mm_int4_staged(device int8_t* A [[buffer(0)]], device uint2* W [[buffer(1)]],
                           device int* C [[buffer(2)]], constant int4& dims [[buffer(3)]],
                           uint2 tgid [[threadgroup_position_in_grid]],
                           ushort lid [[thread_index_in_threadgroup]]) {
  const int M = dims.x, N = dims.y, K = dims.z;
  const int ns = N / 4;
  threadgroup ushort wt[KT * (N_TILE / 4)];
  auto tb = tensor<threadgroup int4b_format, dextents<int, 2>, tensor_inline>((threadgroup uchar*)wt, dextents<int, 2>(N_TILE, KT));
  auto a = tensor(A, dextents<int, 2>(K, M));
  auto c = tensor(C, dextents<int, 2>(N, M));
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(M_TILE, N_TILE, KT, false, false, false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, execution_simdgroups<SG>> op;
  auto ta0 = a.slice(0, tgid.y * M_TILE);
  auto ct = op.get_destination_cooperative_tensor<decltype(ta0), decltype(tb), int>();
  #pragma unroll_full
  for (uint16_t i = 0; i < ct.get_capacity(); ++i) {
    if (ct.is_valid_element(i)) ct[i] = 0;
  }
  const int o_slice0 = tgid.x * (N_TILE / 4);
  constexpr int blocks = (KT / 4) * (N_TILE / 4);
  for (int k0 = 0; k0 < K; k0 += KT) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int bidx = lid; bidx < blocks; bidx += SG * 32) {
      int bi = bidx / (N_TILE / 4);
      int bo = bidx % (N_TILE / 4);
      uint2 v = W[(k0 / 4 + bi) * ns + o_slice0 + bo];
      v ^= uint2(0x88888888u);
      wt[(bi * 4 + 0) * (N_TILE / 4) + bo] = ushort(v.x);
      wt[(bi * 4 + 1) * (N_TILE / 4) + bo] = ushort(v.x >> 16);
      wt[(bi * 4 + 2) * (N_TILE / 4) + bo] = ushort(v.y);
      wt[(bi * 4 + 3) * (N_TILE / 4) + bo] = ushort(v.y >> 16);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto ta = a.slice(k0, tgid.y * M_TILE);
    op.run(ta, tb, ct);
  }
  auto tc = c.slice(tgid.x * N_TILE, tgid.y * M_TILE);
  ct.store(tc);
}

// ML Drift's shipped short-run variant, as generated: 64 x 64 tiles, K tile 32, nibbles expanded
// to int8 (after ^0x8888) into a threadgroup int8 tile, int8 x int8 products.
uint32_t expand_int4_to_int8(uint32_t val) {
  uint32_t x = val & 0xFFFF;
  x = (x | (x << 8)) & 0x00FF00FF;
  x = (x | (x << 4)) & 0x0F0F0F0F;
  uint32_t signs = x & 0x08080808;
  uint32_t sign_extensions = signs * 0x1E;
  return x | sign_extensions;
}
kernel void mm_expand_int8(device int8_t* A [[buffer(0)]], device uint2* W [[buffer(1)]],
                           device int* C [[buffer(2)]], constant int4& dims [[buffer(3)]],
                           uint2 tgid [[threadgroup_position_in_grid]],
                           ushort lid [[thread_index_in_threadgroup]]) {
  const int M = dims.x, N = dims.y, K = dims.z;
  const int ns = N / 4;
  threadgroup int8_t w_loc[32 * 64];
  threadgroup uint* w_loc_x4 = (threadgroup uint*)(w_loc);
  auto tb = tensor(w_loc, dextents<int, 2>(64, 32));
  auto a = tensor(A, dextents<int, 2>(K, M));
  auto c = tensor(C, dextents<int, 2>(N, M));
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(64, 64, 32, false, false, false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
  auto ta0 = a.slice(0, tgid.y * 64);
  auto ct = op.get_destination_cooperative_tensor<decltype(ta0), decltype(tb), int>();
  #pragma unroll_full
  for (uint16_t i = 0; i < ct.get_capacity(); ++i) {
    if (ct.is_valid_element(i)) ct[i] = 0;
  }
  int sub_i = lid / 16;
  int sub_o = lid % 16;
  int w_o_slice = tgid.x * 16 + sub_o;
  for (int k0 = 0; k0 < K; k0 += 32) {
    uint2 u = W[(k0 / 4 + sub_i) * ns + w_o_slice];
    u ^= uint2(0x88888888u);
    uint w0 = expand_int4_to_int8(u.x), w1 = expand_int4_to_int8(u.x >> 16);
    uint w2 = expand_int4_to_int8(u.y), w3 = expand_int4_to_int8(u.y >> 16);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    w_loc_x4[(sub_i * 4 + 0) * 16 + sub_o] = w0;
    w_loc_x4[(sub_i * 4 + 1) * 16 + sub_o] = w1;
    w_loc_x4[(sub_i * 4 + 2) * 16 + sub_o] = w2;
    w_loc_x4[(sub_i * 4 + 3) * 16 + sub_o] = w3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto ta = a.slice(k0, tgid.y * 64);
    op.run(ta, tb, ct);
  }
  auto tc = c.slice(tgid.x * 64, tgid.y * 64);
  ct.store(tc);
}
"""

func run(M: Int, N: Int, K: Int, kt: Int, mt: Int, sg: Int, iterations: Int) throws {
  let device = MTLCreateSystemDefaultDevice()!
  let options = MTLCompileOptions()
  options.languageVersion = .version4_0
  let library = try device.makeLibrary(source: source.replacingOccurrences(of: "KT_VALUE", with: String(kt)).replacingOccurrences(of: "MT_VALUE", with: String(mt)).replacingOccurrences(of: "SG_VALUE", with: String(sg)), options: options)
  let queue = device.makeCommandQueue()!

  var rng = SystemRandomNumberGenerator()
  var a = [Int8](repeating: 0, count: M * K)
  for i in a.indices { a[i] = Int8(truncatingIfNeeded: Int.random(in: -127...127, using: &rng)) }
  var w = [UInt8](repeating: 0, count: K * N)  // offset-binary nibbles 0...15, value w - 8
  for i in w.indices { w[i] = UInt8.random(in: 0...15, using: &rng) }

  // int8 [K][N]
  let b8 = w.map { Int8(Int($0) - 8) }
  // direct int4 [K][N], two's complement, low nibble first
  var b4 = [UInt8](repeating: 0, count: K * N / 2)
  for k in 0..<K {
    for n in stride(from: 0, to: N, by: 2) {
      let lo = UInt8(bitPattern: Int8(Int(w[k * N + n]) - 8)) & 0xF
      let hi = UInt8(bitPattern: Int8(Int(w[k * N + n + 1]) - 8)) & 0xF
      b4[(k * N + n) / 2] = lo | (hi << 4)
    }
  }
  // ML Drift decode layout: blocks (i_slice, o_slice) of 8 bytes, block index i_slice * N/4 + o_slice;
  // 16-bit chunk j holds input 4*i_slice+j, outputs 4*o_slice+0..3 (low nibble first), offset binary
  var blocksBuf = [UInt16](repeating: 0, count: K * N / 4)
  for islice in 0..<(K / 4) {
    for oslice in 0..<(N / 4) {
      for j in 0..<4 {
        let k = islice * 4 + j
        var chunk: UInt16 = 0
        for o in 0..<4 { chunk |= UInt16(w[k * N + oslice * 4 + o]) << (4 * o) }
        blocksBuf[(islice * (N / 4) + oslice) * 4 + j] = chunk
      }
    }
  }

  let bufA = device.makeBuffer(bytes: a, length: a.count, options: .storageModeShared)!
  let buf8 = device.makeBuffer(bytes: b8, length: b8.count, options: .storageModeShared)!
  let buf4 = device.makeBuffer(bytes: b4, length: b4.count, options: .storageModeShared)!
  let bufBlocks = device.makeBuffer(bytes: blocksBuf, length: blocksBuf.count * 2, options: .storageModeShared)!
  var dims = SIMD4<Int32>(Int32(M), Int32(N), Int32(K), 0)

  // CPU reference on samples
  let samples = (0..<256).map { _ in (Int.random(in: 0..<M), Int.random(in: 0..<N)) }
  let reference = samples.map { (m, n) -> Int32 in
    var s: Int32 = 0
    for k in 0..<K { s += Int32(a[m * K + k]) * Int32(Int(w[k * N + n]) - 8) }
    return s
  }

  func measure(_ name: String, _ weights: MTLBuffer, weightBytes: Int, nTile: Int = 128, mTile: Int? = nil, threads: Int? = nil) throws {
    let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: name)!)
    let bufC = device.makeBuffer(length: M * N * 4, options: .storageModeShared)!
    var times: [Double] = []
    for _ in 0..<iterations {
      memset(bufC.contents(), 0, M * N * 4)
      let cb = queue.makeCommandBuffer()!
      let enc = cb.makeComputeCommandEncoder()!
      enc.setComputePipelineState(pipeline)
      enc.setBuffer(bufA, offset: 0, index: 0)
      enc.setBuffer(weights, offset: 0, index: 1)
      enc.setBuffer(bufC, offset: 0, index: 2)
      enc.setBytes(&dims, length: 16, index: 3)
      enc.dispatchThreadgroups(MTLSize(width: N / nTile, height: M / (mTile ?? mt), depth: 1),
                               threadsPerThreadgroup: MTLSize(width: threads ?? sg * 32, height: 1, depth: 1))
      enc.endEncoding()
      cb.commit()
      cb.waitUntilCompleted()
      if let error = cb.error { print("\(name): \(error)"); return }
      times.append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
    }
    let c = bufC.contents().bindMemory(to: Int32.self, capacity: M * N)
    var wrong = 0
    for (index, (m, n)) in samples.enumerated() where c[m * N + n] != reference[index] { wrong += 1 }
    let best = times.dropFirst().min() ?? times[0]
    let median = times.dropFirst().sorted()[(times.count - 1) / 2]
    let tops = 2.0 * Double(M) * Double(N) * Double(K) / (median / 1000) / 1e12
    print(String(format: "  %-15@ median %7.3f ms  best %7.3f ms  %5.1f TOPS  weights %5.1f MB  wrong %d/256",
                 name as NSString, median, best, tops, Double(weightBytes) / 1e6, wrong))
  }

  print("M \(M) N \(N) K \(K), tiles M \(mt) K \(kt), \(sg) simdgroups, \(iterations) runs:")
  try measure("mm_int8", buf8, weightBytes: b8.count)
  try measure("mm_int4_direct", buf4, weightBytes: b4.count)
  try measure("mm_int4_staged", bufBlocks, weightBytes: blocksBuf.count * 2)
  try measure("mm_expand_int8", bufBlocks, weightBytes: blocksBuf.count * 2, nTile: 64, mTile: 64, threads: 128)
}

let args = CommandLine.arguments.dropFirst().compactMap { Int($0) }
let (kt, mt, sg) = (args.count > 0 ? args[0] : 128, args.count > 1 ? args[1] : 64, args.count > 2 ? args[2] : 4)
for (m, n, k) in [(128, 10240, 2560), (128, 2560, 10240), (128, 2048, 2560), (128, 2560, 2048), (1024, 10240, 2560)] {
  try run(M: m, N: n, K: k, kt: kt, mt: mt, sg: sg, iterations: 12)
}
