// D2 sweep: the matrix units' ceiling on attention-shaped products (MPP matmul2d), to tell
// whether prefill attention is arithmetic-bound or traffic-bound (PERFORMANCE.md, D2 rows).
// C[m][n] = sum_k A[m][k] * B[k][n], A half [M][K]; B half or int8 [K][N]; C float or half.
//   qk-like: M 4,096 query rows (8 heads x 1,024 tokens / 2 KV heads x 4), N keys, K head size
//   pv-like: M 4,096, N head size, K keys
// Run: swift agello/experiments/mpp_ceiling.swift
import Foundation
import Metal

func source(mTile: Int, nTile: Int, sg: Int) -> String {
  """
  #include <metal_stdlib>
  #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
  using namespace metal;

  template <typename BType, typename CType>
  void mm(device half* A, device BType* B, device CType* C, constant int4& dims, uint2 tgid) {
    const int M = dims.x, N = dims.y, K = dims.z;
    auto a = tensor(A, dextents<int, 2>(K, M));
    auto b = tensor(B, dextents<int, 2>(N, K));
    auto c = tensor(C, dextents<int, 2>(N, M));
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(\(mTile), \(nTile),
        static_cast<int>(dynamic_extent), false, false, false);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroups<\(sg)>> op;
    auto ta = a.slice(0, tgid.y * \(mTile));
    auto tb = b.slice(tgid.x * \(nTile), 0);
    auto tc = c.slice(tgid.x * \(nTile), tgid.y * \(mTile));
    op.run(ta, tb, tc);
  }

  kernel void hh_f(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device float* C [[buffer(2)]],
                   constant int4& dims [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {
    mm<half, float>(A, B, C, dims, tgid);
  }
  kernel void hh_h(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device half* C [[buffer(2)]],
                   constant int4& dims [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {
    mm<half, half>(A, B, C, dims, tgid);
  }
  kernel void hi8_f(device half* A [[buffer(0)]], device int8_t* B [[buffer(1)]], device float* C [[buffer(2)]],
                    constant int4& dims [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {
    mm<int8_t, float>(A, B, C, dims, tgid);
  }
  """
}

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
print("GPU: \(device.name)")

func run(M: Int, N: Int, K: Int, label: String, tiles: [(Int, Int)], iterations: Int = 10) throws {
  let a = (0..<(M * K)).map { _ in Float16(Float.random(in: -1...1)) }
  let bh = (0..<(K * N)).map { _ in Float16(Float.random(in: -1...1)) }
  let b8 = (0..<(K * N)).map { _ in Int8.random(in: -127...127) }
  let bufA = device.makeBuffer(bytes: a, length: a.count * 2, options: .storageModeShared)!
  let bufBh = device.makeBuffer(bytes: bh, length: bh.count * 2, options: .storageModeShared)!
  let bufB8 = device.makeBuffer(bytes: b8, length: b8.count, options: .storageModeShared)!
  let bufC = device.makeBuffer(length: M * N * 4, options: .storageModeShared)!
  var dims = SIMD4<Int32>(Int32(M), Int32(N), Int32(K), 0)
  let flops = 2.0 * Double(M) * Double(N) * Double(K)
  print("\(label): M \(M) N \(N) K \(K) (\(String(format: "%.1f", flops / 1e9)) GFLOP)")
  for (mTile, nTile) in tiles where N % nTile == 0 && M % mTile == 0 {
    let library = try device.makeLibrary(source: source(mTile: mTile, nTile: nTile, sg: 4), options: nil)
    var line = String(format: "  tile %3dx%-3d", mTile, nTile)
    for (name, b) in [("hh_f", bufBh), ("hh_h", bufBh), ("hi8_f", bufB8)] {
      let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: name)!)
      var times: [Double] = []
      for _ in 0..<iterations {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(b, offset: 0, index: 1)
        enc.setBuffer(bufC, offset: 0, index: 2)
        enc.setBytes(&dims, length: 16, index: 3)
        enc.dispatchThreadgroups(MTLSize(width: N / nTile, height: M / mTile, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { print("\(name): \(error)"); return }
        times.append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
      }
      let median = times.dropFirst().sorted()[(times.count - 1) / 2]
      line += String(format: "  %@ %7.3f ms %5.1f TF", name, median, flops / (median / 1000) / 1e12)
    }
    print(line)
  }
}

let tiles = [(64, 128), (32, 256), (16, 512), (128, 64)]
try run(M: 4096, N: 8192, K: 512, label: "qk-like, global", tiles: tiles)
try run(M: 4096, N: 8192, K: 256, label: "qk-like, local", tiles: tiles)
try run(M: 4096, N: 512, K: 8192, label: "pv-like, global", tiles: tiles)
try run(M: 4096, N: 256, K: 8192, label: "pv-like, local", tiles: tiles)
