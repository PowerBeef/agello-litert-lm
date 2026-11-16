// Checks kernels.metal against the host path, bit for bit, on this Mac's GPU (stage 1 of the
// GPU-resident speculative round, Agello's build/drafts/gpu-resident-round-design-2026-11-16.md):
//
// - the embedders' gathers for all 262,144 tokens, read through no-copy buffers over the mapped
//   model file, against the same formula on the CPU, and for the reference ids against the LiteRT
//   interpreter's own rows;
// - the argmax against a port of the CPU sampler's MaxElement, on random rows with planted ties,
//   signed zeros and maxima at both ends, in float32 and float16;
// - the concatenation.
//
// Build and run (the manifest and reference come from tools/model-surgery/embedder_tables.py):
//   python embedder_tables.py <model> --check 1000 --manifest tables.json --dump-reference ref.bin
//   swiftc -O check_kernels.swift -o check_kernels
//   ./check_kernels <model> tables.json kernels.metal [ref.bin]
import Foundation
import Metal

struct Table: Decodable {
  let file_offset: Int
  let bytes: Int
  let rows: Int
  let cols: Int
  let scale_offset: Int
}

struct Layout: Decodable {
  let section: [Int]
  let tables: [Table]
  let mul: Double?
}

func fail(_ message: String) -> Never {
  FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
  exit(1)
}

let args = CommandLine.arguments
guard args.count >= 4 else { fail("usage: check_kernels <model> <tables.json> <kernels.metal> [ref.bin]") }
let layouts = try JSONDecoder().decode([String: Layout].self, from: Data(contentsOf: URL(fileURLWithPath: args[2])))
guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { fail("no Metal device") }
let options = MTLCompileOptions()
options.mathMode = .safe
let library = try device.makeLibrary(source: String(contentsOfFile: args[3], encoding: .utf8), options: options)
func pipeline(_ name: String) throws -> MTLComputePipelineState {
  try device.makeComputePipelineState(function: library.makeFunction(name: name)!)
}
let gather = try pipeline("agello_embed_gather")
let argmax32 = try pipeline("agello_argmax_f32")
let argmax16 = try pipeline("agello_argmax_f16")
let concat = try pipeline("agello_concat")
print("GPU: \(device.name)")

// The model file, mapped read-only, as the runtime maps it.
let fd = open(args[1], O_RDONLY)
guard fd >= 0 else { fail("can't open \(args[1])") }
var info = stat()
fstat(fd, &info)
let fileSize = Int(info.st_size)
guard let mapped = mmap(nil, fileSize, PROT_READ, MAP_SHARED, fd, 0), mapped != MAP_FAILED else { fail("mmap") }
let file = UnsafeRawPointer(mapped)
let page = Int(getpagesize())

struct GatherTable { var rows_offset: UInt32; var scales_offset: UInt32 }
struct GatherShape { var rows: UInt32; var cols: UInt32; var tokens: UInt32; var tables: UInt32; var has_mul: UInt32; var mul: Float }

/// A no-copy buffer over the section's pages, and each table's offsets inside it.
func sectionBuffer(_ layout: Layout) -> (MTLBuffer, [GatherTable]) {
  let start = layout.section[0] / page * page
  let end = (layout.section[1] + page - 1) / page * page
  guard let buffer = device.makeBuffer(
    bytesNoCopy: UnsafeMutableRawPointer(mutating: file + start), length: end - start,
    options: .storageModeShared, deallocator: nil)
  else { fail("no-copy buffer over [\(start), \(end))") }
  return (buffer, layout.tables.map { GatherTable(rows_offset: UInt32($0.file_offset - start), scales_offset: UInt32($0.scale_offset - start)) })
}

/// The host formula, as embedder_tables.py checks it against the interpreter.
func cpuRows(_ layout: Layout, ids: [Int32]) -> [Float] {
  let cols = layout.tables[0].cols
  let rows = layout.tables[0].rows
  var out = [Float](repeating: 0, count: ids.count * layout.tables.count * cols)
  let mul = layout.mul.map(Float.init)
  out.withUnsafeMutableBufferPointer { o in
    for (t, id) in ids.enumerated() {
      let row = id >= 0 && Int(id) < rows ? Int(id) : 0
      for (k, table) in layout.tables.enumerated() {
        let packed = (file + table.file_offset + row * cols / 4).assumingMemoryBound(to: UInt8.self)
        let scale = (file + table.scale_offset).assumingMemoryBound(to: Float.self)[row]
        let base = (t * layout.tables.count + k) * cols
        for j in 0..<cols {
          var q = Int((packed[j / 4] >> UInt8(2 * (j % 4))) & 3)
          if q >= 2 { q -= 4 }
          var v = Float(q) * scale
          if let mul { v = v * mul }
          o[base + j] = v
        }
      }
    }
  }
  return out
}

func gpuRows(_ layout: Layout, buffer: MTLBuffer, tables: [GatherTable], ids: [Int32]) -> (rows: [Float], seconds: Double) {
  let cols = layout.tables[0].cols
  var shape = GatherShape(
    rows: UInt32(layout.tables[0].rows), cols: UInt32(cols), tokens: UInt32(ids.count), tables: UInt32(tables.count),
    has_mul: layout.mul == nil ? 0 : 1, mul: Float(layout.mul ?? 1))
  let idBuffer = device.makeBuffer(bytes: ids, length: ids.count * 4, options: .storageModeShared)!
  let count = ids.count * tables.count * cols
  let out = device.makeBuffer(length: count * 4, options: .storageModeShared)!
  let command = queue.makeCommandBuffer()!
  let encoder = command.makeComputeCommandEncoder()!
  encoder.setComputePipelineState(gather)
  encoder.setBuffer(idBuffer, offset: 0, index: 0)
  encoder.setBuffer(buffer, offset: 0, index: 1)
  encoder.setBytes(tables, length: tables.count * MemoryLayout<GatherTable>.stride, index: 2)
  encoder.setBytes(&shape, length: MemoryLayout<GatherShape>.stride, index: 3)
  encoder.setBuffer(out, offset: 0, index: 4)
  encoder.dispatchThreads(
    MTLSize(width: cols / 4, height: ids.count, depth: tables.count),
    threadsPerThreadgroup: MTLSize(width: min(cols / 4, 64), height: 4, depth: 1))
  encoder.endEncoding()
  command.commit()
  command.waitUntilCompleted()
  if let error = command.error { fail("gather: \(error)") }
  let rows = Array(UnsafeBufferPointer(start: out.contents().assumingMemoryBound(to: Float.self), count: count))
  return (rows, command.gpuEndTime - command.gpuStartTime)
}

func bitsEqual(_ a: [Float], _ b: [Float]) -> Int? {
  for i in a.indices where a[i].bitPattern != b[i].bitPattern { return i }
  return a.count == b.count ? nil : min(a.count, b.count)
}

// MARK: - The embedders

var failures = 0
var references: (ids: [Int32], rows: [String: [Float]])?
if args.count > 4 {
  let data = try Data(contentsOf: URL(fileURLWithPath: args[4]))
  let n = Int(data.withUnsafeBytes { $0.load(as: Int32.self) })
  let ids = data.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)[1...n]) }
  var offset = 4 * (n + 1)
  var rows: [String: [Float]] = [:]
  for name in ["tf_lite_embedder", "tf_lite_per_layer_embedder"] {
    let layout = layouts[name]!
    let count = n * layout.tables.count * layout.tables[0].cols
    rows[name] = data.withUnsafeBytes { Array(UnsafeRawBufferPointer(rebasing: $0[offset..<(offset + count * 4)]).bindMemory(to: Float.self)) }
    offset += count * 4
  }
  references = (ids, rows)
}

for name in ["tf_lite_embedder", "tf_lite_per_layer_embedder"] {
  let layout = layouts[name]!
  let (buffer, tables) = sectionBuffer(layout)
  if let references {
    let got = gpuRows(layout, buffer: buffer, tables: tables, ids: references.ids).rows
    if let index = bitsEqual(got, references.rows[name]!) {
      print("\(name): the interpreter's \(references.ids.count) rows DIFFER at element \(index)")
      failures += 1
    } else {
      print("\(name): the interpreter's \(references.ids.count) rows bit-identical")
    }
  }
  let rows = layout.tables[0].rows
  let batch = 4_096
  var mismatches = 0
  var gpuSeconds = 0.0
  for first in stride(from: 0, to: rows, by: batch) {
    var ids = (first..<min(first + batch, rows)).map(Int32.init)
    if first == 0 { ids[1] = -5; ids[2] = Int32(rows) + 3 }  // out of range: row 0, as the graph does
    let gpu = gpuRows(layout, buffer: buffer, tables: tables, ids: ids)
    gpuSeconds += gpu.seconds
    if let index = bitsEqual(gpu.rows, cpuRows(layout, ids: ids)) {
      mismatches += 1
      if mismatches <= 3 { print("\(name): batch at \(first) differs at element \(index)") }
    }
  }
  print("\(name): all \(rows) tokens \(mismatches == 0 ? "bit-identical" : "DIFFER in \(mismatches) batches") (\(String(format: "%.1f", gpuSeconds * 1000)) ms of GPU)")
  failures += mismatches == 0 ? 0 : 1
  // A round's own sizes: one token (a drafter step), four (verify).
  for n in [1, 4] {
    let ids = (0..<n).map { _ in Int32.random(in: 0..<Int32(rows)) }
    _ = gpuRows(layout, buffer: buffer, tables: tables, ids: ids)
    let times = (0..<20).map { _ in gpuRows(layout, buffer: buffer, tables: tables, ids: ids).seconds }.sorted()
    print("  \(n) token(s): \(String(format: "%.1f", times[10] * 1e6)) µs of GPU (median of 20)")
  }
}

// MARK: - The argmax

/// The CPU sampler's MaxElement (runtime/components/sampling_cpu_util.cc), ported as written.
func maxElement(_ x: [Float]) -> Int {
  let tile = 128
  var maxValue = x[0]
  var maxIndex = 0
  var i = 0
  while i + tile <= x.count {
    var lanes = Array(x[i..<(i + 8)])
    var k = i + 8
    while k < i + tile {
      for j in 0..<8 { lanes[j] = Swift.max(lanes[j], x[k + j]) }
      k += 8
    }
    let tileMax = lanes.max()!
    if tileMax > maxValue {
      for j in 0..<tile where x[i + j] == tileMax {
        maxValue = tileMax
        maxIndex = i + j
        break
      }
    }
    i += tile
  }
  while i < x.count {
    if x[i] > maxValue { maxValue = x[i]; maxIndex = i }
    i += 1
  }
  return maxIndex
}

let vocab = 262_144
var rowsOfLogits: [[Float]] = []
for trial in 0..<96 {
  var row = (0..<vocab).map { _ in Float.random(in: -30...30) }
  switch trial % 8 {
  case 0: break
  case 1: let v: Float = 40; for _ in 0..<3 { row[Int.random(in: 0..<vocab)] = v }  // tied maxima
  case 2: row[0] = 50; row[vocab - 1] = 50  // ends tied
  case 3: row[vocab - 1] = 50
  case 4: row = row.map { -abs($0) - 1 }; row[777] = -0.0; row[555] = 0.0; row[999] = -0.0  // signed zeros
  case 5: row = [Float](repeating: 1.5, count: vocab)  // all equal
  case 6: let v = row.max()! ; row[Int.random(in: 0..<vocab)] = v  // a tie with the random max
  default: row = row.map { Float(Float16($0)) }  // values a half can hold: ties are common
  }
  rowsOfLogits.append(row)
}

func gpuArgmax(_ rows: [[Float]], half: Bool) -> [Int] {
  let flat = rows.flatMap { $0 }
  let input: MTLBuffer = half
    ? device.makeBuffer(bytes: flat.map { Float16($0) }, length: flat.count * 2, options: .storageModeShared)!
    : device.makeBuffer(bytes: flat, length: flat.count * 4, options: .storageModeShared)!
  var n = UInt32(vocab)
  let out = device.makeBuffer(length: rows.count * 4, options: .storageModeShared)!
  let command = queue.makeCommandBuffer()!
  let encoder = command.makeComputeCommandEncoder()!
  encoder.setComputePipelineState(half ? argmax16 : argmax32)
  encoder.setBuffer(input, offset: 0, index: 0)
  encoder.setBytes(&n, length: 4, index: 1)
  encoder.setBuffer(out, offset: 0, index: 2)
  encoder.setThreadgroupMemoryLength(1_024 * 8, index: 0)
  encoder.dispatchThreadgroups(MTLSize(width: rows.count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1_024, height: 1, depth: 1))
  encoder.endEncoding()
  command.commit()
  command.waitUntilCompleted()
  if let error = command.error { fail("argmax: \(error)") }
  return Array(UnsafeBufferPointer(start: out.contents().assumingMemoryBound(to: Int32.self), count: rows.count)).map(Int.init)
}

for half in [false, true] {
  let rows = half ? rowsOfLogits.map { $0.map { Float(Float16($0)) } } : rowsOfLogits
  let want = rows.map(maxElement)
  let got = gpuArgmax(rows, half: half)
  let wrong = zip(got, want).enumerated().filter { $0.element.0 != $0.element.1 }
  print("argmax \(half ? "f16" : "f32"): \(rows.count - wrong.count)/\(rows.count) rows as the CPU sampler picks"
    + (wrong.isEmpty ? "" : "; first wrong row \(wrong[0].offset): \(wrong[0].element.0) vs \(wrong[0].element.1)"))
  failures += wrong.isEmpty ? 0 : 1
}

// MARK: - The concatenation

let first = (0..<2_560).map { _ in Float.random(in: -1...1) }
let second = (0..<2_560).map { _ in Float.random(in: -1...1) }
var sizes = SIMD2<UInt32>(2_560, 2_560)
let a = device.makeBuffer(bytes: first, length: first.count * 4)!
let b = device.makeBuffer(bytes: second, length: second.count * 4)!
let joined = device.makeBuffer(length: 5_120 * 4)!
let command = queue.makeCommandBuffer()!
let encoder = command.makeComputeCommandEncoder()!
encoder.setComputePipelineState(concat)
encoder.setBuffer(a, offset: 0, index: 0)
encoder.setBuffer(b, offset: 0, index: 1)
encoder.setBytes(&sizes, length: 8, index: 2)
encoder.setBuffer(joined, offset: 0, index: 3)
encoder.dispatchThreads(MTLSize(width: 5_120, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
encoder.endEncoding()
command.commit()
command.waitUntilCompleted()
let concatenated = Array(UnsafeBufferPointer(start: joined.contents().assumingMemoryBound(to: Float.self), count: 5_120))
let concatOK = bitsEqual(concatenated, first + second) == nil
print("concat: \(concatOK ? "bit-identical" : "DIFFERS")")
failures += concatOK ? 0 : 1

print(failures == 0 ? "All kernels match the host path." : "\(failures) check(s) failed.")
exit(failures == 0 ? 0 : 1)
