// Copyright 2026 Patrice Déry
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "runtime/executor/agello_resident_round.h"

namespace litert::lm {
namespace {

// The kernels of tools/agello-gpu-round/kernels.metal that the round uses, compiled with fast
// math off. The argmax follows the CPU sampler's GreedyHalf (top_p_cpu_sampler.cc): the first
// maximum, the two zeros equal, and -1 when the maximum is a positive NaN.
constexpr char kKernels[] = R"metal(
#include <metal_stdlib>
using namespace metal;

struct Shape {
  uint rows;
  uint cols;
  uint has_mul;
  float mul;
  uint rows_offset;
};

// One thread per byte of the row (four values). The id outside [0, rows) reads row 0, as the
// embedder graph's bounds select does.
kernel void agello_embed_gather(
    device const int* id [[buffer(0)]], device const uchar* rows [[buffer(1)]],
    device const float* scales [[buffer(2)]], constant Shape& shape [[buffer(3)]],
    device float* out [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  const uint bytes_per_row = shape.cols / 4;
  if (gid >= bytes_per_row) return;
  const int token = id[0];
  const uint row = (token >= 0 && uint(token) < shape.rows) ? uint(token) : 0u;
  const uchar packed = rows[shape.rows_offset + row * bytes_per_row + gid];
  const float scale = scales[row];
  for (uint k = 0; k < 4; ++k) {
    int q = int((packed >> (2 * k)) & 3);
    q = q >= 2 ? q - 4 : q;
    float v = float(q) * scale;
    if (shape.has_mul) v = v * shape.mul;
    out[gid * 4 + k] = v;
  }
}

struct TableRef {
  uint rows_offset;
  uint scales_offset;
};

struct TablesShape {
  uint rows;
  uint cols;
  uint tokens;
  uint tables;
  uint has_mul;
  float mul;
};

// Verify's inputs: one thread per (byte of a row, token, table), four values each; outputs are
// tables x cols per token, in table order (the per-layer embedder's concatenation).
kernel void agello_gather_tables(
    device const int* ids [[buffer(0)]], device const uchar* base [[buffer(1)]],
    constant TableRef* tables [[buffer(2)]], constant TablesShape& shape [[buffer(3)]],
    device float* out [[buffer(4)]], uint3 gid [[thread_position_in_grid]]) {
  const uint bytes_per_row = shape.cols / 4;
  if (gid.x >= bytes_per_row || gid.y >= shape.tokens || gid.z >= shape.tables) return;
  const int token = ids[gid.y];
  const uint row = (token >= 0 && uint(token) < shape.rows) ? uint(token) : 0u;
  const TableRef table = tables[gid.z];
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
  uint nan;
};

static inline Best better(Best a, Best b) {
  Best c = (b.value > a.value || (b.value == a.value && b.index < a.index)) ? b : a;
  c.nan = a.nan | b.nan;
  return c;
}

kernel void agello_argmax_f16(
    device const half* logits [[buffer(0)]], constant uint& vocab [[buffer(1)]],
    device int* out [[buffer(2)]], uint lane [[thread_position_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]], threadgroup Best* shared [[threadgroup(0)]]) {
  Best best = {-INFINITY, 0xffffffffu, 0u};
  for (uint i = lane; i < vocab; i += lanes) {
    const half h = logits[i];
    const ushort bits = as_type<ushort>(h);
    if ((bits & 0x7c00u) == 0x7c00u && (bits & 0x3ffu) != 0u) {
      // A positive NaN is the CPU's maximum (it returns -1); a negative one never wins.
      if ((bits & 0x8000u) == 0u) best.nan = 1u;
      continue;
    }
    Best candidate = {float(h), i, 0u};
    best = better(best, candidate);
  }
  shared[lane] = best;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = lanes / 2; stride > 0; stride /= 2) {
    if (lane < stride) shared[lane] = better(shared[lane], shared[lane + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0) out[0] = shared[0].nan ? -1 : int(shared[0].index);
}
)metal";

struct TableRef {
  uint32_t rows_offset;
  uint32_t scales_offset;
};

struct TablesShape {
  uint32_t rows;
  uint32_t cols;
  uint32_t tokens;
  uint32_t tables;
  uint32_t has_mul;
  float mul;
};

// One table set as the GPU reads it: a no-copy buffer over its section and the tables in it.
struct GpuTableSet {
  id<MTLBuffer> base = nil;
  std::vector<TableRef> tables;
  TablesShape shape{};
};

struct GatherShape {
  uint32_t rows;
  uint32_t cols;
  uint32_t has_mul;
  float mul;
  uint32_t rows_offset;
};

class MetalResidentRound : public ResidentRound {
 public:
  MetalResidentRound(id<MTLCommandQueue> queue, id<MTLComputePipelineState> gather,
                     id<MTLComputePipelineState> argmax, id<MTLComputePipelineState> gather_tables,
                     id<MTLBuffer> table, id<MTLBuffer> scales, id<MTLBuffer> ids,
                     GatherShape shape)
      : queue_(queue),
        gather_(gather),
        argmax_(argmax),
        gather_tables_(gather_tables),
        table_(table),
        scales_(scales),
        ids_(ids),
        shape_(shape) {}

  void SetFirstId(int token_id) override {
    static_cast<int32_t*>([ids_ contents])[0] = token_id;
    last_ = nil;
  }

  bool EncodeInput(int step, void* activations, void* previous, size_t previous_offset) override {
    id<MTLBuffer> input = (__bridge id<MTLBuffer>)activations;
    id<MTLBuffer> source = (__bridge id<MTLBuffer>)previous;
    const size_t half_bytes = static_cast<size_t>(shape_.cols) * sizeof(float);
    if (input == nil || source == nil || input.length < 2 * half_bytes ||
        source.length < previous_offset + half_bytes) {
      return false;
    }
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> compute = [command computeCommandEncoder];
    [compute setComputePipelineState:gather_];
    [compute setBuffer:ids_ offset:static_cast<NSUInteger>(step) * sizeof(int32_t) atIndex:0];
    [compute setBuffer:table_ offset:0 atIndex:1];
    [compute setBuffer:scales_ offset:0 atIndex:2];
    [compute setBytes:&shape_ length:sizeof(shape_) atIndex:3];
    [compute setBuffer:input offset:0 atIndex:4];
    const NSUInteger threads = shape_.cols / 4;
    [compute dispatchThreads:MTLSizeMake(threads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(MIN(threads, gather_.maxTotalThreadsPerThreadgroup), 1, 1)];
    [compute endEncoding];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    [blit copyFromBuffer:source
             sourceOffset:previous_offset
                 toBuffer:input
        destinationOffset:half_bytes
                     size:half_bytes];
    [blit endEncoding];
    [command commit];
    last_ = command;
    return true;
  }

  bool EncodeArgmax(int step, void* logits, int vocab) override {
    id<MTLBuffer> row = (__bridge id<MTLBuffer>)logits;
    if (row == nil || row.length < static_cast<size_t>(vocab) * 2) return false;
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> compute = [command computeCommandEncoder];
    [compute setComputePipelineState:argmax_];
    [compute setBuffer:row offset:0 atIndex:0];
    uint32_t count = static_cast<uint32_t>(vocab);
    [compute setBytes:&count length:sizeof(count) atIndex:1];
    [compute setBuffer:ids_ offset:static_cast<NSUInteger>(step + 1) * sizeof(int32_t) atIndex:2];
    const NSUInteger lanes = MIN(static_cast<NSUInteger>(1024), argmax_.maxTotalThreadsPerThreadgroup);
    [compute setThreadgroupMemoryLength:lanes * 12 atIndex:0];
    [compute dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(lanes, 1, 1)];
    [compute endEncoding];
    [command commit];
    last_ = command;
    return true;
  }

  bool AddVerifyTables(const TableSet& embedder, const TableSet& per_layer) override {
    GpuTableSet sets[2];
    const TableSet* sources[2] = {&embedder, &per_layer};
    const size_t page = static_cast<size_t>(getpagesize());
    for (int i = 0; i < 2; ++i) {
      const TableSet& source = *sources[i];
      if (source.section == nullptr || source.tables.empty() || source.rows <= 0 ||
          source.cols <= 0 || source.cols % 4 != 0 || source.section_bytes > UINT32_MAX ||
          reinterpret_cast<uintptr_t>(source.section) % page != 0) {
        return false;
      }
      const size_t row_bytes = static_cast<size_t>(source.cols) / 4;
      for (const auto& [rows_offset, scales_offset] : source.tables) {
        if (rows_offset + static_cast<size_t>(source.rows) * row_bytes > source.section_bytes ||
            scales_offset % 4 != 0 ||
            scales_offset + static_cast<size_t>(source.rows) * sizeof(float) > source.section_bytes) {
          return false;
        }
        sets[i].tables.push_back({static_cast<uint32_t>(rows_offset),
                                  static_cast<uint32_t>(scales_offset)});
      }
      const size_t length = (source.section_bytes + page - 1) / page * page;
      sets[i].base = [queue_.device newBufferWithBytesNoCopy:const_cast<void*>(source.section)
                                                      length:length
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
      if (sets[i].base == nil) return false;
      sets[i].shape = {static_cast<uint32_t>(source.rows), static_cast<uint32_t>(source.cols), 0,
                       static_cast<uint32_t>(source.tables.size()), source.mul != 1.0f ? 1u : 0u,
                       source.mul};
    }
    embedder_ = sets[0];
    per_layer_ = sets[1];
    // One gather now, waited for: the first use of these buffers took 0.68 s on the M6 (making
    // them resident), which belongs in the load, not in the first reply's first round.
    static_cast<int32_t*>([ids_ contents])[0] = 0;
    id<MTLBuffer> warm_embedding = [queue_.device
        newBufferWithLength:embedder_.tables.size() * embedder_.shape.cols * sizeof(float)
                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> warm_per_layer = [queue_.device
        newBufferWithLength:per_layer_.tables.size() * per_layer_.shape.cols * sizeof(float)
                    options:MTLResourceStorageModeShared];
    if (!EncodeVerifyInputs(1, (__bridge void*)warm_embedding, (__bridge void*)warm_per_layer)) {
      embedder_ = GpuTableSet();
      per_layer_ = GpuTableSet();
      return false;
    }
    [last_ waitUntilCompleted];
    last_ = nil;
    NSLog(@"Agello resident round: verify's inputs on the GPU (%zu and %zu tables)",
          embedder_.tables.size(), per_layer_.tables.size());
    return true;
  }

  bool HasVerifyTables() const override { return embedder_.base != nil && per_layer_.base != nil; }

  bool EncodeVerifyInputs(int tokens, void* embeddings, void* per_layer) override {
    id<MTLBuffer> outputs[2] = {(__bridge id<MTLBuffer>)embeddings, (__bridge id<MTLBuffer>)per_layer};
    GpuTableSet* sets[2] = {&embedder_, &per_layer_};
    if (!HasVerifyTables()) return false;
    for (int i = 0; i < 2; ++i) {
      const size_t needed = static_cast<size_t>(tokens) * sets[i]->tables.size() * sets[i]->shape.cols * sizeof(float);
      if (outputs[i] == nil || outputs[i].length < needed) return false;
    }
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> compute = [command computeCommandEncoder];
    [compute setComputePipelineState:gather_tables_];
    for (int i = 0; i < 2; ++i) {
      TablesShape shape = sets[i]->shape;
      shape.tokens = static_cast<uint32_t>(tokens);
      [compute setBuffer:ids_ offset:0 atIndex:0];
      [compute setBuffer:sets[i]->base offset:0 atIndex:1];
      [compute setBytes:sets[i]->tables.data() length:sets[i]->tables.size() * sizeof(TableRef) atIndex:2];
      [compute setBytes:&shape length:sizeof(shape) atIndex:3];
      [compute setBuffer:outputs[i] offset:0 atIndex:4];
      const NSUInteger width = shape.cols / 4;
      [compute dispatchThreads:MTLSizeMake(width, static_cast<NSUInteger>(tokens), shape.tables)
          threadsPerThreadgroup:MTLSizeMake(MIN(width, static_cast<NSUInteger>(64)), 4, 1)];
    }
    [compute endEncoding];
    [command commit];
    last_ = command;
    return true;
  }

  bool Finish(int count, int* ids) override {
    if (last_ == nil) return false;
    [last_ waitUntilCompleted];
    const bool ok = last_.status == MTLCommandBufferStatusCompleted;
    last_ = nil;
    if (!ok) return false;
    const int32_t* drafted = static_cast<const int32_t*>([ids_ contents]);
    for (int i = 0; i < count; ++i) {
      if (drafted[i + 1] < 0) return false;
      ids[i] = drafted[i + 1];
    }
    return true;
  }

 private:
  id<MTLCommandQueue> queue_;
  id<MTLComputePipelineState> gather_;
  id<MTLComputePipelineState> argmax_;
  id<MTLComputePipelineState> gather_tables_;
  GpuTableSet embedder_;
  GpuTableSet per_layer_;
  id<MTLBuffer> table_;
  id<MTLBuffer> scales_;
  id<MTLBuffer> ids_;
  GatherShape shape_;
  id<MTLCommandBuffer> last_ = nil;
};

}  // namespace

namespace {
// The drafter chain (2) unless AGELLO_RESIDENT_ROUND says 0 (off) or 1 (verify's inputs too).
char ResidentRoundMode() {
  static const char mode = [] {
    const char* value = std::getenv("AGELLO_RESIDENT_ROUND");
    return value != nullptr && (value[0] == '0' || value[0] == '1') ? value[0] : '2';
  }();
  return mode;
}
}  // namespace

bool ResidentRoundEnabled() { return ResidentRoundMode() != '0'; }

bool ResidentRoundGathersVerifyInputs() { return ResidentRoundMode() == '1'; }

std::unique_ptr<ResidentRound> ResidentRound::Create(const void* command_queue, const Table& table,
                                                     int max_steps) {
  @autoreleasepool {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)command_queue;
    if (queue == nil || table.section == nullptr || table.scales == nullptr || table.rows <= 0 ||
        table.cols <= 0 || table.cols % 4 != 0) {
      NSLog(@"Agello resident round: no queue or table; the host drafts");
      return nullptr;
    }
    const size_t page = static_cast<size_t>(getpagesize());
    const size_t row_bytes = static_cast<size_t>(table.cols) / 4;
    if (reinterpret_cast<uintptr_t>(table.section) % page != 0 ||
        table.rows_offset + static_cast<size_t>(table.rows) * row_bytes > table.section_bytes) {
      NSLog(@"Agello resident round: the embedder section isn't page-aligned or the table overruns it");
      return nullptr;
    }
    id<MTLDevice> device = queue.device;
    // The mapped pages, read in place: no copy, and no footprint beyond the pages the GPU reads.
    const size_t length = (table.section_bytes + page - 1) / page * page;
    id<MTLBuffer> rows = [device newBufferWithBytesNoCopy:const_cast<void*>(table.section)
                                                   length:length
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
    id<MTLBuffer> scales = [device newBufferWithBytes:table.scales
                                               length:static_cast<size_t>(table.rows) * sizeof(float)
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> ids = [device newBufferWithLength:static_cast<size_t>(max_steps + 1) * sizeof(int32_t)
                                            options:MTLResourceStorageModeShared];
    if (rows == nil || scales == nil || ids == nil) {
      NSLog(@"Agello resident round: buffers failed; the host drafts");
      return nullptr;
    }
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    options.mathMode = MTLMathModeSafe;
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:@(kKernels) options:options error:&error];
    if (library == nil) {
      NSLog(@"Agello resident round: kernels didn't compile: %@", error);
      return nullptr;
    }
    id<MTLFunction> gather_function = [library newFunctionWithName:@"agello_embed_gather"];
    id<MTLFunction> argmax_function = [library newFunctionWithName:@"agello_argmax_f16"];
    id<MTLFunction> tables_function = [library newFunctionWithName:@"agello_gather_tables"];
    if (gather_function == nil || argmax_function == nil || tables_function == nil ||
        table.section_bytes > UINT32_MAX) {
      NSLog(@"Agello resident round: kernels missing; the host drafts");
      return nullptr;
    }
    id<MTLComputePipelineState> gather = [device newComputePipelineStateWithFunction:gather_function
                                                                               error:&error];
    id<MTLComputePipelineState> argmax = [device newComputePipelineStateWithFunction:argmax_function
                                                                               error:&error];
    id<MTLComputePipelineState> gather_tables =
        [device newComputePipelineStateWithFunction:tables_function error:&error];
    if (gather == nil || argmax == nil || gather_tables == nil) {
      NSLog(@"Agello resident round: pipelines failed: %@", error);
      return nullptr;
    }
    GatherShape shape{static_cast<uint32_t>(table.rows), static_cast<uint32_t>(table.cols),
                      table.mul != 1.0f ? 1u : 0u, table.mul,
                      static_cast<uint32_t>(table.rows_offset)};
    NSLog(@"Agello resident round: on (%d x %d table, mul %g)", table.rows, table.cols, table.mul);
    return std::make_unique<MetalResidentRound>(queue, gather, argmax, gather_tables, rows, scales,
                                                ids, shape);
  }
}

}  // namespace litert::lm
