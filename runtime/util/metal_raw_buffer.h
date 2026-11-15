// Copyright 2026 Patrice Déry (Agello, PowerBeef/agello-litert-lm)
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

// Agello (T3): the CPU view of a Metal buffer. Apple GPUs share memory with the CPU, so the bytes
// of a buffer with shared storage can be copied as they are, in the GPU's own layout, without a
// lock's layout conversion.
#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_METAL_RAW_BUFFER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_METAL_RAW_BUFFER_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace litert::lm {

struct MetalRawBuffer {
  void* contents = nullptr;
  size_t length = 0;
};

// The contents and length of `mtl_buffer` (an id<MTLBuffer>) when its storage is shared;
// nullopt otherwise, and on platforms without Metal.
std::optional<MetalRawBuffer> GetMetalRawBuffer(void* mtl_buffer);

// Host memory for a raw snapshot of `length` bytes, in pages of its own that FreeSnapshotMemory
// returns to the system at once (a freed malloc'd buffer this large stays resident in the
// allocator's cache: +0.47 GB of footprint after a 16K snapshot). Page-aligned; nullptr on
// failure and on platforms without Metal.
void* AllocateSnapshotMemory(size_t length);
void FreeSnapshotMemory(void* addr);

// Agello (sparse KV, step 1): a byte range copied between a Metal buffer and host memory.
struct MetalBlitRange {
  size_t buffer_offset = 0;
  size_t host_offset = 0;
  size_t length = 0;
};

// Copies `ranges` between `mtl_buffer` (an id<MTLBuffer>, shared or private) and `host` (from
// AllocateSnapshotMemory, `host_length` bytes) with one blit on `command_queue` (an
// id<MTLCommandQueue>: the queue the GPU's runs use, so the copy is ordered after them), and
// waits for it. `to_host` copies the buffer's ranges into the host memory, otherwise back. False
// on failure, and on platforms without Metal.
bool BlitMetalRanges(const void* command_queue, void* mtl_buffer, void* host, size_t host_length,
                     const std::vector<MetalBlitRange>& ranges, bool to_host);

// The length of `mtl_buffer` (an id<MTLBuffer>), 0 without one.
size_t MetalBufferLength(void* mtl_buffer);

// Agello (sparse KV, steps 2-3): KV caches as Metal 4 placement sparse buffers, which cost only
// their mapped tiles (an ordinary buffer that any kernel uses is wholly resident). Reads of an
// unmapped tile return zero and writes there are dropped (tier 1).

// The device of `mtl_buffer` (an id<MTLBuffer>), unretained; nullptr without one.
void* MetalDeviceOfBuffer(void* mtl_buffer);

// Whether `mtl_device` (an id<MTLDevice>) has placement sparse buffers (tier 1).
bool PlacementSparseBuffersSupported(void* mtl_device);

// The sparse tile the KV caches use (16 KB).
size_t SparseKvTileBytes();

// A private, hazard-tracked placement sparse buffer of `length` bytes with 16 KB tiles, none
// mapped; retained (ReleaseMetalObject). nullptr on failure.
void* CreateSparseMetalBuffer(void* mtl_device, size_t length);

// Releases an object a Create function above retained.
void ReleaseMetalObject(void* object);

// A byte range of a placement sparse buffer (mapped as the 16 KB tiles it touches).
struct SparseRange {
  void* buffer = nullptr;  // id<MTLBuffer>
  size_t offset = 0;
  size_t length = 0;
};

// Maps tiles of placement sparse buffers from placement heaps, one heap per step, through a
// Metal 4 queue of its own, and waits for each update. Steps unmap last in, first out.
class SparseKvMapper {
 public:
  // `command_queue` (an id<MTLCommandQueue>) runs the kernels that use the buffers: an unmap
  // waits for its work first.
  SparseKvMapper(void* mtl_device, const void* command_queue);
  ~SparseKvMapper();
  SparseKvMapper(const SparseKvMapper&) = delete;
  SparseKvMapper& operator=(const SparseKvMapper&) = delete;

  bool ok() const;
  // A new heap holding the tiles the ranges touch that no earlier step mapped, mapped into them.
  bool MapStep(const std::vector<SparseRange>& ranges);
  // Unmaps the last step's ranges, once the queue's work is done, and frees its heap.
  bool UnmapLastStep();
  int steps() const;
  size_t mapped_bytes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_METAL_RAW_BUFFER_H_
