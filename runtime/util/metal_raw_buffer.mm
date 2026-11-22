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

#import <Metal/Metal.h>

#include "runtime/util/metal_raw_buffer.h"

#include <sys/mman.h>
#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <memory>
#include <optional>
#include <vector>

namespace litert::lm {

std::optional<MetalRawBuffer> GetMetalRawBuffer(void* mtl_buffer) {
  if (mtl_buffer == nullptr) return std::nullopt;
  id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)mtl_buffer;
  if (buffer.storageMode != MTLStorageModeShared || buffer.contents == nullptr) {
    return std::nullopt;
  }
  return MetalRawBuffer{buffer.contents, static_cast<size_t>(buffer.length)};
}

// The mapping starts one page before the returned address; that page holds its total length.
void* AllocateSnapshotMemory(size_t length) {
  const size_t page = static_cast<size_t>(getpagesize());
  const size_t total = page + (length + page - 1) / page * page;
  void* base = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (base == MAP_FAILED) return nullptr;
  *static_cast<size_t*>(base) = total;
  return static_cast<char*>(base) + page;
}

void FreeSnapshotMemory(void* addr) {
  if (addr == nullptr) return;
  char* base = static_cast<char*>(addr) - getpagesize();
  munmap(base, *reinterpret_cast<size_t*>(base));
}

bool BlitMetalRanges(const void* command_queue, void* mtl_buffer, void* host, size_t host_length,
                     const std::vector<MetalBlitRange>& ranges, bool to_host) {
  if (command_queue == nullptr || mtl_buffer == nullptr || host == nullptr) return false;
  if (host_length == 0) return true;
  @autoreleasepool {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)const_cast<void*>(command_queue);
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)mtl_buffer;
    // The host memory as a buffer of its own pages (AllocateSnapshotMemory's are page-aligned
    // and whole): written or read only by this blit.
    const size_t page = static_cast<size_t>(getpagesize());
    const size_t wrapped = (host_length + page - 1) / page * page;
    id<MTLBuffer> host_buffer = [buffer.device newBufferWithBytesNoCopy:host
                                                                 length:wrapped
                                                                options:MTLResourceStorageModeShared
                                                            deallocator:nil];
    if (host_buffer == nil) {
      fprintf(stderr, "Agello KV blit: host memory of %zu bytes can't be wrapped\n", wrapped);
      return false;
    }
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    command_buffer.label = to_host ? @"Agello KV snapshot" : @"Agello KV restore";
    id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
    for (const MetalBlitRange& range : ranges) {
      if (range.length == 0) continue;
      if (to_host) {
        [blit copyFromBuffer:buffer
                 sourceOffset:range.buffer_offset
                     toBuffer:host_buffer
            destinationOffset:range.host_offset
                         size:range.length];
      } else {
        [blit copyFromBuffer:host_buffer
                 sourceOffset:range.host_offset
                     toBuffer:buffer
            destinationOffset:range.buffer_offset
                         size:range.length];
      }
    }
    [blit endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.error != nil) {
      fprintf(stderr, "Agello KV blit failed: %s\n",
              command_buffer.error.localizedDescription.UTF8String);
      return false;
    }
    return true;
  }
}

size_t MetalBufferLength(void* mtl_buffer) {
  if (mtl_buffer == nullptr) return 0;
  return static_cast<size_t>(((__bridge id<MTLBuffer>)mtl_buffer).length);
}

void* MetalDeviceOfBuffer(void* mtl_buffer) {
  if (mtl_buffer == nullptr) return nullptr;
  return (__bridge void*)((__bridge id<MTLBuffer>)mtl_buffer).device;
}

bool PlacementSparseBuffersSupported(void* mtl_device) {
  if (mtl_device == nullptr) return false;
  if (@available(macOS 26.4, iOS 26.4, *)) {
    return ((__bridge id<MTLDevice>)mtl_device).supportsPlacementSparse;
  }
  return false;
}

size_t SparseKvTileBytes() { return 16 * 1024; }

void* CreateSparseMetalBuffer(void* mtl_device, size_t length) {
  if (mtl_device == nullptr || length == 0) return nullptr;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)mtl_device;
    id<MTLBuffer> buffer =
        [device newBufferWithLength:length
                            options:MTLResourceStorageModePrivate |
                                    MTLResourceHazardTrackingModeTracked
            placementSparsePageSize:MTLSparsePageSize16];
    if (buffer == nil || buffer.sparseBufferTier != MTLBufferSparseTier1) return nullptr;
    buffer.label = @"Agello sparse KV cache";
    return (__bridge_retained void*)buffer;
  }
  return nullptr;
}

void ReleaseMetalObject(void* object) {
  if (object != nullptr) CFRelease(object);
}

struct SparseKvMapper::Impl {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> command_queue = nil;
  id<MTL4CommandQueue> mapping_queue = nil;
  id<MTLSharedEvent> event = nil;
  uint64_t event_value = 0;
  struct Step {
    id<MTLHeap> heap = nil;
    std::vector<SparseRange> ranges;  // the tiles this step mapped
    size_t bytes = 0;
  };
  std::vector<Step> steps;
  size_t mapped_bytes = 0;
  // Each buffer's mapped tiles: a range's tiles may straddle another's (rows that don't start on
  // a tile), so a step maps only the tiles no earlier step did.
  std::vector<std::pair<void*, std::vector<bool>>> tiles;

  std::vector<bool>& TilesOf(void* buffer) {
    for (auto& [b, bits] : tiles) {
      if (b == buffer) return bits;
    }
    const size_t length = static_cast<size_t>(((__bridge id<MTLBuffer>)buffer).length);
    tiles.emplace_back(buffer, std::vector<bool>((length + SparseKvTileBytes() - 1) /
                                                 SparseKvTileBytes()));
    return tiles.back().second;
  }

  void WaitForCommandQueue() {
    if (command_queue == nil) return;
    id<MTLCommandBuffer> fence = [command_queue commandBuffer];
    [fence commit];
    [fence waitUntilCompleted];
  }

  // Applies `mode` to the ranges, grouped by buffer, then waits for the mapping queue.
  bool Update(const std::vector<SparseRange>& ranges, id<MTLHeap> heap, bool map)
      API_AVAILABLE(macos(26.0), ios(26.0)) {
    const size_t tile = SparseKvTileBytes();
    size_t heap_tile = 0;
    std::vector<MTL4UpdateSparseBufferMappingOperation> operations;
    void* current = nullptr;
    auto flush = [&] {
      if (current == nullptr || operations.empty()) return;
      [mapping_queue updateBufferMappings:(__bridge id<MTLBuffer>)current
                                     heap:map ? heap : nil
                               operations:operations.data()
                                    count:operations.size()];
      operations.clear();
    };
    for (const SparseRange& range : ranges) {
      if (range.buffer != current) {
        flush();
        current = range.buffer;
      }
      MTL4UpdateSparseBufferMappingOperation operation;
      operation.mode = map ? MTLSparseTextureMappingModeMap : MTLSparseTextureMappingModeUnmap;
      operation.bufferRange = NSMakeRange(range.offset / tile, range.length / tile);
      operation.heapOffset = heap_tile;
      operations.push_back(operation);
      heap_tile += range.length / tile;
    }
    flush();
    ++event_value;
    [mapping_queue signalEvent:event value:event_value];
    return [event waitUntilSignaledValue:event_value timeoutMS:10000];
  }
};

namespace {

API_AVAILABLE(macos(26.0), ios(26.0))
void AppendTileRuns(const std::vector<bool>& bits,
                    std::vector<MTL4CopySparseBufferMappingOperation>* copy_ops,
                    std::vector<MTL4UpdateSparseBufferMappingOperation>* unmap_ops) {
  size_t t = 0;
  const size_t n = bits.size();
  while (t < n) {
    while (t < n && !bits[t]) ++t;
    if (t == n) break;
    const size_t start = t;
    while (t < n && bits[t]) ++t;
    const size_t count = t - start;
    if (copy_ops != nullptr) {
      MTL4CopySparseBufferMappingOperation op;
      op.sourceRange = NSMakeRange(start, count);
      op.destinationOffset = start;
      copy_ops->push_back(op);
    }
    if (unmap_ops != nullptr) {
      MTL4UpdateSparseBufferMappingOperation op;
      op.mode = MTLSparseTextureMappingModeUnmap;
      op.bufferRange = NSMakeRange(start, count);
      op.heapOffset = 0;
      unmap_ops->push_back(op);
    }
  }
}

}  // namespace

struct SparseKvMappingSnapshot::Impl {
  id<MTL4CommandQueue> mapping_queue = nil;
  id<MTLSharedEvent> event = nil;
  struct SavedBuffer {
    void* live_buffer = nullptr;
    id<MTLBuffer> shadow_buffer = nil;
    std::vector<bool> tiles;
  };
  std::vector<SavedBuffer> buffers;
  std::vector<SparseKvMapper::Impl::Step> steps;
  size_t mapped_bytes = 0;

  ~Impl() {
    if (buffers.empty()) return;
    if (@available(macOS 26.0, iOS 26.0, *)) {
      @autoreleasepool {
        if (mapping_queue != nil && event != nil) {
          std::vector<MTL4UpdateSparseBufferMappingOperation> unmap_ops;
          for (const SavedBuffer& b : buffers) {
            unmap_ops.clear();
            AppendTileRuns(b.tiles, nullptr, &unmap_ops);
            if (!unmap_ops.empty() && b.shadow_buffer != nil) {
              [mapping_queue updateBufferMappings:b.shadow_buffer
                                             heap:nil
                                       operations:unmap_ops.data()
                                            count:unmap_ops.size()];
            }
          }
          [mapping_queue signalEvent:event value:1];
          [event waitUntilSignaledValue:1 timeoutMS:10000];
        }
        buffers.clear();
        steps.clear();
      }
    }
  }
};

SparseKvMappingSnapshot::SparseKvMappingSnapshot() = default;
SparseKvMappingSnapshot::~SparseKvMappingSnapshot() = default;
SparseKvMappingSnapshot::SparseKvMappingSnapshot(SparseKvMappingSnapshot&&) noexcept = default;
SparseKvMappingSnapshot& SparseKvMappingSnapshot::operator=(SparseKvMappingSnapshot&&) noexcept =
    default;
size_t SparseKvMappingSnapshot::mapped_bytes() const {
  return impl_ ? impl_->mapped_bytes : 0;
}
bool SparseKvMappingSnapshot::empty() const { return !impl_ || impl_->steps.empty(); }

SparseKvMapper::SparseKvMapper(void* mtl_device, const void* command_queue)
    : impl_(std::make_unique<Impl>()) {
  if (mtl_device == nullptr) return;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    impl_->device = (__bridge id<MTLDevice>)mtl_device;
    impl_->command_queue = (__bridge id<MTLCommandQueue>)const_cast<void*>(command_queue);
    impl_->mapping_queue = [impl_->device newMTL4CommandQueue];
    impl_->event = [impl_->device newSharedEvent];
  }
}

SparseKvMapper::~SparseKvMapper() = default;

bool SparseKvMapper::ok() const {
  return impl_->device != nil && impl_->mapping_queue != nil && impl_->event != nil;
}

bool SparseKvMapper::MapStep(const std::vector<SparseRange>& ranges) {
  if (!ok()) return false;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    @autoreleasepool {
      const size_t tile = SparseKvTileBytes();
      // The tiles the ranges touch that no step has mapped, as runs of whole tiles.
      std::vector<SparseRange> fresh;
      size_t bytes = 0;
      for (const SparseRange& range : ranges) {
        if (range.length == 0) continue;
        std::vector<bool>& bits = impl_->TilesOf(range.buffer);
        const size_t first = range.offset / tile;
        const size_t last = std::min(bits.size(), (range.offset + range.length + tile - 1) / tile);
        for (size_t t = first; t < last; ++t) {
          if (bits[t]) continue;
          bits[t] = true;
          if (!fresh.empty() && fresh.back().buffer == range.buffer &&
              fresh.back().offset + fresh.back().length == t * tile) {
            fresh.back().length += tile;
          } else {
            fresh.push_back({range.buffer, t * tile, tile});
          }
          bytes += tile;
        }
      }
      if (bytes == 0) {
        impl_->steps.push_back(Impl::Step{});
        return true;
      }
      MTLHeapDescriptor* descriptor = [[MTLHeapDescriptor alloc] init];
      descriptor.type = MTLHeapTypePlacement;
      descriptor.storageMode = MTLStorageModePrivate;
      descriptor.size = bytes;
      descriptor.maxCompatiblePlacementSparsePageSize = MTLSparsePageSize16;
      id<MTLHeap> heap = [impl_->device newHeapWithDescriptor:descriptor];
      if (heap == nil) {
        fprintf(stderr, "Agello sparse KV: no heap of %zu bytes\n", bytes);
        return false;
      }
      heap.label = @"Agello sparse KV step";
      if (!impl_->Update(fresh, heap, /*map=*/true)) return false;
      impl_->steps.push_back(Impl::Step{heap, std::move(fresh), bytes});
      impl_->mapped_bytes += bytes;
      return true;
    }
  }
  return false;
}

bool SparseKvMapper::UnmapLastStep() {
  if (!ok() || impl_->steps.empty()) return false;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    @autoreleasepool {
      // Nothing in flight may still use the tiles: the queue runs in order, so an empty command
      // buffer committed after its work completes after it.
      impl_->WaitForCommandQueue();
      Impl::Step& step = impl_->steps.back();
      if (!step.ranges.empty() && !impl_->Update(step.ranges, nil, /*map=*/false)) return false;
      const size_t tile = SparseKvTileBytes();
      for (const SparseRange& range : step.ranges) {
        std::vector<bool>& bits = impl_->TilesOf(range.buffer);
        for (size_t t = range.offset / tile; t < (range.offset + range.length) / tile; ++t) {
          bits[t] = false;
        }
      }
      impl_->mapped_bytes -= step.bytes;
      impl_->steps.pop_back();
      return true;
    }
  }
  return false;
}

std::optional<SparseKvMappingSnapshot> SparseKvMapper::DetachMappings() {
  if (!ok()) return std::nullopt;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    @autoreleasepool {
      impl_->WaitForCommandQueue();
      SparseKvMappingSnapshot snapshot;
      if (impl_->steps.empty()) return snapshot;
      auto snap_impl = std::make_unique<SparseKvMappingSnapshot::Impl>();
      snap_impl->mapping_queue = impl_->mapping_queue;
      snap_impl->event = [impl_->device newSharedEvent];
      if (snap_impl->event == nil) return std::nullopt;

      // Allocate every virtual shadow sparse buffer (0 physical bytes) before mutating any live
      // mapping so a failure leaves the live mapper untouched.
      for (const auto& [live_ptr, bits] : impl_->tiles) {
        bool any = false;
        for (bool b : bits) {
          if (b) {
            any = true;
            break;
          }
        }
        if (!any) continue;
        id<MTLBuffer> live_buf = (__bridge id<MTLBuffer>)live_ptr;
        id<MTLBuffer> shadow =
            [impl_->device newBufferWithLength:live_buf.length
                                       options:MTLResourceStorageModePrivate |
                                               MTLResourceHazardTrackingModeTracked
                       placementSparsePageSize:MTLSparsePageSize16];
        if (shadow == nil || shadow.sparseBufferTier != MTLBufferSparseTier1) {
          return std::nullopt;
        }
        shadow.label = @"Agello sparse KV snapshot";
        snap_impl->buffers.push_back({live_ptr, shadow, bits});
      }

      std::vector<MTL4CopySparseBufferMappingOperation> copy_ops;
      std::vector<MTL4UpdateSparseBufferMappingOperation> unmap_ops;
      for (const auto& saved : snap_impl->buffers) {
        copy_ops.clear();
        unmap_ops.clear();
        AppendTileRuns(saved.tiles, &copy_ops, &unmap_ops);
        if (copy_ops.empty()) continue;
        id<MTLBuffer> live_buf = (__bridge id<MTLBuffer>)saved.live_buffer;
        [impl_->mapping_queue copyBufferMappingsFromBuffer:live_buf
                                                  toBuffer:saved.shadow_buffer
                                                operations:copy_ops.data()
                                                     count:copy_ops.size()];
        [impl_->mapping_queue updateBufferMappings:live_buf
                                              heap:nil
                                        operations:unmap_ops.data()
                                             count:unmap_ops.size()];
      }
      ++impl_->event_value;
      [impl_->mapping_queue signalEvent:impl_->event value:impl_->event_value];
      if (![impl_->event waitUntilSignaledValue:impl_->event_value timeoutMS:10000]) {
        return std::nullopt;
      }

      snap_impl->steps = std::move(impl_->steps);
      impl_->steps.clear();
      snap_impl->mapped_bytes = impl_->mapped_bytes;
      impl_->mapped_bytes = 0;
      for (auto& [_, bits] : impl_->tiles) {
        std::fill(bits.begin(), bits.end(), false);
      }
      snapshot.impl_ = std::move(snap_impl);
      return snapshot;
    }
  }
  return std::nullopt;
}

bool SparseKvMapper::RestoreMappings(SparseKvMappingSnapshot&& snapshot) {
  if (!ok()) return false;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    @autoreleasepool {
      impl_->WaitForCommandQueue();
      // Keep any currently mapped heaps (e.g. a helper's step) alive until the unmap completes.
      std::vector<Impl::Step> old_steps = std::move(impl_->steps);
      impl_->steps.clear();
      impl_->mapped_bytes = 0;

      std::vector<MTL4CopySparseBufferMappingOperation> copy_ops;
      std::vector<MTL4UpdateSparseBufferMappingOperation> unmap_ops;
      bool queued = false;
      for (auto& [live_ptr, bits] : impl_->tiles) {
        unmap_ops.clear();
        AppendTileRuns(bits, nullptr, &unmap_ops);
        if (!unmap_ops.empty()) {
          [impl_->mapping_queue updateBufferMappings:(__bridge id<MTLBuffer>)live_ptr
                                                heap:nil
                                          operations:unmap_ops.data()
                                               count:unmap_ops.size()];
          std::fill(bits.begin(), bits.end(), false);
          queued = true;
        }
      }

      if (snapshot.impl_ != nullptr) {
        for (const auto& saved : snapshot.impl_->buffers) {
          copy_ops.clear();
          unmap_ops.clear();
          AppendTileRuns(saved.tiles, &copy_ops, &unmap_ops);
          if (copy_ops.empty()) continue;
          id<MTLBuffer> live_buf = (__bridge id<MTLBuffer>)saved.live_buffer;
          [impl_->mapping_queue copyBufferMappingsFromBuffer:saved.shadow_buffer
                                                    toBuffer:live_buf
                                                  operations:copy_ops.data()
                                                       count:copy_ops.size()];
          [impl_->mapping_queue updateBufferMappings:saved.shadow_buffer
                                                heap:nil
                                          operations:unmap_ops.data()
                                               count:unmap_ops.size()];
          queued = true;
        }
      }

      if (queued) {
        ++impl_->event_value;
        [impl_->mapping_queue signalEvent:impl_->event value:impl_->event_value];
        if (![impl_->event waitUntilSignaledValue:impl_->event_value timeoutMS:10000]) {
          return false;
        }
      }

      if (snapshot.impl_ != nullptr) {
        for (const auto& saved : snapshot.impl_->buffers) {
          impl_->TilesOf(saved.live_buffer) = saved.tiles;
        }
        impl_->steps = std::move(snapshot.impl_->steps);
        impl_->mapped_bytes = snapshot.impl_->mapped_bytes;
        snapshot.impl_->buffers.clear();
        snapshot.impl_->steps.clear();
        snapshot.impl_->mapped_bytes = 0;
      }
      // `old_steps` drops here, after the mapping queue finished unmapping them.
      return true;
    }
  }
  return false;
}

int SparseKvMapper::steps() const { return static_cast<int>(impl_->steps.size()); }

size_t SparseKvMapper::mapped_bytes() const { return impl_->mapped_bytes; }

}  // namespace litert::lm
