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

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <mutex>
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
  std::shared_ptr<std::mutex> queue_mutex = std::make_shared<std::mutex>();
  id<MTLSharedEvent> event = nil;
  uint64_t event_value = 0;
  std::atomic<bool> poisoned{false};
  struct Step {
    id<MTLHeap> heap = nil;
    std::vector<SparseRange> ranges;  // the tiles this step mapped
    size_t bytes = 0;
  };
  struct QuarantinedWork {
    uint64_t signal_value = 0;
    std::vector<id<MTLHeap>> heaps;
    std::vector<id<MTLBuffer>> buffers;
  };
  std::vector<Step> steps;
  std::atomic<size_t> mapped_bytes{0};
  // Each buffer's mapped tiles: a range's tiles may straddle another's (rows that don't start on
  // a tile), so a step maps only the tiles no earlier step did.
  std::vector<std::pair<void*, std::vector<bool>>> tiles;
  // In-flight heaps and shadow buffers retained after a mapping_queue timeout so
  // Metal 4 never sees a released resource while queued operations drain.
  std::vector<QuarantinedWork> quarantine;

  void QuarantineLocked(uint64_t signal_value, std::vector<id<MTLHeap>> heaps,
                        std::vector<id<MTLBuffer>> buffers) {
    if (heaps.empty() && buffers.empty()) return;
    quarantine.push_back(QuarantinedWork{signal_value, std::move(heaps), std::move(buffers)});
  }

  bool HasBuffer(void* buffer) const {
    for (const auto& [b, _] : tiles) {
      if (b == buffer) return true;
    }
    return false;
  }

  std::vector<bool>& TilesOf(void* buffer) {
    for (auto& [b, bits] : tiles) {
      if (b == buffer) return bits;
    }
    const size_t length = static_cast<size_t>(((__bridge id<MTLBuffer>)buffer).length);
    tiles.emplace_back(buffer, std::vector<bool>((length + SparseKvTileBytes() - 1) /
                                                 SparseKvTileBytes()));
    return tiles.back().second;
  }

  void RollbackTiles(const std::vector<SparseRange>& ranges) {
    const size_t tile = SparseKvTileBytes();
    for (const SparseRange& range : ranges) {
      for (auto& [b, bits] : tiles) {
        if (b != range.buffer) continue;
        const size_t first = range.offset / tile;
        const size_t last = std::min(bits.size(), (range.offset + range.length) / tile);
        for (size_t t = first; t < last; ++t) {
          bits[t] = false;
        }
        break;
      }
    }
  }

  bool WaitForCommandQueue() {
    if (command_queue == nil) return true;
    id<MTLCommandBuffer> fence = [command_queue commandBuffer];
    if (fence == nil) return false;
    [fence commit];
    [fence waitUntilCompleted];
    return fence.status == MTLCommandBufferStatusCompleted && fence.error == nil;
  }

  // Applies `mode` to the ranges, grouped by buffer, then waits for the mapping queue.
  bool Update(const std::vector<SparseRange>& ranges, id<MTLHeap> heap, bool map,
              uint64_t* out_signal = nullptr)
      API_AVAILABLE(macos(26.0), ios(26.0)) {
    std::lock_guard<std::mutex> lock(*queue_mutex);
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
    if (out_signal != nullptr) *out_signal = event_value;
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
  std::shared_ptr<std::mutex> queue_mutex;
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
          std::unique_lock<std::mutex> lock;
          if (queue_mutex) lock = std::unique_lock<std::mutex>(*queue_mutex);
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

SparseKvMapper::~SparseKvMapper() {
  if (!impl_) return;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    @autoreleasepool {
      if (impl_->device == nil || impl_->mapping_queue == nil || impl_->event == nil) return;
      if (impl_->poisoned.load(std::memory_order_acquire)) {
        // Wait for any timed-out mapping_queue operations to signal before dropping
        // quarantined heaps and shadow buffers at mapper teardown.
        uint64_t pending_signal = 0;
        {
          std::lock_guard<std::mutex> lock(*impl_->queue_mutex);
          pending_signal = impl_->event_value;
        }
        if (pending_signal > 0 && impl_->event.signaledValue < pending_signal) {
          [impl_->event waitUntilSignaledValue:pending_signal timeoutMS:10000];
        }
        std::lock_guard<std::mutex> lock(*impl_->queue_mutex);
        impl_->quarantine.clear();
        impl_->steps.clear();
        impl_->mapped_bytes.store(0, std::memory_order_release);
        return;
      }
      if (impl_->steps.empty()) return;
      impl_->WaitForCommandQueue();
      std::lock_guard<std::mutex> lock(*impl_->queue_mutex);
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
      if (queued) {
        ++impl_->event_value;
        [impl_->mapping_queue signalEvent:impl_->event value:impl_->event_value];
        [impl_->event waitUntilSignaledValue:impl_->event_value timeoutMS:10000];
      }
      impl_->quarantine.clear();
      impl_->steps.clear();
      impl_->mapped_bytes.store(0, std::memory_order_release);
    }
  }
}

bool SparseKvMapper::ok() const {
  return impl_ != nullptr && !impl_->poisoned.load(std::memory_order_acquire) &&
         impl_->device != nil && impl_->mapping_queue != nil && impl_->event != nil;
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
        impl_->RollbackTiles(fresh);
        fprintf(stderr, "Agello sparse KV: no heap of %zu bytes\n", bytes);
        return false;
      }
      heap.label = @"Agello sparse KV step";
      uint64_t signal_value = 0;
      if (!impl_->Update(fresh, heap, /*map=*/true, &signal_value)) {
        // Poison the mapper and quarantine the in-flight heap while mapping_queue
        // may still execute the queued updateBufferMappings operation.
        impl_->poisoned.store(true, std::memory_order_release);
        {
          std::lock_guard<std::mutex> lock(*impl_->queue_mutex);
          impl_->QuarantineLocked(signal_value, {heap}, {});
        }
        fprintf(stderr, "Agello sparse KV: MapStep timed out; mapper poisoned\n");
        return false;
      }
      impl_->steps.push_back(Impl::Step{heap, std::move(fresh), bytes});
      impl_->mapped_bytes.fetch_add(bytes, std::memory_order_acq_rel);
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
      if (!impl_->WaitForCommandQueue()) return false;
      Impl::Step& step = impl_->steps.back();
      if (!step.ranges.empty() && !impl_->Update(step.ranges, nil, /*map=*/false)) {
        // Keep step (and its heap) alive and leave tiles/mapped_bytes intact;
        // poison the mapper so no further mapping ops run.
        impl_->poisoned.store(true, std::memory_order_release);
        fprintf(stderr, "Agello sparse KV: UnmapLastStep timed out; mapper poisoned\n");
        return false;
      }
      const size_t tile = SparseKvTileBytes();
      for (const SparseRange& range : step.ranges) {
        std::vector<bool>& bits = impl_->TilesOf(range.buffer);
        for (size_t t = range.offset / tile; t < (range.offset + range.length) / tile; ++t) {
          bits[t] = false;
        }
      }
      impl_->mapped_bytes.fetch_sub(step.bytes, std::memory_order_acq_rel);
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
      if (!impl_->WaitForCommandQueue()) return std::nullopt;
      SparseKvMappingSnapshot snapshot;
      if (impl_->steps.empty()) return snapshot;
      auto snap_impl = std::make_unique<SparseKvMappingSnapshot::Impl>();
      snap_impl->mapping_queue = impl_->mapping_queue;
      snap_impl->queue_mutex = impl_->queue_mutex;
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
          // No mappings were copied yet; prevent ~Impl from unmapping empty shadows.
          snap_impl->buffers.clear();
          return std::nullopt;
        }
        shadow.label = @"Agello sparse KV snapshot";
        snap_impl->buffers.push_back({live_ptr, shadow, bits});
      }

      {
        std::lock_guard<std::mutex> lock(*impl_->queue_mutex);
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
          // Poison the mapper, retain all in-flight shadow buffers in quarantine
          // (while clearing snap_impl->buffers so ~Impl does not double-unmap them),
          // and keep impl_->steps, impl_->mapped_bytes, and impl_->tiles intact.
          impl_->poisoned.store(true, std::memory_order_release);
          std::vector<id<MTLBuffer>> in_flight_shadows;
          in_flight_shadows.reserve(snap_impl->buffers.size());
          for (auto& saved : snap_impl->buffers) {
            if (saved.shadow_buffer != nil) {
              in_flight_shadows.push_back(saved.shadow_buffer);
            }
          }
          snap_impl->buffers.clear();
          impl_->QuarantineLocked(impl_->event_value, {}, std::move(in_flight_shadows));
          fprintf(stderr, "Agello sparse KV: DetachMappings timed out; mapper poisoned\n");
          return std::nullopt;
        }
      }

      snap_impl->steps = std::move(impl_->steps);
      impl_->steps.clear();
      snap_impl->mapped_bytes = impl_->mapped_bytes.exchange(0, std::memory_order_acq_rel);
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
      // Validate every snapshot buffer against live registered buffers before mutating
      // or bridging any raw pointer.
      if (snapshot.impl_ != nullptr) {
        for (const auto& saved : snapshot.impl_->buffers) {
          if (!impl_->HasBuffer(saved.live_buffer)) {
            fprintf(stderr, "Agello sparse KV: snapshot buffer no longer registered\n");
            return false;
          }
        }
      }

      if (!impl_->WaitForCommandQueue()) return false;
      // Keep any currently mapped heaps (e.g. a helper's step) alive until the unmap completes.
      std::vector<Impl::Step> old_steps = std::move(impl_->steps);
      const size_t old_mapped_bytes = impl_->mapped_bytes.exchange(0, std::memory_order_acq_rel);
      impl_->steps.clear();

      bool timed_out = false;
      uint64_t timed_out_signal = 0;
      {
        std::lock_guard<std::mutex> lock(*impl_->queue_mutex);
        std::vector<MTL4CopySparseBufferMappingOperation> copy_ops;
        std::vector<MTL4UpdateSparseBufferMappingOperation> unmap_ops;
        bool queued = false;
        for (const auto& [live_ptr, bits] : impl_->tiles) {
          unmap_ops.clear();
          AppendTileRuns(bits, nullptr, &unmap_ops);
          if (!unmap_ops.empty()) {
            [impl_->mapping_queue updateBufferMappings:(__bridge id<MTLBuffer>)live_ptr
                                                  heap:nil
                                            operations:unmap_ops.data()
                                                 count:unmap_ops.size()];
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
            timed_out = true;
            timed_out_signal = impl_->event_value;
          }
        }
      }

      if (timed_out) {
        // Poison the mapper and quarantine all in-flight heaps and shadow buffers
        // from both the live steps and the snapshot until the queue drains.
        impl_->poisoned.store(true, std::memory_order_release);
        std::vector<id<MTLHeap>> in_flight_heaps;
        std::vector<id<MTLBuffer>> in_flight_shadows;
        if (snapshot.impl_ != nullptr) {
          in_flight_heaps.reserve(snapshot.impl_->steps.size());
          for (auto& step : snapshot.impl_->steps) {
            if (step.heap != nil) in_flight_heaps.push_back(step.heap);
          }
          in_flight_shadows.reserve(snapshot.impl_->buffers.size());
          for (auto& saved : snapshot.impl_->buffers) {
            if (saved.shadow_buffer != nil) {
              in_flight_shadows.push_back(saved.shadow_buffer);
            }
          }
          // Clear snapshot so ~SparseKvMappingSnapshot does not issue duplicate unmaps.
          snapshot.impl_->buffers.clear();
          snapshot.impl_->steps.clear();
          snapshot.impl_->mapped_bytes = 0;
        }
        {
          std::lock_guard<std::mutex> lock(*impl_->queue_mutex);
          impl_->QuarantineLocked(timed_out_signal, std::move(in_flight_heaps),
                                  std::move(in_flight_shadows));
        }
        // Keep old_steps, mapped_bytes, and tiles mutually consistent and retained.
        impl_->steps = std::move(old_steps);
        impl_->mapped_bytes.store(old_mapped_bytes, std::memory_order_release);
        fprintf(stderr, "Agello sparse KV: RestoreMappings timed out; mapper poisoned\n");
        return false;
      }

      for (auto& [_, bits] : impl_->tiles) {
        std::fill(bits.begin(), bits.end(), false);
      }
      if (snapshot.impl_ != nullptr) {
        for (const auto& saved : snapshot.impl_->buffers) {
          impl_->TilesOf(saved.live_buffer) = saved.tiles;
        }
        impl_->steps = std::move(snapshot.impl_->steps);
        impl_->mapped_bytes.store(snapshot.impl_->mapped_bytes, std::memory_order_release);
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

size_t SparseKvMapper::mapped_bytes() const {
  return impl_ ? impl_->mapped_bytes.load(std::memory_order_acquire) : 0;
}

}  // namespace litert::lm
