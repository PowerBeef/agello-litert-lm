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

#include <memory>
#include <optional>
#include <vector>

#include "runtime/util/metal_raw_buffer.h"

namespace litert::lm {

std::optional<MetalRawBuffer> GetMetalRawBuffer(void* mtl_buffer) { return std::nullopt; }

void* AllocateSnapshotMemory(size_t length) { return nullptr; }

void FreeSnapshotMemory(void* addr) {}

bool BlitMetalRanges(const void* command_queue, void* mtl_buffer, void* host, size_t host_length,
                     const std::vector<MetalBlitRange>& ranges, bool to_host) {
  return false;
}

size_t MetalBufferLength(void* mtl_buffer) { return 0; }

void* MetalDeviceOfBuffer(void* mtl_buffer) { return nullptr; }
bool PlacementSparseBuffersSupported(void* mtl_device) { return false; }
size_t SparseKvTileBytes() { return 16 * 1024; }
void* CreateSparseMetalBuffer(void* mtl_device, size_t length) { return nullptr; }
void ReleaseMetalObject(void* object) {}

struct SparseKvMapper::Impl {};
SparseKvMapper::SparseKvMapper(void* mtl_device, const void* command_queue) {}
SparseKvMapper::~SparseKvMapper() = default;
bool SparseKvMapper::ok() const { return false; }
bool SparseKvMapper::MapStep(const std::vector<SparseRange>& ranges) { return false; }
bool SparseKvMapper::UnmapLastStep() { return false; }
int SparseKvMapper::steps() const { return 0; }
size_t SparseKvMapper::mapped_bytes() const { return 0; }

}  // namespace litert::lm
