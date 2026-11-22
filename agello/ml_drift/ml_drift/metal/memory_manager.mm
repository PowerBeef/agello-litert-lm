// Copyright 2026 The ML Drift Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ml_drift/metal/memory_manager.h"

#include <mach/mach.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <map>
#include <numeric>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/memory_management.h"
#include "ml_drift/common/memory_management/types.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"
#include "ml_drift/metal/common.h"
#include "ml_drift/metal/environment.h"
#include "ml_drift/metal/metal_spatial_tensor.h"

namespace ml_drift {
namespace metal {
namespace {

// A heuristic to avoid offset assignment of buffers for better memory footprint. Offset assignment
// prevents memory compression when buffers are not used or filled with zeros.
constexpr uint64_t kMaxTotalOffsetSizeToEnableOffsetAssignment = 128L * 1024 * 1024;  // 128 MB

// returns true if actual memory for this storage type is buffer
bool IsBufferBased(const GpuInfo& gpu_info, const TensorStorageType& type) {
  const bool family_apple1 =
      gpu_info.IsApple() && gpu_info.apple_info.IsFamilyApple1();
  if (!family_apple1 &&
      (type == TensorStorageType::kTexture2D || type == TensorStorageType::kSingleTexture2D)) {
    return true;
  }
  return type == TensorStorageType::kBuffer || type == TensorStorageType::kImageBuffer;
}

void AddUsage(ValueId id, int task_index,
              std::map<ValueId, int2>* usage_records) {
  auto it = usage_records->find(id);
  if (it == usage_records->end()) {
    // initializing start index(.x) and end index(.y)
    (*usage_records)[id].x = task_index;
    (*usage_records)[id].y = task_index;
  } else {
    // updating end index(.y)
    (*usage_records)[id].y = task_index;
  }
}

// Calculates the total size of the assignment.
size_t TotalSize(const ObjectsAssignment<size_t>& assignment,
                 size_t alignment = 1) {
  size_t total_size = 0;
  for (auto object_size : assignment.object_sizes) {
    total_size += AlignByN(object_size, alignment);
  }
  return total_size;
}

// Returns a vector of T* sorted in increasing memory size.
template <typename T>
std::vector<const T*> GetSortedVector(const std::vector<std::unique_ptr<T>>& items) {
  std::vector<const T*> sorted_items;
  sorted_items.reserve(items.size());
  for (const auto& item : items) {
    sorted_items.push_back(item.get());
  }
  absl::c_sort(sorted_items, [](const T* i1, const T* i2) {
    return i1->GetMemorySizeInBytes() < i2->GetMemorySizeInBytes();
  });
  return sorted_items;
}

// Takes the smallest available buffer which can hold `size`. Agello: not one more than twice
// `size` and 16 MB over it (a new buffer is made instead). The pool is shared by every graph of
// a model, and once a graph runs with a buffer, all of that buffer stays resident (wired) until
// it's freed: verify's and the 128-token prefill's small tensors used to borrow the 1,024-token
// prefill's 524 MB score buffers, so a 32K window held 1.3 GB more on every chat, whether or not
// a prompt ever needed the 1,024-token prefill.
const Buffer* TakeAvailableBuffer(std::vector<const Buffer*>& available_buffers, size_t size) {
  auto it = absl::c_find_if(available_buffers,
                            [size](const Buffer* b) { return b->GetMemorySizeInBytes() >= size; });
  if (it == available_buffers.end()) {
    return nullptr;
  }
  // Allow decode/verify to share verify_8's ~40-50 MB buffer while still blocking
  // multi-hundred-MB prefill score buffers if private pools are disabled.
  constexpr uint64_t kMaxWasteBytes = 64ull << 20;
  const uint64_t found = (*it)->GetMemorySizeInBytes();
  if (found > 2 * static_cast<uint64_t>(size) && found - size > kMaxWasteBytes) {
    return nullptr;
  }

  const Buffer* buffer = *it;
  available_buffers.erase(it);
  return buffer;
}

// Takes the first available texture which has the same descriptor as `td`.
const MetalSpatialTensor* TakeAvailableTexture(
    std::vector<const MetalSpatialTensor*>& available_tensors, const TensorDescriptor& td) {
  auto it = absl::c_find_if(available_tensors, [td](const MetalSpatialTensor* t) {
    if (td != t->GetDescriptor()) {
      return false;
    }

    BHWDC s1 = td.GetBHWDCShape();
    BHWDC s2 = t->GetDescriptor().GetBHWDCShape();
    return s1.b == s2.b && s1.h == s2.h && s1.w == s2.w && s1.d == s2.d && s1.c == s2.c;
  });
  if (it == available_tensors.end()) {
    return nullptr;
  }

  const MetalSpatialTensor* tensor = *it;
  available_tensors.erase(it);
  return tensor;
}

uint64_t GetUniqueKey(const TensorDescriptor& tensor_desc) {
  return absl::HashOf(ToStringWithShape(tensor_desc));
}

}  // namespace

absl::StatusOr<MemoryManager::ModelId> MemoryManager::AllocateMemory(
    const Environment& env, const GpuModel& gpu_model, const ExternalTensorsInfo& external_tensors,
    std::vector<std::unique_ptr<MetalSpatialTensor>>& external_mutable_tensors) {
  ModelId model_id = next_model_id_++;
  device_ = env.device();
  for (const auto& [id, desc] : gpu_model.tensors) {
    tensors_descs_[Key(model_id, id)] = desc;
  }
  // Agello: the large prefill gets a pool of its own (private_models_). A graph's positions per
  // run are its positions input's batch ([positions, 1, 1, 1, 1]); the caches and masks have a
  // batch of 1. Every prefill graph (64 positions or more per run) gets one, released when another
  // graph runs: the 128-token prefill's window-sized buffers (the global cache's converted copies,
  // its scores) then live only while a prompt is read (sparse KV, step 4: a short chat at 64K
  // 1,132 -> 857 MB on the M6); decode, verify and the drafter, a few positions each, share one.
  // AGELLO_PRIVATE_POOLS=none: only the 1,024-token prefill (window step 2); =all: every graph.
  {
    constexpr int kPrivatePoolRows = 512;
    constexpr int kPrefillRows = 64;
    static const bool shared = std::getenv("AGELLO_SHARED_POOL") != nullptr;
    int rows = 0;
    for (const auto& [id, _] : gpu_model.input_ids_and_refs) {
      auto it = gpu_model.tensors.find(id);
      if (it == gpu_model.tensors.end()) continue;
      rows = std::max(rows, it->second.GetBHWDCShape().b);
    }
    static const std::string mode = [] {
      const char* value = std::getenv("AGELLO_PRIVATE_POOLS");
      return std::string(value != nullptr && value[0] != '\0' ? value : "prefill");
    }();
    const int threshold = mode == "none" ? kPrivatePoolRows : kPrefillRows;
    if (!shared && (mode == "all" || rows >= threshold)) private_models_.insert(model_id);
    if (std::getenv("AGELLO_LOG_POOL") != nullptr) {
      fprintf(stderr, "Agello pool: model %u, %d positions per run%s\n", model_id, rows,
              private_models_.contains(model_id) ? ", a pool of its own" : "");
    }
  }
  ABSL_ASSIGN_OR_RETURN(external_mutable_tensors,
                        AllocateExternalTensors(model_id, external_tensors, env));
  ABSL_RETURN_IF_ERROR(AllocateMemoryForConstTensors(model_id, gpu_model, env));
  std::map<ValueId, int2> buffer_usages;
  std::map<ValueId, int2> texture_usages;
  ABSL_RETURN_IF_ERROR(
      CollectUsageInformation(model_id, gpu_model, env.GetInfo(), &buffer_usages, &texture_usages));
  producer_names_.clear();
  if (std::getenv("AGELLO_LOG_POOL") != nullptr) {
    for (const auto& node : gpu_model.nodes) {
      for (ValueId id : node.outputs) producer_names_[id] = node.name;
    }
  }
  ABSL_RETURN_IF_ERROR(AllocateMemoryForBuffers(model_id, env, buffer_usages));
  ABSL_RETURN_IF_ERROR(AllocateMemoryForTextures(model_id, env, texture_usages));
  return model_id;
}

namespace {
constexpr uint64_t kMinReleaseBytes = 256ull << 20;

int64_t FootprintMegabytes() {
  task_vm_info_data_t info;
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) !=
      KERN_SUCCESS) {
    return -1;
  }
  return static_cast<int64_t>(info.phys_footprint >> 20);
}

bool LogPool() {
  static const bool log = std::getenv("AGELLO_LOG_POOL") != nullptr;
  return log;
}
}  // namespace

void MemoryManager::SetRebind(ModelId model_id, Rebind rebind) {
  if (rebind) {
    rebind_[model_id] = std::move(rebind);
  } else {
    rebind_.erase(model_id);
  }
}

std::vector<const Buffer*> MemoryManager::OwnBuffers(ModelId model_id) const {
  std::vector<const Buffer*> own;
  for (const auto& [buffer, models] : buffer_models_) {
    if (models.size() == 1 && models.contains(model_id)) {
      own.push_back(buffer);
    }
  }
  return own;
}

bool MemoryManager::BeginRun(ModelId model_id, bool is_prefill) {
  if (is_prefill) prefill_models_.insert(model_id);
  bool due = false;
  if (last_run_model_.has_value() && *last_run_model_ != model_id) {
    const ModelId last = *last_run_model_;
    if (prefill_models_.contains(last) && !released_.contains(last) && rebind_.contains(last)) {
      uint64_t bytes = 0;
      for (const Buffer* buffer : OwnBuffers(last)) bytes += buffer->GetMemorySizeInBytes();
      if (bytes >= kMinReleaseBytes) {
        pending_release_ = last;
        due = true;
      }
    }
  }
  last_run_model_ = model_id;
  return due;
}

absl::Status MemoryManager::ReleasePending() {
  if (!pending_release_.has_value()) return absl::OkStatus();
  const ModelId model_id = *pending_release_;
  pending_release_.reset();
  const int64_t before = LogPool() ? FootprintMegabytes() : 0;
  const std::vector<const Buffer*> own = OwnBuffers(model_id);
  const absl::flat_hash_set<const Buffer*> own_set(own.begin(), own.end());
  absl::flat_hash_set<ValueId> ids;
  @autoreleasepool {
    for (const PoolView& view : pool_views_) {
      if (view.key.first != model_id || !own_set.contains(view.buffer)) continue;
      // A view on no buffer keeps its descriptor; a 2D texture view's texture goes with it.
      ABSL_RETURN_IF_ERROR(CreateTensorSharedBuffer(nil, tensors_descs_[view.key],
                                                    &value_id_to_buffer_[view.key], view.offset));
      ids.insert(view.key.second);
    }
    // The kernels drop their references to the buffers, then the pool does.
    ABSL_RETURN_IF_ERROR(rebind_[model_id](ids));
    uint64_t bytes = 0;
    for (const Buffer* buffer : own) {
      bytes += buffer->GetMemorySizeInBytes();
      if (LogPool() && buffer->GetMemorySizeInBytes() >= (64ull << 20)) {
        // The pool's own reference is one: more means a kernel or a texture still holds it.
        const long retains = CFGetRetainCount((__bridge CFTypeRef)buffer->GetMemoryPtr());
        int views = 0, image2d = 0;
        std::string ids_text;
        for (const PoolView& view : pool_views_) {
          if (view.key.first != model_id || view.buffer != buffer) continue;
          ++views;
          image2d += view.image2d;
          if (ids_text.size() < 120) absl::StrAppend(&ids_text, " ", view.key.second, view.image2d ? "t" : "");
        }
        fprintf(stderr, "Agello pool: freeing %llu MB, retain count %ld, %d views (%d 2D textures):%s\n",
                static_cast<unsigned long long>(buffer->GetMemorySizeInBytes() >> 20), retains, views,
                image2d, ids_text.c_str());
      }
      // A few buffers stay referenced from outside the pool after the rebind (a retain count of 3,
      // not 2: both 256 MB score buffers of the 1,024-token prefill at 64K and one more), so the
      // object outlives the pool's reference. Nothing runs on them any more (the queue drained),
      // so their pages are discarded first: an empty buffer holds no memory whoever keeps it.
      [buffer->GetMemoryPtr() setPurgeableState:MTLPurgeableStateEmpty];
      *const_cast<Buffer*>(buffer) = Buffer(nil, buffer->GetMemorySizeInBytes());
    }
    if (LogPool()) {
      fprintf(stderr, "Agello pool: freed model %u's %zu own buffers (%llu MB, %zu tensors);"
              " footprint %lld -> %lld MB\n", model_id, own.size(),
              static_cast<unsigned long long>(bytes >> 20), ids.size(),
              static_cast<long long>(before), static_cast<long long>(FootprintMegabytes()));
    }
  }
  released_.insert(model_id);
  return absl::OkStatus();
}

absl::Status MemoryManager::EnsureBuffers(ModelId model_id) {
  if (!released_.contains(model_id)) return absl::OkStatus();
  released_.erase(model_id);
  const std::vector<const Buffer*> own = OwnBuffers(model_id);
  const absl::flat_hash_set<const Buffer*> own_set(own.begin(), own.end());
  absl::flat_hash_set<ValueId> ids;
  @autoreleasepool {
    for (const Buffer* buffer : own) {
      ABSL_RETURN_IF_ERROR(CreateBuffer(buffer->GetMemorySizeInBytes(), /*data=*/nullptr, device_,
                                        const_cast<Buffer*>(buffer)));
    }
    for (const PoolView& view : pool_views_) {
      if (view.key.first != model_id || !own_set.contains(view.buffer)) continue;
      if (view.image2d) {
        ABSL_RETURN_IF_ERROR(CreateTensorSharedImage2DBuffer(
            view.buffer->GetMemoryPtr(), tensors_descs_[view.key], view.row_alignment,
            &value_id_to_buffer_[view.key], view.offset));
      } else {
        ABSL_RETURN_IF_ERROR(CreateTensorSharedBuffer(view.buffer->GetMemoryPtr(),
                                                      tensors_descs_[view.key],
                                                      &value_id_to_buffer_[view.key], view.offset));
      }
      ids.insert(view.key.second);
    }
  }
  if (LogPool()) {
    fprintf(stderr, "Agello pool: allocated model %u's %zu own buffers again (%zu tensors)\n",
            model_id, own.size(), ids.size());
  }
  return rebind_[model_id](ids);
}

MetalSpatialTensor* MemoryManager::GetSpatialTensor(Key key) {
  if (external_immutable_tensors_.find(key) != external_immutable_tensors_.end()) {
    return external_immutable_tensors_[key];
  } else if (external_mutable_tensors_.find(key) != external_mutable_tensors_.end()) {
    return external_mutable_tensors_[key];
  } else if (const_tensors_.find(key) != const_tensors_.end()) {
    return &const_tensors_[key];
  } else if (value_id_to_buffer_.find(key) != value_id_to_buffer_.end()) {
    return &value_id_to_buffer_[key];
  } else if (value_id_to_texture_.find(key) != value_id_to_texture_.end()) {
    return &value_id_to_texture_[key];
  }
  return nullptr;
}

MemoryManager::TensorMemoryType MemoryManager::GetTensorMemoryType(const GpuInfo& gpu_info,
                                                                   Key key) {
  if (external_immutable_tensors_.find(key) != external_immutable_tensors_.end()) {
    return TensorMemoryType::kExternal;
  } else if (external_mutable_tensors_.find(key) != external_mutable_tensors_.end()) {
    return TensorMemoryType::kExternal;
  } else if (const_tensors_.find(key) != const_tensors_.end()) {
    return TensorMemoryType::kConst;
  } else if (IsBufferBased(gpu_info, tensors_descs_[key].GetStorageType())) {
    return TensorMemoryType::kBuffer;
  } else {
    return TensorMemoryType::kStrongShape;
  }
}

absl::Status MemoryManager::CollectUsageInformation(ModelId model_id, const GpuModel& gpu_model,
                                                    const GpuInfo& gpu_info,
                                                    std::map<ValueId, int2>* buffer_usages,
                                                    std::map<ValueId, int2>* texture_usages) {
  for (auto input : gpu_model.input_ids_and_refs) {
    ValueId id = input.first;
    const auto& memory_type = GetTensorMemoryType(gpu_info, Key(model_id, id));
    if (memory_type == TensorMemoryType::kBuffer) {
      AddUsage(id, 0, buffer_usages);
    } else if (memory_type == TensorMemoryType::kStrongShape) {
      AddUsage(id, 0, texture_usages);
    }
  }

  for (int node_index = 0; node_index < gpu_model.nodes.size(); ++node_index) {
    const auto& node = gpu_model.nodes[node_index];
    auto ids = node.inputs;
    ids.insert(ids.end(), node.outputs.begin(), node.outputs.end());
    for (ValueId id : ids) {
      const auto& memory_type = GetTensorMemoryType(gpu_info, Key(model_id, id));
      if (memory_type == TensorMemoryType::kBuffer) {
        AddUsage(id, node_index, buffer_usages);
      } else if (memory_type == TensorMemoryType::kStrongShape) {
        AddUsage(id, node_index, texture_usages);
      }
    }
  }

  for (auto output : gpu_model.output_ids_and_refs) {
    ValueId id = output.first;
    const auto& memory_type = GetTensorMemoryType(gpu_info, Key(model_id, id));
    if (memory_type == TensorMemoryType::kBuffer) {
      AddUsage(id, gpu_model.nodes.size(), buffer_usages);
    } else if (memory_type == TensorMemoryType::kStrongShape) {
      AddUsage(id, gpu_model.nodes.size(), texture_usages);
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::unique_ptr<MetalSpatialTensor>>>
MemoryManager::AllocateExternalTensors(ModelId model_id,
                                       const ExternalTensorsInfo& external_tensors,
                                       const Environment& env) {
  for (const auto& [id, tensor] : external_tensors.immutable_tensors) {
    auto* metal_spatial_tensor = dynamic_cast<MetalSpatialTensor*>(tensor);
    if (!metal_spatial_tensor) {
      return absl::InvalidArgumentError("Expected MetalSpatialTensor.");
    }
    external_immutable_tensors_[Key(model_id, id)] = metal_spatial_tensor;
  }

  std::vector<std::unique_ptr<MetalSpatialTensor>> temp_tensors;
  absl::flat_hash_map<uint64_t, MetalSpatialTensor*> mutable_tensors;
  for (const auto& [id, _] : external_tensors.mutable_tensors) {
    auto key = Key(model_id, id);
    auto& tensor_desc = tensors_descs_[key];
    auto tensor_desc_unique_key = GetUniqueKey(tensor_desc);
    if (mutable_tensors.contains(tensor_desc_unique_key)) {
      external_mutable_tensors_[key] = mutable_tensors[tensor_desc_unique_key];
      continue;
    }
    temp_tensors.push_back(std::make_unique<MetalSpatialTensor>());
    auto* metal_spatial_tensor = temp_tensors.back().get();
    ABSL_RETURN_IF_ERROR(CreateTensor(env.device(), tensors_descs_[key], metal_spatial_tensor));
    mutable_tensors[tensor_desc_unique_key] = metal_spatial_tensor;
    // It will be reset to nullptr by the caller later.
    external_mutable_tensors_[key] = metal_spatial_tensor;
  }
  return std::move(temp_tensors);
}

absl::Status MemoryManager::AllocateMemoryForConstTensors(ModelId model_id,
                                                          const GpuModel& gpu_model,
                                                          const Environment& env) {
  for (auto& description : gpu_model.const_tensors) {
    ABSL_RETURN_IF_ERROR(const_tensors_[Key(model_id, description.first)].CreateFromDescriptor(
        description.second, env.device()));
  }
  return absl::OkStatus();
}

std::vector<const Buffer*> MemoryManager::AvailableBuffers(ModelId model_id) const {
  if (private_models_.contains(model_id)) return {};
  std::vector<const Buffer*> available = GetSortedVector(shared_buffers_);
  std::erase_if(available, [this](const Buffer* b) { return private_buffers_.contains(b); });
  return available;
}

absl::Status MemoryManager::AllocateMemoryForBuffers(ModelId model_id, const Environment& env,
                                                     const std::map<ValueId, int2>& buffer_usages) {
  // From Apple documentation:
  // For buffers in the device address space, align the offset to the data type
  // consumed by the compute function (which is always less than or equal to 16
  // bytes).
  // For buffers in the constant address space, align the offset to 256
  // bytes in macOS. In iOS, align the offset to the maximum of either the data
  // type consumed by the compute function, or 4 bytes. A 16-byte alignment is
  // safe in iOS if you don't need to consider the data type.
  if (buffer_usages.empty()){
    return absl::OkStatus();
  }
#if defined(TARGET_IOS) || defined(TARGET_TVOS)
  const size_t kConstAlignment = 16;
#elif defined(TARGET_MACOS)
  const size_t kConstAlignment = 256;
#else
  const size_t kConstAlignment = 256;
#endif
  size_t min_common_alignment = kConstAlignment;
  std::vector<TensorUsageRecord<size_t>> buffer_usage_records;
  std::map<ValueId, int> value_id_to_shared_buffer_tensors;
  for (auto& usage : buffer_usages) {
    const auto& td = tensors_descs_[Key(model_id, usage.first)];
    const auto& shape = td.GetBHWDCShape();
    const auto& storage_dims = td.GetStorageDims();
    const size_t element_size = SizeOf(td.GetDataType());
    size_t buffer_size;
    if (td.GetStorageType() == TensorStorageType::kTexture2D ||
        td.GetStorageType() == TensorStorageType::kSingleTexture2D) {
      const size_t row_bytes_alignment = [env.device()
          minimumLinearTextureAlignmentForPixelFormat:DataTypeToPixelFormat(td.GetDataType(),
                                                                            td.GetElementSize(),
                                                                            false)];
      min_common_alignment =
          std::lcm(min_common_alignment, row_bytes_alignment);
      const int vec_size = td.GetStorageType() == TensorStorageType::kTexture2D ? 4 : shape.c;
      const size_t bytes_per_row = storage_dims[0] * vec_size * element_size;
      buffer_size =
          AlignByN(bytes_per_row, row_bytes_alignment) * storage_dims[1];
    } else {
      buffer_size = storage_dims[0] * 4 * element_size;
    }
    value_id_to_shared_buffer_tensors[usage.first] =
        buffer_usage_records.size();
    buffer_usage_records.push_back({buffer_size,
                                    static_cast<TaskId>(usage.second.x),
                                    static_cast<TaskId>(usage.second.y)});
  }

  ObjectsAssignment<size_t> buffer_assignment;
  ABSL_RETURN_IF_ERROR(AssignObjectsToTensors(buffer_usage_records, MemoryStrategy::kGreedyBest,
                                              &buffer_assignment));

  OffsetsAssignment offset_assignment;
  ABSL_RETURN_IF_ERROR(AssignOffsetsToTensors(buffer_usage_records, MemoryStrategy::kGreedyBySize,
                                              &offset_assignment, min_common_alignment));

  bool use_offset_assignment = false;
  if (offset_assignment.total_size <= TotalSize(buffer_assignment) &&
      offset_assignment.total_size <= kMaxTotalOffsetSizeToEnableOffsetAssignment &&
      offset_assignment.total_size <= env.GetInfo().GetMaxBufferSize()) {
    use_offset_assignment = true;
  }

  std::vector<const Buffer*> mtl_buffers;
  std::vector<std::unique_ptr<Buffer>> new_buffers;

  if (use_offset_assignment) {
    size_t required_size = offset_assignment.total_size;
    std::vector<const Buffer*> available_buffers = AvailableBuffers(model_id);
    const Buffer* selected_buffer = TakeAvailableBuffer(available_buffers, required_size);
    if (!selected_buffer) {
      new_buffers.push_back(std::make_unique<Buffer>());
      ABSL_RETURN_IF_ERROR(
          CreateBuffer(required_size, /*data=*/nullptr, env.device(), new_buffers.back().get()));
      selected_buffer = new_buffers.back().get();
    }
    mtl_buffers.push_back(selected_buffer);
  } else {
    std::vector<const Buffer*> available_buffers = AvailableBuffers(model_id);
    mtl_buffers.resize(buffer_assignment.object_sizes.size());
    for (int i = 0; i < buffer_assignment.object_sizes.size(); ++i) {
      size_t size = buffer_assignment.object_sizes[i];
      const Buffer* selected_buffer = TakeAvailableBuffer(available_buffers, size);
      if (!selected_buffer) {
        new_buffers.push_back(std::make_unique<Buffer>());
        ABSL_RETURN_IF_ERROR(
            CreateBuffer(size, /*data=*/nullptr, env.device(), new_buffers.back().get()));
        selected_buffer = new_buffers.back().get();
      }
      mtl_buffers[i] = selected_buffer;
    }
  }

  if (LogPool()) {
    uint64_t reused = 0, created = 0;
    std::string owners;
    for (const Buffer* buffer : mtl_buffers) {
      const bool is_new = absl::c_any_of(new_buffers, [buffer](const auto& b) { return b.get() == buffer; });
      (is_new ? created : reused) += buffer->GetMemorySizeInBytes();
      if (!is_new && buffer_models_.contains(buffer)) {
        absl::StrAppend(&owners, " ", buffer->GetMemorySizeInBytes() >> 20, "MB<-");
        for (ModelId m : buffer_models_[buffer]) absl::StrAppend(&owners, m, ".");
      }
    }
    std::string big_new;
    for (const auto& b : new_buffers) {
      if (b->GetMemorySizeInBytes() >= (64ull << 20)) {
        id<MTLBuffer> handle = b->GetMemoryPtr();
        absl::StrAppend(&big_new, " ", b->GetMemorySizeInBytes() >> 20, "MB@",
                        absl::Hex(reinterpret_cast<uintptr_t>(handle != nil ? [handle contents] : nullptr)));
      }
    }
    fprintf(stderr, "Agello pool: model %u new buffers of 64 MB or more:%s\n", model_id,
            big_new.c_str());
    fprintf(stderr, "Agello pool: model %u allocates %zu tensors in %zu buffers (%s): %llu MB new, "
            "%llu MB reused%s\n", model_id, buffer_usages.size(), mtl_buffers.size(),
            use_offset_assignment ? "offsets" : "objects",
            static_cast<unsigned long long>(created >> 20),
            static_cast<unsigned long long>(reused >> 20), owners.c_str());
  }
  if (LogPool()) {
    // Tenants: the largest tensors each big buffer holds, with their producers.
    absl::flat_hash_map<const Buffer*, std::vector<std::pair<uint64_t, std::string>>> tenants;
    for (auto& usage : buffer_usages) {
      const int index = value_id_to_shared_buffer_tensors[usage.first];
      const Buffer* b = use_offset_assignment ? mtl_buffers[0]
                                              : mtl_buffers[buffer_assignment.object_ids[index]];
      if (b->GetMemorySizeInBytes() < (64ull << 20)) continue;
      const auto& td = tensors_descs_[Key(model_id, usage.first)];
      const BHWDC s = td.GetBHWDCShape();
      auto it = producer_names_.find(usage.first);
      tenants[b].push_back({buffer_usage_records[index].tensor_size,
                            absl::StrCat(usage.first, " ", it != producer_names_.end() ? it->second : "input",
                                         " [", s.b, ",", s.h, ",", s.w, ",", s.d, ",", s.c, "] ",
                                         ToString(td.GetDataType()), " ",
                                         buffer_usage_records[index].tensor_size >> 20, "MB")});
    }
    for (auto& [b, list] : tenants) {
      absl::c_sort(list, [](const auto& x, const auto& y) { return x.first > y.first; });
      std::string text;
      for (size_t i = 0; i < list.size() && i < 4; ++i) absl::StrAppend(&text, " | ", list[i].second);
      fprintf(stderr, "Agello pool: model %u buffer %llu MB, %zu tensors:%s\n", model_id,
              static_cast<unsigned long long>(b->GetMemorySizeInBytes() >> 20), list.size(),
              text.c_str());
    }
  }
  for (auto& buf : new_buffers) {
    if (private_models_.contains(model_id)) private_buffers_.insert(buf.get());
    shared_buffers_.push_back(std::move(buf));
  }

  for (auto& usage : buffer_usages) {
    const auto& td = tensors_descs_[Key(model_id, usage.first)];
    const int buffer_index =
        buffer_assignment
            .object_ids[value_id_to_shared_buffer_tensors[usage.first]];
    auto& tensor = value_id_to_buffer_[Key(model_id, usage.first)];
    uint64_t base_buffer_offset = 0;
    const Buffer* selected_buffer = nullptr;
    if (use_offset_assignment) {
      selected_buffer = mtl_buffers[0];
      base_buffer_offset =
          offset_assignment
              .offsets[value_id_to_shared_buffer_tensors[usage.first]];
    } else {
      selected_buffer = mtl_buffers[buffer_index];
      base_buffer_offset = 0;
    }
    buffer_models_[selected_buffer].insert(model_id);
    if (td.GetStorageType() == TensorStorageType::kTexture2D ||
        td.GetStorageType() == TensorStorageType::kSingleTexture2D) {
      size_t row_bytes_alignment = [env.device()
          minimumLinearTextureAlignmentForPixelFormat:DataTypeToPixelFormat(td.GetDataType(),
                                                                            td.GetElementSize(),
                                                                            false)];
      @autoreleasepool {
        ABSL_RETURN_IF_ERROR(CreateTensorSharedImage2DBuffer(
            selected_buffer->GetMemoryPtr(), td, row_bytes_alignment, &tensor, base_buffer_offset));
      }
      pool_views_.push_back({Key(model_id, usage.first), selected_buffer, base_buffer_offset,
                             /*image2d=*/true, row_bytes_alignment});
    } else {
      @autoreleasepool {
        ABSL_RETURN_IF_ERROR(CreateTensorSharedBuffer(selected_buffer->GetMemoryPtr(), td, &tensor,
                                                      base_buffer_offset));
      }
      pool_views_.push_back({Key(model_id, usage.first), selected_buffer, base_buffer_offset});
    }
  }
  return absl::OkStatus();
}

absl::Status MemoryManager::AllocateMemoryForTextures(
    ModelId model_id, const Environment& env, const std::map<ValueId, int2>& texture_usages) {
  if (texture_usages.empty()) {
    return absl::OkStatus();
  }
  struct TensorDescComparator {
    TensorDescriptor tensor_desc;

    bool operator==(const TensorDescComparator& t) const {
      return tensor_desc == t.tensor_desc &&
             tensor_desc.GetBHWDCShape() == t.tensor_desc.GetBHWDCShape();
    }
  };

  std::vector<TensorUsageRecord<TensorDescComparator>> usage_records;
  std::map<ValueId, ValueId> remap_from_value_ids;
  for (auto& usage : texture_usages) {
    remap_from_value_ids[usage.first] = usage_records.size();
    usage_records.push_back({{tensors_descs_[Key(model_id, usage.first)]},
                             static_cast<TaskId>(usage.second.x),
                             static_cast<TaskId>(usage.second.y)});
  }

  ObjectsAssignment<TensorDescComparator> assignment;
  ABSL_RETURN_IF_ERROR(
      AssignObjectsToTensors(usage_records, MemoryStrategy::kEquality, &assignment));

  std::vector<const MetalSpatialTensor*> mtl_textures(assignment.object_sizes.size(), nullptr);
  std::vector<std::unique_ptr<MetalSpatialTensor>> new_textures;

  std::vector<const MetalSpatialTensor*> available_textures =
      GetSortedVector(shared_texture_tensors_);

  for (int i = 0; i < assignment.object_sizes.size(); ++i) {
    const auto& td = assignment.object_sizes[i].tensor_desc;
    const MetalSpatialTensor* selected_tensor = TakeAvailableTexture(available_textures, td);
    if (selected_tensor) {
      mtl_textures[i] = selected_tensor;
    } else {
      new_textures.push_back(std::make_unique<MetalSpatialTensor>());
      ABSL_RETURN_IF_ERROR(CreateTensor(env.device(), td, new_textures.back().get()));
      mtl_textures[i] = new_textures.back().get();
    }
  }

  if (std::getenv("AGELLO_LOG_POOL") != nullptr) {
    uint64_t created = 0;
    std::string big;
    for (const auto& tex : new_textures) {
      created += tex->GetMemorySizeInBytes();
      if (tex->GetMemorySizeInBytes() >= (32ull << 20)) {
        absl::StrAppend(&big, " ", tex->GetMemorySizeInBytes() >> 20, "MB");
      }
    }
    fprintf(stderr, "Agello pool: model %u textures: %zu objects, %zu new (%llu MB):%s\n", model_id,
            mtl_textures.size(), new_textures.size(), static_cast<unsigned long long>(created >> 20),
            big.c_str());
  }
  for (auto& tex : new_textures) {
    shared_texture_tensors_.push_back(std::move(tex));
  }

  for (auto& usage : texture_usages) {
    const auto& td = tensors_descs_[Key(model_id, usage.first)];
    const auto id = assignment.object_ids[remap_from_value_ids[usage.first]];
    ABSL_RETURN_IF_ERROR(
        CreateTensorSharedTexture(mtl_textures[id]->GetTextureHandle(), td,
                                  &value_id_to_texture_[Key(model_id, usage.first)]));
  }
  return absl::OkStatus();
}

uint64_t MemoryManager::GetIntermediateTensorsSize() const {
  uint64_t total_memory = 0;
  for (const auto& t : shared_texture_tensors_) {
    total_memory += t->GetMemorySizeInBytes();
  }
  for (const auto& b : shared_buffers_) {
    total_memory += b->GetMemorySizeInBytes();
  }

  return total_memory;
}

uint64_t MemoryManager::GetConstantTensorsSize() const {
  uint64_t total_size = 0;
  for (const auto& t : const_tensors_) {
    total_size += t.second.GetMemorySizeInBytes();
  }
  return total_size;
}

absl::Status MemoryManager::SetExternalTensor(const Key& key, MetalSpatialTensor* tensor_ptr) {
  auto it = external_mutable_tensors_.find(key);
  if (it == external_mutable_tensors_.end()) {
    return absl::InvalidArgumentError("No external tensor with this id.");
  }
  external_mutable_tensors_[key] = tensor_ptr;
  return absl::OkStatus();
}

API_AVAILABLE(ios(18.0), macos(15.0))
void MemoryManager::AddConstantsToResidencySet(id<MTLResidencySet> residency_set) const {
  if (residency_set == nil) return;

  for (const auto& [key, tensor] : const_tensors_) {
    if (tensor.GetBufferHandle() != nil) {
      [residency_set addAllocation:tensor.GetBufferHandle()];
    }
    if (tensor.GetTextureHandle() != nil) {
      [residency_set addAllocation:tensor.GetTextureHandle()];
    }
  }
}

API_AVAILABLE(ios(18.0), macos(15.0))
void MemoryManager::AddIntermediatesToResidencySet(id<MTLResidencySet> residency_set) const {
  if (residency_set == nil) return;

  for (const auto& buffer : shared_buffers_) {
    if (buffer->GetMemoryPtr() != nil) {
      [residency_set addAllocation:buffer->GetMemoryPtr()];
    }
  }

  for (const auto& tensor : shared_texture_tensors_) {
    if (tensor->GetTextureHandle() != nil) {
      [residency_set addAllocation:tensor->GetTextureHandle()];
    }
    if (tensor->GetBufferHandle() != nil) {
      [residency_set addAllocation:tensor->GetBufferHandle()];
    }
  }
}

API_AVAILABLE(ios(18.0), macos(15.0))
void MemoryManager::AddExternalImmutableToResidencySet(id<MTLResidencySet> residency_set) const {
  if (residency_set == nil) return;

  for (const auto& [key, tensor] : external_immutable_tensors_) {
    if (tensor != nullptr) {
      if (tensor->GetBufferHandle() != nil) {
        [residency_set addAllocation:tensor->GetBufferHandle()];
      }
      if (tensor->GetTextureHandle() != nil) {
        [residency_set addAllocation:tensor->GetTextureHandle()];
      }
    }
  }
}

API_AVAILABLE(ios(18.0), macos(15.0))
void MemoryManager::AddExternalMutableToResidencySet(id<MTLResidencySet> residency_set) const {
  if (residency_set == nil) return;

  for (const auto& [key, tensor] : external_mutable_tensors_) {
    if (tensor != nullptr) {
      if (tensor->GetBufferHandle() != nil) {
        [residency_set addAllocation:tensor->GetBufferHandle()];
      }
      if (tensor->GetTextureHandle() != nil) {
        [residency_set addAllocation:tensor->GetTextureHandle()];
      }
    }
  }
}

}  // namespace metal
}  // namespace ml_drift
