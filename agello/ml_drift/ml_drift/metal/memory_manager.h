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

#ifndef ML_DRIFT_METAL_MEMORY_MANAGER_H_
#define ML_DRIFT_METAL_MEMORY_MANAGER_H_

#import <Metal/Metal.h>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/model.h"
#include "ml_drift/common/task/gpu_tensor.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/types.h"
#include "ml_drift/metal/buffer.h"
#include "ml_drift/metal/environment.h"
#include "ml_drift/metal/metal_spatial_tensor.h"

namespace ml_drift {
namespace metal {

class MemoryManager {
 public:
  using ModelId = uint32_t;
  using Key = std::pair<ModelId, ValueId>;

  absl::StatusOr<ModelId> AllocateMemory(
      const Environment& env, const GpuModel& gpu_model,
      const ExternalTensorsInfo& external_tensors,
      std::vector<std::unique_ptr<MetalSpatialTensor>>&
          external_mutable_tensors);
  MetalSpatialTensor* GetSpatialTensor(Key key);
  // Can be used only with ids from mutable_tensors in external_tensors
  // Must be called after initialization and before execution
  absl::Status SetExternalTensor(const Key& key,
                                 MetalSpatialTensor* tensor_ptr);
  uint64_t GetIntermediateTensorsSize() const;
  uint64_t GetConstantTensorsSize() const;

  // Agello: a prefill graph's own pool buffers are freed between its runs. Once a graph runs
  // with a pool buffer, all of the buffer stays resident (wired) until it's freed, and marking it
  // purgeable releases nothing: after one message of 1,024 tokens or more, the 1,024-token
  // prefill's buffers (1.3 GB at a 32K window) stayed with the chat for the engine's life. A
  // buffer only one graph references, of a graph that runs as a prefill with at least
  // kMinReleaseBytes of such buffers, is freed when another graph runs next, and allocated again
  // before the graph runs again; the tensor views on it are recreated in place and the graph's
  // kernels re-pointed (no recompilation). Each context registers how to re-point its kernels.
  using Rebind = std::function<absl::Status(const absl::flat_hash_set<ValueId>&)>;
  void SetRebind(ModelId model_id, Rebind rebind);
  void UnregisterModel(ModelId model_id);
  // Before a run of `model_id`: records whether it's a prefill and says whether another
  // graph's own buffers are due for release. The caller then makes sure no work that uses them
  // is in flight or pending, and calls ReleasePending.
  bool BeginRun(ModelId model_id, bool is_prefill);
  absl::Status ReleasePending();
  // Allocates `model_id`'s own buffers again if they were freed, and re-points its kernels.
  absl::Status EnsureBuffers(ModelId model_id);

  // Adds allocated internal constant GPU resources to the given residency set.
  // The residency_set parameter must be of type id<MTLResidencySet>.
  API_AVAILABLE(ios(18.0), macos(15.0))
  void AddConstantsToResidencySet(id<MTLResidencySet> residency_set) const;

  // Adds allocated intermediate GPU resources to the given residency set.
  // The residency_set parameter must be of type id<MTLResidencySet>.
  API_AVAILABLE(ios(18.0), macos(15.0))
  void AddIntermediatesToResidencySet(id<MTLResidencySet> residency_set) const;

  // Adds allocated external immutable GPU resources to the given residency set.
  // The residency_set parameter must be of type id<MTLResidencySet>.
  API_AVAILABLE(ios(18.0), macos(15.0))
  void AddExternalImmutableToResidencySet(
      id<MTLResidencySet> residency_set) const;

  // Adds allocated external mutable GPU resources to the given residency set.
  // The residency_set parameter must be of type id<MTLResidencySet>.
  API_AVAILABLE(ios(18.0), macos(15.0))
  void AddExternalMutableToResidencySet(
      id<MTLResidencySet> residency_set) const;

 private:
  enum class TensorMemoryType {
    kStrongShape,
    kBuffer,
    kConst,
    kExternal
  };
  absl::Status CollectUsageInformation(ModelId model_id,
                                       const GpuModel& gpu_model,
                                       const GpuInfo& gpu_info,
                                       std::map<ValueId, int2>* buffer_usages,
                                       std::map<ValueId, int2>* texture_usages);
  absl::StatusOr<std::vector<std::unique_ptr<MetalSpatialTensor>>>
  AllocateExternalTensors(ModelId model_id,
                          const ExternalTensorsInfo& external_tensors,
                          const Environment& env);
  absl::Status AllocateMemoryForConstTensors(ModelId model_id,
                                             const GpuModel& gpu_model,
                                             const Environment& env);
  absl::Status AllocateMemoryForBuffers(
      ModelId model_id, const Environment& env,
      const std::map<ValueId, int2>& buffer_usages);
  absl::Status AllocateMemoryForTextures(
      ModelId model_id, const Environment& env,
      const std::map<ValueId, int2>& texture_usages);
  TensorMemoryType GetTensorMemoryType(const GpuInfo& gpu_info, Key key);

  absl::flat_hash_map<Key, MetalSpatialTensor*, absl::Hash<Key>,
                      std::equal_to<Key>>
      external_immutable_tensors_;
  absl::flat_hash_map<Key, MetalSpatialTensor*, absl::Hash<Key>,
                      std::equal_to<Key>>
      external_mutable_tensors_;

  absl::flat_hash_map<Key, TensorDescriptor, absl::Hash<Key>,
                      std::equal_to<Key>>
      tensors_descs_;
  std::map<Key, MetalSpatialTensor> const_tensors_;

  std::vector<std::unique_ptr<Buffer>> shared_buffers_;
  absl::flat_hash_map<Key, MetalSpatialTensor, absl::Hash<Key>,
                      std::equal_to<Key>>
      value_id_to_buffer_;

  std::vector<std::unique_ptr<MetalSpatialTensor>> shared_texture_tensors_;
  absl::flat_hash_map<Key, MetalSpatialTensor, absl::Hash<Key>,
                      std::equal_to<Key>>
      value_id_to_texture_;
  ModelId next_model_id_ = 0;

  // Agello: what freeing and reallocating a graph's own buffers needs.
  struct PoolView {
    Key key;
    const Buffer* buffer;
    uint64_t offset;
    // A 2D texture view on the buffer, with its row alignment, or a buffer view.
    bool image2d = false;
    size_t row_alignment = 0;
  };
  std::vector<const Buffer*> OwnBuffers(ModelId model_id) const;
  // The pool's buffers `model_id` may take: none for a private model, the shared ones otherwise.
  std::vector<const Buffer*> AvailableBuffers(ModelId model_id) const;
  id<MTLDevice> device_ = nil;
  std::vector<PoolView> pool_views_;
  absl::flat_hash_map<const Buffer*, absl::flat_hash_set<ModelId>> buffer_models_;
  absl::flat_hash_set<ModelId> prefill_models_;
  // Agello (the window's step 2): graphs with a pool of their own, the large prefill's (inputs of
  // kPrivatePoolRows positions or more), and the buffers only they use. They take no buffer from
  // the shared pool and lend none to it, so every buffer they touch is their own and is freed
  // after a long prompt (BeginRun); the everyday graphs never hold its pages.
  absl::flat_hash_set<ModelId> private_models_;
  // AGELLO_LOG_POOL: each value's producing node, for the log of a large buffer's tenants.
  absl::flat_hash_map<ValueId, std::string> producer_names_;
  absl::flat_hash_set<const Buffer*> private_buffers_;
  absl::flat_hash_set<ModelId> released_;
  absl::flat_hash_map<ModelId, Rebind> rebind_;
  std::optional<ModelId> last_run_model_;
  std::optional<ModelId> pending_release_;
};

}  // namespace metal
}  // namespace ml_drift

#endif  // ML_DRIFT_METAL_MEMORY_MANAGER_H_
