// Copyright 2026 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_STATE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_STATE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/state_interface.h"
#include "runtime/proto/executor_metadata.pb.h"

namespace litert::lm {

class SparseKvMappingSnapshot;

class LitertState : public StateInterface {
 public:
  enum class AllocationPolicy {
    // In-place update. The same buffer is used for both input and output KV
    // cache tensors.
    kInplace = 0,
    // Ping-pong update. Two sets of buffers are used and swapped between
    // inputs and outputs at each step. Required when the backend/model
    // does not support updating buffers in-place.
    kPingPong = 1,
    // GPU-optimized in-place update. Similar to kInplace, but optimized for
    // GPU backends that handle KV cache updates internally. In this mode,
    // KV cache buffers are allocated as GPU buffers (outputs) and we skip
    // passing them as inputs during execution to avoid overhead.
    kGpuOptimizedInplace = 2,
  };

  static absl::StatusOr<std::unique_ptr<LitertState>> Create(
      Environment& env, CompiledModel& compiled_model,
      absl::string_view signature_name,
      const proto::ExecutorMetadata* executor_metadata,
      AllocationPolicy allocation_policy, int batch_size,
      bool clear_kv_cache_before_prefill = true);

  int GetNumEntries() const override { return num_entries_; }

  int GetBatchSize() const override { return batch_size_; }

  absl::StatusOr<std::string> Serialize() const override {
    return absl::UnimplementedError("Not implemented");
  }

  absl::Status Load(absl::string_view serialized_state) override {
    return absl::UnimplementedError("Not implemented");
  }

  absl::Status SelectAndCopyFrom(StateInterface& other,
                                 int batch_index) override;

  absl::Status BroadcastAndCopyFrom(StateInterface& other) override;

  absl::StatusOr<std::unique_ptr<StateInterface>> DeepCopy() const override;

  // Agello (E4): a copy that holds only the first `positions` slots of each KV cache (a raw
  // snapshot of Metal buffers; any other state is copied whole). Positions at or past the
  // conversation's step are written before they're read, so the executor's snapshots for a
  // helper need no more, and their pages past the filled slots are never touched.
  absl::StatusOr<std::unique_ptr<StateInterface>> DeepCopyFilled(int positions) const;

  // Agello (P3.3): detaches placement-sparse global KV mappings via Metal 4
  // copyBufferMappingsFromBuffer (zero bytes copied, zero duplicate global KV heaps) and
  // blit-snapshots any non-sparse caches (e.g. local ring buffers). Falls back to
  // DeepCopyFilled when sparse KV is inactive.
  absl::StatusOr<std::unique_ptr<StateInterface>> DetachSnapshot(int positions);
  bool IsSparseSnapshot() const { return sparse_snapshot_; }
  absl::Status RestoreSparseSnapshot(LitertState& other);

  bool Contains(absl::string_view tensor_name) const override;

  absl::Status Clear() override;

  // Returns true if all state buffers are backed by host memory.
  bool IsHostMemory() const;

  // Agello telemetry: the bytes a copy of this state moves (every buffer of both banks).
  int64_t ByteSize() const;

  // Agello telemetry: the bytes a snapshot holds and a restore from it moves (E4: the filled
  // slots), or ByteSize() for any other state.
  int64_t CopiedBytes() const;

  // Copies the contents of all state buffers (both banks for kPingPong) and
  // the ping-pong bank index from `other` into this state's existing buffers,
  // without reallocating them. `other` must have the same allocation policy,
  // batch size, number of entries and set of state buffers. The buffer types
  // may differ (e.g. restoring a host memory snapshot into GPU buffers).
  //
  // This is currently used to restore host memory snapshots of Metal state
  // buffers, which costs a full host -> device copy per restore.
  // TODO: b/572018960 - Revisit once LiteRT supports keeping Metal snapshots
  // in device memory so that restores can swap buffers instead of copying.
  absl::Status CopyContentsFrom(const LitertState& other);

  // Agello (ring rewind): the ring caches' bytes. A ring cache (a KV cache with fewer slots
  // than the largest one) can't rewind far on its own: the positions written after a step
  // overwrite slots that the step's window reads. The executor copies the rings at a rewind
  // point and writes them back when it rewinds to it.
  struct RingBuffer {
    std::string name;
    std::vector<uint8_t> bytes;
    bool is_k = false;
    size_t a_dim = 0;   // [1, heads, A, C]: K caches have their slots along A, V caches along C
    size_t slices = 0;  // groups of four along C
    size_t width = 1;   // bytes per element
  };
  struct RingCopy {
    std::vector<RingBuffer> buffers;
    int64_t bytes = 0;
  };

  // The bytes in `window_slots` that differ between two copies of the same rings.
  // With `per_slot`, also the count for each slot that differs, and whether it's in a K cache.
  static int64_t DifferentWindowBytes(
      const RingCopy& a, const RingCopy& b, const std::vector<int>& window_slots,
      int ring_slots, std::vector<std::pair<int, int64_t>>* per_slot = nullptr);

  // The ring caches' slots, 0 when there are none.
  int RingSlots() const;

  // A copy of the ring caches; empty without rings, or when the buffers aren't shared Metal
  // ones that the host can read.
  absl::StatusOr<RingCopy> CopyRings() const;

  // Writes `copy` back into the ring caches.
  absl::Status RestoreRings(const RingCopy& copy);

  // Agello (sparse KV; PERFORMANCE.md, "The window: a fixed 64K with sparse KV caches"): the
  // global KV caches (8,192 slots or more) are Metal 4 placement sparse buffers that cost only
  // their mapped tiles, mapped in steps of 4,096 positions ahead of the fill (AGELLO_SPARSE_KV=0:
  // ordinary caches; =full: every step mapped at creation). Before a run that writes positions
  // through `last_position`, EnsureKvMapped maps what it needs; TrimKvMapping unmaps the steps
  // past `last_position` (a new conversation's first run). No-ops without sparse caches.
  absl::Status EnsureKvMapped(int last_position);
  absl::Status TrimKvMapping(int last_position);
  // The bytes mapped into the sparse caches, 0 without them.
  int64_t SparseKvMappedBytes() const;

  ~LitertState() override;

  // Resizes the KV cache to the given number of entries (sequence length).
  // Note: Resize is a no-op if the requested size is smaller than the current
  // size.
  absl::Status Resize(CompiledModel& compiled_model,
                      absl::string_view signature_name, int num_entries);

  struct StateBuffers {
    absl::flat_hash_map<absl::string_view, TensorBuffer> input_buffers;
    absl::flat_hash_map<absl::string_view, TensorBuffer> output_buffers;
  };

  // For backends that support inplace update, this returns a single set of
  // state cache buffers that can be used for both input and output (i.e,
  // input_buffers and output_buffers point to the same data).
  // For backends that don't support inplace update, this returns two distinct
  // sets of state cache buffers, one for input and one for output. On each
  // call, the input/output buffers will be swapped.
  absl::StatusOr<StateBuffers> GetStateBuffers(
      CompiledModel& compiled_model, absl::string_view signature_name);

 private:
  struct StateBuffer {
    TensorBuffer buffer;
    proto::StateBuffer::Type type = proto::StateBuffer::TYPE_UNSPECIFIED;
    std::optional<int> dynamic_dim;
  };

  absl::Status SyncShapes(CompiledModel& compiled_model,
                          absl::string_view signature_name);

  static absl::StatusOr<std::unique_ptr<LitertState>> HeuristicBasedCreate(
      Environment& env, CompiledModel& compiled_model,
      absl::string_view signature_name, AllocationPolicy allocation_policy,
      int batch_size, bool clear_kv_cache_before_prefill);

  static absl::StatusOr<std::unique_ptr<LitertState>> MetadataBasedCreate(
      Environment& env, CompiledModel& compiled_model,
      absl::string_view signature_name,
      const proto::ExecutorMetadata& executor_metadata,
      AllocationPolicy allocation_policy, int batch_size,
      bool clear_kv_cache_before_prefill);

  LitertState(
      int batch_size, int num_entries, Environment& env,
      absl::flat_hash_map<std::string, StateBuffer> bank_1_state_buffers,
      std::optional<absl::flat_hash_map<std::string, StateBuffer>>
          bank_2_state_buffers,
      AllocationPolicy allocation_policy)
      : batch_size_(batch_size),
        num_entries_(num_entries),
        env_(env),
        bank_1_state_buffers_(std::move(bank_1_state_buffers)),
        bank_2_state_buffers_(std::move(bank_2_state_buffers)),
        allocation_policy_(allocation_policy) {}

  // Batch size of the KV cache buffers.
  int batch_size_;
  // Number of entries in the KV cache.
  int num_entries_;
  // Environment to create new TensorBuffers (required for resizing).
  Environment& env_;
  // Primary state buffers.
  absl::flat_hash_map<std::string, StateBuffer> bank_1_state_buffers_;
  // Secondary state buffers - only used when inplace update is not
  // supported.
  std::optional<absl::flat_hash_map<std::string, StateBuffer>>
      bank_2_state_buffers_;

  bool bank_1_is_input_ = true;
  // Agello (T3): this state is a raw snapshot of Metal buffers (DeepCopy): host buffers holding
  // each MTLBuffer's bytes in the GPU's layout, only meaningful to CopyContentsFrom.
  bool raw_metal_snapshot_ = false;
  // Agello (E4): a raw snapshot of the first `positions` slots of each KV cache only.
  std::optional<int> snapshot_positions_;
  int64_t copied_bytes_ = -1;

  absl::StatusOr<std::unique_ptr<StateInterface>> DeepCopyImpl(
      std::optional<int> positions) const;

  // Agello (sparse KV, step 1): a snapshot of the first `positions` slots of every state buffer
  // by GPU blits on `command_queue` into compact host memory (the filled runs back to back), so
  // the snapshot holds only those bytes and the buffers need no CPU view.
  absl::StatusOr<std::unique_ptr<StateInterface>> BlitSnapshot(
      int positions, const void* command_queue) const;
  // Agello (sparse KV, step 1): this state is such a snapshot.
  bool blit_snapshot_ = false;
  // Agello (P3.3): this state holds detached Metal 4 sparse KV page-table mappings
  // (plus optional host blit snapshots of non-sparse ring caches).
  bool sparse_snapshot_ = false;
  std::shared_ptr<SparseKvMappingSnapshot> sparse_mapping_snapshot_;

  AllocationPolicy allocation_policy_;

  // Agello (sparse KV): replaces the global KV caches with placement sparse buffers when asked
  // and supported; called once, after creation. Clears any candidate buffer that was not
  // replaced when `clear_buffer` is true.
  absl::Status ConvertToSparseKv(const void* command_queue, bool clear_buffer);
  struct SparseKv;
  std::shared_ptr<SparseKv> sparse_kv_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_STATE_H_
