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

#include "runtime/executor/litert/state.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <variant>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_environment_options.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/executor/common_utils.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/state_interface.h"
#include "runtime/util/metal_raw_buffer.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/tensor_buffer_util.h"

namespace litert::lm {

namespace {

// Agello (sparse KV, step 0, a test): AGELLO_KV_POISON=1 fills the KV data buffers (a megabyte or
// more: INT8 values, not the scales) with 0x7F after each clear, so any reader that depends on the
// cache holding zeros past the fill changes the reply. A placement sparse cache's reused tiles
// keep old data, so the sparse KV needs no reader of zeros.
// Agello (sparse KV): by default the global caches are sparse and mapped ahead of the fill
// (step 3), on a GPU with placement sparse buffers (others keep ordinary caches).
// AGELLO_SPARSE_KV=0 keeps them ordinary; =full maps every step at creation (step 2).
enum class SparseKvMode { kOff, kFull, kGrow };
SparseKvMode GetSparseKvMode() {
  static const SparseKvMode mode = [] {
    const char* value = std::getenv("AGELLO_SPARSE_KV");
    if (value == nullptr || value[0] == '\0') return SparseKvMode::kGrow;
    if (value[0] == '0' || std::string(value) == "off") return SparseKvMode::kOff;
    return std::string(value) == "full" ? SparseKvMode::kFull : SparseKvMode::kGrow;
  }();
  return mode;
}
constexpr int kSparseStepPositions = 4096;
constexpr int kSparseMinSlots = 8192;
bool LogSparseKv() { return std::getenv("AGELLO_LOG_SPARSE") != nullptr; }

// A global KV cache (8,192 slots or more) that becomes sparse after creation, so its clear is
// skipped (no reader needs zeros past the fill: the poison test; unmapped tiles read zero).
bool SparseKvCandidate(absl::string_view name, const TensorBuffer& buffer) {
  if (GetSparseKvMode() == SparseKvMode::kOff) return false;
  const bool is_k = absl::StartsWith(name, "kv_cache_k_");
  if (!is_k && !absl::StartsWith(name, "kv_cache_v_")) return false;
  auto tensor_type = buffer.TensorType();
  if (!tensor_type) return false;
  auto dims = tensor_type->Layout().Dimensions();
  if (dims.size() != 4 || dims[0] != 1) return false;
  return (is_k ? dims[2] : dims[3]) >= kSparseMinSlots;
}

const void* MetalCommandQueueOf(const Environment& env);

absl::Status PoisonIfAsked(TensorBuffer& buffer) {
  static const bool poison = [] {
    const char* value = std::getenv("AGELLO_KV_POISON");
    return value != nullptr && value[0] == '1';
  }();
  if (!poison) return absl::OkStatus();
  LITERT_ASSIGN_OR_RETURN(size_t size, buffer.PackedSize());
  if (size < (size_t{1} << 20)) return absl::OkStatus();
  LITERT_ASSIGN_OR_RETURN(auto lock, TensorBufferScopedLock::Create(
                                         buffer, TensorBuffer::LockMode::kWrite));
  std::memset(lock.second, 0x7F, size);
  return absl::OkStatus();
}

}  // namespace

namespace {

constexpr int kDynamicDimValue = -1;

absl::StatusOr<std::optional<int>> GetDynamicDimIndex(
    const CompiledModel& compiled_model, absl::string_view signature,
    absl::string_view tensor_name) {
  LITERT_ASSIGN_OR_RETURN(
      const RankedTensorType ranked_tensor_type,
      compiled_model.GetInputTensorType(signature, tensor_name));
  auto dimensions = ranked_tensor_type.Layout().Dimensions();
  std::optional<int> dynamic_dim_index;
  for (int i = 0; i < dimensions.size(); ++i) {
    if (dimensions[i] == kDynamicDimValue) {
      RET_CHECK(!dynamic_dim_index.has_value())
          << "Multiple dynamic dimensions are not supported.";
      dynamic_dim_index = i;
    }
  }
  return dynamic_dim_index;
}

absl::Status ResolveDynamicShape(CompiledModel& compiled_model,
                                 absl::string_view signature,
                                 absl::string_view tensor_name, int new_value) {
  LITERT_ASSIGN_OR_RETURN(
      const RankedTensorType ranked_tensor_type,
      compiled_model.GetInputTensorType(signature, tensor_name));
  auto dimensions = ranked_tensor_type.Layout().Dimensions();

  bool has_dynamic_dim = false;
  std::vector<int> new_shape;
  new_shape.reserve(dimensions.size());
  for (int i = 0; i < dimensions.size(); ++i) {
    if (dimensions[i] == kDynamicDimValue) {
      has_dynamic_dim = true;
      new_shape.push_back(new_value);
    } else {
      new_shape.push_back(dimensions[i]);
    }
  }

  if (has_dynamic_dim) {
    LITERT_RETURN_IF_ERROR(
        compiled_model.ResizeInputTensor(signature, tensor_name, new_shape));
  }

  return absl::OkStatus();
}

absl::StatusOr<TensorBuffer> ResizeTensorBuffer(Environment& env,
                                                TensorBuffer& tensor_buffer,
                                                int dynamic_dim_index,
                                                int num_entries_to_insert) {
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType& tensor_type,
                          tensor_buffer.TensorType());
  RET_CHECK(!tensor_type.Layout().HasStrides());
  auto dimensions = tensor_type.Layout().Dimensions();
  std::vector<int> new_dimensions;
  new_dimensions.reserve(dimensions.size());
  for (int i = 0; i < dimensions.size(); ++i) {
    if (i == dynamic_dim_index) {
      new_dimensions.push_back(dimensions[i] + num_entries_to_insert);
    } else {
      new_dimensions.push_back(dimensions[i]);
    }
  }

  LITERT_ASSIGN_OR_RETURN(TensorBufferType buffer_type,
                          tensor_buffer.BufferType());
  Layout new_layout(Dimensions(new_dimensions.begin(), new_dimensions.end()));
  auto new_out_type =
      RankedTensorType(tensor_type.ElementType(), std::move(new_layout));
  LITERT_ASSIGN_OR_RETURN(size_t new_size, new_out_type.Bytes());

  LITERT_ASSIGN_OR_RETURN(
      TensorBuffer new_tensor_buffer,
      TensorBuffer::CreateManaged(env, buffer_type, new_out_type, new_size));
  LITERT_RETURN_IF_ERROR(new_tensor_buffer.Clear());

  LITERT_ASSIGN_OR_RETURN(auto tensor_buffer_lock_and_addr,
                          TensorBufferScopedLock::Create(
                              tensor_buffer, TensorBuffer::LockMode::kRead));
  auto* tensor_buffer_ptr =
      static_cast<uint8_t*>(tensor_buffer_lock_and_addr.second);
  LITERT_ASSIGN_OR_RETURN(
      auto new_tensor_buffer_lock_and_addr,
      TensorBufferScopedLock::Create(new_tensor_buffer,
                                     TensorBuffer::LockMode::kWrite));
  auto* new_tensor_buffer_ptr =
      static_cast<uint8_t*>(new_tensor_buffer_lock_and_addr.second);
  std::optional<size_t> element_size = GetByteWidth(tensor_type.ElementType());
  RET_CHECK(element_size.has_value());

  ABSL_RETURN_IF_ERROR(ExpandBuffer(tensor_buffer_ptr, dimensions,
                                    new_tensor_buffer_ptr, new_dimensions,
                                    element_size.value()));

  return new_tensor_buffer;
}

absl::Status SelectAndCopyBuffer(TensorBuffer& dst, const TensorBuffer& src,
                                 int batch_index) {
  LITERT_ASSIGN_OR_RETURN(
      auto src_buffer_lock_and_addr,
      TensorBufferScopedLock::Create(src, TensorBuffer::LockMode::kRead));
  const char* src_buffer_ptr =
      static_cast<const char*>(src_buffer_lock_and_addr.second);

  LITERT_ASSIGN_OR_RETURN(
      auto dst_buffer_lock_and_addr,
      TensorBufferScopedLock::Create(dst, TensorBuffer::LockMode::kWrite));
  LITERT_ASSIGN_OR_RETURN(size_t dst_buffer_size, dst.PackedSize());
  char* dst_buffer_ptr =
      static_cast<char*>(const_cast<void*>(dst_buffer_lock_and_addr.second));
  // This copy is based on the assumption that the KV cache buffers are in the
  // layout of [batch * X, ...] or [1, batch * X, ...] where X could be 1 or
  // more and X doesn't make values interleaved across batches which is true
  // for the current LLM models of all backends.
  src_buffer_ptr += batch_index * dst_buffer_size;
  memcpy(dst_buffer_ptr, src_buffer_ptr, dst_buffer_size);
  return absl::OkStatus();
}

absl::Status BroadcastAndCopyBuffer(TensorBuffer& dst, int dst_batch_size,
                                    const TensorBuffer& src) {
  LITERT_ASSIGN_OR_RETURN(
      auto src_buffer_lock_and_addr,
      TensorBufferScopedLock::Create(src, TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(size_t src_buffer_size, src.PackedSize());
  const char* src_buffer_ptr =
      static_cast<const char*>(src_buffer_lock_and_addr.second);

  LITERT_ASSIGN_OR_RETURN(
      auto dst_buffer_lock_and_addr,
      TensorBufferScopedLock::Create(dst, TensorBuffer::LockMode::kWrite));
  char* dst_buffer_ptr =
      static_cast<char*>(const_cast<void*>(dst_buffer_lock_and_addr.second));

  for (int i = 0; i < dst_batch_size; ++i) {
    memcpy(dst_buffer_ptr, src_buffer_ptr, src_buffer_size);
    dst_buffer_ptr += src_buffer_size;
  }
  return absl::OkStatus();
}

// Copies the contents of `src` into `dst`. Both buffers must have the same
// packed size, but may have different buffer types.
absl::Status CopyBufferContents(TensorBuffer& dst, const TensorBuffer& src) {
  LITERT_ASSIGN_OR_RETURN(size_t src_size, src.PackedSize());
  LITERT_ASSIGN_OR_RETURN(size_t dst_size, dst.PackedSize());
  RET_CHECK_EQ(src_size, dst_size);
  LITERT_ASSIGN_OR_RETURN(
      auto src_lock_and_addr,
      TensorBufferScopedLock::Create(src, TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(
      auto dst_lock_and_addr,
      TensorBufferScopedLock::Create(dst, TensorBuffer::LockMode::kWrite));
  memcpy(dst_lock_and_addr.second, src_lock_and_addr.second, dst_size);
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<LitertState>> LitertState::Create(
    Environment& env, CompiledModel& compiled_model,
    absl::string_view signature_name,
    const proto::ExecutorMetadata* executor_metadata,
    AllocationPolicy allocation_policy, int batch_size,
    bool clear_kv_cache_before_prefill) {
  std::unique_ptr<LitertState> state;
  if (executor_metadata == nullptr) {
    ABSL_ASSIGN_OR_RETURN(state, HeuristicBasedCreate(env, compiled_model, signature_name,
                                                      allocation_policy, batch_size,
                                                      clear_kv_cache_before_prefill));
  } else {
    ABSL_ASSIGN_OR_RETURN(state, MetadataBasedCreate(env, compiled_model, signature_name,
                                                     *executor_metadata, allocation_policy,
                                                     batch_size, clear_kv_cache_before_prefill));
  }
  ABSL_RETURN_IF_ERROR(state->ConvertToSparseKv(MetalCommandQueueOf(env)));
  return state;
}

// Agello (sparse KV).
struct LitertState::SparseKv {
  struct Buffer {
    std::string name;
    void* mtl_buffer = nullptr;  // retained id<MTLBuffer>
    bool is_k = false;
    size_t heads = 0, w = 0, slices = 0, vec = 0;
    int slots = 0;
  };
  std::unique_ptr<SparseKvMapper> mapper;
  std::vector<Buffer> buffers;
  int slots = 0;

  ~SparseKv() {
    mapper.reset();
    for (const Buffer& buffer : buffers) ReleaseMetalObject(buffer.mtl_buffer);
  }

  // The tile ranges of positions [from, to) in each buffer (FilledRuns' layout): a K cache holds
  // one row of its slots per head and group of four dims, a V cache its slots in groups of four
  // per head.
  std::vector<SparseRange> StepRanges(int from, int to) const {
    std::vector<SparseRange> ranges;
    for (const Buffer& b : buffers) {
      const int end = std::min(to, b.slots);
      if (end <= from) continue;
      if (b.is_k) {
        for (size_t y = 0; y < b.heads; ++y) {
          for (size_t s = 0; s < b.slices; ++s) {
            ranges.push_back({b.mtl_buffer, (y * b.slices + s) * b.w * b.vec + from * b.vec,
                              static_cast<size_t>(end - from) * b.vec});
          }
        }
      } else {
        for (size_t y = 0; y < b.heads; ++y) {
          ranges.push_back({b.mtl_buffer, y * b.slices * b.w * b.vec + from / 4 * b.w * b.vec,
                            static_cast<size_t>(end - from) / 4 * b.w * b.vec});
        }
      }
    }
    return ranges;
  }
};

LitertState::~LitertState() {
  // The tensor buffers wrap the sparse caches without owning them: they go first.
  bank_1_state_buffers_.clear();
  bank_2_state_buffers_.reset();
  sparse_kv_.reset();
}

absl::Status LitertState::ConvertToSparseKv(const void* command_queue) {
  if (GetSparseKvMode() == SparseKvMode::kOff) return absl::OkStatus();
  if (bank_2_state_buffers_.has_value() || command_queue == nullptr) {
    fprintf(stderr, "Agello sparse KV: off (%s)\n",
            command_queue == nullptr ? "no Metal command queue" : "ping-pong caches");
    return absl::OkStatus();
  }
  auto sparse = std::make_shared<SparseKv>();
  const size_t tile = SparseKvTileBytes();
  for (auto& [name, state_buffer] : bank_1_state_buffers_) {
    TensorBuffer& buffer = state_buffer.buffer;
    if (!SparseKvCandidate(name, buffer) || !buffer.IsMetalMemory()) continue;
    LITERT_ASSIGN_OR_RETURN(void* mtl_buffer, buffer.GetMetalBuffer());
    void* device = MetalDeviceOfBuffer(mtl_buffer);
    if (!PlacementSparseBuffersSupported(device)) {
      fprintf(stderr, "Agello sparse KV: this GPU has no placement sparse buffers\n");
      return absl::OkStatus();
    }
    LITERT_ASSIGN_OR_RETURN(auto tensor_type, buffer.TensorType());
    LITERT_ASSIGN_OR_RETURN(auto buffer_type, buffer.BufferType());
    LITERT_ASSIGN_OR_RETURN(size_t size, buffer.Size());
    LITERT_ASSIGN_OR_RETURN(size_t packed, buffer.PackedSize());
    std::optional<size_t> width = GetByteWidth(tensor_type.ElementType());
    auto dims = tensor_type.Layout().Dimensions();
    if (!width.has_value() || dims.size() != 4) continue;
    SparseKv::Buffer entry;
    entry.name = name;
    entry.is_k = absl::StartsWith(name, "kv_cache_k_");
    entry.heads = dims[1];
    entry.w = dims[2];
    entry.slices = (dims[3] + 3) / 4;
    entry.vec = 4 * *width;
    entry.slots = entry.is_k ? dims[2] : dims[3];
    if (entry.slices * entry.heads * entry.w * entry.vec != packed) continue;
    if (!sparse->mapper) {
      sparse->mapper = std::make_unique<SparseKvMapper>(device, command_queue);
      if (!sparse->mapper->ok()) {
        fprintf(stderr, "Agello sparse KV: no Metal 4 mapping queue\n");
        return absl::OkStatus();
      }
    }
    entry.mtl_buffer = CreateSparseMetalBuffer(device, (size + tile - 1) / tile * tile);
    if (entry.mtl_buffer == nullptr) {
      fprintf(stderr, "Agello sparse KV: no sparse buffer of %zu bytes for %s\n", size,
              name.c_str());
      continue;
    }
    auto created = TensorBuffer::CreateFromMetalBuffer(env_, tensor_type, buffer_type,
                                                       entry.mtl_buffer, size);
    if (!created) {
      ReleaseMetalObject(entry.mtl_buffer);
      return absl::InternalError(absl::StrCat("Sparse KV: can't wrap a buffer for ", name));
    }
    buffer = std::move(*created);
    sparse->slots = std::max(sparse->slots, entry.slots);
    sparse->buffers.push_back(std::move(entry));
  }
  if (sparse->buffers.empty()) return absl::OkStatus();
  sparse_kv_ = std::move(sparse);
  if (LogSparseKv()) {
    fprintf(stderr, "Agello sparse KV: %zu caches of %d slots are sparse (%s)\n",
            sparse_kv_->buffers.size(), sparse_kv_->slots,
            GetSparseKvMode() == SparseKvMode::kFull ? "fully mapped" : "mapped as they fill");
  }
  if (GetSparseKvMode() == SparseKvMode::kFull) return EnsureKvMapped(sparse_kv_->slots - 1);
  return absl::OkStatus();
}

absl::Status LitertState::EnsureKvMapped(int last_position) {
  if (sparse_kv_ == nullptr) return absl::OkStatus();
  SparseKvMapper& mapper = *sparse_kv_->mapper;
  const int total_steps =
      (sparse_kv_->slots + kSparseStepPositions - 1) / kSparseStepPositions;
  const int needed = std::min(total_steps, std::max(last_position, 0) / kSparseStepPositions + 1);
  while (mapper.steps() < needed) {
    const int from = mapper.steps() * kSparseStepPositions;
    RET_CHECK(mapper.MapStep(sparse_kv_->StepRanges(from, from + kSparseStepPositions)))
        << "Sparse KV: mapping positions " << from << " failed";
    if (LogSparseKv()) {
      fprintf(stderr, "Agello sparse KV: mapped through position %d, %.0f MB\n",
              from + kSparseStepPositions - 1, mapper.mapped_bytes() / 1048576.0);
    }
  }
  return absl::OkStatus();
}

absl::Status LitertState::TrimKvMapping(int last_position) {
  if (sparse_kv_ == nullptr || GetSparseKvMode() != SparseKvMode::kGrow) {
    return absl::OkStatus();
  }
  SparseKvMapper& mapper = *sparse_kv_->mapper;
  const int keep = std::max(last_position, 0) / kSparseStepPositions + 1;
  while (mapper.steps() > keep) {
    RET_CHECK(mapper.UnmapLastStep()) << "Sparse KV: unmapping failed";
    if (LogSparseKv()) {
      fprintf(stderr, "Agello sparse KV: trimmed to %d steps, %.0f MB\n", mapper.steps(),
              mapper.mapped_bytes() / 1048576.0);
    }
  }
  return absl::OkStatus();
}

int64_t LitertState::SparseKvMappedBytes() const {
  return sparse_kv_ == nullptr ? 0 : static_cast<int64_t>(sparse_kv_->mapper->mapped_bytes());
}

absl::Status LitertState::SelectAndCopyFrom(StateInterface& other,
                                            int batch_index) {
  auto other_litert = dynamic_cast<LitertState*>(&other);
  RET_CHECK(other_litert != nullptr) << "Only support LitertState.";
  RET_CHECK(!bank_2_state_buffers_.has_value());
  RET_CHECK(!other_litert->bank_2_state_buffers_.has_value());
  RET_CHECK_GT(other_litert->batch_size_, batch_size_);
  RET_CHECK_LT(batch_index, other_litert->batch_size_);
  RET_CHECK_EQ(num_entries_, other_litert->num_entries_);

  for (auto& [input_name, state_buffer] : bank_1_state_buffers_) {
    auto it = other_litert->bank_1_state_buffers_.find(input_name);
    RET_CHECK(it != other_litert->bank_1_state_buffers_.end());
    ABSL_RETURN_IF_ERROR(SelectAndCopyBuffer(state_buffer.buffer,
                                             it->second.buffer, batch_index));
  }
  return absl::OkStatus();
}

absl::Status LitertState::BroadcastAndCopyFrom(StateInterface& other) {
  auto other_litert = dynamic_cast<LitertState*>(&other);
  RET_CHECK(other_litert != nullptr) << "Only support LitertState.";
  RET_CHECK(!bank_2_state_buffers_.has_value());
  RET_CHECK(!other_litert->bank_2_state_buffers_.has_value());
  RET_CHECK_EQ(other_litert->batch_size_, 1);
  RET_CHECK_GT(batch_size_, other_litert->batch_size_);
  RET_CHECK_EQ(num_entries_, other_litert->num_entries_);

  for (auto& [input_name, state_buffer] : bank_1_state_buffers_) {
    auto it = other_litert->bank_1_state_buffers_.find(input_name);
    RET_CHECK(it != other_litert->bank_1_state_buffers_.end());
    ABSL_RETURN_IF_ERROR(BroadcastAndCopyBuffer(
        state_buffer.buffer, batch_size_, it->second.buffer));
  }

  return absl::OkStatus();
}

namespace {

// Agello (E4): a byte range of a buffer.
struct ByteRun {
  size_t offset;
  size_t length;
};

void CopyRuns(void* dst, const void* src, size_t length, const std::vector<ByteRun>* runs) {
  if (runs == nullptr) {
    memcpy(dst, src, length);
    return;
  }
  for (const ByteRun& run : *runs) {
    memcpy(static_cast<uint8_t*>(dst) + run.offset,
           static_cast<const uint8_t*>(src) + run.offset, run.length);
  }
}

// Agello (T3): a raw snapshot of a buffer with its own shared MTLBuffer: a host buffer of the
// MTLBuffer's length holding its bytes as they are. A lock would convert the GPU layout to row
// order and back on restore (the KV caches' transpose cost about 290 ms per snapshot at 16K).
// The host buffer has pages of its own, unmapped when the snapshot is freed.
// Agello (E4): with `runs`, only those bytes are copied; the snapshot keeps the buffer's length
// and layout, and its pages past them are never touched (never resident).
absl::StatusOr<TensorBuffer> RawMetalSnapshot(Environment& env, const TensorBuffer& buffer,
                                              const MetalRawBuffer& raw,
                                              const std::vector<ByteRun>* runs = nullptr) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, buffer.TensorType());
  if (void* memory = AllocateSnapshotMemory(raw.length); memory != nullptr) {
    CopyRuns(memory, raw.contents, raw.length, runs);
    auto litert_type = static_cast<LiteRtRankedTensorType>(tensor_type);
    auto holder = env.GetHolder();
    LiteRtTensorBuffer handle;
    if (holder.runtime->CreateTensorBufferFromHostMemory(&litert_type, memory, raw.length,
                                                         FreeSnapshotMemory,
                                                         &handle) != kLiteRtStatusOk) {
      FreeSnapshotMemory(memory);
      return absl::InternalError("Failed to wrap a KV snapshot's host memory");
    }
    return TensorBuffer::WrapCObject(holder, handle, OwnHandle::kYes);
  }
  LITERT_ASSIGN_OR_RETURN(
      auto copy,
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory, tensor_type, raw.length));
  LITERT_ASSIGN_OR_RETURN(auto lock_and_addr,
                          TensorBufferScopedLock::Create(copy, TensorBuffer::LockMode::kWrite));
  CopyRuns(lock_and_addr.second, raw.contents, raw.length, runs);
  return copy;
}

// Agello (E4): the bytes of a KV cache buffer that hold its first `positions` slots. The caches
// are [1, heads, A, C] tensors stored with the heads outermost, then groups of four along C, then
// A: element ((y * slices + c / 4) * A + a) * 4 + c % 4 (read on the M6 with
// AGELLO_VERIFY_KV_PREFIX: every byte of a filled cache inside these runs, none outside). K caches
// are [1, heads, slots, dim]: one run per head and group of four dims. V caches are
// [1, heads, dim, slots], the slots along C: one run per head. Other buffers, and one whose
// length isn't that layout's, have none: they're copied whole. A ring cache whose positions
// passed its slots is one run of everything.
std::optional<std::vector<ByteRun>> FilledRuns(absl::string_view name, const TensorBuffer& buffer,
                                               size_t length, int positions) {
  const bool is_k = absl::StartsWith(name, "kv_cache_k_");
  const bool is_v = absl::StartsWith(name, "kv_cache_v_");
  if (!is_k && !is_v) return std::nullopt;
  auto tensor_type = buffer.TensorType();
  if (!tensor_type) return std::nullopt;
  auto dims = tensor_type->Layout().Dimensions();
  if (dims.size() != 4 || dims[0] != 1) return std::nullopt;
  std::optional<size_t> width = GetByteWidth(tensor_type->ElementType());
  if (!width.has_value()) return std::nullopt;
  const size_t h = dims[1], w = dims[2], c = dims[3];
  const size_t slices = (c + 3) / 4;
  const size_t vec = 4 * *width;
  if (slices * h * w * vec != length) return std::nullopt;
  const size_t slots = is_k ? w : c;
  const size_t n = std::min<size_t>(slots, static_cast<size_t>(std::max(positions, 0)));
  std::vector<ByteRun> runs;
  if (n == slots) {
    runs.push_back({0, length});
  } else if (is_k) {
    runs.reserve(h * slices);
    for (size_t y = 0; y < h; ++y) {
      for (size_t s = 0; s < slices; ++s) runs.push_back({(y * slices + s) * w * vec, n * vec});
    }
  } else {
    runs.reserve(h);
    for (size_t y = 0; y < h; ++y) runs.push_back({y * slices * w * vec, (n + 3) / 4 * w * vec});
  }
  return runs;
}

// Agello (E4): with AGELLO_VERIFY_KV_PREFIX set, a check of the layout FilledRuns assumes, on a
// live buffer: the share of zero bytes inside the runs of the first `positions` slots (low when
// the layout holds) and the nonzero bytes outside the runs of `positions` + 1,024 (zero on an
// engine whose caches were never filled further, when the layout holds; the 1,024 slots cover
// a prefill chunk's padding and speculative rounds).
void VerifyFilledRuns(absl::string_view name, const TensorBuffer& buffer, const uint8_t* bytes,
                      size_t length, int positions) {
  auto inside = FilledRuns(name, buffer, length, positions);
  auto margin = FilledRuns(name, buffer, length, positions + 1024);
  if (!inside.has_value() || !margin.has_value()) return;
  size_t inside_bytes = 0, inside_zero = 0;
  for (const ByteRun& run : *inside) {
    inside_bytes += run.length;
    for (size_t i = 0; i < run.length; ++i) inside_zero += bytes[run.offset + i] == 0;
  }
  std::vector<bool> covered(length, false);
  for (const ByteRun& run : *margin) {
    std::fill(covered.begin() + run.offset, covered.begin() + run.offset + run.length, true);
  }
  size_t outside_nonzero = 0;
  for (size_t i = 0; i < length; ++i) outside_nonzero += !covered[i] && bytes[i] != 0;
  ABSL_LOG(INFO) << "Agello E4 verify " << name << ": positions " << positions << ", runs "
                 << inside->size() << ", inside " << inside_bytes << " bytes ("
                 << (inside_bytes ? 100.0 * inside_zero / inside_bytes : 0.0)
                 << " % zero), nonzero past the margin " << outside_nonzero << " of " << length;
}

// Agello (sparse KV, step 1): the environment's Metal command queue, the one the GPU's runs use,
// or nullptr.
const void* MetalCommandQueueOf(const Environment& env) {
  auto options = env.GetOptions();
  if (!options) return nullptr;
  auto queue = options->GetOption(EnvironmentOptions::Tag::kMetalCommandQueue);
  if (!queue) return nullptr;
  // The environment hands pointers back as void* (ToStdAny), set as const void*.
  if (const auto* pointer = std::get_if<void*>(&queue.Value())) return *pointer;
  if (const auto* pointer = std::get_if<const void*>(&queue.Value())) return *pointer;
  return nullptr;
}

// AGELLO_KV_BLIT=0 keeps the CPU snapshots (T3, E4), for comparison.
bool KvBlitsEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("AGELLO_KV_BLIT");
    return value == nullptr || value[0] != '0';
  }();
  return enabled;
}

// AGELLO_SPARSE_MAPPING_SNAPSHOT=0 falls back to host blit snapshots for sparse KV.
bool SparseMappingSnapshotEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("AGELLO_SPARSE_MAPPING_SNAPSHOT");
    return value == nullptr || value[0] != '0';
  }();
  return enabled;
}

// A buffer's filled runs (FilledRuns, or the whole buffer) packed back to back in host memory.
struct CompactCopy {
  std::vector<MetalBlitRange> ranges;
  size_t bytes = 0;
};

CompactCopy CompactRanges(absl::string_view name, const TensorBuffer& buffer, size_t length,
                          int positions) {
  CompactCopy copy;
  std::optional<std::vector<ByteRun>> runs = FilledRuns(name, buffer, length, positions);
  if (!runs.has_value()) runs = std::vector<ByteRun>{{0, length}};
  copy.ranges.reserve(runs->size());
  for (const ByteRun& run : *runs) {
    copy.ranges.push_back({run.offset, copy.bytes, run.length});
    copy.bytes += run.length;
  }
  return copy;
}

}  // namespace

absl::StatusOr<std::unique_ptr<StateInterface>> LitertState::BlitSnapshot(
    int positions, const void* command_queue) const {
  int64_t copied_bytes = 0;
  auto snapshot = [&](absl::string_view name,
                      const TensorBuffer& buffer) -> absl::StatusOr<TensorBuffer> {
    LITERT_ASSIGN_OR_RETURN(void* mtl_buffer, buffer.GetMetalBuffer());
    LITERT_ASSIGN_OR_RETURN(size_t length, buffer.PackedSize());
    const CompactCopy copy = CompactRanges(name, buffer, length, positions);
    const size_t bytes = std::max<size_t>(copy.bytes, 1);
    void* memory = AllocateSnapshotMemory(bytes);
    RET_CHECK(memory != nullptr) << "No memory for a KV snapshot of " << bytes << " bytes";
    if (!BlitMetalRanges(command_queue, mtl_buffer, memory, copy.bytes, copy.ranges,
                         /*to_host=*/true)) {
      FreeSnapshotMemory(memory);
      return absl::InternalError(absl::StrCat("KV snapshot blit failed: ", name));
    }
    copied_bytes += static_cast<int64_t>(copy.bytes);
    const int32_t held[] = {static_cast<int32_t>(bytes)};
    auto litert_type = static_cast<LiteRtRankedTensorType>(
        RankedTensorType(ElementType::UInt8, Layout(Dimensions(held, held + 1))));
    auto holder = env_.GetHolder();
    LiteRtTensorBuffer handle;
    if (holder.runtime->CreateTensorBufferFromHostMemory(&litert_type, memory, bytes,
                                                         FreeSnapshotMemory,
                                                         &handle) != kLiteRtStatusOk) {
      FreeSnapshotMemory(memory);
      return absl::InternalError("Failed to wrap a KV snapshot's host memory");
    }
    return TensorBuffer::WrapCObject(holder, handle, OwnHandle::kYes);
  };
  auto snapshot_bank = [&](const absl::flat_hash_map<std::string, StateBuffer>& bank)
      -> absl::StatusOr<absl::flat_hash_map<std::string, StateBuffer>> {
    absl::flat_hash_map<std::string, StateBuffer> copy;
    for (const auto& [name, state_buffer] : bank) {
      ABSL_ASSIGN_OR_RETURN(auto buffer, snapshot(name, state_buffer.buffer));
      copy[name] = StateBuffer{
          .buffer = std::move(buffer),
          .type = state_buffer.type,
          .dynamic_dim = state_buffer.dynamic_dim,
      };
    }
    return copy;
  };
  ABSL_ASSIGN_OR_RETURN(auto bank_1, snapshot_bank(bank_1_state_buffers_));
  std::optional<absl::flat_hash_map<std::string, StateBuffer>> bank_2;
  if (bank_2_state_buffers_.has_value()) {
    ABSL_ASSIGN_OR_RETURN(bank_2, snapshot_bank(*bank_2_state_buffers_));
  }
  auto copy = absl::WrapUnique(new LitertState(batch_size_, num_entries_, env_,
                                               std::move(bank_1), std::move(bank_2),
                                               allocation_policy_));
  copy->bank_1_is_input_ = bank_1_is_input_;
  copy->blit_snapshot_ = true;
  copy->snapshot_positions_ = positions;
  copy->copied_bytes_ = copied_bytes;
  return copy;
}

absl::StatusOr<std::unique_ptr<StateInterface>> LitertState::DeepCopy() const {
  return DeepCopyImpl(std::nullopt);
}

absl::StatusOr<std::unique_ptr<StateInterface>> LitertState::DeepCopyFilled(
    int positions) const {
  return DeepCopyImpl(positions);
}

absl::StatusOr<std::unique_ptr<StateInterface>> LitertState::DetachSnapshot(int positions) {
  if (sparse_kv_ == nullptr || !sparse_kv_->mapper || !sparse_kv_->mapper->ok() ||
      bank_2_state_buffers_.has_value() || !KvBlitsEnabled() ||
      !SparseMappingSnapshotEnabled()) {
    return DeepCopyFilled(positions);
  }
  const void* command_queue = MetalCommandQueueOf(env_);
  if (command_queue == nullptr) return DeepCopyFilled(positions);

  auto is_sparse_name = [&](absl::string_view name) {
    for (const SparseKv::Buffer& b : sparse_kv_->buffers) {
      if (b.name == name) return true;
    }
    return false;
  };
  for (const auto& [name, state_buffer] : bank_1_state_buffers_) {
    if (!is_sparse_name(name) && !state_buffer.buffer.IsMetalMemory()) {
      return DeepCopyFilled(positions);
    }
  }

  // Snapshot any non-sparse buffers (e.g. local ring caches) first so a failure leaves the live
  // sparse mappings intact.
  int64_t copied_bytes = 0;
  absl::flat_hash_map<std::string, StateBuffer> non_sparse_bank;
  for (const auto& [name, state_buffer] : bank_1_state_buffers_) {
    if (is_sparse_name(name)) continue;
    LITERT_ASSIGN_OR_RETURN(void* mtl_buffer, state_buffer.buffer.GetMetalBuffer());
    LITERT_ASSIGN_OR_RETURN(size_t length, state_buffer.buffer.PackedSize());
    const CompactCopy copy = CompactRanges(name, state_buffer.buffer, length, positions);
    const size_t bytes = std::max<size_t>(copy.bytes, 1);
    void* memory = AllocateSnapshotMemory(bytes);
    RET_CHECK(memory != nullptr) << "No memory for a KV snapshot of " << bytes << " bytes";
    if (!BlitMetalRanges(command_queue, mtl_buffer, memory, copy.bytes, copy.ranges,
                         /*to_host=*/true)) {
      FreeSnapshotMemory(memory);
      return absl::InternalError(absl::StrCat("KV snapshot blit failed: ", name));
    }
    copied_bytes += static_cast<int64_t>(copy.bytes);
    const int32_t held[] = {static_cast<int32_t>(bytes)};
    auto litert_type = static_cast<LiteRtRankedTensorType>(
        RankedTensorType(ElementType::UInt8, Layout(Dimensions(held, held + 1))));
    auto holder = env_.GetHolder();
    LiteRtTensorBuffer handle;
    if (holder.runtime->CreateTensorBufferFromHostMemory(&litert_type, memory, bytes,
                                                         FreeSnapshotMemory,
                                                         &handle) != kLiteRtStatusOk) {
      FreeSnapshotMemory(memory);
      return absl::InternalError("Failed to wrap a KV snapshot's host memory");
    }
    non_sparse_bank[name] = StateBuffer{
        .buffer = TensorBuffer::WrapCObject(holder, handle, OwnHandle::kYes),
        .type = state_buffer.type,
        .dynamic_dim = state_buffer.dynamic_dim,
    };
  }

  std::optional<SparseKvMappingSnapshot> detached = sparse_kv_->mapper->DetachMappings();
  if (!detached.has_value()) return DeepCopyFilled(positions);
  if (LogSparseKv()) {
    fprintf(stderr, "Agello sparse KV: detached %.1f MB mappings (blitted %.1f MB non-sparse)\n",
            detached->mapped_bytes() / 1048576.0, copied_bytes / 1048576.0);
  }
  auto copy = absl::WrapUnique(new LitertState(batch_size_, num_entries_, env_,
                                               std::move(non_sparse_bank), std::nullopt,
                                               allocation_policy_));
  copy->bank_1_is_input_ = bank_1_is_input_;
  copy->sparse_snapshot_ = true;
  copy->sparse_mapping_snapshot_ =
      std::make_shared<SparseKvMappingSnapshot>(std::move(*detached));
  copy->snapshot_positions_ = positions;
  copy->copied_bytes_ = copied_bytes;
  return copy;
}

absl::Status LitertState::RestoreSparseSnapshot(LitertState& other) {
  RET_CHECK(other.sparse_snapshot_);
  RET_CHECK(allocation_policy_ == other.allocation_policy_) << "Allocation policy mismatch.";
  RET_CHECK_EQ(batch_size_, other.batch_size_);
  RET_CHECK_EQ(num_entries_, other.num_entries_);
  RET_CHECK(sparse_kv_ != nullptr && sparse_kv_->mapper && sparse_kv_->mapper->ok())
      << "No sparse KV mapper to restore sparse mapping snapshot";
  RET_CHECK(other.sparse_mapping_snapshot_ != nullptr);

  const size_t restored_sparse_bytes = other.sparse_mapping_snapshot_->mapped_bytes();
  RET_CHECK(sparse_kv_->mapper->RestoreMappings(std::move(*other.sparse_mapping_snapshot_)))
      << "Sparse KV: restoring detached mappings failed";

  if (!other.bank_1_state_buffers_.empty()) {
    const void* command_queue = MetalCommandQueueOf(env_);
    RET_CHECK(command_queue != nullptr) << "No Metal command queue to restore non-sparse KV";
    RET_CHECK(other.snapshot_positions_.has_value());
    const int positions = *other.snapshot_positions_;
    for (auto& [name, src_state] : other.bank_1_state_buffers_) {
      auto it = bank_1_state_buffers_.find(name);
      RET_CHECK(it != bank_1_state_buffers_.end()) << "Missing state buffer: " << name;
      LITERT_ASSIGN_OR_RETURN(void* mtl_buffer, it->second.buffer.GetMetalBuffer());
      LITERT_ASSIGN_OR_RETURN(size_t length, it->second.buffer.PackedSize());
      const CompactCopy copy = CompactRanges(name, it->second.buffer, length, positions);
      LITERT_ASSIGN_OR_RETURN(size_t held, src_state.buffer.PackedSize());
      RET_CHECK_GE(held, copy.bytes) << name;
      LITERT_ASSIGN_OR_RETURN(auto src_lock_and_addr,
                              TensorBufferScopedLock::Create(src_state.buffer,
                                                             TensorBuffer::LockMode::kRead));
      RET_CHECK(BlitMetalRanges(command_queue, mtl_buffer,
                                const_cast<void*>(src_lock_and_addr.second),
                                copy.bytes, copy.ranges, /*to_host=*/false))
          << "KV restore blit failed: " << name;
    }
  }
  if (LogSparseKv()) {
    fprintf(stderr, "Agello sparse KV: restored %.1f MB mappings (blitted %.1f MB non-sparse)\n",
            restored_sparse_bytes / 1048576.0, other.CopiedBytes() / 1048576.0);
  }
  bank_1_is_input_ = other.bank_1_is_input_;
  return absl::OkStatus();
}

int64_t LitertState::CopiedBytes() const {
  return copied_bytes_ >= 0 ? copied_bytes_ : ByteSize();
}

absl::StatusOr<std::unique_ptr<StateInterface>> LitertState::DeepCopyImpl(
    std::optional<int> positions) const {
  // Agello (sparse KV, step 1): the filled slots by GPU blits when every buffer is a Metal one.
  if (positions.has_value() && KvBlitsEnabled() && !bank_1_state_buffers_.empty()) {
    auto all_metal = [](const absl::flat_hash_map<std::string, StateBuffer>& bank) {
      for (const auto& [_, state_buffer] : bank) {
        if (!state_buffer.buffer.IsMetalMemory()) return false;
      }
      return true;
    };
    const void* command_queue = MetalCommandQueueOf(env_);
    if (command_queue != nullptr && all_metal(bank_1_state_buffers_) &&
        (!bank_2_state_buffers_.has_value() || all_metal(*bank_2_state_buffers_))) {
      return BlitSnapshot(*positions, command_queue);
    }
    static bool reported = [&] {
      fprintf(stderr, "Agello KV snapshot: no blit (%s); CPU copies instead\n",
              command_queue == nullptr ? "no Metal command queue" : "a buffer isn't a Metal one");
      return true;
    }();
    (void)reported;
  }
  // Agello (T3): raw snapshots when every buffer is a Metal buffer of its own.
  bool raw = !bank_1_state_buffers_.empty();
  auto all_raw = [](const absl::flat_hash_map<std::string, StateBuffer>& bank) {
    for (const auto& [_, state_buffer] : bank) {
      if (!RawMetalStorage(state_buffer.buffer)) return false;
    }
    return true;
  };
  raw = raw && all_raw(bank_1_state_buffers_) &&
        (!bank_2_state_buffers_.has_value() || all_raw(*bank_2_state_buffers_));
  // Agello (E4): the filled slots only, for raw snapshots that were asked for them.
  const bool filled = raw && positions.has_value();
  const bool verify = filled && std::getenv("AGELLO_VERIFY_KV_PREFIX") != nullptr;
  int64_t copied_bytes = 0;
  auto copy_buffer = [&](absl::string_view name,
                         const TensorBuffer& buffer) -> absl::StatusOr<TensorBuffer> {
    if (raw) {
      MetalRawBuffer storage = *RawMetalStorage(buffer);
      std::optional<std::vector<ByteRun>> runs;
      if (filled) runs = FilledRuns(name, buffer, storage.length, *positions);
      if (verify) {
        VerifyFilledRuns(name, buffer, static_cast<const uint8_t*>(storage.contents),
                         storage.length, *positions);
      }
      if (!runs.has_value()) {
        copied_bytes += static_cast<int64_t>(storage.length);
        return RawMetalSnapshot(env_, buffer, storage);
      }
      for (const ByteRun& run : *runs) copied_bytes += static_cast<int64_t>(run.length);
      return RawMetalSnapshot(env_, buffer, storage, &*runs);
    }
    LITERT_ASSIGN_OR_RETURN(auto buf_copy, CopyTensorBuffer(env_, buffer));
    return buf_copy;
  };

  absl::flat_hash_map<std::string, StateBuffer> bank_1_state_buffers;
  for (const auto& [name, state_buffer] : bank_1_state_buffers_) {
    ABSL_ASSIGN_OR_RETURN(auto buf_copy, copy_buffer(name, state_buffer.buffer));
    bank_1_state_buffers[name] = StateBuffer{
        .buffer = std::move(buf_copy),
        .type = state_buffer.type,
        .dynamic_dim = state_buffer.dynamic_dim,
    };
  }

  std::optional<absl::flat_hash_map<std::string, StateBuffer>>
      bank_2_state_buffers;
  if (bank_2_state_buffers_.has_value()) {
    bank_2_state_buffers.emplace();
    auto& map = *bank_2_state_buffers;
    for (const auto& [name, state_buffer] : *bank_2_state_buffers_) {
      ABSL_ASSIGN_OR_RETURN(auto buf_copy, copy_buffer(name, state_buffer.buffer));
      map[name] = StateBuffer{
          .buffer = std::move(buf_copy),
          .type = state_buffer.type,
          .dynamic_dim = state_buffer.dynamic_dim,
      };
    }
  }

  auto copy = absl::WrapUnique(new LitertState(
      batch_size_, num_entries_, env_, std::move(bank_1_state_buffers),
      std::move(bank_2_state_buffers), allocation_policy_));
  copy->bank_1_is_input_ = bank_1_is_input_;
  copy->raw_metal_snapshot_ = raw;
  if (filled) {
    copy->snapshot_positions_ = positions;
    copy->copied_bytes_ = copied_bytes;
  }

  return copy;
}

bool LitertState::IsHostMemory() const {
  auto is_host = [](const StateBuffer& state_buffer) {
    auto buffer_type = state_buffer.buffer.BufferType();
    return buffer_type && *buffer_type == TensorBufferType::kHostMemory;
  };
  for (const auto& [_, state_buffer] : bank_1_state_buffers_) {
    if (!is_host(state_buffer)) return false;
  }
  if (bank_2_state_buffers_.has_value()) {
    for (const auto& [_, state_buffer] : *bank_2_state_buffers_) {
      if (!is_host(state_buffer)) return false;
    }
  }
  return true;
}

int64_t LitertState::ByteSize() const {
  int64_t bytes = 0;
  auto add_bank = [&bytes](const absl::flat_hash_map<std::string, StateBuffer>& bank) {
    for (const auto& [_, state_buffer] : bank) {
      if (auto size = state_buffer.buffer.Size()) bytes += static_cast<int64_t>(*size);
    }
  };
  add_bank(bank_1_state_buffers_);
  if (bank_2_state_buffers_.has_value()) add_bank(*bank_2_state_buffers_);
  return bytes;
}

absl::Status LitertState::CopyContentsFrom(const LitertState& other) {
  if (this == &other) {
    return absl::OkStatus();
  }
  RET_CHECK(allocation_policy_ == other.allocation_policy_)
      << "Allocation policy mismatch.";
  RET_CHECK_EQ(batch_size_, other.batch_size_);
  RET_CHECK_EQ(num_entries_, other.num_entries_);
  RET_CHECK_EQ(bank_2_state_buffers_.has_value(),
               other.bank_2_state_buffers_.has_value());

  if (other.blit_snapshot_) {
    // Agello (sparse KV, step 1): a blit snapshot goes back by blits, run by run.
    const void* command_queue = MetalCommandQueueOf(env_);
    RET_CHECK(command_queue != nullptr) << "No Metal command queue to restore a KV snapshot";
    RET_CHECK(other.snapshot_positions_.has_value());
    const int positions = *other.snapshot_positions_;
    // A sparse cache may have been trimmed since (a helper's new conversation).
    ABSL_RETURN_IF_ERROR(EnsureKvMapped(positions - 1));
    auto restore_bank = [&](absl::flat_hash_map<std::string, StateBuffer>& dst,
                            const absl::flat_hash_map<std::string, StateBuffer>& src)
        -> absl::Status {
      RET_CHECK_EQ(dst.size(), src.size());
      for (auto& [name, state_buffer] : dst) {
        auto it = src.find(name);
        RET_CHECK(it != src.end()) << "Missing state buffer: " << name;
        LITERT_ASSIGN_OR_RETURN(void* mtl_buffer, state_buffer.buffer.GetMetalBuffer());
        LITERT_ASSIGN_OR_RETURN(size_t length, state_buffer.buffer.PackedSize());
        const CompactCopy copy = CompactRanges(name, state_buffer.buffer, length, positions);
        LITERT_ASSIGN_OR_RETURN(size_t held, it->second.buffer.PackedSize());
        RET_CHECK_GE(held, copy.bytes) << name;
        LITERT_ASSIGN_OR_RETURN(auto src_lock_and_addr,
                                TensorBufferScopedLock::Create(it->second.buffer,
                                                               TensorBuffer::LockMode::kRead));
        // The blit only reads the snapshot's memory.
        RET_CHECK(BlitMetalRanges(command_queue, mtl_buffer,
                                  const_cast<void*>(src_lock_and_addr.second),
                                  copy.bytes, copy.ranges, /*to_host=*/false))
            << "KV restore blit failed: " << name;
      }
      return absl::OkStatus();
    };
    ABSL_RETURN_IF_ERROR(restore_bank(bank_1_state_buffers_, other.bank_1_state_buffers_));
    if (bank_2_state_buffers_.has_value()) {
      ABSL_RETURN_IF_ERROR(restore_bank(*bank_2_state_buffers_, *other.bank_2_state_buffers_));
    }
    bank_1_is_input_ = other.bank_1_is_input_;
    return absl::OkStatus();
  }

  const bool raw = other.raw_metal_snapshot_;
  const std::optional<int> positions = other.snapshot_positions_;
  auto copy_bank = [raw, positions](absl::flat_hash_map<std::string, StateBuffer>& dst,
                                    const absl::flat_hash_map<std::string, StateBuffer>& src)
      -> absl::Status {
    RET_CHECK_EQ(dst.size(), src.size());
    for (auto& [name, state_buffer] : dst) {
      auto it = src.find(name);
      RET_CHECK(it != src.end()) << "Missing state buffer: " << name;
      if (raw) {
        // Agello (T3): a raw snapshot goes back byte for byte into the same kind of buffer.
        std::optional<MetalRawBuffer> storage = RawMetalStorage(state_buffer.buffer);
        RET_CHECK(storage.has_value()) << "Raw snapshot restored into another buffer kind: "
                                       << name;
        LITERT_ASSIGN_OR_RETURN(size_t src_size, it->second.buffer.Size());
        RET_CHECK_EQ(src_size, storage->length);
        LITERT_ASSIGN_OR_RETURN(auto src_lock_and_addr,
                                TensorBufferScopedLock::Create(
                                    it->second.buffer, TensorBuffer::LockMode::kRead));
        // Agello (E4): a snapshot of the filled slots puts back only those.
        std::optional<std::vector<ByteRun>> runs;
        if (positions.has_value()) {
          runs = FilledRuns(name, state_buffer.buffer, storage->length, *positions);
        }
        CopyRuns(storage->contents, src_lock_and_addr.second, storage->length,
                 runs.has_value() ? &*runs : nullptr);
        continue;
      }
      ABSL_RETURN_IF_ERROR(
          CopyBufferContents(state_buffer.buffer, it->second.buffer));
    }
    return absl::OkStatus();
  };

  ABSL_RETURN_IF_ERROR(
      copy_bank(bank_1_state_buffers_, other.bank_1_state_buffers_));
  if (bank_2_state_buffers_.has_value()) {
    ABSL_RETURN_IF_ERROR(
        copy_bank(*bank_2_state_buffers_, *other.bank_2_state_buffers_));
  }
  bank_1_is_input_ = other.bank_1_is_input_;
  return absl::OkStatus();
}

bool LitertState::Contains(absl::string_view tensor_name) const {
  return bank_1_state_buffers_.contains(tensor_name);
}

absl::Status LitertState::Clear() {
  for (auto& [_, state_buffer] : bank_1_state_buffers_) {
    LITERT_RETURN_IF_ERROR(state_buffer.buffer.Clear());
    ABSL_RETURN_IF_ERROR(PoisonIfAsked(state_buffer.buffer));
  }
  if (bank_2_state_buffers_.has_value()) {
    for (auto& [_, state_buffer] : *bank_2_state_buffers_) {
      LITERT_RETURN_IF_ERROR(state_buffer.buffer.Clear());
    }
  }
  return absl::OkStatus();
}

absl::Status LitertState::Resize(CompiledModel& compiled_model,
                                 absl::string_view signature_name,
                                 int num_entries) {
  RET_CHECK(!bank_2_state_buffers_.has_value())
          .SetCode(absl::StatusCode::kInvalidArgument)
      << "Out of place KV cache cannot be resized.";

  bool has_dynamic = false;
  for (const auto& [input_name, state_buffer] : bank_1_state_buffers_) {
    if (state_buffer.dynamic_dim.has_value()) {
      has_dynamic = true;
      break;
    }
  }
  if (!has_dynamic) {
    return absl::InvalidArgumentError(
        "KV cache is not dynamic and cannot be resized.");
  }

  int entries_to_add = num_entries - num_entries_;
  if (entries_to_add <= 0) {
    return absl::OkStatus();
  }

  for (const auto& [input_name, state_buffer] : bank_1_state_buffers_) {
    if (state_buffer.dynamic_dim.has_value()) {
      ABSL_RETURN_IF_ERROR(ResolveDynamicShape(compiled_model, signature_name,
                                               input_name, num_entries));
    }
  }

  for (auto& [input_name, state_buffer] : bank_1_state_buffers_) {
    if (state_buffer.dynamic_dim.has_value()) {
      LITERT_ASSIGN_OR_RETURN(
          state_buffer.buffer,
          ResizeTensorBuffer(env_, state_buffer.buffer,
                             state_buffer.dynamic_dim.value(), entries_to_add));
    }
  }
  num_entries_ = num_entries;
  return absl::OkStatus();
}

absl::StatusOr<LitertState::StateBuffers> LitertState::GetStateBuffers(
    CompiledModel& compiled_model, absl::string_view signature_name) {
  LITERT_RETURN_IF_ERROR(SyncShapes(compiled_model, signature_name));
  auto* input_bank = &bank_1_state_buffers_;
  auto* output_bank = &bank_1_state_buffers_;

  if (bank_2_state_buffers_.has_value()) {
    if (bank_1_is_input_) {
      output_bank = &bank_2_state_buffers_.value();
    } else {
      input_bank = &bank_2_state_buffers_.value();
    }
    bank_1_is_input_ = !bank_1_is_input_;
  }

  StateBuffers buffers;
  const bool should_skip_inputs =
      allocation_policy_ == AllocationPolicy::kGpuOptimizedInplace;
  const bool is_prefill = absl::StartsWith(signature_name, "prefill");

  for (const auto& [input_name, state_buffer] : *input_bank) {
    const bool is_local_kv_cache =
        state_buffer.type == proto::StateBuffer::TYPE_LOCAL_KEY_CACHE ||
        state_buffer.type == proto::StateBuffer::TYPE_LOCAL_VALUE_CACHE;
    const bool is_linear_attention =
        state_buffer.type == proto::StateBuffer::TYPE_LINEAR_ATTENTION ||
        IsLinearAttentionStateName(input_name);
    if (should_skip_inputs && !is_linear_attention &&
        (!is_prefill || !is_local_kv_cache)) {
      // For GPU optimized in place updates, we are required to pass the local
      // KV cache buffers as inputs too in the prefill stage, and linear
      // attention / conv state buffers in both prefill and decode.
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(auto duplicated, state_buffer.buffer.Duplicate());
    buffers.input_buffers[input_name] = std::move(duplicated);
  }

  for (const auto& [input_name, state_buffer] : *output_bank) {
    LITERT_ASSIGN_OR_RETURN(auto duplicated, state_buffer.buffer.Duplicate());
    buffers.output_buffers[input_name] = std::move(duplicated);
  }
  return buffers;
}

absl::Status LitertState::SyncShapes(CompiledModel& compiled_model,
                                     absl::string_view signature_name) {
  for (const auto& [input_name, state_buffer] : bank_1_state_buffers_) {
    if (state_buffer.dynamic_dim.has_value()) {
      ABSL_RETURN_IF_ERROR(ResolveDynamicShape(compiled_model, signature_name,
                                               input_name, num_entries_));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<LitertState>> LitertState::HeuristicBasedCreate(
    Environment& env, CompiledModel& compiled_model,
    absl::string_view signature_name, AllocationPolicy allocation_policy,
    int batch_size, bool clear_kv_cache_before_prefill) {
  LITERT_ASSIGN_OR_RETURN(
      auto sig_input_names,
      compiled_model.GetSignatureInputNames(signature_name));
  LITERT_ASSIGN_OR_RETURN(
      auto sig_output_names,
      compiled_model.GetSignatureOutputNames(signature_name));

  std::string k_root_name;
  std::string v_root_name;
  ABSL_RETURN_IF_ERROR(GetKVCacheRootNames(sig_input_names, sig_output_names,
                                           k_root_name, v_root_name));

  std::vector<std::string> key_cache_input_names;
  std::vector<std::string> value_cache_input_names;
  std::string mask_input_name;
  for (auto input_name : sig_input_names) {
    if (absl::StartsWith(input_name, k_root_name) ||
        absl::StartsWith(input_name, "kv_cache_c_")) {
      key_cache_input_names.push_back(std::string(input_name));
    }
    if (absl::StartsWith(input_name, v_root_name) ||
        absl::StartsWith(input_name, "kv_cache_c_")) {
      value_cache_input_names.push_back(std::string(input_name));
    }
    if (absl::StrContains(input_name, "mask")) {
      mask_input_name = input_name;
    }
  }
  if (key_cache_input_names.empty() || value_cache_input_names.empty()) {
    return absl::FailedPreconditionError("No KV cache inputs found.");
  }

  ABSL_ASSIGN_OR_RETURN(std::optional<int> k_dynamic_dim,
                        GetDynamicDimIndex(compiled_model, signature_name,
                                           key_cache_input_names[0]));
  ABSL_ASSIGN_OR_RETURN(std::optional<int> v_dynamic_dim,
                        GetDynamicDimIndex(compiled_model, signature_name,
                                           value_cache_input_names[0]));
  RET_CHECK(k_dynamic_dim.has_value() == v_dynamic_dim.has_value());

  auto create_and_init_buffer =
      [&](absl::string_view input_name, proto::StateBuffer::Type type,
          const std::optional<int>& dynamic_dim, bool is_output,
          bool clear_buffer) -> absl::StatusOr<StateBuffer> {
    if (dynamic_dim.has_value()) {
      ABSL_RETURN_IF_ERROR(ResolveDynamicShape(compiled_model, signature_name,
                                               input_name, /*new_value=*/1));
    }

    // LiteRT macros don't support trinary operators. Hence the following block.
    std::optional<TensorBuffer> buffer;
    if (is_output) {
      LITERT_ASSIGN_OR_RETURN(buffer, compiled_model.CreateOutputBuffer(
                                          signature_name, input_name));
    } else {
      LITERT_ASSIGN_OR_RETURN(
          buffer, compiled_model.CreateInputBuffer(signature_name, input_name));
    }

    if (clear_buffer && !SparseKvCandidate(input_name, *buffer)) {
      LITERT_RETURN_IF_ERROR(buffer->Clear());
      ABSL_RETURN_IF_ERROR(PoisonIfAsked(*buffer));
    }
    return StateBuffer{
        .buffer = std::move(buffer.value()),
        .type = type,
        .dynamic_dim = dynamic_dim,
    };
  };

  const bool is_gpu_optimized =
      allocation_policy == AllocationPolicy::kGpuOptimizedInplace;
  absl::flat_hash_map<std::string, StateBuffer> bank_1_state_buffers;

  for (const auto& input_name : key_cache_input_names) {
    LITERT_ASSIGN_OR_RETURN(
        bank_1_state_buffers[input_name],
        create_and_init_buffer(input_name, proto::StateBuffer::TYPE_UNSPECIFIED,
                               k_dynamic_dim, /*is_output=*/is_gpu_optimized,
                               /*clear_buffer=*/clear_kv_cache_before_prefill));
  }
  for (const auto& input_name : value_cache_input_names) {
    LITERT_ASSIGN_OR_RETURN(
        bank_1_state_buffers[input_name],
        create_and_init_buffer(input_name, proto::StateBuffer::TYPE_UNSPECIFIED,
                               v_dynamic_dim, /*is_output=*/is_gpu_optimized,
                               /*clear_buffer=*/clear_kv_cache_before_prefill));
  }

  std::optional<absl::flat_hash_map<std::string, StateBuffer>>
      bank_2_state_buffers;
  if (allocation_policy == AllocationPolicy::kPingPong) {
    auto& bank_2 = bank_2_state_buffers.emplace();
    for (const auto& input_name : key_cache_input_names) {
      LITERT_ASSIGN_OR_RETURN(
          bank_2[input_name],
          create_and_init_buffer(
              input_name, proto::StateBuffer::TYPE_UNSPECIFIED,
              /*dynamic_dim=*/std::nullopt, /*is_output=*/true,
              /*clear_buffer=*/false));
    }
    for (const auto& input_name : value_cache_input_names) {
      LITERT_ASSIGN_OR_RETURN(
          bank_2[input_name],
          create_and_init_buffer(
              input_name, proto::StateBuffer::TYPE_UNSPECIFIED,
              /*dynamic_dim=*/std::nullopt, /*is_output=*/true,
              /*clear_buffer=*/false));
    }
  }

  int context_size = 1;
  const bool is_dynamic_kv_cache = k_dynamic_dim.has_value();
  if (!is_dynamic_kv_cache) {
    if (!mask_input_name.empty()) {
      // Mask is our best bet for inferring context size. Key and value tensors
      // have different layouts and as such cannot be used directly.
      LITERT_ASSIGN_OR_RETURN(
          const RankedTensorType mask_tensor_type,
          compiled_model.GetInputTensorType(signature_name, mask_input_name));
      auto dims = mask_tensor_type.Layout().Dimensions();
      // Expect [1, 1, Sequence, KV Length]
      RET_CHECK_EQ(dims.size(), 4);
      context_size = dims[3];
    } else {
      // Fallback: get capacity from key cache tensor layout.
      // Usually key cache layout is [batch, num_heads, sequence_length,
      // head_dim]
      LITERT_ASSIGN_OR_RETURN(const RankedTensorType k_tensor_type,
                              compiled_model.GetInputTensorType(
                                  signature_name, key_cache_input_names[0]));
      auto dims = k_tensor_type.Layout().Dimensions();
      RET_CHECK_GE(dims.size(), 3);
      context_size = dims[2];
    }
  }

  return absl::WrapUnique(new LitertState(
      batch_size, context_size, env, std::move(bank_1_state_buffers),
      std::move(bank_2_state_buffers), allocation_policy));
}

absl::StatusOr<std::unique_ptr<LitertState>> LitertState::MetadataBasedCreate(
    Environment& env, CompiledModel& compiled_model,
    absl::string_view signature_name,
    const proto::ExecutorMetadata& executor_metadata,
    AllocationPolicy allocation_policy, int batch_size,
    bool clear_kv_cache_before_prefill) {
  struct StateBufferMeta {
    std::string name;
    proto::StateBuffer::Type type;
    std::optional<int> dynamic_dim;
    bool is_global;
  };

  std::vector<StateBufferMeta> state_buffer_metas;
  std::optional<int> max_supported_sequence_size;

  for (const auto& state_buffer :
       executor_metadata.llm_executor_metadata().state_buffers()) {
    std::vector<absl::string_view> names;
    if (!state_buffer.prefill_input_name().empty()) {
      names.push_back(state_buffer.prefill_input_name());
    }
    if (!state_buffer.prefill_output_name().empty()) {
      names.push_back(state_buffer.prefill_output_name());
    }
    if (!state_buffer.decode_input_name().empty()) {
      names.push_back(state_buffer.decode_input_name());
    }
    if (!state_buffer.decode_output_name().empty()) {
      names.push_back(state_buffer.decode_output_name());
    }
    for (size_t i = 1; i < names.size(); ++i) {
      RET_CHECK_EQ(names[0], names[i])
          << "Current implementation requires all state names in StateBuffer"
             " to be the same: "
          << names[0] << " vs " << names[i];
    }

    RET_CHECK(!names.empty()) << "At least one state name must be defined";
    std::string input_name = std::string(names[0]);

    bool is_global = false;
    switch (state_buffer.type()) {
      case proto::StateBuffer::TYPE_GLOBAL_KEY_CACHE:
      case proto::StateBuffer::TYPE_GLOBAL_VALUE_CACHE:
        is_global = true;
        break;
      case proto::StateBuffer::TYPE_LOCAL_KEY_CACHE:
      case proto::StateBuffer::TYPE_LOCAL_VALUE_CACHE:
      case proto::StateBuffer::TYPE_LINEAR_ATTENTION:
        break;
      default:
        return absl::InvalidArgumentError(absl::StrCat(
            "Unsupported state buffer type: ", state_buffer.type()));
    }

    LITERT_ASSIGN_OR_RETURN(
        const RankedTensorType ranked_tensor_type,
        compiled_model.GetInputTensorType(signature_name, input_name));
    auto dimensions = ranked_tensor_type.Layout().Dimensions();

    std::optional<int> dynamic_dim;
    if (state_buffer.has_sequence_axis()) {
      int axis = state_buffer.sequence_axis();
      RET_CHECK_GE(axis, 0);
      RET_CHECK_LT(axis, dimensions.size());
      if (dimensions[axis] == kDynamicDimValue) {
        dynamic_dim = axis;
      } else if (is_global) {
        int seq_size = dimensions[axis];
        if (max_supported_sequence_size.has_value()) {
          max_supported_sequence_size =
              std::min(*max_supported_sequence_size, seq_size);
        } else {
          max_supported_sequence_size = seq_size;
        }
      }
    } else {
      RET_CHECK(state_buffer.type() ==
                proto::StateBuffer::TYPE_LINEAR_ATTENTION)
          << "Sequence axis must be defined for state buffers in the current "
             "implementation.";
    }

    state_buffer_metas.push_back(StateBufferMeta{
        .name = std::move(input_name),
        .type = state_buffer.type(),
        .dynamic_dim = dynamic_dim,
        .is_global = is_global,
    });
  }

  if (state_buffer_metas.empty()) {
    return absl::InvalidArgumentError(
        "No state buffers found for the current signature");
  }

  auto create_and_init_buffer =
      [&](absl::string_view input_name, proto::StateBuffer::Type type,
          const std::optional<int>& dynamic_dim, bool is_output,
          bool clear_buffer) -> absl::StatusOr<StateBuffer> {
    if (dynamic_dim.has_value()) {
      ABSL_RETURN_IF_ERROR(ResolveDynamicShape(compiled_model, signature_name,
                                               input_name, /*new_value=*/1));
    }

    // LiteRT macros don't support trinary operators. Hence the following block.
    std::optional<TensorBuffer> buffer;
    if (is_output) {
      LITERT_ASSIGN_OR_RETURN(buffer, compiled_model.CreateOutputBuffer(
                                          signature_name, input_name));
    } else {
      LITERT_ASSIGN_OR_RETURN(
          buffer, compiled_model.CreateInputBuffer(signature_name, input_name));
    }
    if (clear_buffer && !SparseKvCandidate(input_name, *buffer)) {
      LITERT_RETURN_IF_ERROR(buffer->Clear());
      ABSL_RETURN_IF_ERROR(PoisonIfAsked(*buffer));
    }
    return StateBuffer{
        .buffer = std::move(buffer.value()),
        .type = type,
        .dynamic_dim = dynamic_dim,
    };
  };

  const bool is_gpu_optimized =
      allocation_policy == AllocationPolicy::kGpuOptimizedInplace;
  absl::flat_hash_map<std::string, StateBuffer> bank_1_state_buffers;
  for (const auto& meta : state_buffer_metas) {
    LITERT_ASSIGN_OR_RETURN(
        bank_1_state_buffers[meta.name],
        create_and_init_buffer(meta.name, meta.type, meta.dynamic_dim,
                               /*is_output=*/is_gpu_optimized,
                               /*clear_buffer=*/clear_kv_cache_before_prefill));
  }

  std::optional<absl::flat_hash_map<std::string, StateBuffer>>
      bank_2_state_buffers;
  if (allocation_policy == AllocationPolicy::kPingPong) {
    auto& bank_2 = bank_2_state_buffers.emplace();
    for (const auto& meta : state_buffer_metas) {
      LITERT_ASSIGN_OR_RETURN(
          bank_2[meta.name],
          create_and_init_buffer(meta.name, meta.type,
                                 /*dynamic_dim=*/std::nullopt,
                                 /*is_output=*/true,
                                 /*clear_buffer=*/false));
    }
  }

  int context_size = 1;
  bool is_dynamic = false;
  for (const auto& meta : state_buffer_metas) {
    if (meta.dynamic_dim.has_value()) {
      is_dynamic = true;
      break;
    }
  }
  if (!is_dynamic) {
    if (max_supported_sequence_size.has_value()) {
      context_size = *max_supported_sequence_size;
    } else {
      context_size = std::numeric_limits<int>::max();
    }
  }

  return absl::WrapUnique(new LitertState(
      batch_size, context_size, env, std::move(bank_1_state_buffers),
      std::move(bank_2_state_buffers), allocation_policy));
}

namespace {

// Agello (ring rewind): a KV cache's slots (the A of a K cache, the C of a V cache; see
// FilledRuns for the layout), or nothing for any other buffer.
std::optional<int> CacheSlots(absl::string_view name, const TensorBuffer& buffer) {
  const bool is_k = absl::StartsWith(name, "kv_cache_k_");
  const bool is_v = absl::StartsWith(name, "kv_cache_v_");
  if (!is_k && !is_v) return std::nullopt;
  auto tensor_type = buffer.TensorType();
  if (!tensor_type) return std::nullopt;
  auto dims = tensor_type->Layout().Dimensions();
  if (dims.size() != 4 || dims[0] != 1) return std::nullopt;
  return static_cast<int>(is_k ? dims[2] : dims[3]);
}

// The slot a byte of a ring cache belongs to: element ((y * slices + c / 4) * A + a) * 4 + c % 4
// of a [1, heads, A, C] tensor, the slot being a for K and c for V.
int SlotOfByte(const LitertState::RingBuffer& ring, size_t offset) {
  const size_t element = offset / ring.width;
  const size_t lane = element % 4;
  const size_t rest = element / 4;
  const size_t a = rest % ring.a_dim;
  const size_t group = (rest / ring.a_dim) % ring.slices;
  return static_cast<int>(ring.is_k ? a : group * 4 + lane);
}

}  // namespace

int LitertState::RingSlots() const {
  int largest = 0, smallest = 0;
  for (const auto& [name, state_buffer] : bank_1_state_buffers_) {
    std::optional<int> slots = CacheSlots(name, state_buffer.buffer);
    if (!slots.has_value()) continue;
    largest = std::max(largest, *slots);
    smallest = smallest == 0 ? *slots : std::min(smallest, *slots);
  }
  return smallest < largest ? smallest : 0;
}

absl::StatusOr<LitertState::RingCopy> LitertState::CopyRings() const {
  RingCopy copy;
  const int ring = RingSlots();
  if (ring == 0) return copy;
  auto copy_bank = [&](const absl::flat_hash_map<std::string, StateBuffer>& bank) -> bool {
    for (const auto& [name, state_buffer] : bank) {
      std::optional<int> slots = CacheSlots(name, state_buffer.buffer);
      if (!slots.has_value() || *slots != ring) continue;
      std::optional<MetalRawBuffer> storage = RawMetalStorage(state_buffer.buffer);
      auto tensor_type = state_buffer.buffer.TensorType();
      if (!storage.has_value() || !tensor_type) return false;
      std::optional<size_t> width = GetByteWidth(tensor_type->ElementType());
      if (!width.has_value()) return false;
      auto dims = tensor_type->Layout().Dimensions();
      const auto* bytes = static_cast<const uint8_t*>(storage->contents);
      copy.buffers.push_back(RingBuffer{
          .name = name,
          .bytes = std::vector<uint8_t>(bytes, bytes + storage->length),
          .is_k = absl::StartsWith(name, "kv_cache_k_"),
          .a_dim = static_cast<size_t>(dims[2]),
          .slices = static_cast<size_t>((dims[3] + 3) / 4),
          .width = *width,
      });
      copy.bytes += static_cast<int64_t>(storage->length);
    }
    return true;
  };
  if (!copy_bank(bank_1_state_buffers_) ||
      (bank_2_state_buffers_.has_value() && !copy_bank(*bank_2_state_buffers_))) {
    return RingCopy{};
  }
  return copy;
}

absl::Status LitertState::RestoreRings(const RingCopy& copy) {
  for (const RingBuffer& ring : copy.buffers) {
    auto find = [&](absl::flat_hash_map<std::string, StateBuffer>& bank) -> StateBuffer* {
      auto it = bank.find(ring.name);
      return it == bank.end() ? nullptr : &it->second;
    };
    StateBuffer* state_buffer = find(bank_1_state_buffers_);
    if (state_buffer == nullptr && bank_2_state_buffers_.has_value()) {
      state_buffer = find(*bank_2_state_buffers_);
    }
    RET_CHECK(state_buffer != nullptr) << "Missing ring cache: " << ring.name;
    std::optional<MetalRawBuffer> storage = RawMetalStorage(state_buffer->buffer);
    RET_CHECK(storage.has_value()) << "Ring cache isn't a shared Metal buffer: " << ring.name;
    RET_CHECK_EQ(storage->length, ring.bytes.size()) << ring.name;
    memcpy(storage->contents, ring.bytes.data(), ring.bytes.size());
  }
  return absl::OkStatus();
}

int64_t LitertState::DifferentWindowBytes(const RingCopy& a, const RingCopy& b,
                                          const std::vector<int>& window_slots, int ring_slots,
                                          std::vector<std::pair<int, int64_t>>* per_slot) {
  std::vector<bool> in_window(ring_slots, false);
  for (int slot : window_slots) {
    if (slot >= 0 && slot < ring_slots) in_window[slot] = true;
  }
  std::vector<int64_t> counts(2 * ring_slots, 0);  // K caches' counts, then V caches'
  int64_t different = 0;
  for (const RingBuffer& x : a.buffers) {
    for (const RingBuffer& y : b.buffers) {
      if (x.name != y.name || x.bytes.size() != y.bytes.size()) continue;
      for (size_t i = 0; i < x.bytes.size(); ++i) {
        if (x.bytes[i] == y.bytes[i]) continue;
        const int slot = SlotOfByte(x, i);
        if (!in_window[slot]) continue;
        ++different;
        ++counts[x.is_k ? slot : ring_slots + slot];
      }
    }
  }
  if (per_slot != nullptr) {
    for (int slot = 0; slot < 2 * ring_slots; ++slot) {
      if (counts[slot] > 0) per_slot->emplace_back(slot, counts[slot]);
    }
  }
  return different;
}

}  // namespace litert::lm
