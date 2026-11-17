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

#include "runtime/executor/llm_litert_mtp_drafter.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model_types.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/litert_logit_mask_runner.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/sampler.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert/state.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_executor_settings_utils.h"
#include "runtime/executor/state_interface.h"
#include <cstring>
#include <variant>

#include "litert/c/litert_model.h"  // from @litert
#include "litert/c/litert_op_code.h"  // from @litert
#include "litert/cc/litert_environment_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"
#include "agello/agello_telemetry.h"  // from @agello_telemetry

namespace litert::lm {

namespace {

constexpr bool kEnableMtpDrafterLogs = false;

// Agello, the replay test (engine review step 7): AGELLO_MTP_REPLAY=1 runs every round twice.
// After the real round, the same round again: its inputs written as before, its three drafter
// runs and its verify committed back to back, one wait at the end, no logits read and no picks.
// That is the ceiling of a GPU-resident round, which would still pay that host preparation. The
// replay writes what the real round wrote (same inputs, same positions), so replies are unchanged.
// Telemetry: mtp.round (the real round) against mtp.replay; gpu.replay is the replay's GPU time.
// Where the replay waits (the resident round's versions): 1, once at the end (the whole round
// chained); a, also before verify (a drafter chain); b, as a with verify's inputs written in that
// gap (the drafter chain's ceiling: the host reads the drafts, looks up verify's embeddings,
// commits); c, after each drafter run, verify chained; e, the drafter chain with verify encoded
// and committed behind an event, released once the drafts are back and verify's inputs written
// (the per-layer tables stay on the CPU).
char MtpReplayMode() {
  static const char mode = [] {
    const char* value = std::getenv("AGELLO_MTP_REPLAY");
    if (value == nullptr || value[0] == '\0' || value[0] == '0') return '\0';
    return value[0] == 'a' || value[0] == 'b' || value[0] == 'c' || value[0] == 'e' ? value[0]
                                                                                     : '1';
  }();
  return mode;
}

// Restores waiting even when a replayed run fails, then waits for what was committed.
// Agello (the GPU-resident round): the environment's Metal command queue, the one the GPU's runs
// use, or nullptr (as state.cc's MetalCommandQueueOf).
const void* EnvironmentMetalQueue(const Environment& env) {
  auto options = env.GetOptions();
  if (!options) return nullptr;
  auto queue = options->GetOption(EnvironmentOptions::Tag::kMetalCommandQueue);
  if (!queue) return nullptr;
  if (const auto* pointer = std::get_if<void*>(&queue.Value())) return *pointer;
  if (const auto* pointer = std::get_if<const void*>(&queue.Value())) return *pointer;
  return nullptr;
}

// Agello (the GPU-resident round): the embedder's INT2 table, scales and multiplier, read from its
// graph (one EMBEDDING_LOOKUP, per-row scales, zero points 0, then a MUL by a constant), and the
// GPU side built over the mapped section. nullptr when anything differs: the host drafts.
std::unique_ptr<ResidentRound> CreateResidentRound(Environment& env, ModelResources& resources,
                                                   int steps) {
  const void* queue = EnvironmentMetalQueue(env);
  auto buffer = resources.GetTFLiteModelBuffer(ModelType::kTfLiteEmbedder);
  auto model = resources.GetTFLiteModel(ModelType::kTfLiteEmbedder);
  if (queue == nullptr || !buffer.ok() || !model.ok() || *model == nullptr) {
    ABSL_LOG(INFO) << "Agello resident round: no Metal queue or embedder; the host drafts.";
    return nullptr;
  }
  ResidentRound::Table table;
  table.section = buffer->data();
  table.section_bytes = buffer->size();
  LiteRtModel handle = (*model)->Get();
  LiteRtParamIndex main_index = 0;
  LiteRtSubgraph subgraph = nullptr;
  LiteRtParamIndex num_ops = 0;
  if (LiteRtGetMainModelSubgraphIndex(handle, &main_index) != kLiteRtStatusOk ||
      LiteRtGetModelSubgraph(handle, main_index, &subgraph) != kLiteRtStatusOk ||
      LiteRtGetNumSubgraphOps(subgraph, &num_ops) != kLiteRtStatusOk) {
    return nullptr;
  }
  int lookups = 0;
  auto weights_of = [](LiteRtOp op, LiteRtParamIndex input, const void** addr,
                       size_t* size) -> bool {
    LiteRtTensor tensor = nullptr;
    LiteRtWeights weights = nullptr;
    return LiteRtGetOpInput(op, input, &tensor) == kLiteRtStatusOk &&
           LiteRtGetTensorWeights(tensor, &weights) == kLiteRtStatusOk &&
           LiteRtGetWeightsBytes(weights, addr, size) == kLiteRtStatusOk;
  };
  for (LiteRtParamIndex i = 0; i < num_ops; ++i) {
    LiteRtOp op = nullptr;
    LiteRtOpCode code;
    if (LiteRtGetSubgraphOp(subgraph, i, &op) != kLiteRtStatusOk ||
        LiteRtGetOpCode(op, &code) != kLiteRtStatusOk) {
      return nullptr;
    }
    if (code == kLiteRtOpCodeTflEmbeddingLookup) {
      ++lookups;
      const void* addr = nullptr;
      size_t size = 0;
      LiteRtTensor tensor = nullptr;
      LiteRtQuantizationPerChannel quantization;
      if (!weights_of(op, 1, &addr, &size) || LiteRtGetOpInput(op, 1, &tensor) != kLiteRtStatusOk ||
          LiteRtGetPerChannelQuantization(tensor, &quantization) != kLiteRtStatusOk ||
          quantization.num_channels <= 0 || quantization.scales == nullptr) {
        return nullptr;
      }
      for (int64_t row = 0; row < quantization.num_channels; ++row) {
        if (quantization.zero_points != nullptr && quantization.zero_points[row] != 0) return nullptr;
      }
      const auto* section = static_cast<const char*>(table.section);
      const auto* rows = static_cast<const char*>(addr);
      if (rows < section || rows + size > section + table.section_bytes ||
          (size * 4) % quantization.num_channels != 0) {
        ABSL_LOG(INFO) << "Agello resident round: the table isn't in the mapped section.";
        return nullptr;
      }
      table.rows_offset = static_cast<size_t>(rows - section);
      table.rows = static_cast<int>(quantization.num_channels);
      table.cols = static_cast<int>(size * 4 / quantization.num_channels);
      table.scales = quantization.scales;
    } else if (code == kLiteRtOpCodeTflMul) {
      const void* addr = nullptr;
      size_t size = 0;
      if (!weights_of(op, 1, &addr, &size) || size != sizeof(float)) return nullptr;
      std::memcpy(&table.mul, addr, sizeof(float));
    }
  }
  if (lookups != 1) return nullptr;
  return ResidentRound::Create(queue, table, steps);
}

// Agello (the GPU-resident round, stage 3): an embedder section's tables with their scales found
// in the mapped section (every EMBEDDING_LOOKUP in op order, which is the concatenation's), and
// its MUL constant. False when a table or its scales lie outside the section.
bool ReadTableSet(ModelResources& resources, ModelType type, ResidentRound::TableSet* set) {
  auto buffer = resources.GetTFLiteModelBuffer(type);
  auto model = resources.GetTFLiteModel(type);
  if (!buffer.ok() || !model.ok() || *model == nullptr) return false;
  set->section = buffer->data();
  set->section_bytes = buffer->size();
  const auto* section = static_cast<const char*>(set->section);
  LiteRtModel handle = (*model)->Get();
  LiteRtParamIndex main_index = 0;
  LiteRtSubgraph subgraph = nullptr;
  LiteRtParamIndex num_ops = 0;
  if (LiteRtGetMainModelSubgraphIndex(handle, &main_index) != kLiteRtStatusOk ||
      LiteRtGetModelSubgraph(handle, main_index, &subgraph) != kLiteRtStatusOk ||
      LiteRtGetNumSubgraphOps(subgraph, &num_ops) != kLiteRtStatusOk) {
    return false;
  }
  for (LiteRtParamIndex i = 0; i < num_ops; ++i) {
    LiteRtOp op = nullptr;
    LiteRtOpCode code;
    LiteRtTensor tensor = nullptr;
    LiteRtWeights weights = nullptr;
    const void* addr = nullptr;
    size_t size = 0;
    if (LiteRtGetSubgraphOp(subgraph, i, &op) != kLiteRtStatusOk ||
        LiteRtGetOpCode(op, &code) != kLiteRtStatusOk) {
      return false;
    }
    if (code != kLiteRtOpCodeTflEmbeddingLookup && code != kLiteRtOpCodeTflMul) continue;
    if (LiteRtGetOpInput(op, 1, &tensor) != kLiteRtStatusOk ||
        LiteRtGetTensorWeights(tensor, &weights) != kLiteRtStatusOk ||
        LiteRtGetWeightsBytes(weights, &addr, &size) != kLiteRtStatusOk) {
      return false;
    }
    if (code == kLiteRtOpCodeTflMul) {
      if (size != sizeof(float)) return false;
      std::memcpy(&set->mul, addr, sizeof(float));
      continue;
    }
    LiteRtQuantizationPerChannel quantization;
    if (LiteRtGetPerChannelQuantization(tensor, &quantization) != kLiteRtStatusOk ||
        quantization.num_channels <= 0 || quantization.scales == nullptr) {
      return false;
    }
    const auto* rows = static_cast<const char*>(addr);
    const auto* scales = reinterpret_cast<const char*>(quantization.scales);
    const size_t scale_bytes = static_cast<size_t>(quantization.num_channels) * sizeof(float);
    if (rows < section || rows + size > section + set->section_bytes || scales < section ||
        scales + scale_bytes > section + set->section_bytes ||
        (size * 4) % quantization.num_channels != 0) {
      return false;
    }
    for (int64_t row = 0; row < quantization.num_channels; ++row) {
      if (quantization.zero_points != nullptr && quantization.zero_points[row] != 0) return false;
    }
    const int rows_count = static_cast<int>(quantization.num_channels);
    const int cols = static_cast<int>(size * 4 / quantization.num_channels);
    if (!set->tables.empty() && (rows_count != set->rows || cols != set->cols)) return false;
    set->rows = rows_count;
    set->cols = cols;
    set->tables.emplace_back(static_cast<size_t>(rows - section), static_cast<size_t>(scales - section));
  }
  return !set->tables.empty();
}

// Agello (the GPU-resident round): whether a round needs no constraint masks: no constraint, no
// state (the host loop masks nothing then), or a state that allows any token for the round's
// `steps` drafts.
bool RoundNeedsNoMasks(const Constraint* constraint, const Constraint::State* state, int steps) {
  return constraint == nullptr || state == nullptr || constraint->AllowsAnyTokens(*state, steps);
}

// The drafts' constraint states, as the host's drafting loop builds them, for drafts the GPU
// picked.
absl::Status ChainDraftConstraintStates(
    const Constraint* constraint, const Constraint::State* state, const std::vector<int>& drafts,
    std::vector<std::unique_ptr<Constraint::State>>& states) {
  if (constraint == nullptr || state == nullptr) return absl::OkStatus();
  for (int draft : drafts) {
    ABSL_ASSIGN_OR_RETURN(auto next, constraint->ComputeNext(*state, draft));
    if (constraint->IsEnded(*next)) next = constraint->Start();
    states.push_back(std::move(next));
    state = states.back().get();
  }
  return absl::OkStatus();
}

struct DeferredGpuWaits {
  DeferredGpuWaits() { ::agello::telemetry::DeferGpuWaits(true); }
  ~DeferredGpuWaits() {
    ::agello::telemetry::DeferGpuWaits(false);
    ::agello::telemetry::WaitForDeferredGpuWork();
  }
  DeferredGpuWaits(const DeferredGpuWaits&) = delete;
  DeferredGpuWaits& operator=(const DeferredGpuWaits&) = delete;
};

#define MTP_DRAFTER_LOG() \
  ABSL_LOG_IF(INFO, kEnableMtpDrafterLogs) << "MTP Drafter - "

constexpr absl::string_view kVerifySignatureRunner = "verify";

absl::StatusOr<std::unique_ptr<Sampler>> CreateGreedySampler(
    const Environment& env, Backend backend, int output_heads,
    int sequence_size, int vocab_size,
    std::optional<ActivationDataType> activation_data_type) {
  proto::SamplerParameters sampler_params;
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  sampler_params.set_k(1);
  sampler_params.set_p(0.0f);
  sampler_params.set_temperature(1.0f);
  sampler_params.set_seed(0);
  return CreateSampler(backend, output_heads, std::move(sampler_params), env,
                       sequence_size, vocab_size, activation_data_type);
}

absl::StatusOr<TensorBuffer> CreateFP16OutputBuffer(
    Environment& env, CompiledModel& compiled_model, size_t signature_index,
    absl::string_view output_name, size_t output_index) {
  LITERT_ASSIGN_OR_RETURN(
      std::vector<Layout> runtime_layouts,
      compiled_model.GetOutputTensorLayouts(signature_index,
                                            /*update_allocation=*/true));
  // Use runtime layout.
  Layout runtime_layout = runtime_layouts[output_index];
  LITERT_ASSIGN_OR_RETURN(
      auto requirements,
      compiled_model.GetOutputBufferRequirements(signature_index, output_name));
  LITERT_ASSIGN_OR_RETURN(auto strides, requirements.Strides());
  if (!strides.empty()) {
    auto dims = runtime_layout.Dimensions();
    runtime_layout = Layout(litert::Dimensions(dims.begin(), dims.end()),
                            litert::Strides(strides.begin(), strides.end()));
  }
  RankedTensorType new_tensor_type(litert::ElementType::Float16,
                                   std::move(runtime_layout));
  LITERT_ASSIGN_OR_RETURN(size_t size, requirements.BufferSize());
  LITERT_ASSIGN_OR_RETURN(auto buffer_types, requirements.SupportedTypes());
  if (buffer_types.empty()) {
    return absl::InternalError("No supported buffer types found.");
  }
  auto buffer_type = buffer_types[0];
  LITERT_ASSIGN_OR_RETURN(
      auto buffer, TensorBuffer::CreateManaged(
                       env, buffer_type, std::move(new_tensor_type), size));
  return buffer;
}

absl::Status ConcatenateEmbeddingsAndActivations(
    const std::vector<float>& embedding_vector,
    TensorBuffer& projected_activations, TensorBuffer& output_activations) {
  size_t chunk_size_bytes = embedding_vector.size() * sizeof(float);
  LITERT_ASSIGN_OR_RETURN(
      auto output_activations_lock_and_addr,
      TensorBufferScopedLock::Create(output_activations,
                                     TensorBuffer::LockMode::kWrite));
  memcpy(static_cast<char*>(output_activations_lock_and_addr.second),
         embedding_vector.data(), chunk_size_bytes);

  LITERT_ASSIGN_OR_RETURN(
      auto projected_activations_lock_and_addr,
      TensorBufferScopedLock::Create(projected_activations,
                                     TensorBuffer::LockMode::kRead));
  memcpy(static_cast<char*>(output_activations_lock_and_addr.second) +
             chunk_size_bytes,
         static_cast<const char*>(projected_activations_lock_and_addr.second),
         chunk_size_bytes);
  return absl::OkStatus();
}

absl::Status ConcatenateEmbeddingsAndActivationsFromVerifierBuffer(
    const std::vector<float>& embedding_vector,
    TensorBuffer& verifier_projected_activations,
    int last_verified_token_id_idx, TensorBuffer& output_activations) {
  size_t chunk_size_bytes = embedding_vector.size() * sizeof(float);

  LITERT_ASSIGN_OR_RETURN(
      auto output_activations_lock_and_addr,
      TensorBufferScopedLock::Create(output_activations,
                                     TensorBuffer::LockMode::kWrite));
  memcpy(static_cast<char*>(output_activations_lock_and_addr.second),
         embedding_vector.data(), chunk_size_bytes);

  LITERT_ASSIGN_OR_RETURN(
      auto verifier_projected_activations_lock_and_addr,
      TensorBufferScopedLock::Create(verifier_projected_activations,
                                     TensorBuffer::LockMode::kRead));
  // Grab the last valid activation from verifier output buffer with shape
  // [batch, draft_steps + 1, hidden_size].
  // Offset is as such the last verified token id's index multiplied by the
  // hiden_size in bytes.
  size_t offset = last_verified_token_id_idx * chunk_size_bytes;
  memcpy(static_cast<char*>(output_activations_lock_and_addr.second) +
             chunk_size_bytes,
         static_cast<const char*>(
             verifier_projected_activations_lock_and_addr.second) +
             offset,
         chunk_size_bytes);
  return absl::OkStatus();
}

absl::StatusOr<int> GetVocabSizeFromLogitsTensor(TensorBuffer& logits_tensor) {
  LITERT_ASSIGN_OR_RETURN(auto logits_tensor_type, logits_tensor.TensorType());
  RET_CHECK_EQ(logits_tensor_type.Layout().Dimensions().size(), 3);
  // logits tensor shape is [batch, seq, vocab].
  return logits_tensor_type.Layout().Dimensions()[2];
}

}  // namespace

absl::Status UpdateCompilationOptions(
    const LlmExecutorSettings& executor_settings,
    litert::Options& compilation_options) {
  switch (executor_settings.GetBackend()) {
    case Backend::GPU: {
      LITERT_ASSIGN_OR_RETURN(auto& gpu_compilation_options,
                              compilation_options.GetGpuOptions());
      gpu_compilation_options.AddExternalTensorPattern("kv_cache_");
      gpu_compilation_options.AddBufferStorageTensorPattern("kv_cache_");
      gpu_compilation_options.AddExternalTensorPattern("param_tensor");
      gpu_compilation_options.AddBufferStorageTensorPattern("param_tensor");
      break;
    }
    case Backend::CPU: {
      break;
    }
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported backend: ", executor_settings.GetBackend()));
  }

  return absl::OkStatus();
}
LlmLiteRtMtpDrafter::~LlmLiteRtMtpDrafter() {
  ABSL_VLOG(1) << "Num drafted tokens: " << num_drafted_tokens_;
  ABSL_VLOG(1) << "Num verified tokens: " << num_verified_tokens_;
  if (num_drafted_tokens_ > 0) {
    ABSL_LOG(INFO) << "MTP Drafter - Success rate: "
                   << static_cast<double>(num_verified_tokens_) /
                          num_drafted_tokens_;
  }
}

absl::StatusOr<std::unique_ptr<LlmLiteRtMtpDrafter>>
LlmLiteRtMtpDrafter::Create(
    Environment& env, ModelResources& resources,
    const LlmExecutorSettings& executor_settings, CompiledModel& base_model,
    EmbeddingLookupManager& embedding_manager,
    std::optional<std::reference_wrapper<EmbeddingLookupManager>> ple_manager,
    const proto::ExecutorMetadata* executor_metadata) {
  ActivationDataType activation_data_type =
      executor_settings.GetActivationDataType().value_or(
          ActivationDataType::FLOAT16);
  auto cache_suffix = std::string(ExecutorSettingsBase::kMtpDrafterCacheSuffix);
  // Agello (PERFORMANCE.md, E3): the drafter compiles its own model, whose one
  // signature isn't among the main model's: a selection made for the main
  // graph (long windows leave out the 1,024-token prefill) doesn't apply here.
  LlmExecutorSettings drafter_settings = executor_settings;
  drafter_settings.SetSelectedSignatures({});
  ABSL_ASSIGN_OR_RETURN(
      auto compilation_options,
      CreateCompilationOptions(drafter_settings, activation_data_type,
                               /*signatures=*/std::nullopt,
                               /*cache_suffix=*/cache_suffix));
  ABSL_RETURN_IF_ERROR(
      UpdateCompilationOptions(drafter_settings, compilation_options));
  ABSL_RETURN_IF_ERROR(SetExternalWeightOptions(
      resources, ModelType::kTfLiteMtpDrafter, compilation_options));
  ABSL_ASSIGN_OR_RETURN(auto model,
                        resources.GetTFLiteModel(ModelType::kTfLiteMtpDrafter));
  LITERT_ASSIGN_OR_RETURN(
      auto compiled_model,
      CompiledModel::Create(env, model->Get(), compilation_options));
  ABSL_ASSIGN_OR_RETURN(
      auto base_model_desc,
      resources.GetTFLiteModel(ModelType::kTfLitePrefillDecode));

  ABSL_ASSIGN_OR_RETURN(
      auto drafter,
      Create(env, std::move(compiled_model), executor_settings, base_model,
             *base_model_desc, embedding_manager, ple_manager,
             executor_metadata));
  if (ResidentRoundEnabled() && executor_settings.GetBackend() == Backend::GPU) {
    drafter->resident_ = CreateResidentRound(env, resources, drafter->num_draft_steps_);
    ResidentRound::TableSet embedder, per_layer;
    if (drafter->resident_ != nullptr &&
        ReadTableSet(resources, ModelType::kTfLiteEmbedder, &embedder) &&
        ReadTableSet(resources, ModelType::kTfLitePerLayerEmbedder, &per_layer) &&
        embedder.tables.size() == 1) {
      drafter->resident_->AddVerifyTables(embedder, per_layer);
    }
  }
  return drafter;
}

absl::StatusOr<std::unique_ptr<LlmLiteRtMtpDrafter>>
LlmLiteRtMtpDrafter::Create(
    Environment& env, CompiledModel mtp_drafter_model,
    const LlmExecutorSettings& executor_settings, CompiledModel& base_model,
    const Model& base_model_desc, EmbeddingLookupManager& embedding_manager,
    std::optional<std::reference_wrapper<EmbeddingLookupManager>> ple_manager,
    const proto::ExecutorMetadata* executor_metadata) {
  ActivationDataType activation_data_type =
      executor_settings.GetActivationDataType().value_or(
          ActivationDataType::FLOAT16);
  const Backend backend = executor_settings.GetBackend();
  bool use_fp16_precision = backend == Backend::GPU &&
                            activation_data_type == ActivationDataType::FLOAT16;

  absl::flat_hash_map<absl::string_view, TensorBuffer>
      mtp_drafter_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      mtp_drafter_output_buffers;
  std::vector<std::string> kv_cache_input_names;
  LITERT_ASSIGN_OR_RETURN(
      SimpleSignature drafter_signature,
      mtp_drafter_model.GetSignature(/*signature_index=*/0));
  {
    for (absl::string_view input_name : drafter_signature.InputNames()) {
      if (absl::StartsWith(input_name, "kv_cache_")) {
        kv_cache_input_names.emplace_back(input_name);
        continue;
      }

      LITERT_ASSIGN_OR_RETURN(auto input_buffer,
                              mtp_drafter_model.CreateInputBuffer(
                                  drafter_signature.Key(), input_name));
      mtp_drafter_input_buffers[input_name] = std::move(input_buffer);
    }

    for (size_t i = 0; i < drafter_signature.OutputNames().size(); ++i) {
      absl::string_view output_name = drafter_signature.OutputNames()[i];
      if (output_name == "logits" && use_fp16_precision) {
        LITERT_ASSIGN_OR_RETURN(
            auto output_buffer,
            CreateFP16OutputBuffer(env, mtp_drafter_model,
                                   /*signature_index=*/0, output_name, i));
        mtp_drafter_output_buffers[output_name] = std::move(output_buffer);
      } else {
        LITERT_ASSIGN_OR_RETURN(auto output_buffer,
                                mtp_drafter_model.CreateOutputBuffer(
                                    drafter_signature.Key(), output_name));
        mtp_drafter_output_buffers[output_name] = std::move(output_buffer);
      }
    }
  }

  LITERT_ASSIGN_OR_RETURN(SimpleSignature verify_signature,
                          base_model.FindSignature(kVerifySignatureRunner));

  absl::flat_hash_map<absl::string_view, TensorBuffer> verifier_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> verifier_output_buffers;
  int num_draft_steps;
  {
    for (absl::string_view input_name : verify_signature.InputNames()) {
      if (absl::StrContains(input_name, "kv")) {
        continue;
      }
      LITERT_ASSIGN_OR_RETURN(
          auto input_buffer,
          base_model.CreateInputBuffer(verify_signature.Key(), input_name));
      verifier_input_buffers[input_name] = std::move(input_buffer);
    }
    LITERT_ASSIGN_OR_RETURN(
        size_t verify_signature_index,
        base_model.GetSignatureIndex(kVerifySignatureRunner));
    for (size_t i = 0; i < verify_signature.OutputNames().size(); ++i) {
      absl::string_view output_name = verify_signature.OutputNames()[i];
      if (absl::StrContains(output_name, "kv")) {
        continue;
      }
      if (output_name == "logits" && use_fp16_precision) {
        LITERT_ASSIGN_OR_RETURN(
            auto output_buffer,
            CreateFP16OutputBuffer(env, base_model, verify_signature_index,
                                   output_name, i));
        verifier_output_buffers[output_name] = std::move(output_buffer);
      } else {
        LITERT_ASSIGN_OR_RETURN(
            auto output_buffer,
            base_model.CreateOutputBuffer(verify_signature.Key(), output_name));
        verifier_output_buffers[output_name] = std::move(output_buffer);
      }
    }

    LITERT_ASSIGN_OR_RETURN(auto input_pos_tensor_type,
                            verify_signature.InputTensorType("input_pos"));
    // Expecred shape: [T = G + 1] where G is the number of draft steps
    const auto& input_pos_dims = input_pos_tensor_type.Layout().Dimensions();
    num_draft_steps = input_pos_dims[0] - 1;
  }

  LITERT_ASSIGN_OR_RETURN(
      int vocab_size,
      GetVocabSizeFromLogitsTensor(verifier_output_buffers["logits"]));

  ActivationDataType drafter_logits_data_type;
  {
    LITERT_ASSIGN_OR_RETURN(auto drafter_logits_type,
                            mtp_drafter_output_buffers["logits"].TensorType());
    if (drafter_logits_type.ElementType() == ElementType::Float16) {
      drafter_logits_data_type = ActivationDataType::FLOAT16;
    } else if (drafter_logits_type.ElementType() == ElementType::Float32) {
      drafter_logits_data_type = ActivationDataType::FLOAT32;
    } else {
      return absl::InvalidArgumentError("Unsupported drafter logits type");
    }
  }

  ActivationDataType verifier_logits_data_type;
  {
    LITERT_ASSIGN_OR_RETURN(auto verifier_logits_type,
                            verifier_output_buffers["logits"].TensorType());
    if (verifier_logits_type.ElementType() == ElementType::Float16) {
      verifier_logits_data_type = ActivationDataType::FLOAT16;
    } else if (verifier_logits_type.ElementType() == ElementType::Float32) {
      verifier_logits_data_type = ActivationDataType::FLOAT32;
    } else {
      return absl::InvalidArgumentError("Unsupported verifier logits type");
    }
  }

  {
    LITERT_ASSIGN_OR_RETURN(
        auto drafter_projected_activations_type,
        drafter_signature.OutputTensorType("projected_activations"));
    if (drafter_projected_activations_type.ElementType() !=
        ElementType::Float32) {
      return absl::InvalidArgumentError(
          "Unsupported drafter projected activations type: must be float32");
    }
  }

  {
    LITERT_ASSIGN_OR_RETURN(auto verifier_activations_type,
                            verify_signature.OutputTensorType("activations"));
    if (verifier_activations_type.ElementType() != ElementType::Float32) {
      return absl::InvalidArgumentError(
          "Unsupported verifier activations type: must be float32");
    }
  }

  ABSL_ASSIGN_OR_RETURN(auto drafter_sampler,
                        CreateGreedySampler(env, backend,
                                            /*output_heads=*/1,
                                            /*sequence_size=*/1, vocab_size,
                                            drafter_logits_data_type));
  ABSL_ASSIGN_OR_RETURN(
      auto verifier_sampler,
      CreateGreedySampler(env, backend,
                          /*output_heads=*/1,
                          /*sequence_size=*/num_draft_steps + 1, vocab_size,
                          verifier_logits_data_type));

  LITERT_ASSIGN_OR_RETURN(auto drafter_id_tensor,
                          CreateTensorBuffer<int32_t>({1, 1}));
  LITERT_ASSIGN_OR_RETURN(
      auto verifier_id_tensor,
      CreateTensorBuffer<int32_t>({1, num_draft_steps + 1}));

  ABSL_ASSIGN_OR_RETURN(
      auto drafter_model_signatures,
      GetModelSignaturesFromInputOutputNames(drafter_signature.InputNames(),
                                             drafter_signature.OutputNames(),
                                             /*strict=*/false));
  ABSL_ASSIGN_OR_RETURN(
      auto verifier_model_signatures,
      GetModelSignaturesFromInputOutputNames(verify_signature.InputNames(),
                                             verify_signature.OutputNames(),
                                             /*strict=*/false));

  const HwAccelerators accelerator = GetHwAcceleratorForBackend(backend);
  ABSL_ASSIGN_OR_RETURN(auto drafter_mask_runner,
                        LiteRtLogitMaskRunner::Create(env, accelerator));
  ABSL_ASSIGN_OR_RETURN(auto verifier_mask_runner,
                        LiteRtLogitMaskRunner::Create(env, accelerator));

  return absl::WrapUnique(new LlmLiteRtMtpDrafter(
      std::move(mtp_drafter_model), std::move(drafter_signature), base_model,
      std::move(verify_signature), base_model_desc, embedding_manager,
      ple_manager, std::move(drafter_sampler), std::move(verifier_sampler),
      std::move(kv_cache_input_names), std::move(mtp_drafter_input_buffers),
      std::move(mtp_drafter_output_buffers), std::move(verifier_input_buffers),
      std::move(verifier_output_buffers), std::move(drafter_id_tensor),
      std::move(verifier_id_tensor), num_draft_steps,
      std::move(drafter_model_signatures), std::move(verifier_model_signatures),
      vocab_size, GetAttentionMaskParams(executor_metadata),
      std::move(drafter_mask_runner), std::move(verifier_mask_runner)));
}

absl::Status LlmLiteRtMtpDrafter::PrepareDrafterInputBuffers(
    int position, absl::flat_hash_map<absl::string_view, TensorBuffer>&
                      output_kv_cache_buffers) {
  for (const auto& kv_cache_input_name : kv_cache_input_names_) {
    LITERT_ASSIGN_OR_RETURN(
        auto kv_cache_buffer_dup,
        output_kv_cache_buffers.at(kv_cache_input_name).Duplicate());
    active_drafter_input_buffers_[kv_cache_input_name] =
        std::move(kv_cache_buffer_dup);
  }
  LITERT_RETURN_IF_ERROR(
      active_drafter_input_buffers_["input_pos"].Write<int32_t>(
          absl::MakeSpan(&position, 1)));
  if (drafter_signatures_.input_attn_mask.has_value()) {
    auto& mask_buf =
        active_drafter_input_buffers_[*drafter_signatures_.input_attn_mask];
    ABSL_RETURN_IF_ERROR(
        InitializeAttentionMask(mask_buf, /*use_fp16_precision=*/false));
    ABSL_RETURN_IF_ERROR(FillAttentionMask(mask_buf,
                                           /*start_timestep=*/position,
                                           /*steps=*/1,
                                           attn_params_.global_type));
  }
  if (drafter_signatures_.input_attn_mask_local.has_value()) {
    auto& mask_buf = active_drafter_input_buffers_[*drafter_signatures_
                                                        .input_attn_mask_local];
    ABSL_RETURN_IF_ERROR(
        InitializeAttentionMask(mask_buf, /*use_fp16_precision=*/false));
    ABSL_RETURN_IF_ERROR(FillAttentionMask(
        mask_buf,
        /*start_timestep=*/position,
        /*steps=*/1, attn_params_.local_type,
        /*token_ids=*/std::nullopt, attn_params_.sliding_window_size,
        RingBufferAttentionMaskMode::kDecode));
  }
  if (drafter_signatures_.input_int32_param.has_value()) {
    ABSL_RETURN_IF_ERROR(FillSingleBufferCacheParamTensor(
        active_drafter_input_buffers_[*drafter_signatures_.input_int32_param],
        position,
        /*update_length=*/1));
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtMtpDrafter::PrepareDrafterOutputBuffers() {
  for (auto& [output_name, output_buffer] : active_drafter_output_buffers_) {
    LITERT_RETURN_IF_ERROR(output_buffer.ClearEvent());
  }
  return absl::OkStatus();
}

absl::StatusOr<LlmLiteRtMtpDrafter::DraftingResult>
LlmLiteRtMtpDrafter::RunDraftingLoop(
    int token_id, std::optional<TensorBuffer>& activations,
    const Constraint* constraint,
    const Constraint::State* verified_constraint_state) {
  AGELLO_SPAN(kMtpDraft);
  if (resident_ != nullptr &&
      RoundNeedsNoMasks(constraint, verified_constraint_state, num_draft_steps_)) {
    auto resident = RunResidentDraftingLoop(token_id, activations);
    if (resident.ok()) {
      ABSL_RETURN_IF_ERROR(ChainDraftConstraintStates(constraint, verified_constraint_state,
                                                      resident->drafted_tokens,
                                                      resident->draft_constraint_states));
      return resident;
    }
    MTP_DRAFTER_LOG() << "resident round fell back to the host: " << resident.status();
  }
  DraftingResult result;
  result.drafted_tokens.reserve(num_draft_steps_);
  result.draft_constraint_states.reserve(num_draft_steps_);
  int last_drafted_token_id = token_id;
  std::vector<float> embedding_vector;
  TensorBuffer* activations_ptr =
      activations.has_value() ? &activations.value() : nullptr;
  const Constraint::State* current_draft_state = verified_constraint_state;
  for (int i = 0; i < num_draft_steps_; ++i) {
    LITERT_RETURN_IF_ERROR(PrepareDrafterOutputBuffers());
    // Concat and lookup embeddings with previous activations.
    // Embedding lookup has shape [B = 1, T = 1, D = 1536]
    // Drafter output activation has shape [B = 1, T = 1, D = 1536]
    // Concatenated embedding + activation has shape [B = 1, T = 1, D = 3072]
    TensorBuffer* drafter_activations_buffer =
        &active_drafter_input_buffers_["activations"];
    ABSL_RETURN_IF_ERROR(embedding_manager_.LookupDecode(last_drafted_token_id,
                                                         embedding_vector));
    if (activations_ptr) {
      ABSL_RETURN_IF_ERROR(ConcatenateEmbeddingsAndActivations(
          embedding_vector, *activations_ptr, *drafter_activations_buffer));
    } else {
      ABSL_RETURN_IF_ERROR(
          ConcatenateEmbeddingsAndActivationsFromVerifierBuffer(
              embedding_vector, verifier_output_buffers_["activations"],
              last_verified_token_id_idx_, *drafter_activations_buffer));
    }

    bool async = true;
    {
      AGELLO_SPAN(kRunDrafter);
      AGELLO_COUNT(kRunsDrafter);
      ::agello::telemetry::SetCurrentRun(::agello::telemetry::kGpuDrafter);
      LITERT_RETURN_IF_ERROR(mtp_drafter_model_.RunAsync(
          drafter_signature_.Key(), active_drafter_input_buffers_,
          active_drafter_output_buffers_, async));
    }

    if (constraint != nullptr && current_draft_state != nullptr) {
      ABSL_ASSIGN_OR_RETURN(auto mask,
                            constraint->ComputeMask(*current_draft_state));
      ABSL_RETURN_IF_ERROR(drafter_mask_runner_->Apply(
          active_drafter_output_buffers_["logits"], mask.get()));
    }

    ABSL_RETURN_IF_ERROR(drafter_sampler_->SampleToIdAndScoreBuffer(
        active_drafter_output_buffers_["logits"], drafter_id_tensor_,
        /*scores_tensor=*/nullptr));
    LITERT_ASSIGN_OR_RETURN(auto id_vector,
                            CopyFromTensorBuffer<int32_t>(drafter_id_tensor_));
    RET_CHECK_EQ(id_vector.size(), 1);
    int sampled_draft_id = id_vector[0];
    result.drafted_tokens.push_back(sampled_draft_id);

    if (constraint != nullptr && current_draft_state != nullptr) {
      ABSL_ASSIGN_OR_RETURN(
          auto next_state,
          constraint->ComputeNext(*current_draft_state, sampled_draft_id));
      if (constraint->IsEnded(*next_state)) {
        next_state = constraint->Start();
      }
      result.draft_constraint_states.push_back(std::move(next_state));
      current_draft_state = result.draft_constraint_states.back().get();
    }

    last_drafted_token_id = sampled_draft_id;
    activations_ptr = &active_drafter_output_buffers_["projected_activations"];
  }
  return result;
}

absl::StatusOr<LlmLiteRtMtpDrafter::DraftingResult>
LlmLiteRtMtpDrafter::RunResidentDraftingLoop(int token_id,
                                             std::optional<TensorBuffer>& activations) {
  if (num_draft_steps_ > 8) return absl::FailedPreconditionError("more than 8 draft steps");
  TensorBuffer& input = active_drafter_input_buffers_["activations"];
  TensorBuffer& logits = active_drafter_output_buffers_["logits"];
  TensorBuffer& projected = active_drafter_output_buffers_["projected_activations"];
  LITERT_ASSIGN_OR_RETURN(void* input_mtl, input.GetMetalBuffer());
  LITERT_ASSIGN_OR_RETURN(void* logits_mtl, logits.GetMetalBuffer());
  LITERT_ASSIGN_OR_RETURN(void* projected_mtl, projected.GetMetalBuffer());
  LITERT_ASSIGN_OR_RETURN(auto logits_type, logits.TensorType());
  if (logits_type.ElementType() != ElementType::Float16) {
    return absl::FailedPreconditionError("drafter logits aren't FP16");
  }
  // Step 0 continues from decode's activations, or from verify's row of the last accepted token.
  void* previous_mtl = nullptr;
  size_t previous_offset = 0;
  if (activations.has_value()) {
    LITERT_ASSIGN_OR_RETURN(previous_mtl, activations->GetMetalBuffer());
  } else {
    if (last_verified_token_id_idx_ < 0) return absl::FailedPreconditionError("no activations");
    LITERT_ASSIGN_OR_RETURN(previous_mtl, verifier_output_buffers_["activations"].GetMetalBuffer());
    LITERT_ASSIGN_OR_RETURN(auto projected_type, projected.TensorType());
    LITERT_ASSIGN_OR_RETURN(size_t row_bytes, projected_type.Bytes());
    previous_offset = static_cast<size_t>(last_verified_token_id_idx_) * row_bytes;
  }
  resident_->SetFirstId(token_id);
  int drafted[8] = {0};
  bool encoded = true;
  {
    // The three runs and the kernels between them committed back to back on one queue.
    DeferredGpuWaits deferred;
    for (int i = 0; i < num_draft_steps_ && encoded; ++i) {
      LITERT_RETURN_IF_ERROR(PrepareDrafterOutputBuffers());
      encoded = resident_->EncodeInput(i, input_mtl, i == 0 ? previous_mtl : projected_mtl,
                                       i == 0 ? previous_offset : 0);
      if (!encoded) break;
      bool async = true;
      {
        AGELLO_SPAN(kRunDrafter);
        AGELLO_COUNT(kRunsDrafter);
        ::agello::telemetry::SetCurrentRun(::agello::telemetry::kGpuDrafter);
        LITERT_RETURN_IF_ERROR(mtp_drafter_model_.RunAsync(
            drafter_signature_.Key(), active_drafter_input_buffers_,
            active_drafter_output_buffers_, async));
      }
      encoded = resident_->EncodeArgmax(i, logits_mtl, vocab_size_);
    }
  }
  // Waits for the last kernel in every case, so the host path never races it.
  const bool finished = resident_->Finish(num_draft_steps_, drafted);
  if (!encoded || !finished) {
    return absl::InternalError("the GPU round didn't finish (or met a NaN maximum)");
  }
  DraftingResult result;
  result.drafted_tokens.assign(drafted, drafted + num_draft_steps_);
  return result;
}

absl::StatusOr<std::pair<std::vector<int>, std::vector<int>>>
LlmLiteRtMtpDrafter::RunResidentRound(
    int position, int token_id, std::optional<TensorBuffer>& activations,
    absl::flat_hash_map<absl::string_view, TensorBuffer>& input_kv_cache_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer>& output_kv_cache_buffers) {
  if (num_draft_steps_ > 7) return absl::FailedPreconditionError("more than 7 draft steps");
  if (!verifier_signatures_.input_embeddings.has_value() ||
      !verifier_signatures_.input_per_layer_embeddings.has_value() || !ple_manager_.has_value()) {
    return absl::FailedPreconditionError("verify takes no per-layer embeddings");
  }
  TensorBuffer& embeddings = active_verifier_input_buffers_[*verifier_signatures_.input_embeddings];
  TensorBuffer& per_layer =
      active_verifier_input_buffers_[*verifier_signatures_.input_per_layer_embeddings];
  // Packed buffers hold plain rows, which the run's input conversion rearranges; another layout
  // would need the kernels to write ML Drift's.
  for (TensorBuffer* buffer : {&embeddings, &per_layer}) {
    LITERT_ASSIGN_OR_RETURN(auto type, buffer->BufferType());
    if (type != TensorBufferType::kMetalBufferPacked) {
      return absl::FailedPreconditionError("verify's embeddings aren't packed Metal buffers");
    }
  }
  LITERT_ASSIGN_OR_RETURN(void* embeddings_mtl, embeddings.GetMetalBuffer());
  LITERT_ASSIGN_OR_RETURN(void* per_layer_mtl, per_layer.GetMetalBuffer());
  TensorBuffer& input = active_drafter_input_buffers_["activations"];
  TensorBuffer& logits = active_drafter_output_buffers_["logits"];
  TensorBuffer& projected = active_drafter_output_buffers_["projected_activations"];
  LITERT_ASSIGN_OR_RETURN(void* input_mtl, input.GetMetalBuffer());
  LITERT_ASSIGN_OR_RETURN(void* logits_mtl, logits.GetMetalBuffer());
  LITERT_ASSIGN_OR_RETURN(void* projected_mtl, projected.GetMetalBuffer());
  LITERT_ASSIGN_OR_RETURN(auto logits_type, logits.TensorType());
  if (logits_type.ElementType() != ElementType::Float16) {
    return absl::FailedPreconditionError("drafter logits aren't FP16");
  }
  void* previous_mtl = nullptr;
  size_t previous_offset = 0;
  if (activations.has_value()) {
    LITERT_ASSIGN_OR_RETURN(previous_mtl, activations->GetMetalBuffer());
  } else {
    if (last_verified_token_id_idx_ < 0) return absl::FailedPreconditionError("no activations");
    LITERT_ASSIGN_OR_RETURN(previous_mtl, verifier_output_buffers_["activations"].GetMetalBuffer());
    LITERT_ASSIGN_OR_RETURN(auto projected_type, projected.TensorType());
    LITERT_ASSIGN_OR_RETURN(size_t row_bytes, projected_type.Bytes());
    previous_offset = static_cast<size_t>(last_verified_token_id_idx_) * row_bytes;
  }
  // Verify's inputs that don't depend on the drafts, written before the chain.
  ABSL_RETURN_IF_ERROR(PrepareVerifierInputBuffers(position, token_id,
                                                   std::vector<int>(num_draft_steps_, 0),
                                                   input_kv_cache_buffers,
                                                   /*embeddings_on_gpu=*/true));
  ABSL_RETURN_IF_ERROR(PrepareVerifierOutputBuffers(output_kv_cache_buffers));
  resident_->SetFirstId(token_id);
  bool encoded = true;
  {
    // The drafter's runs, the kernels between them, verify's inputs and verify, back to back.
    DeferredGpuWaits deferred;
    for (int i = 0; i < num_draft_steps_ && encoded; ++i) {
      LITERT_RETURN_IF_ERROR(PrepareDrafterOutputBuffers());
      encoded = resident_->EncodeInput(i, input_mtl, i == 0 ? previous_mtl : projected_mtl,
                                       i == 0 ? previous_offset : 0);
      if (!encoded) break;
      bool async = true;
      {
        AGELLO_SPAN(kRunDrafter);
        AGELLO_COUNT(kRunsDrafter);
        ::agello::telemetry::SetCurrentRun(::agello::telemetry::kGpuDrafter);
        LITERT_RETURN_IF_ERROR(mtp_drafter_model_.RunAsync(
            drafter_signature_.Key(), active_drafter_input_buffers_,
            active_drafter_output_buffers_, async));
      }
      encoded = resident_->EncodeArgmax(i, logits_mtl, vocab_size_);
    }
    if (encoded) {
      encoded = resident_->EncodeVerifyInputs(num_draft_steps_ + 1, embeddings_mtl, per_layer_mtl);
    }
    if (encoded) {
      AGELLO_SPAN(kRunVerify);
      AGELLO_COUNT(kRunsVerify);
      ::agello::telemetry::SetCurrentRun(::agello::telemetry::kGpuVerify);
      bool async = true;
      LITERT_RETURN_IF_ERROR(base_model_.RunAsync(verify_signature_.Key(),
                                                  active_verifier_input_buffers_,
                                                  active_verifier_output_buffers_, async));
    }
  }
  int drafted[8] = {0};
  const bool finished = resident_->Finish(num_draft_steps_, drafted);
  if (!encoded || !finished) {
    return absl::InternalError("the GPU round didn't finish (or met a NaN maximum)");
  }
  // Verify's greedy picks, on the host as before.
  ABSL_RETURN_IF_ERROR(verifier_sampler_->SampleToIdAndScoreBuffer(
      active_verifier_output_buffers_.at("logits"), verifier_id_tensor_,
      /*scores_tensor=*/nullptr));
  LITERT_ASSIGN_OR_RETURN(auto verified, CopyFromTensorBuffer<int32_t>(verifier_id_tensor_));
  RET_CHECK_EQ(verified.size(), num_draft_steps_ + 1);
  return std::make_pair(std::vector<int>(drafted, drafted + num_draft_steps_),
                        std::vector<int>(verified.begin(), verified.end()));
}

absl::Status LlmLiteRtMtpDrafter::PrepareVerifierInputBuffers(
    int position, int token_id, const std::vector<int>& drafted_tokens,
    absl::flat_hash_map<absl::string_view, TensorBuffer>&
        input_kv_cache_buffers, bool embeddings_on_gpu) {
  {
    LITERT_ASSIGN_OR_RETURN(auto verifier_input_pos_lock_and_addr,
                            TensorBufferScopedLock::Create(
                                active_verifier_input_buffers_["input_pos"],
                                TensorBuffer::LockMode::kWrite));
    auto* prefill_input_pos_ptr =
        static_cast<int32_t*>(verifier_input_pos_lock_and_addr.second);
    for (int i = 0; i < num_draft_steps_ + 1; ++i) {
      *prefill_input_pos_ptr++ = position + i;
    }
  }

  std::vector<int> drafted_tokens_with_input_token;
  drafted_tokens_with_input_token.reserve(num_draft_steps_ + 1);
  drafted_tokens_with_input_token.push_back(token_id);
  drafted_tokens_with_input_token.insert(drafted_tokens_with_input_token.end(),
                                         drafted_tokens.begin(),
                                         drafted_tokens.end());

  if (verifier_signatures_.input_attn_mask.has_value()) {
    auto& mask_buf =
        active_verifier_input_buffers_[*verifier_signatures_.input_attn_mask];
    ABSL_RETURN_IF_ERROR(
        InitializeAttentionMask(mask_buf, /*use_fp16_precision=*/false));
    ABSL_RETURN_IF_ERROR(FillAttentionMask(mask_buf,
                                           /*start_timestep=*/position,
                                           /*steps=*/num_draft_steps_ + 1,
                                           attn_params_.global_type));
  }
  if (verifier_signatures_.input_attn_mask_local.has_value()) {
    auto& mask_buf =
        active_verifier_input_buffers_[*verifier_signatures_
                                            .input_attn_mask_local];
    ABSL_RETURN_IF_ERROR(
        InitializeAttentionMask(mask_buf, /*use_fp16_precision=*/false));
    ABSL_RETURN_IF_ERROR(FillAttentionMask(
        mask_buf,
        /*start_timestep=*/position,
        /*steps=*/num_draft_steps_ + 1, attn_params_.local_type,
        /*token_ids=*/std::nullopt, attn_params_.sliding_window_size,
        RingBufferAttentionMaskMode::kVerify));
  }

  if (verifier_signatures_.input_embeddings.has_value() && !embeddings_on_gpu) {
    ABSL_RETURN_IF_ERROR(embedding_manager_.LookupPrefill(
        drafted_tokens_with_input_token,
        &active_verifier_input_buffers_[*verifier_signatures_.input_embeddings],
        /*offset=*/0));
  }
  if (ple_manager_.has_value() && !embeddings_on_gpu &&
      verifier_signatures_.input_per_layer_embeddings.has_value()) {
    ABSL_RETURN_IF_ERROR(ple_manager_->get().LookupPrefill(
        drafted_tokens_with_input_token,
        &active_verifier_input_buffers_[*verifier_signatures_
                                             .input_per_layer_embeddings],
        /*offset=*/0));
  }

  for (const auto& [input_name, input_buffer] : input_kv_cache_buffers) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
    active_verifier_input_buffers_[input_name] = std::move(input_buffer_dup);
  }
  if (verifier_signatures_.input_int32_param.has_value()) {
    ABSL_RETURN_IF_ERROR(FillSingleBufferCacheParamTensor(
        active_verifier_input_buffers_[*verifier_signatures_.input_int32_param],
        position, num_draft_steps_ + 1));
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtMtpDrafter::PrepareVerifierOutputBuffers(
    absl::flat_hash_map<absl::string_view, TensorBuffer>&
        output_kv_cache_buffers) {
  for (const auto& [output_name, output_buffer] : output_kv_cache_buffers) {
    LITERT_ASSIGN_OR_RETURN(auto output_buffer_dup, output_buffer.Duplicate());
    active_verifier_output_buffers_[output_name] = std::move(output_buffer_dup);
  }
  for (auto& [output_name, output_buffer] : active_verifier_output_buffers_) {
    LITERT_RETURN_IF_ERROR(output_buffer.ClearEvent());
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int>> LlmLiteRtMtpDrafter::RunVerification(
    const std::vector<std::unique_ptr<Constraint::State>>&
        draft_constraint_states) {
  AGELLO_SPAN(kMtpVerify);
  bool async = true;
  {
    AGELLO_SPAN(kRunVerify);
    AGELLO_COUNT(kRunsVerify);
    ::agello::telemetry::SetCurrentRun(::agello::telemetry::kGpuVerify);
    LITERT_RETURN_IF_ERROR(base_model_.RunAsync(
        verify_signature_.Key(), active_verifier_input_buffers_,
        active_verifier_output_buffers_, async));
  }

  // Apply constraint masks to verifier logits for all positions in a single
  // pass.
  if (constraint_ != nullptr) {
    std::vector<std::unique_ptr<LogitMask>> masks;
    masks.reserve(draft_constraint_states.size() + 1);
    if (constraint_state_ != nullptr) {
      ABSL_ASSIGN_OR_RETURN(auto mask,
                            constraint_->ComputeMask(*constraint_state_));
      masks.push_back(std::move(mask));
    } else {
      masks.push_back(nullptr);
    }
    for (const auto& draft_state : draft_constraint_states) {
      if (draft_state != nullptr) {
        ABSL_ASSIGN_OR_RETURN(auto mask,
                              constraint_->ComputeMask(*draft_state));
        masks.push_back(std::move(mask));
      } else {
        masks.push_back(nullptr);
      }
    }
    ABSL_RETURN_IF_ERROR(verifier_mask_runner_->ApplySequence(
        active_verifier_output_buffers_.at("logits"), masks));
  }

  ABSL_RETURN_IF_ERROR(verifier_sampler_->SampleToIdAndScoreBuffer(
      active_verifier_output_buffers_.at("logits"), verifier_id_tensor_,
      /*scores_tensor=*/nullptr));

  LITERT_ASSIGN_OR_RETURN(auto id_vector,
                          CopyFromTensorBuffer<int32_t>(verifier_id_tensor_));
  RET_CHECK_EQ(id_vector.size(), num_draft_steps_ + 1);
  return id_vector;
}

absl::StatusOr<std::vector<std::vector<int>>> LlmLiteRtMtpDrafter::Draft(
    int position, int token_id, std::optional<TensorBuffer> activations,
    StateInterface& state, const Constraint* constraint) {
  ::agello::telemetry::ScopedSpan round_span(::agello::telemetry::kMtpRound);
  auto* litert_state = dynamic_cast<LitertState*>(&state);
  RET_CHECK(litert_state != nullptr);
  LITERT_ASSIGN_OR_RETURN(
      auto state_buffers,
      litert_state->GetStateBuffers(base_model_, verify_signature_.Key()));

  ABSL_RETURN_IF_ERROR(
      PrepareDrafterInputBuffers(position - 1, state_buffers.output_buffers));

  // Initialize or reset constraint state when constraint changes or on first
  // step of decode turn.
  if (constraint == nullptr) {
    constraint_ = nullptr;
    constraint_state_.reset();
  } else if (constraint != constraint_ || activations.has_value() ||
             constraint_state_ == nullptr) {
    constraint_ = constraint;
    constraint_state_ = constraint_->Start();
    // Compute mask on the initial state so the underlying grammar engine (e.g.
    // llguidance) initializes parser rows and lexer state before committing the
    // first token.
    ABSL_RETURN_IF_ERROR(constraint_->ComputeMask(*constraint_state_).status());
    ABSL_ASSIGN_OR_RETURN(constraint_state_, constraint_->ComputeNext(
                                                 *constraint_state_, token_id));
    if (constraint_->IsEnded(*constraint_state_)) {
      constraint_state_ = constraint_->Start();
    }
  }

  const absl::Time draft_start = absl::Now();
  DraftingResult drafting_result;
  std::vector<int> verifier_id_vector;
  absl::Duration drafting_time;
  absl::Duration verify_time;
  // Agello (the GPU-resident round, stage 3): the whole round on the GPU when it can.
  std::optional<std::pair<std::vector<int>, std::vector<int>>> resident_round;
  // With a constraint (Think's budget), when its masks would allow every token.
  if (resident_ != nullptr && resident_->HasVerifyTables() &&
      RoundNeedsNoMasks(constraint_, constraint_state_.get(), num_draft_steps_)) {
    auto round = RunResidentRound(position, token_id, activations, state_buffers.input_buffers,
                                  state_buffers.output_buffers);
    if (round.ok()) {
      resident_round = std::move(*round);
    } else {
      MTP_DRAFTER_LOG() << "resident round fell back to the host: " << round.status();
    }
  }
  if (resident_round.has_value()) {
    drafting_result.drafted_tokens = std::move(resident_round->first);
    verifier_id_vector = std::move(resident_round->second);
    ABSL_RETURN_IF_ERROR(ChainDraftConstraintStates(constraint_, constraint_state_.get(),
                                                    drafting_result.drafted_tokens,
                                                    drafting_result.draft_constraint_states));
    drafting_time = absl::Now() - draft_start;
  } else {
    ABSL_ASSIGN_OR_RETURN(drafting_result,
                          RunDraftingLoop(token_id, activations, constraint_,
                                          constraint_state_.get()));
    drafting_time = absl::Now() - draft_start;

    const absl::Time verify_start = absl::Now();
    ABSL_RETURN_IF_ERROR(PrepareVerifierInputBuffers(
        position, token_id, drafting_result.drafted_tokens,
        state_buffers.input_buffers));
    ABSL_RETURN_IF_ERROR(
        PrepareVerifierOutputBuffers(state_buffers.output_buffers));

    ABSL_ASSIGN_OR_RETURN(
        verifier_id_vector,
        RunVerification(drafting_result.draft_constraint_states));
    verify_time = absl::Now() - verify_start;
  }

  RET_CHECK_EQ(verifier_id_vector.size(),
               drafting_result.drafted_tokens.size() + 1);
  int num_correct_tokens =
      CountAcceptedPrefix(verifier_id_vector, drafting_result.drafted_tokens);
  int bonus_token = verifier_id_vector[num_correct_tokens];
  last_verified_token_id_idx_ = num_correct_tokens;

  // Update verified constraint state according to verified + bonus tokens.
  if (constraint_ != nullptr) {
    if (num_correct_tokens > 0 &&
        num_correct_tokens <=
            static_cast<int>(drafting_result.draft_constraint_states.size())) {
      constraint_state_ = std::move(
          drafting_result.draft_constraint_states[num_correct_tokens - 1]);
    }
    if (bonus_token >= 0 && constraint_state_ != nullptr) {
      ABSL_ASSIGN_OR_RETURN(
          constraint_state_,
          constraint_->ComputeNext(*constraint_state_, bonus_token));
      if (constraint_->IsEnded(*constraint_state_)) {
        constraint_state_ = constraint_->Start();
      }
    }
  }

  round_span.End();
  if (const char mode = MtpReplayMode(); mode != '\0' && constraint_ == nullptr) {
    AGELLO_SPAN(kMtpReplay);
    AGELLO_COUNT(kMtpReplayRounds);
    ABSL_RETURN_IF_ERROR(
        PrepareDrafterInputBuffers(position - 1, state_buffers.output_buffers));
    auto prepare_verify = [&]() -> absl::Status {
      ABSL_RETURN_IF_ERROR(PrepareVerifierInputBuffers(
          position, token_id, drafting_result.drafted_tokens,
          state_buffers.input_buffers));
      return PrepareVerifierOutputBuffers(state_buffers.output_buffers);
    };
    if (mode != 'b' && mode != 'e') ABSL_RETURN_IF_ERROR(prepare_verify());
    DeferredGpuWaits deferred;
    for (int i = 0; i < num_draft_steps_; ++i) {
      LITERT_RETURN_IF_ERROR(PrepareDrafterOutputBuffers());
      ::agello::telemetry::SetCurrentRun(::agello::telemetry::kGpuReplay);
      bool async = true;
      LITERT_RETURN_IF_ERROR(mtp_drafter_model_.RunAsync(
          drafter_signature_.Key(), active_drafter_input_buffers_,
          active_drafter_output_buffers_, async));
      if (mode == 'c') ::agello::telemetry::WaitForDeferredGpuWork();
    }
    if (mode == 'a' || mode == 'b') ::agello::telemetry::WaitForDeferredGpuWork();
    if (mode == 'b') ABSL_RETURN_IF_ERROR(prepare_verify());
    // Mode e: on every way out, a hold still pending is dropped and a held run released, so
    // the queue never waits on an event nobody signals (declared after `deferred`, so it runs
    // before the final wait).
    struct ReleaseHeld {
      bool armed = false;
      ~ReleaseHeld() {
        if (!armed) return;
        ::agello::telemetry::TakeGpuRunHold();
        ::agello::telemetry::ReleaseHeldGpuRun();
      }
    } release_held;
    if (mode == 'e') {
      ::agello::telemetry::MarkDeferredGpuWork();
      ::agello::telemetry::HoldNextGpuRun();
      release_held.armed = true;
    }
    ::agello::telemetry::SetCurrentRun(::agello::telemetry::kGpuReplay);
    bool async = true;
    LITERT_RETURN_IF_ERROR(base_model_.RunAsync(
        verify_signature_.Key(), active_verifier_input_buffers_,
        active_verifier_output_buffers_, async));
    if (mode == 'e') {
      ::agello::telemetry::WaitForMarkedGpuWork();
      ABSL_RETURN_IF_ERROR(prepare_verify());
      ::agello::telemetry::ReleaseHeldGpuRun();
    }
  }

  MTP_DRAFTER_LOG() << "drafted_tokens: "
                    << absl::StrJoin(drafting_result.drafted_tokens, ", ");
  MTP_DRAFTER_LOG() << "bonus_token: " << bonus_token;
  MTP_DRAFTER_LOG() << "num_correct_tokens: " << num_correct_tokens;

  // The first token comes from the decode output and is always correct.
  std::vector<int> output_tokens = std::move(drafting_result.drafted_tokens);
  output_tokens.resize(num_correct_tokens);
  output_tokens.push_back(bonus_token);
  num_drafted_tokens_ += num_draft_steps_;
  num_verified_tokens_ += num_correct_tokens;
  last_round_stats_ = MtpRoundStats{
      .drafted_tokens = num_draft_steps_,
      .accepted_tokens = num_correct_tokens,
      .emitted_tokens = num_correct_tokens + 1,
      .drafting_time = drafting_time,
      .verify_time = verify_time,
  };

  MTP_DRAFTER_LOG() << "drafter output: " << absl::StrJoin(output_tokens, ", ");
  MTP_DRAFTER_LOG() << "--------------------------------------------------";

  return std::vector<std::vector<int>>{std::move(output_tokens)};
}

}  // namespace litert::lm
