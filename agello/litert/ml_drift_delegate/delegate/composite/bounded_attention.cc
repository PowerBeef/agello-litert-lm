// Copyright 2026 Patrice Déry (Agello, PowerBeef/LiteRT)
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

#include "ml_drift_delegate/delegate/composite/bounded_attention.h"

#include <any>
#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <set>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/runtime_batched_matmul_parser.h"

namespace litert::ml_drift {
namespace {

using ::ml_drift::GraphFloat32;
using ::ml_drift::Node;
using ::ml_drift::OperationType;
using ::ml_drift::Value;

// A value read by exactly one node and not a graph output.
bool SingleUse(const GraphFloat32& graph, const Value* value) {
  return !graph.IsGraphOutput(value->id) && graph.FindConsumers(value->id).size() == 1;
}

std::optional<float> ScalarParam(const Node& node) {
  const auto* attr = std::any_cast<::ml_drift::ElementwiseAttributes>(&node.operation.attributes);
  if (attr == nullptr) return std::nullopt;
  const auto* scalar = std::get_if<::ml_drift::ScalarValue>(&attr->param);
  if (scalar == nullptr) return std::nullopt;
  const float* value = std::get_if<float>(scalar);
  if (value == nullptr) return std::nullopt;
  return *value;
}

// The int32 values of a constant value, or nullptr.
const std::vector<int32_t>* IntConstant(const GraphFloat32& graph, const Value* value) {
  const Node* producer = graph.FindProducer(value->id);
  if (producer == nullptr || producer->operation.type != ToString(OperationType::kConstant)) {
    return nullptr;
  }
  const auto* attr = std::any_cast<::ml_drift::ConstTensorAttributes>(&producer->operation.attributes);
  if (attr == nullptr) return nullptr;
  const auto* tensor = std::get_if<::ml_drift::TensorInt32>(&attr->tensor);
  return tensor == nullptr ? nullptr : &tensor->data;
}

}  // namespace

std::optional<int> FindSlidingWindow(const GraphFloat32& graph) {
  // A sliding-window mask keeps key k for position p when p >= k and p < k + W:
  // LOGICAL_AND(GREATER_EQUAL(p, keys), LESS(p, keys + W)), the keys and keys + W
  // being int32 constants. Every such mask must agree on W.
  std::optional<int> window;
  for (const Node* node : graph.nodes()) {
    if (node == nullptr || node->operation.type != ToString(OperationType::kLogicalAnd)) continue;
    const auto inputs = graph.FindInputs(node->id);
    if (inputs.size() != 2) continue;
    const Node* ge = graph.FindProducer(inputs[0]->id);
    const Node* lt = graph.FindProducer(inputs[1]->id);
    if (ge == nullptr || lt == nullptr) continue;
    if (ge->operation.type == ToString(OperationType::kLess)) std::swap(ge, lt);
    if (ge->operation.type != ToString(OperationType::kGreaterEqual) ||
        lt->operation.type != ToString(OperationType::kLess)) {
      continue;
    }
    const auto ge_inputs = graph.FindInputs(ge->id);
    const auto lt_inputs = graph.FindInputs(lt->id);
    if (ge_inputs.size() != 2 || lt_inputs.size() != 2 || ge_inputs[0]->id != lt_inputs[0]->id) {
      continue;
    }
    const std::vector<int32_t>* keys = IntConstant(graph, ge_inputs[1]);
    const std::vector<int32_t>* shifted = IntConstant(graph, lt_inputs[1]);
    if (keys == nullptr || shifted == nullptr || keys->empty() || keys->size() != shifted->size()) {
      continue;
    }
    // Agello (PERFORMANCE.md, E2): the keys are positions, one per cache slot: a
    // run of consecutive integers. The rotary fix's block selector (patch_rope.py
    // --split) also compares against one-element constants 512 apart and read as a
    // 512 window; harmless while the real window was 512 too, but with ring-buffer
    // local caches (ring_kv.py) it was the only match, and the window kernels then
    // started each local layer at an absolute position the ring doesn't hold.
    bool consecutive = keys->size() > 1;
    for (size_t i = 1; consecutive && i < keys->size(); ++i) {
      consecutive = (*keys)[i] == (*keys)[0] + static_cast<int32_t>(i);
    }
    if (!consecutive) continue;
    const int w = (*shifted)[0] - (*keys)[0];
    bool uniform = w > 0;
    for (size_t i = 0; uniform && i < keys->size(); ++i) {
      uniform = (*shifted)[i] - (*keys)[i] == w;
    }
    if (!uniform || (window && *window != w)) return std::nullopt;
    window = w;
  }
  return window;
}

namespace {

// The nodes of one attention tail.
struct Tail {
  Node* qk = nullptr;
  Value* logits = nullptr;
  Node* select = nullptr;
  Value* mask = nullptr;
  Value* fill = nullptr;
  Value* selected = nullptr;  // The select's output.
  Node* maximum = nullptr;    // Optional capping (enable_infinite_float_capping).
  Node* minimum = nullptr;
  std::optional<float> clamp_low;
  std::optional<float> clamp_high;
  Node* softmax = nullptr;
  Value* probabilities = nullptr;
  Node* pv = nullptr;
  Value* params = nullptr;
};

// Matches the tail that ends in `softmax`; nullopt if anything differs from the
// expected shape of a Gemma 4 runtime_bmm attention.
std::optional<Tail> MatchTail(const GraphFloat32& graph, Node* softmax) {
  Tail tail;
  tail.softmax = softmax;
  if (const auto* attr = std::any_cast<::ml_drift::SoftmaxAttributes>(&softmax->operation.attributes);
      attr != nullptr && attr->axis != ::ml_drift::Axis::kChannels) {
    return std::nullopt;
  }
  const auto softmax_inputs = graph.FindInputs(softmax->id);
  const auto softmax_outputs = graph.FindOutputs(softmax->id);
  if (softmax_inputs.size() != 1 || softmax_outputs.size() != 1) return std::nullopt;
  tail.probabilities = softmax_outputs[0];

  // The PV product: the FC form (external weights), bounded by its source,
  // reading [probabilities, weights, params] with no bias yet.
  if (!SingleUse(graph, tail.probabilities)) return std::nullopt;
  tail.pv = graph.FindConsumers(tail.probabilities->id)[0];
  if (tail.pv->operation.type != kRuntimeBatchedMatMulType) return std::nullopt;
  const auto* pv_attr =
      std::any_cast<RuntimeBatchedMatMulAttributes>(&tail.pv->operation.attributes);
  if (pv_attr == nullptr || !pv_attr->external_weights.has_value() ||
      !pv_attr->runtime_check.src_end_ch_index.has_value() ||
      pv_attr->runtime_check.dst_end_ch_index.has_value() || pv_attr->src_exp_input) {
    return std::nullopt;
  }
  const auto pv_inputs = graph.FindInputs(tail.pv->id);
  if (pv_inputs.size() != 3 || pv_inputs[0]->id != tail.probabilities->id) return std::nullopt;
  tail.params = pv_inputs[2];

  // Back from the softmax: [MINIMUM <- MAXIMUM <-] SELECT_V2.
  Value* value = softmax_inputs[0];
  if (!SingleUse(graph, value)) return std::nullopt;
  Node* producer = graph.FindProducer(value->id);
  if (producer != nullptr && producer->operation.type == ToString(OperationType::kMinimum)) {
    tail.minimum = producer;
    tail.clamp_high = ScalarParam(*producer);
    const auto inputs = graph.FindInputs(producer->id);
    if (!tail.clamp_high || inputs.size() != 1) return std::nullopt;
    value = inputs[0];
    if (!SingleUse(graph, value)) return std::nullopt;
    producer = graph.FindProducer(value->id);
    if (producer == nullptr || producer->operation.type != ToString(OperationType::kMaximum)) {
      return std::nullopt;
    }
    tail.maximum = producer;
    tail.clamp_low = ScalarParam(*producer);
    const auto max_inputs = graph.FindInputs(producer->id);
    if (!tail.clamp_low || max_inputs.size() != 1) return std::nullopt;
    value = max_inputs[0];
    if (!SingleUse(graph, value)) return std::nullopt;
    producer = graph.FindProducer(value->id);
  }
  if (producer == nullptr || producer->operation.type != ToString(OperationType::kSelectV2)) {
    return std::nullopt;
  }
  tail.select = producer;
  tail.selected = value;
  const auto select_inputs = graph.FindInputs(tail.select->id);
  if (select_inputs.size() != 3) return std::nullopt;
  tail.mask = select_inputs[0];
  tail.logits = select_inputs[1];
  tail.fill = select_inputs[2];
  if (tail.mask->tensor.type != ::ml_drift::DataType::kBool) return std::nullopt;
  const Node* fill_producer = graph.FindProducer(tail.fill->id);
  if (fill_producer == nullptr ||
      fill_producer->operation.type != ToString(OperationType::kConstant) ||
      tail.fill->tensor.shape.DimensionsProduct() != 1) {
    return std::nullopt;
  }

  // The QK product: bounded by its destination on the same params entry.
  if (!SingleUse(graph, tail.logits)) return std::nullopt;
  tail.qk = graph.FindProducer(tail.logits->id);
  if (tail.qk == nullptr || tail.qk->operation.type != kRuntimeBatchedMatMulType) {
    return std::nullopt;
  }
  const auto* qk_attr =
      std::any_cast<RuntimeBatchedMatMulAttributes>(&tail.qk->operation.attributes);
  if (qk_attr == nullptr || !qk_attr->runtime_check.dst_end_ch_index.has_value() ||
      *qk_attr->runtime_check.dst_end_ch_index != *pv_attr->runtime_check.src_end_ch_index) {
    return std::nullopt;
  }
  const auto qk_inputs = graph.FindInputs(tail.qk->id);
  if (qk_inputs.empty() || qk_inputs.back()->id != tail.params->id) return std::nullopt;
  if (tail.logits->tensor.shape != tail.selected->tensor.shape) return std::nullopt;
  return tail;
}

absl::Status Rewrite(GraphFloat32* graph, const Tail& tail, std::optional<int> window,
                     std::optional<int> global_window = std::nullopt) {
  // 1. The select becomes the bounded mask: (logits, mask), with the capping
  //    folded in. Masked keys get what the capped select gave them.
  for (Value* input : {tail.mask, tail.logits, tail.fill}) {
    ABSL_RETURN_IF_ERROR(graph->RemoveConsumer(tail.select->id, input->id));
  }
  graph->AddConsumer(tail.select->id, tail.logits->id);
  graph->AddConsumer(tail.select->id, tail.mask->id);
  BoundedMaskAttributes mask_attr;
  if (tail.clamp_low && tail.clamp_high) {
    mask_attr.mask_value = *tail.clamp_low;
    mask_attr.clamp_low = tail.clamp_low;
    mask_attr.clamp_high = tail.clamp_high;
  }
  tail.select->operation.type = kBoundedMaskType;
  tail.select->operation.attributes = mask_attr;
  if (graph->FindConsumers(tail.fill->id).empty()) {
    const Node* fill_producer = graph->FindProducer(tail.fill->id);
    if (fill_producer != nullptr) ABSL_RETURN_IF_ERROR(graph->DeleteNode(fill_producer->id));
    ABSL_RETURN_IF_ERROR(graph->DeleteValue(tail.fill->id));
  }

  // 2. Drop the capping nodes: the softmax reads the masked logits directly.
  if (tail.minimum != nullptr) {
    const Value* capped = graph->FindInputs(tail.softmax->id)[0];
    const Value* clamped_low = graph->FindOutputs(tail.maximum->id)[0];
    ABSL_RETURN_IF_ERROR(graph->ReplaceInput(tail.softmax->id, capped->id, tail.selected->id));
    const auto capped_id = capped->id;
    const auto clamped_low_id = clamped_low->id;
    ABSL_RETURN_IF_ERROR(graph->DeleteNode(tail.minimum->id));
    ABSL_RETURN_IF_ERROR(graph->DeleteNode(tail.maximum->id));
    ABSL_RETURN_IF_ERROR(graph->DeleteValue(capped_id));
    ABSL_RETURN_IF_ERROR(graph->DeleteValue(clamped_low_id));
  }

  // 3. The softmax becomes the bounded reduction: (masked logits, params) ->
  //    per row (1 / sum, max) in a 4-channel tensor.
  auto* qk_attr = std::any_cast<RuntimeBatchedMatMulAttributes>(&tail.qk->operation.attributes);
  auto* pv_attr = std::any_cast<RuntimeBatchedMatMulAttributes>(&tail.pv->operation.attributes);
  graph->AddConsumer(tail.softmax->id, tail.params->id);
  tail.softmax->operation.type = kBoundedSoftmaxReduceType;
  BoundedSoftmaxReduceAttributes reduce_attr{.end_ch_index =
                                                 *pv_attr->runtime_check.src_end_ch_index};
  {
    const ::ml_drift::BHWC& scores = tail.selected->tensor.shape;
    reduce_attr.whole_rows = scores.b * scores.h * scores.w;
  }

  // A sliding-window layer (not global) also starts at its window: the QK product
  // skips the keys below it (its fused mask masks them), the reduction and the PV
  // product start there. params[0] is the first new token's position.
  if (window && qk_attr->is_global.has_value() && !*qk_attr->is_global) {
    qk_attr->runtime_check.window_pos_index = 0;
    qk_attr->runtime_check.window_size = *window;
    qk_attr->runtime_check.dst_window = true;
    pv_attr->runtime_check.window_pos_index = 0;
    pv_attr->runtime_check.window_size = *window;
    pv_attr->runtime_check.src_window = true;
    reduce_attr.window_pos_index = 0;
    reduce_attr.window_size = *window;
  }
  // Agello (engine review, step 6): the drafter's global layer reads only its last
  // `global_window` positions. Its keys below the window aren't masked, but the reduction and
  // the PV product start at the window, so they don't count: attention over the window. Only the
  // drafts change; verify decides what is kept.
  if (global_window && qk_attr->is_global.has_value() && *qk_attr->is_global) {
    qk_attr->runtime_check.window_pos_index = 0;
    qk_attr->runtime_check.window_size = *global_window;
    qk_attr->runtime_check.dst_window = true;
    pv_attr->runtime_check.window_pos_index = 0;
    pv_attr->runtime_check.window_size = *global_window;
    pv_attr->runtime_check.src_window = true;
    reduce_attr.window_pos_index = 0;
    reduce_attr.window_size = *global_window;
  }
  tail.softmax->operation.attributes = reduce_attr;
  ::ml_drift::BHWC& reduced_shape = tail.probabilities->tensor.shape;
  reduced_shape.c = 4;

  // 4. The PV product reads the masked logits, with the reduction as src_exp:
  //    [masked logits, weights, reduction, params].
  ABSL_RETURN_IF_ERROR(
      graph->ReplaceInput(tail.pv->id, tail.probabilities->id, tail.selected->id));
  ABSL_RETURN_IF_ERROR(graph->RemoveConsumer(tail.pv->id, tail.params->id));
  graph->AddConsumer(tail.pv->id, tail.probabilities->id);
  graph->AddConsumer(tail.pv->id, tail.params->id);
  pv_attr->src_exp_input = true;
  return absl::OkStatus();
}

// Agello (PERFORMANCE.md, D2): a prefill's attention in slices of query rows. A rewritten tail
// computes a score tensor of [kv heads, query rows, keys]: at a 1,024-token prefill (4,096 query
// rows, eight query heads over two KV heads) and a 32K window that's 524 MB, 2.1 GB at 128K, and
// the reason long windows were held to 128-token chunks. Every query row attends independently,
// so the tail runs on slices of the rows, each with a score tensor of at most kMaxScoreBytes,
// with the same kernels and the same results, and the slices' outputs are concatenated. The
// mask is sliced the same way, once for all the layers that read it.
constexpr int64_t kMaxScoreBytes = 256ll << 20;

// AGELLO_SLICE_SCORE_MB overrides the limit (for the D2 sweep; part of the program cache key).
int64_t MaxScoreBytes() {
  static const int64_t bytes = [] {
    const char* value = std::getenv("AGELLO_SLICE_SCORE_MB");
    const int64_t megabytes = value != nullptr ? std::atoll(value) : 0;
    return megabytes > 0 ? (megabytes << 20) : kMaxScoreBytes;
  }();
  return bytes;
}

int RowSlices(const Tail& tail) {
  const ::ml_drift::BHWC& scores = tail.selected->tensor.shape;
  const int64_t bytes = 2ll * scores.b * scores.h * scores.w * scores.c;
  int slices = 1;
  // At least 256 rows a slice: the graph builder picks other product kernels for runs of 128 rows
  // or fewer, which sum differently (the D2 sweep: 128-row slices changed replies).
  while (bytes / slices > MaxScoreBytes() && scores.w % (2 * slices) == 0 &&
         scores.w / (2 * slices) >= 256) {
    slices *= 2;
  }
  return slices;
}

using MaskSlices = absl::flat_hash_map<std::pair<::ml_drift::ValueId, int>, Value*>;

absl::Status SliceRows(GraphFloat32* graph, const Tail& tail, int slices,
                       MaskSlices* mask_slices) {
  const auto qk_inputs = graph->FindInputs(tail.qk->id);
  const auto mask_inputs = graph->FindInputs(tail.select->id);
  const auto pv_inputs = graph->FindInputs(tail.pv->id);
  const auto pv_outputs = graph->FindOutputs(tail.pv->id);
  if (qk_inputs.size() != 3 || mask_inputs.size() != 2 || pv_inputs.size() != 4 ||
      pv_outputs.size() != 1) {
    ABSL_LOG(WARNING) << "Agello: an attention tail with a context-sized score tensor stays "
                         "whole (unexpected inputs).";
    return absl::OkStatus();
  }
  Value* q = qk_inputs[0];
  Value* k = qk_inputs[1];
  Value* params = qk_inputs[2];
  Value* mask = mask_inputs[1];
  Value* v = pv_inputs[1];
  Value* out = pv_outputs[0];
  const int rows = tail.selected->tensor.shape.w;
  const int slice_rows = rows / slices;
  // Agello (engine review, step 4): a mask with one row per chunk row (the bounds, [1, 1, R, 1],
  // read by query row w % R) is sliced to a slice's rows only when the slice lies within one
  // query head's R rows; a slice of whole heads reads it whole. A [rows, T] mask is sliced like
  // the rows.
  const int chunk_rows = mask->tensor.shape.w;
  const bool per_chunk_row = chunk_rows != rows && chunk_rows > 0 && rows % chunk_rows == 0;
  const bool per_chunk_row_ok =
      per_chunk_row && ((slice_rows <= chunk_rows && chunk_rows % slice_rows == 0) ||
                        slice_rows % chunk_rows == 0);
  if (q->tensor.shape.w != rows || out->tensor.shape.w != rows ||
      (per_chunk_row ? !per_chunk_row_ok : mask->tensor.shape.w != rows)) {
    ABSL_LOG(WARNING) << "Agello: an attention tail with a context-sized score tensor stays "
                         "whole (unexpected row shapes).";
    return absl::OkStatus();
  }
  ::ml_drift::NodeId cursor = tail.pv->id;
  auto insert = [&](const std::string& type, std::any attributes,
                    const std::vector<Value*>& inputs, Value* output) -> absl::Status {
    Node* node = nullptr;
    ABSL_RETURN_IF_ERROR(graph->InsertNodeAfter(cursor, &node));
    cursor = node->id;
    node->operation.type = type;
    node->operation.attributes = std::move(attributes);
    for (Value* input : inputs) graph->AddConsumer(node->id, input->id);
    graph->SetProducer(node->id, output->id);
    return absl::OkStatus();
  };
  auto value_like = [&](const Value* like, int width) {
    Value* value = graph->NewValue();
    value->tensor = like->tensor;
    value->tensor.ref = -1;
    value->tensor.is_variable_input = false;
    value->tensor.shape.w = width;
    value->quant_params = like->quant_params;
    return value;
  };
  auto row_slice = [&](const ::ml_drift::BHWC& shape, int start) {
    ::ml_drift::SliceAttributes attr;
    attr.starts = ::ml_drift::BHWC(0, 0, start, 0);
    attr.ends = ::ml_drift::BHWC(shape.b, shape.h, start + slice_rows, shape.c);
    attr.strides = ::ml_drift::BHWC(1, 1, 1, 1);
    return attr;
  };
  const std::string slice_type = ToString(OperationType::kSlice);
  std::vector<Value*> outputs;
  for (int i = 0; i < slices; ++i) {
    const int start = i * slice_rows;
    Value* mask_slice = mask;
    if (!per_chunk_row || slice_rows < chunk_rows) {
      const int mask_start = per_chunk_row ? start % chunk_rows : start;
      Value*& cached = (*mask_slices)[{mask->id, mask_start}];
      if (cached == nullptr) {
        cached = value_like(mask, slice_rows);
        ABSL_RETURN_IF_ERROR(insert(slice_type, row_slice(mask->tensor.shape, mask_start),
                                    {mask}, cached));
      }
      mask_slice = cached;
    }
    Value* q_slice = value_like(q, slice_rows);
    ABSL_RETURN_IF_ERROR(insert(slice_type, row_slice(q->tensor.shape, start), {q}, q_slice));
    Value* logits = value_like(tail.logits, slice_rows);
    ABSL_RETURN_IF_ERROR(insert(tail.qk->operation.type, tail.qk->operation.attributes,
                                {q_slice, k, params}, logits));
    Value* masked = value_like(tail.selected, slice_rows);
    ABSL_RETURN_IF_ERROR(insert(tail.select->operation.type, tail.select->operation.attributes,
                                {logits, mask_slice}, masked));
    Value* reduced = value_like(tail.probabilities, slice_rows);
    ABSL_RETURN_IF_ERROR(insert(tail.softmax->operation.type,
                                tail.softmax->operation.attributes, {masked, params}, reduced));
    Value* output = value_like(out, slice_rows);
    ABSL_RETURN_IF_ERROR(insert(tail.pv->operation.type, tail.pv->operation.attributes,
                                {masked, v, reduced, params}, output));
    outputs.push_back(output);
  }
  // The whole tail goes; the concatenation produces its output.
  const ::ml_drift::ValueId logits_id = tail.logits->id;
  const ::ml_drift::ValueId selected_id = tail.selected->id;
  const ::ml_drift::ValueId probabilities_id = tail.probabilities->id;
  for (Node* node : {tail.qk, tail.select, tail.softmax, tail.pv}) {
    ABSL_RETURN_IF_ERROR(graph->DeleteNode(node->id));
  }
  for (::ml_drift::ValueId id : {logits_id, selected_id, probabilities_id}) {
    ABSL_RETURN_IF_ERROR(graph->DeleteValue(id));
  }
  ::ml_drift::ConcatAttributes concat;
  concat.axis = ::ml_drift::Axis::kWidth;
  return insert(ToString(OperationType::kConcat), concat, outputs, out);
}

// Agello (engine review, step 4): the boolean [1, 1, R, keys] condition a global layer's mask is
// copied from, one copy per query head: mask <- RESHAPE <- NOT_EQUAL(0) <- CONCAT(height) <-
// [COPY] <- CAST <- condition. Null if the mask isn't built that way.
Value* MatchRowMask(const GraphFloat32& graph, const Value* mask, int keys) {
  const auto producer_of = [&](const Value* value, OperationType type) -> Node* {
    Node* node = graph.FindProducer(value->id);
    return node != nullptr && node->operation.type == ToString(type) ? node : nullptr;
  };
  const auto single_input = [&](const Node* node) -> Value* {
    const auto inputs = graph.FindInputs(node->id);
    return inputs.size() == 1 ? inputs[0] : nullptr;
  };
  Node* reshape = producer_of(mask, OperationType::kReshape);
  Value* value = reshape ? single_input(reshape) : nullptr;
  Node* not_equal = value ? producer_of(value, OperationType::kNotEqual) : nullptr;
  if (not_equal == nullptr || ScalarParam(*not_equal) != std::optional<float>(0.0f)) {
    return nullptr;
  }
  value = single_input(not_equal);
  Node* concat = value ? producer_of(value, OperationType::kConcat) : nullptr;
  if (concat == nullptr) return nullptr;
  const auto* concat_attr =
      std::any_cast<::ml_drift::ConcatAttributes>(&concat->operation.attributes);
  if (concat_attr == nullptr || concat_attr->axis != ::ml_drift::Axis::kHeight) return nullptr;
  const Value* cast_value = nullptr;
  const auto concat_inputs = graph.FindInputs(concat->id);
  for (const Value* input : concat_inputs) {
    const Value* source = input;
    if (Node* copy = producer_of(source, OperationType::kCopy)) source = single_input(copy);
    if (source == nullptr || (cast_value != nullptr && source != cast_value)) return nullptr;
    cast_value = source;
  }
  Node* cast = cast_value ? producer_of(cast_value, OperationType::kCast) : nullptr;
  Value* condition = cast ? single_input(cast) : nullptr;
  if (condition == nullptr) return nullptr;
  const ::ml_drift::BHWC& c = condition->tensor.shape;
  const int heads = static_cast<int>(concat_inputs.size());
  const bool row_shaped = condition->tensor.type == ::ml_drift::DataType::kBool && c.b == 1 &&
                          c.h == 1 && c.c == keys && mask->tensor.shape.w == heads * c.w;
  return row_shaped ? condition : nullptr;
}

// Deletes `value`'s producer and, going back, every producer whose outputs nothing reads.
// `graph_outputs` are the graph's outputs before the rewrite: GraphFloat32::IsGraphOutput also
// counts any value with no reader, which is every value this cleans up.
absl::Status DeleteDeadChain(GraphFloat32* graph, Value* value,
                             const absl::flat_hash_set<::ml_drift::ValueId>& graph_outputs) {
  std::vector<Value*> pending{value};
  while (!pending.empty()) {
    Value* current = pending.back();
    pending.pop_back();
    if (graph_outputs.contains(current->id) || !graph->FindConsumers(current->id).empty()) {
      continue;
    }
    Node* producer = graph->FindProducer(current->id);
    if (producer == nullptr) continue;
    const auto outputs = graph->FindOutputs(producer->id);
    bool dead = true;
    for (const Value* output : outputs) {
      dead = dead && !graph_outputs.contains(output->id) &&
             graph->FindConsumers(output->id).empty();
    }
    if (!dead) continue;
    const auto inputs = graph->FindInputs(producer->id);
    ABSL_RETURN_IF_ERROR(graph->DeleteNode(producer->id));
    for (const Value* output : outputs) ABSL_RETURN_IF_ERROR(graph->DeleteValue(output->id));
    for (Value* input : inputs) pending.push_back(input);
  }
  return absl::OkStatus();
}

// A kMaskBoundsType node on `condition`, right after its first reader, so the boolean mask can
// die early: [1, 1, R, 1] int32.
absl::StatusOr<Value*> AddMaskBounds(GraphFloat32* graph, Value* condition) {
  const Node* first_reader = nullptr;
  for (const Node* node : graph->nodes()) {
    if (node == nullptr) continue;
    for (const Value* input : graph->FindInputs(node->id)) {
      if (input->id == condition->id) {
        first_reader = node;
        break;
      }
    }
    if (first_reader != nullptr) break;
  }
  if (first_reader == nullptr) return absl::InternalError("A mask with no reader");
  Node* node = nullptr;
  ABSL_RETURN_IF_ERROR(graph->InsertNodeAfter(first_reader->id, &node));
  node->operation.type = kMaskBoundsType;
  Value* bounds = graph->NewValue();
  bounds->tensor = condition->tensor;
  bounds->tensor.ref = -1;
  bounds->tensor.is_variable_input = false;
  bounds->tensor.type = ::ml_drift::DataType::kInt32;
  bounds->tensor.shape.c = 1;
  graph->AddConsumer(node->id, condition->id);
  graph->SetProducer(node->id, bounds->id);
  return bounds;
}

// Agello (engine review, step 4): the flag the graph makes from the same mask, "does this row
// keep any key": condition -> RESHAPE -> CAST -> REDUCE_MAXIMUM(channels) -> NOT_EQUAL(0), read
// after each global layer to zero padded rows. Its cast holds a 16-bit copy of the whole mask for
// most of the run (128 MB at 64K). Null if there's none.
Value* MatchRowFlag(const GraphFloat32& graph, const Value* condition) {
  const auto only_consumer = [&](const Value* value, OperationType type) -> Node* {
    const auto consumers = graph.FindConsumers(value->id);
    return consumers.size() == 1 && consumers[0]->operation.type == ToString(type) ? consumers[0]
                                                                                  : nullptr;
  };
  const auto output_of = [&](const Node* node) -> Value* {
    const auto outputs = graph.FindOutputs(node->id);
    return outputs.size() == 1 ? outputs[0] : nullptr;
  };
  for (Node* reshape : graph.FindConsumers(condition->id)) {
    if (reshape->operation.type != ToString(OperationType::kReshape)) continue;
    Value* value = output_of(reshape);
    Node* cast = value ? only_consumer(value, OperationType::kCast) : nullptr;
    value = cast ? output_of(cast) : nullptr;
    Node* reduce = value ? only_consumer(value, OperationType::kReduceMaximum) : nullptr;
    if (reduce == nullptr) continue;
    const auto* reduce_attr = std::any_cast<::ml_drift::ReduceAttributes>(&reduce->operation.attributes);
    if (reduce_attr == nullptr || reduce_attr->dims != std::set<::ml_drift::Axis>{::ml_drift::Axis::kChannels}) {
      continue;
    }
    value = output_of(reduce);
    Node* not_equal = value ? only_consumer(value, OperationType::kNotEqual) : nullptr;
    if (not_equal == nullptr || ScalarParam(*not_equal) != std::optional<float>(0.0f)) continue;
    Value* flag = output_of(not_equal);
    if (flag == nullptr || flag->tensor.type != ::ml_drift::DataType::kBool) continue;
    const ::ml_drift::BHWC& s = flag->tensor.shape;
    const int rows = condition->tensor.shape.w;
    if (s.b == 1 && s.c == 1 && ((s.h == rows && s.w == 1) || (s.w == rows && s.h == 1))) {
      return flag;
    }
  }
  return nullptr;
}

// Replaces `flag` with a kBoundsFlagType node on `bounds`, after the bounds' producer.
absl::StatusOr<Value*> AddBoundsFlag(GraphFloat32* graph, Value* bounds, Value* flag) {
  const Node* producer = graph->FindProducer(bounds->id);
  if (producer == nullptr) return absl::InternalError("Bounds with no producer");
  Node* node = nullptr;
  ABSL_RETURN_IF_ERROR(graph->InsertNodeAfter(producer->id, &node));
  node->operation.type = kBoundsFlagType;
  Value* replacement = graph->NewValue();
  replacement->tensor = flag->tensor;
  replacement->tensor.ref = -1;
  replacement->tensor.is_variable_input = false;
  graph->AddConsumer(node->id, bounds->id);
  graph->SetProducer(node->id, replacement->id);
  for (Node* reader : graph->FindConsumers(flag->id)) {
    ABSL_RETURN_IF_ERROR(graph->ReplaceInput(reader->id, flag->id, replacement->id));
  }
  return replacement;
}

std::optional<float> UniformScale(
    const std::optional<::ml_drift::Tensor<::ml_drift::Linear, ::ml_drift::DataType::kFloat32>>&
        scale) {
  if (!scale.has_value() || scale->data.empty()) return std::nullopt;
  const float s = scale->data[0];
  for (float v : scale->data) {
    if (v != s) return std::nullopt;
  }
  return s;
}

// Agello (P2.3): collapses a rewritten prefill attention tail
// (QK -> BoundedMask -> BoundedSoftmaxReduce -> PV) into one fused MSL 4.1
// FlashAttention node reading the INT8 KV cache directly without materializing
// the [H_kv, Q_rows, cache_size] score matrix. Returns true if fused.
absl::StatusOr<bool> TryFuseFlashAttention(GraphFloat32* graph, const Tail& tail) {
  const auto qk_inputs = graph->FindInputs(tail.qk->id);
  const auto mask_inputs = graph->FindInputs(tail.select->id);
  const auto pv_inputs = graph->FindInputs(tail.pv->id);
  const auto pv_outputs = graph->FindOutputs(tail.pv->id);
  if (qk_inputs.size() != 3 || mask_inputs.size() != 2 || pv_inputs.size() != 4 ||
      pv_outputs.size() != 1) {
    return false;
  }
  Value* q = qk_inputs[0];
  Value* k = qk_inputs[1];
  Value* params = qk_inputs[2];
  Value* mask = mask_inputs[1];
  Value* v = pv_inputs[1];
  Value* out = pv_outputs[0];

  const ::ml_drift::BHWC& q_shape = q->tensor.shape;
  const int rows = q_shape.w;
  const int head_dim = q_shape.c;
  const int kv_heads = q_shape.h;
  const int cache_size = tail.selected->tensor.shape.c;
  if (q_shape.b != 1 || out->tensor.shape != q_shape || rows < 256 ||
      (head_dim != 256 && head_dim != 512) || kv_heads < 1 || cache_size < 4 ||
      (cache_size % 4) != 0) {
    return false;
  }

  const auto* qk_attr =
      std::any_cast<RuntimeBatchedMatMulAttributes>(&tail.qk->operation.attributes);
  const auto* pv_attr =
      std::any_cast<RuntimeBatchedMatMulAttributes>(&tail.pv->operation.attributes);
  const auto* mask_attr =
      std::any_cast<BoundedMaskAttributes>(&tail.select->operation.attributes);
  const auto* reduce_attr =
      std::any_cast<BoundedSoftmaxReduceAttributes>(&tail.softmax->operation.attributes);
  if (qk_attr == nullptr || pv_attr == nullptr || mask_attr == nullptr ||
      reduce_attr == nullptr) {
    return false;
  }
  if (!qk_attr->external_weights.has_value() || !pv_attr->external_weights.has_value()) {
    return false;
  }
  const auto& k_ext = *qk_attr->external_weights;
  const auto& v_ext = *pv_attr->external_weights;
  if (k_ext.desc.type != ::ml_drift::DataType::kUint8 ||
      v_ext.desc.type != ::ml_drift::DataType::kUint8 ||
      k_ext.desc.layout != ::ml_drift::WeightsLayout::kOSpatialIOGroupO4I4 ||
      v_ext.desc.layout != ::ml_drift::WeightsLayout::kOSpatialIOGroupI4O4 ||
      k_ext.weights_shape != ::ml_drift::OHWI(cache_size, kv_heads, 1, head_dim) ||
      v_ext.weights_shape != ::ml_drift::OHWI(head_dim, kv_heads, 1, cache_size)) {
    return false;
  }
  if (k_ext.desc.output_group_size != cache_size / 4 ||
      v_ext.desc.output_group_size != head_dim / 4) {
    return false;
  }
  const auto scale_k = UniformScale(qk_attr->scale);
  const auto scale_v = UniformScale(pv_attr->scale);
  if (!scale_k.has_value() || !scale_v.has_value()) {
    return false;
  }

  // Ring-buffer caches and packed groups are handled by the general path.
  if (qk_attr->runtime_check.src_end_ch_index.has_value() ||
      pv_attr->runtime_check.dst_end_ch_index.has_value()) {
    return false;
  }

  FlashAttentionAttributes attr;
  attr.head_dim = head_dim;
  attr.kv_heads = kv_heads;
  attr.cache_size = cache_size;
  attr.scale_k = *scale_k;
  attr.scale_v = *scale_v;
  attr.mask_value = mask_attr->mask_value;
  attr.clamp_low = mask_attr->clamp_low;
  attr.clamp_high = mask_attr->clamp_high;
  attr.end_ch_index = reduce_attr->end_ch_index;
  attr.window_pos_index = reduce_attr->window_pos_index;
  attr.window_size = reduce_attr->window_size;
  attr.row_end = mask_attr->row_end;

  Node* flash_node = nullptr;
  ABSL_RETURN_IF_ERROR(graph->InsertNodeAfter(tail.pv->id, &flash_node));
  flash_node->operation.type = kFlashAttentionType;
  flash_node->operation.attributes = std::move(attr);
  for (Value* input : {q, k, v, mask, params}) {
    graph->AddConsumer(flash_node->id, input->id);
  }

  const ::ml_drift::ValueId logits_id = tail.logits->id;
  const ::ml_drift::ValueId selected_id = tail.selected->id;
  const ::ml_drift::ValueId probabilities_id = tail.probabilities->id;
  for (Node* node : {tail.qk, tail.select, tail.softmax, tail.pv}) {
    ABSL_RETURN_IF_ERROR(graph->DeleteNode(node->id));
  }
  for (::ml_drift::ValueId id : {logits_id, selected_id, probabilities_id}) {
    ABSL_RETURN_IF_ERROR(graph->DeleteValue(id));
  }
  graph->SetProducer(flash_node->id, out->id);
  return true;
}

}  // namespace

bool FlashAttentionEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("AGELLO_FLASH_ATTENTION");
    return value == nullptr || value[0] != '0';
  }();
  return enabled;
}

bool MaskBoundsEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("AGELLO_MASK_BOUNDS");
    return value == nullptr || value[0] != '0';
  }();
  return enabled;
}

bool SliceAttentionEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("AGELLO_SLICE_ATTENTION");
    return value == nullptr || value[0] != '0';
  }();
  return enabled;
}

bool BoundedAttentionEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("AGELLO_BOUNDED_ATTENTION");
    return value == nullptr || value[0] != '0';
  }();
  return enabled;
}

bool BoundedWindowEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("AGELLO_BOUNDED_WINDOW");
    return value == nullptr || value[0] != '0';
  }();
  return enabled;
}

absl::StatusOr<int> FuseBoundedAttentionSoftmax(GraphFloat32* graph,
                                                bool enable_flash_attention) {
  if (graph == nullptr) return 0;
  const bool flash_enabled = enable_flash_attention && FlashAttentionEnabled();
  const std::optional<int> window =
      BoundedWindowEnabled() ? FindSlidingWindow(*graph) : std::nullopt;
  std::vector<::ml_drift::NodeId> softmax_ids;
  for (Node* node : graph->nodes()) {
    if (node != nullptr && node->operation.type == ToString(OperationType::kSoftmax)) {
      softmax_ids.push_back(node->id);
    }
  }
  int rewritten = 0;
  int flashed = 0;
  MaskSlices mask_slices;
  // Agello (engine review, step 6): the drafter's graph is the one with a handful of attention
  // tails (four: three local, one global; the main graphs have 42). AGELLO_DRAFTER_WINDOW sets
  // its global layer's window in positions; 0 or unset keeps the whole cache.
  static const int drafter_window = [] {
    const char* value = std::getenv("AGELLO_DRAFTER_WINDOW");
    return value != nullptr ? std::max(0, std::atoi(value)) : 0;
  }();
  absl::flat_hash_map<::ml_drift::ValueId, Value*> bounds_of;
  std::vector<Value*> replaced_masks;
  int flags = 0;
  absl::flat_hash_set<::ml_drift::ValueId> graph_outputs;
  for (const Value* output : graph->outputs()) graph_outputs.insert(output->id);
  int bounded = 0;
  std::optional<int> global_window;
  if (drafter_window > 0) {
    int tails = 0;
    for (::ml_drift::NodeId id : softmax_ids) {
      if (Node* s = graph->GetNode(id); s != nullptr && MatchTail(*graph, s)) ++tails;
    }
    if (tails > 0 && tails <= 4) global_window = drafter_window;
  }
  for (::ml_drift::NodeId id : softmax_ids) {
    Node* softmax = graph->GetNode(id);
    if (softmax == nullptr) continue;
    std::optional<Tail> tail = MatchTail(*graph, softmax);
    if (!tail) continue;
    ABSL_RETURN_IF_ERROR(Rewrite(graph, *tail, window, global_window));
    ++rewritten;
    // Agello (engine review, step 4): a global layer's mask as per-row bounds. Only global
    // layers, whose mask is the runtime's own input (a prefix per row); a ring file's local
    // masks come out of floor_mod in the graph and wrap, so they aren't prefixes. And only a
    // prefill's (64 score rows or more): decode's and verify's few rows gain nothing, and their
    // bounds kernel, a thread a row scanning the whole window, cost decode 5 % (M6, 2026-11-15).
    constexpr int kMaskBoundsMinRows = 64;
    const auto* qk_attr =
        std::any_cast<RuntimeBatchedMatMulAttributes>(&tail->qk->operation.attributes);
    const bool global = qk_attr != nullptr && qk_attr->is_global.value_or(false);
    if (MaskBoundsEnabled() && global && tail->selected->tensor.shape.w >= kMaskBoundsMinRows) {
      Value* condition = MatchRowMask(*graph, tail->mask, tail->selected->tensor.shape.c);
      if (condition != nullptr && graph->FindProducer(condition->id) != nullptr) {
        condition = nullptr;  // not an input: built in the graph
      }
      if (condition != nullptr) {
        Value*& bounds = bounds_of[condition->id];
        if (bounds == nullptr) {
          ABSL_ASSIGN_OR_RETURN(bounds, AddMaskBounds(graph, condition));
          if (Value* flag = MatchRowFlag(*graph, condition)) {
            ABSL_RETURN_IF_ERROR(AddBoundsFlag(graph, bounds, flag).status());
            replaced_masks.push_back(flag);  // its chain goes with the masks'
            ++flags;
          }
        }
        ABSL_RETURN_IF_ERROR(graph->ReplaceInput(tail->select->id, tail->mask->id, bounds->id));
        auto* attr = std::any_cast<BoundedMaskAttributes>(&tail->select->operation.attributes);
        attr->row_end = true;
        ++bounded;
        if (std::find(replaced_masks.begin(), replaced_masks.end(), tail->mask) ==
            replaced_masks.end()) {
          replaced_masks.push_back(tail->mask);
        }
      }
    }
    // Agello (P2.3): fuse eligible prefill tails into MSL 4.1 FlashAttention
    // (zero materialized score matrix, skips SliceRows).
    bool fused_flash = false;
    if (flash_enabled) {
      ABSL_ASSIGN_OR_RETURN(fused_flash, TryFuseFlashAttention(graph, *tail));
      if (fused_flash) ++flashed;
    }
    // Agello (D2): a prefill's tail in slices of query rows.
    if (!fused_flash && SliceAttentionEnabled()) {
      if (const int slices = RowSlices(*tail); slices > 1) {
        ABSL_RETURN_IF_ERROR(SliceRows(graph, *tail, slices, &mask_slices));
      }
    }
  }
  for (Value* mask : replaced_masks) {
    ABSL_RETURN_IF_ERROR(DeleteDeadChain(graph, mask, graph_outputs));
  }
  if (std::getenv("AGELLO_LOG_ATTENTION") != nullptr) {
    int nodes = 0;
    for (const Node* node : graph->nodes()) nodes += node != nullptr;
    fprintf(stderr, "Agello attention: %d tails (%d flash), %d read row bounds (%zu bounds, %d row flags, "
            "%zu chains freed), %d nodes left\n", rewritten, flashed, bounded, bounds_of.size(),
            flags, replaced_masks.size(), nodes);
  }
  return rewritten;
}

}  // namespace litert::ml_drift
