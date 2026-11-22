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
#include <optional>

#include <gtest/gtest.h>
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/operations.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/runtime_batched_matmul_parser.h"

namespace litert::ml_drift {
namespace {

using ::ml_drift::BHWC;
using ::ml_drift::DataType;
using ::ml_drift::GraphFloat32;
using ::ml_drift::Node;
using ::ml_drift::OperationType;
using ::ml_drift::Value;

Value* NewValue(GraphFloat32& graph, DataType type, BHWC shape) {
  Value* value = graph.NewValue();
  value->tensor.type = type;
  value->tensor.shape = shape;
  return value;
}

Node* NewNode(GraphFloat32& graph, const std::string& type, std::initializer_list<Value*> inputs,
              Value* output) {
  Node* node = graph.NewNode();
  node->operation.type = type;
  for (Value* input : inputs) graph.AddConsumer(node->id, input->id);
  graph.SetProducer(node->id, output->id);
  return node;
}

// A Gemma 4 decode attention tail: QK -> SELECT_V2 -> MAXIMUM -> MINIMUM -> SOFTMAX -> PV.
struct Attention {
  GraphFloat32 graph;
  Value* logits;
  Value* mask;
  Value* selected;
  Value* probabilities;
  Value* params;
  Value* v;
  Node* select;
  Node* softmax;
  Node* pv;
  Value* out;
};

// A sliding-window mask: LOGICAL_AND(GREATER_EQUAL(p, keys), LESS(p, keys + window)).
void AddWindowMask(GraphFloat32& g, int keys, int window, int offset_of_last = 0) {
  Value* position = NewValue(g, DataType::kInt32, BHWC(1, 1, 1, 1));
  auto constant = [&](int shift, int last_extra) {
    Value* value = NewValue(g, DataType::kInt32, BHWC(1, 1, 1, keys));
    Node* node = g.NewNode();
    node->operation.type = ToString(OperationType::kConstant);
    ::ml_drift::TensorInt32 tensor;
    tensor.shape = BHWC(1, 1, 1, keys);
    for (int k = 0; k < keys; ++k) tensor.data.push_back(k + shift + (k == keys - 1 ? last_extra : 0));
    node->operation.attributes = ::ml_drift::ConstTensorAttributes{tensor};
    g.SetProducer(node->id, value->id);
    return value;
  };
  Value* ge = NewValue(g, DataType::kBool, BHWC(1, 1, 1, keys));
  NewNode(g, ToString(OperationType::kGreaterEqual), {position, constant(0, 0)}, ge);
  Value* lt = NewValue(g, DataType::kBool, BHWC(1, 1, 1, keys));
  NewNode(g, ToString(OperationType::kLess), {position, constant(window, offset_of_last)}, lt);
  Value* both = NewValue(g, DataType::kBool, BHWC(1, 1, 1, keys));
  NewNode(g, ToString(OperationType::kLogicalAnd), {ge, lt}, both);
}

void Build(Attention& a, std::optional<bool> is_global = std::nullopt, int rows = 4,
           int keys = 64, Value* shared_mask = nullptr, Value* shared_params = nullptr) {
  GraphFloat32& g = a.graph;
  Value* q = NewValue(g, DataType::kFloat32, BHWC(1, 2, rows, 256));
  Value* k = NewValue(g, DataType::kFloat32, BHWC(1, 1, 1, keys * 256));
  a.v = NewValue(g, DataType::kFloat32, BHWC(1, 1, 1, 256 * keys));
  a.params = shared_params ? shared_params : NewValue(g, DataType::kInt32, BHWC(1, 1, 1, 7));
  a.mask = shared_mask ? shared_mask : NewValue(g, DataType::kBool, BHWC(1, 1, rows, keys));
  a.logits = NewValue(g, DataType::kFloat32, BHWC(1, 2, rows, keys));

  RuntimeBatchedMatMulAttributes qk_attr;
  qk_attr.runtime_check.dst_end_ch_index = 2;
  qk_attr.external_weights = ExternalWeightsAttributes{};
  qk_attr.is_global = is_global;
  NewNode(g, kRuntimeBatchedMatMulType, {q, k, a.params}, a.logits)->operation.attributes =
      qk_attr;

  Value* fill = NewValue(g, DataType::kFloat32, BHWC(1, 1, 1, 1));
  Node* constant = g.NewNode();
  constant->operation.type = ToString(OperationType::kConstant);
  g.SetProducer(constant->id, fill->id);

  a.selected = NewValue(g, DataType::kFloat32, BHWC(1, 2, rows, keys));
  a.select = NewNode(g, ToString(OperationType::kSelectV2), {a.mask, a.logits, fill}, a.selected);
  Value* raised = NewValue(g, DataType::kFloat32, BHWC(1, 2, rows, keys));
  NewNode(g, ToString(OperationType::kMaximum), {a.selected}, raised)->operation.attributes =
      ::ml_drift::ElementwiseAttributes{::ml_drift::ScalarValue(-65504.0f)};
  Value* capped = NewValue(g, DataType::kFloat32, BHWC(1, 2, rows, keys));
  NewNode(g, ToString(OperationType::kMinimum), {raised}, capped)->operation.attributes =
      ::ml_drift::ElementwiseAttributes{::ml_drift::ScalarValue(65504.0f)};
  a.probabilities = NewValue(g, DataType::kFloat32, BHWC(1, 2, rows, keys));
  a.softmax = NewNode(g, ToString(OperationType::kSoftmax), {capped}, a.probabilities);
  a.softmax->operation.attributes = ::ml_drift::SoftmaxAttributes{::ml_drift::Axis::kChannels};

  Value* out = NewValue(g, DataType::kFloat32, BHWC(1, 2, rows, 256));
  a.out = out;
  RuntimeBatchedMatMulAttributes pv_attr;
  pv_attr.runtime_check.src_end_ch_index = 2;
  pv_attr.external_weights = ExternalWeightsAttributes{};
  a.pv = NewNode(g, kRuntimeBatchedMatMulType, {a.probabilities, a.v, a.params}, out);
  a.pv->operation.attributes = pv_attr;
}

TEST(BoundedAttentionTest, RewritesTheAttentionTail) {
  Attention a;
  Build(a);
  const auto rewritten = FuseBoundedAttentionSoftmax(&a.graph);
  ASSERT_TRUE(rewritten.ok()) << rewritten.status();
  EXPECT_EQ(*rewritten, 1);

  // QK, the bounded mask, the reduction and PV: the capping and the constant are gone.
  EXPECT_EQ(a.graph.nodes().size(), 4u);

  EXPECT_EQ(a.select->operation.type, kBoundedMaskType);
  const auto mask_inputs = a.graph.FindInputs(a.select->id);
  ASSERT_EQ(mask_inputs.size(), 2u);
  EXPECT_EQ(mask_inputs[0]->id, a.logits->id);
  EXPECT_EQ(mask_inputs[1]->id, a.mask->id);
  const auto& mask_attr = std::any_cast<const BoundedMaskAttributes&>(a.select->operation.attributes);
  EXPECT_EQ(mask_attr.mask_value, -65504.0f);
  EXPECT_EQ(mask_attr.clamp_low, -65504.0f);
  EXPECT_EQ(mask_attr.clamp_high, 65504.0f);

  EXPECT_EQ(a.softmax->operation.type, kBoundedSoftmaxReduceType);
  const auto reduce_inputs = a.graph.FindInputs(a.softmax->id);
  ASSERT_EQ(reduce_inputs.size(), 2u);
  EXPECT_EQ(reduce_inputs[0]->id, a.selected->id);
  EXPECT_EQ(reduce_inputs[1]->id, a.params->id);
  EXPECT_EQ(a.probabilities->tensor.shape, BHWC(1, 2, 4, 4));
  EXPECT_EQ(std::any_cast<const BoundedSoftmaxReduceAttributes&>(a.softmax->operation.attributes)
                .end_ch_index,
            2);

  const auto pv_inputs = a.graph.FindInputs(a.pv->id);
  ASSERT_EQ(pv_inputs.size(), 4u);
  EXPECT_EQ(pv_inputs[0]->id, a.selected->id);
  EXPECT_EQ(pv_inputs[1]->id, a.v->id);
  EXPECT_EQ(pv_inputs[2]->id, a.probabilities->id);
  EXPECT_EQ(pv_inputs[3]->id, a.params->id);
  EXPECT_TRUE(
      std::any_cast<const RuntimeBatchedMatMulAttributes&>(a.pv->operation.attributes).src_exp_input);
}

// Agello (D2): a 1,024-token prefill's tail (4,096 query rows) over 32,768 keys has a 512 MiB
// score tensor, so it runs in two slices of 2,048 rows, each at the 256 MiB limit.
TEST(BoundedAttentionTest, SlicesALargeTailByRows) {
  Attention a;
  Build(a, /*is_global=*/true, /*rows=*/4096, /*keys=*/32768);
  const auto rewritten = FuseBoundedAttentionSoftmax(&a.graph);
  ASSERT_TRUE(rewritten.ok()) << rewritten.status();
  EXPECT_EQ(*rewritten, 1);
  // Per slice: a Q slice, a mask slice, QK, the bounded mask, the reduction and PV; then the
  // concatenation, which produces the tail's output.
  EXPECT_EQ(a.graph.nodes().size(), 2u * 6 + 1);
  const Node* concat = a.graph.FindProducer(a.out->id);
  ASSERT_NE(concat, nullptr);
  EXPECT_EQ(concat->operation.type, ToString(OperationType::kConcat));
  const auto parts = a.graph.FindInputs(concat->id);
  ASSERT_EQ(parts.size(), 2u);
  int pv_nodes = 0;
  for (const Value* part : parts) {
    EXPECT_EQ(part->tensor.shape, BHWC(1, 2, 2048, 256));
    const Node* pv = a.graph.FindProducer(part->id);
    ASSERT_NE(pv, nullptr);
    ASSERT_EQ(pv->operation.type, kRuntimeBatchedMatMulType);
    ++pv_nodes;
    const auto pv_inputs = a.graph.FindInputs(pv->id);
    ASSERT_EQ(pv_inputs.size(), 4u);
    EXPECT_EQ(pv_inputs[0]->tensor.shape, BHWC(1, 2, 2048, 32768));
    EXPECT_EQ(pv_inputs[1]->id, a.v->id);
    EXPECT_EQ(pv_inputs[2]->tensor.shape, BHWC(1, 2, 2048, 4));
    EXPECT_EQ(pv_inputs[3]->id, a.params->id);
    // The masked logits come from a bounded mask over this slice's QK and mask slice.
    const Node* mask = a.graph.FindProducer(pv_inputs[0]->id);
    ASSERT_NE(mask, nullptr);
    EXPECT_EQ(mask->operation.type, kBoundedMaskType);
    const auto mask_inputs = a.graph.FindInputs(mask->id);
    ASSERT_EQ(mask_inputs.size(), 2u);
    EXPECT_EQ(mask_inputs[1]->tensor.shape, BHWC(1, 1, 2048, 32768));
    const Node* qk = a.graph.FindProducer(mask_inputs[0]->id);
    ASSERT_NE(qk, nullptr);
    const auto qk_inputs = a.graph.FindInputs(qk->id);
    ASSERT_EQ(qk_inputs.size(), 3u);
    EXPECT_EQ(qk_inputs[0]->tensor.shape, BHWC(1, 2, 2048, 256));
    EXPECT_EQ(qk_inputs[2]->id, a.params->id);
    // The slice's reduction is threaded for the whole tail's 2 x 4,096 rows, so a row sums in
    // the same order however many slices the window makes.
    const Node* reduce = a.graph.FindProducer(pv_inputs[2]->id);
    ASSERT_NE(reduce, nullptr);
    ASSERT_EQ(reduce->operation.type, kBoundedSoftmaxReduceType);
    EXPECT_EQ(std::any_cast<const BoundedSoftmaxReduceAttributes&>(reduce->operation.attributes)
                  .whole_rows,
              2 * 4096);
  }
  EXPECT_EQ(pv_nodes, 2);
}

TEST(BoundedAttentionTest, LeavesASmallTailWhole) {
  Attention a;
  Build(a, /*is_global=*/true, /*rows=*/512, /*keys=*/32768);  // 64 MB of scores.
  ASSERT_TRUE(FuseBoundedAttentionSoftmax(&a.graph).ok());
  EXPECT_EQ(a.graph.nodes().size(), 4u);
}

// Two layers reading one mask share its slices.
TEST(BoundedAttentionTest, SlicesASharedMaskOnce) {
  Attention a;
  Build(a, /*is_global=*/true, /*rows=*/4096, /*keys=*/32768);
  // The second tail in the same graph, on the first one's mask and params.
  struct Shared { Value* mask; Value* params; } shared{a.mask, a.params};
  GraphFloat32& g = a.graph;
  {
    // Build the second tail into the first graph.
    Value* q = NewValue(g, DataType::kFloat32, BHWC(1, 2, 4096, 256));
    Value* k = NewValue(g, DataType::kFloat32, BHWC(1, 1, 1, 32768 * 256));
    Value* v = NewValue(g, DataType::kFloat32, BHWC(1, 1, 1, 256 * 32768));
    Value* logits = NewValue(g, DataType::kFloat32, BHWC(1, 2, 4096, 32768));
    RuntimeBatchedMatMulAttributes qk_attr;
    qk_attr.runtime_check.dst_end_ch_index = 2;
    qk_attr.external_weights = ExternalWeightsAttributes{};
    qk_attr.is_global = true;
    NewNode(g, kRuntimeBatchedMatMulType, {q, k, shared.params}, logits)->operation.attributes =
        qk_attr;
    Value* fill = NewValue(g, DataType::kFloat32, BHWC(1, 1, 1, 1));
    Node* constant = g.NewNode();
    constant->operation.type = ToString(OperationType::kConstant);
    g.SetProducer(constant->id, fill->id);
    Value* selected = NewValue(g, DataType::kFloat32, BHWC(1, 2, 4096, 32768));
    NewNode(g, ToString(OperationType::kSelectV2), {shared.mask, logits, fill}, selected);
    Value* probabilities = NewValue(g, DataType::kFloat32, BHWC(1, 2, 4096, 32768));
    NewNode(g, ToString(OperationType::kSoftmax), {selected}, probabilities)->operation.attributes =
        ::ml_drift::SoftmaxAttributes{::ml_drift::Axis::kChannels};
    Value* out = NewValue(g, DataType::kFloat32, BHWC(1, 2, 4096, 256));
    RuntimeBatchedMatMulAttributes pv_attr;
    pv_attr.runtime_check.src_end_ch_index = 2;
    pv_attr.external_weights = ExternalWeightsAttributes{};
    NewNode(g, kRuntimeBatchedMatMulType, {probabilities, v, shared.params}, out)
        ->operation.attributes = pv_attr;
  }
  const auto rewritten = FuseBoundedAttentionSoftmax(&g);
  ASSERT_TRUE(rewritten.ok()) << rewritten.status();
  EXPECT_EQ(*rewritten, 2);
  int mask_slices = 0;
  for (const Node* node : g.nodes()) {
    if (node->operation.type != ToString(OperationType::kSlice)) continue;
    const auto inputs = g.FindInputs(node->id);
    if (inputs.size() == 1 && inputs[0]->id == shared.mask->id) ++mask_slices;
  }
  EXPECT_EQ(mask_slices, 2);
  // Two tails of 2 slices x (Q slice, QK, mask, reduction, PV), 2 shared mask slices, 2 concats.
  EXPECT_EQ(g.nodes().size(), 2u * 2 * 5 + 2 + 2);
}

TEST(BoundedAttentionTest, LeavesOtherSoftmaxesAlone) {
  GraphFloat32 graph;
  Value* in = NewValue(graph, DataType::kFloat32, BHWC(1, 1, 1, 16));
  Value* out = NewValue(graph, DataType::kFloat32, BHWC(1, 1, 1, 16));
  Node* softmax = NewNode(graph, ToString(OperationType::kSoftmax), {in}, out);
  const auto rewritten = FuseBoundedAttentionSoftmax(&graph);
  ASSERT_TRUE(rewritten.ok());
  EXPECT_EQ(*rewritten, 0);
  EXPECT_EQ(softmax->operation.type, ToString(OperationType::kSoftmax));
}

TEST(BoundedAttentionTest, SkipsATailWhoseProbabilitiesHaveAnotherReader) {
  Attention a;
  Build(a);
  Value* extra = NewValue(a.graph, DataType::kFloat32, BHWC(1, 2, 4, 64));
  NewNode(a.graph, ToString(OperationType::kCopy), {a.probabilities}, extra);
  const auto rewritten = FuseBoundedAttentionSoftmax(&a.graph);
  ASSERT_TRUE(rewritten.ok());
  EXPECT_EQ(*rewritten, 0);
  EXPECT_EQ(a.softmax->operation.type, ToString(OperationType::kSoftmax));
}

TEST(BoundedAttentionTest, FindsTheSlidingWindow) {
  GraphFloat32 graph;
  AddWindowMask(graph, 64, 16);
  EXPECT_EQ(FindSlidingWindow(graph), 16);
  AddWindowMask(graph, 64, 16);  // Another graph mask with the same window.
  EXPECT_EQ(FindSlidingWindow(graph), 16);
}

TEST(BoundedAttentionTest, NoWindowWhenMasksDisagree) {
  GraphFloat32 graph;
  AddWindowMask(graph, 64, 16);
  AddWindowMask(graph, 64, 32);
  EXPECT_EQ(FindSlidingWindow(graph), std::nullopt);
  GraphFloat32 uneven;
  AddWindowMask(uneven, 64, 16, /*offset_of_last=*/1);
  EXPECT_EQ(FindSlidingWindow(uneven), std::nullopt);
}

TEST(BoundedAttentionTest, LocalLayersGetTheWindow) {
  Attention a;
  Build(a, /*is_global=*/false);
  AddWindowMask(a.graph, 64, 16);
  ASSERT_EQ(*FuseBoundedAttentionSoftmax(&a.graph), 1);
  const auto& pv = std::any_cast<const RuntimeBatchedMatMulAttributes&>(a.pv->operation.attributes);
  EXPECT_EQ(pv.runtime_check.window_pos_index, 0);
  EXPECT_EQ(pv.runtime_check.window_size, 16);
  EXPECT_TRUE(pv.runtime_check.src_window);
  EXPECT_FALSE(pv.runtime_check.dst_window);
  const auto& reduce =
      std::any_cast<const BoundedSoftmaxReduceAttributes&>(a.softmax->operation.attributes);
  EXPECT_EQ(reduce.window_pos_index, 0);
  EXPECT_EQ(reduce.window_size, 16);
  bool qk_has_window = false;
  for (Node* node : a.graph.nodes()) {
    if (node->operation.type != kRuntimeBatchedMatMulType || node == a.pv) continue;
    const auto& qk = std::any_cast<const RuntimeBatchedMatMulAttributes&>(node->operation.attributes);
    qk_has_window = qk.runtime_check.dst_window && qk.runtime_check.window_size == 16 &&
                    !qk.runtime_check.src_window;
  }
  EXPECT_TRUE(qk_has_window);
}

TEST(BoundedAttentionTest, GlobalLayersReadEveryFilledKey) {
  Attention a;
  Build(a, /*is_global=*/true);
  AddWindowMask(a.graph, 64, 16);
  ASSERT_EQ(*FuseBoundedAttentionSoftmax(&a.graph), 1);
  const auto& pv = std::any_cast<const RuntimeBatchedMatMulAttributes&>(a.pv->operation.attributes);
  EXPECT_FALSE(pv.runtime_check.window_pos_index.has_value());
  const auto& reduce =
      std::any_cast<const BoundedSoftmaxReduceAttributes&>(a.softmax->operation.attributes);
  EXPECT_FALSE(reduce.window_pos_index.has_value());
}

TEST(BoundedAttentionTest, FusesPrefillTailIntoFlashAttentionWhenEnabled) {
  auto build_int8_prefill = [](Attention& a, int rows, int head_dim, int keys, bool is_global) {
    GraphFloat32& g = a.graph;
    Value* q = NewValue(g, DataType::kFloat32, BHWC(1, 2, rows, head_dim));
    Value* k = NewValue(g, DataType::kUint8, BHWC(1, 2, keys, head_dim));
    a.v = NewValue(g, DataType::kUint8, BHWC(1, 2, head_dim, keys));
    a.params = NewValue(g, DataType::kInt32, BHWC(1, 1, 1, 7));
    a.mask = NewValue(g, DataType::kBool, BHWC(1, 1, rows, keys));
    a.logits = NewValue(g, DataType::kFloat32, BHWC(1, 2, rows, keys));

    RuntimeBatchedMatMulAttributes qk_attr;
    qk_attr.runtime_check.dst_end_ch_index = 2;
    qk_attr.is_global = is_global;
    ExternalWeightsAttributes k_ext;
    k_ext.weights_shape = ::ml_drift::OHWI(keys, 2, 1, head_dim);
    k_ext.desc.type = DataType::kUint8;
    k_ext.desc.layout = ::ml_drift::WeightsLayout::kOSpatialIOGroupO4I4;
    k_ext.desc.output_group_size = keys / 4;
    qk_attr.external_weights = k_ext;
    ::ml_drift::Tensor<::ml_drift::Linear, DataType::kFloat32> scale_k;
    scale_k.shape = ::ml_drift::Linear(keys);
    scale_k.data = std::vector<float>(keys, 0.05f);
    qk_attr.scale = scale_k;
    NewNode(g, kRuntimeBatchedMatMulType, {q, k, a.params}, a.logits)->operation.attributes =
        qk_attr;

    Value* fill = NewValue(g, DataType::kFloat32, BHWC(1, 1, 1, 1));
    Node* constant = g.NewNode();
    constant->operation.type = ToString(OperationType::kConstant);
    g.SetProducer(constant->id, fill->id);
    a.selected = NewValue(g, DataType::kFloat32, BHWC(1, 2, rows, keys));
    a.select = NewNode(g, ToString(OperationType::kSelectV2), {a.mask, a.logits, fill}, a.selected);
    a.probabilities = NewValue(g, DataType::kFloat32, BHWC(1, 2, rows, keys));
    a.softmax = NewNode(g, ToString(OperationType::kSoftmax), {a.selected}, a.probabilities);
    a.softmax->operation.attributes = ::ml_drift::SoftmaxAttributes{::ml_drift::Axis::kChannels};

    a.out = NewValue(g, DataType::kFloat32, BHWC(1, 2, rows, head_dim));
    RuntimeBatchedMatMulAttributes pv_attr;
    pv_attr.runtime_check.src_end_ch_index = 2;
    ExternalWeightsAttributes v_ext;
    v_ext.weights_shape = ::ml_drift::OHWI(head_dim, 2, 1, keys);
    v_ext.desc.type = DataType::kUint8;
    v_ext.desc.layout = ::ml_drift::WeightsLayout::kOSpatialIOGroupI4O4;
    v_ext.desc.output_group_size = head_dim / 4;
    pv_attr.external_weights = v_ext;
    ::ml_drift::Tensor<::ml_drift::Linear, DataType::kFloat32> scale_v;
    scale_v.shape = ::ml_drift::Linear(head_dim);
    scale_v.data = std::vector<float>(head_dim, 0.02f);
    pv_attr.scale = scale_v;
    a.pv = NewNode(g, kRuntimeBatchedMatMulType, {a.probabilities, a.v, a.params}, a.out);
    a.pv->operation.attributes = pv_attr;
  };

  // Prefill D=256 local tail collapses into one kFlashAttentionType node (no SliceRows).
  Attention local_prefill;
  build_int8_prefill(local_prefill, /*rows=*/4096, /*head_dim=*/256, /*keys=*/32768,
                     /*is_global=*/false);
  const auto rewritten = FuseBoundedAttentionSoftmax(&local_prefill.graph,
                                                     /*enable_flash_attention=*/true);
  ASSERT_TRUE(rewritten.ok()) << rewritten.status();
  EXPECT_EQ(*rewritten, 1);
  EXPECT_EQ(local_prefill.graph.nodes().size(), 1u);
  const Node* flash = local_prefill.graph.FindProducer(local_prefill.out->id);
  ASSERT_NE(flash, nullptr);
  EXPECT_EQ(flash->operation.type, kFlashAttentionType);
  EXPECT_EQ(local_prefill.graph.FindInputs(flash->id).size(), 5u);
  const auto& fattr = std::any_cast<const FlashAttentionAttributes&>(flash->operation.attributes);
  EXPECT_EQ(fattr.head_dim, 256);
  EXPECT_EQ(fattr.kv_heads, 2);
  EXPECT_EQ(fattr.cache_size, 32768);
  EXPECT_FLOAT_EQ(fattr.scale_k, 0.05f);
  EXPECT_FLOAT_EQ(fattr.scale_v, 0.02f);
  EXPECT_FALSE(fattr.row_end);

  // Decode tail (rows=4 < 256) stays on bounded QK -> mask -> reduce -> PV.
  Attention decode;
  build_int8_prefill(decode, /*rows=*/4, /*head_dim=*/256, /*keys=*/32768, /*is_global=*/false);
  ASSERT_EQ(*FuseBoundedAttentionSoftmax(&decode.graph, /*enable_flash_attention=*/true), 1);
  EXPECT_EQ(decode.graph.nodes().size(), 4u);
}

}  // namespace
}  // namespace litert::ml_drift
