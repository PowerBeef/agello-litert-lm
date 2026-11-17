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

#include "ml_drift_delegate/delegate/composite/bounded_attention_kernel.h"

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift_delegate/delegate/composite/bounded_attention.h"

namespace litert::ml_drift {

absl::Status CreateBoundedMaskFromNode(const std::vector<::ml_drift::Value*>& inputs,
                                       const std::vector<::ml_drift::Value*>& outputs,
                                       const ::ml_drift::Node& node,
                                       ::ml_drift::GpuModelBuilder* model_builder) {
  const auto* attr = std::any_cast<BoundedMaskAttributes>(&node.operation.attributes);
  if (attr == nullptr || inputs.size() != 2 || outputs.size() != 1) {
    return absl::InvalidArgumentError("Bounded mask expects [logits, mask] and one output.");
  }
  ABSL_ASSIGN_OR_RETURN(auto logits, model_builder->GetTensor(inputs[0]->id));
  ABSL_ASSIGN_OR_RETURN(auto mask, model_builder->GetTensor(inputs[1]->id));
  const ::ml_drift::DataType type = logits.tensor_desc.GetDataType();

  // Elementwise with the logits first, so MergeNodes fuses it into the QK
  // product's epilogue: it then runs only where that product writes. The mask
  // broadcasts like any elementwise second input (GQA-folded rows read their
  // query token's row), as in the SDPA fallback's SelectBoolMask.
  ::ml_drift::ElementwiseDescriptor op_desc;
  op_desc.args.AddFloat("mask_value", attr->mask_value, type);
  if (attr->clamp_low && attr->clamp_high) {
    op_desc.args.AddFloat("clamp_low", *attr->clamp_low, type);
    op_desc.args.AddFloat("clamp_high", *attr->clamp_high, type);
  }
  int component = 0;
  for (const char* c : {"x", "y", "z", "w"}) {
    const std::string value = std::string("in_value.") + c;
    const std::string kept_value =
        attr->clamp_low && attr->clamp_high
            ? "clamp(" + value + ", args.clamp_low, args.clamp_high)"
            : value;
    // Agello (engine review, step 4): with row_end, key S_COORD * 4 + component against the
    // row's bound (an int32 read).
    const std::string keep =
        attr->row_end ? "(S_COORD * 4 + " + std::to_string(component) + ") < in2_value.x"
                      : std::string("in2_value.") + c;
    op_desc.code += std::string("  out_value.") + c + " = " + keep + " ? " + kept_value +
                    " : args.mask_value;\n";
    ++component;
  }
  auto dst = model_builder->AddTensor(logits.tensor_desc);
  ::ml_drift::OperationDef definition;
  definition.src_tensors.push_back(logits.tensor_desc);
  definition.src_tensors.push_back(mask.tensor_desc);
  definition.dst_tensors.push_back(dst.tensor_desc);
  auto op = std::make_unique<::ml_drift::GPUOperation>(::ml_drift::CreateGpuOperation(
      definition, std::move(op_desc), mask.tensor_desc.GetBHWCShape(),
      dst.tensor_desc.GetBHWCShape()));
  model_builder->AddGpuOperation({logits, mask}, {dst}, std::move(op), kBoundedMaskType);
  return model_builder->UpdateOutputTensor(dst, outputs[0]->id);
}

namespace {
// Agello (engine review, step 4): one thread per row of the mask.
class MaskBoundsOp : public ::ml_drift::GPUOperation {
 public:
  MaskBoundsOp() = default;
  ::ml_drift::int3 GetGridSize() const override {
    return ::ml_drift::int3(dst_[0]->Width() * dst_[0]->Batch(), dst_[0]->Height(), 1);
  }
  MaskBoundsOp(MaskBoundsOp&&) = default;
  MaskBoundsOp& operator=(MaskBoundsOp&&) = default;
  MaskBoundsOp(const MaskBoundsOp&) = delete;
  MaskBoundsOp& operator=(const MaskBoundsOp&) = delete;
};
}  // namespace

absl::Status CreateMaskBoundsFromNode(const std::vector<::ml_drift::Value*>& inputs,
                                      const std::vector<::ml_drift::Value*>& outputs,
                                      const ::ml_drift::Node& node,
                                      ::ml_drift::GpuModelBuilder* model_builder) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    return absl::InvalidArgumentError("Mask bounds expect one mask and one output.");
  }
  ABSL_ASSIGN_OR_RETURN(auto mask, model_builder->GetTensor(inputs[0]->id));
  ::ml_drift::TensorDescriptor dst_desc = mask.tensor_desc;
  dst_desc.SetDataType(::ml_drift::DataType::kInt32);
  ::ml_drift::BHWC dst_shape = mask.tensor_desc.GetBHWCShape();
  dst_shape.c = 1;
  dst_desc.SetBHWCShape(dst_shape);
  auto dst = model_builder->AddTensor(dst_desc);

  MaskBoundsOp op;
  op.AddSrcTensor("src_tensor", mask.tensor_desc);
  op.AddDstTensor("dst_tensor", dst.tensor_desc);
  // One thread per row (W, with the batch and the height folded in): one past the last kept
  // key, so a row that isn't a prefix still bounds every key it keeps.
  op.code_ = R"(
MAIN_FUNCTION($0) {
  int X = ucl::GetGlobalId<0>();
  int Y = ucl::GetGlobalId<1>();
  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height()) {
    return;
  }
  int end = 0;
  for (int s = 0; s < args.src_tensor.Slices(); ++s) {
    int4 kept = ucl::Convert<int4>(args.src_tensor.Read(X, Y, s));
    if (kept.x != 0) end = s * 4 + 1;
    if (kept.y != 0) end = s * 4 + 2;
    if (kept.z != 0) end = s * 4 + 3;
    if (kept.w != 0) end = s * 4 + 4;
  }
  args.dst_tensor.Write(ucl::Convert<args.dst_tensor::type>(int4(end, 0, 0, 0)), X, Y, 0);
}
)";
  model_builder->AddGpuOperation({mask}, {dst},
                                 std::make_unique<MaskBoundsOp>(std::move(op)),
                                 kMaskBoundsType);
  return model_builder->UpdateOutputTensor(dst, outputs[0]->id);
}

namespace {
// Agello (engine review, step 4): one thread per row.
class BoundsFlagOp : public ::ml_drift::GPUOperation {
 public:
  explicit BoundsFlagOp(int rows) : rows_(rows) {}
  ::ml_drift::int3 GetGridSize() const override { return ::ml_drift::int3(rows_, 1, 1); }
  BoundsFlagOp(BoundsFlagOp&&) = default;
  BoundsFlagOp& operator=(BoundsFlagOp&&) = default;
  BoundsFlagOp(const BoundsFlagOp&) = delete;
  BoundsFlagOp& operator=(const BoundsFlagOp&) = delete;

 private:
  int rows_;
};
}  // namespace

absl::Status CreateBoundsFlagFromNode(const std::vector<::ml_drift::Value*>& inputs,
                                      const std::vector<::ml_drift::Value*>& outputs,
                                      const ::ml_drift::Node& node,
                                      ::ml_drift::GpuModelBuilder* model_builder) {
  if (inputs.size() != 1 || outputs.size() != 1) {
    return absl::InvalidArgumentError("A bounds flag expects one bounds tensor and one output.");
  }
  ABSL_ASSIGN_OR_RETURN(auto bounds, model_builder->GetTensor(inputs[0]->id));
  const ::ml_drift::BHWC flag_shape = outputs[0]->tensor.shape;
  const int rows = bounds.tensor_desc.GetBHWCShape().w;
  const bool rows_in_height = flag_shape.h == rows && flag_shape.w == 1;
  if (flag_shape.c != 1 || flag_shape.b != 1 || !(rows_in_height || flag_shape.w == rows)) {
    return absl::InvalidArgumentError("A bounds flag needs one boolean per row.");
  }
  ::ml_drift::TensorDescriptor dst_desc = bounds.tensor_desc;
  dst_desc.SetDataType(::ml_drift::DataType::kBool);
  dst_desc.SetBHWCShape(flag_shape);
  auto dst = model_builder->AddTensor(dst_desc);
  BoundsFlagOp op(rows);
  op.AddSrcTensor("src_tensor", bounds.tensor_desc);
  op.AddDstTensor("dst_tensor", dst.tensor_desc);
  op.args_.AddInt("rows", rows);
  const std::string coords = rows_in_height ? "0, R, 0" : "R, 0, 0";
  op.code_ = R"(
MAIN_FUNCTION($0) {
  int R = ucl::GetGlobalId<0>();
  if (R >= args.rows) {
    return;
  }
  int4 bound = ucl::Convert<int4>(args.src_tensor.Read(R, 0, 0));
  int4 kept = int4(bound.x > 0 ? 1 : 0, 0, 0, 0);
  args.dst_tensor.Write(ucl::Convert<args.dst_tensor::type>(kept), )" + coords + R"();
}
)";
  model_builder->AddGpuOperation({bounds}, {dst},
                                 std::make_unique<BoundsFlagOp>(std::move(op)), kBoundsFlagType);
  return model_builder->UpdateOutputTensor(dst, outputs[0]->id);
}

absl::Status CreateBoundedSoftmaxReduceFromNode(const std::vector<::ml_drift::Value*>& inputs,
                                                const std::vector<::ml_drift::Value*>& outputs,
                                                const ::ml_drift::Node& node,
                                                ::ml_drift::GpuModelBuilder* model_builder) {
  const auto* attr = std::any_cast<BoundedSoftmaxReduceAttributes>(&node.operation.attributes);
  if (attr == nullptr || inputs.size() != 2 || outputs.size() != 1) {
    return absl::InvalidArgumentError(
        "Bounded softmax reduction expects [logits, params] and one output.");
  }
  ABSL_ASSIGN_OR_RETURN(auto logits, model_builder->GetTensor(inputs[0]->id));
  ABSL_ASSIGN_OR_RETURN(auto params, model_builder->GetTensor(inputs[1]->id));
  ::ml_drift::SoftmaxRuntimeCheckDesc runtime_check;
  runtime_check.end_ch_index = attr->end_ch_index;
  runtime_check.window_pos_index = attr->window_pos_index;
  runtime_check.window_size = attr->window_size;
  runtime_check.decision_rows = attr->whole_rows;
  auto reduced = model_builder->SoftmaxReduce(logits, runtime_check, &params);
  return model_builder->UpdateOutputTensor(reduced, outputs[0]->id);
}

}  // namespace litert::ml_drift
