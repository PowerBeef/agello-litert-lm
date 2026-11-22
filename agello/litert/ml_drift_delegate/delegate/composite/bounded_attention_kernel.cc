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
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/kernel_info.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/buffer_desc.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/tuning_type.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "ml_drift/common/util.h"  // from @ml_drift
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
// Parallel row-wise mask bound scan: on Apple Metal, 256 threads per row scan
// Slices() in parallel and reduce with simd_max (+ 8-element shared max);
// other backends fall back to one thread per row.
class MaskBoundsOp : public ::ml_drift::GPUOperation {
 public:
  explicit MaskBoundsOp(bool parallel_row_scan)
      : parallel_row_scan_(parallel_row_scan) {
    if (parallel_row_scan_) {
      work_group_size_ = ::ml_drift::int3(256, 1, 1);
    }
  }
  std::vector<::ml_drift::int3> GetPossibleKernelWorkGroups(
      ::ml_drift::TuningType tuning_type, const ::ml_drift::GpuInfo& gpu_info,
      const ::ml_drift::KernelInfo& kernel_info) const override {
    if (parallel_row_scan_) {
      return {work_group_size_};
    }
    return ::ml_drift::GPUOperation::GetPossibleKernelWorkGroups(
        tuning_type, gpu_info, kernel_info);
  }
  ::ml_drift::int3 GetGridSize() const override {
    if (parallel_row_scan_) {
      return ::ml_drift::int3(256, dst_[0]->Width() * dst_[0]->Batch(),
                              dst_[0]->Height());
    }
    return ::ml_drift::int3(dst_[0]->Width() * dst_[0]->Batch(),
                            dst_[0]->Height(), 1);
  }
  MaskBoundsOp(MaskBoundsOp&&) = default;
  MaskBoundsOp& operator=(MaskBoundsOp&&) = default;
  MaskBoundsOp(const MaskBoundsOp&) = delete;
  MaskBoundsOp& operator=(const MaskBoundsOp&) = delete;

 private:
  bool parallel_row_scan_ = false;
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

  const bool parallel_row_scan =
      model_builder->gpu_info().IsApple() && model_builder->gpu_info().IsApiMetal();
  MaskBoundsOp op(parallel_row_scan);
  op.AddSrcTensor("src_tensor", mask.tensor_desc);
  op.AddDstTensor("dst_tensor", dst.tensor_desc);
  if (parallel_row_scan) {
    // 256 threads per row scan Slices() coalesced and reduce with simd_max.
    op.code_ = R"(
MAIN_FUNCTION($0) {
  int tid = ucl::GetLocalId<0>();
  int X = ucl::GetGroupId<1>();
  int Y = ucl::GetGroupId<2>();
  if (X >= args.dst_tensor.Width() || Y >= args.dst_tensor.Height()) {
    return;
  }
  int end = 0;
  int slices = args.src_tensor.Slices();
  for (int s = tid; s < slices; s += 256) {
    int4 kept = ucl::Convert<int4>(args.src_tensor.Read(X, Y, s));
    if (kept.x != 0) end = s * 4 + 1;
    if (kept.y != 0) end = s * 4 + 2;
    if (kept.z != 0) end = s * 4 + 3;
    if (kept.w != 0) end = s * 4 + 4;
  }
  end = simd_max(end);
  threadgroup int smem[8];
  if ((tid & 31) == 0) {
    smem[tid >> 5] = end;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    int best = max(max(max(smem[0], smem[1]), max(smem[2], smem[3])),
                   max(max(smem[4], smem[5]), max(smem[6], smem[7])));
    args.dst_tensor.Write(ucl::Convert<args.dst_tensor::type>(int4(best, 0, 0, 0)), X, Y, 0);
  }
}
)";
  } else {
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
  }
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

namespace {

// Agello (P2.3): MSL 4.1 cooperative-tensor FlashAttention over the INT8 KV cache.
class FlashAttentionOp : public ::ml_drift::GPUOperation {
 public:
  FlashAttentionOp(int wg_size, int bq) : bq_(bq) {
    work_group_size_ = ::ml_drift::int3(wg_size, 1, 1);
  }
  std::vector<::ml_drift::int3> GetPossibleKernelWorkGroups(
      ::ml_drift::TuningType, const ::ml_drift::GpuInfo&,
      const ::ml_drift::KernelInfo&) const override {
    return {work_group_size_};
  }
  ::ml_drift::int3 GetGridSize() const override {
    return ::ml_drift::int3(
        work_group_size_.x,
        ::ml_drift::DivideRoundUp(dst_[0]->Width() * dst_[0]->Batch(), bq_),
        dst_[0]->Height());
  }
  FlashAttentionOp(FlashAttentionOp&&) = default;
  FlashAttentionOp& operator=(FlashAttentionOp&&) = default;
  FlashAttentionOp(const FlashAttentionOp&) = delete;
  FlashAttentionOp& operator=(const FlashAttentionOp&) = delete;

 private:
  int bq_ = 32;
};

std::string GenerateFlashAttentionD256Code(const FlashAttentionAttributes& attr,
                                           ::ml_drift::DataType q_type) {
  // Optimal Apple10 tile (benchmarked on M6): 4 SGs (128 threads), M_SG=16
  // (BQ=64), BK=64, PV_N=64 (4 tiles), FP32 O accumulator (~25.5 KiB TG).
  constexpr int kSimdGroups = 4;
  constexpr int kWgSize = kSimdGroups * 32;
  constexpr int kMsg = 16;
  constexpr int kBq = kSimdGroups * kMsg;  // 64
  constexpr int kBk = 64;
  const bool is_f16 = q_type == ::ml_drift::DataType::kFloat16;
  const std::string scalar = is_f16 ? "half" : "float";
  const std::string vec4 = is_f16 ? "half4" : "float4";
  // kv_loc_u32 is 16 KiB: holds 32 x 256 half (2 passes) or 16 x 256 float (4 passes).
  const int pass_rows = is_f16 ? 32 : 16;
  const int pass_sgs = pass_rows / kMsg;
  const int num_passes = kBq / pass_rows;
  const bool has_clamp = attr.clamp_low.has_value() && attr.clamp_high.has_value();

  std::string c;
  c += "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";
  c += "MAIN_FUNCTION($0) {\n";
  absl::StrAppend(&c, "  constexpr int WG_SIZE = ", kWgSize, ";\n");
  absl::StrAppend(&c, "  constexpr int BQ = ", kBq, ";\n");
  absl::StrAppend(&c, "  constexpr int BK = ", kBk, ";\n");
  c += "  int tid = ucl::GetLocalId<0>();\n";
  c += "  int sgid = tid >> 5;\n";
  c += "  int lane = tid & 31;\n";
  c += "  int sg_r0 = sgid * 16;\n";
  c += "  int num_q_tiles = (args.q_rows + BQ - 1) / BQ;\n";
  if (attr.row_end) {
    c += "  int q_tile_id = (num_q_tiles - 1) - ucl::GetGroupId<1>();\n";
  } else {
    c += "  int q_tile_id = ucl::GetGroupId<1>();\n";
  }
  c += "  int kv_head = ucl::GetGroupId<2>();\n";
  c += "  if (q_tile_id < 0 || q_tile_id >= num_q_tiles || kv_head >= args.kv_heads) return;\n";
  c += "  int q_base = q_tile_id * BQ;\n";
  c += "  int active_keys = clamp(args.params.Read(args.end_ch_index), 0, args.cache_size);\n";
  if (attr.window_pos_index.has_value()) {
    absl::StrAppend(&c,
                    "  int k_start = (max(0, args.params.Read(args.window_pos_index) - ",
                    attr.window_size - 1, ") / 64) * 64;\n");
  } else {
    c += "  int k_start = 0;\n";
  }
  c += "  int k_end = active_keys;\n";
  c += "  int mask_y = kv_head % args.mask_h;\n";

  // Threadgroup buffers (~25.5 KiB total).
  c += "  alignas(16) threadgroup uint kv_loc_u32[64 * 64];\n";
  c += "  threadgroup int8_t* kv_loc = reinterpret_cast<threadgroup int8_t*>(kv_loc_u32);\n";
  c += "  alignas(16) threadgroup half p_tg[BQ * BK];\n";
  c += "  threadgroup float tg_alpha[BQ];\n";
  c += "  threadgroup float tg_m[BQ];\n";

  if (attr.row_end) {
    c += "  threadgroup int tg_bounds[BQ];\n";
    c += "  threadgroup int tg_max_b[2];\n";
    c += "  if (tid < BQ) {\n";
    c += "    int qr = q_base + tid;\n";
    c += "    int b = 0;\n";
    c += "    if (qr < args.q_rows) {\n";
    c += "      b = clamp(ucl::Convert<int4>(args.mask.Read(qr % args.mask_w, mask_y, 0)).x, 0, active_keys);\n";
    c += "    }\n";
    c += "    tg_bounds[tid] = b;\n";
    c += "    int sm = simd_max(b);\n";
    c += "    if (lane == 0) tg_max_b[sgid] = sm;\n";
    c += "  }\n";
    c += "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    c += "  k_end = min(active_keys, max(tg_max_b[0], tg_max_b[1]));\n";
  } else {
    // 64 rows x 64 keys = 64 x uint2 (512 B per buffer, double-buffered = 1 KiB).
    // Each SG owns 4 consecutive slices (16 keys = 1 ushort) and reads both 32-row halves.
    c += "  alignas(8) threadgroup ushort tg_mask_u16[2][BQ * 4];\n";
    c += "  threadgroup uint tg_tile_any[2][4];\n";
    c += "  int mask_qr0 = q_base + lane;\n";
    c += "  int mask_qr1 = q_base + 32 + lane;\n";
    c += "  bool mask_valid0 = mask_qr0 < args.q_rows;\n";
    c += "  bool mask_valid1 = mask_qr1 < args.q_rows;\n";
    c += "  int mask_x0 = mask_valid0 ? (mask_qr0 % args.mask_w) : 0;\n";
    c += "  int mask_x1 = mask_valid1 ? (mask_qr1 % args.mask_w) : 0;\n";
  }

  // Q tensor and K/V/P views.
  absl::StrAppend(&c, "  device ", scalar, "* q_ptr = reinterpret_cast<device ", scalar,
                  "*>(args.q.GetHandle()) + kv_head * args.q_rows * 256;\n");
  c += "  auto q_tensor = metal::tensor(q_ptr, metal::dextents<int, 2>(256, args.q_rows));\n";
  c += "  auto q_sg_tile = q_tensor.slice(0, q_base + sg_r0);\n";
  c += "  auto k_tile = metal::tensor(kv_loc, metal::dextents<int, 2>(256, 64));\n";
  c += "  auto v_tensor = metal::tensor(kv_loc, metal::dextents<int, 2>(256, 64));\n";
  c += "  auto p_tensor = metal::tensor(p_tg, metal::dextents<int, 2>(64, BQ));\n";
  c += "  auto p_sg_tile = p_tensor.slice(0, sg_r0);\n";

  // TensorOps QK (16x64x256, NT) and PV (16x64x64, NN) ops.
  c += R"(  constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 64, 256, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
  mpp::tensor_ops::matmul2d<qk_desc, metal::execution_simdgroup> mul_qk_op;
  auto ctS = mul_qk_op.get_destination_cooperative_tensor<decltype(q_sg_tile), decltype(k_tile), float>();
  auto ctM = mul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(q_sg_tile), decltype(k_tile), float>();
  auto ctL = mul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(q_sg_tile), decltype(k_tile), float>();
  auto ctTileMax = mul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(q_sg_tile), decltype(k_tile), float>();
  auto ctTileSum = mul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(q_sg_tile), decltype(k_tile), float>();
  #pragma clang loop unroll(full)
  for (uint16_t i = 0; i < ctM.get_capacity(); ++i) {
    if (ctM.is_valid_element(i)) {
      ctM[i] = -1e38f;
      ctL[i] = 0.0f;
    }
  }
  constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 64, 64, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<pv_desc, metal::execution_simdgroup> mul_pv_op;
  auto v_slice0 = v_tensor.slice(0, 0);
  auto v_slice1 = v_tensor.slice(64, 0);
  auto v_slice2 = v_tensor.slice(128, 0);
  auto v_slice3 = v_tensor.slice(192, 0);
)";
  for (int t = 0; t < 4; ++t) {
    absl::StrAppend(&c, "  auto ctO", t,
                    " = mul_pv_op.get_destination_cooperative_tensor<decltype(p_sg_tile), "
                    "decltype(v_slice0), float>();\n");
    absl::StrAppend(&c, "  #pragma clang loop unroll(full)\n",
                    "  for (uint16_t i = 0; i < ctO", t, ".get_capacity(); ++i) {\n",
                    "    if (ctO", t, ".is_valid_element(i)) ctO", t, "[i] = 0.0f;\n",
                    "  }\n");
  }

  // Coalesced K (16 consecutive k_s per d_s) and V (64 consecutive d_s per k_s) coords.
  c += "  int k_lk4 = tid & 15;\n";
  c += "  int k_ds_base = tid >> 4;\n";
  c += "  int v_ds = tid & 63;\n";
  c += "  int v_lk4_base = tid >> 6;\n";

  // Main K-tile loop (step 64 keys).
  c += "  int iter = 0;\n";
  c += "  for (int k_base = k_start; k_base < k_end; k_base += 64, ++iter) {\n";
  c += "    int k_s_base = k_base >> 2;\n";
  if (!attr.row_end) {
    c += "    int buf = iter & 1;\n";
    c += "    int s_base = k_s_base + sgid * 4;\n";
    c += "    uint bits0 = 0u;\n";
    c += "    uint bits1 = 0u;\n";
    c += "    if (mask_valid0) {\n";
    c += "      #pragma clang loop unroll(full)\n";
    c += "      for (int q = 0; q < 4; ++q) {\n";
    c += "        int ms = s_base + q;\n";
    c += "        if (ms < args.mask_slices) {\n";
    c += "          int4 mv = ucl::Convert<int4>(args.mask.Read(mask_x0, mask_y, ms));\n";
    c += "          int k0 = ms * 4;\n";
    c += "          uint sh = uint(q * 4);\n";
    c += "          if (mv.x != 0 && k0 + 0 >= k_start && k0 + 0 < active_keys) bits0 |= (1u << sh);\n";
    c += "          if (mv.y != 0 && k0 + 1 >= k_start && k0 + 1 < active_keys) bits0 |= (2u << sh);\n";
    c += "          if (mv.z != 0 && k0 + 2 >= k_start && k0 + 2 < active_keys) bits0 |= (4u << sh);\n";
    c += "          if (mv.w != 0 && k0 + 3 >= k_start && k0 + 3 < active_keys) bits0 |= (8u << sh);\n";
    c += "        }\n";
    c += "      }\n";
    c += "    }\n";
    c += "    if (mask_valid1) {\n";
    c += "      #pragma clang loop unroll(full)\n";
    c += "      for (int q = 0; q < 4; ++q) {\n";
    c += "        int ms = s_base + q;\n";
    c += "        if (ms < args.mask_slices) {\n";
    c += "          int4 mv = ucl::Convert<int4>(args.mask.Read(mask_x1, mask_y, ms));\n";
    c += "          int k0 = ms * 4;\n";
    c += "          uint sh = uint(q * 4);\n";
    c += "          if (mv.x != 0 && k0 + 0 >= k_start && k0 + 0 < active_keys) bits1 |= (1u << sh);\n";
    c += "          if (mv.y != 0 && k0 + 1 >= k_start && k0 + 1 < active_keys) bits1 |= (2u << sh);\n";
    c += "          if (mv.z != 0 && k0 + 2 >= k_start && k0 + 2 < active_keys) bits1 |= (4u << sh);\n";
    c += "          if (mv.w != 0 && k0 + 3 >= k_start && k0 + 3 < active_keys) bits1 |= (8u << sh);\n";
    c += "        }\n";
    c += "      }\n";
    c += "    }\n";
    c += "    tg_mask_u16[buf][lane * 4 + sgid] = ushort(bits0);\n";
    c += "    tg_mask_u16[buf][(32 + lane) * 4 + sgid] = ushort(bits1);\n";
    c += "    uint sg_any = simd_any((bits0 | bits1) != 0u) ? 1u : 0u;\n";
    c += "    if (lane == 0) tg_tile_any[buf][sgid] = sg_any;\n";
    c += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    c += "    uint tile_active = tg_tile_any[buf][0] | tg_tile_any[buf][1] | tg_tile_any[buf][2] | tg_tile_any[buf][3];\n";
    c += "    if (tile_active == 0u) continue;\n";
    c += "    threadgroup uint2* tg_mask = reinterpret_cast<threadgroup uint2*>(tg_mask_u16[buf]);\n";
    c += "    uint sg_mask_or = 0u;\n";
    c += "    #pragma clang loop unroll(full)\n";
    c += "    for (int r = 0; r < 16; ++r) {\n";
    c += "      uint2 m2 = tg_mask[sg_r0 + r];\n";
    c += "      sg_mask_or |= (m2.x | m2.y);\n";
    c += "    }\n";
    c += "    bool sg_active = (sg_mask_or != 0u);\n";
  } else {
    c += "    if (iter > 0) threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    c += "    int sg_max_b = 0;\n";
    c += "    #pragma clang loop unroll(full)\n";
    c += "    for (int r = 0; r < 16; ++r) sg_max_b = max(sg_max_b, tg_bounds[sg_r0 + r]);\n";
    c += "    bool sg_active = (k_base < sg_max_b);\n";
  }

  // Load K tile into kv_loc_u32 (64 x 256 int8 = 16 KiB, 8 coalesced uint4 loads/thread).
  c += "    {\n";
  c += "      int k_s = k_s_base + k_lk4;\n";
  c += "      int row0 = k_lk4 * 4;\n";
  c += "      #pragma clang loop unroll(full)\n";
  c += "      for (int d_s = k_ds_base; d_s < 64; d_s += 8) {\n";
  c += "        uint4 kv = (k_s < args.k_slices) ? (args.cache_k.Read((kv_head * 64 + d_s) * args.k_slices + k_s) ^ uint4(0x80808080u)) : uint4(0u);\n";
  c += "        kv_loc_u32[(row0 + 0) * 64 + d_s] = kv.x;\n";
  c += "        kv_loc_u32[(row0 + 1) * 64 + d_s] = kv.y;\n";
  c += "        kv_loc_u32[(row0 + 2) * 64 + d_s] = kv.z;\n";
  c += "        kv_loc_u32[(row0 + 3) * 64 + d_s] = kv.w;\n";
  c += "      }\n";
  c += "    }\n";
  c += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";

  // QK matmul per active SIMD group (16 x 64 x 256).
  c += "    if (sg_active) {\n";
  c += "      mul_qk_op.run(q_sg_tile, k_tile, ctS);\n";
  c += "    }\n";
  c += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";

  // Load V tile into kv_loc_u32 (64 x 256 int8 = 16 KiB, 8 coalesced uint4 loads/thread).
  c += "    {\n";
  c += "      #pragma clang loop unroll(full)\n";
  c += "      for (int lk4 = v_lk4_base; lk4 < 16; lk4 += 2) {\n";
  c += "        int k_s = k_s_base + lk4;\n";
  c += "        uint4 vv = (k_s < args.k_slices) ? (args.cache_v.Read((kv_head * args.k_slices + k_s) * 64 + v_ds) ^ uint4(0x80808080u)) : uint4(0u);\n";
  c += "        int row0 = lk4 * 4;\n";
  c += "        kv_loc_u32[(row0 + 0) * 64 + v_ds] = vv.x;\n";
  c += "        kv_loc_u32[(row0 + 1) * 64 + v_ds] = vv.y;\n";
  c += "        kv_loc_u32[(row0 + 2) * 64 + v_ds] = vv.z;\n";
  c += "        kv_loc_u32[(row0 + 3) * 64 + v_ds] = vv.w;\n";
  c += "      }\n";
  c += "    }\n";

  // Online softmax on active SGs before the V barrier.
  c += "    bool need_rescale = false;\n";
  c += "    if (sg_active) {\n";
  c += "      #pragma clang loop unroll(full)\n";
  c += "      for (uint16_t i = 0; i < ctS.get_capacity(); ++i) {\n";
  c += "        if (ctS.is_valid_element(i)) {\n";
  c += "          auto coord = ctS.get_multidimensional_index(i);\n";
  c += "          int lq = sg_r0 + coord[1];\n";
  c += "          int lk = coord[0];\n";
  if (attr.row_end) {
    c += "          bool kept = (k_base + lk) < tg_bounds[lq];\n";
  } else {
    c += "          uint2 m2 = tg_mask[lq];\n";
    c += "          uint mw = (lk < 32) ? m2.x : m2.y;\n";
    c += "          bool kept = ((mw >> (lk & 31)) & 1u) != 0u;\n";
  }
  c += "          if (kept) {\n";
  c += "            float s = ctS[i] * args.scale_k;\n";
  if (has_clamp) {
    c += "            s = clamp(s, args.clamp_low, args.clamp_high);\n";
  }
  c += "            ctS[i] = s;\n";
  c += "          } else {\n";
  c += "            ctS[i] = -1e38f;\n";
  c += "          }\n";
  c += "        }\n";
  c += "      }\n";
  c += "      mpp::tensor_ops::reduce_rows(ctS, ctTileMax, mpp::tensor_ops::reduction_operation::max, -1e38f);\n";
  c += "      bool local_changed = false;\n";
  c += "      #pragma clang loop unroll(full)\n";
  c += "      for (uint16_t i = 0; i < ctM.get_capacity(); ++i) {\n";
  c += "        if (ctM.is_valid_element(i)) {\n";
  c += "          float m_old = ctM[i];\n";
  c += "          float m_new = max(m_old, ctTileMax[i]);\n";
  c += "          bool ch = (m_old > -1e30f) && (m_new > m_old);\n";
  c += "          local_changed = local_changed || ch;\n";
  c += "          float alpha = ch ? exp(m_old - m_new) : 1.0f;\n";
  c += "          ctM[i] = m_new;\n";
  c += "          ctL[i] *= alpha;\n";
  c += "          auto rc = ctM.get_multidimensional_index(i);\n";
  c += "          tg_alpha[sg_r0 + rc[0]] = alpha;\n";
  c += "          tg_m[sg_r0 + rc[0]] = m_new;\n";
  c += "        }\n";
  c += "      }\n";
  c += "      need_rescale = simd_any(local_changed);\n";
  c += "      simdgroup_barrier(mem_flags::mem_threadgroup);\n";
  c += "      #pragma clang loop unroll(full)\n";
  c += "      for (uint16_t i = 0; i < ctS.get_capacity(); ++i) {\n";
  c += "        if (ctS.is_valid_element(i)) {\n";
  c += "          auto coord = ctS.get_multidimensional_index(i);\n";
  c += "          int lq = sg_r0 + coord[1];\n";
  c += "          float s = ctS[i];\n";
  c += "          float m = tg_m[lq];\n";
  c += "          float p = (s > -1e30f && m > -1e30f) ? exp(s - m) : 0.0f;\n";
  c += "          ctS[i] = p;\n";
  c += "          p_tg[lq * BK + coord[0]] = half(p);\n";
  c += "        }\n";
  c += "      }\n";
  c += "      mpp::tensor_ops::reduce_rows(ctS, ctTileSum, mpp::tensor_ops::reduction_operation::sum, 0.0f);\n";
  c += "      #pragma clang loop unroll(full)\n";
  c += "      for (uint16_t i = 0; i < ctL.get_capacity(); ++i) {\n";
  c += "        if (ctL.is_valid_element(i)) ctL[i] += ctTileSum[i];\n";
  c += "      }\n";
  c += "    }\n";

  // Synchronize V in kv_loc_u32, p_tg, and tg_alpha in one barrier.
  c += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";

  // Rescale ctO and run PV matmul on active SGs (4 tiles of 16 x 64 x 64).
  c += "    if (sg_active) {\n";
  c += "      if (need_rescale) {\n";
  c += "        #pragma clang loop unroll(full)\n";
  c += "        for (uint16_t i = 0; i < ctO0.get_capacity(); ++i) {\n";
  c += "          if (ctO0.is_valid_element(i)) {\n";
  c += "            float a = tg_alpha[sg_r0 + ctO0.get_multidimensional_index(i)[1]];\n";
  c += "            ctO0[i] *= a;\n";
  c += "            ctO1[i] *= a;\n";
  c += "            ctO2[i] *= a;\n";
  c += "            ctO3[i] *= a;\n";
  c += "          }\n";
  c += "        }\n";
  c += "      }\n";
  c += "      mul_pv_op.run(p_sg_tile, v_slice0, ctO0);\n";
  c += "      mul_pv_op.run(p_sg_tile, v_slice1, ctO1);\n";
  c += "      mul_pv_op.run(p_sg_tile, v_slice2, ctO2);\n";
  c += "      mul_pv_op.run(p_sg_tile, v_slice3, ctO3);\n";
  c += "    }\n";
  c += "  }\n";  // end K-loop

  // Final normalization factor (scale_v / l) stored in tg_alpha.
  c += "  #pragma clang loop unroll(full)\n";
  c += "  for (uint16_t i = 0; i < ctL.get_capacity(); ++i) {\n";
  c += "    if (ctL.is_valid_element(i)) {\n";
  c += "      auto rc = ctL.get_multidimensional_index(i);\n";
  c += "      float l = ctL[i];\n";
  c += "      tg_alpha[sg_r0 + rc[0]] = (l > 0.0f) ? (args.scale_v / l) : 0.0f;\n";
  c += "    }\n";
  c += "  }\n";
  c += "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";

  // Multi-pass coalesced epilogue reusing kv_loc_u32 (16 KiB).
  absl::StrAppend(&c, "  threadgroup ", scalar, "* o_stage = reinterpret_cast<threadgroup ",
                  scalar, "*>(kv_loc_u32);\n");
  absl::StrAppend(&c, "  threadgroup ", vec4, "* o_stage4 = reinterpret_cast<threadgroup ",
                  vec4, "*>(kv_loc_u32);\n");
  absl::StrAppend(&c, "  constexpr int PASS_ROWS = ", pass_rows, ";\n");
  absl::StrAppend(&c, "  constexpr int PASS_SGS = ", pass_sgs, ";\n");
  absl::StrAppend(&c, "  for (int pass = 0; pass < ", num_passes, "; ++pass) {\n");
  c += "    if (pass > 0) threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  c += "    if (sgid >= pass * PASS_SGS && sgid < (pass + 1) * PASS_SGS) {\n";
  c += "      int local_sg = sgid - pass * PASS_SGS;\n";
  for (int t = 0; t < 4; ++t) {
    absl::StrAppend(&c, "      #pragma clang loop unroll(full)\n",
                    "      for (uint16_t i = 0; i < ctO", t, ".get_capacity(); ++i) {\n",
                    "        if (ctO", t, ".is_valid_element(i)) {\n",
                    "          auto coord = ctO", t, ".get_multidimensional_index(i);\n",
                    "          int lr = local_sg * 16 + coord[1];\n",
                    "          float inv_l = tg_alpha[sg_r0 + coord[1]];\n",
                    "          o_stage[lr * 256 + ", t * 64, " + coord[0]] = ", scalar,
                    "(ctO", t, "[i] * inv_l);\n",
                    "        }\n",
                    "      }\n");
  }
  c += "    }\n";
  c += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  c += "    int ep_ds = tid & 63;\n";
  c += "    int ep_lr_base = tid >> 6;\n";
  c += "    #pragma clang loop unroll(full)\n";
  c += "    for (int lr = ep_lr_base; lr < PASS_ROWS; lr += 2) {\n";
  c += "      int qr = q_base + pass * PASS_ROWS + lr;\n";
  c += "      if (qr < args.q_rows) {\n";
  c += "        args.dst.Write(o_stage4[lr * 64 + ep_ds], qr, kv_head, ep_ds);\n";
  c += "      }\n";
  c += "    }\n";
  c += "  }\n";
  c += "}\n";
  return c;
}

std::string GenerateFlashAttentionD512Code(const FlashAttentionAttributes& attr,
                                           ::ml_drift::DataType q_type) {
  // Optimal Apple10 D=512 tile (benchmarked on M6): 4 SGs (128 threads),
  // M_SG=16 (BQ=64), BK=64, no-split-K (two 256-dim halves into 16 KiB kv_loc),
  // 16x-scaled running-normalized FP16 O accumulators (~25 KiB TG, zero spills,
  // all 4 SGs active in softmax, half the cache_k TLB misses of BK=32).
  constexpr int kBq = 64;
  constexpr int kBk = 64;
  const bool is_f16 = q_type == ::ml_drift::DataType::kFloat16;
  const std::string scalar = is_f16 ? "half" : "float";
  const std::string vec4 = is_f16 ? "half4" : "float4";
  // kv_loc_u32 is 16 KiB: holds 16 x 512 half (4 passes) or 8 x 512 float (8 passes).
  const int pass_rows = is_f16 ? 16 : 8;
  const int num_passes = kBq / pass_rows;
  const bool has_clamp = attr.clamp_low.has_value() && attr.clamp_high.has_value();

  std::string c;
  c += "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";
  c += "MAIN_FUNCTION($0) {\n";
  absl::StrAppend(&c, "  constexpr int BQ = ", kBq, ";\n");
  absl::StrAppend(&c, "  constexpr int BK = ", kBk, ";\n");
  c += "  int tid = ucl::GetLocalId<0>();\n";
  c += "  int sgid = tid >> 5;\n";
  c += "  int lane = tid & 31;\n";
  c += "  int sg_r0 = sgid * 16;\n";
  c += "  int num_q_tiles = (args.q_rows + BQ - 1) / BQ;\n";
  if (attr.row_end) {
    // Launch heaviest causal query tiles first for tail-wave load balance.
    c += "  int q_tile_id = (num_q_tiles - 1) - ucl::GetGroupId<1>();\n";
  } else {
    c += "  int q_tile_id = ucl::GetGroupId<1>();\n";
  }
  c += "  int kv_head = ucl::GetGroupId<2>();\n";
  c += "  if (q_tile_id < 0 || q_tile_id >= num_q_tiles || kv_head >= args.kv_heads) return;\n";
  c += "  int q_base = q_tile_id * BQ;\n";
  c += "  int active_keys = clamp(args.params.Read(args.end_ch_index), 0, args.cache_size);\n";
  if (attr.window_pos_index.has_value()) {
    absl::StrAppend(&c,
                    "  int k_start = (max(0, args.params.Read(args.window_pos_index) - ",
                    attr.window_size - 1, ") / 64) * 64;\n");
  } else {
    c += "  int k_start = 0;\n";
  }
  c += "  int k_end = active_keys;\n";
  c += "  int mask_y = kv_head % args.mask_h;\n";

  // Threadgroup buffers (~25.5 KiB total).
  c += "  alignas(16) threadgroup uint kv_loc_u32[64 * 64];\n";
  c += "  threadgroup int8_t* kv_loc = reinterpret_cast<threadgroup int8_t*>(kv_loc_u32);\n";
  c += "  alignas(16) threadgroup half p_tg[BQ * BK];\n";
  c += "  threadgroup float tg_o_scale[BQ];\n";
  c += "  threadgroup float tg_m[BQ];\n";

  if (attr.row_end) {
    c += "  threadgroup int tg_bounds[BQ];\n";
    c += "  threadgroup int tg_max_b[2];\n";
    c += "  if (tid < BQ) {\n";
    c += "    int qr = q_base + tid;\n";
    c += "    int b = 0;\n";
    c += "    if (qr < args.q_rows) {\n";
    c += "      b = clamp(ucl::Convert<int4>(args.mask.Read(qr % args.mask_w, mask_y, 0)).x, 0, active_keys);\n";
    c += "    }\n";
    c += "    tg_bounds[tid] = b;\n";
    c += "    int sm = simd_max(b);\n";
    c += "    if (lane == 0) tg_max_b[sgid] = sm;\n";
    c += "  }\n";
    c += "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    c += "  k_end = min(active_keys, max(tg_max_b[0], tg_max_b[1]));\n";
  } else {
    c += "  alignas(8) threadgroup ushort tg_mask_u16[2][BQ * 4];\n";
    c += "  threadgroup uint tg_tile_any[2][4];\n";
    c += "  int mask_qr0 = q_base + lane;\n";
    c += "  int mask_qr1 = q_base + 32 + lane;\n";
    c += "  bool mask_valid0 = mask_qr0 < args.q_rows;\n";
    c += "  bool mask_valid1 = mask_qr1 < args.q_rows;\n";
    c += "  int mask_x0 = mask_valid0 ? (mask_qr0 % args.mask_w) : 0;\n";
    c += "  int mask_x1 = mask_valid1 ? (mask_qr1 % args.mask_w) : 0;\n";
  }

  // Q lo/hi views (256 cols each), 256-col K/V staging view, P view.
  absl::StrAppend(&c, "  device ", scalar, "* q_ptr = reinterpret_cast<device ", scalar,
                  "*>(args.q.GetHandle()) + kv_head * args.q_rows * 512;\n");
  c += "  auto q_tensor = metal::tensor(q_ptr, metal::dextents<int, 2>(512, args.q_rows));\n";
  c += "  auto q_lo = q_tensor.slice(0, q_base + sg_r0);\n";
  c += "  auto q_hi = q_tensor.slice(256, q_base + sg_r0);\n";
  c += "  auto kv_tensor = metal::tensor(kv_loc, metal::dextents<int, 2>(256, 64));\n";
  c += "  auto p_tensor = metal::tensor(p_tg, metal::dextents<int, 2>(64, BQ));\n";
  c += "  auto p_sg_tile = p_tensor.slice(0, sg_r0);\n";

  // Two-half QK (16x64x256, NT) and 8-tile FP16 running-normalized PV (16x64x64, NN).
  c += R"(  constexpr auto qk0_desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 64, 256, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
  constexpr auto qk1_desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 64, 256, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<qk0_desc, metal::execution_simdgroup> qk0_op;
  mpp::tensor_ops::matmul2d<qk1_desc, metal::execution_simdgroup> qk1_op;
  auto ctS = qk0_op.get_destination_cooperative_tensor<decltype(q_lo), decltype(kv_tensor), float>();
  auto ctM = qk0_op.get_row_reduction_destination_cooperative_tensor<decltype(q_lo), decltype(kv_tensor), float>();
  auto ctL = qk0_op.get_row_reduction_destination_cooperative_tensor<decltype(q_lo), decltype(kv_tensor), float>();
  auto ctTileMax = qk0_op.get_row_reduction_destination_cooperative_tensor<decltype(q_lo), decltype(kv_tensor), float>();
  auto ctTileSum = qk0_op.get_row_reduction_destination_cooperative_tensor<decltype(q_lo), decltype(kv_tensor), float>();
  #pragma clang loop unroll(full)
  for (uint16_t i = 0; i < ctM.get_capacity(); ++i) {
    if (ctM.is_valid_element(i)) {
      ctM[i] = -1e38f;
      ctL[i] = 0.0f;
    }
  }
  constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 64, 64, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<pv_desc, metal::execution_simdgroup> pv_op;
  auto v0 = kv_tensor.slice(0, 0);
  auto v1 = kv_tensor.slice(64, 0);
  auto v2 = kv_tensor.slice(128, 0);
  auto v3 = kv_tensor.slice(192, 0);
)";
  for (int t = 0; t < 8; ++t) {
    absl::StrAppend(&c, "  auto ctO", t,
                    " = pv_op.get_destination_cooperative_tensor<decltype(p_sg_tile), "
                    "decltype(v0), half>();\n");
  }
  c += "  #pragma clang loop unroll(full)\n";
  c += "  for (uint16_t i = 0; i < ctO0.get_capacity(); ++i) {\n";
  c += "    if (ctO0.is_valid_element(i)) {\n";
  c += "      ctO0[i] = 0; ctO1[i] = 0; ctO2[i] = 0; ctO3[i] = 0;\n";
  c += "      ctO4[i] = 0; ctO5[i] = 0; ctO6[i] = 0; ctO7[i] = 0;\n";
  c += "    }\n";
  c += "  }\n";

  // Coalesced K (16 consecutive k_s per d_s) and V (64 consecutive d_s per k_s) coords.
  c += "  int k_lk4 = tid & 15;\n";
  c += "  int k_ds_base = tid >> 4;\n";
  c += "  int v_ds = tid & 63;\n";
  c += "  int v_lk4_base = tid >> 6;\n";

  // Main K-tile loop (step 64 keys).
  c += "  int iter = 0;\n";
  c += "  for (int k_base = k_start; k_base < k_end; k_base += 64, ++iter) {\n";
  c += "    int k_s_base = k_base >> 2;\n";
  if (!attr.row_end) {
    c += "    int buf = iter & 1;\n";
    c += "    int s_base = k_s_base + sgid * 4;\n";
    c += "    uint bits0 = 0u;\n";
    c += "    uint bits1 = 0u;\n";
    c += "    if (mask_valid0) {\n";
    c += "      #pragma clang loop unroll(full)\n";
    c += "      for (int q = 0; q < 4; ++q) {\n";
    c += "        int ms = s_base + q;\n";
    c += "        if (ms < args.mask_slices) {\n";
    c += "          int4 mv = ucl::Convert<int4>(args.mask.Read(mask_x0, mask_y, ms));\n";
    c += "          int k0 = ms * 4;\n";
    c += "          uint sh = uint(q * 4);\n";
    c += "          if (mv.x != 0 && k0 + 0 >= k_start && k0 + 0 < active_keys) bits0 |= (1u << sh);\n";
    c += "          if (mv.y != 0 && k0 + 1 >= k_start && k0 + 1 < active_keys) bits0 |= (2u << sh);\n";
    c += "          if (mv.z != 0 && k0 + 2 >= k_start && k0 + 2 < active_keys) bits0 |= (4u << sh);\n";
    c += "          if (mv.w != 0 && k0 + 3 >= k_start && k0 + 3 < active_keys) bits0 |= (8u << sh);\n";
    c += "        }\n";
    c += "      }\n";
    c += "    }\n";
    c += "    if (mask_valid1) {\n";
    c += "      #pragma clang loop unroll(full)\n";
    c += "      for (int q = 0; q < 4; ++q) {\n";
    c += "        int ms = s_base + q;\n";
    c += "        if (ms < args.mask_slices) {\n";
    c += "          int4 mv = ucl::Convert<int4>(args.mask.Read(mask_x1, mask_y, ms));\n";
    c += "          int k0 = ms * 4;\n";
    c += "          uint sh = uint(q * 4);\n";
    c += "          if (mv.x != 0 && k0 + 0 >= k_start && k0 + 0 < active_keys) bits1 |= (1u << sh);\n";
    c += "          if (mv.y != 0 && k0 + 1 >= k_start && k0 + 1 < active_keys) bits1 |= (2u << sh);\n";
    c += "          if (mv.z != 0 && k0 + 2 >= k_start && k0 + 2 < active_keys) bits1 |= (4u << sh);\n";
    c += "          if (mv.w != 0 && k0 + 3 >= k_start && k0 + 3 < active_keys) bits1 |= (8u << sh);\n";
    c += "        }\n";
    c += "      }\n";
    c += "    }\n";
    c += "    tg_mask_u16[buf][lane * 4 + sgid] = ushort(bits0);\n";
    c += "    tg_mask_u16[buf][(32 + lane) * 4 + sgid] = ushort(bits1);\n";
    c += "    uint sg_any = simd_any((bits0 | bits1) != 0u) ? 1u : 0u;\n";
    c += "    if (lane == 0) tg_tile_any[buf][sgid] = sg_any;\n";
    c += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    c += "    uint tile_active = tg_tile_any[buf][0] | tg_tile_any[buf][1] | tg_tile_any[buf][2] | tg_tile_any[buf][3];\n";
    c += "    if (tile_active == 0u) continue;\n";
    c += "    threadgroup uint2* tg_mask = reinterpret_cast<threadgroup uint2*>(tg_mask_u16[buf]);\n";
    c += "    uint sg_mask_or = 0u;\n";
    c += "    #pragma clang loop unroll(full)\n";
    c += "    for (int r = 0; r < 16; ++r) {\n";
    c += "      uint2 m2 = tg_mask[sg_r0 + r];\n";
    c += "      sg_mask_or |= (m2.x | m2.y);\n";
    c += "    }\n";
    c += "    bool sg_active = (sg_mask_or != 0u);\n";
  } else {
    c += "    if (iter > 0) threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    c += "    int sg_max_b = 0;\n";
    c += "    #pragma clang loop unroll(full)\n";
    c += "    for (int r = 0; r < 16; ++r) sg_max_b = max(sg_max_b, tg_bounds[sg_r0 + r]);\n";
    c += "    bool sg_active = (k_base < sg_max_b);\n";
  }

  // Load K lo (d_s 0..63) -> qk0_op.
  c += "    {\n";
  c += "      int k_s = k_s_base + k_lk4;\n";
  c += "      int row0 = k_lk4 * 4;\n";
  c += "      #pragma clang loop unroll(full)\n";
  c += "      for (int d_s = k_ds_base; d_s < 64; d_s += 8) {\n";
  c += "        uint4 kv = (k_s < args.k_slices) ? (args.cache_k.Read((kv_head * 128 + d_s) * args.k_slices + k_s) ^ uint4(0x80808080u)) : uint4(0u);\n";
  c += "        kv_loc_u32[(row0 + 0) * 64 + d_s] = kv.x;\n";
  c += "        kv_loc_u32[(row0 + 1) * 64 + d_s] = kv.y;\n";
  c += "        kv_loc_u32[(row0 + 2) * 64 + d_s] = kv.z;\n";
  c += "        kv_loc_u32[(row0 + 3) * 64 + d_s] = kv.w;\n";
  c += "      }\n";
  c += "    }\n";
  c += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  c += "    if (sg_active) qk0_op.run(q_lo, kv_tensor, ctS);\n";
  c += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";

  // Load K hi (d_s 64..127) -> qk1_op (accumulate into ctS).
  c += "    {\n";
  c += "      int k_s = k_s_base + k_lk4;\n";
  c += "      int row0 = k_lk4 * 4;\n";
  c += "      #pragma clang loop unroll(full)\n";
  c += "      for (int d_s = k_ds_base; d_s < 64; d_s += 8) {\n";
  c += "        uint4 kv = (k_s < args.k_slices) ? (args.cache_k.Read((kv_head * 128 + 64 + d_s) * args.k_slices + k_s) ^ uint4(0x80808080u)) : uint4(0u);\n";
  c += "        kv_loc_u32[(row0 + 0) * 64 + d_s] = kv.x;\n";
  c += "        kv_loc_u32[(row0 + 1) * 64 + d_s] = kv.y;\n";
  c += "        kv_loc_u32[(row0 + 2) * 64 + d_s] = kv.z;\n";
  c += "        kv_loc_u32[(row0 + 3) * 64 + d_s] = kv.w;\n";
  c += "      }\n";
  c += "    }\n";
  c += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  c += "    if (sg_active) qk1_op.run(q_hi, kv_tensor, ctS);\n";
  c += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";

  // Load V lo (d_s 0..63) into kv_loc_u32 while preparing softmax.
  c += "    {\n";
  c += "      #pragma clang loop unroll(full)\n";
  c += "      for (int lk4 = v_lk4_base; lk4 < 16; lk4 += 2) {\n";
  c += "        int k_s = k_s_base + lk4;\n";
  c += "        uint4 vv = (k_s < args.k_slices) ? (args.cache_v.Read((kv_head * args.k_slices + k_s) * 128 + v_ds) ^ uint4(0x80808080u)) : uint4(0u);\n";
  c += "        int row0 = lk4 * 4;\n";
  c += "        kv_loc_u32[(row0 + 0) * 64 + v_ds] = vv.x;\n";
  c += "        kv_loc_u32[(row0 + 1) * 64 + v_ds] = vv.y;\n";
  c += "        kv_loc_u32[(row0 + 2) * 64 + v_ds] = vv.z;\n";
  c += "        kv_loc_u32[(row0 + 3) * 64 + v_ds] = vv.w;\n";
  c += "      }\n";
  c += "    }\n";

  // 16x-scaled running-normalized online softmax across ALL 4 SGs.
  c += "    bool need_rescale = false;\n";
  c += "    if (sg_active) {\n";
  c += "      #pragma clang loop unroll(full)\n";
  c += "      for (uint16_t i = 0; i < ctS.get_capacity(); ++i) {\n";
  c += "        if (ctS.is_valid_element(i)) {\n";
  c += "          auto coord = ctS.get_multidimensional_index(i);\n";
  c += "          int lq = sg_r0 + coord[1];\n";
  c += "          int lk = coord[0];\n";
  if (attr.row_end) {
    c += "          bool kept = (k_base + lk) < tg_bounds[lq];\n";
  } else {
    c += "          uint2 m2 = tg_mask[lq];\n";
    c += "          uint mw = (lk < 32) ? m2.x : m2.y;\n";
    c += "          bool kept = ((mw >> (lk & 31)) & 1u) != 0u;\n";
  }
  c += "          if (kept) {\n";
  c += "            float s = ctS[i] * args.scale_k;\n";
  if (has_clamp) {
    c += "            s = clamp(s, args.clamp_low, args.clamp_high);\n";
  }
  c += "            ctS[i] = s;\n";
  c += "          } else {\n";
  c += "            ctS[i] = -1e38f;\n";
  c += "          }\n";
  c += "        }\n";
  c += "      }\n";
  c += "      mpp::tensor_ops::reduce_rows(ctS, ctTileMax, mpp::tensor_ops::reduction_operation::max, -1e38f);\n";
  c += "      #pragma clang loop unroll(full)\n";
  c += "      for (uint16_t i = 0; i < ctM.get_capacity(); ++i) {\n";
  c += "        if (ctM.is_valid_element(i)) {\n";
  c += "          float m_old = ctM[i];\n";
  c += "          float m_new = max(m_old, ctTileMax[i]);\n";
  c += "          float alpha = (m_old > -1e30f && m_new > m_old) ? exp(m_old - m_new) : 1.0f;\n";
  c += "          ctM[i] = m_new;\n";
  c += "          ctL[i] *= alpha;\n";
  c += "          auto rc = ctM.get_multidimensional_index(i);\n";
  c += "          tg_m[sg_r0 + rc[0]] = m_new;\n";
  c += "        }\n";
  c += "      }\n";
  c += "      simdgroup_barrier(mem_flags::mem_threadgroup);\n";
  c += "      #pragma clang loop unroll(full)\n";
  c += "      for (uint16_t i = 0; i < ctS.get_capacity(); ++i) {\n";
  c += "        if (ctS.is_valid_element(i)) {\n";
  c += "          auto coord = ctS.get_multidimensional_index(i);\n";
  c += "          float s = ctS[i];\n";
  c += "          float m = tg_m[sg_r0 + coord[1]];\n";
  c += "          ctS[i] = (s > -1e30f && m > -1e30f) ? exp(s - m) : 0.0f;\n";
  c += "        }\n";
  c += "      }\n";
  c += "      mpp::tensor_ops::reduce_rows(ctS, ctTileSum, mpp::tensor_ops::reduction_operation::sum, 0.0f);\n";
  c += "      bool has_prev = false;\n";
  c += "      #pragma clang loop unroll(full)\n";
  c += "      for (uint16_t i = 0; i < ctL.get_capacity(); ++i) {\n";
  c += "        if (ctL.is_valid_element(i)) {\n";
  c += "          auto rc = ctL.get_multidimensional_index(i);\n";
  c += "          float l_old_s = ctL[i];\n";
  c += "          float l_new = l_old_s + ctTileSum[i];\n";
  c += "          ctL[i] = l_new;\n";
  c += "          float inv_l = (l_new > 0.0f) ? (1.0f / l_new) : 0.0f;\n";
  c += "          float os = l_old_s * inv_l;\n";
  c += "          has_prev = has_prev || (l_old_s > 0.0f && os < 1.0f);\n";
  c += "          tg_o_scale[sg_r0 + rc[0]] = os;\n";
  c += "          tg_m[sg_r0 + rc[0]] = 16.0f * inv_l;\n";
  c += "        }\n";
  c += "      }\n";
  c += "      need_rescale = simd_any(has_prev);\n";
  c += "      simdgroup_barrier(mem_flags::mem_threadgroup);\n";
  c += "      #pragma clang loop unroll(full)\n";
  c += "      for (uint16_t i = 0; i < ctS.get_capacity(); ++i) {\n";
  c += "        if (ctS.is_valid_element(i)) {\n";
  c += "          auto coord = ctS.get_multidimensional_index(i);\n";
  c += "          int lq = sg_r0 + coord[1];\n";
  c += "          p_tg[lq * BK + coord[0]] = half(ctS[i] * tg_m[lq]);\n";
  c += "        }\n";
  c += "      }\n";
  c += "    }\n";

  // Synchronize V lo and p_tg.
  c += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  c += "    if (sg_active) {\n";
  c += "      if (need_rescale) {\n";
  c += "        #pragma clang loop unroll(full)\n";
  c += "        for (uint16_t i = 0; i < ctO0.get_capacity(); ++i) {\n";
  c += "          if (ctO0.is_valid_element(i)) {\n";
  c += "            half a = half(tg_o_scale[sg_r0 + ctO0.get_multidimensional_index(i)[1]]);\n";
  c += "            ctO0[i] *= a; ctO1[i] *= a; ctO2[i] *= a; ctO3[i] *= a;\n";
  c += "            ctO4[i] *= a; ctO5[i] *= a; ctO6[i] *= a; ctO7[i] *= a;\n";
  c += "          }\n";
  c += "        }\n";
  c += "      }\n";
  c += "      pv_op.run(p_sg_tile, v0, ctO0);\n";
  c += "      pv_op.run(p_sg_tile, v1, ctO1);\n";
  c += "      pv_op.run(p_sg_tile, v2, ctO2);\n";
  c += "      pv_op.run(p_sg_tile, v3, ctO3);\n";
  c += "    }\n";
  c += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";

  // Load V hi (d_s 64..127) -> pv_op for ctO4..7.
  c += "    {\n";
  c += "      #pragma clang loop unroll(full)\n";
  c += "      for (int lk4 = v_lk4_base; lk4 < 16; lk4 += 2) {\n";
  c += "        int k_s = k_s_base + lk4;\n";
  c += "        uint4 vv = (k_s < args.k_slices) ? (args.cache_v.Read((kv_head * args.k_slices + k_s) * 128 + 64 + v_ds) ^ uint4(0x80808080u)) : uint4(0u);\n";
  c += "        int row0 = lk4 * 4;\n";
  c += "        kv_loc_u32[(row0 + 0) * 64 + v_ds] = vv.x;\n";
  c += "        kv_loc_u32[(row0 + 1) * 64 + v_ds] = vv.y;\n";
  c += "        kv_loc_u32[(row0 + 2) * 64 + v_ds] = vv.z;\n";
  c += "        kv_loc_u32[(row0 + 3) * 64 + v_ds] = vv.w;\n";
  c += "      }\n";
  c += "    }\n";
  c += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  c += "    if (sg_active) {\n";
  c += "      pv_op.run(p_sg_tile, v0, ctO4);\n";
  c += "      pv_op.run(p_sg_tile, v1, ctO5);\n";
  c += "      pv_op.run(p_sg_tile, v2, ctO6);\n";
  c += "      pv_op.run(p_sg_tile, v3, ctO7);\n";
  c += "    }\n";
  c += "  }\n";  // end K-loop

  // Multi-pass coalesced epilogue reusing kv_loc_u32 (16 KiB).
  // ctO already holds running-normalized 16x-scaled sums; multiply by scale_v / 16.
  c += "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  c += "  float final_scale = args.scale_v * 0.0625f;\n";
  absl::StrAppend(&c, "  threadgroup ", scalar, "* o_stage = reinterpret_cast<threadgroup ",
                  scalar, "*>(kv_loc_u32);\n");
  absl::StrAppend(&c, "  threadgroup ", vec4, "* o_stage4 = reinterpret_cast<threadgroup ",
                  vec4, "*>(kv_loc_u32);\n");
  absl::StrAppend(&c, "  constexpr int PASS_ROWS = ", pass_rows, ";\n");
  absl::StrAppend(&c, "  for (int pass = 0; pass < ", num_passes, "; ++pass) {\n");
  c += "    if (pass > 0) threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  c += "    int pass_r0 = pass * PASS_ROWS;\n";
  c += "    int pass_r1 = pass_r0 + PASS_ROWS;\n";
  c += "    if (sg_r0 < pass_r1 && (sg_r0 + 16) > pass_r0) {\n";
  c += "      #pragma clang loop unroll(full)\n";
  c += "      for (uint16_t i = 0; i < ctO0.get_capacity(); ++i) {\n";
  c += "        if (ctO0.is_valid_element(i)) {\n";
  c += "          auto coord = ctO0.get_multidimensional_index(i);\n";
  c += "          int lr = (sg_r0 + coord[1]) - pass_r0;\n";
  c += "          if (lr >= 0 && lr < PASS_ROWS) {\n";
  c += "            int base = lr * 512 + coord[0];\n";
  for (int t = 0; t < 8; ++t) {
    absl::StrAppend(&c, "            o_stage[base + ", t * 64, "] = ", scalar,
                    "(float(ctO", t, "[i]) * final_scale);\n");
  }
  c += "          }\n";
  c += "        }\n";
  c += "      }\n";
  c += "    }\n";
  c += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  // 128 threads write PASS_ROWS x 128 vec4s coalesced (1 vec4/row per thread).
  c += "    #pragma clang loop unroll(full)\n";
  c += "    for (int lr = 0; lr < PASS_ROWS; ++lr) {\n";
  c += "      int qr = q_base + pass_r0 + lr;\n";
  c += "      if (qr < args.q_rows) {\n";
  c += "        args.dst.Write(o_stage4[lr * 128 + tid], qr, kv_head, tid);\n";
  c += "      }\n";
  c += "    }\n";
  c += "  }\n";
  c += "}\n";
  return c;
}

}  // namespace

absl::Status CreateFlashAttentionFromNode(const std::vector<::ml_drift::Value*>& inputs,
                                          const std::vector<::ml_drift::Value*>& outputs,
                                          const ::ml_drift::Node& node,
                                          ::ml_drift::GpuModelBuilder* model_builder) {
  const auto* attr = std::any_cast<FlashAttentionAttributes>(&node.operation.attributes);
  if (attr == nullptr || inputs.size() != 5 || outputs.size() != 1) {
    return absl::InvalidArgumentError(
        "Flash attention expects [Q, cache_k, cache_v, mask, params] and one output.");
  }
  if (attr->head_dim != 256 && attr->head_dim != 512) {
    return absl::InvalidArgumentError("Flash attention supports head_dim 256 or 512.");
  }

  ABSL_ASSIGN_OR_RETURN(auto q, model_builder->GetTensor(inputs[0]->id));
  ABSL_ASSIGN_OR_RETURN(auto cache_k, model_builder->GetTensor(inputs[1]->id));
  ABSL_ASSIGN_OR_RETURN(auto cache_v, model_builder->GetTensor(inputs[2]->id));
  ABSL_ASSIGN_OR_RETURN(auto mask, model_builder->GetTensor(inputs[3]->id));
  ABSL_ASSIGN_OR_RETURN(auto params, model_builder->GetTensor(inputs[4]->id));

  ::ml_drift::TensorDescriptor q_lin_desc(
      q.tensor_desc.GetDataType(), ::ml_drift::TensorStorageType::kBuffer,
      q.tensor_desc.GetLayout(),
      ::ml_drift::TensorDescriptor::PhysicalLayout1D::kDHWBCC4);
  q_lin_desc.SetBHWCShape(q.tensor_desc.GetBHWCShape());
  auto q_lin = model_builder->AddTensor(q_lin_desc);
  model_builder->Copy(q, q_lin);
  const ::ml_drift::BHWC q_shape = q_lin.tensor_desc.GetBHWCShape();
  const ::ml_drift::DataType q_type = q_lin.tensor_desc.GetDataType();
  auto dst = model_builder->AddTensor(q_shape, q_type);

  constexpr int kSimdGroups = 4;
  constexpr int kBq = 64;
  FlashAttentionOp op(kSimdGroups * 32, kBq);

  op.AddSrcTensor("q", q_lin.tensor_desc);
  ::ml_drift::BufferDescriptor kv_desc;
  kv_desc.element_type = ::ml_drift::DataType::kUint32;
  kv_desc.element_size = 4;
  op.AddSrcBuffer("cache_k", kv_desc);
  op.AddSrcBuffer("cache_v", kv_desc);
  op.AddSrcTensor("mask", mask.tensor_desc);
  ::ml_drift::BufferDescriptor params_desc;
  params_desc.element_type = ::ml_drift::DataType::kInt32;
  params_desc.element_size = 1;
  params_desc.memory_type = ::ml_drift::MemoryType::kConstant;
  op.AddSrcBuffer("params", params_desc);
  op.AddDstTensor("dst", dst.tensor_desc);

  op.args_.AddInt("q_rows", q_shape.w);
  op.args_.AddInt("kv_heads", q_shape.h);
  op.args_.AddInt("cache_size", attr->cache_size);
  op.args_.AddInt("k_slices", attr->cache_size / 4);
  op.args_.AddInt("end_ch_index", attr->end_ch_index);
  op.args_.AddFloat("scale_k", attr->scale_k, ::ml_drift::DataType::kFloat32);
  op.args_.AddFloat("scale_v", attr->scale_v, ::ml_drift::DataType::kFloat32);
  if (attr->window_pos_index.has_value()) {
    op.args_.AddInt("window_pos_index", *attr->window_pos_index);
  }
  if (attr->clamp_low.has_value() && attr->clamp_high.has_value()) {
    op.args_.AddFloat("clamp_low", *attr->clamp_low, ::ml_drift::DataType::kFloat32);
    op.args_.AddFloat("clamp_high", *attr->clamp_high, ::ml_drift::DataType::kFloat32);
  }
  const ::ml_drift::BHWC mask_shape = mask.tensor_desc.GetBHWCShape();
  op.args_.AddInt("mask_w", mask_shape.w);
  op.args_.AddInt("mask_h", mask_shape.h);
  op.args_.AddInt("mask_slices", ::ml_drift::DivideRoundUp(mask_shape.c, 4));

  if (attr->head_dim == 256) {
    op.code_ = GenerateFlashAttentionD256Code(*attr, q_type);
  } else {
    op.code_ = GenerateFlashAttentionD512Code(*attr, q_type);
  }

  model_builder->AddGpuOperation(
      {q_lin, cache_k, cache_v, mask, params}, {dst},
      std::make_unique<FlashAttentionOp>(std::move(op)),
      absl::StrCat(kFlashAttentionType, "_d", attr->head_dim));
  return model_builder->UpdateOutputTensor(dst, outputs[0]->id);
}

}  // namespace litert::ml_drift
