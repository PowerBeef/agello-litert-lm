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

#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "absl/strings/match.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernels/conv_apple_mpp.h"
#include "ml_drift/common/kernels/quantize_and_dequantize.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"

namespace ml_drift {
namespace {

GpuInfo MakeAppleMetal4GpuInfo() {
  GpuInfo gpu_info;
  GetGpuInfoFromDeviceDescription("Apple A19 Pro", GpuApi::kMetal, &gpu_info);
  gpu_info.metal_info.language_version = MetalLanguageVersion::kMetal4_1;
  gpu_info.metal_info.max_work_group_size_x = 1024;
  gpu_info.metal_info.max_work_group_size_y = 1024;
  gpu_info.metal_info.max_work_group_size_z = 1024;
  return gpu_info;
}

TEST(AgelloKernelCodegenTest, QuantizationSeedsClampedCoordsAndFp32Range) {
  const GpuInfo gpu_info = MakeAppleMetal4GpuInfo();
  OperationDef op_def;
  op_def.src_tensors.push_back(
      {DataType::kFloat16, TensorStorageType::kBuffer, Layout::kHWC});
  op_def.dst_tensors.push_back(
      {DataType::kInt8, TensorStorageType::kBuffer, Layout::kHWC});
  op_def.dst_tensors.push_back(
      {DataType::kFloat16, TensorStorageType::kBuffer, Layout::kHWC});

  std::unique_ptr<GPUOperation> op = CreateQuantization(
      op_def, PackedType::kInt8C4, gpu_info, BHWC(1, 1, 4, 32),
      /*calculate_sum=*/false);
  ASSERT_NE(op, nullptr);

  // Check generated UCL code before Metal lowering.
  EXPECT_TRUE(absl::StrContains(
      op->code_, "args.src_tensor.Read(x_clamped, y_clamped, 0).x"));
  EXPECT_FALSE(absl::StrContains(op->code_, "args.src_tensor.Read(0, 0, 0)"));
  EXPECT_TRUE(absl::StrContains(
      op->code_,
      "float src_range = ucl::Convert<float>(max_value) - "
      "ucl::Convert<float>(min_value);"));
  EXPECT_TRUE(absl::StrContains(
      op->code_,
      "float scale = src_range > 0.0f ? ucl::Convert<float>(args.dst_range) / "
      "src_range : 0.0f;"));

  // AssembleCode must succeed cleanly on Apple Metal 4.
  EXPECT_TRUE(op->AssembleCode(gpu_info).ok());
}

TEST(AgelloKernelCodegenTest, ConvAppleMppDynamicKSoftmaxBoundsAndTailMask) {
  const GpuInfo gpu_info = MakeAppleMetal4GpuInfo();
  TensorDescriptor src_desc(DataType::kFloat16, TensorStorageType::kBuffer,
                            Layout::kHWC,
                            TensorDescriptor::PhysicalLayout1D::kDHWBCC4);
  TensorDescriptor dst_desc(DataType::kFloat16, TensorStorageType::kBuffer,
                            Layout::kHWC);
  TensorDescriptor src_exp_desc(DataType::kFloat32, TensorStorageType::kBuffer,
                                Layout::kHWC);

  ConvRuntimeCheckDesc runtime_check;
  runtime_check.src_end_ch_index = 2;

  ConvAppleMPP op = CreateConvAppleMPPExternalWeights(
      src_desc, dst_desc, OHWI(256, 2, 1, 1024), /*bias=*/nullptr,
      &src_exp_desc, /*different_weights_for_height=*/true, runtime_check);
  ASSERT_TRUE(op.AssembleCode(gpu_info).ok());

  EXPECT_TRUE(absl::StrContains(op.code_, "int src_end_ch = max(0, "));
  EXPECT_TRUE(
      absl::StrContains(op.code_, "int src_slices_dynamic = min((src_end_ch + 3) / 4, "));
  EXPECT_TRUE(
      absl::StrContains(op.code_, "bool in_k = raw_slice_id < src_slices_dynamic;"));
  EXPECT_FALSE(
      absl::StrContains(op.code_, "bool in_k = raw_slice_id < args.src"));
  EXPECT_TRUE(
      absl::StrContains(op.code_, "if (ch_base + 0 >= src_end_ch) src.x = 0.0h;"));
  EXPECT_TRUE(
      absl::StrContains(op.code_, "if (ch_base + 3 >= src_end_ch) src.w = 0.0h;"));
}

TEST(AgelloKernelCodegenTest, ConvAppleMppDynamicKNonSoftmaxUsesManualSrcRead) {
  const GpuInfo gpu_info = MakeAppleMetal4GpuInfo();
  TensorDescriptor src_desc(DataType::kFloat16, TensorStorageType::kBuffer,
                            Layout::kHWC,
                            TensorDescriptor::PhysicalLayout1D::kDHWBCC4);
  TensorDescriptor dst_desc(DataType::kFloat16, TensorStorageType::kBuffer,
                            Layout::kHWC);

  ConvRuntimeCheckDesc runtime_check;
  runtime_check.src_end_ch_index = 2;

  ConvAppleMPP op = CreateConvAppleMPPExternalWeights(
      src_desc, dst_desc, OHWI(256, 1, 1, 128), /*bias=*/nullptr,
      /*src_exp=*/nullptr, /*different_weights_for_height=*/false,
      runtime_check);
  ASSERT_TRUE(op.AssembleCode(gpu_info).ok());

  // Dynamic K must route through manual_src_reading (preserving kDHWBCC4 row
  // stride) and bound in_k to src_slices_dynamic.
  EXPECT_TRUE(absl::StrContains(op.code_, "threadgroup half a_loc["));
  EXPECT_FALSE(absl::StrContains(op.code_, "auto a_tensor = tensor(s_ptr"));
  EXPECT_TRUE(
      absl::StrContains(op.code_, "bool in_k = raw_slice_id < src_slices_dynamic;"));
}

TEST(AgelloKernelCodegenTest, ConvAppleMppWindowDstUsesKEndBound) {
  const GpuInfo gpu_info = MakeAppleMetal4GpuInfo();
  TensorDescriptor src_desc(DataType::kFloat16, TensorStorageType::kBuffer,
                            Layout::kHWC,
                            TensorDescriptor::PhysicalLayout1D::kDHWBCC4);
  TensorDescriptor dst_desc(DataType::kFloat16, TensorStorageType::kBuffer,
                            Layout::kHWC);

  ConvRuntimeCheckDesc runtime_check;
  runtime_check.src_end_ch_index = 2;
  runtime_check.window_pos_index = 0;
  runtime_check.window_size = 512;
  runtime_check.dst_window = true;

  ConvAppleMPP op = CreateConvAppleMPPExternalWeights(
      src_desc, dst_desc, OHWI(256, 2, 1, 1024), /*bias=*/nullptr,
      /*src_exp=*/nullptr, /*different_weights_for_height=*/true,
      runtime_check);
  ASSERT_TRUE(op.AssembleCode(gpu_info).ok());

  EXPECT_TRUE(absl::StrContains(op.code_, "bool in_k = raw_slice_id < k_end;"));
}

}  // namespace
}  // namespace ml_drift
