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

#import <Metal/Metal.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_model.h"
#include "ml_drift/common/kernels/quantize_and_dequantize.h"
#include "ml_drift/common/precision.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/metal/buffer.h"
#include "ml_drift/metal/compute_task.h"
#include "ml_drift/metal/environment.h"
#include "ml_drift/metal/inference_context.h"
#include "ml_drift/metal/memory_manager.h"
#include "ml_drift/metal/metal_spatial_tensor.h"

namespace ml_drift {
namespace metal {
namespace {

GpuModel MakeSingleBufferModel(ValueId id, const BHWC& shape) {
  GpuModel model;
  TensorDescriptor desc(DataType::kFloat16, TensorStorageType::kBuffer,
                        Layout::kHWC);
  desc.SetBHWCShape(shape);
  model.tensors[id] = desc;
  model.input_ids_and_refs.push_back({id, id});
  return model;
}

TEST(AgelloMetalSafetyTest, BufferCreateRejectsOversizedAllocationAndNullArgs) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (device == nil) {
    GTEST_SKIP() << "No Metal device available.";
  }

  constexpr size_t kHugeBytes = static_cast<size_t>(1) << 50;
  Buffer buffer;
  absl::Status status = CreateBuffer(kHugeBytes, nullptr, device, &buffer);
  EXPECT_EQ(status.code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(buffer.GetMemoryPtr(), nil);

  BufferDescriptor desc;
  desc.size = -1;
  status = buffer.CreateFromBufferDescriptor(desc, device);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(buffer.GetMemoryPtr(), nil);

  desc.size = 64;
  EXPECT_EQ(buffer.CreateFromBufferDescriptor(desc, nil).code(),
            absl::StatusCode::kInvalidArgument);
  desc.data.resize(16);
  EXPECT_EQ(buffer.CreateFromBufferDescriptor(desc, device).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(buffer.GetMemoryPtr(), nil);

  EXPECT_EQ(CreateBuffer(64, nullptr, device, nullptr).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(CreateBuffer(64, nullptr, nil, &buffer).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(AgelloMetalSafetyTest, CreateTensorSharedBufferRejectsNilBuffer) {
  TensorDescriptor buf_desc(DataType::kFloat16, TensorStorageType::kBuffer,
                            Layout::kHWC);
  buf_desc.SetBHWCShape(BHWC(1, 1, 1, 4));
  MetalSpatialTensor tensor;
  EXPECT_EQ(CreateTensorSharedBuffer(nil, buf_desc, &tensor).code(),
            absl::StatusCode::kInvalidArgument);

  TensorDescriptor tex_desc(DataType::kFloat16, TensorStorageType::kTexture2D,
                            Layout::kHWC);
  tex_desc.SetBHWCShape(BHWC(1, 1, 1, 4));
  EXPECT_EQ(CreateTensorSharedImage2DBuffer(nil, tex_desc, 64, &tensor).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(AgelloMetalSafetyTest, MemoryManagerTeardownClearsPendingRelease) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (device == nil) {
    GTEST_SKIP() << "No Metal device available.";
  }
  constexpr uint64_t kLargeBytes = 256ull << 20;
  if ([device maxBufferLength] < kLargeBytes) {
    GTEST_SKIP() << "Device maxBufferLength below 256 MiB.";
  }

  MemoryManager mm;
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF16;
  create_info.storage_type = TensorStorageType::kBuffer;

  // 4096 * 32768 * 2 bytes = 256 MiB (hits kMinReleaseBytes without touching pages).
  GpuModel model0 = MakeSingleBufferModel(/*id=*/1, BHWC(1, 1, 4096, 32768));
  GpuModel model1 = MakeSingleBufferModel(/*id=*/1, BHWC(1, 1, 1, 4));

  auto ctx0 = std::make_unique<InferenceContext>(&mm);
  ASSERT_TRUE(ctx0->InitFromGpuModel(create_info, &model0, device).ok());

  // Destroying an uninitialized InferenceContext sharing the same MemoryManager
  // must not unregister model 0.
  {
    InferenceContext unused(&mm);
  }

  InferenceContext ctx1(&mm);
  ASSERT_TRUE(ctx1.InitFromGpuModel(create_info, &model1, device).ok());

  EXPECT_FALSE(ctx0->PoolReleaseDue(/*is_prefill=*/true));
  EXPECT_TRUE(ctx1.PoolReleaseDue(/*is_prefill=*/false));

  // Destroy ctx0 while its release is pending; ReleasePoolBuffers must not throw
  // std::bad_function_call or dereference ctx0.
  ctx0.reset();
  EXPECT_TRUE(ctx1.ReleasePoolBuffers().ok());
}

TEST(AgelloMetalSafetyTest, EnsureBuffersRollsBackCleanlyOnRebindFailure) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (device == nil) {
    GTEST_SKIP() << "No Metal device available.";
  }
  constexpr uint64_t kLargeBytes = 256ull << 20;
  if ([device maxBufferLength] < kLargeBytes) {
    GTEST_SKIP() << "Device maxBufferLength below 256 MiB.";
  }

  Environment env(device);
  MemoryManager mm;
  ExternalTensorsInfo ext;
  std::vector<std::unique_ptr<MetalSpatialTensor>> ext_mut0;
  std::vector<std::unique_ptr<MetalSpatialTensor>> ext_mut1;

  GpuModel model0 = MakeSingleBufferModel(/*id=*/1, BHWC(1, 1, 4096, 32768));
  GpuModel model1 = MakeSingleBufferModel(/*id=*/1, BHWC(1, 1, 1, 4));

  auto m0_or = mm.AllocateMemory(env, model0, ext, ext_mut0);
  ASSERT_TRUE(m0_or.ok()) << m0_or.status();
  const MemoryManager::ModelId m0 = *m0_or;

  auto m1_or = mm.AllocateMemory(env, model1, ext, ext_mut1);
  ASSERT_TRUE(m1_or.ok()) << m1_or.status();
  const MemoryManager::ModelId m1 = *m1_or;

  bool fail_next_rebind = false;
  int rebind_calls = 0;
  mm.SetRebind(m0, [&](const absl::flat_hash_set<ValueId>& ids) -> absl::Status {
    ++rebind_calls;
    if (fail_next_rebind) {
      fail_next_rebind = false;
      return absl::InternalError("injected rebind failure");
    }
    return absl::OkStatus();
  });

  EXPECT_FALSE(mm.BeginRun(m0, /*is_prefill=*/true));
  EXPECT_TRUE(mm.BeginRun(m1, /*is_prefill=*/false));
  ASSERT_TRUE(mm.ReleasePending().ok());
  ASSERT_NE(mm.GetSpatialTensor({m0, 1}), nullptr);
  EXPECT_EQ(mm.GetSpatialTensor({m0, 1})->GetBufferHandle(), nil);

  // Inject failure during EnsureBuffers rebind: must roll back views/buffers to
  // nil and keep m0 in released_ so a later EnsureBuffers succeeds.
  fail_next_rebind = true;
  absl::Status status = mm.EnsureBuffers(m0);
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_EQ(mm.GetSpatialTensor({m0, 1})->GetBufferHandle(), nil);

  // Subsequent EnsureBuffers must reallocate and restore the tensor buffer.
  ASSERT_TRUE(mm.EnsureBuffers(m0).ok());
  EXPECT_NE(mm.GetSpatialTensor({m0, 1})->GetBufferHandle(), nil);
  EXPECT_GE(rebind_calls, 4);
}

TEST(AgelloMetalSafetyTest, QuantizationInt8Fp16OutlierOverflowAndZeroRow) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (device == nil) {
    GTEST_SKIP() << "No Metal device available.";
  }

  Environment env(device);
  const BHWC src_shape(1, 1, 4, 32);
  const BHWC params_shape(1, 1, 4, 2);

  TensorFloat32 src_cpu;
  src_cpu.shape = src_shape;
  src_cpu.data.assign(src_shape.DimensionsProduct(), 0.0f);
  // Row 0: large BOS outlier at (0, 0, 0).
  src_cpu.data[src_shape.LinearIndex({0, 0, 0, 0})] = 49152.0f;
  // Row 1: small range [0, 1] isolated from Row 0.
  for (int c = 0; c < src_shape.c; ++c) {
    src_cpu.data[src_shape.LinearIndex({0, 0, 1, c})] =
        static_cast<float>(c) / 31.0f;
  }
  // Row 2: symmetric wide range [-40960, +40960] whose span exceeds FP16 max.
  src_cpu.data[src_shape.LinearIndex({0, 0, 2, 0})] = -40960.0f;
  src_cpu.data[src_shape.LinearIndex({0, 0, 2, 1})] = 40960.0f;
  // Row 3: all zeros (src_range == 0).

  OperationDef op_def;
  op_def.src_tensors.push_back(
      {DataType::kFloat16, TensorStorageType::kBuffer, Layout::kHWC});
  op_def.dst_tensors.push_back(
      {DataType::kInt8, TensorStorageType::kBuffer, Layout::kHWC});
  op_def.dst_tensors.push_back(
      {DataType::kFloat16, TensorStorageType::kBuffer, Layout::kHWC});

  auto op = CreateQuantization(op_def, PackedType::kInt8C4, env.GetInfo(),
                               src_shape, /*calculate_sum=*/false);
  ASSERT_TRUE(op->AssembleCode(env.GetInfo()).ok());

  TensorDescriptor src_td = op_def.src_tensors[0];
  src_td.UploadData(src_cpu);
  TensorDescriptor dst_i8_td = op_def.dst_tensors[0];
  dst_i8_td.SetBHWCShape(src_shape);
  TensorDescriptor dst_params_td = op_def.dst_tensors[1];
  dst_params_td.SetBHWCShape(params_shape);

  MetalSpatialTensor src_gpu;
  MetalSpatialTensor dst_i8_gpu;
  MetalSpatialTensor dst_params_gpu;
  ASSERT_TRUE(src_gpu.CreateFromDescriptor(src_td, device).ok());
  ASSERT_TRUE(dst_i8_gpu.CreateFromDescriptor(dst_i8_td, device).ok());
  ASSERT_TRUE(dst_params_gpu.CreateFromDescriptor(dst_params_td, device).ok());

  ComputeTask task;
  task.Init(std::move(op));
  ASSERT_TRUE(task.Compile(&env).ok());
  ASSERT_TRUE(task.SetSrcTensor(&src_gpu, 0).ok());
  ASSERT_TRUE(task.SetDstTensor(&dst_i8_gpu, 0).ok());
  ASSERT_TRUE(task.SetDstTensor(&dst_params_gpu, 1).ok());
  ASSERT_TRUE(task.UpdateParams().ok());

  id<MTLCommandQueue> queue = [device newCommandQueue];
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  task.Encode(enc);
  [enc endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  ASSERT_EQ(cmd.status, MTLCommandBufferStatusCompleted);

  TensorDescriptor out_i8_desc;
  TensorDescriptor out_params_desc;
  ASSERT_TRUE(dst_i8_gpu.ToDescriptor(&out_i8_desc, device).ok());
  ASSERT_TRUE(dst_params_gpu.ToDescriptor(&out_params_desc, device).ok());

  Tensor<BHWC, DataType::kInt8> dst_i8;
  out_i8_desc.DownloadData(&dst_i8);
  TensorFloat32 dst_params;
  out_params_desc.DownloadData(&dst_params);

  for (float p : dst_params.data) {
    EXPECT_TRUE(std::isfinite(p));
  }

  // Row 1 must reflect [0, 1], unaffected by Row 0's 49152 outlier.
  EXPECT_NEAR(dst_params.data[params_shape.LinearIndex({0, 0, 1, 0})],
              1.0f / 255.0f, 0.005f);
  EXPECT_NEAR(dst_params.data[params_shape.LinearIndex({0, 0, 1, 1})],
              128.0f / 255.0f, 0.005f);
  EXPECT_EQ(dst_i8.data[src_shape.LinearIndex({0, 0, 1, 0})], -128);
  EXPECT_EQ(dst_i8.data[src_shape.LinearIndex({0, 0, 1, 31})], 127);

  // Row 2 has span 81920 (> FP16 max); inv_scale must remain finite (~321.25).
  EXPECT_NEAR(dst_params.data[params_shape.LinearIndex({0, 0, 2, 0})],
              81920.0f / 255.0f, 1.0f);
  EXPECT_EQ(dst_i8.data[src_shape.LinearIndex({0, 0, 2, 0})], -128);
  EXPECT_EQ(dst_i8.data[src_shape.LinearIndex({0, 0, 2, 1})], 127);

  // Row 3 (all zeros) must produce 0 scale/offset and finite -128 outputs.
  EXPECT_FLOAT_EQ(dst_params.data[params_shape.LinearIndex({0, 0, 3, 0})],
                  0.0f);
  EXPECT_FLOAT_EQ(dst_params.data[params_shape.LinearIndex({0, 0, 3, 1})],
                  0.0f);
  for (int c = 0; c < src_shape.c; ++c) {
    EXPECT_EQ(dst_i8.data[src_shape.LinearIndex({0, 0, 3, c})], -128);
  }
}

}  // namespace
}  // namespace metal
}  // namespace ml_drift
