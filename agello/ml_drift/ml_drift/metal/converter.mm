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

#include "ml_drift/metal/converter.h"

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "ml_drift/common/kernels/conversion.h"

namespace ml_drift {
namespace metal {

absl::Status TensorToBHWCBufferConverter::Init(Environment* env,
                                               const TensorDescriptor& src_desc,
                                               DataType buffer_type) {
  BufferDescriptor dst_desc;
  dst_desc.element_type = buffer_type;
  dst_desc.element_size = 1;
  dst_desc.memory_type = MemoryType::kGlobal;

  const auto& gpu_info = env->GetInfo();
  TensorToBhwcBuffer gpu_op =
      CreateTensorToBhwcBufferOp(gpu_info, src_desc, dst_desc);
  gpu_op.RecalculateWorkGroupsSize(gpu_info, src_desc.GetBHWCShape());
  ABSL_RETURN_IF_ERROR(gpu_op.AssembleCode(gpu_info));
  task_.Init(std::make_unique<TensorToBhwcBuffer>(std::move(gpu_op)));
  return task_.Compile(env);
}

namespace {
// Agello: a converter's task kept its source and destination bound until its next encode, so
// between runs it held the buffers it last converted: a pool buffer the large prefill's release
// had freed (a third reference on two 256 MB and one 128 MB buffer at 64K), and the executor's
// input buffers. Once encoded, the command buffer keeps what it needs; transfers of a megabyte or
// more are unbound (views on no buffer, as the pool's release makes), small ones keep their
// bindings so decode pays nothing.
constexpr uint64_t kUnbindBytes = 1ull << 20;

absl::Status Unbind(ComputeTask& task, const MetalSpatialTensor& tensor, bool tensor_is_src,
                    size_t buffer_bytes) {
  MetalSpatialTensor none(nil, nil, /*memory_owner=*/false, /*texture_mem_owner=*/false,
                          tensor.GetDescriptor());
  Buffer no_buffer(nil, buffer_bytes);
  if (tensor_is_src) {
    ABSL_RETURN_IF_ERROR(task.SetSrcTensor(&none, 0));
    return task.SetDstBuffer(&no_buffer, 0);
  }
  ABSL_RETURN_IF_ERROR(task.SetSrcBuffer(&no_buffer, 0));
  return task.SetDstTensor(&none, 0);
}
}  // namespace

absl::Status TensorToBHWCBufferConverter::Encode(
    id<MTLComputeCommandEncoder> encoder, MetalSpatialTensor* src,
    Buffer* dst) {
  ABSL_RETURN_IF_ERROR(task_.SetSrcTensor(src, 0));
  ABSL_RETURN_IF_ERROR(task_.SetDstBuffer(dst, 0));
  ABSL_RETURN_IF_ERROR(task_.UpdateParams());
  task_.Encode(encoder);
  if (src->GetMemorySizeInBytes() >= kUnbindBytes) {
    return Unbind(task_, *src, /*tensor_is_src=*/true, dst->GetMemorySizeInBytes());
  }
  return absl::OkStatus();
}

absl::Status TensorToBHWCBufferConverter::Encode(
    id<MTLComputeCommandEncoder> encoder, MetalSpatialTensor* src,
    id<MTLBuffer> dst) {
  Buffer wrapped = CreateBufferShared(dst);
  return Encode(encoder, src, &wrapped);
}

absl::Status BHWCBufferToTensorConverter::Init(
    Environment* env, DataType buffer_type, const TensorDescriptor& dst_desc) {
  BufferDescriptor src_desc;
  src_desc.element_type = buffer_type;
  src_desc.element_size = 1;
  src_desc.memory_type = MemoryType::kGlobal;

  const auto& gpu_info = env->GetInfo();
  BhwcBufferToTensor gpu_op =
      CreateBhwcBufferToTensorOp(gpu_info, src_desc, dst_desc);
  gpu_op.RecalculateWorkGroupsSize(gpu_info, dst_desc.GetBHWCShape());
  ABSL_RETURN_IF_ERROR(gpu_op.AssembleCode(gpu_info));
  task_.Init(std::make_unique<GPUOperation>(std::move(gpu_op)));
  return task_.Compile(env);
}

absl::Status BHWCBufferToTensorConverter::Encode(
    id<MTLComputeCommandEncoder> encoder, Buffer* src,
    MetalSpatialTensor* dst) {
  ABSL_RETURN_IF_ERROR(task_.SetSrcBuffer(src, 0));
  ABSL_RETURN_IF_ERROR(task_.SetDstTensor(dst, 0));
  ABSL_RETURN_IF_ERROR(task_.UpdateParams());
  task_.Encode(encoder);
  if (dst->GetMemorySizeInBytes() >= kUnbindBytes) {
    return Unbind(task_, *dst, /*tensor_is_src=*/false, src->GetMemorySizeInBytes());
  }
  return absl::OkStatus();
}

absl::Status BHWCBufferToTensorConverter::Encode(
    id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> src,
    MetalSpatialTensor* dst) {
  Buffer wrapped = CreateBufferShared(src);
  return Encode(encoder, &wrapped, dst);
}

}  // namespace metal
}  // namespace ml_drift
