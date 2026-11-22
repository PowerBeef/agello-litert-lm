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

#include "ml_drift/common/kernels/conv_apple_mpp.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <variant>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_replace.h"
#include "ml_drift/common/data_type.h"
#include "ml_drift/common/gpu_info.h"
#include "ml_drift/common/kernels/fully_connected_util.h"
#include "ml_drift/common/operations.h"
#include "ml_drift/common/shape.h"
#include "ml_drift/common/task/buffer_desc.h"
#include "ml_drift/common/task/gpu_object_desc.h"
#include "ml_drift/common/task/gpu_operation.h"
#include "ml_drift/common/task/tensor_desc.h"
#include "ml_drift/common/task/weights_layout.h"
#include "ml_drift/common/tensor.h"
#include "ml_drift/common/types.h"
#include "ml_drift/common/util.h"

namespace ml_drift {
namespace {
void AddRuntimeParams(GPUOperation& op,
                      const ConvRuntimeCheckDesc& runtime_check) {
  bool has_runtime_check = false;
  if (runtime_check.src_end_ch_index.has_value()) {
    op.args_.AddInt("src_end_ch_index", *runtime_check.src_end_ch_index);
    has_runtime_check = true;
  }
  if (runtime_check.dst_end_ch_index.has_value()) {
    op.args_.AddInt("dst_end_ch_index", *runtime_check.dst_end_ch_index);
    has_runtime_check = true;
  }
  if (runtime_check.window_pos_index.has_value()) {
    op.args_.AddInt("window_pos_index", *runtime_check.window_pos_index);
    has_runtime_check = true;
  }
  if (runtime_check.packed_groups.has_value()) {
    op.args_.AddInt("packed_params_offset",
                    runtime_check.packed_groups->params_offset);
    has_runtime_check = true;
  }
  if (has_runtime_check) {
    BufferDescriptor buffer_desc;
    buffer_desc.element_type = DataType::kInt32;
    buffer_desc.element_size = 1;
    buffer_desc.memory_type = MemoryType::kConstant;
    op.AddSrcBuffer("params", buffer_desc);
  }
}

std::string ReadFloatWeights(const ConvAppleMPP::ConvParams& params) {
  const bool weights_conversion =
      params.weights_desc.layout == WeightsLayout::kOSpatialIOGroupI4O4;
  const bool quantized_weights =
      weights_conversion && SizeInBitsOf(params.weights_desc.type) <= 8;
  std::string c;
  if (quantized_weights && params.scale_zp_shape.i != 1) {
    // grouped quantization
    c += "    int src_group_id = (k + sub_i) / " +
         std::to_string(params.src_group_slices) + +";\n";
    c += "    if (last_src_group_id != src_group_id) {\n";
    c += "      last_src_group_id = src_group_id;\n";
    c += fc::ReadScaleZeroPointBlock(
        "w_o_slice", params.scale_zp_shape, params.has_zero_point,
        fc::GetDataTypeForWeights(params.weights_desc.type));
    c += "    }\n";
  }
  if (params.weights_desc.type == DataType::kUint8) {
    c += "    uint4 u8_i4o4 = args.weights.Read(w_wg_offset);\n";
    c += "    ucl::U32x4ToU8x16AsVec4x4<half>(u8_i4o4, w0, w1, w2, w3);\n";
  } else if (params.weights_desc.type == DataType::kUint4) {
    c += "    uint2 u4_i4o4 = args.weights.Read(w_wg_offset);\n";
    c += "    ucl::U32x2ToU4x16AsVec4x4<half>(u4_i4o4, w0, w1, w2, w3);\n";
  } else if (params.weights_desc.type == DataType::kUint2) {
    c += "    uint u2_i4o4 = args.weights.Read(w_wg_offset);\n";
    c += "    ucl::U32x1ToU2x16AsVec4x4<half>(u2_i4o4, w0, w1, w2, w3);\n";
  } else {
    c += "    args.weights.ReadVec16AsVec4x4(w0, w1, w2, w3, w_wg_offset);\n";
  }
  if (quantized_weights) {
    c += R"(
    w0 = w0 * w_scale + w_bias;
    w1 = w1 * w_scale + w_bias;
    w2 = w2 * w_scale + w_bias;
    w3 = w3 * w_scale + w_bias;
)";
  }
  return c;
}

void AddXKernelParams(const Convolution2DAttributes& attr, GPUOperation* op) {
  const OHWI& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  op->args_.AddInt("stride_x", attr.strides.w);
  op->args_.AddInt("padding_x", -attr.padding.prepended.w);
  op->args_.AddInt("kernel_size_x", weights_shape.w);
  op->args_.AddInt("dilation_x", attr.dilations.w);
}

void AddYKernelParams(const Convolution2DAttributes& attr, GPUOperation* op) {
  const OHWI& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);
  op->args_.AddInt("stride_y", attr.strides.h);
  op->args_.AddInt("padding_y", -attr.padding.prepended.h);
  op->args_.AddInt("kernel_size_y", weights_shape.h);
  op->args_.AddInt("dilation_y", attr.dilations.h);
}

void ApplyDequantParams(ConvAppleMPP::ConvParams& params,
                        const TensorDescriptor& dst,
                        const ConvAppleMPPDequantParams* dequant) {
  if (dequant == nullptr) return;
  params.has_dequant = true;
  params.dequant_has_zero_point = dequant->weights_zero_point != nullptr;
  params.dequant_params_has_batch = dequant->src_params->HasAxis(Axis::kBatch);
  params.dequant_dst_type = dst.GetDataType();
}

void AttachDequantArgs(ConvAppleMPP& conv,
                       const ConvAppleMPPDequantParams* dequant) {
  if (dequant == nullptr) return;
  conv.AddSrcTensor("params_tensor", *dequant->src_params);
  conv.AddSrcTensor("weights_sum", *dequant->weights_sum);
  conv.AddSrcTensor("weights_scale", *dequant->weights_scale);
  if (dequant->weights_zero_point != nullptr) {
    conv.AddSrcTensor("weights_zero_point", *dequant->weights_zero_point);
  }
  conv.args_.AddFloat("in_channels", dequant->in_channels);
}

std::string EmitRowCoords(const ConvAppleMPP::ConvParams& params,
                          bool has_batch, const std::string& row_expr,
                          const std::string& pfx, const std::string& indent) {
  std::string c;
  if (params.runtime_check.packed_groups.has_value()) {
    c += indent + "int " + pfx + "_h = 0;\n";
    c += indent + "int " + pfx + "_w = w_group_offset + (" + row_expr + ");\n";
    c += indent + "bool " + pfx + "_ok = ((" + row_expr +
         ") < w_group_size) && (" + pfx + "_w < args.dst.Width());\n";
  } else if (params.batched_weights) {
    c += indent + "int " + pfx + "_h = w_batch_id;\n";
    if (has_batch) {
      c += indent + "int " + pfx + "_b = (" + row_expr +
           ") % args.dst.Batch();\n";
      c += indent + "int " + pfx + "_w = (" + row_expr +
           ") / args.dst.Batch();\n";
    } else {
      c += indent + "int " + pfx + "_w = (" + row_expr + ");\n";
    }
    c += indent + "bool " + pfx + "_ok = (" + pfx + "_w < args.dst.Width());\n";
  } else {
    if (has_batch) {
      c += indent + "int " + pfx + "_b = (" + row_expr +
           ") % args.dst.Batch();\n";
      c += indent + "int " + pfx + "_rem = (" + row_expr +
           ") / args.dst.Batch();\n";
      c += indent + "int " + pfx + "_w = " + pfx + "_rem % args.dst.Width();\n";
      c += indent + "int " + pfx + "_h = " + pfx + "_rem / args.dst.Width();\n";
    } else {
      c += indent + "int " + pfx + "_w = (" + row_expr +
           ") % args.dst.Width();\n";
      c += indent + "int " + pfx + "_h = (" + row_expr +
           ") / args.dst.Width();\n";
    }
    c += indent + "bool " + pfx + "_ok = (" + pfx +
         "_h < args.dst.Height());\n";
  }
  return c;
}
}  // namespace

absl::Status ConvAppleMPP::PrepareCode(const GpuInfo& gpu_info) {
  TensorDescriptor dst_desc;
  ABSL_RETURN_IF_ERROR(GetTensorDescriptor("dst", &dst_desc));
  code_ = GetKernelCode(src_desc_, dst_desc);
  ResetConstExprResolved();
  return absl::OkStatus();
}

std::string ConvAppleMPP::GetKernelCode(const TensorDescriptor& src,
                                        const TensorDescriptor& dst) const {
  const bool weights_conversion =
      params_.weights_desc.layout == WeightsLayout::kOSpatialIOGroupI4O4;
  const bool quantized_weights =
      weights_conversion && SizeInBitsOf(params_.weights_desc.type) <= 8;
  const bool manual_src_reading =
      params_.softmax_input_activation ||
      src.GetStorageType() != TensorStorageType::kBuffer ||
      !src.IsCC4Layout() || !params_.x_kernel_is_1 || !params_.y_kernel_is_1;
  const bool manual_k_tiling =
      params_.runtime_check.src_end_ch_index.has_value() ||
      manual_src_reading || weights_conversion;
  const int k_tile =
      manual_k_tiling ? 32 : AlignByN(params_.weights_shape.i, 4);
  const bool has_batch = src.HasAxis(Axis::kBatch);
  const bool dst_has_batch = dst.HasAxis(Axis::kBatch);
  const DataType accum_type = params_.weights_data_type == DataType::kInt8
                                  ? DataType::kInt32
                                  : DataType::kFloat16;
  const DataType out_type =
      params_.has_dequant ? params_.dequant_dst_type : accum_type;
  std::string c;
  c += "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";
  if (weights_conversion && params_.weights_data_type != DataType::kFloat16 &&
      params_.weights_desc.type == DataType::kUint4) {
    c += R"(
uint32_t expand_int4_to_int8(uint32_t val) {
  uint32_t x = val & 0xFFFF;
  x = (x | (x << 8)) & 0x00FF00FF;
  x = (x | (x << 4)) & 0x0F0F0F0F;
  uint32_t signs = x & 0x08080808;
  uint32_t sign_extensions = signs * 0x1E;
  return x | sign_extensions;
}
)";
  }
  c += R"(
MAIN_FUNCTION($0) {
  int slice_tile_id = ucl::GetGroupId<0>();
  int spatial_tile_id = ucl::GetGroupId<1>();
)";
  if (!params_.runtime_check.packed_groups.has_value()) {
    c += R"(  {
    int gx = slice_tile_id;
    int gy = spatial_tile_id;
    int GX = (args.dst.Slices() + N_TILE_SLICES - 1) / N_TILE_SLICES;
    int GY = (SPATIAL_SIZE + M_TILE - 1) / M_TILE;
    if (GX > 1 && GY > 1) {
      int linear = gy * GX + gx;
      int band_size = 4 * GX;
      int full_bands = GY >> 2;
      int full_count = full_bands * band_size;
      if (linear < full_count) {
        int band_id = linear / band_size;
        int in_band = linear - band_id * band_size;
        int full_x = GX & ~3;
        int full_band_tiles = full_x << 2;
        if (in_band < full_band_tiles) {
          int blk_x = in_band >> 4;
          int idx = in_band & 15;
          int lx = (idx & 1) | ((idx >> 1) & 2);
          int ly = ((idx >> 1) & 1) | ((idx >> 2) & 2);
          slice_tile_id = (blk_x << 2) + lx;
          spatial_tile_id = (band_id << 2) + ly;
        } else {
          int rem = in_band - full_band_tiles;
          int tail_w = GX - full_x;
          slice_tile_id = full_x + (rem % tail_w);
          spatial_tile_id = (band_id << 2) + (rem / tail_w);
        }
      } else {
        int rem = linear - full_count;
        int y0 = full_bands << 2;
        int rem_h = GY - y0;
        if (rem_h >= 2) {
          int band2_size = 2 * GX;
          if (rem < band2_size) {
            int full_x2 = GX & ~1;
            int full_b2 = full_x2 << 1;
            if (rem < full_b2) {
              int blk_x = rem >> 2;
              int idx = rem & 3;
              slice_tile_id = (blk_x << 1) + (idx & 1);
              spatial_tile_id = y0 + (idx >> 1);
            } else {
              int t = rem - full_b2;
              slice_tile_id = full_x2;
              spatial_tile_id = y0 + t;
            }
          } else {
            int t = rem - band2_size;
            slice_tile_id = t;
            spatial_tile_id = y0 + 2;
          }
        } else {
          slice_tile_id = rem;
          spatial_tile_id = y0;
        }
      }
    }
  }
)";
  }
  c += R"(  int sub_spatial_id = ucl::GetLocalId<0>() % M_TILE;
  int sub_slice_id = ucl::GetLocalId<0>() / M_TILE;
  int slices_per_wg = WG_SIZE / M_TILE;
  int spatial_id = spatial_tile_id * M_TILE + sub_spatial_id;
)";
  if (has_batch) {
    c += "  int dst_b = spatial_id % args.dst.Batch();\n";
    c += "  spatial_id = spatial_id / args.dst.Batch();\n";
  }
  if (params_.batched_weights) {
    c += "  int dst_w = spatial_id;\n";
    c += "  int dst_h = ucl::GetGroupId<2>();\n";
  } else {
    c += "  int dst_w = spatial_id % args.dst.Width();\n";
    c += "  int dst_h = spatial_id / args.dst.Width();\n";
  }
  if (params_.runtime_check.packed_groups.has_value()) {
    c += "  int w_batch_id = dst_h;\n";
    c += "  dst_h = 0;\n";
    c += "  int w_group_size = args.params.Read(args.packed_params_offset + "
         "w_batch_id);\n";
    c += "  int w_group_offset = args.params.Read(args.packed_params_offset + "
         "w_batch_id + " +
         std::to_string(params_.runtime_check.packed_groups->num_groups) +
         ");\n";
    c += "  int wg_first_w = spatial_tile_id * M_TILE;\n";
    c += "  if (wg_first_w >= w_group_size) return;\n";
    c += "  dst_w = w_group_offset + dst_w;\n";
  } else if (params_.batched_weights) {
    c += "  int w_batch_id = dst_h;\n";
  }
  if (params_.runtime_check.dst_end_ch_index.has_value()) {
    c += "  int dst_end_slice_runtime = " +
         params_.runtime_check.GetRuntimeEndSlice(
             "args.params.Read(args.dst_end_ch_index)", "args.dst.Slices()") +
         ";\n";
    c += "  if (slice_tile_id * N_TILE_SLICES >= dst_end_slice_runtime) "
         "return;\n";
  }
  const ConvRuntimeCheckDesc& window = params_.runtime_check;
  const bool use_window_src = window.window_pos_index.has_value() &&
                              window.src_window && manual_k_tiling &&
                              params_.x_kernel_is_1 && params_.y_kernel_is_1 &&
                              !window.packed_groups.has_value();
  const bool use_window_dst = window.window_pos_index.has_value() &&
                              window.dst_window && params_.x_kernel_is_1 &&
                              params_.y_kernel_is_1 &&
                              !window.packed_groups.has_value();
  if (use_window_src || use_window_dst) {
    c += "  int window_start_slice = " +
         RuntimeWindowStartSlice(window.window_size) + ";\n";
  }
  if (weights_conversion) {
    ABSL_CHECK(n_tile_ == 64);
    ABSL_CHECK(k_tile == 32);
    ABSL_CHECK(simdgroups_ == 4);
    const std::string w_loc_type =
        params_.weights_data_type == DataType::kFloat16 ? "half" : "int8_t";
    const std::string w_loc_x4_type =
        params_.weights_data_type == DataType::kFloat16 ? "half4" : "uint";
    c += "  threadgroup " + w_loc_type + " w_loc[K_TILE * N_TILE];\n";
    c += "  threadgroup " + w_loc_x4_type + "* w_loc_x4 = (threadgroup " +
         w_loc_x4_type + "*)(w_loc);\n";
    c += R"(
  auto b_tile = tensor(w_loc, dextents<int, 2>(N_TILE, K_TILE));

  int loc_id = ucl::GetLocalId<0>();
  int sub_i = loc_id / 16;
  int sub_o = loc_id % 16;

  int w_o_slice = min(slice_tile_id * N_TILE_SLICES + sub_o, args.dst.Slices() - 1);
  int w_stride = args.dst.Slices() * 8;
)";
    const std::string batch_part =
        params_.batched_weights ? "w_batch_id * args.src.Slices()" : "0";
    c += "  int w_wg_offset = (" + batch_part +
         " + sub_i) * args.dst.Slices() + w_o_slice;\n";
    if (quantized_weights && params_.weights_data_type == DataType::kFloat16) {
      c += "  half4 w_scale, w_bias;\n";
      if (params_.scale_zp_shape.i != 1) {
        c += "  int last_src_group_id = -1;\n";
      } else if (params_.scale_zp_shape.o != 1) {
        c += fc::ReadScaleZeroPointLinear(
            "w_o_slice", params_.scale_zp_shape, params_.has_zero_point,
            fc::GetDataTypeForWeights(params_.weights_desc.type));
      } else {
        c += fc::ReadScaleZeroPointScalar(
            params_.has_zero_point,
            fc::GetDataTypeForWeights(params_.weights_desc.type));
      }
    }
  } else {
    c += R"(
  device SType* w_ptr = reinterpret_cast<device SType*>(args.weights.GetPtr());
  int b_rows = args.src.Slices() * 4;
  int b_cols = args.dst.Slices() * 4;
)";
    std::string spatial_size;
    if (!params_.x_kernel_is_1) {
      spatial_size += " * args.kernel_size_x";
    }
    if (!params_.y_kernel_is_1) {
      spatial_size += " * args.kernel_size_y";
    }
    if (!spatial_size.empty()) {
      c += "  b_rows = b_rows" + spatial_size + ";\n";
    }
    if (params_.batched_weights) {
      c += "  w_ptr += w_batch_id * b_rows * b_cols;\n";
    }
    c += "  auto b_tensor = tensor(w_ptr, dextents<int, 2>(b_cols, b_rows));\n";
    c += "  auto b_tile = b_tensor.slice(slice_tile_id * N_TILE, 0);\n";
  }
  if (manual_src_reading) {
    if (has_batch) {
      c += "  args.src.SetBatchRef(dst_b);\n";
    }
    c += R"(
  threadgroup half a_loc[K_TILE * M_TILE];
  threadgroup half4* a_loc_x4 = (threadgroup half4*)(a_loc);
  auto a_tile = tensor(a_loc, dextents<int, 2>(K_TILE, M_TILE));
)";
  } else {
    c += R"(
  device SType* s_ptr = reinterpret_cast<device SType*>(args.src.GetHandle());
  int a_rows = SPATIAL_SIZE;
  int a_cols = args.src.Slices() * 4;
)";
    if (params_.runtime_check.packed_groups.has_value()) {
      c += "  s_ptr += w_group_offset * a_cols;\n";
    } else {
      if (params_.batched_weights) {
        c += "  s_ptr += w_batch_id * a_rows * a_cols;\n";
      }
    }
    c += "  auto a_tensor = tensor(s_ptr, dextents<int, 2>(a_cols, a_rows));\n";
    c += "  auto a_tile = a_tensor.slice(0, spatial_tile_id * M_TILE);\n";
  }
  c += R"(
  constexpr auto matmul_desc = mpp::tensor_ops::matmul2d_descriptor(M_TILE, N_TILE, K_TILE, false, false, false, MAT_MUL_MODE);
  mpp::tensor_ops::matmul2d<matmul_desc, execution_simdgroups<SIMDGROUPS>> matmul_op;

  auto c_tile = matmul_op.get_destination_cooperative_tensor<decltype(a_tile), decltype(b_tile), ACCUM_SCALAR>();
  #pragma unroll_full
  for (uint16_t i = 0; i < c_tile.get_capacity(); ++i) {
    if(c_tile.is_valid_element(i)) c_tile[i] = 0;
  }
)";
  std::string src_end_slice = "args.src.Slices()";
  if (params_.runtime_check.src_end_ch_index.has_value()) {
    c += "  int src_slices_dynamic = " +
         params_.runtime_check.GetRuntimeEndSlice(
             "args.params.Read(args.src_end_ch_index)", "args.src.Slices()") +
         ";\n";
    src_end_slice = "src_slices_dynamic";
  }
  if (!params_.x_kernel_is_1 || !params_.y_kernel_is_1) {
    c += "  int w_offset = 0;\n";
  }
  if (!params_.y_kernel_is_1) {
    c += "  for (int ky = 0; ky < args.kernel_size_y; ++ky) {\n";
    c += "  int temp_y = dst_h * args.stride_y + args.padding_y;\n";
    c += "  int src_y = ky * args.dilation_y + temp_y;\n";
    c += "  bool in_y = src_y >= 0 && src_y < args.src.Height();\n";
    c += "  src_y = clamp(src_y, 0, args.src.Height() - 1);\n";
  } else {
    c += "  int src_y = min(dst_h, args.src.Height() - 1);\n";
  }
  if (!params_.x_kernel_is_1) {
    c += "  for (int kx = 0; kx < args.kernel_size_x; ++kx) {\n";
    c += "  int temp_x = dst_w * args.stride_x + args.padding_x;\n";
    c += "  int src_x = kx * args.dilation_x + temp_x;\n";
    c += "  bool in_x = src_x >= 0 && src_x < args.src.Width();\n";
    c += "  src_x = clamp(src_x, 0, args.src.Width() - 1);\n";
  } else {
    c += "  int src_x = min(dst_w, args.src.Width() - 1);\n";
  }
  if (params_.softmax_input_activation) {
    c += "  half2 exp_val = args.src_exp.Read(src_x, src_y, 0).xy;\n";
  }
  if (manual_k_tiling) {
    std::string k_start = "0";
    std::string k_end = src_end_slice;
    if (use_window_src) {
      k_start = "window_start_slice";
      if (weights_conversion) {
        c += "  w_wg_offset += (window_start_slice / K_TILE_SLICES) * w_stride;\n";
      }
    }
    if (use_window_dst) {
      c += "  int k_end = (slice_tile_id + 1) * N_TILE_SLICES <= window_start_slice ? 0 : " +
           src_end_slice + ";\n";
      k_end = "k_end";
    }
    c += "  for (int k = " + k_start + "; k < " + k_end + "; k += K_TILE_SLICES) {\n";
    if (weights_conversion) {
      if (params_.weights_data_type == DataType::kFloat16) {
        c += "    half4 w0, w1, w2, w3;\n";
        c += ReadFloatWeights(params_);
      } else if (params_.weights_desc.type == DataType::kUint8) {
        c += R"(
    uint4 u8_i4o4 = args.weights.Read(w_wg_offset);
    uint w0 = u8_i4o4.x ^ 0x80808080u;
    uint w1 = u8_i4o4.y ^ 0x80808080u;
    uint w2 = u8_i4o4.z ^ 0x80808080u;
    uint w3 = u8_i4o4.w ^ 0x80808080u;
)";
      } else {
        c += R"(
    uint2 u4_i4o4 = args.weights.Read(w_wg_offset);
    u4_i4o4.x ^= 0x88888888u;
    u4_i4o4.y ^= 0x88888888u;
    uint w0 = expand_int4_to_int8(u4_i4o4.x);
    uint w1 = expand_int4_to_int8(u4_i4o4.x >> 16);
    uint w2 = expand_int4_to_int8(u4_i4o4.y);
    uint w3 = expand_int4_to_int8(u4_i4o4.y >> 16);
)";
      }
      c += "    w_wg_offset += w_stride;\n";
      c += R"(
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // store it in i32o64 or half4 in i32o16
    w_loc_x4[(sub_i * 4 + 0) * 16 + sub_o] = w0;
    w_loc_x4[(sub_i * 4 + 1) * 16 + sub_o] = w1;
    w_loc_x4[(sub_i * 4 + 2) * 16 + sub_o] = w2;
    w_loc_x4[(sub_i * 4 + 3) * 16 + sub_o] = w3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
)";
    } else {
      if (!params_.x_kernel_is_1 || !params_.y_kernel_is_1) {
        c += "    b_tile = b_tensor.slice(slice_tile_id * N_TILE, w_offset);\n";
        c += "    w_offset += K_TILE_SLICES * 4;\n";
      } else {
        c += "    b_tile = b_tensor.slice(slice_tile_id * N_TILE, k * 4);\n";
      }
    }
    if (manual_src_reading) {
      c += R"(
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll_full
    for (uint16_t i = 0; i < K_TILE_SLICES / slices_per_wg; ++i) {
      int tile_k_id = i * slices_per_wg + sub_slice_id;
      int slice_id = min(k + tile_k_id, args.src.Slices() - 1);
      half4 src = args.src.Read(src_x, src_y, slice_id);
)";
      std::string check;
      if (!params_.x_kernel_is_1) {
        check += "in_x";
      }
      if (!params_.y_kernel_is_1) {
        if (!check.empty()) {
          check += " && ";
        }
        check += "in_y";
      }
      if (!check.empty()) {
        c += "      src *= ucl::Convert<half>((" + check + "));\n";
      }
      if (params_.softmax_input_activation) {
        c += "      src = exp(src - exp_val.y) * exp_val.x;\n";
      }
      c += R"(
      a_loc_x4[sub_spatial_id * K_TILE_SLICES + tile_k_id] = src;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
)";
    } else {
      c += "    a_tile = a_tensor.slice(k * 4, spatial_tile_id * M_TILE);\n";
    }
    c += "    matmul_op.run(a_tile, b_tile, c_tile);\n";
    c += "  }\n";
  } else {
    if (use_window_dst) {
      c += "  if ((slice_tile_id + 1) * N_TILE_SLICES > window_start_slice) {\n";
      c += "    matmul_op.run(a_tile, b_tile, c_tile);\n";
      c += "  }\n";
    } else {
      c += "  matmul_op.run(a_tile, b_tile, c_tile);\n";
    }
  }
  if (!params_.x_kernel_is_1) {
    c += "  }\n";
  }
  if (!params_.y_kernel_is_1) {
    c += "  }\n";
  }
  const int strip_m = std::min(m_tile_, 32);
  if (params_.has_dequant && (m_tile_ % strip_m == 0) && (n_tile_ % 16 == 0)) {
    // Param-cached 32-row half strip epilogue: shrinks threadgroup memory from
    // 32,768 B (int32[128*64], 1 TG/core cap) to 9,728-10,496 B (3 TG/core),
    // with zero global loads or div/mod inside the c_tile register loop and
    // fully coalesced vec4 writes for all linked elementwise ops.
    c += "  {\n";
    c += "    threadgroup float tg_s_scale[M_TILE];\n";
    c += "    threadgroup float tg_s_zp[M_TILE];\n";
    if (params_.dequant_has_zero_point) {
      c += "    threadgroup float tg_s_zp_term[M_TILE];\n";
    }
    c += "    threadgroup float4 tg_w_scale4[N_TILE_SLICES];\n";
    c += "    threadgroup float4 tg_w_ssum4[N_TILE_SLICES];\n";
    if (params_.dequant_has_zero_point) {
      c += "    threadgroup float4 tg_w_zp4[N_TILE_SLICES];\n";
    }
    c += "    threadgroup float* tg_w_scale = (threadgroup float*)(tg_w_scale4);\n";
    c += "    threadgroup float* tg_w_ssum = (threadgroup float*)(tg_w_ssum4);\n";
    if (params_.dequant_has_zero_point) {
      c += "    threadgroup float* tg_w_zp = (threadgroup float*)(tg_w_zp4);\n";
    }
    c += "    int ep_tid = ucl::GetLocalId<0>();\n";
    c += "    if (ep_tid < M_TILE) {\n";
    c += "      int p_row = spatial_tile_id * M_TILE + ep_tid;\n";
    c += EmitRowCoords(params_, dst_has_batch, "p_row", "pr", "      ");
    c += "      if (pr_ok) {\n";
    std::string pr_coords = "pr_w, pr_h, 0";
    if (params_.dequant_params_has_batch) {
      pr_coords += ", pr_b";
    }
    c += "        float4 sp = args.params_tensor.Read<float>(" + pr_coords + ");\n";
    c += "        tg_s_scale[ep_tid] = sp.x;\n";
    c += "        tg_s_zp[ep_tid] = sp.y;\n";
    if (params_.dequant_has_zero_point) {
      c += "        tg_s_zp_term[ep_tid] = sp.x * sp.z + sp.y * args.in_channels;\n";
    }
    c += "      } else {\n";
    c += "        tg_s_scale[ep_tid] = 0.0f;\n";
    c += "        tg_s_zp[ep_tid] = 0.0f;\n";
    if (params_.dequant_has_zero_point) {
      c += "        tg_s_zp_term[ep_tid] = 0.0f;\n";
    }
    c += "      }\n";
    c += "    }\n";
    c += "    if (ep_tid < N_TILE_SLICES) {\n";
    c += "      int w_sl = slice_tile_id * N_TILE_SLICES + ep_tid;\n";
    c += "      if (w_sl < args.dst.Slices()) {\n";
    c += "        float4 ws = args.weights_scale.Read<float>(w_sl);\n";
    c += "        float4 wm = ucl::Convert<float4>(args.weights_sum.Read(w_sl));\n";
    c += "        tg_w_scale4[ep_tid] = ws;\n";
    c += "        tg_w_ssum4[ep_tid] = ws * wm;\n";
    if (params_.dequant_has_zero_point) {
      c += "        tg_w_zp4[ep_tid] = args.weights_zero_point.Read<float>(w_sl);\n";
    }
    c += "      } else {\n";
    c += "        tg_w_scale4[ep_tid] = float4(0.0f);\n";
    c += "        tg_w_ssum4[ep_tid] = float4(0.0f);\n";
    if (params_.dequant_has_zero_point) {
      c += "        tg_w_zp4[ep_tid] = float4(0.0f);\n";
    }
    c += "      }\n";
    c += "    }\n";
    c += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    c += "    threadgroup OUT_SCALAR tmp_h[N_TILE * STRIP_M];\n";
    c += "    threadgroup OUT_VEC4* tmp_h_x4 = (threadgroup OUT_VEC4*)(tmp_h);\n";
    c += "    int ep_sub_m = ep_tid % STRIP_M;\n";
    c += "    int ep_sub_s = ep_tid / STRIP_M;\n";
    c += "    int ep_spw = WG_SIZE / STRIP_M;\n";
    c += "    #pragma unroll_full\n";
    c += "    for (int m_base = 0; m_base < M_TILE; m_base += STRIP_M) {\n";
    c += "      if (m_base > 0) threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    c += "      #pragma unroll_full\n";
    c += "      for (uint16_t i = 0; i < c_tile.get_capacity(); ++i) {\n";
    c += "        if (c_tile.is_valid_element(i)) {\n";
    c += "          auto coord = c_tile.get_multidimensional_index(i);\n";
    c += "          int lm = coord[1] - m_base;\n";
    c += "          if (lm >= 0 && lm < STRIP_M) {\n";
    c += "            int n = coord[0];\n";
    c += "            int m = coord[1];\n";
    c += "            float res = float(c_tile[i]) * (tg_w_scale[n] * tg_s_scale[m]) + tg_s_zp[m] * tg_w_ssum[n];\n";
    if (params_.dequant_has_zero_point) {
      c += "            res += tg_s_zp_term[m] * tg_w_zp[n];\n";
    }
    c += "            tmp_h[lm * N_TILE + n] = ucl::Convert<OUT_SCALAR>(res);\n";
    c += "          }\n";
    c += "        }\n";
    c += "      }\n";
    c += "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    c += "      int ep_row = spatial_tile_id * M_TILE + m_base + ep_sub_m;\n";
    c += EmitRowCoords(params_, dst_has_batch, "ep_row", "ep", "      ");
    c += "      if (ep_ok) {\n";
    c += "        for (int i = 0; i < N_TILE_SLICES / ep_spw; ++i) {\n";
    c += "          int tile_n_id = i * ep_spw + ep_sub_s;\n";
    c += "          int slice_id = slice_tile_id * N_TILE_SLICES + tile_n_id;\n";
    c += "          if (slice_id < args.dst.Slices()) {\n";
    c += "            OUT_VEC4 res_value = tmp_h_x4[ep_sub_m * N_TILE_SLICES + tile_n_id];\n";
    std::string ep_coords = "ep_w, ep_h, slice_id";
    if (dst_has_batch) {
      ep_coords += ", ep_b";
    }
    c += "            args.dst.Write(res_value, " + ep_coords + ");\n";
    c += "          }\n";
    c += "        }\n";
    c += "      }\n";
    c += "    }\n";
    c += "  }\n";
  } else if (manual_src_reading && !params_.has_dequant &&
             (m_tile_ % strip_m == 0) && (m_tile_ > strip_m) &&
             (n_tile_ % 16 == 0)) {
    // PV attention (a_loc 4 KiB + tmp 8 KiB = 12 KiB -> 2 TG/core instead of
    // 20 KiB -> 1 TG/core cap).
    c += "  {\n";
    c += "    threadgroup OUT_SCALAR tmp_h[N_TILE * STRIP_M];\n";
    c += "    threadgroup OUT_VEC4* tmp_h_x4 = (threadgroup OUT_VEC4*)(tmp_h);\n";
    c += "    int ep_tid = ucl::GetLocalId<0>();\n";
    c += "    int ep_sub_m = ep_tid % STRIP_M;\n";
    c += "    int ep_sub_s = ep_tid / STRIP_M;\n";
    c += "    int ep_spw = WG_SIZE / STRIP_M;\n";
    c += "    #pragma unroll_full\n";
    c += "    for (int m_base = 0; m_base < M_TILE; m_base += STRIP_M) {\n";
    c += "      if (m_base > 0) threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    c += "      #pragma unroll_full\n";
    c += "      for (uint16_t i = 0; i < c_tile.get_capacity(); ++i) {\n";
    c += "        if (c_tile.is_valid_element(i)) {\n";
    c += "          auto coord = c_tile.get_multidimensional_index(i);\n";
    c += "          int lm = coord[1] - m_base;\n";
    c += "          if (lm >= 0 && lm < STRIP_M) {\n";
    c += "            tmp_h[lm * N_TILE + coord[0]] = c_tile[i];\n";
    c += "          }\n";
    c += "        }\n";
    c += "      }\n";
    c += "      threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    c += "      int ep_row = spatial_tile_id * M_TILE + m_base + ep_sub_m;\n";
    c += EmitRowCoords(params_, dst_has_batch, "ep_row", "ep", "      ");
    c += "      if (ep_ok) {\n";
    c += "        for (int i = 0; i < N_TILE_SLICES / ep_spw; ++i) {\n";
    c += "          int tile_n_id = i * ep_spw + ep_sub_s;\n";
    c += "          int slice_id = slice_tile_id * N_TILE_SLICES + tile_n_id;\n";
    c += "          if (slice_id < args.dst.Slices()) {\n";
    c += "            OUT_VEC4 res_value = tmp_h_x4[ep_sub_m * N_TILE_SLICES + tile_n_id];\n";
    if (params_.has_bias) {
      c += "            res_value += args.biases.Read(slice_id);\n";
    }
    std::string ep_coords = "ep_w, ep_h, slice_id";
    if (dst_has_batch) {
      ep_coords += ", ep_b";
    }
    c += "            args.dst.Write(res_value, " + ep_coords + ");\n";
    c += "          }\n";
    c += "        }\n";
    c += "      }\n";
    c += "    }\n";
    c += "  }\n";
  } else {
    c += R"(
  threadgroup ACCUM_SCALAR tmp[N_TILE * M_TILE];
  auto c_local = tensor(tmp, dextents<int, 2>(N_TILE, M_TILE));
  c_tile.store(c_local);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup ACCUM_VEC4* tmp_x4 = (threadgroup ACCUM_VEC4*)(tmp);
)";
    const std::string oob_check = params_.batched_weights
                                      ? "dst_w >= args.dst.Width()"
                                      : "dst_h >= args.dst.Height()";
    c += "  if (" + oob_check + ") return;\n";
    if (params_.runtime_check.packed_groups.has_value()) {
      c += "  if (dst_w >= w_group_offset + w_group_size) return;\n";
    }
    if (params_.has_dequant) {
      std::string p_coords = "dst_w, dst_h, 0";
      if (params_.dequant_params_has_batch) {
        p_coords += ", dst_b";
      }
      c += "  float4 src_params = args.params_tensor.Read<float>(" + p_coords +
           ");\n";
      c += "  float src_scale = src_params.x;\n";
      c += "  float src_zero_point = src_params.y;\n";
      if (params_.dequant_has_zero_point) {
        c += "  float src_sum = src_params.z;\n";
      }
    }
    c += R"(
  for (int i = 0; i < N_TILE_SLICES / slices_per_wg; ++i) {
    int tile_n_id = i * slices_per_wg + sub_slice_id;
    int slice_id = slice_tile_id * N_TILE_SLICES + tile_n_id;
    if (slice_id < args.dst.Slices()) {
)";
    if (params_.has_dequant) {
      c += "      float4 resf32 = ucl::Convert<float4>(tmp_x4[sub_spatial_id * "
           "N_TILE_SLICES + tile_n_id]);\n";
      c += "      float4 weights_scale = "
           "args.weights_scale.Read<float>(slice_id);\n";
      c += "      resf32 *= weights_scale * src_scale;\n";
      c += "      float4 weights_sum = "
           "ucl::Convert<float4>(args.weights_sum.Read(slice_id));\n";
      c += "      resf32 += weights_scale * src_zero_point * weights_sum;\n";
      if (params_.dequant_has_zero_point) {
        c += "      float4 weights_zero_point = "
             "args.weights_zero_point.Read<float>(slice_id);\n";
        c += "      resf32 += src_scale * weights_zero_point * src_sum;\n";
        c += "      resf32 += src_zero_point * weights_zero_point * "
             "args.in_channels;\n";
      }
      c += "      OUT_VEC4 res_value = ucl::Convert<OUT_VEC4>(resf32);\n";
    } else {
      c += "      OUT_VEC4 res_value = tmp_x4[sub_spatial_id * N_TILE_SLICES + "
           "tile_n_id];\n";
      if (params_.has_bias) {
        c += "      res_value += args.biases.Read(slice_id);\n";
      }
    }
    std::string coords = "dst_w, dst_h, slice_id";
    if (dst_has_batch) {
      coords += ", dst_b";
    }
    c += "      args.dst.Write(res_value, " + coords + ");\n";
    c += "    }\n";
    c += "  }\n";
  }
  c += "}\n";
  const std::string type =
      params_.weights_data_type == DataType::kFloat16 ? "half" : "int8_t";
  std::string spatial_size = "args.dst.Width()";
  if (!params_.batched_weights) {
    spatial_size += " * args.dst.Height()";
  }
  if (has_batch) {
    spatial_size += " * args.dst.Batch()";
  }
  if (params_.runtime_check.packed_groups.has_value()) {
    spatial_size = "w_group_size";
  }
  const std::string mat_mul_mode =
      manual_k_tiling
          ? "mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate"
          : "mpp::tensor_ops::matmul2d_descriptor::mode::multiply";
  absl::StrReplaceAll(
      {
          {"M_TILE", std::to_string(m_tile_)},
          {"STRIP_M", std::to_string(strip_m)},
          {"N_TILE_SLICES", std::to_string(n_tile_ / 4)},
          {"N_TILE", std::to_string(n_tile_)},
          {"K_TILE_SLICES", std::to_string(k_tile / 4)},
          {"K_TILE", std::to_string(k_tile)},
          {"SIMDGROUPS", std::to_string(simdgroups_)},
          {"WG_SIZE", std::to_string(32 * simdgroups_)},
          {"ACCUM_SCALAR", ToUclDataType(accum_type, 1)},
          {"ACCUM_VEC4", ToUclDataType(accum_type, 4)},
          {"OUT_SCALAR", ToUclDataType(out_type, 1)},
          {"OUT_VEC4", ToUclDataType(out_type, 4)},
          {"Type", ToUclDataType(DataType::kFloat16, 4)},
          {"SType", type},
          {"SPATIAL_SIZE", spatial_size},
          {"MAT_MUL_MODE", mat_mul_mode},
      },
      &c);
  return c;
}

ConvAppleMPP::ConvAppleMPP(const TensorDescriptor& src,
                           const ConvParams& params, int m_tile, int n_tile)
    : params_(params), m_tile_(m_tile), n_tile_(n_tile), simdgroups_(4) {
  src.CopyWithoutData(&src_desc_);
  work_group_size_ = int3(32 * simdgroups_, 1, 1);
  if (params_.weights_shape.o % 128 != 0) {
    n_tile_ = 64;
  }
}

int3 ConvAppleMPP::GetGridSize() const {
  const int groups_x = DivideRoundUp(dst_[0]->Channels(), n_tile_);
  int spatial_size = dst_[0]->Width() * dst_[0]->Batch();
  int groups_z = 1;
  if (params_.batched_weights) {
    groups_z = dst_[0]->Height();
  } else {
    spatial_size *= dst_[0]->Height();
  }
  if (params_.runtime_check.packed_groups.has_value()) {
    spatial_size = params_.runtime_check.packed_groups->max_group_size;
    groups_z = params_.runtime_check.packed_groups->num_groups;
  }
  const int groups_y = DivideRoundUp(spatial_size, m_tile_);
  return int3(groups_x * work_group_size_.x, groups_y, groups_z);
}

bool SupportsConvAppleMPP(const GpuInfo& gpu_info) {
  return gpu_info.IsApiMetal() && gpu_info.IsApple() &&
         gpu_info.apple_info.gpu_family >= AppleInfo::Family::kApple10 &&
         gpu_info.metal_info.IsMslVersionEqualOrHigher(4, 0);
}

bool SupportsConvAppleMPP(const GpuInfo& gpu_info,
                          const ExternalWeights& weights) {
  if (!SupportsConvAppleMPP(gpu_info)) {
    return false;
  }
  const bool supported_type = weights.desc.type == DataType::kFloat32 ||
                              weights.desc.type == DataType::kFloat16 ||
                              weights.desc.type == DataType::kUint8 ||
                              weights.desc.type == DataType::kUint4 ||
                              weights.desc.type == DataType::kUint2;
  const int dst_slices = DivideRoundUp(weights.shape.o, 4);
  if (weights.desc.layout != WeightsLayout::kOSpatialIOGroupI4O4 ||
      weights.desc.output_group_size != dst_slices || !supported_type) {
    return false;
  }
  return true;
}

ConvAppleMPP CreateConvAppleMPP(const TensorDescriptor& src,
                                const TensorDescriptor& dst,
                                const Convolution2DAttributes& attr) {
  const OHWI& weights_shape =
      std::visit([](const auto& w) { return w.shape; }, attr.weights);

  ConvAppleMPP::ConvParams params;
  params.weights_desc.layout = WeightsLayout::kUnknown;
  params.weights_data_type = DataType::kFloat16;
  params.weights_shape = weights_shape;
  params.has_bias = !attr.bias.data.empty();
  params.InitKernelXY(attr);

  ConvAppleMPP conv(src, params);
  conv.AddSrcTensor("src", src);
  conv.AddDstTensor("dst", dst);
  if (!params.x_kernel_is_1) {
    AddXKernelParams(attr, &conv);
  }
  if (!params.y_kernel_is_1) {
    AddYKernelParams(attr, &conv);
  }

  conv.UploadWeights(GetFloatWeights(attr));
  if (!attr.bias.data.empty()) {
    conv.UploadBias(attr.bias);
  }
  return conv;
}

ConvAppleMPP CreateConvAppleMPP(
    const TensorDescriptor& src, const TensorDescriptor& dst,
    const Tensor<OHWI, DataType::kFloat32>& weights,
    const Tensor<Linear, DataType::kFloat32>& bias) {
  ConvAppleMPP::ConvParams params;
  params.weights_desc.layout = WeightsLayout::kUnknown;
  params.weights_data_type = DataType::kFloat16;
  params.weights_shape = weights.shape;
  params.has_bias = !bias.data.empty();

  ConvAppleMPP conv(src, params);
  conv.AddSrcTensor("src", src);
  conv.AddDstTensor("dst", dst);

  conv.UploadWeights(weights);
  if (!bias.data.empty()) {
    conv.UploadBias(bias);
  }
  return conv;
}

ConvAppleMPP CreateConvAppleMPPExternalWeights(
    const TensorDescriptor& src, const TensorDescriptor& dst,
    const OHWI& weights_shape, const TensorDescriptor* bias,
    const TensorDescriptor* src_exp, bool different_weights_for_height,
    const ConvRuntimeCheckDesc& runtime_check) {
  ConvAppleMPP::ConvParams params;
  params.weights_desc.layout = WeightsLayout::kUnknown;
  params.weights_data_type = DataType::kFloat16;
  params.weights_shape = weights_shape;
  params.has_bias = bias != nullptr;
  params.batched_weights = different_weights_for_height;
  params.runtime_check = runtime_check;
  params.softmax_input_activation = src_exp != nullptr;

  // Agello (PERFORMANCE.md, D2 sweep): AGELLO_PV_TILE="MxN" sets the tile of the products that
  // compute probabilities from a softmax reduction (attention's x V), keeping the 8,192-element
  // destination: each output tile recomputes its rows' exponentials, so wider tiles compute
  // fewer. N stops at the output's width (M grows to match).
  int m_tile = 64;
  int n_tile = 128;
  if (params.softmax_input_activation) {
    if (const char* tile = std::getenv("AGELLO_PV_TILE"); tile != nullptr) {
      int m = 0, n = 0;
      if (std::sscanf(tile, "%dx%d", &m, &n) == 2 && m * n == 8192 && n % 64 == 0 &&
          (m == 16 || m == 32 || m == 64 || m == 128)) {
        const int width = AlignByN(weights_shape.o, 64);
        if (n > width) {
          n = width;
          m = 8192 / n;
        }
        if (m == 16 || m == 32 || m == 64 || m == 128) {
          m_tile = m;
          n_tile = n;
        }
      }
    }
  }
  ConvAppleMPP conv(src, params, m_tile, n_tile);
  conv.AddSrcTensor("src", src);
  conv.AddDstTensor("dst", dst);

  BufferDescriptor weights_desc;
  weights_desc.element_type = DataType::kFloat16;
  weights_desc.element_size = 1;
  conv.AddSrcBuffer("weights", weights_desc);

  if (bias) {
    conv.AddSrcTensor("biases", *bias);
  }

  if (src_exp) {
    conv.AddSrcTensor("src_exp", *src_exp);
  }

  AddRuntimeParams(conv, runtime_check);
  return conv;
}

ConvAppleMPP CreateConvAppleMPPExternalWeights(
    const TensorDescriptor& src, const TensorDescriptor& dst,
    const ExternalWeights& weights, const TensorDescriptor* bias,
    const TensorDescriptor* src_exp, bool different_weights_for_height,
    const ConvRuntimeCheckDesc& runtime_check, const BHWC* dst_shape) {
  ConvAppleMPP::ConvParams params;
  params.weights_desc.layout = WeightsLayout::kUnknown;
  params.weights_data_type = DataType::kFloat16;
  params.weights_shape = weights.shape;
  params.has_bias = bias != nullptr;
  params.batched_weights = different_weights_for_height;
  params.runtime_check = runtime_check;
  params.softmax_input_activation = src_exp != nullptr;

  params.weights_desc = weights.desc;
  params.scale_zp_shape = weights.scale_zp_shape;
  params.has_zero_point = weights.zero_point != nullptr;
  params.src_group_slices =
      DivideRoundUp(weights.shape.i, 4) / weights.scale_zp_shape.i;

  int m_tile = 64;
  if (runtime_check.packed_groups.has_value() && dst_shape) {
    const int average_task_size =
        DivideRoundUp(dst_shape->w, runtime_check.packed_groups->num_groups);
    if (average_task_size <= 32) {
      m_tile = 32;
    }
  }
  ConvAppleMPP conv(src, params, m_tile, /*n_tile=*/64);
  conv.AddSrcTensor("src", src);
  conv.AddDstTensor("dst", dst);

  BufferDescriptor buffer_desc;
  if (SizeInBitsOf(weights.desc.type) >= 16) {
    // float weights
    buffer_desc.element_type = weights.desc.type;
    buffer_desc.element_size = 16;
  } else {
    // quantized weights
    buffer_desc.element_type = DataType::kUint32;
    buffer_desc.element_size = SizeInBitsOf(weights.desc.type) / 2;
  }
  buffer_desc.memory_type = MemoryType::kGlobal;
  conv.AddSrcBuffer("weights", buffer_desc);

  fc::AddWeightsScaleZeroPointArguments(weights, &conv);

  if (bias) {
    conv.AddSrcTensor("biases", *bias);
  }

  if (src_exp) {
    conv.AddSrcTensor("src_exp", *src_exp);
  }

  AddRuntimeParams(conv, runtime_check);
  return conv;
}

ConvAppleMPP CreateConvAppleMPPInt8(
    const TensorDescriptor& src, const TensorDescriptor& dst,
    const Tensor<OHWI, DataType::kInt8>& weights) {
  ConvAppleMPP::ConvParams params;
  params.weights_desc.layout = WeightsLayout::kUnknown;
  params.weights_data_type = DataType::kInt8;
  params.weights_shape = weights.shape;
  params.has_bias = false;

  ConvAppleMPP conv(src, params);
  conv.AddSrcTensor("src", src);
  conv.AddDstTensor("dst", dst);
  conv.UploadWeights(weights);
  return conv;
}

ConvAppleMPP CreateConvAppleMPPInt8(const TensorDescriptor& src,
                                    const TensorDescriptor& dst,
                                    const OHWI& weights_shape,
                                    const ConvAppleMPPDequantParams* dequant) {
  ConvAppleMPP::ConvParams params;
  params.weights_desc.layout = WeightsLayout::kUnknown;
  params.weights_data_type = DataType::kInt8;
  params.weights_shape = weights_shape;
  params.has_bias = false;
  params.batched_weights = weights_shape.h != 1;
  ApplyDequantParams(params, dst, dequant);

  ConvAppleMPP conv(src, params);
  conv.AddSrcTensor("src", src);
  conv.AddDstTensor("dst", dst);

  BufferDescriptor weights_desc;
  weights_desc.element_type = DataType::kInt8;
  weights_desc.element_size = 1;
  conv.AddSrcBuffer("weights", weights_desc);
  AttachDequantArgs(conv, dequant);
  return conv;
}

// Creates an Apple MPP convolution operation with INT8 external weights.
ConvAppleMPP CreateConvAppleMPPInt8(const TensorDescriptor& src,
                                    const TensorDescriptor& dst,
                                    const ExternalWeights& weights,
                                    const ConvAppleMPPDequantParams* dequant) {
  ConvAppleMPP::ConvParams params;
  params.weights_desc.layout = WeightsLayout::kUnknown;
  params.weights_data_type = DataType::kInt8;
  params.weights_shape = weights.shape;
  params.has_bias = false;
  params.batched_weights = weights.shape.h != 1;
  params.weights_desc = weights.desc;
  ApplyDequantParams(params, dst, dequant);

  ConvAppleMPP conv(src, params, /*m_tile=*/64, /*n_tile=*/64);
  conv.AddSrcTensor("src", src);
  conv.AddDstTensor("dst", dst);

  BufferDescriptor weights_desc;
  weights_desc.element_type = DataType::kUint32;
  weights_desc.element_size = SizeInBitsOf(weights.desc.type) / 2;
  weights_desc.memory_type = MemoryType::kGlobal;
  conv.AddSrcBuffer("weights", weights_desc);
  AttachDequantArgs(conv, dequant);
  return conv;
}

}  // namespace ml_drift
