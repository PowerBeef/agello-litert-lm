// Copyright 2025 The ODML Authors.
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

#include "runtime/components/top_p_cpu_sampler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "agello/agello_telemetry.h"  // from @agello_telemetry
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/components/sampling_cpu_util.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/tensor_buffer_util.h"
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {
namespace {

// Computes exact full-vocabulary token log-probability:
// log P(sampled_id) = logits[sampled_id] - logsumexp(logits / temp).
float ComputeTokenLogProb(const float* logits, int vocab_size, int sampled_id,
                          float temperature) {
  if (logits == nullptr || vocab_size <= 0 || sampled_id < 0 ||
      sampled_id >= vocab_size) {
    return -std::numeric_limits<float>::infinity();
  }
  // When temperature <= 0.0 (greedy decoding), evaluate token
  // log-probabilities against the nominal model logits (temperature = 1.0)
  // rather than dividing by epsilon which causes non-argmax logits to
  // underflow to 0.0 and collapse the score to 0.0.
  const float temp = (temperature <= 0.0f) ? 1.0f : temperature;

  float max_logit = logits[0];
  for (int v = 1; v < vocab_size; ++v) {
    if (logits[v] > max_logit) {
      max_logit = logits[v];
    }
  }

  if (!std::isfinite(max_logit)) {
    return (logits[sampled_id] == max_logit)
               ? 0.0f
               : -std::numeric_limits<float>::infinity();
  }

  double sum_exp = 0.0;
  for (int v = 0; v < vocab_size; ++v) {
    sum_exp += std::exp(static_cast<double>((logits[v] - max_logit) / temp));
  }

  if (!std::isfinite(sum_exp) || sum_exp <= 0.0) {
    return 0.0f;
  }

  float log_prob = static_cast<float>(
      (static_cast<double>(logits[sampled_id] - max_logit) / temp) -
      std::log(sum_exp));
  return std::min(0.0f, log_prob);
}

std::vector<float> ComputeExactLogProbsForBatch(
    absl::Span<const float> logits_data_span,
    absl::Span<const int> flat_sampled_ids,
    const std::vector<std::vector<float>>& sampled_scores, int batch_size,
    int sequence_size, float temperature) {
  std::vector<float> scores(batch_size * sequence_size);
  const int total_tokens = batch_size * sequence_size;
  const int vocab_size =
      (total_tokens > 0 && logits_data_span.size() % total_tokens == 0)
          ? logits_data_span.size() / total_tokens
          : 0;
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < sequence_size; ++j) {
      const int idx = i * sequence_size + j;
      const int sampled_id = flat_sampled_ids[idx];
      if (vocab_size > 0 && sampled_id >= 0 && sampled_id < vocab_size) {
        scores[idx] = ComputeTokenLogProb(
            logits_data_span.data() + static_cast<size_t>(idx) * vocab_size,
            vocab_size, sampled_id, temperature);
      } else {
        scores[idx] = std::log(sampled_scores[i][j]);
      }
    }
  }
  return scores;
}

absl::Status ValidateTensor(const TensorBuffer& tensor, int max_num_dims,
                            int batch_size, const std::string& tensor_name) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor.TensorType());
  auto dims = tensor_type.Layout().Dimensions();
  LITERT_ASSIGN_OR_RETURN(int num_significant_dims, NumSignificantDims(tensor));
  if (num_significant_dims > max_num_dims) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The output ", tensor_name, " tensor must have <=", max_num_dims,
        " significant dimension, but got ", num_significant_dims));
  }
  if (dims[0] != batch_size) {
    return absl::InvalidArgumentError(
        absl::StrCat("The output ", tensor_name,
                     " tensor must have the same batch size as the input "
                     "logits tensor, but got ",
                     dims[0], " vs ", batch_size));
  }
  return absl::OkStatus();
}

// Converts an array of float16 values to float32 values.
void ConvertFp16ToFp32(absl::Span<const uint16_t> fp16_values,
                       std::vector<float>& out) {
  out.resize(fp16_values.size());
  for (int i = 0; i < fp16_values.size(); ++i) {
    tflite::half half_val;
    std::memcpy(&half_val, fp16_values.data() + i, sizeof(uint16_t));
    out[i] = static_cast<float>(half_val);
  }
}

// Agello (P6 / P1.2): the first maximum of a row of FP16 logits (IEEE half bits), compared as
// unsigned order-preserving keys (flipping all bits of negatives and the sign bit of positives).
// On ARM64, scans 32 halves (64 bytes) per iteration via NEON and tracks the first winning 4096-
// element (8 KiB) block so the index lookup stays hot in L1 cache.
// Returns -1 when the maximum is a NaN, which the float path handles instead.
int GreedyHalf(const uint16_t* row, int vocab) {
  if (vocab <= 0) return -1;
  auto key = [](uint16_t h) -> uint16_t {
    return (h & 0x8000u) ? static_cast<uint16_t>(~h) : static_cast<uint16_t>(h | 0x8000u);
  };
#if defined(__ARM_NEON) || defined(__aarch64__)
  constexpr int kBlock = 4096;
  const uint16x8_t v_sign = vdupq_n_u16(0x8000u);
  uint16_t best = 0;
  int best_block_start = 0;
  bool have_best = false;
  for (int block = 0; block < vocab; block += kBlock) {
    const int end = std::min(block + kBlock, vocab);
    uint16_t block_best = 0;
    int i = block;
    if (end - i >= 32) {
      uint16x8_t vmax0 = vdupq_n_u16(0);
      uint16x8_t vmax1 = vdupq_n_u16(0);
      uint16x8_t vmax2 = vdupq_n_u16(0);
      uint16x8_t vmax3 = vdupq_n_u16(0);
      for (; i + 32 <= end; i += 32) {
        const uint16x8_t v0 = vld1q_u16(row + i);
        const uint16x8_t v1 = vld1q_u16(row + i + 8);
        const uint16x8_t v2 = vld1q_u16(row + i + 16);
        const uint16x8_t v3 = vld1q_u16(row + i + 24);
        const uint16x8_t m0 =
            vreinterpretq_u16_s16(vshrq_n_s16(vreinterpretq_s16_u16(v0), 15));
        const uint16x8_t m1 =
            vreinterpretq_u16_s16(vshrq_n_s16(vreinterpretq_s16_u16(v1), 15));
        const uint16x8_t m2 =
            vreinterpretq_u16_s16(vshrq_n_s16(vreinterpretq_s16_u16(v2), 15));
        const uint16x8_t m3 =
            vreinterpretq_u16_s16(vshrq_n_s16(vreinterpretq_s16_u16(v3), 15));
        vmax0 = vmaxq_u16(vmax0, veorq_u16(v0, vorrq_u16(m0, v_sign)));
        vmax1 = vmaxq_u16(vmax1, veorq_u16(v1, vorrq_u16(m1, v_sign)));
        vmax2 = vmaxq_u16(vmax2, veorq_u16(v2, vorrq_u16(m2, v_sign)));
        vmax3 = vmaxq_u16(vmax3, veorq_u16(v3, vorrq_u16(m3, v_sign)));
      }
      block_best = vmaxvq_u16(
          vmaxq_u16(vmaxq_u16(vmax0, vmax1), vmaxq_u16(vmax2, vmax3)));
    }
    for (; i < end; ++i) block_best = std::max(block_best, key(row[i]));
    if (!have_best || block_best > best) {
      best = block_best;
      best_block_start = block;
      have_best = true;
    }
  }
  const uint16_t h = (best & 0x8000u) ? static_cast<uint16_t>(best & 0x7fffu)
                                      : static_cast<uint16_t>(~best);
  if ((h & 0x7c00u) == 0x7c00u && (h & 0x3ffu) != 0) return -1;
  // -0 and +0 compare equal as floats: the float path keeps the first of them.
  if ((h & 0x7fffu) == 0) {
    for (int i = 0; i < vocab; ++i) {
      if ((row[i] & 0x7fffu) == 0) return i;
    }
    return -1;
  }
  const int block_end = std::min(best_block_start + kBlock, vocab);
  const uint16x8_t v_target = vdupq_n_u16(h);
  int index = best_block_start;
  for (; index + 8 <= block_end; index += 8) {
    if (vmaxvq_u16(vceqq_u16(vld1q_u16(row + index), v_target)) != 0) {
      while (row[index] != h) ++index;
      return index;
    }
  }
  while (index < block_end && row[index] != h) ++index;
  return index;
#else
  uint16_t best = 0;
  for (int i = 0; i < vocab; ++i) best = std::max(best, key(row[i]));
  int index = 0;
  while (key(row[index]) != best) ++index;
  const uint16_t h = row[index];
  if ((h & 0x7c00u) == 0x7c00u && (h & 0x3ffu) != 0) return -1;
  // -0 and +0 compare equal as floats: the float path keeps the first of them.
  if ((h & 0x7fffu) == 0) {
    for (int i = 0; i < index; ++i) {
      if ((row[i] & 0x7fffu) == 0) return i;
    }
  }
  return index;
#endif
}

}  // namespace

// Agello (P1.2): zero-copy greedy argmax over ML Drift's DSHWBC4 layout
// ([vocab/4][rows][4]) in shared unified memory. Eliminates both the 2-4 MiB
// tiled-transpose shadow copy in LiteRtLockMetalMemory and logits_f16_ memcpy
// on every verify / verify_8 speculative round.
bool TopPSampler::GreedyHalfDSHWBC4(const uint16_t* src, int rows, int vocab,
                                    int* out_ids) {
  if (src == nullptr || out_ids == nullptr || rows <= 0 || vocab <= 0 ||
      (vocab % 4) != 0) {
    return false;
  }
  const int slices = vocab / 4;
  constexpr int kSliceBlock = 1024;  // 4,096 tokens per row; stays hot in L1.
  auto key = [](uint16_t h) -> uint16_t {
    return (h & 0x8000u) ? static_cast<uint16_t>(~h)
                         : static_cast<uint16_t>(h | 0x8000u);
  };
  std::vector<uint16_t> best(rows, 0);
  std::vector<int> best_block(rows, 0);
  bool have_best = false;

#if defined(__ARM_NEON) || defined(__aarch64__)
  const uint16x8_t v_sign = vdupq_n_u16(0x8000u);
  auto neon_key = [v_sign](uint16x8_t v) -> uint16x8_t {
    const uint16x8_t m =
        vreinterpretq_u16_s16(vshrq_n_s16(vreinterpretq_s16_u16(v), 15));
    return veorq_u16(v, vorrq_u16(m, v_sign));
  };
  if (rows == 4) {
    for (int sb = 0; sb < slices; sb += kSliceBlock) {
      const int sb_end = std::min(sb + kSliceBlock, slices);
      uint16x8_t vm01 = vdupq_n_u16(0);
      uint16x8_t vm23 = vdupq_n_u16(0);
      int s = sb;
      for (; s + 2 <= sb_end; s += 2) {
        const uint16_t* p = src + static_cast<size_t>(s) * 16;
        vm01 = vmaxq_u16(vm01, vmaxq_u16(neon_key(vld1q_u16(p)),
                                         neon_key(vld1q_u16(p + 16))));
        vm23 = vmaxq_u16(vm23, vmaxq_u16(neon_key(vld1q_u16(p + 8)),
                                         neon_key(vld1q_u16(p + 24))));
      }
      for (; s < sb_end; ++s) {
        const uint16_t* p = src + static_cast<size_t>(s) * 16;
        vm01 = vmaxq_u16(vm01, neon_key(vld1q_u16(p)));
        vm23 = vmaxq_u16(vm23, neon_key(vld1q_u16(p + 8)));
      }
      const uint16_t blk[4] = {
          vmaxv_u16(vget_low_u16(vm01)),
          vmaxv_u16(vget_high_u16(vm01)),
          vmaxv_u16(vget_low_u16(vm23)),
          vmaxv_u16(vget_high_u16(vm23)),
      };
      for (int r = 0; r < 4; ++r) {
        if (!have_best || blk[r] > best[r]) {
          best[r] = blk[r];
          best_block[r] = sb;
        }
      }
      have_best = true;
    }
  } else if (rows == 8) {
    for (int sb = 0; sb < slices; sb += kSliceBlock) {
      const int sb_end = std::min(sb + kSliceBlock, slices);
      uint16x8_t vm01 = vdupq_n_u16(0);
      uint16x8_t vm23 = vdupq_n_u16(0);
      uint16x8_t vm45 = vdupq_n_u16(0);
      uint16x8_t vm67 = vdupq_n_u16(0);
      for (int s = sb; s < sb_end; ++s) {
        const uint16_t* p = src + static_cast<size_t>(s) * 32;
        vm01 = vmaxq_u16(vm01, neon_key(vld1q_u16(p)));
        vm23 = vmaxq_u16(vm23, neon_key(vld1q_u16(p + 8)));
        vm45 = vmaxq_u16(vm45, neon_key(vld1q_u16(p + 16)));
        vm67 = vmaxq_u16(vm67, neon_key(vld1q_u16(p + 24)));
      }
      const uint16_t blk[8] = {
          vmaxv_u16(vget_low_u16(vm01)),  vmaxv_u16(vget_high_u16(vm01)),
          vmaxv_u16(vget_low_u16(vm23)),  vmaxv_u16(vget_high_u16(vm23)),
          vmaxv_u16(vget_low_u16(vm45)),  vmaxv_u16(vget_high_u16(vm45)),
          vmaxv_u16(vget_low_u16(vm67)),  vmaxv_u16(vget_high_u16(vm67)),
      };
      for (int r = 0; r < 8; ++r) {
        if (!have_best || blk[r] > best[r]) {
          best[r] = blk[r];
          best_block[r] = sb;
        }
      }
      have_best = true;
    }
  } else
#endif
  {
    std::vector<uint16_t> blk(rows);
    for (int sb = 0; sb < slices; sb += kSliceBlock) {
      const int sb_end = std::min(sb + kSliceBlock, slices);
      std::fill(blk.begin(), blk.end(), 0);
      for (int s = sb; s < sb_end; ++s) {
        const uint16_t* slice_ptr = src + static_cast<size_t>(s) * rows * 4;
        for (int r = 0; r < rows; ++r) {
          const uint16_t* p = slice_ptr + r * 4;
          uint16_t m = std::max(std::max(key(p[0]), key(p[1])),
                                std::max(key(p[2]), key(p[3])));
          if (m > blk[r]) blk[r] = m;
        }
      }
      for (int r = 0; r < rows; ++r) {
        if (!have_best || blk[r] > best[r]) {
          best[r] = blk[r];
          best_block[r] = sb;
        }
      }
      have_best = true;
    }
  }

  for (int r = 0; r < rows; ++r) {
    const uint16_t h = (best[r] & 0x8000u)
                           ? static_cast<uint16_t>(best[r] & 0x7fffu)
                           : static_cast<uint16_t>(~best[r]);
    if ((h & 0x7c00u) == 0x7c00u && (h & 0x3ffu) != 0) return false;
    if ((h & 0x7fffu) == 0) {
      int found = -1;
      for (int s = 0; s < slices && found < 0; ++s) {
        const uint16_t* p = src + (static_cast<size_t>(s) * rows + r) * 4;
        for (int c = 0; c < 4; ++c) {
          if ((p[c] & 0x7fffu) == 0) {
            found = s * 4 + c;
            break;
          }
        }
      }
      if (found < 0) return false;
      out_ids[r] = found;
      continue;
    }
    const int sb_end = std::min(best_block[r] + kSliceBlock, slices);
    int found = -1;
    for (int s = best_block[r]; s < sb_end; ++s) {
      const uint16_t* p = src + (static_cast<size_t>(s) * rows + r) * 4;
      if (p[0] == h) {
        found = s * 4 + 0;
        break;
      }
      if (p[1] == h) {
        found = s * 4 + 1;
        break;
      }
      if (p[2] == h) {
        found = s * 4 + 2;
        break;
      }
      if (p[3] == h) {
        found = s * 4 + 3;
        break;
      }
    }
    if (found < 0) return false;
    out_ids[r] = found;
  }
  return true;
}

absl::StatusOr<std::unique_ptr<TopPSampler>> TopPSampler::Create(
    int k, float p, float temperature, int batch_size, int sequence_size,
    int seed, bool compute_exact_log_probs) {
  if (k <= 0) {
    return absl::InvalidArgumentError("k must be positive.");
  }
  if (p < 0.0f || p > 1.0f) {
    return absl::InvalidArgumentError("p must be in [0, 1].");
  }
  if (batch_size <= 0) {
    return absl::InvalidArgumentError("batch_size must be positive.");
  }
  if (sequence_size <= 0) {
    return absl::InvalidArgumentError("sequence_size must be positive.");
  }
  if (temperature < 0.0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Temperature must be >= 0, but got ", temperature));
  }
  return absl::WrapUnique(new TopPSampler(
      k, p, temperature, batch_size, sequence_size, seed,
      compute_exact_log_probs));
}

absl::Status TopPSampler::SampleToIdAndScoreBuffer(
    const TensorBuffer& logits_tensor, TensorBuffer& ids_tensor,
    TensorBuffer* scores_tensor) {
  auto status = ValidateTensor(logits_tensor, /*max_num_dims=*/3, batch_size_,
                               "input logits");
  if (!status.ok()) {
    return status;
  }
  status =
      ValidateTensor(ids_tensor, /*max_num_dims=*/2, batch_size_, "output ids");
  if (!status.ok()) {
    return status;
  }

  LITERT_ASSIGN_OR_RETURN(auto logits_tensor_type, logits_tensor.TensorType());
  absl::Span<float> logits_data_span;
  if (logits_tensor_type.ElementType() == ElementType::Float32) {
    auto logits_data_or = ReferTensorBufferAsSpan<float>(logits_tensor);
    if (!logits_data_or) {  // Download the data if it is not in host memory.
      LITERT_ASSIGN_OR_RETURN(logits_data_,
                              CopyFromTensorBuffer<float>(logits_tensor));
      logits_data_span = absl::MakeSpan(logits_data_);
    } else {
      logits_data_span = *logits_data_or;
    }
  } else if (logits_tensor_type.ElementType() == ElementType::Float16) {
    LITERT_ASSIGN_OR_RETURN(auto logits_size, logits_tensor.PackedSize());
    const size_t num_elements = logits_size / sizeof(uint16_t);
    const int rows = batch_size_ * sequence_size_;
    const bool greedy_eligible =
        k_ == 1 && rows > 0 && num_elements % rows == 0 &&
        (scores_tensor == nullptr || !compute_exact_log_probs_);

    // Agello (P1.2): multi-row Metal FP16 logits (verify [1, 4, vocab] and
    // verify_8 [1, 8, vocab]) live in DSHWBC4 layout in shared unified memory.
    // Scan MTLBuffer.contents directly in one sequential NEON pass without
    // locking/transposing into a 2-4 MiB host shadow buffer.
    if (greedy_eligible && rows > 1 && batch_size_ == 1) {
      auto buf_type = logits_tensor.BufferType();
      auto dims = logits_tensor_type.Layout().Dimensions();
      const int vocab = static_cast<int>(num_elements / rows);
      if (buf_type.HasValue() &&
          *buf_type == TensorBufferType::kMetalBufferFp16 &&
          dims.size() == 3 && dims[0] == 1 && dims[1] == rows &&
          dims[2] == vocab && vocab > 0 && (vocab % 4) == 0) {
        if (auto raw = RawMetalStorage(logits_tensor);
            raw.has_value() && raw->contents != nullptr &&
            raw->length >= logits_size) {
          if (logits_tensor.HasEvent()) {
            LITERT_ASSIGN_OR_RETURN(auto event, logits_tensor.GetEvent());
            LITERT_RETURN_IF_ERROR(event.Wait(/*timeout_in_ms=*/-1));
          }
          const auto* dshwbc4_ptr =
              static_cast<const uint16_t*>(raw->contents);
          std::vector<int> ids(rows);
          if (GreedyHalfDSHWBC4(dshwbc4_ptr, rows, vocab, ids.data())) {
            ::agello::telemetry::ProbeLogitsHalfDSHWBC4(dshwbc4_ptr, rows,
                                                        vocab);
            ids_tensor.Write(absl::MakeConstSpan(ids));
            if (scores_tensor != nullptr) {
              status = ValidateTensor(*scores_tensor, /*max_num_dims=*/2,
                                      batch_size_, "output scores");
              if (!status.ok()) {
                return status;
              }
              std::vector<float> scores(rows, 0.0f);
              scores_tensor->Write(absl::MakeConstSpan(scores));
            }
            return absl::OkStatus();
          }
        }
      }
    }

    TensorBuffer& mutable_logits_tensor =
        const_cast<TensorBuffer&>(logits_tensor);
    // Agello (P1.2): lock the FP16 logits tensor for read-only in-place access
    // instead of copying 512 KB-4 MB per step into logits_f16_.
    auto lock_and_addr = TensorBufferScopedLock::Create<const uint16_t>(
        mutable_logits_tensor, TensorBuffer::LockMode::kRead);
    const uint16_t* fp16_ptr = nullptr;
    if (lock_and_addr.HasValue()) {
      fp16_ptr = lock_and_addr->second;
    } else {
      logits_f16_.resize(num_elements);
      LITERT_RETURN_IF_ERROR(
          mutable_logits_tensor.Read(absl::MakeSpan(logits_f16_)));
      fp16_ptr = logits_f16_.data();
    }
    // Agello (P6): greedy sampling (MTP's drafter and verifier, every step)
    // picks the first maximum of the halves in place: no copy, no conversion.
    if (greedy_eligible) {
      const int vocab = static_cast<int>(num_elements / rows);
      std::vector<int> ids(rows);
      bool greedy = vocab > 0;
      for (int r = 0; r < rows && greedy; ++r) {
        ids[r] = GreedyHalf(fp16_ptr + static_cast<size_t>(r) * vocab, vocab);
        greedy = ids[r] >= 0;
      }
      if (greedy) {
        ::agello::telemetry::ProbeLogitsHalf(fp16_ptr, rows, vocab);
        ids_tensor.Write(absl::MakeConstSpan(ids));
        if (scores_tensor != nullptr) {
          status = ValidateTensor(*scores_tensor, /*max_num_dims=*/2,
                                  batch_size_, "output scores");
          if (!status.ok()) {
            return status;
          }
          // A greedy pick's probability is 1, as in TopKTopPSampling.
          std::vector<float> scores(rows, 0.0f);
          scores_tensor->Write(absl::MakeConstSpan(scores));
        }
        return absl::OkStatus();
      }
    }
    ConvertFp16ToFp32(absl::MakeConstSpan(fp16_ptr, num_elements),
                      logits_data_);
    logits_data_span = absl::MakeSpan(logits_data_);
  } else {
    return absl::InvalidArgumentError(
        "Unsupported logits data type for sampler.");
  }

  // Agello telemetry: NaN/Inf, max, top-1 margin and entropy of the logits (sampled calls).
  ::agello::telemetry::ProbeLogits(
      logits_data_span.data(), static_cast<int64_t>(batch_size_) * sequence_size_,
      static_cast<int64_t>(logits_data_span.size()) /
          std::max<int64_t>(1, static_cast<int64_t>(batch_size_) * sequence_size_));

  std::vector<std::vector<float>> sampled_scores;
  auto sampled_ids =
      TopKTopPSampling(logits_data_span, k_, p_, temperature_, generator_,
                       batch_size_, sequence_size_, sampled_scores);
  if (!sampled_ids.ok()) {
    return sampled_ids.status();
  }
  std::vector<int> flat_sampled_ids(batch_size_ * sequence_size_);
  for (int i = 0; i < batch_size_; ++i) {
    for (int j = 0; j < sequence_size_; ++j) {
      flat_sampled_ids[i * sequence_size_ + j] = (*sampled_ids)[i][j];
    }
  }
  ids_tensor.Write(absl::MakeConstSpan(flat_sampled_ids));
  if (scores_tensor != nullptr) {
    status = ValidateTensor(*scores_tensor, /*max_num_dims=*/2, batch_size_,
                            "output scores");
    if (!status.ok()) {
      return status;
    }
    std::vector<float> scores(batch_size_ * sequence_size_);
    if (!compute_exact_log_probs_) {
      for (int i = 0; i < batch_size_; ++i) {
        for (int j = 0; j < sequence_size_; ++j) {
          // The scores are the log of the probability of the sampled token.
          scores[i * sequence_size_ + j] = std::log(sampled_scores[i][j]);
        }
      }
    } else {
      scores = ComputeExactLogProbsForBatch(
          logits_data_span, absl::MakeConstSpan(flat_sampled_ids),
          sampled_scores, batch_size_, sequence_size_, temperature_);
    }
    scores_tensor->Write(absl::MakeConstSpan(scores));
  }
  return absl::OkStatus();
}

absl::Status TopPSampler::UpdateConfig(
    const proto::SamplerParameters& sampler_params, int batch_size,
    std::shared_ptr<std::default_random_engine> rand_gen) {
  k_ = sampler_params.k();
  p_ = sampler_params.p();
  temperature_ = sampler_params.temperature();
  batch_size_ = batch_size;
  compute_exact_log_probs_ = sampler_params.compute_exact_log_probs();
  if (rand_gen != nullptr) {
    generator_ = rand_gen;
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
