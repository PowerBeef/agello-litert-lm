// Copyright 2026 Patrice Déry
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_AGELLO_RESIDENT_ROUND_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_AGELLO_RESIDENT_ROUND_H_

#include <cstddef>
#include <memory>

namespace litert::lm {

// Agello (the GPU-resident speculative round, stage 2): the drafter's three steps on the GPU,
// with no host step between them. After each drafter run, the GPU picks the draft (the CPU
// sampler's greedy rule over the FP16 logits), gathers that token's embedding from the INT2
// table the model file maps, and writes the next step's input (embedding, then the drafter's
// projected activations); the host waits once, then reads the drafted ids. The kernels give the
// host path's results bit for bit (tools/agello-gpu-round, stage 1). Pointers are Objective-C
// objects bridged to void* (id<MTLCommandQueue>, id<MTLBuffer>), so this header stays plain C++.
class ResidentRound {
 public:
  // The embedding table: `section` is the mapped tf_lite_embedder section (page-aligned), holding
  // `rows` rows of `cols` 2-bit fields at `rows_offset`; `scales` is the rows' float32 scales
  // (copied); `mul` multiplies every value (sqrt(2560) for E4B).
  struct Table {
    const void* section = nullptr;
    size_t section_bytes = 0;
    size_t rows_offset = 0;
    const float* scales = nullptr;
    int rows = 0;
    int cols = 0;
    float mul = 1.0f;
  };

  // nullptr when Metal isn't there or the setup fails (logged); the caller keeps the host path.
  static std::unique_ptr<ResidentRound> Create(const void* command_queue, const Table& table,
                                               int max_steps);
  virtual ~ResidentRound() = default;

  // Step 0's input token, known to the host. Call before any Encode of a round.
  virtual void SetFirstId(int token_id) = 0;
  // Encodes step `step`'s input: the embedding of draft `step` (step 0's: SetFirstId's) into the
  // first `cols` floats of `activations`, then `cols` floats of `previous` from byte
  // `previous_offset` into the rest.
  virtual bool EncodeInput(int step, void* activations, void* previous, size_t previous_offset) = 0;
  // Encodes the greedy pick over `vocab` FP16 logits (contiguous from offset 0) as draft
  // `step + 1`.
  virtual bool EncodeArgmax(int step, void* logits, int vocab) = 0;
  // Waits for the round's GPU work and writes drafts 1..`count` to `ids`. False on a GPU error
  // or a NaN maximum (the CPU sampler's float fallback decides those), so the caller drafts
  // again on the host.
  virtual bool Finish(int count, int* ids) = 0;
};

// AGELLO_RESIDENT_ROUND=1 turns the round on (off by default until the iPhone measures it).
bool ResidentRoundEnabled();

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_AGELLO_RESIDENT_ROUND_H_
