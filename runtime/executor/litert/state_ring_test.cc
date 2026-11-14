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
//
// Agello (ring rewind): the window comparison attributes each byte to its slot, in the layout
// FilledRuns documents: element ((y * slices + c / 4) * A + a) * 4 + c % 4 of [1, heads, A, C],
// the slot being a for K caches and c for V caches.

#include <cstdint>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "runtime/executor/litert/state.h"

namespace litert::lm {
namespace {

TEST(LitertStateRingTest, DifferentWindowBytesFindsTheSlot) {
  constexpr int kSlots = 8;
  auto ring = [](bool is_k) {
    LitertState::RingBuffer buffer;
    buffer.name = is_k ? "kv_cache_k_0" : "kv_cache_v_0";
    buffer.is_k = is_k;
    buffer.width = 1;
    // K: [1, 2 heads, 8 slots, 4 dims]; V: [1, 2 heads, 4 dims, 8 slots].
    buffer.a_dim = is_k ? kSlots : 4;
    buffer.slices = is_k ? 1 : kSlots / 4;
    buffer.bytes.assign(2 * kSlots * 4, 0);
    return buffer;
  };
  LitertState::RingCopy a, b;
  a.buffers = {ring(/*is_k=*/true), ring(/*is_k=*/false)};
  b = a;
  // K, head 1, slot 5, dim 2: ((1 * 1 + 0) * 8 + 5) * 4 + 2.
  b.buffers[0].bytes[54] = 1;
  // V, head 0, dim 3, slot 6 (group 1, lane 2): ((0 * 2 + 1) * 4 + 3) * 4 + 2.
  b.buffers[1].bytes[30] = 1;

  std::vector<std::pair<int, int64_t>> per_slot;
  EXPECT_EQ(LitertState::DifferentWindowBytes(a, b, {5, 6}, kSlots, &per_slot), 2);
  // K's counts come first, then V's (offset by the slot count).
  EXPECT_THAT(per_slot, ::testing::ElementsAre(std::make_pair(5, int64_t{1}),
                                               std::make_pair(kSlots + 6, int64_t{1})));
  EXPECT_EQ(LitertState::DifferentWindowBytes(a, b, {5}, kSlots), 1);
  EXPECT_EQ(LitertState::DifferentWindowBytes(a, b, {6}, kSlots), 1);
  EXPECT_EQ(LitertState::DifferentWindowBytes(a, b, {0, 1, 2, 3, 4, 7}, kSlots), 0);
}

}  // namespace
}  // namespace litert::lm
