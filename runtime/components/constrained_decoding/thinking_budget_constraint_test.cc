// Copyright 2026 The ODML Authors.
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

#include "runtime/components/constrained_decoding/thinking_budget_constraint.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/bitmap.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/constraint_provider.h"
#include "runtime/components/constrained_decoding/fake_constraint.h"
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "runtime/components/constrained_decoding/llg_constraint_provider.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::Return;

using Tokenizer = ::litert::support::Tokenizer;
using TokenizerType = ::litert::support::TokenizerType;
using TokenIds = ::litert::support::TokenIds;

// Returns the tokens allowed by ComputeMask() in `state`.
std::vector<int> AllowedTokens(const ThinkingBudgetConstraint& constraint,
                               const Constraint::State& state) {
  std::vector<int> allowed;
  auto mask = constraint.ComputeMask(state);
  if (!mask.ok()) {
    ADD_FAILURE() << "ComputeMask failed: " << mask.status();
    return allowed;
  }
  if ((*mask)->GetType() != MaskType::kBitmap) {
    ADD_FAILURE() << "Expected a bitmap mask.";
    return allowed;
  }
  const auto& bitmap_mask = static_cast<const BitmapLogitMask&>(**mask);
  for (int token = 0; token < constraint.GetVocabularySize(); ++token) {
    if (bitmap_mask.IsAllowed(token)) {
      allowed.push_back(token);
    }
  }
  return allowed;
}

class MockTokenizer : public Tokenizer {
 public:
  MOCK_METHOD(TokenizerType, GetTokenizerType, (), (const, override));
  MOCK_METHOD(absl::StatusOr<TokenIds>, TextToTokenIds, (absl::string_view),
              (override));
  MOCK_METHOD(absl::StatusOr<int>, TokenToId, (absl::string_view), (override));
  MOCK_METHOD(absl::StatusOr<std::string>, TokenIdsToText,
              (absl::Span<const int>, bool), (override));
  MOCK_METHOD(std::vector<std::string>, GetTokens, (), (const, override));
  MOCK_METHOD(int, GetVocabSize, (), (const, override));
};

TEST(ThinkingBudgetConstraintTest, TestStart) {
  std::vector<int> start_tokens = {10, 11};
  std::vector<int> end_tokens = {12, 13};
  ThinkingBudgetConstraint constraint(nullptr, 5, start_tokens, end_tokens,
                                      100);

  auto state_ptr = constraint.Start();
  ASSERT_NE(state_ptr, nullptr);
  auto* state =
      static_cast<ThinkingBudgetConstraint::ThinkingState*>(state_ptr.get());

  EXPECT_FALSE(state->in_thinking);
  EXPECT_EQ(state->thinking_token_count, 0);
  EXPECT_EQ(state->forced_end_token_index, -1);
  EXPECT_EQ(state->matching_start_index, 0);
}

TEST(ThinkingBudgetConstraintTest, TestThinkingOnlyCountTokens) {
  std::vector<int> start_tokens = {10, 11};
  std::vector<int> end_tokens = {12, 13};
  ThinkingBudgetConstraint constraint(nullptr, 5, start_tokens, end_tokens,
                                      100);

  auto state = constraint.Start();

  // Match start tokens first.
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 10));
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 11));

  // Feed some normal tokens.
  for (int i = 0; i < 3; ++i) {
    ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 50 + i));
    auto* s =
        static_cast<ThinkingBudgetConstraint::ThinkingState*>(state.get());
    EXPECT_TRUE(s->in_thinking);
    EXPECT_EQ(s->thinking_token_count, i + 1);
  }
}

TEST(ThinkingBudgetConstraintTest, TestThinkingSkipStartTokens) {
  std::vector<int> start_tokens = {10, 11};
  std::vector<int> end_tokens = {12, 13};
  ThinkingBudgetConstraint constraint(nullptr, 5, start_tokens, end_tokens,
                                      100);

  auto state = constraint.Start();

  // Feed start tokens: 10, 11
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 10));
  auto* s = static_cast<ThinkingBudgetConstraint::ThinkingState*>(state.get());
  EXPECT_FALSE(s->in_thinking);
  EXPECT_EQ(s->thinking_token_count, 0);  // Start tokens shouldn't count.
  EXPECT_EQ(s->matching_start_index, 1);

  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 11));
  s = static_cast<ThinkingBudgetConstraint::ThinkingState*>(state.get());
  EXPECT_TRUE(s->in_thinking);
  EXPECT_EQ(s->thinking_token_count, 0);
  EXPECT_EQ(s->matching_start_index, -1);  // Finished matching start.

  // Feed a normal token.
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 50));
  s = static_cast<ThinkingBudgetConstraint::ThinkingState*>(state.get());
  EXPECT_TRUE(s->in_thinking);
  EXPECT_EQ(s->thinking_token_count, 1);
}

TEST(ThinkingBudgetConstraintTest, TestThinkingBudgetExceeded) {
  std::vector<int> start_tokens = {10, 11};
  std::vector<int> end_tokens = {12, 13};
  ThinkingBudgetConstraint constraint(nullptr, 3, start_tokens, end_tokens,
                                      100);

  auto state = constraint.Start();

  // Match start tokens first.
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 10));
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 11));

  // Feed 3 normal tokens.
  for (int i = 0; i < 3; ++i) {
    ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 50 + i));
  }
  auto* s = static_cast<ThinkingBudgetConstraint::ThinkingState*>(state.get());
  EXPECT_TRUE(s->in_thinking);
  EXPECT_EQ(s->thinking_token_count, 3);
  EXPECT_EQ(s->forced_end_token_index,
            0);  // Budget exceeded, should start forcing.

  // Verify bitmap only allows first end token (12).
  ASSERT_OK_AND_ASSIGN(auto bitmap, constraint.ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(12));
  EXPECT_FALSE(bitmap->Get(50));

  // Feed the forced end tokens.
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 12));
  s = static_cast<ThinkingBudgetConstraint::ThinkingState*>(state.get());
  EXPECT_TRUE(s->in_thinking);
  EXPECT_EQ(s->forced_end_token_index, 1);

  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 13));
  s = static_cast<ThinkingBudgetConstraint::ThinkingState*>(state.get());
  EXPECT_FALSE(s->in_thinking);  // Should be out of thinking now.
  EXPECT_EQ(s->forced_end_token_index, -1);
}

TEST(ThinkingBudgetConstraintTest, TestNaturalEnd) {
  std::vector<int> start_tokens = {10, 11};
  std::vector<int> end_tokens = {12, 13};
  ThinkingBudgetConstraint constraint(nullptr, 5, start_tokens, end_tokens,
                                      100);

  auto state = constraint.Start();

  // Match start tokens first.
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 10));
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 11));

  // Feed 2 normal tokens.
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 50));
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 51));

  // Feed end tokens naturally: 12, 13
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 12));
  auto* s = static_cast<ThinkingBudgetConstraint::ThinkingState*>(state.get());
  EXPECT_TRUE(s->in_thinking);
  EXPECT_EQ(s->natural_end_match_index, 1);

  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 13));
  s = static_cast<ThinkingBudgetConstraint::ThinkingState*>(state.get());
  EXPECT_FALSE(s->in_thinking);  // Transitions out naturally.
  EXPECT_EQ(s->natural_end_match_index, 0);
}

TEST(ThinkingBudgetConstraintTest, TestWithUserConstraint) {
  std::vector<int> start_tokens = {10, 11};
  std::vector<int> end_tokens = {12, 13};
  // User constraint forces [20, 21, 1] (1 is stop token).
  FakeConstraint user_constraint({20, 21, 1}, 100);

  ThinkingBudgetConstraint constraint(&user_constraint, 3, start_tokens,
                                      end_tokens, 100);

  auto state = constraint.Start();

  // Match start tokens first.
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 10));
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 11));

  // During thinking, user constraint should not restrict anything (except when
  // forcing end, but here we are not).
  ASSERT_OK_AND_ASSIGN(auto bitmap, constraint.ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(20));
  EXPECT_TRUE(bitmap->Get(99));  // Any token allowed by default in thinking.

  // Feed 3 normal tokens to exceed budget.
  for (int i = 0; i < 3; ++i) {
    ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 50 + i));
  }

  // Feed forced end tokens.
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 12));
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 13));

  auto* s = static_cast<ThinkingBudgetConstraint::ThinkingState*>(state.get());
  EXPECT_FALSE(s->in_thinking);
  ASSERT_NE(s->user_state, nullptr);

  // Now user constraint should be active.
  ASSERT_OK_AND_ASSIGN(bitmap, constraint.ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(20));
  EXPECT_FALSE(bitmap->Get(99));  // Only 20 allowed now by FakeConstraint.

  // Feed user tokens.
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 20));
  ASSERT_OK_AND_ASSIGN(bitmap, constraint.ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(21));
  EXPECT_FALSE(bitmap->Get(20));

  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 21));
  EXPECT_FALSE(constraint.IsEnded(*state));

  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 1));
  EXPECT_TRUE(constraint.IsEnded(*state));
}

TEST(ThinkingBudgetConstraintTest, TestSkipThinkingWithUserConstraint) {
  std::vector<int> start_tokens = {10, 11};
  std::vector<int> end_tokens = {12, 13};
  // User constraint forces [20, 21, 1] (1 is stop token).
  FakeConstraint user_constraint({20, 21, 1}, 100);

  ThinkingBudgetConstraint constraint(&user_constraint, 3, start_tokens,
                                      end_tokens, 100);

  auto state = constraint.Start();
  EXPECT_THAT(AllowedTokens(constraint, *state), ElementsAre(10, 20));

  // The model skips thinking and emits the grammar's first token.
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 20));

  auto* s = static_cast<ThinkingBudgetConstraint::ThinkingState*>(state.get());
  EXPECT_FALSE(s->in_thinking);
  EXPECT_EQ(s->matching_start_index, -1);
  ASSERT_NE(s->user_state, nullptr);

  // The wrapped constraint advanced exactly once.
  EXPECT_THAT(AllowedTokens(constraint, *state), ElementsAre(21));
}

TEST(ThinkingBudgetConstraintTest, TestComputeMask) {
  std::vector<int> start_tokens = {10, 11};
  std::vector<int> end_tokens = {12, 13};
  FakeConstraint user_constraint({20, 21, 1}, 100);
  ThinkingBudgetConstraint constraint(&user_constraint, 2, start_tokens,
                                      end_tokens, 100);

  auto state = constraint.Start();

  // Initially in start matching: ComputeMask allows the first start token
  // plus whatever the user constraint allows, and nothing else.
  ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));
  ASSERT_NE(mask, nullptr);
  EXPECT_EQ(mask->GetType(), MaskType::kBitmap);
  auto* bitmap_mask = static_cast<BitmapLogitMask*>(mask.get());
  EXPECT_TRUE(bitmap_mask->IsAllowed(10));
  EXPECT_TRUE(bitmap_mask->IsAllowed(20));
  EXPECT_FALSE(bitmap_mask->IsAllowed(11));
  EXPECT_FALSE(bitmap_mask->IsAllowed(50));
  ASSERT_OK_AND_ASSIGN(auto bitmap, constraint.ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(10));
  EXPECT_TRUE(bitmap->Get(20));
  EXPECT_FALSE(bitmap->Get(11));
  EXPECT_FALSE(bitmap->Get(50));

  // Match start tokens. After a partial match only the next start token is
  // allowed.
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 10));
  ASSERT_OK_AND_ASSIGN(mask, constraint.ComputeMask(*state));
  bitmap_mask = static_cast<BitmapLogitMask*>(mask.get());
  EXPECT_TRUE(bitmap_mask->IsAllowed(11));
  EXPECT_FALSE(bitmap_mask->IsAllowed(20));
  EXPECT_FALSE(bitmap_mask->IsAllowed(50));
  ASSERT_OK_AND_ASSIGN(bitmap, constraint.ComputeBitmap(*state));
  EXPECT_TRUE(bitmap->Get(11));
  EXPECT_FALSE(bitmap->Get(20));
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 11));

  // In thinking, token 0
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 50));
  ASSERT_OK_AND_ASSIGN(mask, constraint.ComputeMask(*state));
  bitmap_mask = static_cast<BitmapLogitMask*>(mask.get());
  EXPECT_TRUE(bitmap_mask->IsAllowed(50));

  // Token 1 (hits budget of 2)
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 51));
  // Now budget exceeded, forced end token 12 should be the only allowed token
  ASSERT_OK_AND_ASSIGN(mask, constraint.ComputeMask(*state));
  bitmap_mask = static_cast<BitmapLogitMask*>(mask.get());
  EXPECT_TRUE(bitmap_mask->IsAllowed(12));
  EXPECT_FALSE(bitmap_mask->IsAllowed(13));
  EXPECT_FALSE(bitmap_mask->IsAllowed(50));

  // Feed forced end tokens
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 12));
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 13));

  // Out of thinking -> user constraint active (FakeConstraint allows token 20)
  ASSERT_OK_AND_ASSIGN(mask, constraint.ComputeMask(*state));
  bitmap_mask = static_cast<BitmapLogitMask*>(mask.get());
  EXPECT_TRUE(bitmap_mask->IsAllowed(20));
  EXPECT_FALSE(bitmap_mask->IsAllowed(21));
  EXPECT_FALSE(bitmap_mask->IsAllowed(12));
}

class BanAllNonBitmapConstraint : public FakeConstraint {
 public:
  explicit BanAllNonBitmapConstraint(int vocabulary_size)
      : FakeConstraint(/*token_ids=*/{}, vocabulary_size) {}
  absl::StatusOr<std::unique_ptr<LogitMask>> ComputeMask(
      const State&) const override {
    auto mask = std::make_unique<CompositeLogitMask>();
    mask->AddMask(BitmapLogitMask::CreateAllDisallowed(GetVocabularySize()));
    return mask;
  }
};

TEST(ThinkingBudgetConstraintTest, TestStepZeroWithNonBitmapUserMask) {
  BanAllNonBitmapConstraint user_constraint(/*vocabulary_size=*/5);
  ThinkingBudgetConstraint constraint(&user_constraint, /*budget=*/5,
                                      /*start_token_ids=*/{2, 3},
                                      /*end_token_ids=*/{4}, /*vocab_size=*/5);

  auto state = constraint.Start();
  ASSERT_OK_AND_ASSIGN(auto mask, constraint.ComputeMask(*state));
  EXPECT_EQ(mask->GetType(), MaskType::kCustom);

  std::vector<float> logits(5, 1.5f);
  ASSERT_OK(mask->Apply(absl::MakeSpan(logits)));
  constexpr float kNegInf = -std::numeric_limits<float>::infinity();
  EXPECT_THAT(logits, ElementsAre(kNegInf, kNegInf, 1.5f, kNegInf, kNegInf));
}

TEST(ThinkingBudgetConstraintTest, TestMismatchAfterPartialStartMatchFails) {
  ThinkingBudgetConstraint constraint(nullptr, 5, {10, 11}, {12, 13}, 100);
  auto state = constraint.Start();
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 10));
  EXPECT_FALSE(constraint.ComputeNext(*state, 50).ok());
}

TEST(ThinkingBudgetConstraintTest, TestSkipThinkingWithLlgConstraint) {
  constexpr int kVocabSize = 7;
  constexpr int kEos = 1;
  constexpr int kA = 2;
  constexpr int kB = 3;
  constexpr int kChannelStart = 4;
  constexpr int kThought = 5;
  constexpr int kChannelEnd = 6;
  MockTokenizer tokenizer;
  EXPECT_CALL(tokenizer, GetTokens())
      .WillOnce(Return(std::vector<std::string>{
          "<pad>", "<eos>", "a", "b", "<|channel>", "thought", "<channel|>"}));
  EXPECT_CALL(tokenizer, TextToTokenIds(::testing::_))
      .WillRepeatedly([](absl::string_view text) {
        if (text == "a") return TokenIds{kA};
        if (text == "b") return TokenIds{kB};
        if (text == "ab") return TokenIds{kA, kB};
        return TokenIds{};
      });
  LlGuidanceConfig config;
  config.eos_id = kEos;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ConstraintProvider> provider,
                       LlgConstraintProvider::Create(tokenizer, config));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Constraint> grammar,
                       provider->CreateConstraint(LlGuidanceConstraintArg{
                           .constraint_type = LlgConstraintType::kLark,
                           .constraint_string = R"(start: "a" "b")"}));
  ThinkingBudgetConstraint constraint(
      grammar.get(), /*budget=*/5,
      /*start_token_ids=*/{kChannelStart, kThought},
      /*end_token_ids=*/{kChannelEnd}, kVocabSize);

  // The model may open the thought channel, or skip thinking with a token the
  // grammar allows.
  auto state = constraint.Start();
  EXPECT_THAT(AllowedTokens(constraint, *state),
              ElementsAre(kA, kChannelStart));

  // LLGuidance requires a mask to be computed on the same state before commit.
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, kA));

  // The grammar saw the token exactly once, and runs to completion.
  EXPECT_THAT(AllowedTokens(constraint, *state), ElementsAre(kB));
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, kB));
  EXPECT_THAT(AllowedTokens(constraint, *state), ElementsAre(kEos));
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, kEos));
  EXPECT_TRUE(constraint.IsEnded(*state));
}

// Agello: whether `state` and every state reachable from it in `steps` of `tokens` allow every
// token, by the masks.
bool MasksAllowEveryToken(const ThinkingBudgetConstraint& constraint,
                          const Constraint::State& state, int steps,
                          const std::vector<int>& tokens, int vocab_size) {
  if (static_cast<int>(AllowedTokens(constraint, state).size()) != vocab_size) {
    return false;
  }
  if (steps == 0) return true;
  for (int token : tokens) {
    auto next = constraint.ComputeNext(state, token);
    if (!next.ok() || !MasksAllowEveryToken(constraint, **next, steps - 1, tokens,
                                            vocab_size)) {
      return false;
    }
  }
  return true;
}

// Agello (the GPU-resident round): AllowsAnyTokens never promises more than the masks give, over
// every state six tokens deep (start, end and plain tokens, so the start sequence, natural ends,
// the budget and the forced end all occur), and it says yes where a round runs: thinking far from
// the budget, and in the content phase.
TEST(ThinkingBudgetConstraintTest, AllowsAnyTokensNeverPromisesMoreThanTheMasks) {
  constexpr int kVocab = 32;
  const std::vector<int> tokens = {0, 10, 11, 20, 21};
  ThinkingBudgetConstraint constraint(/*user_constraint=*/nullptr, /*budget=*/3, {10, 11},
                                      {20, 21}, kVocab);
  std::vector<std::unique_ptr<Constraint::State>> frontier;
  frontier.push_back(constraint.Start());
  int promised = 0;
  for (int depth = 0; depth <= 6; ++depth) {
    std::vector<std::unique_ptr<Constraint::State>> next_frontier;
    for (const auto& state : frontier) {
      for (int steps = 0; steps <= 3; ++steps) {
        if (!constraint.AllowsAnyTokens(*state, steps)) continue;
        ++promised;
        EXPECT_TRUE(MasksAllowEveryToken(constraint, *state, steps, tokens, kVocab))
            << "depth " << depth << ", steps " << steps;
      }
      if (depth == 6) continue;
      // The tokens the mask allows: a forced token's rivals are errors by design.
      const std::vector<int> allowed = AllowedTokens(constraint, *state);
      for (int token : tokens) {
        if (std::find(allowed.begin(), allowed.end(), token) == allowed.end()) continue;
        auto next = constraint.ComputeNext(*state, token);
        ASSERT_TRUE(next.ok()) << next.status();
        next_frontier.push_back(*std::move(next));
      }
    }
    frontier = std::move(next_frontier);
  }
  EXPECT_GT(promised, 0);

  // After the start sequence: zero thinking tokens of three.
  auto state = constraint.Start();
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 10));
  EXPECT_FALSE(constraint.AllowsAnyTokens(*state, 0));
  ASSERT_OK_AND_ASSIGN(state, constraint.ComputeNext(*state, 11));
  EXPECT_TRUE(constraint.AllowsAnyTokens(*state, 2));
  EXPECT_FALSE(constraint.AllowsAnyTokens(*state, 3));
  // Skipped thinking: the content phase allows any token for good.
  auto content = constraint.Start();
  ASSERT_OK_AND_ASSIGN(content, constraint.ComputeNext(*content, 0));
  EXPECT_TRUE(constraint.AllowsAnyTokens(*content, 100));
}

TEST(ThinkingBudgetConstraintTest, AllowsAnyTokensIsFalseWithAUserConstraint) {
  FakeConstraint user({1, 2}, 32);
  ThinkingBudgetConstraint constraint(&user, /*budget=*/100, {}, {20, 21}, 32);
  EXPECT_FALSE(constraint.AllowsAnyTokens(*constraint.Start(), 0));
}

}  // namespace
}  // namespace litert::lm
