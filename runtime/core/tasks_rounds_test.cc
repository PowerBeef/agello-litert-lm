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

// Agello: how a reply decoded in rounds of several tokens (speculative decoding) ends. The
// pruned tasks_test.cc needs a SentencePiece model this fork doesn't keep; this test needs none.

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/no_repeat_ngram_config.h"
#include "runtime/components/constrained_decoding/repetition_penalty_config.h"
#include "runtime/components/constrained_decoding/suppress_tokens_config.h"
#include "runtime/components/stop_token_detector.h"
#include "runtime/core/tasks.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/fake_llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {
namespace {

using ::litert::support::Tokenizer;
using ::litert::support::TokenIds;
using ::litert::support::TokenizerType;

constexpr int kVocabSize = 3000;

// Every id reads as " t<id>".
class WordTokenizer : public Tokenizer {
 public:
  TokenizerType GetTokenizerType() const override { return TokenizerType::kUnspecified; }
  absl::StatusOr<TokenIds> TextToTokenIds(absl::string_view text) override {
    return absl::UnimplementedError("Not needed here.");
  }
  absl::StatusOr<int> TokenToId(absl::string_view token) override {
    return absl::UnimplementedError("Not needed here.");
  }
  absl::StatusOr<std::string> TokenIdsToText(absl::Span<const int> ids,
                                             bool skip_special_tokens) override {
    std::string text;
    for (int id : ids) absl::StrAppend(&text, " t", id);
    return text;
  }
  std::vector<std::string> GetTokens() const override {
    std::vector<std::string> tokens;
    tokens.reserve(kVocabSize);
    for (int id = 0; id < kVocabSize; ++id) tokens.push_back(absl::StrCat(" t", id));
    return tokens;
  }
  int GetVocabSize() const override { return kVocabSize; }
};

// Decode calls return several tokens each, as a speculative round does: `rounds[i]` is the
// i-th call's tokens, and the step advances by their number.
class RoundsFakeLlmExecutor : public FakeLlmExecutor {
 public:
  RoundsFakeLlmExecutor(const std::vector<int>& prompt, std::vector<std::vector<int>> rounds)
      : FakeLlmExecutor(kVocabSize, {prompt}, /*decode_tokens_set=*/{{0}}),
        rounds_(std::move(rounds)) {}

  absl::StatusOr<std::vector<std::vector<int>>> Decode(
      const ExecutorDecodeParams& decode_params) override {
    if (next_round_ >= rounds_.size()) {
      return absl::InvalidArgumentError("No round left to decode.");
    }
    const std::vector<int>& tokens = rounds_[next_round_++];
    ABSL_ASSIGN_OR_RETURN(int step, GetCurrentStep());
    ABSL_RETURN_IF_ERROR(
        FakeLlmExecutor::SetCurrentStep(step + static_cast<int>(tokens.size())));
    return std::vector<std::vector<int>>{tokens};
  }

 private:
  std::vector<std::vector<int>> rounds_;
  size_t next_round_ = 0;
};

struct Ending {
  std::vector<int> reply;
  int step = 0;
};

absl::StatusOr<Ending> Decode(const std::vector<int>& prompt, std::vector<int> stop,
                              std::vector<std::vector<int>> rounds) {
  RoundsFakeLlmExecutor executor(prompt, std::move(rounds));
  WordTokenizer tokenizer;
  std::optional<BenchmarkInfo> benchmark_info;
  ABSL_ASSIGN_OR_RETURN(auto prompt_buffer, Tokenizer::TokenIdsToTensorBuffer(prompt));
  ExecutorInputs inputs(ExecutorTextData(std::move(prompt_buffer)), std::nullopt, std::nullopt);
  ABSL_RETURN_IF_ERROR(
      Tasks::Prefill(executor, inputs, /*wait_for_completion=*/true, benchmark_info).status());
  StopTokenDetector detector(/*batch_size=*/1);
  ABSL_RETURN_IF_ERROR(detector.AddStopTokenSequence(stop));
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback = nullptr;
  ABSL_ASSIGN_OR_RETURN(
      Responses responses,
      Tasks::Decode(executor, tokenizer, detector, /*num_output_candidates=*/1, benchmark_info,
                    /*sampler=*/std::nullopt, RepetitionPenaltyConfig::Default(),
                    NoRepeatNgramConfig::Default(), SuppressTokensConfig::Default(),
                    /*constraint=*/nullptr, /*decoded_ids=*/std::nullopt, callback,
                    /*cancelled=*/nullptr));
  if (responses.GetTaskState() != TaskState::kDone || responses.GetTokenIds().size() != 1) {
    return absl::InternalError("The reply didn't end.");
  }
  ABSL_ASSIGN_OR_RETURN(int step, executor.GetCurrentStep());
  return Ending{responses.GetTokenIds()[0], step};
}

// The step is where single-token steps leave it: the stop sequence's last token pending, every
// token before it kept. Rolling back one more (the code before this test) re-pended the reply's
// last token as an id without its embedding, and a reply "400" read as "40" to the next turn
// (Agello's probe 35, 2026-11-17).
TEST(TasksRoundsTest, TheStopTokenStaysPending) {
  const std::vector<int> prompt = {2, 90, 547, 58, 735, 210, 466, 2294};
  const int start = static_cast<int>(prompt.size());
  struct Case {
    std::string name;
    std::vector<int> stop;
    std::vector<std::vector<int>> rounds;
    std::vector<int> reply;
    // Positions the reply takes, its stop sequence included.
    int positions;
  };
  const std::vector<Case> cases = {
      {"stop mid-round", {2294}, {{224, 24, 8}, {66, 2294, 246, 18}}, {224, 24, 8, 66}, 5},
      {"stop first in its round", {2294}, {{224}, {2294, 24, 8}}, {224}, 2},
      {"stop last in its round", {2294}, {{224, 24}, {8, 2294}}, {224, 24, 8}, 4},
      {"stop sequence across rounds", {2295, 2294}, {{224, 2295}, {2294, 18, 246}}, {224}, 3},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    ASSERT_OK_AND_ASSIGN(Ending ending, Decode(prompt, c.stop, c.rounds));
    EXPECT_EQ(ending.reply, c.reply);
    EXPECT_EQ(ending.step, start + c.positions);
  }
}

// The same reply in rounds and in single-token steps ends on the same step.
TEST(TasksRoundsTest, RoundsEndWhereSingleStepsEnd) {
  const std::vector<int> prompt = {2, 90, 547, 58, 735, 210, 466, 2294};
  ASSERT_OK_AND_ASSIGN(Ending rounds, Decode(prompt, {2294}, {{224, 24, 8}, {66, 2294, 246, 18}}));
  ASSERT_OK_AND_ASSIGN(Ending steps, Decode(prompt, {2294}, {{224}, {24}, {8}, {66}, {2294}}));
  EXPECT_EQ(rounds.reply, steps.reply);
  EXPECT_EQ(rounds.step, steps.step);
}

}  // namespace
}  // namespace litert::lm
