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

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/agello_gemma4_template.h"
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/gemma4_data_processor_config.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/test_utils.h"  // NOLINT
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {
namespace {

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::ResultOf;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::VariantWith;

using Tokenizer = ::litert::support::Tokenizer;
using TokenizerType = ::litert::support::TokenizerType;
using TokenIds = ::litert::support::TokenIds;

class FakeTokenizer : public Tokenizer {
 public:
  TokenizerType GetTokenizerType() const override {
    return TokenizerType::kSentencePiece;
  }
  absl::StatusOr<TokenIds> TextToTokenIds(absl::string_view text) override {
    if (text == "<|tool_response>") return TokenIds{50};
    if (text == "<|channel>thought\n") return TokenIds{100, 10};
    if (text == "<channel|>") return TokenIds{101};
    return TokenIds{42};
  }
  absl::StatusOr<int> TokenToId(absl::string_view) override { return 42; }
  absl::StatusOr<std::string> TokenIdsToText(absl::Span<const int>,
                                             bool) override {
    return "x";
  }
  std::vector<std::string> GetTokens() const override { return {}; }
  int GetVocabSize() const override { return 256; }
};

class MockSession : public Engine::Session {
 public:
  MOCK_METHOD(absl::Status, GenerateContentStream,
              (const std::vector<InputData>&,
               absl::AnyInvocable<void(absl::StatusOr<Responses>)>),
              (override));
  MOCK_METHOD(absl::Status, GenerateContentStream,
              (const std::vector<InputData>&,
               absl::AnyInvocable<void(absl::StatusOr<Responses>)>,
               const DecodeConfig&),
              (override));
  MOCK_METHOD(absl::StatusOr<Responses>, GenerateContent,
              (const std::vector<InputData>&), (override));
  MOCK_METHOD(absl::StatusOr<Responses>, RunTextScoring,
              (const std::vector<absl::string_view>&, bool), (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<TaskController>>,
              RunTextScoringAsync,
              (const std::vector<absl::string_view>&,
               absl::AnyInvocable<void(absl::StatusOr<Responses>)>, bool),
              (override));
  MOCK_METHOD(absl::Status, RunPrefill, (const std::vector<InputData>&),
              (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<TaskController>>, RunPrefillAsync,
              (const std::vector<InputData>&,
               absl::AnyInvocable<void(absl::StatusOr<Responses>)>),
              (override));
  MOCK_METHOD(absl::StatusOr<Responses>, RunDecode, (), (override));
  MOCK_METHOD(absl::StatusOr<Responses>, RunDecode, (const DecodeConfig&),
              (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<TaskController>>, RunDecodeAsync,
              (absl::AnyInvocable<void(absl::StatusOr<Responses>)>),
              (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<TaskController>>, RunDecodeAsync,
              (absl::AnyInvocable<void(absl::StatusOr<Responses>)>,
               const DecodeConfig&),
              (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Engine::Session>>, Clone, (),
              (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Engine::Session>>, CloneAsync,
              (absl::AnyInvocable<void(absl::StatusOr<Responses>)>),
              (override));
  MOCK_METHOD(absl::StatusOr<BenchmarkInfo>, GetBenchmarkInfo, (), (override));
  MOCK_METHOD(absl::StatusOr<BenchmarkInfo*>, GetMutableBenchmarkInfo, (),
              (override));
  MOCK_METHOD(void, CancelProcess, (), (override));
  MOCK_METHOD(absl::Status, WaitUntilDone, (), (override));
  MOCK_METHOD(absl::Status, SaveCheckpoint, (absl::string_view), (override));
  MOCK_METHOD(absl::Status, SaveContext, (absl::string_view), (override));
  MOCK_METHOD(absl::Status, RewindToCheckpoint, (absl::string_view),
              (override));
  MOCK_METHOD(absl::Status, RestoreContext, (absl::string_view), (override));
  MOCK_METHOD(const SessionConfig&, GetSessionConfig, (), (const, override));
};

class MockEngine : public Engine {
 public:
  MOCK_METHOD(const EngineSettings&, GetEngineSettings, (), (const, override));
  MOCK_METHOD(const Tokenizer&, GetTokenizer, (), (const, override));
  MOCK_METHOD(absl::StatusOr<AudioExecutorProperties>,
              GetAudioExecutorProperties, (), (const, override));
  MOCK_METHOD(absl::StatusOr<VisionExecutorProperties>,
              GetVisionExecutorProperties, (), (const, override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Session>>, CreateSession,
              (const SessionConfig&), (override));
  MOCK_METHOD(absl::Status, WaitUntilDone, (absl::Duration), (override));
};

std::string LoadGemma4Template() {
  std::ifstream in(
      "runtime/conversation/agello_testdata/gemma4_e4b_file_template.jinja");
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

TEST(AgelloConversationTest,
     PreservesThoughtsAcrossMultiStepToolsAndStripsOnceOnFinalAnswer) {
  const std::string raw_template = LoadGemma4Template();
  ASSERT_FALSE(raw_template.empty());
  const std::string fixed_template = FixGemma4ToolTemplate(raw_template);

  ASSERT_OK_AND_ASSIGN(
      auto model_assets, ModelAssets::Create("unused.litertlm"));
  ASSERT_OK_AND_ASSIGN(
      auto engine_settings,
      EngineSettings::CreateDefault(model_assets, Backend::CPU));
  proto::LlmMetadata metadata;
  metadata.mutable_llm_model_type()->mutable_gemma4();
  metadata.set_jinja_prompt_template(raw_template);
  auto* stop = metadata.add_stop_tokens();
  stop->mutable_token_ids()->add_ids(106);
  engine_settings.GetMutableLlmMetadata() = metadata;

  SessionConfig session_config = SessionConfig::CreateDefault();
  ASSERT_OK(session_config.MaybeUpdateAndValidate(engine_settings));

  FakeTokenizer tokenizer;
  auto mock_session = std::make_unique<MockSession>();
  auto* session_ptr = mock_session.get();
  ON_CALL(*session_ptr, GetSessionConfig())
      .WillByDefault(ReturnRef(session_config));
  ON_CALL(*session_ptr, WaitUntilDone()).WillByDefault(Return(absl::OkStatus()));

  MockEngine engine;
  ON_CALL(engine, GetEngineSettings()).WillByDefault(ReturnRef(engine_settings));
  ON_CALL(engine, GetTokenizer()).WillByDefault(ReturnRef(tokenizer));
  EXPECT_CALL(engine, CreateSession(testing::_))
      .WillOnce(Return(testing::ByMove(std::move(mock_session))));

  JsonPreface preface;
  preface.tools = nlohmann::ordered_json::parse(
      R"([{"type":"function","function":{"name":"get_time","description":"time"}}])");

  ASSERT_OK_AND_ASSIGN(
      auto config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config)
          .SetPreface(preface)
          .SetEnableConstrainedDecoding(false)
          .SetOverwritePromptTemplate(PromptTemplate(fixed_template))
          .SetPrefillPrefaceOnInit(false)
          .SetFilterChannelContentFromKvCache(true)
          .SetOverwriteProcessorConfig(Gemma4DataProcessorConfig())
          .SetChannels({litert::lm::Channel{
              .channel_name = "thought",
              .start = "<|channel>thought\n",
              .end = "<channel|>",
          }})
          .Build(engine));
  ASSERT_OK_AND_ASSIGN(auto conversation, Conversation::Create(engine, config));

  auto get_text = [](const InputText& text) {
    auto s = text.GetRawTextString();
    return s.ok() ? std::string(*s) : std::string();
  };

  // Step 1: User asks -> model thinks (`plan_step_1`) + tool_call.
  // Checkpoint saved once before step 1; NO rewind after tool_call so thought
  // stays live in KV for step 2.
  {
    testing::InSequence seq;
    EXPECT_CALL(*session_ptr,
                RunPrefillAsync(ElementsAre(VariantWith<InputText>(ResultOf(
                                    get_text, HasSubstr("What time is it?")))),
                                testing::_))
        .WillOnce([](const std::vector<InputData>&,
                     absl::AnyInvocable<void(absl::StatusOr<Responses>)> cb) {
          cb(Responses(TaskState::kDone));
          return nullptr;
        });
    EXPECT_CALL(*session_ptr, SaveCheckpoint("channel_content_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*session_ptr, RunDecodeAsync(testing::_, testing::_))
        .WillOnce([](absl::AnyInvocable<void(absl::StatusOr<Responses>)> cb,
                     const DecodeConfig&) {
          cb(Responses(TaskState::kProcessing,
                       {"<|channel>thought\nplan_step_1<channel|>"
                        "<|tool_call>call:get_time{}<tool_call|>"}));
          cb(Responses(TaskState::kDone));
          return nullptr;
        });
    EXPECT_CALL(*session_ptr, RewindToCheckpoint(testing::_)).Times(0);
  }
  ASSERT_OK_AND_ASSIGN(
      auto msg1, conversation->SendMessage(
                     Message{{"role", "user"}, {"content", "What time is it?"}}));
  ASSERT_TRUE(msg1.contains("tool_calls"));

  // Step 2: Tool response -> model thinks (`plan_step_2`) + final answer.
  // Checkpoint is NOT overwritten; ONE rewind at turn end strips BOTH thoughts
  // and refills tool_call + tool_response + final answer cleanly.
  {
    testing::InSequence seq;
    EXPECT_CALL(*session_ptr,
                RunPrefillAsync(ElementsAre(VariantWith<InputText>(ResultOf(
                                    get_text, HasSubstr("response:get_time")))),
                                testing::_))
        .WillOnce([](const std::vector<InputData>&,
                     absl::AnyInvocable<void(absl::StatusOr<Responses>)> cb) {
          cb(Responses(TaskState::kDone));
          return nullptr;
        });
    EXPECT_CALL(*session_ptr, SaveCheckpoint(testing::_)).Times(0);
    EXPECT_CALL(*session_ptr, RunDecodeAsync(testing::_, testing::_))
        .WillOnce([](absl::AnyInvocable<void(absl::StatusOr<Responses>)> cb,
                     const DecodeConfig& dc) {
          // Verify first-token tool-response stop suppression is active.
          EXPECT_TRUE(dc.GetSuppressTokensConfig().has_value());
          EXPECT_TRUE(dc.GetSuppressTokensConfig()->first_token_only());
          cb(Responses(TaskState::kProcessing,
                       {"<|channel>thought\nplan_step_2<channel|>It is noon."}));
          cb(Responses(TaskState::kDone));
          return nullptr;
        });
    EXPECT_CALL(*session_ptr,
                RewindToCheckpoint("channel_content_checkpoint"))
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(
        *session_ptr,
        RunPrefillAsync(
            ElementsAre(VariantWith<InputText>(ResultOf(
                get_text,
                AllOf(HasSubstr("<|tool_call>call:get_time{}<tool_call|>"),
                      HasSubstr("<|tool_response>response:get_time"),
                      HasSubstr("It is noon.<turn|>\n"),
                      Not(HasSubstr("plan_step_1")),
                      Not(HasSubstr("plan_step_2")))))),
            testing::_))
        .WillOnce([](const std::vector<InputData>&,
                     absl::AnyInvocable<void(absl::StatusOr<Responses>)> cb) {
          cb(Responses(TaskState::kDone));
          return nullptr;
        });
  }
  ASSERT_OK(conversation->SendMessage(nlohmann::json::parse(
      R"({"role":"tool","content":[{"name":"get_time","response":{"time":"12:00"}}]})")));
}

}  // namespace
}  // namespace litert::lm
