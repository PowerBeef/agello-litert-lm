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

#include "runtime/components/constrained_decoding/llg_constraint_provider.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "runtime/util/test_utils.h"  // NOLINT
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

using Tokenizer = ::litert::support::Tokenizer;
using TokenizerType = ::litert::support::TokenizerType;
using TokenIds = ::litert::support::TokenIds;

namespace {

using ::testing::Return;

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

class LlgConstraintProviderTest : public ::testing::Test {
 protected:
  MockTokenizer tokenizer_;
};

TEST_F(LlgConstraintProviderTest, CreateSuccess) {
  LlGuidanceConfig config;
  config.eos_id = 1;

  EXPECT_CALL(tokenizer_, GetTokens())
      .WillOnce(Return(std::vector<std::string>{"<pad>", "<eos>", "a", "b"}));

  auto provider = LlgConstraintProvider::Create(tokenizer_, config);
  EXPECT_TRUE(provider.ok());
}

TEST_F(LlgConstraintProviderTest, CreateFailsWithoutEosId) {
  LlGuidanceConfig config;
  auto provider = LlgConstraintProvider::Create(tokenizer_, config);
  EXPECT_FALSE(provider.ok());
  EXPECT_EQ(provider.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(LlgConstraintProviderTest, CreateConstraintProviderSuccess) {
  LlGuidanceConfig config;
  config.eos_id = 1;

  EXPECT_CALL(tokenizer_, GetTokens())
      .WillOnce(Return(std::vector<std::string>{"<pad>", "<eos>", "a", "b"}));

  auto constraint_provider = LlgConstraintProvider::Create(tokenizer_, config);
  ASSERT_TRUE(constraint_provider.ok());
}

TEST_F(LlgConstraintProviderTest, CreateConstraintSuccess) {
  LlGuidanceConfig config;
  config.eos_id = 1;

  EXPECT_CALL(tokenizer_, GetTokens())
      .WillOnce(Return(std::vector<std::string>{"<pad>", "<eos>", "a", "b"}));

  auto constraint_provider = LlgConstraintProvider::Create(tokenizer_, config);
  ASSERT_TRUE(constraint_provider.ok());
  auto provider = std::move(constraint_provider.value());

  auto constraint = provider->CreateConstraint(LlGuidanceConstraintArg{
      .constraint_type = LlgConstraintType::kRegex, .constraint_string = "a+"});
  EXPECT_TRUE(constraint.ok());
}

TEST_F(LlgConstraintProviderTest,
       DecodesSentencePieceSpacesBytesAndSpecialTokensForJsonEnums) {
  LlGuidanceConfig config;
  config.eos_id = 1;

  // Vocab mimicking Gemma 4 SentencePiece pieces: control tokens, byte fallback,
  // `▁`-prefixed words, HTML tag, and pipe-delimited Gemma 4 markers.
  const std::vector<std::string> vocab = {
      "<pad>",             // 0  special
      "<eos>",             // 1  eos (special)
      "{",                 // 2
      "\"",                // 3
      "kind",              // 4
      ":",                 // 5
      "talk",              // 6
      "\xE2\x96\x81now",   // 7  `▁now` -> " now"
      "}",                 // 8
      "<0x20>",            // 9  byte 0x20 (' ')
      "<|turn>",           // 10 special
      "<turn|>",           // 11 special
      "<|channel>",        // 12 special
      "<unused0>",         // 13 special
      "<table>",           // 14 normal HTML text
      "<0xFF>",            // 15 raw byte 0xFF (length 1, not special)
      "now",               // 16 so `<0x20>` + `"now"` is also a valid path
  };
  EXPECT_CALL(tokenizer_, GetTokens()).WillOnce(Return(vocab));
  EXPECT_CALL(tokenizer_, TextToTokenIds(::testing::_))
      .WillRepeatedly([](absl::string_view text) {
        // Minimal greedy SentencePiece-like tokenizer matching `vocab` above.
        TokenIds ids;
        size_t pos = 0;
        while (pos < text.size()) {
          if (text.substr(pos, 4) == "kind") {
            ids.push_back(4);
            pos += 4;
          } else if (text.substr(pos, 4) == "talk") {
            ids.push_back(6);
            pos += 4;
          } else if (text.substr(pos, 4) == " now") {
            ids.push_back(7);
            pos += 4;
          } else if (text.substr(pos, 3) == "now") {
            ids.push_back(16);
            pos += 3;
          } else if (text[pos] == '{') {
            ids.push_back(2);
            ++pos;
          } else if (text[pos] == '"') {
            ids.push_back(3);
            ++pos;
          } else if (text[pos] == ':') {
            ids.push_back(5);
            ++pos;
          } else if (text[pos] == '}') {
            ids.push_back(8);
            ++pos;
          } else if (text[pos] == ' ') {
            ids.push_back(9);
            ++pos;
          } else {
            return TokenIds{};
          }
        }
        return ids;
      });

  ASSERT_OK_AND_ASSIGN(auto provider,
                       LlgConstraintProvider::Create(tokenizer_, config));

  // Multi-word JSON schema enum: previously failed with "Failed to commit token: ▁now".
  ASSERT_OK_AND_ASSIGN(
      auto enum_constraint,
      provider->CreateConstraint(LlGuidanceConstraintArg{
          .constraint_type = LlgConstraintType::kJsonSchema,
          .constraint_string =
              R"({"type":"object","properties":{"kind":{"type":"string","enum":["talk now"]}},"required":["kind"],"additionalProperties":false})"}));

  auto state = enum_constraint->Start();
  const std::vector<int> seq = {2, 3, 4, 3, 5, 3, 6, 7, 3, 8};  // {"kind":"talk now"}
  for (size_t i = 0; i < seq.size(); ++i) {
    ASSERT_OK_AND_ASSIGN(auto bm, enum_constraint->ComputeBitmap(*state));
    EXPECT_TRUE(bm->Get(seq[i])) << "step " << i << " token " << seq[i];
    if (i == 0) {
      EXPECT_FALSE(bm->Get(10));  // `<|turn>` must not be allowed
    }
    if (i == 7) {
      EXPECT_TRUE(bm->Get(7));  // `▁now` decoded to " now"
    }
    ASSERT_OK_AND_ASSIGN(state, enum_constraint->ComputeNext(*state, seq[i]));
  }
  ASSERT_OK_AND_ASSIGN(auto bm_end, enum_constraint->ComputeBitmap(*state));
  EXPECT_TRUE(bm_end->Get(1));  // `<eos>`

  // Byte fallback token `<0x20>` decodes to raw ASCII space (0x20), not literal "<0x20>".
  ASSERT_OK_AND_ASSIGN(
      auto space_constraint,
      provider->CreateConstraint(LlGuidanceConstraintArg{
          .constraint_type = LlgConstraintType::kRegex,
          .constraint_string = " "}));
  auto sp_state = space_constraint->Start();
  ASSERT_OK_AND_ASSIGN(auto sp_bm, space_constraint->ComputeBitmap(*sp_state));
  EXPECT_TRUE(sp_bm->Get(9));   // `<0x20>` -> " "
  EXPECT_FALSE(sp_bm->Get(6));  // `talk` rejected
  ASSERT_OK_AND_ASSIGN(sp_state, space_constraint->ComputeNext(*sp_state, 9));
  ASSERT_OK_AND_ASSIGN(auto sp_end, space_constraint->ComputeBitmap(*sp_state));
  EXPECT_TRUE(sp_end->Get(1));  // `<eos>`

  // Free JSON string allows normal HTML tags (`<table>`) while blocking
  // Gemma 4 control tokens (`<|turn>`, `<turn|>`, `<|channel>`, `<unused0>`).
  ASSERT_OK_AND_ASSIGN(
      auto str_constraint,
      provider->CreateConstraint(LlGuidanceConstraintArg{
          .constraint_type = LlgConstraintType::kJsonSchema,
          .constraint_string =
              R"({"type":"object","properties":{"kind":{"type":"string"}},"required":["kind"],"additionalProperties":false})"}));
  auto s_state = str_constraint->Start();
  for (int tok : {2, 3, 4, 3, 5, 3}) {  // {"kind":"
    ASSERT_OK_AND_ASSIGN(auto bm, str_constraint->ComputeBitmap(*s_state));
    ASSERT_TRUE(bm->Get(tok));
    ASSERT_OK_AND_ASSIGN(s_state, str_constraint->ComputeNext(*s_state, tok));
  }
  ASSERT_OK_AND_ASSIGN(auto s_bm, str_constraint->ComputeBitmap(*s_state));
  EXPECT_TRUE(s_bm->Get(14));   // `<table>` allowed inside JSON string
  EXPECT_FALSE(s_bm->Get(0));   // `<pad>` blocked
  EXPECT_FALSE(s_bm->Get(10));  // `<|turn>` blocked
  EXPECT_FALSE(s_bm->Get(11));  // `<turn|>` blocked
  EXPECT_FALSE(s_bm->Get(12));  // `<|channel>` blocked
  EXPECT_FALSE(s_bm->Get(13));  // `<unused0>` blocked
}

}  // namespace
}  // namespace litert::lm
