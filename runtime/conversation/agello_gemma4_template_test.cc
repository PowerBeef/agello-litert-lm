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

#include "runtime/conversation/agello_gemma4_template.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/prompt_template.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using json = nlohmann::ordered_json;

// The published E4B file's template (gemma-4-E4B-it-fp16-rope.litertlm, the litert-community
// file's, unchanged by patch_rope.py).
std::string FileTemplate() {
  const auto path = std::filesystem::path(::testing::SrcDir()) /
                    "litert_lm/runtime/conversation/agello_testdata/gemma4_e4b_file_template.jinja";
  std::ifstream file(path);
  return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

json User(absl::string_view text) { return {{"role", "user"}, {"content", std::string(text)}}; }

json ToolCall() {
  return {{"role", "assistant"},
          {"tool_calls",
           json::array({{{"type", "function"},
                         {"function",
                          {{"name", "get_current_date_time"}, {"arguments", json::object()}}}}})}};
}

json ToolResponse() {
  return {{"role", "tool"},
          {"content", json::array({{{"type", "tool_response"},
                                    {"name", "get_current_date_time"},
                                    {"response", {{"human", "Tuesday, November 17, 2026"}}}}})}};
}

json Answer(absl::string_view text) {
  return {{"role", "assistant"}, {"content", std::string(text)}};
}

std::string Render(const std::string& jinja, json messages, bool generation_prompt) {
  PromptTemplate prompt_template(jinja);
  PromptTemplateInput input;
  input.messages = std::move(messages);
  input.add_generation_prompt = generation_prompt;
  auto rendered = prompt_template.Apply(input);
  EXPECT_TRUE(rendered.ok()) << rendered.status();
  return rendered.ok() ? *rendered : "";
}

int Count(absl::string_view text, absl::string_view piece) {
  int count = 0;
  for (size_t at = text.find(piece); at != absl::string_view::npos;
       at = text.find(piece, at + piece.size())) {
    ++count;
  }
  return count;
}

TEST(AgelloGemma4TemplateTest, TheFileTemplateIsFixed) {
  const std::string file = FileTemplate();
  ASSERT_GT(file.size(), 10000u);
  EXPECT_NE(FixGemma4ToolTemplate(file), file);
}

TEST(AgelloGemma4TemplateTest, AnotherTemplateIsLeftAlone) {
  const std::string other = "{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}";
  EXPECT_EQ(FixGemma4ToolTemplate(other), other);
}

// What decoding leaves in the cache is the render: the tool call ends with the stop token
// `<|tool_response>`, and the tool response continues from it.
TEST(AgelloGemma4TemplateTest, TheToolResponseContinuesTheToolCall) {
  const std::string fixed = FixGemma4ToolTemplate(FileTemplate());
  const std::string before_response =
      Render(fixed, json::array({User("What year is it?"), ToolCall()}), false);
  const std::string with_response =
      Render(fixed, json::array({User("What year is it?"), ToolCall(), ToolResponse()}), true);
  EXPECT_TRUE(absl::EndsWith(before_response, "<tool_call|><|tool_response>")) << before_response;
  ASSERT_TRUE(absl::StartsWith(with_response, before_response)) << with_response;
  EXPECT_EQ(with_response.substr(before_response.size()),
            "response:get_current_date_time{human:<|\"|>Tuesday, November 17, 2026<|\"|>}"
            "<tool_response|>");
  EXPECT_EQ(Count(with_response, "<|tool_response>"), 1);

  // The file's template closed the turn and rendered the token again.
  const std::string old_render =
      Render(FileTemplate(), json::array({User("What year is it?"), ToolCall(), ToolResponse()}),
             true);
  EXPECT_TRUE(absl::StrContains(old_render, "<tool_call|><turn|>\n<|tool_response>")) << old_render;
}

// The answer, generated right after `<tool_response|>`, is rendered there in later turns.
TEST(AgelloGemma4TemplateTest, TheAnswerFollowsTheToolResponse) {
  const std::string fixed = FixGemma4ToolTemplate(FileTemplate());
  const json chat = json::array(
      {User("What year is it?"), ToolCall(), ToolResponse(), Answer("It is 2026."),
       User("And next year?")});
  const std::string rendered = Render(fixed, chat, true);
  EXPECT_TRUE(absl::StrContains(rendered, "<tool_response|>It is 2026.<turn|>\n<|turn>user\n"))
      << rendered;
  EXPECT_TRUE(absl::EndsWith(rendered, "And next year?<turn|>\n<|turn>model\n")) << rendered;
}

// Chats without tools render as before, byte for byte.
TEST(AgelloGemma4TemplateTest, ChatsWithoutToolsDontChange) {
  const json chat = json::array(
      {{{"role", "system"}, {"content", "You are Agello."}}, User("Hi"), Answer("Hello!"),
       User("How are you?")});
  EXPECT_EQ(Render(FixGemma4ToolTemplate(FileTemplate()), chat, true),
            Render(FileTemplate(), chat, true));
}

// Null values in tool arguments and responses render as `null`, never `None` or empty.
TEST(AgelloGemma4TemplateTest, NullToolArgumentRendersAsNull) {
  const std::string fixed = FixGemma4ToolTemplate(FileTemplate());
  const json call = {
      {"role", "assistant"},
      {"tool_calls",
       json::array({{{"type", "function"},
                     {"function",
                      {{"name", "set_value"},
                       {"arguments", {{"key", "test"}, {"value", nullptr}}}}}}})}};
  const std::string rendered = Render(fixed, json::array({User("Test"), call}), false);
  EXPECT_TRUE(absl::StrContains(rendered, "value:null")) << rendered;
  EXPECT_FALSE(absl::StrContains(rendered, "value:None")) << rendered;
}

// An empty messages list renders without indexing `messages[0]`.
TEST(AgelloGemma4TemplateTest, EmptyMessagesRendersCleanly) {
  const std::string fixed = FixGemma4ToolTemplate(FileTemplate());
  EXPECT_EQ(Render(fixed, json::array(), false), "");
  EXPECT_EQ(Render(fixed, json::array(), true), "<|turn>model\n");
  const std::string with_missing_content =
      Render(fixed, json::array({{{"role", "user"}}}), false);
  EXPECT_TRUE(absl::StrContains(with_missing_content, "<|turn>user\n")) << with_missing_content;
}

}  // namespace
}  // namespace litert::lm
