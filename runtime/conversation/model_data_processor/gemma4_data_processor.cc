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

#include "runtime/conversation/model_data_processor/gemma4_data_processor.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/multimodal_processor_helper.h"
#if !defined(LITERT_LM_FST_CONSTRAINTS_DISABLED)
#include "runtime/components/constrained_decoding/gemma_model_constraint_provider.h"
#endif
#include "runtime/components/tool_use/parser_utils.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/data_utils.h"
#include "runtime/conversation/model_data_processor/gemma4_data_processor_config.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"
#include "support/preprocessor/audio_preprocessor.h"
#include "support/preprocessor/image_preprocessor.h"
#include "sentencepiece_model.pb.h"  // from @sentencepiece

namespace litert::lm {
namespace {

constexpr absl::string_view kZwj = "\xE2\x80\x8D";

// Inserts U+200D inside Gemma 4 control-token boundaries (`<|`, `|>`) and
// legacy multimodal tags so untrusted user/tool content cannot forge turn,
// tool, or media boundaries. Deterministic and idempotent.
std::string NeutralizeGemma4ControlTokens(absl::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (c == '<') {
      const absl::string_view tail = text.substr(i);
      if ((i + 1 < text.size() && text[i + 1] == '|') ||
          absl::StartsWith(tail, "<start_of_image>") ||
          absl::StartsWith(tail, "<end_of_image>") ||
          absl::StartsWith(tail, "<start_of_audio>") ||
          absl::StartsWith(tail, "<end_of_audio>")) {
        out.push_back('<');
        out.append(kZwj.data(), kZwj.size());
        continue;
      }
    } else if (c == '|' && i + 1 < text.size() && text[i + 1] == '>') {
      out.push_back('|');
      out.append(kZwj.data(), kZwj.size());
      continue;
    }
    out.push_back(c);
  }
  return out;
}

// Neutralizes multimodal placeholder and delimiter tokens while keeping
// `<|channel>thought\n...<channel|>` intact for assistant/system turns.
// Deterministic and idempotent.
std::string NeutralizeGemma4MediaTokens(absl::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (size_t i = 0; i < text.size(); ++i) {
    const absl::string_view tail = text.substr(i);
    if (absl::StartsWith(tail, "<|image|>")) {
      out.push_back('<');
      out.append(kZwj.data(), kZwj.size());
      out.append("|image|");
      out.append(kZwj.data(), kZwj.size());
      out.push_back('>');
      i += absl::string_view("<|image|>").size() - 1;
      continue;
    }
    if (absl::StartsWith(tail, "<|audio|>")) {
      out.push_back('<');
      out.append(kZwj.data(), kZwj.size());
      out.append("|audio|");
      out.append(kZwj.data(), kZwj.size());
      out.push_back('>');
      i += absl::string_view("<|audio|>").size() - 1;
      continue;
    }
    if (absl::StartsWith(tail, "<|image>") ||
        absl::StartsWith(tail, "<|audio>") ||
        absl::StartsWith(tail, "<start_of_image>") ||
        absl::StartsWith(tail, "<end_of_image>") ||
        absl::StartsWith(tail, "<start_of_audio>") ||
        absl::StartsWith(tail, "<end_of_audio>")) {
      out.push_back('<');
      out.append(kZwj.data(), kZwj.size());
      continue;
    }
    if (absl::StartsWith(tail, "<image|>")) {
      out.append("<image|");
      out.append(kZwj.data(), kZwj.size());
      out.push_back('>');
      i += absl::string_view("<image|>").size() - 1;
      continue;
    }
    if (absl::StartsWith(tail, "<audio|>")) {
      out.append("<audio|");
      out.append(kZwj.data(), kZwj.size());
      out.push_back('>');
      i += absl::string_view("<audio|>").size() - 1;
      continue;
    }
    out.push_back(text[i]);
  }
  return out;
}

void NeutralizeJsonStrings(nlohmann::ordered_json& value,
                           bool all_control_tokens) {
  if (value.is_string()) {
    const std::string& raw = value.get_ref<const std::string&>();
    std::string sanitized = all_control_tokens
                                ? NeutralizeGemma4ControlTokens(raw)
                                : NeutralizeGemma4MediaTokens(raw);
    if (sanitized != raw) {
      value = std::move(sanitized);
    }
  } else if (value.is_array()) {
    for (auto& element : value) {
      NeutralizeJsonStrings(element, all_control_tokens);
    }
  } else if (value.is_object()) {
    bool key_changed = false;
    for (auto& [key, entry] : value.items()) {
      NeutralizeJsonStrings(entry, all_control_tokens);
      std::string sanitized_key = all_control_tokens
                                      ? NeutralizeGemma4ControlTokens(key)
                                      : NeutralizeGemma4MediaTokens(key);
      if (sanitized_key != key) {
        key_changed = true;
      }
    }
    if (key_changed) {
      nlohmann::ordered_json rebuilt = nlohmann::ordered_json::object();
      for (auto& [key, entry] : value.items()) {
        std::string sanitized_key = all_control_tokens
                                        ? NeutralizeGemma4ControlTokens(key)
                                        : NeutralizeGemma4MediaTokens(key);
        rebuilt[sanitized_key] = std::move(entry);
      }
      value = std::move(rebuilt);
    }
  }
}

}  // namespace

absl::StatusOr<std::unique_ptr<Gemma4DataProcessor>>
Gemma4DataProcessor::Create(Gemma4DataProcessorConfig config,
                            std::optional<Preface> preface,
                            const Tokenizer* tokenizer,
                            const std::vector<std::vector<int>>& stop_token_ids,
                            bool enable_constrained_decoding) {
  // For Gemma4 models without mel-spectrogram extraction, it use 640 PCM
  // samples as one audio token.
  ABSL_ASSIGN_OR_RETURN(
      auto audio_preprocessor,
      AudioPreprocessorMiniAudio::Create(
          litert::support::AudioPreprocessorConfig::CreateDefaultGemma4Config(
              config.skip_mel_spectrogram_extraction)));
#if defined(LITERT_LM_FST_CONSTRAINTS_DISABLED)
  // Agello: without the FST provider only tool-call constraints are missing
  // (CreateConstraint returns Unimplemented, which conversations skip);
  // response formats still work through the llguidance provider.
  return absl::WrapUnique(
      new Gemma4DataProcessor(config, preface, ImagePreprocessor::Create(),
                              std::move(audio_preprocessor)));
#else
  std::unique_ptr<LiteRtLmGemmaModelConstraintProvider,
                  decltype(&LiteRtLmGemmaModelConstraintProvider_Destroy)>
      constraint_provider(nullptr,
                          &LiteRtLmGemmaModelConstraintProvider_Destroy);
  if (enable_constrained_decoding) {
    if (tokenizer->GetTokenizerType() != TokenizerType::kSentencePiece) {
      ABSL_LOG(WARNING)
          << "Constrained decoding is only supported for SentencePiece "
             "tokenizer.";
    } else {
      std::vector<const int*> stop_token_ids_ptrs;
      std::vector<size_t> stop_token_lengths;
      stop_token_ids_ptrs.reserve(stop_token_ids.size());
      stop_token_lengths.reserve(stop_token_ids.size());
      for (const auto& stop_tokens : stop_token_ids) {
        stop_token_ids_ptrs.push_back(stop_tokens.data());
        stop_token_lengths.push_back(stop_tokens.size());
      }
      auto sp_tokenizer =
          reinterpret_cast<const SentencePieceTokenizer*>(tokenizer);
      auto serialized_model_proto =
          sp_tokenizer->GetProcessor().model_proto().SerializeAsString();
      LiteRtLmGemmaModelConstraintProvider* provider =
          LiteRtLmGemmaModelConstraintProvider_Create(
              serialized_model_proto.data(), serialized_model_proto.size(),
              stop_token_ids_ptrs.data(), stop_token_lengths.data(),
              stop_token_ids.size());
      if (provider == nullptr) {
        return absl::InternalError(
            "Failed to create GemmaModelConstraintProvider.");
      }
      constraint_provider.reset(provider);
    }
  }
  return absl::WrapUnique(new Gemma4DataProcessor(
      std::move(constraint_provider), config, preface,
      ImagePreprocessor::Create(), std::move(audio_preprocessor)));
#endif
}

absl::StatusOr<nlohmann::ordered_json>
Gemma4DataProcessor::MessageToTemplateInput(
    const nlohmann::ordered_json& message) const {
  nlohmann::ordered_json result = NormalizeMessageContent(message);
  if (!result.is_object()) {
    return result;
  }

  std::string role;
  if (result.contains("role") && result["role"].is_string()) {
    role = result["role"].get<std::string>();
  }
  const bool is_trusted_role = (role == "assistant" || role == "model" ||
                                role == "system" || role == "developer");
  const bool all_control_tokens = !is_trusted_role;

  if (role == "tool" && result.contains("name") && result["name"].is_string()) {
    NeutralizeJsonStrings(result["name"], /*all_control_tokens=*/true);
  }

  if (result.contains("content")) {
    auto& content = result["content"];
    if (content.is_string()) {
      NeutralizeJsonStrings(content, all_control_tokens);
    } else if (content.is_array()) {
      for (auto& item : content) {
        if (item.is_string()) {
          NeutralizeJsonStrings(item, all_control_tokens);
          continue;
        }
        if (!item.is_object()) {
          continue;
        }
        std::string item_type;
        if (item.contains("type") && item["type"].is_string()) {
          item_type = item["type"].get<std::string>();
        }
        if (item_type == "image" || item_type == "audio") {
          continue;
        }
        if (item_type == "text") {
          if (item.contains("text") && item["text"].is_string()) {
            NeutralizeJsonStrings(item["text"], all_control_tokens);
          }
        } else if (role == "tool" || item_type == "tool_response") {
          if (item.contains("name") && item["name"].is_string()) {
            NeutralizeJsonStrings(item["name"], /*all_control_tokens=*/true);
          }
          if (item.contains("response")) {
            NeutralizeJsonStrings(item["response"],
                                  /*all_control_tokens=*/true);
          }
          if (item.contains("tool_response")) {
            NeutralizeJsonStrings(item["tool_response"],
                                  /*all_control_tokens=*/true);
          }
          if (item.contains("text") && item["text"].is_string()) {
            NeutralizeJsonStrings(item["text"], /*all_control_tokens=*/true);
          }
        }
      }
    }
  }

  if (result.contains("tool_calls") && result["tool_calls"].is_array()) {
    for (auto& tool_call : result["tool_calls"]) {
      if (tool_call.is_object() && tool_call.contains("function") &&
          tool_call["function"].is_object()) {
        auto& fn = tool_call["function"];
        if (fn.contains("name") && fn["name"].is_string()) {
          NeutralizeJsonStrings(fn["name"], /*all_control_tokens=*/false);
        }
        if (fn.contains("arguments")) {
          NeutralizeJsonStrings(fn["arguments"], /*all_control_tokens=*/false);
        }
      }
    }
  }

  return result;
}

absl::StatusOr<std::vector<InputData>>
Gemma4DataProcessor::ToInputDataVectorImpl(
    const std::string& rendered_template_prompt,
    const nlohmann::ordered_json& messages,
    const Gemma4DataProcessorArguments& args) const {
  MultimodalPromptProcessingConfig multi_config{
      .delimiter_regex = R"regex((<\|image\|>|<\|audio\|>))regex",
      .image_token_regex = R"regex((<\|image\|>))regex",
      .audio_token_regex = R"regex((<\|audio\|>))regex",
      .boi_token = config_.boi_token,
      .eoi_token = config_.eoi_token,
      .image_prefix = "",
      .image_suffix = "",
      .add_image_end = true,
      .boa_token = config_.boa_token,
      .eoa_token = config_.eoa_token,
      .audio_prefix = "",
      .audio_suffix = "",
      .add_audio_end = true,
  };
  ImagePreprocessParameter image_preprocess_parameter;
  image_preprocess_parameter.SetPatchifyConfig(
      ImagePreprocessParameter::PatchifyConfig{
          .patch_width = config_.patch_width,
          .patch_height = config_.patch_height,
          .max_num_patches = config_.max_num_patches,
          .pooling_kernel_size = config_.pooling_kernel_size,
          .merge_patches = config_.merge_patches,
      });
  return ProcessMultimodalPrompt(
      rendered_template_prompt, messages, image_preprocessor_.get(),
      audio_preprocessor_.get(), multi_config, image_preprocess_parameter,
      args.visual_token_budget);
}

absl::StatusOr<Message> Gemma4DataProcessor::ToMessageImpl(
    const Responses& responses,
    const Gemma4DataProcessorArguments& args) const {
  return ResponseTextToMessage(
      responses.GetTexts()[0], preface_, config_.code_fence_start,
      config_.code_fence_end, GetSyntaxType(config_.syntax_type),
      {.escape_fence_strings = config_.escape_fence_strings,
       .tool_code_regex = config_.tool_code_regex,
       .return_error_on_parse_failure = ReturnErrorOnParseFailure()});
}

absl::StatusOr<std::unique_ptr<Constraint>>
Gemma4DataProcessor::CreateConstraint(
    const nlohmann::ordered_json& tools) const {
#if defined(LITERT_LM_FST_CONSTRAINTS_DISABLED)
  return absl::UnimplementedError(
      "Tool-call constraints need the FST constraint provider, which this "
      "build doesn't include.");
#else
  if (constraint_provider_c_ == nullptr) {
    return nullptr;
  }
  if (!tools.is_array()) {
    return absl::InvalidArgumentError("Tools must be an array.");
  }
  nlohmann::ordered_json functions = nlohmann::ordered_json::array();
  for (const auto& tool : tools) {
    if (tool.contains("function")) {
      functions.push_back(tool["function"]);
    } else {
      functions.push_back(tool);
    }
  }

  LiteRtLmGemmaModelConstraintOptions gemma_options = {
      .funcall_format = kLiteRtLmGemmaFuncallFormatFcStyle,
      .code_fence_start = config_.code_fence_start.c_str(),
      .code_fence_end = config_.code_fence_end.c_str(),
      .open_quote = config_.open_quote.c_str(),
      .close_quote = config_.close_quote.c_str(),
      .function_response_start = config_.function_response_start.c_str()};
  switch (config_.constraint_mode) {
    case Gemma4DataProcessorConfig::ConstraintMode::kFunctionCallOnly:
      gemma_options.constraint_mode =
          kLiteRtLmGemmaConstraintModeFunctionCallOnly;
      break;
    case Gemma4DataProcessorConfig::ConstraintMode::kTextAndOr:
    default:
      gemma_options.constraint_mode = kLiteRtLmGemmaConstraintModeTextAndOr;
      break;
  }
  std::string functions_str = functions.dump(
      /*indent=*/-1, /*indent_char=*/' ', /*ensure_ascii=*/false,
      nlohmann::json::error_handler_t::replace);
  LiteRtLmConstraint* constraint =
      LiteRtLmGemmaModelConstraintProvider_CreateConstraintFromTools(
          constraint_provider_c_.get(), functions_str.c_str(), &gemma_options);
  if (constraint == nullptr) {
    return absl::InternalError("Failed to create constraint with tools.");
  }
  return absl::WrapUnique(reinterpret_cast<Constraint*>(constraint));
#endif
}

absl::string_view Gemma4DataProcessor::CodeFenceStart() const {
  return config_.code_fence_start;
}

absl::string_view Gemma4DataProcessor::CodeFenceEnd() const {
  return config_.code_fence_end;
}

}  // namespace litert::lm
