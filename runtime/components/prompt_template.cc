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

#include "runtime/components/prompt_template.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/rust/minijinja_template.rs.h"
#include "re2/re2.h"  // from @com_googlesource_code_re2

namespace litert::lm {
namespace {

constexpr absl::string_view kUtf8ReplacementChar = "\xEF\xBF\xBD";

// Replaces any invalid UTF-8 byte sequences with U+FFFD so strings crossed into
// Rust (`cxx::rust::String`) never throw `std::invalid_argument`.
std::string SanitizeUtf8(absl::string_view input) {
  std::string out;
  out.reserve(input.size());
  const size_t n = input.size();
  size_t i = 0;
  while (i < n) {
    const uint8_t b0 = static_cast<uint8_t>(input[i]);
    if (b0 <= 0x7F) {
      out.push_back(input[i]);
      ++i;
      continue;
    }
    if (b0 >= 0xC2 && b0 <= 0xDF && i + 1 < n) {
      const uint8_t b1 = static_cast<uint8_t>(input[i + 1]);
      if (b1 >= 0x80 && b1 <= 0xBF) {
        out.append(input.data() + i, 2);
        i += 2;
        continue;
      }
    } else if (b0 >= 0xE0 && b0 <= 0xEF && i + 2 < n) {
      const uint8_t b1 = static_cast<uint8_t>(input[i + 1]);
      const uint8_t b2 = static_cast<uint8_t>(input[i + 2]);
      const bool b1_valid =
          (b0 == 0xE0) ? (b1 >= 0xA0 && b1 <= 0xBF)
                       : ((b0 == 0xED) ? (b1 >= 0x80 && b1 <= 0x9F)
                                       : (b1 >= 0x80 && b1 <= 0xBF));
      if (b1_valid && b2 >= 0x80 && b2 <= 0xBF) {
        out.append(input.data() + i, 3);
        i += 3;
        continue;
      }
    } else if (b0 >= 0xF0 && b0 <= 0xF4 && i + 3 < n) {
      const uint8_t b1 = static_cast<uint8_t>(input[i + 1]);
      const uint8_t b2 = static_cast<uint8_t>(input[i + 2]);
      const uint8_t b3 = static_cast<uint8_t>(input[i + 3]);
      const bool b1_valid =
          (b0 == 0xF0) ? (b1 >= 0x90 && b1 <= 0xBF)
                       : ((b0 == 0xF4) ? (b1 >= 0x80 && b1 <= 0x8F)
                                       : (b1 >= 0x80 && b1 <= 0xBF));
      if (b1_valid && b2 >= 0x80 && b2 <= 0xBF && b3 >= 0x80 && b3 <= 0xBF) {
        out.append(input.data() + i, 4);
        i += 4;
        continue;
      }
    }
    out.append(kUtf8ReplacementChar.data(), kUtf8ReplacementChar.size());
    ++i;
  }
  return out;
}

// Post-process the template to make it compatible with Minijinja.
//
// Minijinja is a Rust implementation of Jinja2, but it is not 100% compatible
// with the Python implementation. In particular, it does not support calling
// arbitrary Python methods on objects (e.g. `s.startswith("foo")`).
//
// This function uses regexes to rewrite common Python idioms found in
// tokenizer_config.json templates into Minijinja-compatible syntax.
std::string EditTemplateForMinijinja(absl::string_view template_content) {
  std::string modified_template = std::string(template_content);
  RE2::GlobalReplace(&modified_template, R"regex(\.startswith\((.*?)\))regex",
                     R"( is startingwith \1)");
  RE2::GlobalReplace(&modified_template, R"regex(\.endswith\((.*?)\))regex",
                     R"( is endingwith \1)");
  RE2::GlobalReplace(&modified_template,
                     R"regex(\.replace\((.*?),(.*?)\))regex",
                     R"( | replace(\1,\2))");
  RE2::GlobalReplace(&modified_template, R"regex(\.split\((.*?)\)\[0\])regex",
                     R"( | split(\1) | first)");
  RE2::GlobalReplace(&modified_template, R"regex(\.split\((.*?)\)\[-1\])regex",
                     R"( | split(\1) | last)");
  RE2::GlobalReplace(&modified_template, R"regex(\.split\((.*?)\))regex",
                     R"( | split(\1))");
  RE2::GlobalReplace(&modified_template, R"regex(\.join\((.*?)\))regex",
                     R"( | join(\1))");
  RE2::GlobalReplace(&modified_template, R"regex(\.lstrip\(\))regex",
                     " | lstrip");
  RE2::GlobalReplace(&modified_template, R"regex(\.lstrip\((.*?)\))regex",
                     R"( | lstrip(\1))");
  RE2::GlobalReplace(&modified_template, R"regex(\.rstrip\(\))regex",
                     " | rstrip");
  RE2::GlobalReplace(&modified_template, R"regex(\.rstrip\((.*?)\))regex",
                     R"( | rstrip(\1))");
  RE2::GlobalReplace(&modified_template, R"regex(\.strip\(\))regex", " | trim");
  RE2::GlobalReplace(&modified_template, R"regex(\.strip\((.*?)\))regex",
                     R"( | trim(\1))");
  RE2::GlobalReplace(&modified_template, R"regex(\.items\(\))regex",
                     " | items");
  RE2::GlobalReplace(&modified_template, R"regex({% generation %})regex", "");
  RE2::GlobalReplace(&modified_template, R"regex({% endgeneration %})regex",
                     "");
  return modified_template;
}

}  // namespace

using json = nlohmann::ordered_json;

struct PromptTemplate::MinijinjaTemplateImpl {
  rust::Box<MinijinjaTemplate> tmpl;
};

PromptTemplate::PromptTemplate(absl::string_view template_content)
    : minijinja_template_(std::make_unique<MinijinjaTemplateImpl>(
          MinijinjaTemplateImpl{new_minijinja_template(
              EditTemplateForMinijinja(SanitizeUtf8(template_content)))})) {
  const auto caps = minijinja_template_->tmpl->get_capabilities();
  capabilities_ = PromptTemplateCapabilities{
      .supports_single_turn = caps.supports_single_turn};
}

PromptTemplate::~PromptTemplate() = default;

PromptTemplate::PromptTemplate(const PromptTemplate& other)
    : minijinja_template_(
          other.minijinja_template_
              ? std::make_unique<MinijinjaTemplateImpl>(MinijinjaTemplateImpl{
                    other.minijinja_template_->tmpl->clone_template()})
              : nullptr),
      capabilities_(other.capabilities_) {}

PromptTemplate& PromptTemplate::operator=(const PromptTemplate& other) {
  if (this != &other) {
    minijinja_template_ =
        other.minijinja_template_
            ? std::make_unique<MinijinjaTemplateImpl>(MinijinjaTemplateImpl{
                  other.minijinja_template_->tmpl->clone_template()})
            : nullptr;
    capabilities_ = other.capabilities_;
  }
  return *this;
}

PromptTemplate::PromptTemplate(PromptTemplate&& other)
    : minijinja_template_(std::move(other.minijinja_template_)) {
  capabilities_ = other.capabilities_;
}

PromptTemplate& PromptTemplate::operator=(PromptTemplate&& other) {
  if (this != &other) {
    minijinja_template_ = std::move(other.minijinja_template_);
    capabilities_ = other.capabilities_;
  }
  return *this;
}

absl::StatusOr<std::string> PromptTemplate::Apply(
    const PromptTemplateInput& input) const {
  if (minijinja_template_ == nullptr) {
    return absl::FailedPreconditionError("PromptTemplate was moved from.");
  }
  nlohmann::ordered_json minijinja_inputs;
  minijinja_inputs["messages"] = input.messages;
  minijinja_inputs["tools"] = input.tools;
  minijinja_inputs["add_generation_prompt"] = input.add_generation_prompt;
  minijinja_inputs["extra_context"] = input.extra_context;
  minijinja_inputs["now"] = absl::ToUnixSeconds(input.now);
  minijinja_inputs["bos_token"] = input.bos_token;
  minijinja_inputs["eos_token"] = input.eos_token;
  auto result = minijinja_template_->tmpl->apply(minijinja_inputs.dump(
      /*indent=*/-1, /*indent_char=*/' ', /*ensure_ascii=*/false,
      nlohmann::json::error_handler_t::replace));
  if (!result.is_ok) {
    return absl::InternalError(
        absl::StrCat("Failed to apply template: ", std::string(result.error)));
  }
  return std::string(result.content);
}

absl::string_view PromptTemplate::GetTemplateSource() const {
  if (minijinja_template_ == nullptr) {
    return "";
  }
  auto source = minijinja_template_->tmpl->source();
  return absl::string_view(source.data(), source.size());
}

}  // namespace litert::lm
