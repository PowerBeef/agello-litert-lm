// Copyright 2025 The Google AI Edge Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOOL_USE_PARSER_COMMON_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOOL_USE_PARSER_COMMON_H_

#include <cstddef>
#include <cstdint>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "runtime/components/tool_use/rust/parsers.rs.h"

namespace litert::lm {

// RFC 3629 UTF-8 validation before constructing cxx rust::Str so invalid
// byte-fallback tokens never throw std::invalid_argument / abort (R-01).
inline bool IsValidUtf8(absl::string_view text) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(text.data());
  const size_t n = text.size();
  size_t i = 0;
  while (i < n) {
    const uint8_t c = bytes[i];
    if (c <= 0x7F) {
      i += 1;
    } else if (c >= 0xC2 && c <= 0xDF) {
      if (i + 1 >= n || (bytes[i + 1] & 0xC0) != 0x80) return false;
      i += 2;
    } else if (c == 0xE0) {
      if (i + 2 >= n || bytes[i + 1] < 0xA0 || bytes[i + 1] > 0xBF ||
          (bytes[i + 2] & 0xC0) != 0x80) {
        return false;
      }
      i += 3;
    } else if ((c >= 0xE1 && c <= 0xEC) || (c >= 0xEE && c <= 0xEF)) {
      if (i + 2 >= n || (bytes[i + 1] & 0xC0) != 0x80 ||
          (bytes[i + 2] & 0xC0) != 0x80) {
        return false;
      }
      i += 3;
    } else if (c == 0xED) {
      if (i + 2 >= n || bytes[i + 1] < 0x80 || bytes[i + 1] > 0x9F ||
          (bytes[i + 2] & 0xC0) != 0x80) {
        return false;
      }
      i += 3;
    } else if (c == 0xF0) {
      if (i + 3 >= n || bytes[i + 1] < 0x90 || bytes[i + 1] > 0xBF ||
          (bytes[i + 2] & 0xC0) != 0x80 || (bytes[i + 3] & 0xC0) != 0x80) {
        return false;
      }
      i += 4;
    } else if (c >= 0xF1 && c <= 0xF3) {
      if (i + 3 >= n || (bytes[i + 1] & 0xC0) != 0x80 ||
          (bytes[i + 2] & 0xC0) != 0x80 || (bytes[i + 3] & 0xC0) != 0x80) {
        return false;
      }
      i += 4;
    } else if (c == 0xF4) {
      if (i + 3 >= n || bytes[i + 1] < 0x80 || bytes[i + 1] > 0x8F ||
          (bytes[i + 2] & 0xC0) != 0x80 || (bytes[i + 3] & 0xC0) != 0x80) {
        return false;
      }
      i += 4;
    } else {
      return false;
    }
  }
  return true;
}

nlohmann::ordered_json ConvertJsonValue(const JsonValue& json_value);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOOL_USE_PARSER_COMMON_H_
