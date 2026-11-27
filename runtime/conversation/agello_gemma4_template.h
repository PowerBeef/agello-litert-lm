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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_AGELLO_GEMMA4_TEMPLATE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_AGELLO_GEMMA4_TEMPLATE_H_

#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {

// Agello: the tool-call boundary of the Gemma 4 E2B/E4B files' chat template, as upstream's
// template has it since f240215c (2026-11-13). Decoding stops a tool call on `<|tool_response>`
// and keeps it pending, so the file's template, which renders it again at the start of the tool
// response, made the model read it twice, after a `<turn|>` that closed its own turn: then it
// often ended the answer at once (Agello's probe 36, an empty reply after the date tool).
//
// Returns `jinja` with the Gemma 4 tool-call and schema fixes:
// 1. model messages with tool calls end with `<|tool_response>` when last or followed by a tool
//    message, and close with `<turn|>\n` when followed by a non-tool message;
// 2. assistant `content` preamble renders before `tool_calls` so `<tool_call|><|tool_response>`
//    stays contiguous;
// 3. the tool response that follows a tool call leaves out its first `<|tool_response>`, and
//    `role == 'tool'` preserves string/text/mapping content and top-level `message['name']`;
// 4. the answer after a tool response gets no `<|turn>model\n` header, as it is generated;
// 5. no generation prompt follows a tool call;
// 6. `format_argument` renders null/None values as `null` instead of Python's `None`;
// 7. `messages[0]['role']` checks guard against an empty `messages` list;
// 8. `format_parameters` restricts `standard_keys` filtering to fallback non-`properties` schemas,
//    fixes comma placement on `OBJECT`/`ARRAY` schemas lacking `description`/`nullable`, and
//    supports JSON Schema union types (`type: ["string", "null"]`);
// 9. `format_function_declaration` closes `parameters` and `response` braces unconditionally and
//    handles missing `parameters['type']`, non-`OBJECT`/union `response['type']`, and flat tools.
// Returns `jinja` unchanged when any of the pieces it edits isn't there (another template).
std::string FixGemma4ToolTemplate(absl::string_view jinja);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_AGELLO_GEMMA4_TEMPLATE_H_
