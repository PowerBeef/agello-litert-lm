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

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {

std::string FixGemma4ToolTemplate(absl::string_view jinja) {
  // The pieces as the published files have them, whitespace included.
  const std::vector<std::pair<absl::string_view, absl::string_view>> edits = {
      // 3. The answer that follows a tool response continues the model's turn.
      {"        {%- if role != 'tool' -%}\n"
       "            {{- '<|turn>' + role + '\\n' }}\n",
       "        {%- if role != 'tool' and not (role == 'model' and "
       "ns.prev_message_type == 'tool_response') -%}\n"
       "            {{- '<|turn>' + role + '\\n' }}\n"},
      // 2. The model message already ended with the first `<|tool_response>`.
      {"                {%- for item in message['content'] -%}\n"
       "                    {{- '<|tool_response>' -}}\n",
       "                {%- set continues_tool_call = ns.prev_message_type == 'tool_call' -%}\n"
       "                {%- for item in message['content'] -%}\n"
       "                    {%- if not (loop.first and continues_tool_call) -%}\n"
       "                        {{- '<|tool_response>' -}}\n"
       "                    {%- endif -%}\n"},
      // 1. A model message with tool calls stays a tool call and ends with `<|tool_response>`.
      {"                {%- set ns.prev_message_type = None -%} \n"
       "            {%- endif -%}\n"
       "        {%- if ns.prev_message_type == None -%}\n",
       "                {%- if not message['tool_calls'] -%}\n"
       "                    {%- set ns.prev_message_type = None -%}\n"
       "                {%- endif -%}\n"
       "            {%- endif -%}\n"
       "        {%- if role != 'tool' and message['tool_calls'] -%}\n"
       "            {{- '<|tool_response>' -}}\n"
       "        {%- endif -%}\n"
       "        {%- if ns.prev_message_type == None -%}\n"},
      // 4. No generation prompt after a tool call.
      {"    {%- if ns.prev_message_type != 'tool_response' -%}\n"
       "        {{- '<|turn>model\\n' -}}\n",
       "    {%- if ns.prev_message_type != 'tool_response' and "
       "ns.prev_message_type != 'tool_call' -%}\n"
       "        {{- '<|turn>model\\n' -}}\n"},
      // 5. Render null tool arguments/responses as `null`, not Python's `None`.
      {"        {{- ']' -}}\n"
       "    {%- else -%}\n"
       "        {{- argument -}}\n"
       "    {%- endif -%}\n"
       "{%- endmacro -%}\n",
       "        {{- ']' -}}\n"
       "    {%- elif argument is none -%}\n"
       "        {{- 'null' -}}\n"
       "    {%- else -%}\n"
       "        {{- argument -}}\n"
       "    {%- endif -%}\n"
       "{%- endmacro -%}\n"},
      // 6. Guard `messages[0]['role']` when rendering with an empty `messages` list.
      {"{%- if (enable_thinking is defined and enable_thinking) or tools or "
       "messages[0]['role'] in ['system', 'developer'] -%}\n",
       "{%- if (enable_thinking is defined and enable_thinking) or tools or "
       "(messages and messages[0]['role'] in ['system', 'developer']) -%}\n"},
      {"    {%- if messages[0]['role'] in ['system', 'developer'] -%}\n",
       "    {%- if messages and messages[0]['role'] in ['system', 'developer'] -%}\n"},
  };
  std::string fixed(jinja);
  for (const auto& [from, to] : edits) {
    // Each piece once, or the template isn't the one these edits were written for.
    const size_t at = fixed.find(from);
    if (at == std::string::npos || fixed.find(from, at + 1) != std::string::npos) {
      return std::string(jinja);
    }
    fixed.replace(at, from.size(), to);
  }
  return fixed;
}

}  // namespace litert::lm
