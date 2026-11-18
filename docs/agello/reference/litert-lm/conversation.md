# LiteRT-LM: the Conversation layer

> **Scope:** `litert::lm::Conversation` and `ConversationConfig` in `runtime/conversation/`: messages,
> prompt templates (minijinja), model data processors (Gemma 4 in depth), channels and thinking
> (including how reasoning is removed from the KV cache), tool calling, constraints (llguidance,
> external, the closed FST provider), history, rewind and cancellation.
> **Out of scope:** sessions, execution managers and checkpoints below the session API
> ([engine-and-sessions.md](engine-and-sessions.md)); executors, pending tokens and MTP
> ([executors.md](executors.md)); samplers, tokenizers, logit masks and constraint internals
> ([components.md](components.md)); the C and Swift APIs, including the Swift automatic tool loop
> ([c-api.md](c-api.md), [swift-api.md](swift-api.md)); the Gemma 4 chat format as a model property
> ([../models/tokenizer-and-chat-format.md](../models/tokenizer-and-chat-format.md)).
> **Pinned to:** `main` `b7b856a0`: LiteRT-LM at the root (Google's `df6df5b8` plus our commits);
> Agello ships release `agello-one-20261115` (`6da75396`), which predates our changes of 2026-11-17
> to this layer (noted where they apply).
> **Applies to:** all platforms (the layer is portable C++ and Rust); Apple specifics in
> [On Apple](#on-apple). **Last verified:** 2026-11-17

## Summary

`Conversation` turns JSON chat messages into text the model reads and the model's text back into
messages. It owns the chat history, one `Engine::Session` (and so one KV cache), a Jinja prompt
template and a model-specific `ModelDataProcessor`. For Gemma 4 E2B/E4B every turn after the first
prefills only the difference between two template renders, so earlier turns stay in the KV cache and
are never re-read. When a reply contains reasoning (the `thought` channel), our build rewinds the
session to just after the user turn and prefills the reply again without the reasoning, before it
reports the turn as done; the release Agello ships does the same.
JSON response formats from the C and Swift APIs go through the open llguidance provider; the closed
Gemma FST provider, which our binary leaves out, only builds tool-call constraints, which Agello
doesn't use.

## Concepts

### Messages

A `Message` is an `nlohmann::ordered_json` object. The layer reads these fields:

| Field | Meaning |
| --- | --- |
| `role` | `system`, `developer`, `user`, `assistant` (the Gemma 4 template renders it as `model`), `tool` |
| `content` | a string, one part object, or an array of parts; normalized to an array |
| `content[].type` | `text` (`text`), `image` / `audio` (`path` or base64 `blob`), `tool_response` |
| `tool_calls` | on assistant messages: `[{"type": "function", "function": {"name", "arguments"}}]` |
| `channels` | on replies: `{channel_name: text}` extracted from the output (`thought` for Gemma 4) |
| `reasoning_content` | on replies: the text of reasoning channels, joined with newlines |

`Message`, `JsonPreface` and `Channel` are defined in
[`runtime/conversation/io_types.h:26-64`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/io_types.h#L26-L64). `NormalizeMessageContent` wraps string and
object contents into an array of parts, and leaves a message without `content` (an assistant tool
call) as it is ([`runtime/conversation/model_data_processor/data_utils.cc:66-84`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/data_utils.cc#L66-L84)). Media
parts are loaded from `path` (memory-mapped) or `blob` (base64 decoded into memory)
([`runtime/conversation/model_data_processor/data_utils.cc:37-64`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/data_utils.cc#L37-L64)); the template never sees
the blob, because `ApplyTemplate` strips `blob` fields first
([`runtime/conversation/conversation.cc:1196-1200`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L1196-L1200),
[`runtime/conversation/prompt_utils.cc:34-50`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/prompt_utils.cc#L34-L50)).

### Preface and history

The **preface** (`JsonPreface`) is what a conversation starts with: `messages` (system prompt,
earlier turns, few-shot examples), `tools` and `extra_context`. It is fixed at creation. The
**history** (`history_`) holds only what was sent with `SendMessage`/`SendMessageAsync` and the
completed replies ([`runtime/conversation/conversation.cc:617-630`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L617-L630),
[`runtime/conversation/conversation.cc:780-782`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L780-L782)); `GetHistory()` never returns preface
messages. The distinction matters for Agello, which recreates a conversation from saved chat history:
the C API puts the system message and every message given with
`litert_lm_conversation_config_set_messages` into the preface
([`c/conversation.cc:449-479`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/c/conversation.cc#L449-L479)), and the first message sent then prefills all of them (see
[What is prefilled](#what-is-prefilled-for-a-turn)). <!-- c:LM-CNV-022 -->

### Channels

A `Channel` is a named span of model output between a `start` and an `end` string. The Gemma 4
E2B/E4B files declare one: `thought`, from `<|channel>thought\n` to `<channel|>`. A channel is a
reasoning channel when `is_reasoning_channel` is set or its name is `thought`
([`runtime/conversation/channel_util.cc:96-112`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/channel_util.cc#L96-L112)). Channel text is streamed and stored
separately from the answer, and can be removed from the KV cache
([Channels and thinking](#channels-and-thinking)).

### ConversationConfig

Built with `ConversationConfig::Builder` (`CreateDefault(engine)` is `Builder().Build(engine)`).
Defaults at the pin ([`runtime/conversation/conversation.h:268-283`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.h#L268-L283),
[`runtime/conversation/conversation.cc:134-137`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L134-L137)): <!-- c:LM-CNV-001 -->

| Builder setter | Default | Effect |
| --- | --- | --- |
| `SetSessionConfig` | `SessionConfig::CreateDefault()` | sampler, stop tokens, modalities, MTP for this conversation's session |
| `SetPreface` | empty `JsonPreface` | system/initial messages, tools, extra context |
| `SetOverwritePromptTemplate` | none: the file's `jinja_prompt_template` (Gemma 4: with our tool-boundary edits, below) | Jinja template |
| `SetOverwriteProcessorConfig` | none: derived from `llm_model_type` | model data processor config |
| `SetEnableConstrainedDecoding` | `false` | tool-call constraints from the preface's tools |
| `SetConstraintProviderConfig` | none | `LlGuidanceConfig` or `ExternalConstraintConfig` provider for per-message constraints |
| `SetPrefillPrefaceOnInit` | `false` | prefill the preface at creation instead of with the first message |
| `SetChannels` | none: the file's channels | channel definitions |
| `SetFilterChannelContentFromKvCache` | `true` | remove channel text (reasoning) from the KV cache after each reply |
| `SetReturnErrorOnParseFailure` | `true` | an unparsable tool call fails the reply |
| `SetReturnErrorOnMaxTokensReached` | `false` | a full KV cache ends the reply normally instead of with an error |
| `SetThinkingConfig` | none | `ThinkingConfig(enable_thinking, thinking_token_budget)` |
| `SetStreamToolCalls` | `false`, channel `"tool_call"` | also stream tool-call text as it is generated |
| `SetEnableRewinding` | `true` | channel filter uses step rewinds (else saved contexts) |

`CreateInternal` adjusts and validates the inputs:

1. It copies the `SessionConfig`, turns off session-level prompt templating
   (`SetApplyPromptTemplateInSession(false)`) and validates it against the engine
   ([`runtime/conversation/conversation.cc:155-158`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L155-L158)). Templates are the Conversation's job;
   the session only adds the BOS token (see [Rendering](#rendering-with-minijinja)).
2. It picks the template: the overwrite, else the metadata's `jinja_prompt_template`, else a Jinja
   template generated from the older `prompt_templates` metadata, else an error
   ([`runtime/conversation/conversation.cc:160-184`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L160-L184)).
   For a Gemma 4 model the metadata's template first goes through `FixGemma4ToolTemplate`, four
   edits at the tool-call boundary, left out when the template isn't the one they were written for
   (ours: `9a84a518`;
   [`runtime/conversation/agello_gemma4_template.h:24-38`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/agello_gemma4_template.h#L24-L38)).
3. It takes channels from the overwrite or from `LlmMetadata.channels`, rejecting empty names
   ([`runtime/conversation/conversation.cc:186-204`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L186-L204)).
4. It derives the processor config from the session's `llm_model_type`
   ([`runtime/conversation/conversation.cc:206-215`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L206-L215)); `kGemma4` maps to
   `CreateGemma4DataProcessorConfig`
   ([`runtime/conversation/model_data_processor/model_data_processor_factory.cc:659-685`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/model_data_processor_factory.cc#L659-L685)).

### OptionalArgs

Per-message settings ([`runtime/conversation/conversation.h:387-468`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.h#L387-L468)):

| Field | Effect | Through the C API |
| --- | --- | --- |
| `has_pending_message` | prefill without decoding (single-turn templates only) | no |
| `repetition_penalty_config`, `no_repeat_ngram_config`, `suppress_tokens_config` | decode-time constraints ([components.md](components.md)) | yes |
| `decoding_constraint` | a per-message constraint for the conversation's provider | regex and JSON schema only |
| `args` | processor arguments; for Gemma 4 the `visual_token_budget` | yes (budget) |
| `max_output_tokens` | decode limit; thinking and answer tokens both count | yes |
| `task_group_id` | registers task controllers for `CancelGroup` | no |
| `extra_context` | template variables, merged over the preface's | yes (the `extra_context` argument) |
| `thinking_config` | overrides the conversation's `ThinkingConfig` | yes |

`ThinkingConfig()` means thinking on with an unlimited budget (`-1`); `ResolveThinkingConfig` prefers
the message's, then the conversation's ([`runtime/conversation/thinking_config.h:21-36`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/thinking_config.h#L21-L36),
[`runtime/conversation/conversation.cc:93-102`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L93-L102)).

### Prompt templates and processors

- **`PromptTemplate`** wraps a minijinja template compiled through a Rust FFI. Its only capability
  flag, `supports_single_turn`, is true when the source contains `is_appending_to_prefill`
  ([`runtime/components/prompt_template.cc:85-92`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/prompt_template.cc#L85-L92),
  [`runtime/components/rust/minijinja_template.rs:69-73`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/rust/minijinja_template.rs#L69-L73)).
- **`ModelDataProcessor`** converts between messages and the session's `InputData`:
  `ToInputDataVector` (rendered text plus messages to text, image and audio inputs), `ToMessage`
  (output text to a message, with tool-call parsing), `MessageToTemplateInput`, `FormatTools`,
  `CreateConstraint` (tool-call constraints), `CodeFenceStart/End` and `CloneState`
  ([`runtime/conversation/model_data_processor/model_data_processor.h:60-134`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/model_data_processor.h#L60-L134)).

## How it works

### Creating a conversation

```mermaid
sequenceDiagram
  participant C as Caller (Swift actor or C)
  participant CC as ConversationConfig
  participant E as Engine
  participant CV as Conversation
  participant S as Session
  C->>CC: Builder().Set...().Build(engine)
  CC->>CC: copy SessionConfig, templating off, validate
  CC->>CC: template, channels, processor config from LlmMetadata
  C->>CV: Conversation::Create(engine, config)
  CV->>E: CreateSession(session_config)
  E-->>CV: Session (own KV context)
  CV->>CV: CreateModelDataProcessor (registry)
  CV->>CV: CreateConstraintProvider (if configured)
  opt prefill_preface_on_init and preface not empty
    CV->>S: RunPrefill(rendered preface)
  end
```

Each `Conversation` gets its own `Engine::Session`
([`runtime/conversation/conversation.cc:422-441`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L422-L441)); what two sessions on one engine share
and how the engine switches between their KV contexts is in
[engine-and-sessions.md](engine-and-sessions.md). The processor comes from
`ModelDataProcessorRegistry`, keyed by the `DataProcessorConfig` variant; built-in processors register
once, Gemma 4's only when compiled with `ENABLE_GEMMA4_DATA_PROCESSOR`, and an unregistered family
fails with `Unimplemented`
([`runtime/conversation/model_data_processor/model_data_processor.h:267-280`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/model_data_processor.h#L267-L280),
[`runtime/conversation/model_data_processor/model_data_processor_factory.cc:75-106`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/model_data_processor_factory.cc#L75-L106)).
With `prefill_preface_on_init`, the preface is rendered alone and prefilled before `Create` returns
([`runtime/conversation/conversation.cc:442-483`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L442-L483)); neither the C API nor the Swift wrapper
sets it, so for them the preface is prefilled with the first message.

### What is prefilled for a turn

Gemma 4's template is a whole-conversation template: it can only render a full message list.
After the first message, `Conversation` therefore renders twice and prefills the difference.

The template in the E2B/E4B files has no `is_appending_to_prefill`, so `supports_single_turn` is false
and `Conversation` renders each message through `GetSingleTurnTextFromFullHistory`: one full render
for the first message (empty history, `prefill_preface_on_init` false), and two full renders plus a
prefix difference for later messages
([`runtime/conversation/conversation.cc:323-337`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L323-L337);
[template text](../generated/models/e4b-ours.md)). <!-- c:LM-CNV-018 --> That also means
`has_pending_message=true` (prefill several messages, decode once) is rejected with `InvalidArgument`
for these models ([`runtime/conversation/conversation.cc:323-332`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L323-L332)).

**First message** (history empty, `prefill_preface_on_init` false). This path applies only to
templates without single-turn support, such as the one in the E2B/E4B files. The prefill text is one full
render of the preface (system message, initial messages, tools, extra context, plus
`enable_thinking` and the message's `extra_context`) and the new message(s), with
`add_generation_prompt = true` ([`runtime/conversation/conversation.cc:285-295`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L285-L295)).
<!-- c:LM-CNV-020 --> The preface messages are also passed to `ToInputDataVector`, so images in them
are loaded ([`runtime/conversation/conversation.cc:660-671`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L660-L671)).

**Every later message:**

1. `old` = render(preface + history), `add_generation_prompt = false`.
2. `new` = render(preface + history + new message(s)), `add_generation_prompt = true`.
3. If `new` doesn't start with `old`, the send fails with `InternalError` ("The new rendered template
   string does not start with the previous rendered template string").
4. The prefill text is `new` minus `old`.

([`runtime/conversation/conversation.cc:297-320`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L297-L320)) <!-- c:LM-CNV-021 --> Only that
difference reaches the session; the earlier turns are already in the KV cache. Both renders use the
same `enable_thinking` and `extra_context` (the message's `extra_context` is merged over the
preface's; [`runtime/conversation/conversation.cc:260-274`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L260-L274)). Google describes the same
mechanism in `docs/api/cpp/conversation.md` ("History").

For Gemma 4 E2B/E4B and a plain text exchange the pieces are (strings from the file's template):

| Render | Ends with |
| --- | --- |
| `old` after one exchange | `<\|turn>user\nHi<turn\|>\n<\|turn>model\nHello<turn\|>\n` |
| `new` with the next user message | `...<turn\|>\n<\|turn>user\nHow are you?<turn\|>\n<\|turn>model\n` |
| prefill text | `<\|turn>user\nHow are you?<turn\|>\n<\|turn>model\n` |

The difference is computed on strings; whether the KV cache holds exactly `old` depends on how the
previous reply ended (see [Limits](#limits-and-failure-modes)).

So, for Agello: a send on a conversation it keeps prefills one user turn plus `<|turn>model\n`; a
send on a conversation it has just recreated from saved history (a reseed) prefills the whole saved
history with the system turn, because the C API made that history the preface.

### Rendering with minijinja

`PromptTemplate::Apply` serializes the input (messages, tools, `add_generation_prompt`,
`extra_context`, `now`, `bos_token`, `eos_token`) to JSON and calls the Rust renderer
([`runtime/components/prompt_template.cc:129-148`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/prompt_template.cc#L129-L148)). Details that change output:

- Before parsing, Python string idioms common in Hugging Face templates (`.startswith`, `.split`,
  `.strip`, `.items`, ...) are rewritten into minijinja tests and filters with regexes, and
  `{% generation %}` markers are dropped ([`runtime/components/prompt_template.cc:41-75`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/prompt_template.cc#L41-L75)).
- The environment keeps trailing newlines, trims blocks, and adds `strftime_now`, `raise_exception`,
  `tojson` (with `", "` and `": "` separators), `lstrip`, `rstrip` and the `none` test
  ([`runtime/components/rust/minijinja_template.rs:177-192`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/rust/minijinja_template.rs#L177-L192)).
- Every `extra_context` key becomes a top-level template variable, which is how `enable_thinking`
  reaches the template; each render re-creates the environment and re-parses the template source
  ([`runtime/components/rust/minijinja_template.rs:231-255`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/rust/minijinja_template.rs#L231-L255)).
- `bos_token` is an empty string ([`runtime/components/prompt_template.h:94-98`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/prompt_template.h#L94-L98)), so
  `{{ bos_token }}` renders nothing. The session prepends the start token (ID 2 in the Gemma 4 files)
  on the first prefill of a fresh session instead
  ([`runtime/core/session_utils.cc:98-102`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/session_utils.cc#L98-L102), [`runtime/core/session_advanced.cc:120`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/core/session_advanced.cc#L120)).

### The Gemma 4 template in the E2B and E4B files

Agello's E4B file, Google's E4B (`2eee7ac3`) and Google's E2B (`b3ca0d2f`) carry the same
`jinja_prompt_template`, start token 2, stop tokens 1, 50 and 106, and one channel, `thought`, from
`<|channel>thought\n` to `<channel|>`
([e4b-ours](../generated/models/e4b-ours.md), [e4b-google](../generated/models/e4b-google.md),
[e2b-google](../generated/models/e2b-google.md), section 0). <!-- c:LM-CNV-017 --> What the template
does:

- **System turn.** Opened with `<|turn>system\n` when thinking is on, tools are given or the first
  message is `system`/`developer`. With `enable_thinking` true it starts with `<|think|>`; tool
  declarations follow the system text as `<|tool>declaration:NAME{...}<tool|>`.
- **Turns.** Each message renders as `<|turn>ROLE\n...<turn|>\n`, with `assistant` renamed `model`.
  Tool calls render as `<|tool_call>call:NAME{key:value,...}<tool_call|>` with strings quoted by
  `<|"|>`; a `tool` message renders as `<|tool_response>response:NAME{...}<tool_response|>` without a
  turn header.
- **Generation prompt.** `<|turn>model\n`, omitted right after a tool response so the model continues
  its turn. <!-- c:LM-CNV-025 -->
- **Reasoning.** Model text passes through a `strip_thinking` macro that removes every
  `<|channel>...<channel|>` span and trims, so earlier reasoning is never rendered back into the
  prompt. <!-- c:LM-CNV-026 -->

This template is not the one Google publishes on Hugging Face today. `chat_template.jinja` in
`google/gemma-4-E4B-it` at `ee0ef602` is headed "Google Gemma 4 Canonical Chat Template", published
2026-07-09, "Fixed tool-calling loops, turn closures, and thinking content-ordering"
([hf template](../generated/hf/google__gemma-4-E4B-it/chat_template.jinja)). Among the differences:
it writes `<|think|>\n` where the file writes `<|think|>`, renders `reasoning_content` back as a
thought channel for the current turn, and resolves tool responses by `tool_call_id`.
<!-- c:LM-CNV-027 --> The runtime uses the file's template unless the caller overrides it
(`chatTemplate` in Swift, `litert_lm_conversation_config_set_prompt_template` in C); our build
first edits its tool-call boundary (`FixGemma4ToolTemplate`, ours: `9a84a518`;
[ConversationConfig](#conversationconfig) and
[What the KV cache holds](#what-the-kv-cache-holds-at-a-turn-boundary)). The `-gpu`
files carry a third variant that opens `<|channel>thought\n` after a tool response when thinking is
on ([e4b-gpu](../generated/models/e4b-gpu.md)). The chat format as a model property is covered in
[../models/tokenizer-and-chat-format.md](../models/tokenizer-and-chat-format.md).

### Model data processors and multimodal input

`Gemma4DataProcessorConfig` defaults ([`runtime/conversation/model_data_processor/gemma4_data_processor_config.h:24-82`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/gemma4_data_processor_config.h#L24-L82)),
overridden by any metadata field that differs from the proto default
([`runtime/conversation/model_data_processor/model_data_processor_factory.cc:235-260`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/model_data_processor_factory.cc#L235-L260)):
<!-- c:LM-CNV-031 -->

| Field | Default | In the E2B/E4B files |
| --- | --- | --- |
| `boi_token` / `eoi_token` | `<\|image>` / `<image\|>` | same |
| `boa_token` / `eoa_token` | `<\|audio>` / `<audio\|>` | same |
| `patch_width` x `patch_height` | 16 x 16 | same |
| `max_num_patches` | 2520 | 2520 |
| `pooling_kernel_size` | 3 | default |
| `merge_patches` | false (12B only) | default |
| `code_fence_start` / `code_fence_end` | `<\|tool_call>` / `<tool_call\|>` | same |
| `open_quote` / `close_quote` | `<\|"\|>` | same |
| `function_response_start` | `<\|tool_response>` | same |
| `syntax_type` | `fc` | default |
| `constraint_mode` | `kTextAndOr` | default |

The Gemma 4 processor doesn't override `MessageToTemplateInput` or `FormatTools`: tool declarations
and tool calls reach the template as JSON and the template formats them
([`runtime/conversation/model_data_processor/gemma4_data_processor.h:45-116`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/gemma4_data_processor.h#L45-L116)).

`ToInputDataVectorImpl` hands the rendered text to `ProcessMultimodalPrompt`
([`runtime/conversation/model_data_processor/gemma4_data_processor.cc:110-144`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/model_data_processor/gemma4_data_processor.cc#L110-L144)), which:

1. Collects the image and audio parts of the messages being prefilled, in order.
2. Splits the text at `<|image|>`, `<start_of_image>`, `<|audio|>` and `<start_of_audio>`.
3. Replaces each image placeholder with `InputText(text so far + "<|image>")`, the preprocessed
   `InputImage` and `InputImageEnd`; each audio placeholder likewise with `<|audio>`, `InputAudio` and
   `InputAudioEnd` ([`runtime/conversation/model_data_processor/multimodal_processor_helper.cc:115-188`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/multimodal_processor_helper.cc#L115-L188)).
4. Appends the rest of the text ([`runtime/conversation/model_data_processor/multimodal_processor_helper.cc:200-203`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/multimodal_processor_helper.cc#L200-L203))
   and fails if there are more or fewer media parts than placeholders
   ([`runtime/conversation/model_data_processor/multimodal_processor_helper.cc:190-198`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/multimodal_processor_helper.cc#L190-L198)).

The image budget: with a `visual_token_budget`, `max_num_patches` becomes budget x
`pooling_kernel_size`² ([`runtime/conversation/model_data_processor/multimodal_processor_helper.cc:91-103`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/multimodal_processor_helper.cc#L91-L103)).
Without one, the file's 2520 patches at the default 3 x 3 pooling mean 280 tokens per image.
<!-- c:LM-CNV-034 --> `SendMessageAsync` rejects a budget that is not positive or exceeds the engine's
max vision tokens per image ([`runtime/conversation/conversation.cc:597-602`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L597-L602)). The token
counts the model itself supports are on [../models/gemma4-e4b.md](../models/gemma4-e4b.md); the
preprocessors and the vision and audio executors are on [components.md](components.md) and
[executors.md](executors.md).

`ToMessageImpl` parses tool calls only when the preface has tools; otherwise the whole text becomes one
text part ([`runtime/conversation/model_data_processor/data_utils.cc:86-108`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/data_utils.cc#L86-L108)).

### Sending a message

```mermaid
sequenceDiagram
  participant U as Caller thread
  participant CV as Conversation
  participant S as SessionAdvanced
  participant X as Execution manager (engine threads)
  U->>CV: SendMessageAsync(message, callback, args)
  CV->>CV: render old/new, prefill text (minijinja)
  CV->>CV: ToInputDataVector (load and preprocess media)
  CV->>CV: history_.push_back(message)
  CV->>S: RunPrefillAsync(inputs, prefill callback)
  S->>S: tokenize (+BOS on a fresh session)
  S->>X: queue prefill task
  X-->>CV: prefill done
  CV->>S: SaveCheckpoint("channel_content_checkpoint") [filter on]
  CV->>S: RunDecodeAsync(internal callback, DecodeConfig)
  loop each decoded chunk
    X-->>CV: Responses(kProcessing, text)
    CV-->>U: Message chunk (text, or channel text)
  end
  X-->>CV: Responses(kDone)
  CV->>CV: final message, history_.push_back(reply)
  alt reply has channels and filter on
    CV->>S: RewindToCheckpoint, RunPrefillAsync(reply without reasoning)
    X-->>CV: refill done
  end
  CV-->>U: empty Message (turn complete)
```

Rendering, media preprocessing and (inside `RunPrefillAsync`) tokenization run on the thread that
calls `SendMessageAsync` ([`runtime/conversation/conversation.cc:609-612`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L609-L612),
[`runtime/conversation/conversation.cc:688-691`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L688-L691),
[`runtime/core/session_advanced.cc:117-143`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/core/session_advanced.cc#L117-L143)). The prefill callback starts decode with the
`DecodeConfig` from `CreateDecodeConfig`; a prefill error or a non-`Done` end state skips decode
([`runtime/conversation/conversation.cc:846-908`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L846-L908)). Which thread runs the tasks and delivers
the callbacks is on [engine-and-sessions.md](engine-and-sessions.md). `SendMessage` is the same call
followed by `WaitUntilDone` and a wait for the completion sentinel; it returns the last history entry
([`runtime/conversation/conversation.cc:504-559`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L504-L559)).

**The internal callback** (`CreateInternalCallback`) parses the stream:

- Plain text goes out through `ToMessage` as soon as it can't be the start of a delimiter; a trailing
  partial match of any channel start is held for the next chunk
  ([`runtime/conversation/internal_callback_util.cc:395-419`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L395-L419)).
- Named-channel text streams as `{"role": "assistant", "channels": {"thought": ...},
  "reasoning_content": ...}`, again holding back a partial end delimiter
  ([`runtime/conversation/internal_callback_util.cc:76-91`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L76-L91),
  [`runtime/conversation/internal_callback_util.cc:212-232`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L212-L232)).
- A tool-call block is a channel with an empty name: buffered with its fences and emitted through
  `ToMessage` only when the end fence arrives; with `stream_tool_calls` its inner text also streams on
  the `tool_call` channel ([`runtime/conversation/internal_callback_util.cc:154-169`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L154-L169),
  [`runtime/conversation/internal_callback_util.cc:439-462`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L439-L462),
  [`runtime/conversation/internal_callback_util.cc:476-489`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L476-L489)).
- At `kDone`, `SendCompleteMessage` flushes, extracts the channel spans from the whole text (an
  unterminated span runs to the end), builds the final message, stores the spans under `channels` and
  `reasoning_content`, runs `complete_message_callback` (which appends the reply to history) and then
  sends the empty `Message` that means "done"
  ([`runtime/conversation/internal_callback_util.cc:95-149`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L95-L149),
  [`runtime/conversation/channel_util.cc:54-91`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/channel_util.cc#L54-L91),
  [`runtime/conversation/channel_util.cc:114-139`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/channel_util.cc#L114-L139)).

### Channels and thinking

**Turning thinking on.** A resolved `ThinkingConfig` with `enable_thinking` does two things:

1. The template gets `enable_thinking = true`, which for Gemma 4 opens the system turn with
   `<|think|>`.
2. `CreateDecodeConfig` sets the thinking token budget and tokenizes the first channel's start and end
   strings into thinking start/end token IDs (no start IDs if the rendered prompt already opened the
   channel). Without thinking, the budget is 0
   ([`runtime/conversation/conversation.cc:365-393`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L365-L393)). Decode then wraps any other
   constraint in a `ThinkingBudgetConstraint` whenever the budget is non-zero, `-1` included
   ([`runtime/core/tasks.cc:709-722`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/core/tasks.cc#L709-L722)); how the budget ends the thought is on
   [components.md](components.md).

**Removing reasoning from the KV cache.** The model writes its reasoning into the KV
cache as it generates it, but the template never renders it back (`strip_thinking`). Left alone, the
cache would hold text that the next render's `old` string doesn't contain, and the reasoning would
take context room. With `filter_channel_content_from_kv_cache` (default true):

1. **Checkpoint.** After the user turn's prefill, which ends with the generation prompt
   `<|turn>model\n`, and before decode, the prefill callback saves the session checkpoint
   `channel_content_checkpoint` and records `checkpoint_message_index_ = history.size()` and
   `checkpoint_after_user_prefill_ = true`
   ([`runtime/conversation/conversation.cc:872-885`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L872-L885)). <!-- c:LM-CNV-050 --> (Templates
   whose generation prompt opens a channel, such as Qwen 3's `<think>\n`, checkpoint before the user
   prefill instead so the open tag goes too;
   [`runtime/conversation/conversation.cc:643-658`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L643-L658). The Gemma 4 E2B/E4B template never
   does.)
2. **Decode** runs; the reply streams to the caller with its reasoning on the `thought` channel.
3. **Rewind and refill.** If the final message has `channels`, `complete_message_callback` sets
   `defer_completion`, rewinds to the checkpoint and computes the refill as
   render(history up to the checkpoint + reply, `add_generation_prompt = false`) minus
   render(history up to the checkpoint), then prefills it. The second render has
   `add_generation_prompt = true` when the checkpoint was saved after the user prefill, as for
   Gemma 4, and false when it was saved before it
   ([`runtime/conversation/conversation.cc:775-834`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L775-L834),
   [`runtime/conversation/conversation.cc:1161-1194`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L1161-L1194)).
4. **Completion.** The caller's "done" sentinel is held back until the refill prefill finishes
   ([`runtime/conversation/conversation.cc:749-770`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L749-L770)). <!-- c:LM-CNV-052 -->

For a Gemma 4 reply `<|channel>thought\n...<channel|>Hello`, the cache goes from
`...<|turn>model\n<|channel>thought\n...<channel|>Hello` to `...<|turn>model\n` (rewind) to
`...<|turn>model\nHello<turn|>\n` (refill): exactly what the next `old` render contains.

The rewind is cheap: `RewindToCheckpoint` restores the saved step, session state and task
dependencies, drops later checkpoints and calls the executor's `SetCurrentStep`, which moves
`current_step` back, marks the MTP drafter unprimed and drops a deferred speculative round. When the
KV caches include rings (some caches have fewer slots than others), it also puts back the ring copy
taken before a run overwrote the window of the step `SaveCheckpoint` marked, and refuses a rewind of
more than the ring's slots less 512 positions to any other step (ours: `6cfd1a28`). Without rings,
or when no run reached that far, no KV data is copied; the refill overwrites the abandoned positions
([`runtime/core/session_advanced.cc:506-547`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/core/session_advanced.cc#L506-L547),
[`runtime/executor/llm_litert_compiled_model_executor.cc:2069-2101`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/executor/llm_litert_compiled_model_executor.cc#L2069-L2101),
[`runtime/executor/llm_litert_compiled_model_executor.cc:2176-2234`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/executor/llm_litert_compiled_model_executor.cc#L2176-L2234),
[`runtime/executor/litert/state.cc:1402-1412`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/executor/litert/state.cc#L1402-L1412)). <!-- c:LM-CNV-055 -->
The cost is the refill prefill (the answer's tokens plus `<turn|>\n`), paid before the turn reports
done. When rewinding is disabled (the C API does that only for GPU artisan executors with autosized
ring buffers), the filter uses `SaveContext`/`RestoreContext`, which store and restore a
`ProcessedContext` ([`c/conversation.cc:543-553`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/c/conversation.cc#L543-L553),
[`runtime/core/session_advanced.cc:547-592`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/core/session_advanced.cc#L547-L592)).

This eager form comes from upstream commit `97e6dc7d` (2026-10-07), "Eagerly filter channel content
from KV cache after decode", which is in our base, so our build and the release Agello ships both
filter this way. Google's v0.18.0, which Agello shipped until 2026-10-28, predates it and refilled
lazily, at the next user message ([binaries.md](../generated/binaries.md), the record at its end).
<!-- c:LM-CNV-053 -->

**Settings.** The Swift wrapper passes the flag to C only if
`ExperimentalFlags.filterChannelContentFromKvCache` is non-nil; it is nil by default, so the C++
default (true) applies ([`swift/Engine.swift:360-366`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Engine.swift#L360-L366),
[`swift/ExperimentalFlags.swift:36`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ExperimentalFlags.swift#L36), [`swift/ExperimentalFlags.swift:167-171`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ExperimentalFlags.swift#L167-L171), [`c/conversation.cc:530-533`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/c/conversation.cc#L530-L533)).
<!-- c:LM-CNV-057 -->

### Tool calling

The pieces for Gemma 4:

1. **Declarations.** Swift `Tool`s become a JSON tools list ([swift-api.md](swift-api.md)); the C API
   puts it in `JsonPreface.tools`; the template writes `<|tool>declaration:...<tool|>` entries into
   the system turn.
2. **Generation.** The model writes `<|tool_call>call:NAME{...}<tool_call|>`; the internal callback
   buffers the block until the end fence.
3. **Parsing.** `ParseTextAndToolCalls` splits text from fenced blocks and parses each block with the
   FC grammar (ANTLR, compiled into a Rust lexer and parser): `call:NAME{key:value,...}`, values
   being numbers, booleans, null,
   objects, arrays or strings between `<|"|>`, `<escape>` or `<ctrl46>` delimiters. Each call becomes
   `{"type": "function", "function": {...}}` in `tool_calls`
   ([`runtime/components/tool_use/parser_utils.cc:109-156`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/tool_use/parser_utils.cc#L109-L156),
   [`runtime/components/tool_use/fc_parser_utils.cc:33-48`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/tool_use/fc_parser_utils.cc#L33-L48),
   [`runtime/components/tool_use/antlr/AntlrFcParser.g4:23-29`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/tool_use/antlr/AntlrFcParser.g4#L23-L29),
   [`runtime/components/tool_use/antlr/AntlrFcLexer.g4:26-41`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/tool_use/antlr/AntlrFcLexer.g4#L26-L41),
   at Google's base: the prune `edb2c7db` removed the `.g4` sources from our tree and kept the Rust
   generated from them,
   [`runtime/components/tool_use/antlr/generated/antlrfclexer.rs:1`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/tool_use/antlr/generated/antlrfclexer.rs#L1)). <!-- c:LM-CNV-060 -->
4. **Failure.** With `return_error_on_parse_failure` (default true) an unparsable block fails the
   reply with `InvalidArgument`; otherwise the block stays as a text part with an `error` field
   ([`runtime/components/tool_use/parser_utils.cc:138-151`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/tool_use/parser_utils.cc#L138-L151)).
5. **Responses.** The caller sends a `tool` message; the template renders it as
   `<|tool_response>...<tool_response|>` and adds no generation prompt, so the model continues.

The Swift wrapper's automatic loop runs up to 25 model calls per `sendMessage`
([`swift/Conversation.swift:48`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L48), [`swift/Conversation.swift:154-180`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L154-L180)); that
loop, and our patch that answers failing tool calls with an error response (ours: `f3408e06`), are on
[swift-api.md](swift-api.md). The parsers are linked unless `//build_config:disable_tool_use_parser`
is set; `--config=agello` doesn't set it
([`runtime/components/tool_use/BUILD:218-242`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/tool_use/BUILD#L218-L242), [`.bazelrc:553-566`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/.bazelrc#L553-L566)).

Tool calls are not constrained in our build: a tool-call constraint needs the FST provider
([Constraints](#constraints)).

### What the KV cache holds at a turn boundary

The render difference assumes that the cache holds the previous render. Traced in the code, it
holds something slightly different wherever no refill rewrites the reply. Our build changed three
of these boundaries on 2026-11-17 (Agello's probes 35 and 36); the release Agello ships
(`agello-one-20261115`) predates those changes, so this section says which one it describes.
Nothing here was observed in a token dump ([Open questions](#open-questions)).

**The stop token stays pending.** With the executor's own sampler (the default; the C++ runtime, the
C API and the Swift wrapper never set `UseExternalSampler`, and only Google's web package, which our
tree removed, passes a caller's `useExternalSampler` through), each decode step or MTP round keeps
its last token as the *pending* token: counted in `current_step`, not yet in the KV cache. When that
token is a stop token, `Tasks` drops it from the text and ends the loop, so a reply's `<turn|>` (106)
is left pending when it is the last token of its step or round. In our build it is also left pending
when an MTP round emitted tokens after it: `DecodeOneStep` rolls back only those (ours: `e62e6ff1`,
as upstream's `bcd7532b`)
([`runtime/executor/llm_litert_compiled_model_executor.cc:1611-1632`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/executor/llm_litert_compiled_model_executor.cc#L1611-L1632),
[`runtime/core/tasks.cc:362-377`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/core/tasks.cc#L362-L377),
[`runtime/core/tasks.cc:817-822`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/core/tasks.cc#L817-L822),
[`runtime/engine/engine_settings.h:277-280`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.h#L277-L280),
[`js/packages/core/src/session_config.ts:104-105`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/js/packages/core/src/session_config.ts#L104-L105), at Google's base; in the release,
[`runtime/executor/llm_litert_compiled_model_executor.cc:1580-1601`](https://github.com/PowerBeef/agello-litert-lm/blob/6da75396ece779d3c7b6b605ec8c2c06861ce1eb/runtime/executor/llm_litert_compiled_model_executor.cc#L1580-L1601)). <!-- c:X-GAP-001 -->
The next prefill writes the pending token first, at its own position, then the new difference. When
`<turn|>` is pending, a user message's difference starts with `<|turn>user\n`, because the old render
already ended with `<turn|>\n` (claim LM-CNV-021), so the `\n` (107) that the template puts after
`<turn|>` is never written, in either build
([`runtime/executor/llm_litert_compiled_model_executor.cc:694-748`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/executor/llm_litert_compiled_model_executor.cc#L694-L748);
in the release, [`runtime/executor/llm_litert_compiled_model_executor.cc:694-720`](https://github.com/PowerBeef/agello-litert-lm/blob/6da75396ece779d3c7b6b605ec8c2c06861ce1eb/runtime/executor/llm_litert_compiled_model_executor.cc#L694-L720);
[template text](../generated/models/e4b-ours.md)). <!-- c:X-GAP-002 -->

**It is written with its embedding in our build.** Decode creates the pending token as an id only.
The executor looks up embeddings for a prefill's last token and, at the next decode, for the token
about to be decoded; `FillInputBufferWithToken` writes a token without embedding as its `int32` id in
the first 4 bytes of the row and zeros after it. Our build's prefill first looks up the embeddings
and per-layer embeddings of a pending token that has none (`LookupPendingTokenEmbeddings`, ours:
`a82611eb`, as upstream's `0101cf3b` does), so the Gemma 4 prefill signatures, which take embeddings
and per-layer embeddings rather than token ids, get the stop token's own rows. In the release Agello
ships, that position gets the id-and-zeros row in both inputs, and its KV entry is not computed from
`<turn|>`
([`runtime/executor/llm_litert_compiled_model_executor.cc:876-929`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/executor/llm_litert_compiled_model_executor.cc#L876-L929),
[`runtime/executor/llm_litert_compiled_model_executor.cc:1060-1095`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/executor/llm_litert_compiled_model_executor.cc#L1060-L1095),
[`runtime/executor/llm_litert_compiled_model_executor.cc:499-537`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/executor/llm_litert_compiled_model_executor.cc#L499-L537),
[`runtime/executor/llm_litert_compiled_model_executor.cc:539-563`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/executor/llm_litert_compiled_model_executor.cc#L539-L563);
in the release, [`runtime/executor/llm_litert_compiled_model_executor.cc:499-537`](https://github.com/PowerBeef/agello-litert-lm/blob/6da75396ece779d3c7b6b605ec8c2c06861ce1eb/runtime/executor/llm_litert_compiled_model_executor.cc#L499-L537)). <!-- c:X-GAP-003 -->

**After a reply with reasoning (filter on)** the refill writes the render's tokens, ending `<turn|>`
with the final `\n` pending and its embedding looked up, so the next difference continues the render
exactly (claim LM-CNV-052). After the rewind, the refill's `RollBackProcessedTokens` re-pends the `\n`
that ended `<|turn>model\n` as an id-only token; our build writes it from its looked-up embedding, so
the whole refill matches the render, while the release rewrites that position from an id-and-zeros
row
([`runtime/executor/llm_litert_compiled_model_executor.cc:565-595`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/executor/llm_litert_compiled_model_executor.cc#L565-L595),
[`runtime/executor/llm_litert_compiled_model_executor.cc:2069-2101`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/executor/llm_litert_compiled_model_executor.cc#L2069-L2101),
[`runtime/executor/llm_executor_processed_tokens.cc:92-132`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/llm_executor_processed_tokens.cc#L92-L132)). <!-- c:X-GAP-004 -->

**With MTP**, a round can emit the stop token before its last token. Our build's `DecodeOneStep::Run`
then moves `current_step` back by the number of tokens after the stop token, which leaves the stop
token pending, as after a single-token step. The release moves it back by `sequence_length - step`,
which drops the stop token too; the next prefill re-pends the token before it (rewriting that
position from an id-and-zeros row), so neither `<turn|>` nor `\n` is in the cache at that boundary,
and a reply streamed as "400" read as "40" to the next turn (Agello's probe 35)
([`runtime/core/tasks.cc:362-377`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/core/tasks.cc#L362-L377);
in the release, [`runtime/core/tasks.cc:363-373`](https://github.com/PowerBeef/agello-litert-lm/blob/6da75396ece779d3c7b6b605ec8c2c06861ce1eb/runtime/core/tasks.cc#L363-L373)). <!-- c:X-GAP-005 -->

**Tool calls.** Our build edits the Gemma 4 file template's tool-call boundary
(`FixGemma4ToolTemplate`, ours: `9a84a518`, after upstream's `f240215c`): a model message that is a
tool call ends with `<|tool_response>`, the stop token decoding keeps pending, and the tool response
continues from it, with no generation prompt and no turn header before the answer. The release
renders the file's template as it is: `<tool_call|><turn|>\n<|tool_response>`, and `<|turn>model\n`
before an answer that the model generated right after `<tool_response|>`. Our build also keeps the
answer to a tool response from opening with a stop token (a first-token-only
`SuppressTokensConfig`, ours: `586e2e4c`): under a one-line system prompt the model had emitted
`<|tool_response>` there, and the reply ended empty (Agello's probe 36;
[`runtime/conversation/conversation.cc:729-737`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L729-L737)).

Gemma 4 E2B/E4B ("slot" = a position written from an id-and-zeros row):

| Boundary | The render has (our build) | Our build's KV cache holds | The release's KV cache holds |
| --- | --- | --- | --- |
| Reply without reasoning, then a user message | `Hello<turn\|>\n<\|turn>user\n` | `Hello<turn\|><\|turn>user\n` | `Hello`, the `<turn\|>` slot, `<\|turn>user\n` |
| Same, MTP stopped before the round's last token | `Hello<turn\|>\n<\|turn>user\n` | `Hello<turn\|><\|turn>user\n` | `Hell`, `o` as a slot, `<\|turn>user\n` |
| Reply that is only a tool call, without reasoning (stopped on 50), then the tool response | `<tool_call\|><\|tool_response>response:...` | the same | `<tool_call\|>`, the stop token's slot, `<\|tool_response>response:...` (render: `<tool_call\|><turn\|>\n<\|tool_response>`) |
| Answer generated after a tool response, without reasoning | `<tool_response\|>18 °C...<turn\|>\n` | `<tool_response\|>18 °C...`, then the stop token pending | the same (render: `<tool_response\|><\|turn>model\n18 °C...`) |
| Reply with reasoning (filter on) | `<\|turn>model\nHello<turn\|>\n` | `<\|turn>model\nHello<turn\|>`, `\n` pending | `<\|turn>model`, `\n` as a slot, `Hello<turn\|>`, `\n` pending |

The tool rows hold for a reply with no text besides the call whose stop token was the last of its
decode step or MTP round; a call that stopped on `<turn|>` (106) instead leaves that token where our
build's render has `<|tool_response>`. They follow from the templates: the tool message's difference
is `response:...<tool_response|>` in our build and starts with `<|tool_response>` in the release, and
the answer is generated right after `<tool_response|>`
([`runtime/conversation/conversation.cc:297-320`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L297-L320),
[`runtime/conversation/agello_gemma4_template_test.cc:96-126`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/agello_gemma4_template_test.cc#L96-L126);
claims MOD-TOK-049, MOD-TOK-050 for the file's template). <!-- c:X-GAP-006 -->

**Token counts.** `GetTokenCount` returns the executor's `current_step`, which counts the pending token
([`runtime/conversation/conversation.cc:936-938`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L936-L938),
[`runtime/executor/llm_litert_compiled_model_executor.h:126-128`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/executor/llm_litert_compiled_model_executor.h#L126-L128),
[`runtime/executor/llm_executor_processed_tokens.cc:31-33`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/executor/llm_executor_processed_tokens.cc#L31-L33)).
Against the token count of the render of the same history (taking the model's tokens to be the
render's encoding of its text), in our build it is 1 lower per reply without reasoning, except a
tool-call reply that stopped on `<|tool_response>`, which counts equal, and equal after a reasoning
refill. In the release Agello ships it is 1 lower for a tool-call reply too, 2 lower instead when MTP
stopped before a round's last token, and a further 3 lower for each answer generated right after a
tool response without reasoning (the `<|turn>model\n` its later renders put before it).
<!-- c:X-GAP-007 --> An estimate computed from renders therefore runs a little high, the safe side for
a context budget.

### Constraints

**Providers.** `ConstraintProviderConfig` is `std::variant<ExternalConstraintConfig,
LlGuidanceConfig, FstConfig>`. `FstConfig` is a placeholder with a deleted constructor, and
`CreateConstraintProvider` builds only the external and llguidance providers
([`runtime/components/constrained_decoding/constraint_provider_config.h:26-30`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/constrained_decoding/constraint_provider_config.h#L26-L30),
[`runtime/components/constrained_decoding/fst_constraint_config.h:21-29`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/constrained_decoding/fst_constraint_config.h#L21-L29),
[`runtime/components/constrained_decoding/constraint_provider_factory.cc:32-62`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/constrained_decoding/constraint_provider_factory.cc#L32-L62)).
<!-- c:LM-CNV-065 --> The closed Gemma FST provider is not a `ConstraintProvider`: model data
processors call it directly, through a C interface, to build tool-call constraints.

| Path | Built by | Used for | In our build |
| --- | --- | --- | --- |
| llguidance | `LlgConstraintProvider` (open, Rust) | per-message regex, JSON schema, Lark | yes |
| External | `ExternalConstraintProvider` | a caller's own `Constraint` (C++ only) | yes |
| Gemma FST | `LiteRtLmGemmaModelConstraintProvider` (closed, prebuilt) | tool calls from the preface's tools | no |

**llguidance.** The provider builds an llguidance tokenizer from every vocabulary token and creates
regex, JSON-schema, Lark or llguidance-internal constraints
([`runtime/components/constrained_decoding/llg_constraint_provider.cc:42-58`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/constrained_decoding/llg_constraint_provider.cc#L42-L58),
[`runtime/components/constrained_decoding/llg_constraint_provider.cc:142-165`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/constrained_decoding/llg_constraint_provider.cc#L142-L165)). Its EOS
token, unless set, is the first single-token stop sequence of the session; the C API never sets it,
so for the Gemma 4 files (stop tokens 1, 50, 106 in that order) it is token 1
([`runtime/components/constrained_decoding/constraint_provider_factory.cc:43-57`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/constrained_decoding/constraint_provider_factory.cc#L43-L57),
[`runtime/engine/engine_settings.cc:760-769`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine_settings.cc#L760-L769)). <!-- c:LM-CNV-066 --> The C API maps
regex and JSON-schema constraints to an `LlGuidanceConstraintArg` in
`OptionalArgs.decoding_constraint` (no Lark or external constraints through C)
([`c/conversation.cc:137-151`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/c/conversation.cc#L137-L151),
[`c/conversation.cc:159-175`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/c/conversation.cc#L159-L175),
[`c/conversation.cc:524-528`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/c/conversation.cc#L524-L528)).
`litert_lm_conversation_config_set_constraint_provider` accepts only
`kLiteRtLmConstraintProviderTypeLlGuidance`, in our build and in the release Agello ships (others
return `InvalidArgument`;
[`c/conversation.cc:253-268`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/c/conversation.cc#L253-L268)), and conversation creation acts only on
llguidance. <!-- c:LM-CNV-068 --> Swift
`ConversationConfig(enableResponseFormat: true)` selects that provider and enables constrained
decoding; `ResponseFormat` offers `.json(schema:)` and `.regex(pattern:)`
([`swift/Engine.swift:336-352`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Engine.swift#L336-L352), [`swift/ResponseFormat.swift:20-71`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ResponseFormat.swift#L20-L71)).
<!-- c:LM-CNV-069 --> The mask computation and the logit-mask runner are on
[components.md](components.md).

**Which constraint a decode gets** (`CreateDecodeConfig`,
[`runtime/conversation/conversation.cc:394-413`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L394-L413)): <!-- c:LM-CNV-070 -->

```mermaid
flowchart TD
  A[CreateDecodeConfig] --> B{decoding_constraint given<br/>and provider present?}
  B -- yes --> C[constraint_ = provider.CreateConstraint<br/>llguidance or external]
  B -- no --> D{constrained decoding enabled,<br/>constraint_ empty, preface has tools?}
  D -- yes --> E[processor.CreateConstraint tools<br/>Gemma FST provider]
  E -- Unimplemented --> F[no tool constraint]
  E -- ok --> G[constraint_ = tool-call constraint]
  D -- no --> H[keep constraint_ as it is]
  C --> I[DecodeConfig.SetConstraint constraint_]
  F --> I
  G --> I
  H --> I
```

**Our build.** Apple builds use `--config=agello`, which defines
`LITERT_LM_FST_CONSTRAINTS_DISABLED=1` (ours: `43f0f5c3`). That selects
`//build_config:disable_fst_constraints`, so the Gemma 3, Gemma 4 and FunctionGemma processors compile
with the define and without `gemma_model_constraint_provider_lib`
([`.bazelrc:553-556`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/.bazelrc#L553-L556), [`build_config/BUILD:219-243`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/build_config/BUILD#L219-L243),
[`runtime/conversation/model_data_processor/BUILD:685-692`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/BUILD#L685-L692),
[`runtime/conversation/model_data_processor/BUILD:721-726`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/BUILD#L721-L726)). <!-- c:LM-CNV-072 --> With
the define, `Gemma4DataProcessor::Create` succeeds even when constrained decoding is on, and
`CreateConstraint` returns `Unimplemented`, which `CreateDecodeConfig` skips (ours: `d1488ce7`)
([`runtime/conversation/model_data_processor/gemma4_data_processor.cc:62-68`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/model_data_processor/gemma4_data_processor.cc#L62-L68),
[`runtime/conversation/model_data_processor/gemma4_data_processor.cc:160-163`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/model_data_processor/gemma4_data_processor.cc#L160-L163)).
<!-- c:LM-CNV-073 --> Before `d1488ce7` (2026-10-14), as in Google's base and in v0.18.0, the
define made `Create` fail with `FailedPrecondition` ("Constrained decoding was disabled at build
time.") whenever constrained decoding was enabled; since the Swift wrapper enables it for every
response format, JSON output failed although it never needed the FST provider
([`runtime/conversation/model_data_processor/gemma4_data_processor.cc:62-66`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/gemma4_data_processor.cc#L62-L66),
at Google's base). Our build and the release Agello ships both carry `d1488ce7`.
<!-- c:LM-CNV-074 -->

**Our binary** defines none of the provider's symbols and 94 of llguidance's; in the record of
2026-10-16 at the end of [binaries.md](../generated/binaries.md), Google's prebuilt v0.18.0, which
Agello shipped until 2026-10-28, defined 17 of the provider's (and 127 of llguidance's).
<!-- c:LM-CNV-076 --> In builds that compile the provider in (without the define), enabling
constrained decoding makes `Gemma4DataProcessor::Create` create the closed provider from the
serialized SentencePiece model and the stop-token sequences
([`runtime/conversation/model_data_processor/gemma4_data_processor.cc:69-106`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/model_data_processor/gemma4_data_processor.cc#L69-L106)).

**Answer: a JSON response format.** In our build and in the release Agello ships it takes the
llguidance path. Swift's `enableResponseFormat` sets the llguidance provider
([`swift/Engine.swift:336-345`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Engine.swift#L336-L345); in the release,
[`swift/Engine.swift:330-339`](https://github.com/PowerBeef/agello-litert-lm/blob/6da75396ece779d3c7b6b605ec8c2c06861ce1eb/swift/Engine.swift#L330-L339)); the C API turns the format into an
`LlGuidanceConstraintArg` and sets the provider
([`c/conversation.cc:137-151`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/c/conversation.cc#L137-L151),
[`c/conversation.cc:524-528`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/c/conversation.cc#L524-L528)); `CreateDecodeConfig` builds the constraint through
that provider ([`runtime/conversation/conversation.cc:394-410`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L394-L410)). The FST provider builds
constraints only for tool calls from the preface's tools, though in builds that include it
`Gemma4DataProcessor::Create` still creates it whenever constrained decoding is enabled
([`runtime/conversation/model_data_processor/gemma4_data_processor.cc:69-106`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/model_data_processor/gemma4_data_processor.cc#L69-L106)).
Agello's conversations never combine constrained decoding with tools. <!-- c:LM-CNV-078 -->

## On Apple

Nothing in this layer depends on the platform: it is C++ (`runtime/conversation`) plus Rust
(minijinja, llguidance, the ANTLR tool-call parsers), compiled into `CLiteRTLM` for iOS and macOS.
The Apple-specific facts are about what gets linked and what Agello sets:

- **No FST provider in our xcframework.** `agello/build-xcframework.sh` builds with
  `--config=agello` ([`agello/build-xcframework.sh:9-10`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/agello/build-xcframework.sh#L9-L10)), which drops the Gemma constraint
  provider (above). <!-- c:LM-CNV-072 --> In the open-source BUILD that our tree keeps, the
  provider, closed source, is a prebuilt `libGemmaModelConstraintProvider.dylib` for `ios_arm64`,
  `ios_sim_arm64` and `macos_arm64` (`.so` on Android and Linux, `.dll` on Windows), linked through
  `gemma_model_constraint_provider_shared_lib`
  ([`runtime/components/constrained_decoding/BUILD:252-263`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/components/constrained_decoding/BUILD#L252-L263)).
  Our `--config=agello` build leaves it out, and our binary defines none of its symbols
  ([binaries.md](../generated/binaries.md)). <!-- c:LM-CNV-075 -->
- **Agello turns the channel filter on** explicitly, so reasoning never stays in a chat's KV cache
  (Agello `Agello/Inference/ChatEngine.swift` lines 198-205 at `a691212`). <!-- c:LM-CNV-058 -->
  Its comment there ("at the next message the runtime rewinds and refills") describes Google's
  v0.18.0, which Agello shipped until 2026-10-28; in our build and the release Agello ships the
  refill happens at the end of each reply.
- **Agello's JSON helpers** each run on a new conversation with `enableResponseFormat: true` and
  speculative decoding off unless `AGELLO_JSON_MTP=1`; its chat conversations don't enable
  constrained decoding (Agello `ChatEngine.swift` lines 392-422 and 626-656 at `a691212`).
  <!-- c:LM-CNV-079 --> The sticky constraint described in
  [Limits](#limits-and-failure-modes) therefore can't reach a chat.
- **Template rendering and media preprocessing run on the calling thread** inside the C call (in
  Swift, whatever task calls `sendMessage` or `sendMessageStream`), not on the engine's threads.

## Settings

Rows for [../configuration.md](../configuration.md). Defaults are our build's.

| Name | Layer | Default | Effect |
| --- | --- | --- | --- |
| `SetFilterChannelContentFromKvCache` / `litert_lm_conversation_config_set_filter_channel_content_from_kv_cache` / `ExperimentalFlags.filterChannelContentFromKvCache` | C++ / C / Swift | true / unset / nil | remove reasoning from the KV cache after each reply |
| `SetEnableConstrainedDecoding` / `..._set_enable_constrained_decoding` / `enableResponseFormat`, `ExperimentalFlags.enableConversationConstrainedDecoding` | C++ / C / Swift | false | tool-call constraints (needs the FST provider); Swift's `enableResponseFormat` also selects llguidance |
| `SetConstraintProviderConfig` / `..._set_constraint_provider` | C++ / C | none | provider for per-message constraints |
| `decoding_constraint` / `litert_lm_conversation_optional_args_set_constraint` / `responseFormat` | C++ / C / Swift | none | a per-message regex or JSON-schema constraint |
| `SetThinkingConfig`, `OptionalArgs.thinking_config` / `ThinkingConfig` | C++ / C / Swift | none | `<\|think\|>` in the system turn, thinking budget |
| `SetOverwritePromptTemplate` / `..._set_prompt_template` / `chatTemplate` | C++ / C / Swift | the file's template | replaces the Jinja template |
| `SetStreamToolCalls` / `enableToolCallStreaming` + `ExperimentalFlags.enableConversationToolCallStreaming` | C++ / Swift | false, `"tool_call"` | stream tool-call text |
| `SetReturnErrorOnParseFailure` | C++ | true | unparsable tool call fails the reply |
| `SetReturnErrorOnMaxTokensReached` | C++ | false | full KV cache ends the reply normally |
| `SetPrefillPrefaceOnInit` | C++ | false | prefill the preface at creation |
| `SetEnableRewinding` | C++ (C forces false for artisan ring buffers) | true | step rewinds vs saved contexts |
| `visual_token_budget` | C++ / C / Swift (`visualTokenBudget`) | file's `max_num_patches` / 9 | image resolution in tokens |
| `LITERT_LM_FST_CONSTRAINTS_DISABLED` / `//build_config:enable_fst_constraints` | Bazel define / flag | not set / true | drop the closed FST provider (`--config=agello` sets it) |

## Limits and failure modes

- **A response-format constraint sticks.** `Conversation` stores the constraint it creates in
  `constraint_` and never clears it; every later decode gets it again, so a JSON schema passed with
  one message also constrains later messages of the same conversation that pass none
  ([`runtime/conversation/conversation.cc:394-412`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L394-L412),
  [`runtime/conversation/conversation.h:820-822`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.h#L820-L822)). <!-- c:LM-CNV-071 --> `Clone` doesn't
  copy it. Use one conversation per constrained request, as Agello does.
- **Cancellation leaves the KV cache dirty.** On a cancelled task, `cancel_callback` pops the
  messages the send added and clears the channel checkpoint, but doesn't rewind the session: the
  cancelled turn's tokens stay in the cache while the history no longer mentions them
  ([`runtime/conversation/conversation.cc:632-641`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L632-L641),
  [`runtime/conversation/internal_callback_util.cc:292-309`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L292-L309)). <!-- c:LM-CNV-081 -->
  `CancelProcess` cancels every task of the session, and both `Conversation` and `SessionAdvanced`
  document that reuse afterwards is not supported ("poisoned")
  ([`runtime/conversation/conversation.h:655-661`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.h#L655-L661),
  [`runtime/core/session_advanced.h:194-211`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/session_advanced.h#L194-L211)). <!-- c:LM-CNV-082 --> Agello recreates the
  conversation from saved history after a cancel (its CLAUDE.md, "One live conversation per
  engine"). `CancelGroup` needs a `task_group_id`, which the C
  API can't pass ([`runtime/conversation/conversation.cc:976-987`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L976-L987),
  [`c/conversation.cc:105-157`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/c/conversation.cc#L105-L157)).
- **The diff assumes the KV cache holds the last render.** Without reasoning, nothing re-renders the
  reply: the cache holds the reply as generated, ended by the stop token (whether that token is
  written to the cache at the next prefill is an executor matter: [executors.md](executors.md)),
  while the next `old` render ends with `<turn|>\n`. With reasoning and the filter on, the refill
  makes the cache's tokens match the render (in the release Agello ships, one position is written
  from an id-and-zeros row). Traced in
  [What the KV cache holds at a turn boundary](#what-the-kv-cache-holds-at-a-turn-boundary).
- **Template prefix mismatch** fails the send with `InternalError` ("does not start with the previous
  rendered template string") when a template renders history differently depending on what follows
  ([`runtime/conversation/conversation.cc:313-318`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L313-L318)).
- **Per-message thinking toggles don't reach the system turn.** Both renders use the message's
  `enable_thinking`, so the difference never contains the system turn: whatever the first prefill
  wrote there (`<|think|>` or nothing) stays for the conversation's life
  ([`runtime/conversation/conversation.cc:260-266`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L260-L266)). Agello sets thinking per conversation.
- **Full KV cache.** With `return_error_on_max_tokens_reached` false (default), a reply cut by the
  context limit completes normally and enters the history; with true, the turn is removed and the
  caller gets `ResourceExhausted` ([`runtime/conversation/internal_callback_util.cc:319-343`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L319-L343)).
  An unterminated thought then fills the whole `channels.thought` and the answer is empty.
- **Unparsable tool calls** fail the whole reply by default (above).
- **Media count mismatch**: more or fewer images or audio parts than placeholders fails with
  `InvalidArgument`.
- **`has_pending_message`** is rejected for Gemma 4 E2B/E4B files (no single-turn support).
- **No tool-call constraints in our build**: `enableConversationConstrainedDecoding` with tools has no
  effect (silently, `Unimplemented` is skipped); a build with the FST provider builds one.
- **Re-parsing per render.** Each `Apply` re-creates the minijinja environment and re-parses the
  whole Gemma 4 template, and later turns render the whole history twice; the
  cost grows with history length and runs on the caller's thread. Not measured.

## Where in the code

| Concept | Symbol | Citation |
| --- | --- | --- |
| Config and builder | `ConversationConfig::Builder`, `CreateInternal` | [`runtime/conversation/conversation.h:143-284`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.h#L143-L284), [`runtime/conversation/conversation.cc:139-224`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L139-L224) |
| Per-message args | `OptionalArgs` | [`runtime/conversation/conversation.h:387-468`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.h#L387-L468) |
| Creation | `Conversation::Create` | [`runtime/conversation/conversation.cc:416-493`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L416-L493) |
| Prefill text (full history) | `GetSingleTurnTextFromFullHistory` | [`runtime/conversation/conversation.cc:254-321`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L254-L321) |
| Prefill text (single-turn templates) | `RenderSingleTurnTemplate` | [`runtime/conversation/prompt_utils.cc:82-181`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/prompt_utils.cc#L82-L181) |
| Send | `SendMessageAsync` | [`runtime/conversation/conversation.cc:593-911`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L593-L911); the tool-response first-token bans, [`runtime/conversation/conversation.cc:562-591`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L562-L591) |
| Decode config, constraints, thinking | `CreateDecodeConfig` | [`runtime/conversation/conversation.cc:339-414`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L339-L414) |
| Stream parsing | `CreateInternalCallback`, `SendCompleteMessage` | [`runtime/conversation/internal_callback_util.cc:95-149`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L95-L149), [`runtime/conversation/internal_callback_util.cc:248-498`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L248-L498) |
| Channels | `ExtractChannelContent`, `GetOpenChannelName` | [`runtime/conversation/channel_util.cc:38-171`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/channel_util.cc#L38-L171) |
| Rewind and refill | `RewindAndGetInputDataVector`, `GetPrefillTextForMessages` | [`runtime/conversation/conversation.cc:1054-1194`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L1054-L1194) |
| Template engine | `PromptTemplate`, `MinijinjaTemplate` | [`runtime/components/prompt_template.cc:85-148`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/prompt_template.cc#L85-L148), [`runtime/components/rust/minijinja_template.rs:177-259`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/rust/minijinja_template.rs#L177-L259) |
| Processor interface and registry | `ModelDataProcessor`, `ModelDataProcessorRegistry` | [`runtime/conversation/model_data_processor/model_data_processor.h:60-299`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/model_data_processor.h#L60-L299) |
| Gemma 4 processor | `Gemma4DataProcessor` | [`runtime/conversation/model_data_processor/gemma4_data_processor.cc:49-216`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/model_data_processor/gemma4_data_processor.cc#L49-L216) |
| Multimodal assembly | `ProcessMultimodalPrompt` | [`runtime/conversation/model_data_processor/multimodal_processor_helper.cc:40-206`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/multimodal_processor_helper.cc#L40-L206) |
| Tool-call parsing | `ParseTextAndToolCalls`, `ParseFcExpression` | [`runtime/components/tool_use/parser_utils.cc:88-167`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/tool_use/parser_utils.cc#L88-L167) |
| Constraint providers | `CreateConstraintProvider`, `LlgConstraintProvider` | [`runtime/components/constrained_decoding/constraint_provider_factory.cc:32-62`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/constrained_decoding/constraint_provider_factory.cc#L32-L62) |
| FST build switch | `disable_fst_constraints` | [`build_config/BUILD:219-243`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/build_config/BUILD#L219-L243) |
| C API mapping | `litert_lm_conversation_create`, `CreateOptionalArgs` | [`c/conversation.cc:105-157`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/c/conversation.cc#L105-L157), [`c/conversation.cc:438-578`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/c/conversation.cc#L438-L578) |
| Gemma 4 tool boundary (ours) | `FixGemma4ToolTemplate` | [`runtime/conversation/agello_gemma4_template.cc:25-71`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/agello_gemma4_template.cc#L25-L71) |

Other entry points: `GetTokenCount` returns the session's current step and `CountTokens` measures a
prospective message without changing anything
([`runtime/conversation/conversation.cc:936-964`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L936-L964)); `Clone` copies the session, history
and processor state but not `constraint_` or the channel checkpoint
([`runtime/conversation/conversation.cc:989-1019`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L989-L1019)); `RunTextScoring` scores on a cloned
session ([`runtime/conversation/conversation.cc:913-934`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L913-L934)).

## Open questions

- **Turn boundaries: traced, not observed.** The code trace is in
  [What the KV cache holds at a turn boundary](#what-the-kv-cache-holds-at-a-turn-boundary). A token
  dump of a real chat (two turns and a tool call, with and without MTP) would confirm it. Whether the
  `\n` our build never writes after `<turn|>` changes replies was not measured; the release's
  id-and-zeros rows and tool boundary did (Agello's probes 35 and 36, which our build's changes of
  2026-11-17 answer).
- **Effect of the older template.** Whether Gemma 4 E2B/E4B answer differently with the Hugging Face
  template (`<|think|>\n`, reasoning kept within a tool-calling turn) than with the file's was not
  measured.
- **Rendering cost** of the double full-history render per turn at long histories was not measured.
- **MTP with constraints.** Traced: in the release Agello ships, the drafter's llguidance states
  share one mutable matcher, so verification masks and the state after a rejection are wrong
  ([../troubleshooting.md](../troubleshooting.md#json-constrained-replies-break-with-mtp-on)). Our
  build gives each state its own matcher and keeps each state's mask (ours: `1382157b`, 2026-11-17);
  Agello's rule of no MTP for JSON helpers still applies to the release.
