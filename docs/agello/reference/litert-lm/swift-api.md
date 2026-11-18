# LiteRT-LM Swift API (`LiteRTLM`)

> **Scope:** the Swift package in `swift/`: the `Engine` actor and the `Conversation` class, threads,
> lifecycle, streaming (`AsyncThrowingStream`, streams kept by ID, our weak-reference fix), tools
> (`Tool`, `@ToolParam`, automatic calling, our error-response patch), response formats,
> `ExperimentalFlags`, errors, `ModelInfo`, `EmbeddingEngine`, `BuildInfo` and `Telemetry` (ours),
> the `apple_fm` Foundation Models adapter; what Agello ships (our build's wrapper with
> the release's binary); how Agello uses it. **Out of scope:** the C calls underneath
> ([c-api.md](c-api.md)), what the engine and conversation do ([engine-and-sessions.md](engine-and-sessions.md),
> [conversation.md](conversation.md)), every public symbol
> ([`generated/swift-api-inventory.md`](../generated/swift-api-inventory.md)), building the
> xcframework ([build-and-release.md](build-and-release.md)).
> **Pinned to:** `main` `b7b856a0`: LiteRT-LM at the root; Agello ships release `agello-one-20261115`
> (`6da75396`) with our build's Swift wrapper; Agello at `a6912128`.
> **Applies to:** Apple (iOS, macOS). **Last verified:** 2026-11-17

## Summary

`LiteRTLM` is a Swift wrapper over the C API: an `Engine` actor that loads a model, `Conversation`
objects that exchange JSON messages with it, an `EmbeddingEngine` actor, `ModelInfo` for reading a
file's capabilities, and process-wide `ExperimentalFlags`. Generation calls block the thread that
makes them (`sendMessage`) or feed an `AsyncThrowingStream` from LiteRT-LM's own threads
(`sendMessageStream`). Tools are Swift types the wrapper describes to the model and runs itself, up to
25 rounds per message.

Our wrapper starts from Google's at our base (0.19.0 nightly, already ported to C API 1.0.0 upstream)
and carries our patches: a failed tool call answered with an error response, a stream that holds its
conversation weakly, streams kept by ID that end once and stop their native work when cancelled, a
tokenizer API and the embedding overflow option, `BuildInfo` and `Telemetry`, and Swift 6 language
mode. Agello vendors our build's wrapper sources with the binary of our release
`agello-one-20261115`; until 2026-10-28 it shipped Google's v0.18.0 wrapper and binary.

## Concepts

| Type | Kind | Role |
| --- | --- | --- |
| `Engine` | `actor` | Owns the native engine handle; `initialize()`, `createConversation(with:)`, `tokenize(_:)` (ours) |
| `EngineConfig` | `struct`, `Sendable` | Model path, backends, `maxNumTokens`, `cacheDir`, LoRA ranks, activation type |
| `Conversation` | `final class`, `Sendable` | One native conversation: `sendMessage`, `sendMessageStream`, `cancel`, token count, benchmark |
| `ConversationConfig` | `struct` | System message, initial messages, tools, sampler, thinking, response format switch, visual token budget, speculative decoding, chat template |
| `Message`, `Contents`, `Content`, `Role`, `ToolCall` | `struct`/`enum` | The JSON message model |
| `Tool`, `@ToolParam`, `ToolManager` | protocol, property wrapper, class | Tools the model can call |
| `ResponseFormat` | `struct` | JSON schema or regex constraint for one message |
| `ExperimentalFlags` | `struct` of static vars | Process-wide switches, opt-in |
| `LiteRTLMError` | `enum` | Every error the package throws |
| `ModelInfo`, `LLMCapability`, `EmbeddingCapability` | classes | Read a `.litertlm` file without an engine |
| `EmbeddingEngine`, `EmbeddingEngineConfig`, `EmbeddingOptions` | actor, structs | Embedding models |
| `BuildInfo`, `Telemetry` | enums (ours) | Build stamp and engine probes |
| `StreamContext`, `StreamTable` | classes (ours, internal) | One stream's state, and the table its native sends find it in by ID |
| `benchmark(...)`, `BenchmarkInfo` | function, struct | Google's benchmark helper |

### A minimal session

```swift
let config = try EngineConfig(
  modelPath: modelURL.path, backend: .gpu, maxNumTokens: 8192, cacheDir: cachesURL.path)
let engine = Engine(engineConfig: config)
try await engine.initialize()  // seconds; blocks the actor
let conversation = try await engine.createConversation(
  with: ConversationConfig(systemMessage: Message("Be brief.")))
for try await chunk in conversation.sendMessageStream(Message("Hello")) {
  print(chunk.toString, terminator: "")  // text; chunk.channels["thought"] holds thinking
}
```

### Messages

`Role` raw values are `system`, `user`, `assistant` (the Swift case is `.model`) and `tool`; content
parts serialize as `{"type": "text"}`, `{"type": "image"}` with a base64 `blob` or a `path`,
`{"type": "audio"}` likewise, and `{"type": "tool_response", "name", "response"}`
[`swift/Message.swift:19-24`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Message.swift#L19-L24) [`swift/Message.swift:49-72`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Message.swift#L49-L72). <!-- c:LM-SWIFT-051 --> A reply is parsed back with
`Conversation.jsonToMessage`, which accepts `assistant` or `model` as the model role and throws
`message(.invalidContent)` when the JSON has no content, channels or tool calls [`swift/Conversation.swift:861-872`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L861-L872).
Channels (for Gemma 4, `thought`) arrive in `Message.channels`.

### Configuration checks

`EngineConfig` rejects `maxNumTokens <= 0`; `SamplerConfig` rejects `topK <= 0`, `topP` outside
`[0, 1]` and a negative temperature; `ThinkingConfig()` means thinking on with an unlimited budget
(`-1`) [`swift/Config.swift:109-120`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Config.swift#L109-L120) [`swift/Config.swift:155-169`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Config.swift#L155-L169) [`swift/Config.swift:189-192`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Config.swift#L189-L192). <!-- c:LM-SWIFT-053 --> `ConversationConfig`
drops an empty system message and gives any other system message the `system` role
[`swift/Config.swift:251-285`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Config.swift#L251-L285).

## How it works

### Engine: initialization

```mermaid
sequenceDiagram
  participant App
  participant E as Engine (actor)
  participant C as C API
  App->>E: Engine(engineConfig:)
  App->>E: await initialize()
  E->>C: litert_lm_engine_settings_create(path, backend, vision, audio)
  E->>C: setters (max tokens, cache dir, activation type, LoRA, flags)
  E->>C: litert_lm_engine_create(settings)
  Note over E: blocks the actor for the whole load
  E->>C: litert_lm_engine_settings_delete (defer)
  E-->>App: returns; handle set
```

`Engine` is an `actor` whose only mutable state is the native handle; `initialize()` is a synchronous,
actor-isolated method, so it holds the actor (and the thread running it) for the whole load, and a
second call throws `engine(.alreadyInitialized)` [`swift/Engine.swift:32-49`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Engine.swift#L32-L49) [`swift/Engine.swift:63-65`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Engine.swift#L63-L65) [`swift/Engine.swift:73-77`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Engine.swift#L73-L77). <!-- c:LM-SWIFT-001 -->
It creates the settings from `backend.rawValue` (`"cpu"` or `"gpu"`) and the optional vision and audio
backends, applies the configuration, creates the engine and deletes the settings
[`swift/Engine.swift:84-91`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Engine.swift#L84-L91) [`swift/Config.swift:20-40`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Config.swift#L20-L40) [`swift/Engine.swift:185-192`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Engine.swift#L185-L192). <!-- c:LM-SWIFT-002 -->

Settings it applies, in order [`swift/Engine.swift:95-183`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Engine.swift#L95-L183):

| Source | C setter | When |
| --- | --- | --- |
| `maxNumTokens` | `litert_lm_engine_settings_set_max_num_tokens` | non-nil |
| `cacheDir` | `..._set_cache_dir` | non-nil (otherwise the model's folder) |
| `activationDataType` | `..._set_activation_data_type` | non-nil |
| `loraRank`, `audioLoraRank` | `..._set_lora_rank`, `..._set_supported_lora_ranks` (and audio) | non-nil; ranks only when > 0 |
| benchmark arguments or `ExperimentalFlags.enableBenchmark` | `..._enable_benchmark` (+ token counts) | benchmark run, or flag true |
| `ExperimentalFlags.enableSpeculativeDecoding` | `..._set_enable_speculative_decoding` | non-nil |
| `ExperimentalFlags.visualTokenBudget` | `..._set_max_vision_tokens_per_image` | non-nil |
| `ExperimentalFlags.gpuEnableMetalResidencySet` | `..._set_gpu_enable_metal_residency_set` | non-nil |
| `ExperimentalFlags.enableYnnpack` (Google's) | `..._set_enable_ynnpack` | non-nil |

<!-- c:LM-SWIFT-004 -->

`Backend.cpu(threadCount:)` carries a thread count, but `Engine.initialize` never calls
`litert_lm_engine_settings_set_num_threads`: the count is ignored for language models, and only
`EmbeddingEngine` applies it [`swift/Config.swift:20-40`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Config.swift#L20-L40) [`swift/Engine.swift:95-183`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Engine.swift#L95-L183) [`swift/EmbeddingEngine.swift:90-100`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/EmbeddingEngine.swift#L90-L100).
<!-- c:LM-SWIFT-003 -->

`deinit` deletes the native engine [`swift/Engine.swift:457-462`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Engine.swift#L457-L462). Each `Conversation` keeps a strong reference to
its `Engine`, so the native engine outlives every Swift conversation made from it, which is what the
C API's ownership rule requires [`swift/Conversation.swift:68-110`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L68-L110) [`swift/Engine.swift:394-400`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Engine.swift#L394-L400) [`c/engine.h:1111-1120`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/c/engine.h#L1111-L1120).
<!-- c:LM-SWIFT-005 -->

### Engine: creating a conversation

`createConversation(with:)` checks the system messages (a `systemMessage` plus an initial system
message, or two initial system messages, throw `config(.multipleSystemMessages)`)
[`swift/Engine.swift:209-220`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Engine.swift#L209-L220), builds a `ToolManager` from the tools, then translates the config to C objects that
it deletes right after `litert_lm_conversation_create`:

| `ConversationConfig` field or flag | C call | Note |
| --- | --- | --- |
| `samplerConfig` | `litert_lm_sampler_params_create(kLiteRtLmSamplerTypeTopP)` + top-k, top-p, temperature, seed | always type top-p; nil keeps the model's sampler |
| `loraPath`, `audioLoraPath` | `litert_lm_session_config_set_lora_path` (and audio) | |
| `enableSpeculativeDecoding` | `litert_lm_session_config_set_enable_speculative_decoding` | nil inherits the engine |
| `systemMessage` | `litert_lm_conversation_config_set_system_message` | the contents as a JSON array |
| `tools` | `..._set_tools` | `ToolManager.toolsJsonDescription` (`[]` with no tools) |
| `initialMessages` | `..._set_messages` | JSON array |
| `chatTemplate` | `..._set_prompt_template` | |
| `enableResponseFormat` | `..._set_constraint_provider(LlGuidance)` + `..._set_enable_constrained_decoding(true)` | |
| `ExperimentalFlags.enableConversationConstrainedDecoding` | `..._set_enable_constrained_decoding` | only when `enableResponseFormat` is false |
| `enableToolCallStreaming` and the streaming flags | `..._set_stream_tool_calls` | both the field and the flag must be true |
| `ExperimentalFlags.filterChannelContentFromKvCache` | `..._set_filter_channel_content_from_kv_cache` | non-nil only; the C++ default is true |
| `thinkingConfig` | `litert_lm_thinking_config_*`, `..._set_thinking_config` | |

When a `SamplerConfig` is given, the sampler type is always `kLiteRtLmSamplerTypeTopP`
[`swift/Engine.swift:250-274`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Engine.swift#L250-L274). <!-- c:LM-SWIFT-007 --> With `enableResponseFormat`, the conversation gets the
llguidance constraint provider and constrained decoding; otherwise constrained decoding follows
`ExperimentalFlags.enableConversationConstrainedDecoding` (default `false`), and tool-call streaming
needs both `enableToolCallStreaming` and the experimental flag [`swift/Engine.swift:336-359`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Engine.swift#L336-L359) [`swift/ExperimentalFlags.swift:27-39`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ExperimentalFlags.swift#L27-L39) [`swift/ExperimentalFlags.swift:110-133`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ExperimentalFlags.swift#L110-L133).
<!-- c:LM-SWIFT-008 --> The channel filter flag is passed only when set, and the C++ builder's own
default is to filter [`swift/Engine.swift:360-366`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Engine.swift#L360-L366) [`runtime/conversation/conversation.h:273-283`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.h#L273-L283). Which constraint provider and grammar
engine a JSON format reaches in our build and in the release Agello ships is in
[conversation.md](conversation.md).

### Conversation: threads and lifetime

`Conversation` is a `final class` declared `Sendable`: every stored property is a `let` (the logger,
the handle, the `ToolManager`, the automatic-tool flag, the engine, the response-format switch, the
visual token budget, and `streamOwner`, its key in the stream table), there is no lock and no actor of
its own, and the handle is declared `nonisolated(unsafe)` for Swift 6 language mode (ours:
`24f35f81`). `deinit` calls `litert_lm_conversation_delete`, then ends any of its streams still
registered with `conversation(.notAlive)` [`swift/Conversation.swift:68-110`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L68-L110). <!-- c:LM-SWIFT-010 --> Two consequences:

- Nothing in Swift serializes calls on one conversation; ordering is whatever the native engine
  does with concurrent tasks (see [engine-and-sessions.md](engine-and-sessions.md)). Agello
  serializes all engine work in its own actor.
- `deinit` runs on whichever thread drops the last reference, and deleting a busy native
  conversation waits for its work (up to 10 minutes; see
  [c-api.md](c-api.md#deleting-while-work-runs)).

### Conversation: `sendMessage`

`sendMessage` is `async` only in name: it calls the blocking `litert_lm_conversation_send_message`
on the thread running the task and holds it for the whole reply [`swift/Conversation.swift:218-224`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L218-L224) [`c/conversation.h:496-497`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/c/conversation.h#L496-L497).
<!-- c:LM-SWIFT-011 --> On a Swift concurrency thread pool, that is one cooperative thread taken for
seconds.

The automatic tool loop makes at most 25 native sends (`recurringToolCallLimit = 25`). A reply
without `tool_calls` is returned if it has content or channels, and otherwise throws
`conversation(.invalidResponse)` [`swift/Conversation.swift:167-172`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L167-L172). A reply with `tool_calls` is
returned as is when `automaticToolCalling` is false; otherwise its tools run and their responses
are sent as the next message. The thinking config is passed on the first send only, and if the
25th reply still has tool calls, the loop throws
`conversation(.recurringToolCallLimitExceeded(limit: 25))` [`swift/Conversation.swift:136-181`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L136-L181) [`swift/Conversation.swift:48`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L48).
<!-- c:LM-SWIFT-012 -->

### Conversation: `sendMessageStream`

```mermaid
sequenceDiagram
  participant T as Caller task
  participant C as Conversation
  participant N as C API / engine
  participant X as execution thread
  participant K as callback thread
  T->>C: sendMessageStream(message)
  C->>C: StreamContext(continuation, weak conversation), onTermination
  C->>N: send_message_stream(json, streamCallback, ID registered in StreamTable)
  C-->>T: AsyncThrowingStream
  loop decoding
    X->>X: streamCallback(partial JSON chunk), context found by ID
    X-->>T: continuation.yield(Message) if content or channels
  end
  K->>K: streamCallback(unflushed text, then final chunk), ID retired
  alt pending tool calls, under maxToolRounds
    K->>T: Task { run tools; sendToStream(tool responses) under a new ID }
  else no tool calls
    K-->>T: finish(), once
  end
```

`sendMessageStream` returns an `AsyncThrowingStream<Message, Error>` whose build closure starts the
native stream at once and installs an `onTermination` handler: when the task iterating the stream is
cancelled before the stream ended, the handler marks the stream cancelled and calls `cancel()` once,
on a global dispatch queue, which stops the native work and leaves the conversation cancelled, as
`cancel()` does (ours: `4ca2cf33`) [`swift/Conversation.swift:493-552`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L493-L552) [`swift/ConversationStream.swift:176-196`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ConversationStream.swift#L176-L196).
<!-- c:LM-SWIFT-013 --> Two of its parameters are ours: `followUpMaxOutputTokens`, the output cap of
each round after a tool response, and `maxToolRounds` (25 when nil). `sendToStream` serializes the
message, builds the per-turn optional args, registers the stream's `StreamContext` in `StreamTable`
under a new integer ID and passes that ID as `callback_data`; if the stream fails to start, it
retires the ID and throws `conversation(.failedToStartStream(status:message:))`, and if the stream was
cancelled while the send was being set up, it cancels the native work [`swift/Conversation.swift:588-644`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L588-L644). <!-- c:LM-SWIFT-014 -->

`streamCallback` is the C callback, so it runs on LiteRT-LM's threads: the execution thread for
chunks streamed during decoding (`Tasks::Decode` receives the callback), and the callback thread for
the task's final callback, which first delivers any text not yet flushed (`SendCompleteMessage`) and
then the final chunk [`runtime/core/tasks.cc:808-815`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/core/tasks.cc#L808-L815) [`runtime/conversation/internal_callback_util.cc:95-149`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L95-L149) [`swift/ConversationStream.swift:331-336`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ConversationStream.swift#L331-L336) [`runtime/framework/resource_management/threaded_execution_manager.cc:1127-1133`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/framework/resource_management/threaded_execution_manager.cc#L1127-L1133)
[`runtime/framework/resource_management/threaded_execution_manager.cc:529-541`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/framework/resource_management/threaded_execution_manager.cc#L529-L541). <!-- c:LM-SWIFT-015 --> For each chunk it copies the final flag, the error and
the text, and `handleStreamChunk` finds the stream's context by ID (a retired ID finds nothing, and
the chunk is dropped); an error retires the ID and finishes the stream with
`conversation(.invalidResponse(<error text>))`; text is parsed as JSON, `tool_calls` are added to the
context's pending calls, and only chunks with `content` or `channels` are yielded [`swift/ConversationStream.swift:273-304`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ConversationStream.swift#L273-L304) [`swift/ConversationStream.swift:331-336`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ConversationStream.swift#L331-L336).
<!-- c:LM-SWIFT-016 --> A chunk that carries only tool calls is not yielded, so the consumer sees
tool calls only when they share a chunk with content (Agello reports the tools that ran from its own
tool code instead).

On the final chunk the handler retires the send's ID. A cancelled stream finishes with Swift's
`CancellationError`; with no pending tool calls the stream finishes; with pending calls and fewer
rounds than `maxToolRounds`, it starts a Swift `Task` that runs the tools and sends their responses as
the next round under a new ID, and past the limit it finishes with
`conversation(.recurringToolCallLimitExceeded(limit:))` (ours: `4ca2cf33`) [`swift/ConversationStream.swift:306-329`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ConversationStream.swift#L306-L329) [`swift/ConversationStream.swift:208-218`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ConversationStream.swift#L208-L218).
This path never looks at `automaticToolCalling`, so a streamed reply runs the conversation's tools
even when automatic tool calling is off [`swift/Conversation.swift:495-570`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L495-L570). <!-- c:LM-SWIFT-017 --> A tool round
passes the stream's repetition, n-gram and suppressed-token settings, its `followUpMaxOutputTokens`
(or `maxOutputTokens` when that is nil) and its response format, but not the extra context or the
thinking config [`swift/Conversation.swift:554-570`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L554-L570) [`swift/ConversationStream.swift:78-90`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ConversationStream.swift#L78-L90).

### Error chunks and the retained context

Until 2026-10-16 the wrapper retained a `StreamContext` for each send and released it on the first
final chunk and on every error chunk, as upstream's did at our base [`swift/Conversation.swift:819-848`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Conversation.swift#L819-L848), and
LiteRT-LM can report a send's end more than once: with tools declared, a Gemma 4 tool-call block that
the FC parser rejects is reported as an error at least twice, and decoding goes on in between.

1. When the block's end fence streams, the internal callback's `SendMessage` hands the failed
   `ToMessage` result to the user callback: `InvalidArgument`, because `return_error_on_parse_failure`
   defaults to true and neither C nor Swift can change it.
2. At the end of decode, `SendCompleteMessage` parses the whole reply again and fails on the same
   block.

A second report reached a context the first had released. Two changes of ours close that path, in
our build and in the release Agello ships. The C callback passes on only the first final chunk of a
send and drops the rest (ours: `8acc748c`; [c-api.md](c-api.md#threads-and-callbacks)).
<!-- c:X-GAP-015 --> And each native send finds its `StreamContext` through `StreamTable` by an
integer ID passed as `callback_data`, not a retained pointer: an error chunk or the final chunk
retires the ID, the context finishes the stream once (later finishes do nothing), and a chunk for a
retired ID returns without touching anything (ours: `4ca2cf33`) [`swift/ConversationStream.swift:101-159`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ConversationStream.swift#L101-L159) [`swift/ConversationStream.swift:221-255`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ConversationStream.swift#L221-L255)
[`swift/ConversationStream.swift:273-336`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ConversationStream.swift#L273-L336) [`swift/Conversation.swift:623-639`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L623-L639); the release: [`swift/ConversationStream.swift:273-336`](https://github.com/PowerBeef/agello-litert-lm/blob/6da75396ece779d3c7b6b605ec8c2c06861ce1eb/swift/ConversationStream.swift#L273-L336)
[`swift/Conversation.swift:620-636`](https://github.com/PowerBeef/agello-litert-lm/blob/6da75396ece779d3c7b6b605ec8c2c06861ce1eb/swift/Conversation.swift#L620-L636). <!-- c:X-GAP-016 -->

Sources for the native side: [`runtime/conversation/internal_callback_util.cc:61-71`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L61-L71),
[`runtime/conversation/internal_callback_util.cc:447-466`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L447-L466),
[`runtime/conversation/internal_callback_util.cc:330-343`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L330-L343),
[`runtime/conversation/internal_callback_util.cc:95-149`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L95-L149),
[`runtime/conversation/model_data_processor/gemma4_data_processor.cc:146-155`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/model_data_processor/gemma4_data_processor.cc#L146-L155),
[`runtime/conversation/model_data_processor/data_utils.cc:86-108`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/data_utils.cc#L86-L108),
[`runtime/components/tool_use/parser_utils.cc:138-151`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/components/tool_use/parser_utils.cc#L138-L151),
[`runtime/conversation/model_data_processor/model_data_processor.h:123-133`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/model_data_processor/model_data_processor.h#L123-L133).
Agello streams its chat replies on conversations that declare its tools whenever tools or web search
are on, so a malformed tool call reaches this path in the app; with the release Agello ships, the
stream ends once, with the first error. <!-- c:MOD-TOK-053 --> <!-- c:X-GAP-025 -->

The other terminal paths of the conversation layer deliver one terminal event: `SendCompleteMessage`
returns after reporting an error without sending the end marker, each error, cancel and failure branch
of the internal callback makes one call and returns, and in our build the deferred end marker is
suppressed when the channel-filter refill reports an error itself
([`runtime/conversation/internal_callback_util.cc:292-343`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L292-L343),
[`runtime/conversation/conversation.cc:749-770`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/conversation/conversation.cc#L749-L770)).
A non-final chunk whose text isn't valid JSON or doesn't convert to a `Message` retires the send's
ID, finishes the stream with the error and cancels the native work; later chunks of that send are
dropped ([`swift/ConversationStream.swift:273-304`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ConversationStream.swift#L273-L304)).

### Our fix: a stream holds its conversation weakly

Upstream's `StreamContext` held `let conversation: Conversation` [`swift/Conversation.swift:781-784`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Conversation.swift#L781-L784). Ours holds
`weak var conversation: Conversation?` (ours: `0b2ca12fc4`), now in `swift/ConversationStream.swift`
since streams are kept by ID (ours: `4ca2cf33`); the release Agello ships has the same
[`swift/ConversationStream.swift:109-117`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ConversationStream.swift#L109-L117) [`swift/ConversationStream.swift:109-117`](https://github.com/PowerBeef/agello-litert-lm/blob/6da75396ece779d3c7b6b605ec8c2c06861ce1eb/swift/ConversationStream.swift#L109-L117). <!-- c:LM-SWIFT-019 -->

Why it matters, step by step:

1. The engine marks a task finished only after its final callback returns: the callback-thread
   closure calls the callback, then `UpdateTaskState` [`runtime/framework/resource_management/threaded_execution_manager.cc:529-541`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/framework/resource_management/threaded_execution_manager.cc#L529-L541) [`runtime/framework/resource_management/threaded_execution_manager.cc:529-541`](https://github.com/PowerBeef/agello-litert-lm/blob/6da75396ece779d3c7b6b605ec8c2c06861ce1eb/runtime/framework/resource_management/threaded_execution_manager.cc#L529-L541).
2. Upstream's `streamCallback` released its context at the end of the final callback [`swift/Conversation.swift:876-913`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Conversation.swift#L876-L913).
3. With a strong reference, if the app had already dropped the conversation (a stream abandoned
   early, or a conversation replaced while its cancelled stream was finishing), that release was the
   last one: `Conversation.deinit` ran on the callback thread and called
   `litert_lm_conversation_delete`.
4. Deleting the conversation destroys its session, and `SessionAdvanced`'s destructor waits for the
   session's tasks (`Engine::kDefaultTimeout`, 10 minutes) without cancelling them
   [`runtime/core/session_advanced.cc:484-489`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/runtime/core/session_advanced.cc#L484-L489) [`runtime/core/session_advanced.cc:484-491`](https://github.com/PowerBeef/agello-litert-lm/blob/6da75396ece779d3c7b6b605ec8c2c06861ce1eb/runtime/core/session_advanced.cc#L484-L491) [`runtime/engine/engine.h:453`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/engine.h#L453). The task it waits for is the one whose callback is
   still running, so the wait could only end at the timeout, and the engine's single callback thread
   was blocked the whole time.

<!-- c:LM-SWIFT-020 -->

With the weak reference the conversation is deleted where the app drops it, and that deletion waits
for the stream's work to finish. If the conversation is gone when a tool round is due, the round's
`Task` finishes the stream with `conversation(.notAlive)`, and a conversation's `deinit` ends its
streams still registered the same way [`swift/Conversation.swift:517-525`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L517-L525) [`swift/ConversationStream.swift:319-328`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ConversationStream.swift#L319-L328) [`swift/Conversation.swift:102-110`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L102-L110).
<!-- c:LM-SWIFT-021 --> Dropping a conversation in the middle of a stream still blocks the dropping
thread until the generation ends; call `cancel()` first. Agello's device probe 9 checks the fix
(Agello `Vendor/README.md`).

### Cancellation

`cancel()` marks the conversation's registered streams cancelled and calls
`litert_lm_conversation_cancel_process` [`swift/Conversation.swift:646-655`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L646-L655). A stream cancelled during generation
then ends with an error chunk whose text is `CANCELLED: Task cancelled`, which the wrapper throws as
`conversation(.invalidResponse("CANCELLED: Task cancelled"))`; a stream cancelled between tool rounds,
or whose final chunk arrives without an error after the cancel, ends with Swift's `CancellationError`
instead (ours: `4ca2cf33`). Upstream documents that a session is not reusable after a cancel
[`runtime/conversation/internal_callback_util.cc:302-308`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/conversation/internal_callback_util.cc#L302-L308) [`c/conversation.cc:45-86`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/c/conversation.cc#L45-L86) [`swift/ConversationStream.swift:273-311`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ConversationStream.swift#L273-L311) [`swift/Conversation.swift:554-560`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L554-L560) [`runtime/core/session_advanced.h:192-207`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/core/session_advanced.h#L192-L207). <!-- c:LM-SWIFT-022 --> Agello therefore
treats every cancelled conversation as spent and reseeds a new one from saved history (Agello
`ChatEngine.swift:95-99`).

### Tools

A tool is a `Decodable` type with a static `name` and `description`, an empty `init()` and
`run() async throws -> Any`; its parameters are properties wrapped in `@ToolParam(description:)`,
of type `String`, `Int`, `Bool`, `Double` or `Float` (JSON `number`), an `Optional` of one
(`nullable`), or an `Array` of one [`swift/Tool.swift:211-221`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Tool.swift#L211-L221) [`swift/Tool.swift:18-49`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Tool.swift#L18-L49) [`swift/Tool.swift:84-101`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Tool.swift#L84-L101).
<!-- c:LM-SWIFT-025 --> A parameter is required unless it is optional or has a default; reading a
required parameter that was never set is a `fatalError` [`swift/Tool.swift:84-101`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Tool.swift#L84-L101) [`swift/Tool.swift:107-121`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Tool.swift#L107-L121).

**Schema.** `getSchema()` reflects the tool with `Mirror` and emits
`{"type": "function", "function": {"name", "description"}}`; when the tool has at least one
`@ToolParam`, it adds `"parameters": {"type": "object", "properties"}`, with a `"required"` list
when any parameter is required. With `ExperimentalFlags.convertCamelToSnakeCaseInToolDescription` (default `true`), tool and
parameter names are converted from camelCase to snake_case, after dropping the property wrapper's
leading underscore [`swift/Tool.swift:225-273`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Tool.swift#L225-L273) [`swift/ExperimentalFlags.swift:88-108`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ExperimentalFlags.swift#L88-L108). <!-- c:LM-SWIFT-027 --> The conversation passes the
array of schemas to the C API as its tools, and the model's chat template renders them (see
[conversation.md](conversation.md)).

**Execution.** `ToolManager.execute(name:arguments:)` looks the name up in the registry built at
conversation creation, makes a default instance to collect default values (and `null` for optional
parameters), merges the model's arguments over them, decodes a new instance with `JSONDecoder` and
runs it; the result is normalized for JSON (dictionaries and arrays recursively, strings, numbers and
booleans as they are, `Void` as `""`, anything else through `String(describing:)`)
[`swift/ToolManager.swift:68-122`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ToolManager.swift#L68-L122) [`swift/ToolManager.swift:128-155`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ToolManager.swift#L128-L155). The snake_case flag is read again on every execution, for the default
keys and the decoder's key strategy, although its documentation says it is read only when a
conversation is created; the registry's names, on the other hand, are fixed at creation
[`swift/ToolManager.swift:68-122`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ToolManager.swift#L68-L122) [`swift/ToolManager.swift:36-60`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ToolManager.swift#L36-L60) [`swift/ExperimentalFlags.swift:88-108`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ExperimentalFlags.swift#L88-L108). <!-- c:LM-SWIFT-028 --> Changing the flag while
conversations are alive can therefore make arguments fail to decode.

**Our patch: failed calls answered with an error.** Upstream's `handleToolCalls` skipped calls whose
arguments weren't a JSON object and turned any error from a tool into
`conversation(.toolExecutionError(name:error:))`, which ended the whole reply [`swift/Conversation.swift:386-409`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Conversation.swift#L386-L409).
Ours (ours: `f3408e0671`), in our build and in the release Agello ships, answers every call
[`swift/Conversation.swift:405-437`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L405-L437) [`swift/Conversation.swift:439-468`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L439-L468) [`swift/Conversation.swift:402-434`](https://github.com/PowerBeef/agello-litert-lm/blob/6da75396ece779d3c7b6b605ec8c2c06861ce1eb/swift/Conversation.swift#L402-L434): <!-- c:LM-SWIFT-023 -->

| The model's call | Tool response sent back |
| --- | --- |
| `arguments` missing or `null` | the tool runs with no arguments (`{}`) |
| `arguments` an object, or a JSON string holding an object | the tool runs |
| `arguments` anything else | `{"error": "The arguments for '<name>' must be a JSON object."}` |
| an unknown tool name (`tool(.notFound)`) | `{"error": "There is no tool named '<name>'. Use only the tools you were given, or answer without one."}` |
| arguments that don't decode (`DecodingError`) | `{"error": "The arguments for '<name>' don't match its parameters. Check their names and types, then try again."}` |
| the tool throws anything else | `{"error": "'<name>' failed: <localizedDescription>"}` |

The model reads the error in the tool response and can correct the call or answer without it. The
`toolExecutionError` case still exists in `LiteRTLMError`, but nothing throws it any more
[`swift/LiteRTLMError.swift:327-338`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/LiteRTLMError.swift#L327-L338). <!-- c:LM-SWIFT-024 --> Agello's device probe 15 checks this (Agello
`Vendor/README.md`).

### Response formats

`ResponseFormat.json(schema:)` takes a JSON schema as a string (checked to parse as JSON) or a
dictionary; `ResponseFormat.regex(pattern:)` takes a pattern; they map to
`kLiteRtLmConstraintTypeJsonSchema` and `kLiteRtLmConstraintTypeRegex` [`swift/ResponseFormat.swift:20-72`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ResponseFormat.swift#L20-L72). Passing
one to a conversation created without `enableResponseFormat: true` throws
`conversation(.responseFormatNotEnabled)` [`swift/Conversation.swift:136-181`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L136-L181) [`swift/Conversation.swift:495-552`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L495-L552). <!-- c:LM-SWIFT-030 -->
The format is not applied to a user message when automatic tool calling is on and the conversation
has tools: then it applies only to the tool-response message, so the final answer after the tools
is the constrained one [`swift/Conversation.swift:112-122`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L112-L122). <!-- c:LM-SWIFT-031 -->

### `ExperimentalFlags`

`ExperimentalFlags` is a struct of static properties over one process-wide store, shared by every
engine and conversation in the process. In our build the store sits behind an `NSLock` (ours:
`82b05ea1`, which replaced a `Mutex` that crashed optimized builds); the release's own wrapper at
`6da75396` keeps Google's unsynchronized `static var`s. Each setter ignores the new value and logs an
error unless `optIntoExperimentalAPIs()` was called first, and the opt-in can't be undone
[`swift/ExperimentalFlags.swift:24-81`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ExperimentalFlags.swift#L24-L81) [`swift/ExperimentalFlags.swift:21-46`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ExperimentalFlags.swift#L21-L46). <!-- c:LM-SWIFT-032 -->

| Flag | Default | Read when | Effect |
| --- | --- | --- | --- |
| `enableBenchmark` | `false` | `Engine.initialize`; every `getBenchmarkInfo()` | benchmark info collected; `getBenchmarkInfo` throws `benchmarkNotEnabled` while false |
| `convertCamelToSnakeCaseInToolDescription` | `true` | conversation creation and every tool execution | snake_case tool and parameter names |
| `enableConversationConstrainedDecoding` | `false` | conversation creation | constrained decoding without a response format |
| `enableConversationToolCallStreaming` | `false` | conversation creation | with the config field, streams tool-call tokens |
| `conversationToolCallStreamingChannelName` | `"tool_call"` | conversation creation | channel for streamed tool calls |
| `enableSpeculativeDecoding` | `nil` | `Engine.initialize` | engine-level MTP on or off |
| `visualTokenBudget` | `nil` | `Engine.initialize`; every native send whose conversation has no budget of its own | maximum vision tokens per image (engine); per-message budget |
| `filterChannelContentFromKvCache` | `nil` | conversation creation | keeps thinking out of the KV cache when true (C++ default true) |
| `gpuEnableMetalResidencySet` | `nil` | `Engine.initialize` | `MTLResidencySet` for the GPU backend |
| `enableYnnpack` | `nil` (or `true` from the environment) | `Engine.initialize` | YNNPACK before XNNPACK on CPU |

[`swift/ExperimentalFlags.swift:27-39`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ExperimentalFlags.swift#L27-L39) [`swift/ExperimentalFlags.swift:83-206`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ExperimentalFlags.swift#L83-L206)
[`swift/Engine.swift:147-183`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Engine.swift#L147-L183) [`swift/Conversation.swift:716-721`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L716-L721) [`swift/Conversation.swift:276-281`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L276-L281)

`visualTokenBudget` is read at two levels: `Engine.initialize` turns it into the engine's maximum
vision tokens per image, and every native send, in `makeOptionalArgs`, uses
`ConversationConfig.visualTokenBudget` or, if that is nil, the flag's value at send time; the release
Agello ships reads it the same way [`swift/Engine.swift:168-172`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Engine.swift#L168-L172) [`swift/Conversation.swift:276-281`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L276-L281) [`swift/Conversation.swift:273-278`](https://github.com/PowerBeef/agello-litert-lm/blob/6da75396ece779d3c7b6b605ec8c2c06861ce1eb/swift/Conversation.swift#L273-L278).
<!-- c:LM-SWIFT-033 --> `enableYnnpack` is Google's: it is in the upstream wrapper at our base
(`df6df5b8`), so our build and the release Agello ships have it. When unset, its getter returns
`true`, without any opt-in, if the environment variable `ENABLE_YNNPACK` or `enable_ynnpack` is
`"true"` or a launch argument contains `enable_ynnpack=true`, and `nil` otherwise [`swift/ExperimentalFlags.swift:181-206`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ExperimentalFlags.swift#L181-L206) [`swift/ExperimentalFlags.swift:205-238`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ExperimentalFlags.swift#L205-L238).
<!-- c:LM-SWIFT-034 --> Google's `benchmark(...)` function opts the process in and sets
`enableBenchmark` for its duration [`swift/Benchmark.swift:63-75`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Benchmark.swift#L63-L75). <!-- c:LM-SWIFT-035 -->

Because the flags are global, an app that runs two engines (Agello's chat engine and embedder) or
changes a flag between conversations must set every flag immediately before the call that reads it.
`Engine.updateGPUEnableMetalResidencySet(_:)` is the one engine method gated on the opt-in
[`swift/Engine.swift:411-426`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Engine.swift#L411-L426).

### Errors

`LiteRTLMError` has seven top-level cases, `engine`, `embeddingEngine`, `conversation`, `config`,
`tool`, `message` and `modelInfo`, each wrapping a nested enum [`swift/LiteRTLMError.swift:19-27`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/LiteRTLMError.swift#L19-L27).
<!-- c:LM-SWIFT-036 --> Every C call goes through three helpers: `check` turns a
non-OK status into the caller's error with the text `"<function> failed with status <code>: <last
error message>"`, reading and clearing the thread-local message right after the call on the same
thread; `create` also rejects a NULL handle; `get` returns an out-parameter
[`swift/LiteRTLMError.swift:84-91`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/LiteRTLMError.swift#L84-L91) [`swift/LiteRTLMError.swift:105-115`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/LiteRTLMError.swift#L105-L115) [`swift/LiteRTLMError.swift:62-71`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/LiteRTLMError.swift#L62-L71). <!-- c:LM-SWIFT-037 --> The numeric status
survives only inside that string, except in `failedToStartStream(status:message:)`
[`swift/Conversation.swift:588-644`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L588-L644). Equality between two `EngineError`s with detail strings is true when the strings
match or either is empty [`swift/LiteRTLMError.swift:226-255`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/LiteRTLMError.swift#L226-L255).

| Case | Thrown by |
| --- | --- |
| `engine(.alreadyInitialized / .notInitialized)` | `initialize()` twice; `createConversation` before `initialize` |
| `engine(.failedToCreateSettings / .failedToCreateEngine / ...)` | a C call during `initialize` or `createConversation`, with the C message |
| `config(.multipleSystemMessages / .invalidMaxNumTokens / .invalidTopK / ...)` | config checks in Swift |
| `conversation(.invalidResponse(...))` | a C failure during a send, an error chunk (including `CANCELLED: ...`), a reply without content |
| `conversation(.recurringToolCallLimitExceeded(limit:))` | the 25th reply of a `sendMessage` tool loop still has tool calls; a stream past `maxToolRounds` |
| `conversation(.notAlive)` | a call on a conversation without a handle; a stream whose conversation is gone at a tool round, or was deleted |
| Swift's `CancellationError` | a stream cancelled between tool rounds, or whose final chunk arrives without an error after a cancel |
| `conversation(.failedToStartStream(status:message:))` | `send_message_stream` refused to start |
| `conversation(.responseFormatNotEnabled)` | a `ResponseFormat` without `enableResponseFormat` |
| `conversation(.benchmarkNotEnabled / .benchmarkInfoUnavailable)` | `getBenchmarkInfo()` |
| `tool(.notFound)` | `ToolManager.execute` (answered to the model by our patch) |
| `message(.invalidContent / .failedToConvertToJson)` | `jsonToMessage` |
| `modelInfo(.failedToLoadModel)` | `ModelInfo(throwingModelPath:)` |

### `ModelInfo`

`ModelInfo(modelPath:)` (failable) and `ModelInfo(throwingModelPath:)` open the file with
`litert_lm_loaded_file_create` [`swift/ModelInfo.swift:298-312`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ModelInfo.swift#L298-L312). The accessors don't throw:
`modelInfoValue` turns a failed C call or a `NotFound` into a fallback (`maxContextTokens()` 0,
`maxVisionTokenBudget()` -1, `false` for capabilities, `nil` for arrays and strings), and
`fetchIntArray` returns `nil` when the count query fails, in our build and in the release Agello
ships [`swift/ModelInfo.swift:78-93`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ModelInfo.swift#L78-L93) [`swift/ModelInfo.swift:367-370`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ModelInfo.swift#L367-L370) [`swift/ModelInfo.swift:95-117`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ModelInfo.swift#L95-L117). Agello reads it before every load to intersect its plan with the file's
real capabilities (MTP, thinking, function calling, modalities, context and visual-token limits).

### `EmbeddingEngine`

An `actor` like `Engine`; `initialize()` creates the native embedding engine (applying
`Backend.cpu(threadCount:)` when the count is positive), `computeEmbedding` and
`computeEmbeddingBatch` run synchronously on the actor, and `close()` or `deinit` delete the native
engine [`swift/EmbeddingEngine.swift:34-58`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/EmbeddingEngine.swift#L34-L58) [`swift/EmbeddingEngine.swift:90-100`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/EmbeddingEngine.swift#L90-L100) [`swift/EmbeddingEngine.swift:300-306`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/EmbeddingEngine.swift#L300-L306). <!-- c:LM-SWIFT-041 -->
The text encoder's activation type is `EmbeddingEngineConfig.activationDataType`
(ours: `1a90cdb4`, not in the release's own wrapper). `EmbeddingOptions` exposes `normalize`,
`insertSpecialTokens`, `outputSize`, `visionTokensPerImage` and `inputOverflowStrategy`
(`chunkAndAverage`, `truncate` or `error`; ours: `4ca2cf33`), which it passes to
`litert_lm_embedding_options_set_input_overflow_strategy` when set; left nil, a call gets the C++
default, and an input longer than the longest signature is an error [`swift/EmbeddingEngineConfig.swift:71-122`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/EmbeddingEngineConfig.swift#L71-L122) [`swift/EmbeddingEngine.swift:346-351`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/EmbeddingEngine.swift#L346-L351) [`runtime/engine/embedding_engine.h:31-52`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/runtime/engine/embedding_engine.h#L31-L52).
<!-- c:LM-SWIFT-042 -->

### `BuildInfo` and `Telemetry` (ours)

Both are in our build and in the release Agello ships.

- `BuildInfo` (ours: `c7c744ebc6`): `json` returns `litert_lm_get_build_info()`, `fields` decodes it
  as `[String: String]`, and `id` is its `build_id` (12 hex digits for `--config=agello` builds,
  `dev` otherwise) [`swift/BuildInfo.swift:18-30`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/BuildInfo.swift#L18-L30) [`c/build_info.h:29-33`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/c/build_info.h#L29-L33). <!-- c:LM-SWIFT-043 -->
- `Telemetry` (ours: `4aed9274a4`, `a3a0c0adeb`): `Level` from `off` (0) to `deep` (4), a `level`
  property over `litert_lm_telemetry_get_level`/`_set_level`, `snapshotJSON()` (which frees the C
  string), `snapshot()`, `reset()` and `writeTrace(to:)` [`swift/Telemetry.swift:18-62`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Telemetry.swift#L18-L62). <!-- c:LM-SWIFT-044 -->
  Levels and environment variables are in [c-api.md](c-api.md#our-additions-telemetry-and-build-info).

### The Foundation Models adapter (`apple_fm`)

`swift/apple_fm/` is a second library product, `LiteRTLMFoundationModels`, that makes a LiteRT-LM
model a backend for Apple's Foundation Models framework [`Package.swift:61-71`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/Package.swift#L61-L71). It compiles only when
`FoundationModels` can be imported and the compiler is 6.4 or newer, and `LiteRTLanguageModel` is
available from iOS 27 and macOS 27. Each turn, its executor builds a fresh `Conversation` from the
whole Foundation Models transcript (system text, history, the last prompt or tool output); one
`Engine` per configuration is shared process-wide by an engine cache
[`swift/apple_fm/LiteRTLanguageModel.swift:31-39`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/apple_fm/LiteRTLanguageModel.swift#L31-L39) [`swift/apple_fm/LiteRTLanguageModel.swift:51-52`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/apple_fm/LiteRTLanguageModel.swift#L51-L52) [`swift/apple_fm/LiteRTLanguageModel.swift:163-181`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/apple_fm/LiteRTLanguageModel.swift#L163-L181) [`swift/apple_fm/LiteRTLanguageModel.swift:146-152`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/apple_fm/LiteRTLanguageModel.swift#L146-L152). <!-- c:LM-SWIFT-049 --> Tools are described
in the prompt and detected in the text (a JSON object with a `tool_call` key), not through
LiteRT-LM's tool calling; guided generation adds the schema to the prompt and constrains decoding
with it.

When the Swift API inventory was last generated from our build, the `LiteRTLMFoundationModels` target didn't
compile against the macOS 27 SDK, so the inventory lists only `LiteRTLM`
([`generated/swift-api-inventory.md`](../generated/swift-api-inventory.md)). <!-- c:LM-SWIFT-050 -->
The compiler errors were not recorded; the protocol names the adapter uses (`LanguageModel`,
`LanguageModelExecutor`, `LanguageModelExecutorGenerationChannel`) do exist in the macOS 27 SDK's
`FoundationModels` interface, so the failure is in details not checked here. Agello doesn't use
the adapter.

## Shipped wrapper and agello-next

Agello vendors our build's wrapper with the binary of our release:

| | The release's commit (`6da75396`) | Our build (`b7b856a0`) | What Agello vendors (`a6912128`) |
| --- | --- | --- | --- |
| Swift sources | ours on Google's 0.19.0-nightly.20261013 wrapper | the same, plus the three commits below | our build's `swift/`, byte for byte |
| Binary | `dist/CLiteRTLM.xcframework`, one framework with iPhone and Mac slices (ours: `3b2327cfcd`) [`Package.swift:36-41`](https://github.com/PowerBeef/agello-litert-lm/blob/6da75396ece779d3c7b6b605ec8c2c06861ce1eb/Package.swift#L36-L41) | the same [`Package.swift:37-42`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/Package.swift#L37-L42) | the release's `CLiteRTLM.xcframework.zip`, by URL and checksum |
| `swift-tools-version` | 5.9 [`Package.swift:1`](https://github.com/PowerBeef/agello-litert-lm/blob/6da75396ece779d3c7b6b605ec8c2c06861ce1eb/Package.swift#L1) | 6.0, Swift 6 language mode (ours: `24f35f81`) [`Package.swift:1-2`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/Package.swift#L1-L2) | 6.0 |
| `ExperimentalFlags` | Google's `static var`s [`swift/ExperimentalFlags.swift:21-46`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ExperimentalFlags.swift#L21-L46) | behind an `NSLock` (ours: `82b05ea1`) | as our build |
| `EmbeddingEngineConfig.activationDataType` | absent [`swift/EmbeddingEngineConfig.swift:20-60`](https://github.com/PowerBeef/agello-litert-lm/blob/6da75396ece779d3c7b6b605ec8c2c06861ce1eb/swift/EmbeddingEngineConfig.swift#L20-L60) | ours: `1a90cdb4` | as our build |

<!-- c:LM-SWIFT-045 -->

Both wrappers carry the tool-call error responses (ours: `f3408e0671`), the weak
`StreamContext.conversation` (ours: `0b2ca12fc4`), streams kept by ID with bounded tool rounds, the
tokenizer API and the embedding overflow option (ours: `4ca2cf33`), and `BuildInfo` and `Telemetry`,
and both speak C API 1.0.0: `c/` is the same at the two commits. Our build's wrapper differs from
Google's at our base in `swift/BUILD`, `BuildInfo.swift`, `Conversation.swift`,
`ConversationStream.swift`, `EmbeddingEngine.swift`, `EmbeddingEngineConfig.swift`, `Engine.swift`,
`ExperimentalFlags.swift`, `LiteRTLMError.swift`, `Telemetry.swift` and `ToolManager.swift`
([`generated/patch-log.md`](../generated/patch-log.md)); its port to C API 1.0.0 (`check` / `create` /
`get`) is Google's, from that base [`swift/LiteRTLMError.swift:105-115`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/LiteRTLMError.swift#L105-L115) [`swift/Engine.swift:84-91`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Engine.swift#L84-L91). <!-- c:LM-SWIFT-064 --> The
notice at the top of `swift/Conversation.swift` names our patches [`swift/Conversation.swift:15-24`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L15-L24).

Agello at `a6912128` vendors our build's wrapper: every file under `Vendor/LiteRT-LM/swift/` is
byte-identical to the fork's `swift/` at `b7b856a0` (unchanged since `82b05ea1`), and
`Vendor/LiteRT-LM/Package.swift` is the fork's except for the binary target, which is the release's
zip; the binary is the release's, built from `6da75396` [`Package.swift:37-42`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/Package.swift#L37-L42) (Agello
`Vendor/LiteRT-LM/Package.swift:37-43`, `Vendor/README.md`). <!-- c:LM-SWIFT-046 -->

## How Agello uses it

All of it lives in `LiteRTChatEngine` (`Agello/Inference/ChatEngine.swift`) and `LiteRTEmbedder`
(`Agello/Knowledge/Embedding.swift`) at `a6912128`.

| What | How | Agello source |
| --- | --- | --- |
| Flags | opts in, `enableBenchmark = false` (its stats come from telemetry), `filterChannelContentFromKvCache = true`, then `enableSpeculativeDecoding` from its plan before each engine | `ChatEngine.swift:195-205`, `:245` |
| Engine | `.gpu`, vision and audio executors on `.cpu()` when enabled, `maxNumTokens` from its budget, a cache directory: one `kv-all-<build id>` folder for every context size, everything else in the cache folder deleted at each load <!-- c:X-CFG-008 -->; `initialize()` awaited from a detached task | `ChatEngine.swift:355-371` |
| Validation | a one-token reply on a throwaway conversation before committing to a new engine | `ChatEngine.swift:377-390` |
| Chat conversation | system prompt, replayed history, tools (`get_current_date_time`, a calculator, `web_search`), `SamplerConfig` (seeded on a regenerate), `ThinkingConfig(budget:)`, visual token budget, MTP per conversation (off when it samples with a seed) | `ChatEngine.swift:392-421` |
| JSON helpers | `enableResponseFormat: true`, speculative decoding off unless `AGELLO_JSON_MTP=1`, `sendMessage(..., responseFormat: .json(schema:))` | `ChatEngine.swift:621-650` |
| Replies | `sendMessageStream(_:maxOutputTokens:thinkingConfig:followUpMaxOutputTokens:maxToolRounds:)`, `Task.isCancelled` checked per chunk, `cancel()` on cancellation, when output would overflow the KV room, or after a stream error | `ChatEngine.swift:874-925`, `:960-980` |
| Cancel | `cancel()` on the helper or main conversation; a cancelled conversation is reseeded | `ChatEngine.swift:95-99`, `:987-999` |
| Unload | `unloadLocked` sets the conversation to nil, then the engine | `ChatEngine.swift:329-337` |
| Embeddings | `EmbeddingEngine` on `.gpu` only (two tries); `EmbeddingOptions(normalize: true, outputSize: 768, inputOverflowStrategy: .truncate)`; batches of 8 | `Embedding.swift:64-69`, `:76-86`, `:140-161` |

Agello's load opts into experimental APIs, sets `enableBenchmark` to false (its reply stats come from
the runtime's telemetry) and `filterChannelContentFromKvCache` to true, and sets
`enableSpeculativeDecoding` from its plan right before building each engine. <!-- c:LM-SWIFT-056 -->
It builds engines on the GPU, with the vision and audio executors on the CPU backend, and awaits
`initialize()` from a detached task. <!-- c:LM-SWIFT-057 --> Because a load can report success with a
section that failed to map, it proves each new engine with a one-token reply before using it.
<!-- c:LM-SWIFT-058 --> Its JSON helpers turn speculative decoding off unless `AGELLO_JSON_MTP=1` asks
for it, because the runtime's grammar states shared one matcher across a speculative round.
<!-- c:LM-SWIFT-059 --> It cancels through `Conversation.cancel()` and reseeds afterwards.
<!-- c:LM-SWIFT-060 --> Its embedder asks for normalized 768-dimension vectors, truncating long
inputs, in native batches of 8, on the GPU only. <!-- c:LM-SWIFT-061 -->

On unload, `unloadLocked` sets its `Conversation` to nil before its `Engine`. The native
conversation is deleted before the engine regardless of that order: each Swift `Conversation` holds
its `Engine` strongly, so `Engine.deinit` (`litert_lm_engine_delete`) runs only after every
`Conversation`, including any helper conversation still referenced, has been released
(`Conversation.deinit` calls `litert_lm_conversation_delete`) [`swift/Conversation.swift:79`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L79) [`swift/Conversation.swift:102-110`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L102-L110) [`swift/Engine.swift:394-400`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Engine.swift#L394-L400) [`swift/Engine.swift:457-462`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Engine.swift#L457-L462)
(Agello `ChatEngine.swift:329-337`).

## On Apple

- **The package.** Platforms iOS 27 and macOS 27, the slices' own minimum
  ([build-and-release.md](build-and-release.md)), and one xcframework with iPhone and Mac slices
  [`Package.swift:21-25`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/Package.swift#L21-L25) [`Package.swift:37-42`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/Package.swift#L37-L42). Agello runs the iPhone app on the Mac as
  "Designed for iPad/iPhone"; which slice that loads is covered in
  [apple-platform.md](../apple-platform.md).
- **Swift language mode.** Our build's package declares `swift-tools-version: 6.0`, so the wrapper
  builds in Swift 6 language mode (ours: `24f35f81`): the flags sit behind a lock, and the immutable
  native handles and the tool registry are declared `nonisolated(unsafe)` [`Package.swift:1-2`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/Package.swift#L1-L2). The
  package at the release's commit declares 5.9, Swift 5 mode [`Package.swift:1`](https://github.com/PowerBeef/agello-litert-lm/blob/6da75396ece779d3c7b6b605ec8c2c06861ce1eb/Package.swift#L1); Agello vendors the
  6.0 one.
- **Threads.** `Engine` and `EmbeddingEngine` loads and embedding calls block a Swift concurrency
  thread for their duration; `sendMessage` blocks one for the whole reply; streamed chunks arrive on
  LiteRT-LM's threads, not the main actor. Hop to the main actor before touching UI state (Agello
  yields into its own `AsyncThrowingStream` and consumes it from `AppState`).
- **Metal residency set.** Off unless `ExperimentalFlags.gpuEnableMetalResidencySet` is set before
  `initialize()`; Agello leaves it unset.

## Settings

Swift-level settings (`EngineConfig`, `ConversationConfig`, `SamplerConfig`, `ThinkingConfig`,
`EmbeddingEngineConfig`, `EmbeddingOptions`, `ExperimentalFlags`) are listed with defaults and their C
counterparts in [configuration.md](../configuration.md) (layer "Swift"); the flag table above says when
each flag is read.

## Limits and failure modes

| Symptom | Cause | What to do |
| --- | --- | --- |
| The engine stalls for 10 minutes after a stream (upstream's wrapper, or ours before `0b2ca12fc4`) | the last reference to a conversation was released in the final stream callback | use the weak-reference patch; never let a stream own its conversation |
| The conversation fails after the task reading its stream was cancelled | cancelling that task cancels the native work, as `cancel()` does | create a new conversation |
| The next message fails after a cancel | the session is poisoned after `CancelProcess` | create a new conversation |
| A streamed reply runs tools although `automaticToolCalling` is false | the stream path ignores the flag | don't register tools on conversations that mustn't call them |
| `getBenchmarkInfo()` throws `benchmarkNotEnabled` or `benchmarkInfoUnavailable` | the flag is false at call time (`benchmarkNotEnabled`), or was false when the engine was created, so the engine collected nothing | opt in and set it before `initialize()` |
| A flag has no effect | not opted in (the setter only logs), or set after the engine or conversation read it | opt in, then set before the reading call |
| Tool arguments fail to decode | `convertCamelToSnakeCaseInToolDescription` changed after the conversation was created | set it once, before creating conversations |
| A streamed reply ends with an error after a malformed tool call | the parser rejected the block; the stream ends once, with the first error, while decoding may run on | see [Error chunks and the retained context](#error-chunks-and-the-retained-context) |
| A streamed reply ends with `recurringToolCallLimitExceeded` | more tool rounds than `maxToolRounds` | raise the limit, or answer without tools |
| A long embedding input throws | `inputOverflowStrategy` left nil gets the C++ default, *error* | pass `.truncate` or `.chunkAndAverage` (Agello truncates) |
| A cooperative thread is blocked for seconds | `sendMessage`, `initialize`, embedding calls are blocking C calls | run them off the main actor; expect one thread per call |
| `fatalError` inside a tool | a required `@ToolParam` read without a value | give parameters defaults or make them optional |

## Where in the code

| Concept | Symbol | Citation |
| --- | --- | --- |
| Engine actor | `Engine` | [`swift/Engine.swift:32-49`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Engine.swift#L32-L49) |
| Settings sequence | `Engine.initializeInternal` | [`swift/Engine.swift:95-183`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Engine.swift#L95-L183) |
| Conversation | `Conversation` | [`swift/Conversation.swift:68-110`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L68-L110) |
| Blocking send and tool loop | `Conversation.sendMessage` | [`swift/Conversation.swift:136-181`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L136-L181) |
| Stream | `Conversation.sendMessageStream`, `sendToStream` | [`swift/Conversation.swift:493-552`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L493-L552), [`swift/Conversation.swift:588-644`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L588-L644) |
| Stream state (ours) | `StreamContext`, `StreamTable`, `handleStreamChunk`, `streamCallback` | [`swift/ConversationStream.swift:101-219`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ConversationStream.swift#L101-L219), [`swift/ConversationStream.swift:221-271`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ConversationStream.swift#L221-L271), [`swift/ConversationStream.swift:273-336`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ConversationStream.swift#L273-L336) |
| Tool error responses (ours) | `handleToolCalls`, `toolArguments`, `toolErrorMessage` | [`swift/Conversation.swift:405-437`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L405-L437), [`swift/Conversation.swift:439-468`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Conversation.swift#L439-L468) |
| Tools | `Tool`, `ToolParam`, `ToolManager` | [`swift/Tool.swift:211-221`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Tool.swift#L211-L221), [`swift/Tool.swift:84-101`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/Tool.swift#L84-L101), [`swift/ToolManager.swift:68-122`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ToolManager.swift#L68-L122) |
| Response format | `ResponseFormat` | [`swift/ResponseFormat.swift:20-72`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ResponseFormat.swift#L20-L72) |
| Flags | `ExperimentalFlags` | [`swift/ExperimentalFlags.swift:24-81`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/ExperimentalFlags.swift#L24-L81) |
| Errors | `LiteRTLMError`, `check` | [`swift/LiteRTLMError.swift:19-27`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/LiteRTLMError.swift#L19-L27), [`swift/LiteRTLMError.swift:84-91`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/LiteRTLMError.swift#L84-L91) |
| Model info | `ModelInfo` | [`swift/ModelInfo.swift:298-312`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/ModelInfo.swift#L298-L312) |
| Embeddings | `EmbeddingEngine` | [`swift/EmbeddingEngine.swift:34-58`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/EmbeddingEngine.swift#L34-L58) |
| Build info, telemetry (ours) | `BuildInfo`, `Telemetry` | [`swift/BuildInfo.swift:18-30`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/BuildInfo.swift#L18-L30), [`swift/Telemetry.swift:18-62`](https://github.com/PowerBeef/agello-litert-lm/blob/b7b856a0cf7e2304f556e985218bc698768233f7/swift/Telemetry.swift#L18-L62) |
| Foundation Models | `LiteRTLanguageModel` | [`swift/apple_fm/LiteRTLanguageModel.swift:51-52`](https://github.com/google-ai-edge/LiteRT-LM/blob/df6df5b84d5845b49889f631ed4cac80b4325dd3/swift/apple_fm/LiteRTLanguageModel.swift#L51-L52) |

## Open questions

- Why `LiteRTLMFoundationModels` fails to compile against the macOS 27 SDK (the generator only
  detected that it failed).
- Whether concurrent sends on one `Conversation` are safe in the native engine was not checked; the
  wrapper does nothing to prevent them.
- **More than one terminal report.** Traced: an unparsable tool call is reported at least twice by
  the conversation layer, and the C callback and the wrapper each end the send once
  ([Error chunks and the retained context](#error-chunks-and-the-retained-context)). Not reproduced;
  the execution manager's own error paths (a task's callback getting an error and an end state) were
  not traced chunk by chunk.
- Agello keeps speculative decoding off for its JSON helpers because drafted tokens broke the JSON
  mask; the traced mechanism (one llguidance state shared across a speculative round,
  [../troubleshooting.md](../troubleshooting.md#json-constrained-replies-break-with-mtp-on)) has a fix
  in our build (`1382157b`, 2026-11-17) that the release Agello ships doesn't have; not checked on
  the device here.
