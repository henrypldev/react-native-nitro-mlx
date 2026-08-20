# Turn-scoped inference primitives

Date: 2026-08-17 · Status: Proposed · Revised: 2026-08-18

## Summary

Add the primitives a consumer application needs to build an agent harness on top of
`react-native-nitro-mlx`. The package does not gain a harness. It gains three things:

1. **Turn Contexts** — named, retained conversation state over one Resident Model, so a
   caller can hold many independent instruction/history sets without reloading weights.
2. **`LLM.runTurn`** — a turn-scoped call that returns Tool Call Requests to the caller
   instead of executing them, so the caller owns the loop.
3. **Token counting and structured output** — the two supporting operations a harness needs
   to budget context and parse a model plan.

Scheduling, agents, handoffs, artifacts, approvals, budgets, checkpoints, and run outcomes
are consumer concerns. They are explicitly out of scope.

## Motivation

The package already supplies local MLX model loading, streaming generation, tool-call
lifecycle events, normalized generation outcomes, managed history, cancellation, embeddings,
STT, and TTS. What it does not supply is any way to vary turn configuration without paying
for model residency.

Three properties of the current native module block a consumer-built harness.

**`load()` always reloads weights.** `HybridLLM.swift:756` begins with `resetModelState()`
and then calls `modelFactory.loadContainer(...)`. It never compares the requested model ID
with the loaded one. A consumer that changes instructions, tools, or history must call
`load()` again and re-read the weights from disk.

**Turn configuration lives in the singleton.** `systemPrompt`, `tools`, `manageHistory`, and
`messageHistory` are properties of one `HybridLLM` instance. Two logical roles cannot exist
at once; one overwrites the other.

**Tool calls never reach the caller.** The loop at `HybridLLM.swift:945` executes tool
handlers in Swift and continues generating in the same call. A harness needs control between
the tool call and the continuation, for approval, budget checks, re-planning, and
cancellation.

The upstream Swift library already supports the shape we need.
`MLXLMCommon.ChatSession` yields `Generation.toolCall(...)` to its caller when `toolDispatch`
is `nil` (`ChatSession.swift:760` in 3.31.4, inside `streamMap` at `:567`), and
`respond(to messages:)` is documented for exactly this pattern: *"Use this to continue an
existing session with non-user roles, such as one or more tool results, while preserving the
session's KV cache."* The work is to expose that capability across the Nitro bridge, not to
invent it.

## Prior art

This design was checked against the conventions of the field. The findings, with citations,
are in `docs/research/2026-08-17-agent-harness-inference-conventions.md`. Sources include the
Anthropic Messages API, the OpenAI Chat Completions and Responses APIs, `llama-server`,
Ollama, MCP, the Vercel AI SDK, the OpenAI Agents SDK, LangGraph, and Pydantic AI.

Four decisions here match convention without exception: returning tool calls rather than
executing them, keying tool results by call ID, counting tokens before a turn, and shipping
no agent loop. The closest single precedent is the Vercel AI SDK, where a tool defined
without an `execute` function is returned to the caller and ends the loop — a documented
feature, not an escape hatch.

Where this document deviates, the deviation is named and justified in place.

## Goals

- Load one model once and keep it resident while turn configuration changes freely.
- Let a caller hold many isolated Turn Contexts over that one Resident Model.
- Return Tool Call Requests to the caller so the caller owns the tool loop.
- Preserve the KV cache across tool round trips, so a loop is not dominated by prefill.
- Guarantee at most one active LLM Generation Turn, across new and legacy entry points.
- Report token counts from the loaded tokenizer, so a caller can budget context exactly.
- Offer a practical structured-output path for planning turns.
- Preserve all existing `LLM` and `ChatSession` behavior.
- Keep the orchestration surface out of this package.

## Non-goals

- An agent runtime, scheduler, runner, or run state machine.
- Agents, handoffs, artifacts, checkpoints, budgets, authorization, or approval.
- Concurrent MLX generation.
- A background daemon, or any privileged iOS data integration.
- Automatic eviction of Turn Contexts the caller still holds.
- Guaranteed schema conformance in version one (see *Structured output*).
- Persisting a Turn Context across app restarts.

## Domain language

The canonical terms are recorded in `CONTEXT.md`: Resident Model, Turn Context, LLM
Generation Turn, Tool Call Request, LLM Generation Outcome.

Two distinctions matter:

1. A Turn Context is not a model. It holds instructions, a transcript, and a KV cache over
   weights it does not own.
2. An LLM Generation Turn is one model pass. A caller's loop over several turns is the
   caller's concept, and this package has no name for it.

## Current-state gaps

| Concern | Current behavior | Required behavior |
| --- | --- | --- |
| Model residency | `load()` re-reads weights for any call | `load()` skips weight I/O for the loaded model ID |
| Instructions | Singleton `systemPrompt` | Supplied per Turn Context or per turn |
| History | One native managed history | Many isolated Turn Contexts |
| Tools | Installed during `load()`, flat parameter list | Supplied per turn as JSON Schema |
| Tool loop | Native executes handlers and continues | Tool Call Requests returned to the caller |
| Prefill | Fresh session per unmanaged turn | Warm KV cache retained per Turn Context |
| Token budget | Counts available only after generation | Counted before a turn from the loaded tokenizer |
| Usage | One aggregate `tokenCount` | Prompt and completion counts reported separately |
| Structured output | None | Schema-directed turn with a typed failure |

## Design decisions

### The package supplies primitives, the consumer supplies policy

Whether a tool may run, how many steps a task gets, what an agent is, when to ask the user,
and what to persist are product decisions. They depend on the application's UI, database,
and permissions. Putting them here would force every consumer through one opinion and would
tie their iteration speed to this package's release cycle.

The package therefore stops at the boundary where a decision stops being about inference.

### Turn Contexts ship in version one

An earlier draft ran every turn against a fresh session, accepted repeated prefill, and
deferred warm contexts as a possible later optimization. That is the wrong order once the
caller owns the tool loop, because each tool round trip becomes a separate turn over a
growing transcript.

`ChatSession.respond(to messages:)` preserves the KV cache across appended tool results, so
warmth and caller control are not in tension. A Turn Context is a retained `ChatSession`: a
small Swift object plus its cache, not a second copy of the weights.

### The Turn Context is opaque on purpose

The public interface exposes a context only as an ID with a `release()` method. Nothing in
the TypeScript surface names `ChatSession`. That opacity is deliberate: the native
implementation may later move below `ChatSession` to `TokenIterator` without a public
change. See *Structured output* for the reason that matters.

### Serialized turns

One LLM Generation Turn runs at a time, process-wide, across `runTurn`, `generate`,
`stream`, `streamWithEvents`, and `ChatSession`. A `runTurn` call rejects before starting if
another turn is active. `LLM.stop()` cancels whichever turn is active. This package queues
nothing; a caller that wants a queue builds one.

This decision is recorded in ADR 0002. It deviates from `llama-server`'s N-slot concurrency
and matches Ollama's default of one parallel slot.

## Public TypeScript interface

Names may be refined during implementation; semantics and ownership should stay stable.

### Model residency

```ts
LLM.load(modelId: string, options?: LLMLoadOptions): Promise<void>
LLM.unload(): void
LLM.isLoaded: boolean
LLM.loadedModelId: string | null
```

`load()` with the already-loaded model ID performs no weight I/O. It still resets tools,
generation configuration, seed context, and managed history, so the observable state after
`load()` is unchanged. Only the elapsed time and the disk reads change, which keeps `load()`
usable as a reset.

### Messages

```ts
export type LLMMessage =
  | { role: 'system' | 'user'; content: string }
  | { role: 'assistant'; content: string; toolCalls?: LLMToolCall[] }
  | {
      role: 'tool'
      toolCallId: string
      content: string
      /** Tool name, when the caller can supply it. Not consumed by every chat template. */
      name?: string
      /** True when the tool failed. The model must see failures to recover from them. */
      isError?: boolean
    }
```

The assistant variant is required, not optional. Without it, `history` on a cold turn cannot
seed a transcript that contains a tool exchange, and `runTurn` would regress against the
package's own `ChatSession`, whose `AssistantChatMessage` already carries `toolCalls`
(`package/src/chat.ts:49-58`). Upstream renders it model-agnostically:
`Chat.Message.assistant(_ content:, toolCalls:)` at `Chat.swift:67-76`, through
`MessageGenerator.addToolMetadata` at `Chat.swift:137-158`, which emits a `tool_calls` array
that the model's own Jinja template consumes.

`isError` follows the MCP rule: *"Any errors that originate from the tool SHOULD be reported
inside the result object … not as a protocol-level error response. Otherwise, the LLM would
not be able to see that an error occurred and self-correct."* A harness that returns a denied
or failed tool as an ordinary string loses that signal.

`name` is carried where a caller can supply it, but upstream `Chat.Message` has no field for
it, so it may be dropped during rendering. This is a stated limitation rather than a
promise.

### Tool schemas

```ts
export interface ToolSchema {
  name: string
  description: string
  /** Serialized JSON Schema for the tool arguments. Root must be an object schema. */
  parameters: string
}
```

This replaces the flat `ToolParameter` list for turn-scoped calls. `ToolParameter` today is
`{ name, type: string, description, required }` (`package/src/specs/LLM.nitro.ts:191-196`)
with no nesting, no `items`, no `enum`, and no `properties`; the Zod bridge silently flattens
a nested object to `type: 'object'` and an enum to `'string'`
(`package/src/tool-utils.ts:15-56`).

Every convention source uses full JSON Schema: MCP permits any JSON Schema 2020-12 keyword in
`inputSchema` as of revision `2026-07-28`, Anthropic uses `input_schema`, OpenAI uses
`parameters`. Upstream already supports more than the current bridge exposes —
`ToolParameterType` is an `indirect enum` with `.array(elementType:)` and
`.object(properties:)`, and `Tool(schema:handler:)` accepts a raw JSON Schema dictionary.

Without this change, `responseSchema` and `tools` would speak different schema languages in
one package, and the schema-to-synthetic-tool conversion in Milestone 3 would have no target
type.

`ToolSchema` carries no handler. The package never executes a tool supplied to a Turn Context
or to a turn.

### Turn Contexts

```ts
export interface LLMContextOptions {
  instructions?: string
  history?: LLMMessage[]
  tools?: ToolSchema[]
  generationConfig?: LLMGenerationConfig
}

export interface LLMContext {
  readonly id: string
  release(): void
}

LLM.createContext(options?: LLMContextOptions): Promise<LLMContext>
LLM.releaseContext(id: string): void
LLM.releaseAllContexts(): void
LLM.contextIds: string[]
```

Contexts belong to the Resident Model. `unload()` releases all of them. `release()` is
idempotent. Supplying an unknown or released context ID rejects before a turn starts.

### Turn execution

```ts
export interface LLMTurnRequest {
  /** Messages appended before generation: one user message, or the previous turn's tool results. */
  messages: LLMMessage[]
  /** Reuse a warm Turn Context. Omit for a cold, isolated turn. */
  contextId?: string
  /** Used only when contextId is omitted. */
  instructions?: string
  history?: LLMMessage[]
  tools?: ToolSchema[]
  generationConfig?: LLMGenerationConfig
  /** Serialized JSON schema. See "Structured output". */
  responseSchema?: string
  tokenBatchSize?: number
}

export type LLMTurnFinishReason =
  | 'completed'
  | 'tool_calls'
  | 'length'
  | 'stopped'
  | 'unloaded'
  | 'superseded'
  | 'failed'

export interface LLMToolCall {
  id: string
  name: string
  /**
   * Parsed arguments. The native tool-call parser produced these; a malformed
   * tool call never becomes an LLMToolCall and never reaches the caller.
   */
  arguments: Record<string, unknown>
}

export interface LLMTurnUsage {
  promptTokens: number
  completionTokens: number
  /** Tokens served from a warm Turn Context rather than prefilled, when derivable. */
  cachedPromptTokens?: number
}

export interface LLMTurnOutcome {
  finishReason: LLMTurnFinishReason
  /** The unnormalized native reason, for diagnostics. Never branch on this. */
  rawFinishReason?: string
  content: string
  thinking?: string
  /** Populated when finishReason is 'tool_calls'. Empty otherwise. */
  toolCalls: LLMToolCall[]
  usage: LLMTurnUsage
  stats: GenerationStats
  error?: string
  /** Failure stage: 'prepare' | 'generate' | 'schema'. */
  stage?: string
}

LLM.runTurn(
  request: LLMTurnRequest,
  onEvent: (event: StreamEvent) => void,
): Promise<LLMTurnOutcome>
```

`runTurn` follows the ADR 0001 contract. Invalid requests, an unloaded model, an unknown
context, and an already-active turn reject before the turn starts. Once the turn starts,
completion, cancellation, and runtime failure all resolve as an outcome.

**`arguments` is an object, not a string.** An earlier draft specified a raw JSON string
"as emitted by the model". That was false: `ToolCall.Function.arguments` is
`[String: JSONValue]` (`Tool/ToolCall.swift:12`), parsed by `ToolCallProcessor` before it is
yielded, and a malformed call never becomes a `ToolCall` at all. A string would therefore be
a re-serialization in the package's own key order — neither faithful to the model nor useful
for handling malformed output, and reversed by every caller immediately. The field is split
across the industry, but the pattern is consistent: the string form is the streaming form and
the object form is the settled form. Anthropic states it directly — the deltas are partial
JSON strings, the final `tool_use.input` is always an object. `runTurn` returns a settled
outcome. If Nitro cannot carry a map cleanly, a string is an acceptable **wire** compromise,
but the TypeScript wrapper parses it before returning, as it already does for
`StreamEventEnvelope`.

**`id` is always present.** `ToolCallProcessor` assigns one when the model omits it
(`Tool/ToolCallProcessor.swift:453-468`), which makes the correlation invariant total rather
than conditional. Pydantic AI generates one for the same reason.

**Usage is split.** `GenerationStats` today reports one aggregate `tokenCount`
(`package/src/specs/LLM.nitro.ts:6-12`), so a caller cannot attribute context growth to
prefill versus generation — and Milestone 2 cannot check `countTokens` against a real turn.
`GenerateCompletionInfo` already carries `promptTokenCount` and `generationTokenCount`
(`Evaluate.swift:1971-1975`). `cachedPromptTokens` is the number that proves a Turn Context is
actually warm; it is optional because it may not be derivable in every path.

**`finishReason` is reportable, not load-bearing.** A caller's loop should branch on
`toolCalls.length > 0`, not on the enum. Three of the four agent frameworks examined do
exactly that — LangGraph's `tools_condition` is twelve lines and routes purely on
`len(ai_message.tool_calls) > 0`. The enum mixes two axes that OpenAI's Responses API and
Pydantic AI keep apart (lifecycle status versus stop reason); that is workable here only
because `toolCalls` is populated alongside it, and the documentation must say so.

### Token counting

```ts
export interface LLMTokenCountRequest {
  contextId?: string
  instructions?: string
  history?: LLMMessage[]
  tools?: ToolSchema[]
  messages?: LLMMessage[]
}

LLM.countTokens(request: LLMTokenCountRequest): Promise<number>
```

Counts the assembled prompt with the loaded tokenizer and the model's chat template. With a
`contextId`, the count includes that context's accumulated transcript.

### The caller's loop

This is the shape a consumer harness builds on. It is documentation, not package code.

```ts
const ctx = await LLM.createContext({ instructions, tools: schemas })
let messages: LLMMessage[] = [{ role: 'user', content: input }]

while (true) {
  const turn = await LLM.runTurn({ contextId: ctx.id, messages }, onEvent)
  if (turn.toolCalls.length === 0) break

  messages = await Promise.all(
    turn.toolCalls.map(async call => {
      const result = await runToolWithPolicy(call)
      return {
        role: 'tool' as const,
        toolCallId: call.id,
        name: call.name,
        content: result.content,
        isError: result.failed,
      }
    }),
  )
}

ctx.release()
```

Budgets, approval, cancellation, retries, delegation, and persistence all live in
`runToolWithPolicy` and in the loop. None of them enter this package.

## Tool call contract

**Several calls per turn.** A single model pass may emit more than one Tool Call Request.
`toolCalls` is therefore an array and may hold more than one entry. This matches the field:
OpenAI's `parallel_tool_calls` defaults to `true`.

**All results, in the next turn.** The caller must return one `tool` message per returned
call, in the next `runTurn`. Returning fewer rejects before the turn starts. Anthropic states
the underlying constraint plainly — tool result blocks must immediately follow their
corresponding tool use blocks in the message history — and the chat templates here assume the
same.

**Order does not matter.** Results are matched by `toolCallId`, not by position. The caller
may execute the tools in any order, and concurrently.

**Streaming.** `runTurn` emits the existing `StreamEvent` sequence for tokens and thinking,
and reuses `ToolCallStartEvent { id, name, arguments }` when a tool call is recognized, so a
harness can render "calling search_files…" before the turn ends. Note the limitation:
`ToolCallProcessor` buffers until a complete call parses, so this event fires once per call
rather than as a stream of argument deltas. Per-token argument streaming — Anthropic's
`input_json_delta`, OpenAI's `response.function_call_arguments.delta`, Vercel's
`tool-input-delta` — is not achievable without upstream changes and is not offered.

`runTurn` emits no tool execution events, because it executes no tools.

**`toolChoice` is not offered in version one.** `'auto' | 'none' | 'required' | { name }` is
first-class at OpenAI, Anthropic, and Vercel, and it is how a forced tool call would be
implemented properly. `mlx-swift-lm 3.31.4` has no such parameter — `tool_choice`,
`toolChoice`, `response_format`, and `grammar` are all absent from `Libraries/`. This is a
stated gap, not an oversight, and it bounds what *Structured output* can promise.

## Thinking content

`LLMTurnOutcome.thinking` reports what the model emitted during the turn. The contract across
turns is:

- **Warm turns:** whatever the model generated, including thinking tokens, stays in the Turn
  Context's KV cache. The package does not strip it, and cannot without discarding the cache.
  This differs from the network APIs, where the caller chooses what to echo back.
- **Cold turns:** the caller decides. `LLMMessage` has no thinking field, so replayed history
  carries none. A caller that wants thinking preserved must fold it into assistant `content`.

The memory cost of retained thinking tokens per context is measured in Milestone 1.

The network APIs treat this as load-bearing — OpenAI recommends passing back reasoning items
with the last function call, and Anthropic requires thinking blocks be echoed back unchanged
on the same model. Here the KV cache does that job for warm turns, and no equivalent exists
for cold turns.

## Native design

### Context registry

`HybridLLM` gains a dictionary from context ID to a retained `ChatSession`, constructed with
`toolDispatch: nil` so tool calls reach the caller. The registry is `@MainActor`-isolated
like the rest of the module's mutable state.

- `createContext` builds the session over the current container and returns a generated ID.
- `runTurn` with a `contextId` calls `streamDetails(to: messages)` on that session.
- `runTurn` without a `contextId` builds a throwaway session, runs one pass, and releases it.
- `releaseContext` calls `synchronize()` before dropping the session, so no async cache work
  outlives the release.
- `unload` releases every context before clearing the container.

### Turn execution

The generation pass collects `Generation.chunk`, `Generation.info`, and
`Generation.toolCall` values. When the pass ends with one or more tool calls, the turn
finishes with `tool_calls` and returns them. There is no `while true` continuation loop in
the `runTurn` path.

`beginTurn` is refactored so the legacy entry points and `runTurn` share one implementation
that takes an effective turn configuration. The legacy path keeps its `toolDispatch` and its
continuation loop; the new path supplies neither.

### Serialization and cancellation

The existing `generationTasks` guard already rejects a second concurrent generation. It is
extended to cover `runTurn`. Cancellation reuses the current reasons — `stopped`,
`superseded`, `unloaded` — and maps them onto `LLMTurnFinishReason`, preserving the native
value in `rawFinishReason`.

A cancelled warm turn leaves its Turn Context's KV cache in a mutated state. The context is
marked as needing a rebuild, exactly as the managed session is today, so the next turn
against it is correct even if it is cold.

## Generation configuration

`LLMGenerationConfig` gains three fields that upstream already supports and a harness needs:

```ts
seed?: number   // GenerateParameters.seed: UInt64?  (Evaluate.swift:97)
topK?: number   // GenerateParameters.topK           (Evaluate.swift:87)
minP?: number   // GenerateParameters.minP           (Evaluate.swift:90)
```

`seed` matters most. A harness that replays a failed turn, or tests against a recorded
transcript, needs the same `(seed, prompt, parameters)` to produce the same tokens. Upstream
documents exactly that guarantee. OpenAI ships `seed` with the caveat that determinism is not
guaranteed; the same caveat applies here.

**Stop sequences are a known gap.** Upstream has no per-request equivalent beyond
`extraEOSTokens` on the model configuration, so they are not offered.

## Structured output

Version one implements `responseSchema` as a **forced tool call**: the package converts the
schema into a single synthetic tool, offers it as the only tool for that turn, and instructs
the model to call it. The outcome returns the parsed arguments as `content` with
`finishReason: 'completed'`.

This is conventional, not a workaround. It is Pydantic AI's default output mode — "supported
by virtually all models and has been shown to work very well" — with native constrained
decoding as the mode that "is not supported by all models". OpenAI still frames function
calling and structured outputs as a choice by purpose rather than old versus new.

Precise wording matters here. The synthetic tool is the only tool offered and the model is
**asked** to call it. It is not forced, because no `toolChoice` mechanism exists upstream
(see *Tool call contract*).

- A turn whose arguments do not parse resolves with `finishReason: 'failed'` and
  `stage: 'schema'`, so the caller can retry deliberately instead of guessing.
- `responseSchema` is exclusive with tools. Supplying it together with request `tools`, or
  against a Turn Context that declares tools, rejects before the turn starts.
- The schema is not injected into the prompt by the grammar. Three vendors — SGLang, vLLM,
  and llama.cpp — recommend restating the expected shape in the instructions so the model's
  intent matches the constraint. The same advice applies to the synthetic tool, whose
  description should carry the shape.

### Real constrained decoding, and why it is not version one

The path is much shorter than an earlier draft claimed, and it is already inside the upstream
dependency.

`mlx-swift-lm` `main` ships **`MLXGuidedGeneration`**, an XGrammar-backed module (vendored
XGrammar v0.1.30) that constrains any MLX model to a JSON Schema, an EBNF grammar, or a
structural tag by masking token logits at each decoding step, on macOS 14 and iOS 17 and
later. Its API is `GrammarConstraint(tokenizer:jsonSchema:…)`, `computeMask()`,
`commitToken(_:)`, `rollback(_:)`. It landed in PR #334 on 2026-07-15.

Two things block adoption in version one.

**It is not in a tagged release.** `package/MLXReactNative.podspec:27` pins
`upToNextMinorVersion` from 3.31.4, tagged 2026-06-30, two weeks before the module landed.
Adopting it means pinning a commit SHA, which the podspec does not do today.

**`ChatSession` cannot take a grammar constraint.** `ChatSession.swift:648` builds
`TokenIterator(input:model:cache:parameters:)` — the parameters-based initializer, which
derives its `LogitProcessor` from `GenerateParameters`. The direct-injection initializer
`TokenIterator(input:model:cache:processor:sampler:…)` exists but `ChatSession` does not use
it, and `ChatSession` exposes no processor parameter. `MLXGuidedGeneration` therefore drives
`TokenIterator` itself through a separate `GuidedGenerationLoop`.

Turn Contexts are built on `ChatSession`. So warm contexts and constrained decoding sit on
opposite sides of an upstream seam, and version one cannot have both.

This is recorded as an open architectural question with two exits, neither of which changes
the public interface, because the Turn Context is already opaque:

1. Upstream adds a processor parameter to `ChatSession`, or accepts one from us.
2. The native implementation moves the Turn Context below `ChatSession` to `TokenIterator`,
   taking over message accumulation, template rendering, KV cache lifetime, and tool-call
   parsing — roughly what `ChatSession` does in about 870 lines.

Two further notes for whoever takes this on:

- The C++ core supplies only a CPU bitmask-apply function. There is no Metal kernel; the MLX
  mask application must be written. `llguidance` ships a Metal kernel in Python that can serve
  as a reference.
- Tool schemas with nested `$defs` need each tool's definitions hoisted to the envelope root
  with namespacing, because XGrammar resolves JSON Pointers from the document root and nested
  `$defs` leave every `$ref` dangling. This was upstream bug #432, fixed by #434.

## Compatibility

`LLM.generate`, `LLM.stream`, `LLM.streamWithEvents`, `LLM.systemPrompt`,
`LLM.clearHistory`, `manageHistory`, `additionalContext`, and `ChatSession` keep their
current behavior. No existing consumer needs a change.

The one behavior change is that `load()` stops re-reading weights for the loaded model ID.
Observable state after the call is identical.

The serialization rule spans both paths. A `runTurn` call rejects while a legacy `stream` is
active, and a legacy call rejects while a `runTurn` is active. `LLM.stop()` cancels whichever
turn is active. `runTurn` never reads or writes the legacy managed history.

## Stated limitations

- A Turn Context does not survive an app restart. `ChatSession.saveCache(to:)` and
  `loadPromptCache(url:)` exist upstream and would enable it; that is a later primitive.
- A Turn Context that outgrows its window cannot be trimmed. `canTrimPromptCache` and
  `trimPromptCache(_:numTokens:)` are public upstream (`KVCache.swift:1862`, `:1871`) and are
  the mechanism a harness would need. Not exposed in version one — a choice on the record,
  not an absence.
- Structured output is a strong tendency, not a guarantee, and cannot be combined with a
  warm Turn Context once real constrained decoding arrives.
- `toolChoice` is not offered.
- Stop sequences are not offered.
- A tool result's `name` may be dropped during rendering, because upstream `Chat.Message`
  has no field for it.
- Per-token streaming of tool-call arguments is not achievable; the event fires once per
  parsed call.
- Only one model is resident. A caller that needs two models loads them in turn.

## Testing strategy

### TypeScript

- Request validation: empty messages, unknown role, missing `toolCallId` on a tool message,
  an assistant message with malformed `toolCalls`, `responseSchema` together with `tools`,
  `parameters` that is not valid JSON Schema, non-finite numbers.
- Tool result completeness: fewer results than calls rejects; results out of order succeed;
  an unknown `toolCallId` rejects.
- Context lifecycle: create, use, release, double release, use after release.
- Rejection while a turn is active, in both directions against legacy entry points.
- Outcome normalization for every finish reason, including `tool_calls` with one and several
  calls, and `rawFinishReason` passthrough.
- Argument parsing: the wrapper returns objects, not strings.
- Event mapping and safe-callback isolation for `onEvent`.
- Cold and warm request shapes produce the expected native call.

### Pure Swift

Extend the existing `swiftc` test pattern for Foundation-only logic:

- Effective turn configuration planning for legacy versus turn-scoped requests.
- Context registry bookkeeping: insertion, release, release-all, unknown ID.
- Safe clamping of numeric options.
- JSON Schema validation and schema-to-synthetic-tool conversion, including a nested schema.
- `LLMMessage` to `Chat.Message` mapping for all four roles, including assistant with tool
  calls.

### On-device

- One model load serves many Turn Contexts; no second download or container load.
- Two contexts never leak history to each other.
- Tool calls are returned, not executed, and their arguments parse.
- A full loop: user message, tool calls, tool results, final content.
- `stop()` during a warm turn settles, and the next turn against that context is correct.
- Memory across twenty warm turns on one context, and across ten contexts, with and without
  thinking-heavy output.
- Measured cold versus warm turn time, and `cachedPromptTokens`, to confirm the KV cache is
  actually retained.
- `seed` reproduces the same tokens for the same prompt and parameters.

The package test script must widen beyond `bun test src/*.test.ts` if any new test file sits
in a subdirectory.

## Milestones

### Milestone 0 — Residency and baseline

1. Make `load()` skip weight I/O for the already-loaded model ID.
2. Add `loadedModelId`.
3. Record baselines on an iPhone with Qwen3 1.7B 4-bit, using the entry points that already
   exist: model load memory and time; turn time on the unmanaged path, which builds a fresh
   session and is therefore cold; turn time on the managed path over a growing transcript,
   which retains its session and is therefore warm.
4. Add regression tests protecting the current entry points.

The gap between the managed and unmanaged measurements is the evidence that Turn Contexts are
worth building. Per-context memory is measured in Milestone 1, once contexts exist.

Exit: a repeated `load()` reads no weights, and the baselines exist.

### Milestone 1 — Turn Contexts and `runTurn`

1. Add the Nitro wire types, flattened where the bridge requires it. The `LLMMessage` union
   and the `LLMToolCall.arguments` map are the two shapes most likely to need flattening;
   hide the flattening in the TypeScript wrapper.
2. Add the context registry, built with `toolDispatch: nil`.
3. Implement the cold and warm `runTurn` paths.
4. Return Tool Call Requests with parsed arguments and `finishReason: 'tool_calls'`.
5. Add `ToolSchema` with JSON Schema parameters, and map it to upstream `Tool`.
6. Report split usage, and `cachedPromptTokens` where derivable.
7. Refactor `beginTurn` so legacy and turn-scoped paths share one implementation.
8. Regenerate Nitrogen output after the handwritten spec compiles.
9. Add the TypeScript, pure Swift, and on-device tests above.

Exit: a caller-driven tool loop completes over a single model load, two contexts never leak
history, and a warm turn reports fewer prefilled prompt tokens than a cold one.

### Milestone 2 — Token counting

1. Implement `countTokens` over an assembled prompt, including tools and a context transcript.
2. Check it against `usage.promptTokens` from a real turn.

Exit: counted tokens agree with the reported prompt tokens within a stated tolerance.

### Milestone 3 — Structured output

1. Convert `responseSchema` into a single synthetic tool, offered alone.
2. Report a typed schema failure when the arguments do not parse.
3. Test against a tool-trained model on device, including a nested schema.

Exit: a schema turn returns parsed JSON or a typed failure, never a silent bad value.

### Milestone 4 — Documentation and example

1. README section: building an agent loop on `runTurn` and Turn Contexts.
2. A `runTurn` versus `ChatSession` selection guide.
3. A Turn Context lifetime and memory guide.
4. A small example-app screen that runs a two-tool loop.

The example is validation and documentation. It is not a harness and is not part of the
public interface.

### Deferred — real constrained decoding

Not scheduled. Blocked on a tagged `mlx-swift-lm` release containing `MLXGuidedGeneration`
and on resolving the `ChatSession` processor seam. See *Structured output*.

## Risks

### The caller leaks Turn Contexts

The package never evicts a context the caller holds, so a leak grows memory. Mitigate with
`contextIds` for inspection, a documented lifetime guide, an example that releases in a
`finally` block, and a development-mode warning above a configurable context count.

### The KV cache is not retained as expected

The whole warm path rests on `ChatSession` keeping its cache across `respond(to messages:)`.
Milestone 0 measures cold versus warm turn time before Milestone 1 depends on it, and
`cachedPromptTokens` makes the property observable at run time. If the measurement
contradicts the upstream documentation, the primitive still works — it is only slower, and
the interface does not change.

### Structured output is unreliable on small models

Reported as a typed schema failure rather than a silent bad value, so a caller can retry.
Restating the shape in the tool description is the documented mitigation. Real constrained
decoding remains available behind the seam described above.

### The `ChatSession` seam becomes permanent

If upstream never exposes a processor parameter, the only exit is reimplementing `ChatSession`
below the Turn Context. The cost is bounded and known — about 870 lines of upstream behavior —
and the public interface absorbs it without a change, because the context is opaque. Track the
upstream module and revisit when it is tagged.

### Nitro cannot express the wire types

The `LLMMessage` union and the `LLMToolCall.arguments` map are the exposure. Flatten where the
bridge requires it, and hide the flattening in the TypeScript wrapper as the package already
does for `StreamEventEnvelope`. If the map cannot cross, carry JSON text on the wire and parse
in the wrapper — but do not expose the string in the public type.

### The package grows an orchestrator anyway

The clearest defence is the vocabulary. `CONTEXT.md` no longer defines Agent, Handoff, or
Artifact, and ADR 0002 records the boundary. A future contributor who wants to add a
scheduler must first argue against a written decision.

## Open questions requiring implementation evidence

1. How much memory does one Turn Context cost per thousand transcript tokens, on the baseline
   device and model, with and without retained thinking content?
2. Does a cancelled warm turn always leave a recoverable cache, or is a rebuild always
   required?
3. Which model families emit tool-call arguments reliably enough for the synthetic-tool
   structured-output path?
4. Is `cachedPromptTokens` derivable on every path, or only when a context is reused?
5. Is a development-mode context-count warning useful, or noise?

None of these changes the interface or the boundary.
