# Turn-scoped inference primitives

Date: 2026-08-17 · Status: Proposed

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
is `nil` (`ChatSession.swift:757`), and `respond(to messages:)` is documented for exactly
this pattern: *"Use this to continue an existing session with non-user roles, such as one or
more tool results, while preserving the session's KV cache."* The work is to expose that
capability across the Nitro bridge, not to invent it.

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
- Guaranteed schema conformance through constrained decoding (see *Structured output*).
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
| Tools | Installed during `load()` | Supplied per Turn Context or per turn, schema only |
| Tool loop | Native executes handlers and continues | Tool Call Requests returned to the caller |
| Prefill | Fresh session per unmanaged turn | Warm KV cache retained per Turn Context |
| Token budget | Counts available only after generation | Counted before a turn from the loaded tokenizer |
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

### Serialized turns

One LLM Generation Turn runs at a time, process-wide, across `runTurn`, `generate`,
`stream`, `streamWithEvents`, and `ChatSession`. A `runTurn` call rejects before starting if
another turn is active. `LLM.stop()` cancels whichever turn is active. This package queues
nothing; a caller that wants a queue builds one.

This decision is recorded in ADR 0002.

### Cold turns are for one-shot work

On a warm turn the `ChatSession` keeps the assistant tool-call message in its own transcript,
so the caller's next turn continues correctly. On a cold turn there is no transcript, and the
caller would have to rebuild that assistant message itself — in a format that is
chat-template-specific per model family.

Rather than export template internals, the rule is documented: **cold turns are for one-shot
generation; tool loops use a Turn Context.** A cold turn that ends in Tool Call Requests
returns them, but the caller cannot faithfully continue that exchange without a context.

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

`ToolSchema` is a new exported type: `ToolDefinition` without the `handler` field, so name,
description, and parameters only. The package never executes a tool supplied to a Turn
Context or to a turn.

Contexts belong to the Resident Model. `unload()` releases all of them. `release()` is
idempotent. Supplying an unknown or released context ID rejects before a turn starts.

### Turn execution

```ts
export interface LLMMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string
  /** Required when role is 'tool'. Matches the id of the Tool Call Request. */
  toolCallId?: string
}

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
  /** Raw JSON text as emitted by the model. The caller parses and validates it. */
  arguments: string
}

export interface LLMTurnOutcome {
  finishReason: LLMTurnFinishReason
  content: string
  thinking?: string
  /** Populated when finishReason is 'tool_calls'. Empty otherwise. */
  toolCalls: LLMToolCall[]
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

`runTurn` emits the existing `StreamEvent` sequence for tokens and thinking. It does not emit
tool execution events, because it executes no tools.

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
  if (turn.finishReason !== 'tool_calls') break

  messages = await Promise.all(
    turn.toolCalls.map(async call => ({
      role: 'tool' as const,
      toolCallId: call.id,
      content: await runToolWithPolicy(call),
    })),
  )
}

ctx.release()
```

Budgets, approval, cancellation, retries, delegation, and persistence all live in
`runToolWithPolicy` and in the loop. None of them enter this package.

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
`superseded`, `unloaded` — and maps them onto `LLMTurnFinishReason`.

A cancelled warm turn leaves its Turn Context's KV cache in a mutated state. The context is
marked as needing a rebuild, exactly as the managed session is today, so the next turn
against it is correct even if it is cold.

## Structured output

`mlx-swift-lm` has no JSON-schema decoder. It offers only the `LogitProcessor` protocol
(`Evaluate.swift:34`), which is the raw material for a constrained decoder rather than a
decoder.

Version one therefore implements `responseSchema` as a **forced tool call**. The package
converts the schema into a single synthetic tool, sends it as the only tool for that turn,
and requires the model to call it. The chat template already formats tool calls, and the
existing native parser already extracts arguments as JSON. The outcome returns that JSON as
`content` with `finishReason: 'completed'`.

- The cost is close to zero new native code.
- Reliability is good on tool-trained models such as Qwen3, and it is a tendency, not a
  guarantee. A model may still emit arguments that do not parse.
- A turn whose arguments do not parse resolves with `finishReason: 'failed'` and
  `stage: 'schema'`, so the caller can retry deliberately instead of guessing.
- `responseSchema` is exclusive with tools. Supplying it together with request `tools`, or
  against a Turn Context that declares tools, rejects before the turn starts. A planning
  context should therefore declare no tools.

A real `LogitProcessor` grammar that makes malformed output impossible is deferred. It needs
an incremental JSON parser and a vocabulary mask, and it is a project of its own.

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

- A cold turn that ends in Tool Call Requests cannot be faithfully continued, because the
  assistant tool-call message format is chat-template-specific. Use a Turn Context for loops.
- A Turn Context does not survive an app restart. `ChatSession.saveCache(to:)` and
  `loadPromptCache(url:)` exist upstream and would enable it; that is a later primitive.
- Structured output is a strong tendency, not a guarantee.
- Only one model is resident. A caller that needs two models loads them in turn.

## Testing strategy

### TypeScript

- Request validation: empty messages, unknown role, missing `toolCallId` on a tool message,
  `responseSchema` together with `tools`, non-finite numbers.
- Context lifecycle: create, use, release, double release, use after release.
- Rejection while a turn is active, in both directions against legacy entry points.
- Outcome normalization for every finish reason, including `tool_calls` with zero, one, and
  several calls.
- Event mapping and safe-callback isolation for `onEvent`.
- Cold and warm request shapes produce the expected native call.

### Pure Swift

Extend the existing `swiftc` test pattern for Foundation-only logic:

- Effective turn configuration planning for legacy versus turn-scoped requests.
- Context registry bookkeeping: insertion, release, release-all, unknown ID.
- Safe clamping of numeric options.
- Schema-to-synthetic-tool conversion.

### On-device

- One model load serves many Turn Contexts; no second download or container load.
- Two contexts never leak history to each other.
- Tool calls are returned, not executed.
- `stop()` during a warm turn settles, and the next turn against that context is correct.
- Memory across twenty warm turns on one context, and across ten contexts.
- Measured cold versus warm turn time, to confirm the KV cache is actually retained.

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

1. Add the Nitro wire types, flattened where the bridge requires it.
2. Add the context registry, built with `toolDispatch: nil`.
3. Implement the cold and warm `runTurn` paths.
4. Return Tool Call Requests with `finishReason: 'tool_calls'`.
5. Refactor `beginTurn` so legacy and turn-scoped paths share one implementation.
6. Regenerate Nitrogen output after the handwritten spec compiles.
7. Add the TypeScript, pure Swift, and on-device tests above.

Exit: a caller-driven tool loop completes over a single model load, and two contexts never
leak history.

### Milestone 2 — Token counting

1. Implement `countTokens` over an assembled prompt.
2. Check it against the prompt token count reported in generation statistics.

Exit: counted tokens agree with the real prefill count within a stated tolerance.

### Milestone 3 — Structured output

1. Convert `responseSchema` into a single forced synthetic tool.
2. Report a typed schema failure when the arguments do not parse.
3. Test against a tool-trained model on device.

Exit: a schema turn returns parsed JSON or a typed failure, never a silent bad value.

### Milestone 4 — Documentation and example

1. README section: building an agent loop on `runTurn` and Turn Contexts.
2. A `runTurn` versus `ChatSession` selection guide.
3. A Turn Context lifetime and memory guide.
4. A small example-app screen that runs a two-tool loop.

The example is validation and documentation. It is not a harness and is not part of the
public interface.

## Risks

### The caller leaks Turn Contexts

The package never evicts a context the caller holds, so a leak grows memory. Mitigate with
`contextIds` for inspection, a documented lifetime guide, an example that releases in a
`finally` block, and a development-mode warning above a configurable context count.

### The KV cache is not retained as expected

The whole warm path rests on `ChatSession` keeping its cache across `respond(to messages:)`.
Milestone 0 measures cold versus warm turn time before Milestone 1 depends on it. If the
measurement contradicts the upstream documentation, the primitive still works — it is only
slower, and the interface does not change.

### Structured output is unreliable on small models

Reported as a typed schema failure rather than a silent bad value, so a caller can retry. The
constrained decoder remains available as a later project if measurements demand it.

### Nitro cannot express the wire types

`LLMTurnOutcome` nests an array of objects and a union-typed finish reason. Flatten where the
bridge requires it, and hide the flattening in the TypeScript wrapper as the package already
does for `StreamEventEnvelope`.

### The package grows an orchestrator anyway

The clearest defence is the vocabulary. `CONTEXT.md` no longer defines Agent, Handoff, or
Artifact, and ADR 0002 records the boundary. A future contributor who wants to add a
scheduler must first argue against a written decision.

## Open questions requiring implementation evidence

1. How much memory does one Turn Context cost per thousand transcript tokens, on the baseline
   device and model?
2. Does a cancelled warm turn always leave a recoverable cache, or is a rebuild always
   required?
3. Which model families emit tool-call arguments reliably enough for the forced-tool
   structured-output path?
4. Is a development-mode context-count warning useful, or noise?

None of these changes the interface or the boundary.
