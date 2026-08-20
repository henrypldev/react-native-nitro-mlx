# Turn-Scoped Inference Primitives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give consumer applications the primitives to build an agent harness: Turn Contexts over one resident model, `LLM.runTurn` that returns tool calls instead of executing them, token counting, and forced-tool structured output.

**Architecture:** TypeScript wire types cross the Nitro bridge flat (JSON strings for unions and maps) and a public wrapper in `turn.ts` maps them to the spec's discriminated-union types. Swift gains three pure Foundation-only planners (load decision, request validation, tool schemas) tested with the repo's `swiftc` pattern, plus a generic context registry. `HybridLLM` keeps model residency and gains a `runTurn` path built on `ChatSession` with `toolDispatch: nil`.

**Tech Stack:** TypeScript, Nitro Modules (Nitrogen codegen), Swift, mlx-swift-lm 3.31.x (`ChatSession`, `Chat.Message`, `GenerateParameters`), bun test, standalone `swiftc` tests.

**Spec:** `docs/superpowers/specs/2026-08-17-turn-scoped-inference-primitives-design.md` — read it before starting any task. The research behind it is `docs/research/2026-08-17-agent-harness-inference-conventions.md`.

## Global Constraints

- All work is inside `package/` and `example/`. No new npm or CocoaPods dependencies.
- Nitro cannot express discriminated unions or `Record<string, unknown>`. Unions and maps cross the bridge as JSON strings; the TypeScript wrapper hides this (pattern: `StreamEventEnvelope` in `package/src/specs/LLM.nitro.ts:130` mapped by `mapStreamEventEnvelope`).
- Do NOT run `bun specs` (Nitrogen regeneration) until Task 8. Generated files are updated only after the handwritten Swift side compiles (project convention, and Tasks 4–7 do not touch generated code).
- New TS test files live flat in `package/src/` (the test script is `bun test src/*.test.ts`; nested dirs do not run).
- Pure Swift files must import Foundation only — no MLX, no NitroModules — so `swiftc` standalone tests work.
- Existing behavior of `LLM.generate/stream/streamWithEvents/systemPrompt/clearHistory`, `manageHistory`, and `ChatSession` must not change. Run `bun --cwd package test` after every Swift or TS change.
- CocoaPods commands need `LANG=en_US.UTF-8` set.
- Commit after every task with a conventional-commit message. Never commit generated `nitrogen/` output separately from the handwritten spec change that produced it.
- Terminology in code comments and docs follows `CONTEXT.md`: Resident Model, Turn Context, LLM Generation Turn, Tool Call Request.

**Verification commands used throughout:**

```bash
bun --cwd package test          # TS tests
bun --cwd package typecheck     # tsc --noEmit
bun --cwd package specs         # typecheck + nitrogen + build (Task 8 onward only)
# Swift pure tests follow the existing pattern, e.g.:
bun --cwd package test:ios-llm-error
```

---

### Task 1: Model residency — `ModelLoadPlan` and the `load()` skip

**Files:**

- Create: `package/ios/Sources/ModelLoadPlan.swift`
- Create: `package/ios/Tests/ModelLoadPlanTests.swift`
- Modify: `package/ios/Sources/HybridLLM.swift:719-734` (`resetModelState`), `:756-822` (`load`)
- Modify: `package/package.json` (add `test:ios-load-plan` script)

**Interfaces:**

- Produces: `ModelLoadAction` enum (`.loadContainer` / `.reuseContainer`) and `ModelLoadPlan.action(requestedModelId:loadedModelId:hasContainer:)`. Also splits `resetModelState()` into `resetTurnConfiguration()` + container/model teardown; Task 8 reuses `resetTurnConfiguration()`.

- [ ] **Step 1: Write the failing Swift test**

`package/ios/Tests/ModelLoadPlanTests.swift`:

```swift
import Foundation

var failures = 0
func expect(_ condition: Bool, _ message: String) {
    if !condition { failures += 1; print("FAIL: \(message)") } else { print("PASS: \(message)") }
}

// Same model, container resident -> reuse
expect(
    ModelLoadPlan.action(requestedModelId: "m/a", loadedModelId: "m/a", hasContainer: true) == .reuseContainer,
    "same model with container reuses"
)
// Different model -> load
expect(
    ModelLoadPlan.action(requestedModelId: "m/b", loadedModelId: "m/a", hasContainer: true) == .loadContainer,
    "different model loads"
)
// No container (never loaded, or unloaded) -> load
expect(
    ModelLoadPlan.action(requestedModelId: "m/a", loadedModelId: nil, hasContainer: false) == .loadContainer,
    "no container loads"
)
// Same id string but container gone (unload happened) -> load
expect(
    ModelLoadPlan.action(requestedModelId: "m/a", loadedModelId: "m/a", hasContainer: false) == .loadContainer,
    "same id without container loads"
)
// Empty loaded id treated as not loaded
expect(
    ModelLoadPlan.action(requestedModelId: "m/a", loadedModelId: "", hasContainer: true) == .loadContainer,
    "empty loaded id loads"
)

if failures > 0 { exit(1) }
print("All ModelLoadPlan tests passed")
```

- [ ] **Step 2: Add the test script and run it to verify it fails**

In `package/package.json` scripts, after `test:ios-stt-audio`:

```json
"test:ios-load-plan": "swiftc ios/Sources/ModelLoadPlan.swift ios/Tests/ModelLoadPlanTests.swift -o /tmp/ModelLoadPlanTests && /tmp/ModelLoadPlanTests",
```

Run: `bun --cwd package test:ios-load-plan`
Expected: FAIL — `cannot find 'ModelLoadPlan' in scope`

- [ ] **Step 3: Write the implementation**

`package/ios/Sources/ModelLoadPlan.swift`:

```swift
import Foundation

/// Decision for `LLM.load`: reuse the resident container or read weights again.
enum ModelLoadAction: Equatable {
    case loadContainer
    case reuseContainer
}

enum ModelLoadPlan {
    /// `loadedModelId` uses `nil`/empty to mean "nothing loaded"; the resident
    /// container is reused only when the requested id matches it exactly.
    static func action(
        requestedModelId: String,
        loadedModelId: String?,
        hasContainer: Bool
    ) -> ModelLoadAction {
        guard hasContainer, let loaded = loadedModelId, !loaded.isEmpty,
            loaded == requestedModelId
        else {
            return .loadContainer
        }
        return .reuseContainer
    }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `bun --cwd package test:ios-load-plan`
Expected: `All ModelLoadPlan tests passed`

- [ ] **Step 5: Split `resetModelState` and wire the plan into `load()`**

In `HybridLLM.swift`, replace `resetModelState()` (currently `:719`) with:

```swift
    /// Resets everything `load()` reconfigures, but not model residency.
    /// Shared by the full reset and the same-model reuse path.
    private func resetTurnConfiguration() {
        session = nil
        tools = []
        toolSchemas = []
        seedMessages = []
        messageHistory = []
        structuredHistory = []
        manageHistory = false
        generationParameters = GenerateParameters()
        tokenBatchSize = 4
        toolExecution = .parallel
        contextConfig = nil
    }

    private func resetModelState() {
        resetTurnConfiguration()
        container = nil
        modelId = ""
        Memory.clearCache()
    }
```

In `load()` (currently `:756`), replace the unconditional `resetModelState()` + download + `loadContainer` block with:

```swift
            let action = ModelLoadPlan.action(
                requestedModelId: modelId,
                loadedModelId: self.modelId,
                hasContainer: container != nil
            )

            let loadedContainer: ModelContainer
            switch action {
            case .reuseContainer:
                log("Reusing resident container for \(modelId)")
                resetTurnConfiguration()
                options?.onProgress?(1.0)
                loadedContainer = container!
            case .loadContainer:
                resetModelState()

                let memoryAfterCleanup = getMemoryUsage()
                let gpuAfterCleanup = getGPUMemoryUsage()
                log("After cleanup - Host: \(memoryAfterCleanup), GPU: \(gpuAfterCleanup)")

                if !(await ModelDownloader.shared.isDownloaded(modelId: modelId)) {
                    log("Model not cached, downloading before load: \(modelId)")
                    _ = try await ModelDownloader.shared.download(
                        modelId: modelId,
                        progressCallback: { fraction in
                            options?.onProgress?(fraction)
                        }
                    )
                }

                let modelDir = await ModelDownloader.shared.getModelDirectory(modelId: modelId)
                log("Loading from directory: \(modelDir.path)")

                loadedContainer = try await modelFactory.loadContainer(
                    from: modelDir,
                    using: tokenizerLoader
                )

                try Task.checkCancellation()

                let memoryAfterContainer = getMemoryUsage()
                let gpuAfterContainer = getGPUMemoryUsage()
                log("Model loaded - Host: \(memoryAfterContainer), GPU: \(gpuAfterContainer)")
            }
```

Keep everything after (tools/config application, `self.container = loadedContainer`, `self.modelId = modelId`, `rebuildManagedSession()`, `acceptsGeneration = true`) unchanged.

- [ ] **Step 6: Verify nothing else regressed**

Run: `bun --cwd package test && bun --cwd package test:ios-load-plan && bun --cwd package test:ios-generation-task && bun --cwd package test:ios-history-trim`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add package/ios/Sources/ModelLoadPlan.swift package/ios/Tests/ModelLoadPlanTests.swift package/ios/Sources/HybridLLM.swift package/package.json
git commit -m "feat(ios): skip weight reload when load() targets the resident model"
```

---

### Task 2: `LLM.loadedModelId` on the TypeScript wrapper

**Files:**

- Modify: `package/src/llm.ts` (add getter near the existing `isLoaded` getter)
- Modify: `package/src/index.ts` only if `LLM` docs reference it (no export change needed — `LLM` is already exported)
- Test: `package/src/turn.test.ts` (create; this file grows in Tasks 4 and 11)

**Interfaces:**

- Consumes: native `readonly modelId: string` and `readonly isLoaded: boolean` (already in `LLM.nitro.ts`; `modelId` is `""` when unloaded because `resetModelState()` clears it).
- Produces: `LLM.loadedModelId: string | null` — the public residency query named in the spec.

- [ ] **Step 1: Write the failing test**

`package/src/turn.test.ts`:

```ts
import { describe, expect, it, mock } from 'bun:test'

const nativeState = {
  isLoaded: false,
  modelId: '',
}

mock.module('react-native-nitro-modules', () => ({
  NitroModules: {
    createHybridObject: () => nativeState,
  },
}))

const { LLM } = await import('./llm')

describe('LLM.loadedModelId', () => {
  it('is null when no model is loaded', () => {
    nativeState.isLoaded = false
    nativeState.modelId = ''
    expect(LLM.loadedModelId).toBeNull()
  })

  it('is the model id when loaded', () => {
    nativeState.isLoaded = true
    nativeState.modelId = 'mlx-community/Qwen3-1.7B-4bit'
    expect(LLM.loadedModelId).toBe('mlx-community/Qwen3-1.7B-4bit')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun --cwd package test`
Expected: FAIL — `loadedModelId` is `undefined`

- [ ] **Step 3: Implement the getter**

In `package/src/llm.ts`, next to the existing `get isLoaded()` accessor on the `LLM` object:

```ts
  /**
   * The id of the Resident Model, or null when nothing is loaded.
   * `load()` with this exact id performs no weight I/O.
   */
  get loadedModelId(): string | null {
    const instance = getInstance()
    return instance.isLoaded && instance.modelId !== '' ? instance.modelId : null
  },
```

- [ ] **Step 4: Run tests**

Run: `bun --cwd package test && bun --cwd package typecheck`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add package/src/llm.ts package/src/turn.test.ts
git commit -m "feat: expose LLM.loadedModelId for residency checks"
```

---

### Task 3: Baseline measurement screen (Milestone 0 evidence)

**Files:**

- Create: `example/app/benchmark.tsx`
- Modify: `example/app/index.tsx` (add a navigation link; match the file's existing link style)

**Interfaces:**

- Consumes: `LLM.load`, `LLM.loadedModelId`, `LLM.generate` from `react-native-nitro-mlx`.
- Produces: on-screen numbers a human records into the PR description: repeat-load time, cold turn time (unmanaged path), warm turn time (managed path over a growing transcript).

**Note:** MLX generation crashes on the simulator; this screen runs on a physical device only. The build itself is verified on the simulator.

- [ ] **Step 1: Write the screen**

`example/app/benchmark.tsx`:

```tsx
import { useState } from 'react'
import { Button, ScrollView, Text } from 'react-native'
import { LLM, MLXModel } from 'react-native-nitro-mlx'

const MODEL = MLXModel.Qwen3_1_7B_4bit
const PROMPT = 'List three facts about the Moon.'

async function timed(label: string, fn: () => Promise<unknown>): Promise<string> {
  const start = performance.now()
  await fn()
  return `${label}: ${(performance.now() - start).toFixed(0)} ms`
}

export default function Benchmark() {
  const [lines, setLines] = useState<string[]>([])
  const log = (line: string) => setLines(prev => [...prev, line])

  const run = async () => {
    setLines([])
    log(await timed('initial load', () => LLM.load(MODEL)))
    log(await timed('repeat load (same id, must be ~0)', () => LLM.load(MODEL)))
    log(`loadedModelId: ${LLM.loadedModelId}`)

    // Cold: unmanaged path builds a fresh session per turn.
    await LLM.load(MODEL)
    for (let i = 1; i <= 3; i++) {
      log(await timed(`cold turn ${i}`, () => LLM.generate(PROMPT)))
    }

    // Warm: managed path retains its session across turns.
    await LLM.load(MODEL, { manageHistory: true })
    for (let i = 1; i <= 3; i++) {
      log(await timed(`warm turn ${i} (history grows)`, () => LLM.generate(PROMPT)))
    }
    log('done — record these numbers in the PR')
  }

  return (
    <ScrollView contentContainerStyle={{ padding: 16, gap: 8 }}>
      <Button title="Run baseline" onPress={run} />
      {lines.map(line => (
        <Text key={line}>{line}</Text>
      ))}
    </ScrollView>
  )
}
```

In `example/app/index.tsx`, add a link/button navigating to `/benchmark`, styled like the file's existing navigation entries (read the file first and copy its pattern — expo-router `Link` or `router.push`).

- [ ] **Step 2: Verify it builds**

Run: `bun --cwd example typecheck 2>/dev/null || bunx tsc -p example --noEmit`
Expected: no type errors. (If the example has no typecheck script, the `tsc` fallback is the gate.)

- [ ] **Step 3: Manual device step (record, do not block)**

On a physical iPhone: run the app, open Benchmark, tap Run. Record the numbers. Expected shape: repeat load < 100 ms; warm turn 3 noticeably faster than cold turn 3. If the warm/cold gap is absent, STOP and flag it — Milestone 1's value rests on this measurement (spec, "The KV cache is not retained as expected").

- [ ] **Step 4: Commit**

```bash
git add example/app/benchmark.tsx example/app/index.tsx
git commit -m "feat(example): baseline benchmark screen for residency and warm-turn evidence"
```

---

### Task 4: Nitro wire types and TypeScript validators

**Files:**

- Modify: `package/src/specs/LLM.nitro.ts` (types + method declarations + `LLMGenerationConfig` fields)
- Modify: `package/src/runtime.ts` (validators)
- Test: `package/src/turn.test.ts` (extend)

**Interfaces:**

- Produces (wire, all flat for Nitro): `LLMTurnMessage`, `LLMToolSchema`, `LLMToolCallWire`, `LLMTurnUsage`, `LLMTurnFinishReason`, `LLMTurnOutcome`, `LLMTurnRequest`, `LLMTurnContextOptions`, `LLMTokenCountRequest`; native methods `createTurnContext`, `releaseTurnContext`, `releaseAllTurnContexts`, `turnContextIds`, `runTurn`, `countTokens`; `LLMGenerationConfig.seed/topK/minP`.
- Produces (validators in `runtime.ts`): `validateTurnMessages`, `validateToolSchemas`, `validateTurnRequest`, `validateTurnContextOptions`, `validateTokenCountRequest` — exact signatures in Step 3.
- **Do NOT run `bun specs` in this task.** Swift implementations arrive in Task 8; regeneration before then breaks the iOS build.

- [ ] **Step 1: Add wire types to `LLM.nitro.ts`**

After the existing `LLMMessage` interface (`:144-147`):

```ts
/**
 * Flat wire form of a turn message. `toolCallsJson` is a serialized array of
 * `{ id, name, arguments }` objects and is only meaningful on assistant
 * messages; the public wrapper in `turn.ts` maps this to a discriminated union.
 * @internal
 */
export interface LLMTurnMessage {
  role: string
  content: string
  toolCallId?: string
  name?: string
  isError?: boolean
  toolCallsJson?: string
}

/** Tool exposed to a turn. `parameters` is a serialized JSON Schema; root must be an object schema. */
export interface LLMToolSchema {
  name: string
  description: string
  parameters: string
}

/**
 * Wire form of a Tool Call Request. `argumentsJson` is serialized from the
 * already-parsed native arguments; the wrapper parses it back to an object.
 * @internal
 */
export interface LLMToolCallWire {
  id: string
  name: string
  argumentsJson: string
}

export interface LLMTurnUsage {
  promptTokens: number
  completionTokens: number
  /** Tokens served from a warm Turn Context rather than prefilled, when derivable. */
  cachedPromptTokens?: number
}

export type LLMTurnFinishReason =
  'completed' | 'tool_calls' | 'length' | 'stopped' | 'unloaded' | 'superseded' | 'failed'

export interface LLMTurnOutcome {
  finishReason: LLMTurnFinishReason
  /** Unnormalized native reason, for diagnostics. Never branch on this. */
  rawFinishReason?: string
  content: string
  thinking?: string
  /** Populated when finishReason is 'tool_calls'. Empty otherwise. */
  toolCalls: LLMToolCallWire[]
  usage: LLMTurnUsage
  stats: GenerationStats
  error?: string
  /** Failure stage: 'prepare' | 'generate' | 'schema'. */
  stage?: string
}

export interface LLMTurnRequest {
  /** Messages appended before generation. */
  messages: LLMTurnMessage[]
  /** Reuse a warm Turn Context. Omit for a cold, isolated turn. */
  contextId?: string
  /** Cold turns only; rejected when contextId is present. */
  instructions?: string
  history?: LLMTurnMessage[]
  tools?: LLMToolSchema[]
  generationConfig?: LLMGenerationConfig
  /** Serialized JSON Schema. Exclusive with tools (request or context). */
  responseSchema?: string
  tokenBatchSize?: number
}

export interface LLMTurnContextOptions {
  instructions?: string
  history?: LLMTurnMessage[]
  tools?: LLMToolSchema[]
  generationConfig?: LLMGenerationConfig
}

export interface LLMTokenCountRequest {
  contextId?: string
  instructions?: string
  history?: LLMTurnMessage[]
  tools?: LLMToolSchema[]
  messages?: LLMTurnMessage[]
}
```

In `LLMGenerationConfig` (`:152-173`), add:

```ts
  /** Seed for reproducible sampling. Same (seed, prompt, parameters) -> same tokens. */
  seed?: number
  /** Top-k sampling cutoff. 0 disables. */
  topK?: number
  /** Min-p sampling threshold. 0 disables. */
  minP?: number
```

In the `LLM` interface (after `streamWithEvents`):

```ts
  /**
   * Create a Turn Context: retained instructions, transcript, and warm KV
   * cache over the Resident Model. Returns the context id.
   */
  createTurnContext(options?: LLMTurnContextOptions): Promise<string>
  /** Release a Turn Context. Idempotent. */
  releaseTurnContext(id: string): void
  releaseAllTurnContexts(): void
  readonly turnContextIds: string[]

  /**
   * Run one LLM Generation Turn. Returns Tool Call Requests to the caller
   * instead of executing them. Never touches legacy managed history.
   */
  runTurn(
    request: LLMTurnRequest,
    onEvent: (event: StreamEventEnvelope) => void,
  ): Promise<LLMTurnOutcome>

  /** Count tokens for an assembled prompt with the loaded tokenizer and chat template. */
  countTokens(request: LLMTokenCountRequest): Promise<number>
```

- [ ] **Step 2: Write failing validator tests**

Append to `package/src/turn.test.ts`:

```ts
const { validateTurnMessages, validateToolSchemas, validateTurnRequest } =
  await import('./runtime')

describe('validateTurnMessages', () => {
  it('rejects an empty array when required', () => {
    expect(() => validateTurnMessages([], 'messages', { requireNonEmpty: true })).toThrow(
      /messages must not be empty/,
    )
  })

  it('rejects an unknown role', () => {
    expect(() =>
      validateTurnMessages([{ role: 'oracle', content: 'x' }], 'messages', {}),
    ).toThrow(/unknown role/)
  })

  it('rejects a tool message without toolCallId', () => {
    expect(() =>
      validateTurnMessages([{ role: 'tool', content: 'x' }], 'messages', {}),
    ).toThrow(/toolCallId/)
  })

  it('rejects toolCallsJson on a non-assistant message', () => {
    expect(() =>
      validateTurnMessages(
        [{ role: 'user', content: 'x', toolCallsJson: '[]' }],
        'messages',
        {},
      ),
    ).toThrow(/assistant/)
  })

  it('accepts a full loop shape', () => {
    expect(() =>
      validateTurnMessages(
        [
          { role: 'user', content: 'hi' },
          {
            role: 'assistant',
            content: '',
            toolCallsJson: '[{"id":"c1","name":"f","arguments":{}}]',
          },
          { role: 'tool', toolCallId: 'c1', content: 'ok', isError: false },
        ],
        'messages',
        {},
      ),
    ).not.toThrow()
  })
})

describe('validateToolSchemas', () => {
  it('rejects parameters that are not JSON', () => {
    expect(() =>
      validateToolSchemas(
        [{ name: 'f', description: 'd', parameters: '{oops' }],
        'tools',
      ),
    ).toThrow(/JSON/)
  })

  it('rejects a non-object root schema', () => {
    expect(() =>
      validateToolSchemas(
        [{ name: 'f', description: 'd', parameters: '{"type":"string"}' }],
        'tools',
      ),
    ).toThrow(/object schema/)
  })

  it('rejects duplicate tool names', () => {
    const tool = { name: 'f', description: 'd', parameters: '{"type":"object"}' }
    expect(() => validateToolSchemas([tool, tool], 'tools')).toThrow(/duplicate/)
  })
})

describe('validateTurnRequest', () => {
  const user = [{ role: 'user', content: 'hi' }]

  it('rejects responseSchema together with tools', () => {
    expect(() =>
      validateTurnRequest({
        messages: user,
        responseSchema: '{"type":"object"}',
        tools: [{ name: 'f', description: 'd', parameters: '{"type":"object"}' }],
      }),
    ).toThrow(/exclusive/)
  })

  it('rejects cold-turn fields on a warm request', () => {
    expect(() =>
      validateTurnRequest({
        messages: user,
        contextId: 'ctx-1',
        instructions: 'be brief',
      }),
    ).toThrow(/contextId/)
  })

  it('accepts a minimal cold request', () => {
    expect(() => validateTurnRequest({ messages: user })).not.toThrow()
  })
})
```

Run: `bun --cwd package test`
Expected: FAIL — validators not exported

- [ ] **Step 3: Implement validators in `runtime.ts`**

Append (following the file's existing `validate*` style, throwing `TypeError`):

```ts
const TURN_ROLES = new Set(['system', 'user', 'assistant', 'tool'])

export interface TurnMessageLike {
  role: string
  content: string
  toolCallId?: string
  name?: string
  isError?: boolean
  toolCallsJson?: string
}

export function validateTurnMessages(
  value: unknown,
  name: string,
  options: { requireNonEmpty?: boolean },
): TurnMessageLike[] {
  if (!Array.isArray(value)) {
    throw new TypeError(`${name} must be an array`)
  }
  if (options.requireNonEmpty && value.length === 0) {
    throw new TypeError(`${name} must not be empty`)
  }
  return value.map((message, index) => {
    const label = `${name}[${index}]`
    const role = (message as TurnMessageLike)?.role
    if (typeof role !== 'string' || !TURN_ROLES.has(role)) {
      throw new TypeError(`${label} has unknown role: ${String(role)}`)
    }
    if (typeof (message as TurnMessageLike).content !== 'string') {
      throw new TypeError(`${label}.content must be a string`)
    }
    const m = message as TurnMessageLike
    if (role === 'tool' && (typeof m.toolCallId !== 'string' || m.toolCallId === '')) {
      throw new TypeError(
        `${label} is a tool message and requires a non-empty toolCallId`,
      )
    }
    if (m.toolCallsJson !== undefined && role !== 'assistant') {
      throw new TypeError(`${label} carries tool calls but only assistant messages may`)
    }
    return m
  })
}

export interface ToolSchemaLike {
  name: string
  description: string
  parameters: string
}

export function validateToolSchemas(value: unknown, name: string): ToolSchemaLike[] {
  if (!Array.isArray(value)) {
    throw new TypeError(`${name} must be an array`)
  }
  const seen = new Set<string>()
  return value.map((tool, index) => {
    const label = `${name}[${index}]`
    const t = tool as ToolSchemaLike
    assertNonEmptyString(t?.name, `${label}.name`)
    assertNonEmptyString(t?.description, `${label}.description`)
    assertNonEmptyString(t?.parameters, `${label}.parameters`)
    if (seen.has(t.name)) {
      throw new TypeError(`${name} contains a duplicate tool name: ${t.name}`)
    }
    seen.add(t.name)
    let parsed: unknown
    try {
      parsed = JSON.parse(t.parameters)
    } catch {
      throw new TypeError(`${label}.parameters is not valid JSON`)
    }
    if (
      typeof parsed !== 'object' ||
      parsed === null ||
      Array.isArray(parsed) ||
      (parsed as { type?: unknown }).type !== 'object'
    ) {
      throw new TypeError(
        `${label}.parameters root must be an object schema ("type": "object")`,
      )
    }
    return t
  })
}

export interface TurnRequestLike {
  messages: unknown
  contextId?: string
  instructions?: string
  history?: unknown
  tools?: unknown
  responseSchema?: string
  tokenBatchSize?: number
}

export function validateTurnRequest(request: TurnRequestLike): void {
  validateTurnMessages(request.messages, 'runTurn messages', { requireNonEmpty: true })
  const hasTools = Array.isArray(request.tools) && request.tools.length > 0
  if (request.responseSchema !== undefined) {
    assertNonEmptyString(request.responseSchema, 'runTurn responseSchema')
    validateToolSchemas(
      [{ name: '__schema__', description: '-', parameters: request.responseSchema }],
      'runTurn responseSchema',
    )
    if (hasTools) {
      throw new TypeError('runTurn responseSchema is exclusive with tools')
    }
  }
  if (request.contextId !== undefined) {
    assertNonEmptyString(request.contextId, 'runTurn contextId')
    if (request.instructions !== undefined || request.history !== undefined || hasTools) {
      throw new TypeError(
        'runTurn instructions, history, and tools are cold-turn fields; remove them or remove contextId',
      )
    }
  }
  if (hasTools) {
    validateToolSchemas(request.tools, 'runTurn tools')
  }
  if (request.history !== undefined) {
    validateTurnMessages(request.history, 'runTurn history', {})
  }
}

export function validateTurnContextOptions(options: {
  instructions?: string
  history?: unknown
  tools?: unknown
}): void {
  if (options.instructions !== undefined) {
    assertNonEmptyString(options.instructions, 'createContext instructions')
  }
  if (options.history !== undefined) {
    validateTurnMessages(options.history, 'createContext history', {})
  }
  if (options.tools !== undefined) {
    validateToolSchemas(options.tools, 'createContext tools')
  }
}

export function validateTokenCountRequest(request: {
  contextId?: string
  instructions?: string
  history?: unknown
  tools?: unknown
  messages?: unknown
}): void {
  if (request.contextId !== undefined) {
    assertNonEmptyString(request.contextId, 'countTokens contextId')
  }
  if (request.history !== undefined) {
    validateTurnMessages(request.history, 'countTokens history', {})
  }
  if (request.messages !== undefined) {
    validateTurnMessages(request.messages, 'countTokens messages', {})
  }
  if (request.tools !== undefined) {
    validateToolSchemas(request.tools, 'countTokens tools')
  }
}
```

- [ ] **Step 4: Run tests and typecheck**

Run: `bun --cwd package test && bun --cwd package typecheck`
Expected: PASS. Do NOT run `bun specs`.

- [ ] **Step 5: Commit**

```bash
git add package/src/specs/LLM.nitro.ts package/src/runtime.ts package/src/turn.test.ts
git commit -m "feat: add turn-scoped wire types and request validators"
```

---

### Task 5: `TurnContextRegistry` (pure Swift)

**Files:**

- Create: `package/ios/Sources/TurnContextRegistry.swift`
- Create: `package/ios/Tests/TurnContextRegistryTests.swift`
- Modify: `package/package.json` (add `test:ios-turn-registry`)

**Interfaces:**

- Produces: `TurnContextRegistry<Entry>` with `insert(_:) -> String`, `entry(for:)`, `update(_:with:)`, `release(_:) -> Entry?`, `releaseAll() -> [Entry]`, `ids: [String]`, `count: Int`. Task 8 instantiates it with `Entry == TurnContextEntry` (which holds a `ChatSession`); keeping it generic keeps this file Foundation-only.

- [ ] **Step 1: Write the failing test**

`package/ios/Tests/TurnContextRegistryTests.swift`:

```swift
import Foundation

var failures = 0
func expect(_ condition: Bool, _ message: String) {
    if !condition { failures += 1; print("FAIL: \(message)") } else { print("PASS: \(message)") }
}

var registry = TurnContextRegistry<String>()

let a = registry.insert("session-a")
let b = registry.insert("session-b")
expect(a != b, "ids are unique")
expect(a.hasPrefix("ctx-"), "ids carry the ctx- prefix")
expect(registry.count == 2, "count tracks inserts")
expect(registry.ids == [a, b].sorted(), "ids are sorted")
expect(registry.entry(for: a) == "session-a", "lookup returns the entry")
expect(registry.entry(for: "ctx-999") == nil, "unknown id returns nil")

registry.update(a, with: "session-a2")
expect(registry.entry(for: a) == "session-a2", "update replaces the entry")

expect(registry.release(a) == "session-a2", "release returns the entry")
expect(registry.release(a) == nil, "double release returns nil (idempotent)")
expect(registry.count == 1, "release shrinks count")

let drained = registry.releaseAll()
expect(drained == ["session-b"], "releaseAll returns remaining entries")
expect(registry.count == 0, "releaseAll empties the registry")

if failures > 0 { exit(1) }
print("All TurnContextRegistry tests passed")
```

- [ ] **Step 2: Add script and verify failure**

```json
"test:ios-turn-registry": "swiftc ios/Sources/TurnContextRegistry.swift ios/Tests/TurnContextRegistryTests.swift -o /tmp/TurnContextRegistryTests && /tmp/TurnContextRegistryTests",
```

Run: `bun --cwd package test:ios-turn-registry`
Expected: FAIL — `cannot find 'TurnContextRegistry' in scope`

- [ ] **Step 3: Implement**

`package/ios/Sources/TurnContextRegistry.swift`:

```swift
import Foundation

/// Bookkeeping for Turn Contexts. Generic over the entry type so this file
/// stays Foundation-only and testable with the standalone swiftc pattern;
/// HybridLLM instantiates it with an entry holding a ChatSession.
struct TurnContextRegistry<Entry> {
    private var entries: [String: Entry] = [:]
    private var nextId: UInt64 = 0

    var count: Int { entries.count }
    var ids: [String] { entries.keys.sorted() }

    mutating func insert(_ entry: Entry) -> String {
        nextId += 1
        let id = "ctx-\(nextId)"
        entries[id] = entry
        return id
    }

    func entry(for id: String) -> Entry? {
        entries[id]
    }

    mutating func update(_ id: String, with entry: Entry) {
        guard entries[id] != nil else { return }
        entries[id] = entry
    }

    @discardableResult
    mutating func release(_ id: String) -> Entry? {
        entries.removeValue(forKey: id)
    }

    mutating func releaseAll() -> [Entry] {
        let sortedIds = ids
        let released = sortedIds.compactMap { entries[$0] }
        entries.removeAll()
        return released
    }
}
```

- [ ] **Step 4: Run the test**

Run: `bun --cwd package test:ios-turn-registry`
Expected: `All TurnContextRegistry tests passed`

- [ ] **Step 5: Commit**

```bash
git add package/ios/Sources/TurnContextRegistry.swift package/ios/Tests/TurnContextRegistryTests.swift package/package.json
git commit -m "feat(ios): add Turn Context registry bookkeeping"
```

---

### Task 6: `TurnRequestPlanner` (pure Swift)

**Files:**

- Create: `package/ios/Sources/TurnRequestPlanner.swift`
- Create: `package/ios/Tests/TurnRequestPlannerTests.swift`
- Modify: `package/package.json` (add `test:ios-turn-planner`)

**Interfaces:**

- Produces (Foundation-only mirror types, mapped from Nitro types by HybridLLM in Task 8):
  - `TurnMessagePlan { role, content, toolCallId, name, isError, toolCallsJson }`
  - `TurnPlanError: Error, Equatable` — cases listed in Step 3
  - `TurnRequestPlanner.plan(messages:contextId:contextKnown:contextHasTools:pendingToolCallIds:hasColdFields:requestHasTools:hasResponseSchema:) throws -> TurnPlan`
  - `TurnPlan { mode: .cold | .warm, providedToolCallIds: [String] }`
  - `TurnRequestPlanner.clampToolContinuations(_:) -> Int` is NOT here — runTurn has no continuation loop. Numeric clamping stays with `normalizedInt` in HybridLLM.

- [ ] **Step 1: Write the failing test**

`package/ios/Tests/TurnRequestPlannerTests.swift`:

```swift
import Foundation

var failures = 0
func expect(_ condition: Bool, _ message: String) {
    if !condition { failures += 1; print("FAIL: \(message)") } else { print("PASS: \(message)") }
}
func expectThrows(_ expected: TurnPlanError, _ message: String, _ body: () throws -> Void) {
    do {
        try body()
        failures += 1
        print("FAIL: \(message) — no error thrown")
    } catch let error as TurnPlanError {
        expect(error == expected, "\(message) — got \(error)")
    } catch {
        failures += 1
        print("FAIL: \(message) — wrong error type \(error)")
    }
}

func msg(
    _ role: String, _ content: String = "x",
    toolCallId: String? = nil, toolCallsJson: String? = nil
) -> TurnMessagePlan {
    TurnMessagePlan(
        role: role, content: content, toolCallId: toolCallId,
        name: nil, isError: nil, toolCallsJson: toolCallsJson
    )
}

// Empty messages reject
expectThrows(.emptyMessages, "empty messages reject") {
    _ = try TurnRequestPlanner.plan(
        messages: [], contextId: nil, contextKnown: false, contextHasTools: false,
        pendingToolCallIds: [], hasColdFields: false,
        requestHasTools: false, hasResponseSchema: false
    )
}

// Unknown role rejects
expectThrows(.unknownRole("oracle"), "unknown role rejects") {
    _ = try TurnRequestPlanner.plan(
        messages: [msg("oracle")], contextId: nil, contextKnown: false, contextHasTools: false,
        pendingToolCallIds: [], hasColdFields: false,
        requestHasTools: false, hasResponseSchema: false
    )
}

// Tool message without id rejects
expectThrows(.missingToolCallId, "tool message without id rejects") {
    _ = try TurnRequestPlanner.plan(
        messages: [msg("tool")], contextId: nil, contextKnown: false, contextHasTools: false,
        pendingToolCallIds: [], hasColdFields: false,
        requestHasTools: false, hasResponseSchema: false
    )
}

// Unknown context rejects
expectThrows(.unknownContext("ctx-9"), "unknown context rejects") {
    _ = try TurnRequestPlanner.plan(
        messages: [msg("user")], contextId: "ctx-9", contextKnown: false, contextHasTools: false,
        pendingToolCallIds: [], hasColdFields: false,
        requestHasTools: false, hasResponseSchema: false
    )
}

// Cold fields on a warm request reject
expectThrows(.coldFieldsOnWarmTurn, "cold fields on warm request reject") {
    _ = try TurnRequestPlanner.plan(
        messages: [msg("user")], contextId: "ctx-1", contextKnown: true, contextHasTools: false,
        pendingToolCallIds: [], hasColdFields: true,
        requestHasTools: false, hasResponseSchema: false
    )
}

// responseSchema vs request tools rejects
expectThrows(.schemaExclusiveWithTools, "schema with request tools rejects") {
    _ = try TurnRequestPlanner.plan(
        messages: [msg("user")], contextId: nil, contextKnown: false, contextHasTools: false,
        pendingToolCallIds: [], hasColdFields: false,
        requestHasTools: true, hasResponseSchema: true
    )
}

// responseSchema vs context tools rejects
expectThrows(.schemaExclusiveWithTools, "schema with context tools rejects") {
    _ = try TurnRequestPlanner.plan(
        messages: [msg("user")], contextId: "ctx-1", contextKnown: true, contextHasTools: true,
        pendingToolCallIds: [], hasColdFields: false,
        requestHasTools: false, hasResponseSchema: true
    )
}

// Pending tool calls: fewer results reject
expectThrows(.incompleteToolResults(missing: ["c2"]), "missing tool result rejects") {
    _ = try TurnRequestPlanner.plan(
        messages: [msg("tool", toolCallId: "c1")],
        contextId: "ctx-1", contextKnown: true, contextHasTools: true,
        pendingToolCallIds: ["c1", "c2"], hasColdFields: false,
        requestHasTools: false, hasResponseSchema: false
    )
}

// Pending tool calls: unknown id rejects
expectThrows(.unknownToolCallId("cX"), "unknown tool result id rejects") {
    _ = try TurnRequestPlanner.plan(
        messages: [msg("tool", toolCallId: "cX")],
        contextId: "ctx-1", contextKnown: true, contextHasTools: true,
        pendingToolCallIds: ["c1"], hasColdFields: false,
        requestHasTools: false, hasResponseSchema: false
    )
}

// Pending tool calls: complete set in any order passes, mode is warm
do {
    let plan = try TurnRequestPlanner.plan(
        messages: [msg("tool", toolCallId: "c2"), msg("tool", toolCallId: "c1")],
        contextId: "ctx-1", contextKnown: true, contextHasTools: true,
        pendingToolCallIds: ["c1", "c2"], hasColdFields: false,
        requestHasTools: false, hasResponseSchema: false
    )
    expect(plan.mode == .warm, "warm plan for context request")
    expect(plan.providedToolCallIds.sorted() == ["c1", "c2"], "provided ids collected")
} catch {
    failures += 1
    print("FAIL: complete tool results should pass — \(error)")
}

// Minimal cold request passes
do {
    let plan = try TurnRequestPlanner.plan(
        messages: [msg("user")], contextId: nil, contextKnown: false, contextHasTools: false,
        pendingToolCallIds: [], hasColdFields: true,
        requestHasTools: true, hasResponseSchema: false
    )
    expect(plan.mode == .cold, "cold plan without context")
} catch {
    failures += 1
    print("FAIL: minimal cold request should pass — \(error)")
}

if failures > 0 { exit(1) }
print("All TurnRequestPlanner tests passed")
```

- [ ] **Step 2: Add script and verify failure**

```json
"test:ios-turn-planner": "swiftc ios/Sources/TurnRequestPlanner.swift ios/Tests/TurnRequestPlannerTests.swift -o /tmp/TurnRequestPlannerTests && /tmp/TurnRequestPlannerTests",
```

Run: `bun --cwd package test:ios-turn-planner`
Expected: FAIL — types not found

- [ ] **Step 3: Implement**

`package/ios/Sources/TurnRequestPlanner.swift`:

```swift
import Foundation

/// Foundation-only mirror of the wire turn message, so validation is
/// testable without MLX or Nitro. HybridLLM maps the generated type into this.
struct TurnMessagePlan {
    let role: String
    let content: String
    let toolCallId: String?
    let name: String?
    let isError: Bool?
    let toolCallsJson: String?
}

enum TurnPlanError: Error, Equatable {
    case emptyMessages
    case unknownRole(String)
    case missingToolCallId
    case toolCallsOnNonAssistant
    case unknownContext(String)
    case coldFieldsOnWarmTurn
    case schemaExclusiveWithTools
    case incompleteToolResults(missing: [String])
    case unknownToolCallId(String)
}

enum TurnMode: Equatable {
    case cold
    case warm
}

struct TurnPlan: Equatable {
    let mode: TurnMode
    let providedToolCallIds: [String]
}

enum TurnRequestPlanner {
    private static let roles: Set<String> = ["system", "user", "assistant", "tool"]

    static func plan(
        messages: [TurnMessagePlan],
        contextId: String?,
        contextKnown: Bool,
        contextHasTools: Bool,
        pendingToolCallIds: [String],
        hasColdFields: Bool,
        requestHasTools: Bool,
        hasResponseSchema: Bool
    ) throws -> TurnPlan {
        guard !messages.isEmpty else { throw TurnPlanError.emptyMessages }

        var providedToolCallIds: [String] = []
        for message in messages {
            guard roles.contains(message.role) else {
                throw TurnPlanError.unknownRole(message.role)
            }
            if message.role == "tool" {
                guard let id = message.toolCallId, !id.isEmpty else {
                    throw TurnPlanError.missingToolCallId
                }
                providedToolCallIds.append(id)
            }
            if message.toolCallsJson != nil, message.role != "assistant" {
                throw TurnPlanError.toolCallsOnNonAssistant
            }
        }

        let mode: TurnMode
        if let contextId {
            guard contextKnown else { throw TurnPlanError.unknownContext(contextId) }
            guard !hasColdFields else { throw TurnPlanError.coldFieldsOnWarmTurn }
            mode = .warm

            // Every Tool Call Request from the previous turn needs a result,
            // matched by id; order is free (spec: Tool call contract).
            let pending = Set(pendingToolCallIds)
            let provided = Set(providedToolCallIds)
            if let stray = provided.subtracting(pending).sorted().first {
                throw TurnPlanError.unknownToolCallId(stray)
            }
            let missing = pending.subtracting(provided).sorted()
            if !missing.isEmpty {
                throw TurnPlanError.incompleteToolResults(missing: missing)
            }
        } else {
            mode = .cold
        }

        if hasResponseSchema {
            let toolsInPlay = requestHasTools || (mode == .warm && contextHasTools)
            if toolsInPlay { throw TurnPlanError.schemaExclusiveWithTools }
        }

        return TurnPlan(mode: mode, providedToolCallIds: providedToolCallIds)
    }
}
```

- [ ] **Step 4: Run the test**

Run: `bun --cwd package test:ios-turn-planner`
Expected: `All TurnRequestPlanner tests passed`.

- [ ] **Step 5: Commit**

```bash
git add package/ios/Sources/TurnRequestPlanner.swift package/ios/Tests/TurnRequestPlannerTests.swift package/package.json
git commit -m "feat(ios): add turn request planner with tool-result completeness rules"
```

---

### Task 7: `ToolSchemaPlanner` (pure Swift)

**Files:**

- Create: `package/ios/Sources/ToolSchemaPlanner.swift`
- Create: `package/ios/Tests/ToolSchemaPlannerTests.swift`
- Modify: `package/package.json` (add `test:ios-tool-schema`)

**Interfaces:**

- Produces: `ToolSchemaPlanner.parseParameters(_ json: String) throws -> [String: Any]` (throws `ToolSchemaError.invalidJSON` / `.rootNotObjectSchema`), and `ToolSchemaPlanner.syntheticTool(responseSchema: String) throws -> (name: String, description: String, parameters: [String: Any])` with the fixed name `respond_with_structured_output`. Task 9 uses `parseParameters` to build upstream `ToolSpec` dictionaries; Task 14 uses `syntheticTool`.

- [ ] **Step 1: Write the failing test**

`package/ios/Tests/ToolSchemaPlannerTests.swift`:

```swift
import Foundation

var failures = 0
func expect(_ condition: Bool, _ message: String) {
    if !condition { failures += 1; print("FAIL: \(message)") } else { print("PASS: \(message)") }
}

// Valid object schema parses
do {
    let schema = try ToolSchemaPlanner.parseParameters(
        #"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}"#
    )
    expect(schema["type"] as? String == "object", "valid schema parses")
    expect((schema["properties"] as? [String: Any])?["path"] != nil, "nested properties survive")
} catch {
    failures += 1
    print("FAIL: valid schema should parse — \(error)")
}

// Invalid JSON throws
do {
    _ = try ToolSchemaPlanner.parseParameters("{oops")
    failures += 1
    print("FAIL: invalid JSON should throw")
} catch ToolSchemaError.invalidJSON {
    print("PASS: invalid JSON throws")
} catch {
    failures += 1
    print("FAIL: wrong error for invalid JSON — \(error)")
}

// Non-object root throws
do {
    _ = try ToolSchemaPlanner.parseParameters(#"{"type":"string"}"#)
    failures += 1
    print("FAIL: non-object root should throw")
} catch ToolSchemaError.rootNotObjectSchema {
    print("PASS: non-object root throws")
} catch {
    failures += 1
    print("FAIL: wrong error for non-object root — \(error)")
}

// Synthetic tool carries the schema and the fixed name
do {
    let tool = try ToolSchemaPlanner.syntheticTool(
        responseSchema: #"{"type":"object","properties":{"answer":{"type":"string"}}}"#
    )
    expect(tool.name == "respond_with_structured_output", "synthetic tool name is fixed")
    expect(tool.description.contains("only"), "description instructs exclusive use")
    expect(tool.parameters["type"] as? String == "object", "synthetic tool carries the schema")
} catch {
    failures += 1
    print("FAIL: synthetic tool should build — \(error)")
}

if failures > 0 { exit(1) }
print("All ToolSchemaPlanner tests passed")
```

- [ ] **Step 2: Add script and verify failure**

```json
"test:ios-tool-schema": "swiftc ios/Sources/ToolSchemaPlanner.swift ios/Tests/ToolSchemaPlannerTests.swift -o /tmp/ToolSchemaPlannerTests && /tmp/ToolSchemaPlannerTests",
```

Run: `bun --cwd package test:ios-tool-schema`
Expected: FAIL

- [ ] **Step 3: Implement**

`package/ios/Sources/ToolSchemaPlanner.swift`:

```swift
import Foundation

enum ToolSchemaError: Error, Equatable {
    case invalidJSON
    case rootNotObjectSchema
}

enum ToolSchemaPlanner {
    /// Parses a serialized JSON Schema and enforces the wire contract:
    /// the root must be an object schema ("type": "object").
    static func parseParameters(_ json: String) throws -> [String: Any] {
        guard let data = json.data(using: .utf8),
            let parsed = try? JSONSerialization.jsonObject(with: data),
            let dictionary = parsed as? [String: Any]
        else {
            throw ToolSchemaError.invalidJSON
        }
        guard dictionary["type"] as? String == "object" else {
            throw ToolSchemaError.rootNotObjectSchema
        }
        return dictionary
    }

    /// Builds the single synthetic tool used for structured output. The model
    /// is asked, not forced, to call it (no tool_choice exists upstream), so
    /// the description restates the shape expectation.
    static func syntheticTool(
        responseSchema: String
    ) throws -> (name: String, description: String, parameters: [String: Any]) {
        let parameters = try parseParameters(responseSchema)
        return (
            name: "respond_with_structured_output",
            description:
                "Return your final answer by calling this tool. It is the only way to respond: "
                + "call it exactly once, with arguments that match its parameter schema.",
            parameters: parameters
        )
    }
}
```

- [ ] **Step 4: Run the test**

Run: `bun --cwd package test:ios-tool-schema`
Expected: `All ToolSchemaPlanner tests passed`

- [ ] **Step 5: Commit**

```bash
git add package/ios/Sources/ToolSchemaPlanner.swift package/ios/Tests/ToolSchemaPlannerTests.swift package/package.json
git commit -m "feat(ios): add tool schema planner and synthetic structured-output tool"
```

---

### Task 8: Nitrogen regeneration, context methods, and the `runTurn` stub

**Files:**

- Modify: `package/src/specs/LLM.nitro.ts` (none — types landed in Task 4)
- Modify: `package/ios/Sources/HybridLLM.swift` (context entry type, registry, new protocol methods)
- Generated: `package/nitrogen/**` (via `bun specs`)

**Interfaces:**

- Consumes: `TurnContextRegistry` (Task 5), `resetTurnConfiguration` (Task 1), generated Nitro types `LLMTurnContextOptions`, `LLMTurnRequest`, `LLMTurnOutcome` etc. (Task 4).
- Produces: `TurnContextEntry` struct; working `createTurnContext` / `releaseTurnContext` / `releaseAllTurnContexts` / `turnContextIds`; `runTurn` and `countTokens` stubs that reject with `LLMError.generationFailed(stage: "prepare", ...)`; a compiling generated protocol. Tasks 9/10/13/14 replace the stubs.

- [ ] **Step 1: Regenerate and observe the build break**

Run: `bun --cwd package specs`
Expected: nitrogen regenerates, then the Swift protocol gains the six new members and `HybridLLM` no longer conforms. (If nitrogen itself errors on a type, fix the wire type in `LLM.nitro.ts` — nested optional arrays of flat structs are supported; unions of string literals are supported as enums.)

- [ ] **Step 2: Add the context entry and registry state to `HybridLLM`**

Near the other private state (`:361-391`):

```swift
    /// One Turn Context: a retained ChatSession plus the bookkeeping the
    /// session cannot expose. `transcript` mirrors the session's messages so
    /// the context can be rebuilt after a cancelled turn and counted by
    /// countTokens; `pendingToolCallIds` enforces the tool-result contract.
    struct TurnContextEntry {
        var session: ChatSession
        let instructions: String?
        let toolSpecs: [ToolSpec]
        let toolNames: Set<String>
        let parameters: GenerateParameters
        var transcript: [Chat.Message]
        var pendingToolCallIds: [String]
        var needsRebuild: Bool
    }

    private var turnContexts = TurnContextRegistry<TurnContextEntry>()
```

- [ ] **Step 3: Implement the context methods on the hybrid class**

Follow the existing pattern of the public wrapper section (`:80-176`) where the Nitro-facing methods bridge onto the `@MainActor` core:

```swift
    func createTurnContext(options: LLMTurnContextOptions?) throws -> Promise<String> {
        Promise.async { @MainActor in
            try self.core.createTurnContext(options: options)
        }
    }

    func releaseTurnContext(id: String) {
        MainActorSync.write { self.core.releaseTurnContext(id: id) }
    }

    func releaseAllTurnContexts() {
        MainActorSync.write { self.core.releaseAllTurnContexts() }
    }

    var turnContextIds: [String] {
        MainActorSync.read { self.core.turnContextIds }
    }
```

(Adapt the bridge helpers to whatever this section actually uses — read `:80-176` first and copy its idiom exactly; the names `MainActorSync.read/write` and `core` above follow `systemPrompt` at `:93-99`.)

Core implementations:

```swift
    func createTurnContext(options: LLMTurnContextOptions?) throws -> String {
        guard let container else { throw LLMError.notLoaded }

        var toolSpecs: [ToolSpec] = []
        var toolNames: Set<String> = []
        for tool in options?.tools ?? [] {
            let parameters = try ToolSchemaPlanner.parseParameters(tool.parameters)
            toolSpecs.append([
                "type": "function",
                "function": [
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                ],
            ])
            toolNames.insert(tool.name)
        }

        let history = try chatMessagesFromTurnMessages(options?.history ?? [])
        let parameters = buildGenerateParameters(from: options?.generationConfig)
        let session = ChatSession(
            container,
            instructions: options?.instructions,
            history: history,
            generateParameters: parameters,
            tools: toolSpecs.isEmpty ? nil : toolSpecs
        )

        let entry = TurnContextEntry(
            session: session,
            instructions: options?.instructions,
            toolSpecs: toolSpecs,
            toolNames: toolNames,
            parameters: parameters,
            transcript: history,
            pendingToolCallIds: [],
            needsRebuild: false
        )
        return turnContexts.insert(entry)
    }

    func releaseTurnContext(id: String) {
        turnContexts.release(id)
    }

    func releaseAllTurnContexts() {
        _ = turnContexts.releaseAll()
    }

    var turnContextIds: [String] { turnContexts.ids }
```

Note: `ChatSession(container, instructions:history:...)` is the init at `ChatSession.swift:246` (3.31.4). If the resolved version's history-taking init differs, match its signature — the checkout is at `example/ios/build/SourcePackages/checkouts/mlx-swift-lm`.

- [ ] **Step 4: Add the message-mapping helper and the stubs**

```swift
    /// Maps wire turn messages to upstream chat messages. Assistant tool calls
    /// ride in toolCallsJson as [{id, name, arguments}]; upstream renders them
    /// model-agnostically via addToolMetadata (Chat.swift:137-158).
    private func chatMessagesFromTurnMessages(_ messages: [LLMTurnMessage]) throws -> [Chat.Message] {
        try messages.map { message in
            switch message.role {
            case "system": return .system(message.content)
            case "user": return .user(message.content)
            case "assistant":
                guard let json = message.toolCallsJson, !json.isEmpty else {
                    return .assistant(message.content)
                }
                return .assistant(message.content, toolCalls: try parseWireToolCalls(json))
            case "tool":
                return .tool(message.content, id: message.toolCallId)
            default:
                throw LLMError.generationFailed(stage: "prepare", message: "Unknown role \(message.role)")
            }
        }
    }

    private func parseWireToolCalls(_ json: String) throws -> [ToolCall] {
        guard let data = json.data(using: .utf8),
            let array = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        else {
            throw LLMError.generationFailed(stage: "prepare", message: "toolCallsJson is not valid JSON")
        }
        return array.map { entry in
            ToolCall(
                function: .init(
                    name: entry["name"] as? String ?? "",
                    arguments: jsonValueDictionary(from: entry["arguments"] as? [String: Any] ?? [:])
                ),
                id: entry["id"] as? String
            )
        }
    }

    private func jsonValueDictionary(from dictionary: [String: Any]) -> [String: JSONValue] {
        dictionary.mapValues { jsonValue(from: $0) }
    }

    private func jsonValue(from value: Any) -> JSONValue {
        switch value {
        case let string as String: return .string(string)
        case let bool as Bool: return .bool(bool)
        case let int as Int: return .int(int)
        case let double as Double: return .double(double)
        case let array as [Any]: return .array(array.map { jsonValue(from: $0) })
        case let dict as [String: Any]: return .object(jsonValueDictionary(from: dict))
        default: return .null
        }
    }
```

Check `ToolCall` / `JSONValue` member names against the checkout (`Libraries/MLXLMCommon/Tool/ToolCall.swift`) and adjust case names to what the enum actually declares (`.int` vs `.number` etc.). The mapping logic stands; the case names follow upstream.

Stubs:

```swift
    func runTurn(
        request: LLMTurnRequest,
        onEvent: @escaping (StreamEventEnvelope) -> Void
    ) async throws -> LLMTurnOutcome {
        throw LLMError.generationFailed(stage: "prepare", message: "runTurn not implemented yet")
    }

    func countTokens(request: LLMTokenCountRequest) async throws -> Double {
        throw LLMError.generationFailed(stage: "prepare", message: "countTokens not implemented yet")
    }
```

(Signatures must match what nitrogen generated — read the generated Swift protocol in `package/nitrogen/generated/ios/` and conform exactly, including `Promise` wrappers.)

- [ ] **Step 5: Release contexts on load and unload**

In `load()` (both branches, before configuration) and in `resetModelState()`, add:

```swift
        releaseAllTurnContexts()
```

Contexts belong to the Resident Model; `load()` is a reset (spec: Model residency).

- [ ] **Step 6: Verify the build and tests**

Run: `bun --cwd package specs && bun --cwd package test && bun --cwd package test:ios-load-plan && bun --cwd package test:ios-turn-registry`

Then verify the example app compiles for simulator:

```bash
cd example && LANG=en_US.UTF-8 bunx pod-install
xcodebuild -workspace ios/nitromlxexample.xcworkspace -scheme nitromlxexample -sdk iphonesimulator -configuration Debug build | tail -5
```

(Read `example/ios/` for the actual workspace and scheme names before running; substitute them.)
Expected: build succeeds.

- [ ] **Step 7: Commit**

```bash
git add package/src/specs/LLM.nitro.ts package/ios/Sources/HybridLLM.swift package/nitrogen package/lib 2>/dev/null || git add -A package
git commit -m "feat(ios): Turn Context registry methods and runTurn scaffolding across the bridge"
```

---

### Task 9: `runTurn` cold path

**Files:**

- Modify: `package/ios/Sources/HybridLLM.swift` (replace the `runTurn` stub; add `TurnGenerationSink` usage)

**Interfaces:**

- Consumes: `TurnRequestPlanner.plan` (Task 6), `ToolSchemaPlanner.parseParameters` (Task 7), `chatMessagesFromTurnMessages` (Task 8), existing `EventGenerationSink` internals (`:264`), `ThinkingStateMachine`, `TokenBatcher`, `generationTasks` guard.
- Produces: a working cold `runTurn` returning `LLMTurnOutcome` with `toolCalls`, split `usage`, and stream events. Task 10 adds the warm branch inside the same function.

- [ ] **Step 1: Implement the shared turn runner**

Replace the stub with a function that:

1. Guards `acceptsGeneration`, `container != nil`, `ensureNotGenerating()` — same preamble as `beginTurn` (`:878-887`).
2. Maps wire messages to `[TurnMessagePlan]` and calls `TurnRequestPlanner.plan(...)` with `contextId: request.contextId`, `contextKnown: request.contextId.map { turnContexts.entry(for: $0) != nil } ?? false`, `contextHasTools`/`pendingToolCallIds` from the entry, `hasColdFields: request.instructions != nil || request.history != nil || !(request.tools ?? []).isEmpty`, `requestHasTools`, `hasResponseSchema: request.responseSchema != nil`. Planner errors map to `LLMError.generationFailed(stage: "prepare", message: ...)` and reject BEFORE `generationTasks.begin` — preflight failures reject, started turns resolve (ADR 0001).
3. For the cold mode (this task): builds tool specs from `request.tools` via `ToolSchemaPlanner.parseParameters`, builds a throwaway `ChatSession(container, instructions: request.instructions, history: chatMessagesFromTurnMessages(request.history ?? []), generateParameters: buildGenerateParameters(from: request.generationConfig), tools: specs, toolDispatch: nil)`.
4. Registers the turn with `generationTasks.begin` (same shape as `beginTurn` `:889-897`) so `stop()` and exclusion work.
5. Drives `session.streamDetails(to: chatMessagesFromTurnMessages(request.messages))` and folds the stream:

```swift
    private struct TurnAccumulation {
        var toolCalls: [ToolCall] = []
        var promptTokens = 0
        var completionTokens = 0
    }

    private func collectTurn(
        session: ChatSession,
        inputMessages: [Chat.Message],
        sink: GenerationSink,
        progress: GenerationProgress
    ) async throws -> TurnAccumulation {
        var accumulation = TurnAccumulation()
        for try await item in session.streamDetails(to: inputMessages) {
            try Task.checkCancellation()
            switch item {
            case .chunk(let text):
                progress.append(text)
                sink.emitToken(text)
            case .info(let info):
                accumulation.promptTokens += info.promptTokenCount
                accumulation.completionTokens += info.generationTokenCount
            case .toolCall(let call):
                accumulation.toolCalls.append(call)
                sink.emitToolCallStart(call)
            }
        }
        return accumulation
    }
```

(`sink.emitToken` / `emitToolCallStart` name whatever the existing sinks call for token and `tool_call_start` envelope emission — read `EventGenerationSink` at `:264` and route through the same methods, including the `ThinkingStateMachine` handling tokens already get. `Generation`'s exact case payloads are at `Evaluate.swift:2052-2085`.)

6. Builds the outcome:

```swift
        let finishReason: LLMTurnFinishReason
        var rawFinishReason: String? = nil
        if !accumulation.toolCalls.isEmpty {
            finishReason = .toolCalls   // generated enum case for 'tool_calls'
        } else if let maxTokens = effectiveParameters.maxTokens,
            accumulation.completionTokens >= maxTokens {
            finishReason = .length
            rawFinishReason = "maxTokens"
        } else {
            finishReason = .completed
        }

        let accumulatedUsage = LLMTurnUsage(
            promptTokens: Double(accumulation.promptTokens),
            completionTokens: Double(accumulation.completionTokens),
            cachedPromptTokens: nil   // warm turns override this in Task 10
        )

        let outcome = LLMTurnOutcome(
            finishReason: finishReason,
            rawFinishReason: rawFinishReason,
            content: progress.content,
            thinking: sink.thinkingContent.isEmpty ? nil : sink.thinkingContent,
            toolCalls: accumulation.toolCalls.map(wireToolCall),
            usage: accumulatedUsage,
            stats: makeStats(startTime: startTime, progress: progress),
            error: nil,
            stage: nil
        )
```

With the wire mapper:

```swift
    private func wireToolCall(_ call: ToolCall) -> LLMToolCallWire {
        let argumentsData = (try? JSONSerialization.data(
            withJSONObject: anyDictionary(from: call.function.arguments)
        )) ?? Data("{}".utf8)
        return LLMToolCallWire(
            id: call.id ?? "call-\(UInt64.random(in: 0..<UInt64.max))",
            name: call.function.name,
            argumentsJson: String(decoding: argumentsData, as: UTF8.self)
        )
    }

    private func anyDictionary(from values: [String: JSONValue]) -> [String: Any] {
        values.mapValues { anyValue(from: $0) }
    }

    private func anyValue(from value: JSONValue) -> Any {
        switch value {
        case .string(let s): return s
        case .bool(let b): return b
        case .int(let i): return i
        case .double(let d): return d
        case .array(let a): return a.map { anyValue(from: $0) }
        case .object(let o): return anyDictionary(from: o)
        case .null: return NSNull()
        }
    }
```

(Again: match `JSONValue`'s actual cases and `ToolCall`'s actual `id` availability against the checkout. If upstream `ToolCall` has no `id`, generate one — the spec makes `id` total. If the generated finish-reason enum spells its cases differently, follow the generated code.)

7. Cancellation: catch `CancellationError`, map `generationTasks.cancellationReason` to `.stopped` / `.superseded` / `.unloaded` with `rawFinishReason` set to the reason's raw name, mirroring `performTurn`'s catch (`:1023-1048`) minus the managed-history branches. Other errors resolve as `.failed` with `stage: "generate"`.

- [ ] **Step 2: Verify the build**

Run: `bun --cwd package specs` and the example simulator build from Task 8 Step 6.
Expected: builds. All `bun --cwd package test` and `test:ios-*` scripts still pass.

- [ ] **Step 3: Manual device check**

On device (example app; ad-hoc via the Turn Lab screen once Task 15 lands, or a temporary button): a cold `runTurn` with one tool schema returns `finishReason: 'tool_calls'` with parseable `argumentsJson`, and a cold turn without tools returns `completed` with content and non-zero `usage.promptTokens`/`completionTokens`.

- [ ] **Step 4: Commit**

```bash
git add package/ios/Sources/HybridLLM.swift package/nitrogen
git commit -m "feat(ios): cold runTurn path returning Tool Call Requests with split usage"
```

---

### Task 10: `runTurn` warm path, pending-tool bookkeeping, and cancellation rebuild

**Files:**

- Modify: `package/ios/Sources/HybridLLM.swift` (warm branch in `runTurn`)

**Interfaces:**

- Consumes: `TurnContextEntry` (Task 8), the cold-path machinery (Task 9).
- Produces: warm turns over a registry `ChatSession`; `pendingToolCallIds` written after a `tool_calls` outcome and cleared on results; `needsRebuild` set on cancellation and honored on the next turn; transcript mirroring for rebuild and `countTokens`.

- [ ] **Step 1: Implement the warm branch**

In `runTurn`, when the plan mode is `.warm` (entry fetched during planning):

1. If `entry.needsRebuild`: build a fresh `ChatSession(container, instructions: entry.instructions, history: entry.transcript, generateParameters: entry.parameters, tools: entry.toolSpecs.isEmpty ? nil : entry.toolSpecs, toolDispatch: nil)`, set `entry.session` to it, clear `needsRebuild`, and `turnContexts.update(id, with: entry)`.
2. Use `entry.session` and `entry.parameters` instead of building a throwaway session. `request.generationConfig` is nil here by validation (cold-only field).
3. Drive `collectTurn` with the mapped `request.messages`.
4. On a terminal outcome, commit bookkeeping:

```swift
        // Mirror what the session appended, so rebuild and countTokens see
        // the same transcript the KV cache encodes.
        entry.transcript.append(contentsOf: inputChatMessages)
        if !progress.content.isEmpty || !accumulation.toolCalls.isEmpty {
            entry.transcript.append(
                .assistant(progress.content, toolCalls: accumulation.toolCalls.isEmpty ? nil : accumulation.toolCalls)
            )
        }
        entry.pendingToolCallIds = accumulation.toolCalls.map { wireToolCall($0).id }
        turnContexts.update(contextId, with: entry)
```

Keep the wire id generation deterministic per call: compute `wireToolCalls = accumulation.toolCalls.map(wireToolCall)` ONCE and reuse for both the outcome and `pendingToolCallIds` (random fallback ids must match between the two).

5. On cancellation (`CancellationError` catch): set `entry.needsRebuild = true`, leave `transcript` and `pendingToolCallIds` as they were before the turn (do not append partial content), `turnContexts.update`, then return the cancelled outcome as in Task 9.

- [ ] **Step 2: Warm usage — `cachedPromptTokens`**

In the warm branch, `info.promptTokenCount` covers only the newly appended messages (`ChatSession` feeds the iterator just the delta). Add to `TurnContextEntry` (Task 8 struct):

```swift
        /// Running total of tokens already encoded in the warm KV cache:
        /// the sum of prompt and completion tokens of every committed turn.
        /// An estimate — exact only while no trimming has occurred
        /// (spec: open question 4).
        var promptTokensSeen: Int
```

Initialize it to 0 in `createTurnContext`. On each committed warm turn, before building the outcome, set `cachedPromptTokens: Double(entry.promptTokensSeen)` in `accumulatedUsage`, then increment `entry.promptTokensSeen += accumulation.promptTokens + accumulation.completionTokens` as part of the bookkeeping commit. Cold turns keep `cachedPromptTokens: nil`.

- [ ] **Step 3: Verify build + tests**

Run: `bun --cwd package specs`, example simulator build, all `test:ios-*` scripts, `bun --cwd package test`.

- [ ] **Step 4: Manual device check**

Full loop on device: user message → `tool_calls` outcome → `runTurn` again with matching tool results → `completed` outcome whose content uses the results. Then: missing one tool result rejects; `stop()` mid-turn resolves `stopped`; the next turn on the same context succeeds (rebuild path); warm turn 2 reports `cachedPromptTokens > 0` and smaller `promptTokens` than an equivalent cold turn.

- [ ] **Step 5: Commit**

```bash
git add package/ios/Sources/HybridLLM.swift
git commit -m "feat(ios): warm runTurn path with tool-result bookkeeping and cancellation rebuild"
```

---

### Task 11: Public TypeScript wrapper (`turn.ts`)

**Files:**

- Create: `package/src/turn.ts`
- Modify: `package/src/llm.ts` (attach `runTurn`, `createContext`, `countTokens` etc. to the `LLM` object)
- Modify: `package/src/index.ts` (export public types)
- Test: `package/src/turn.test.ts` (extend)

**Interfaces:**

- Consumes: native methods (Task 8–10 signatures), validators (Task 4), `mapStreamEventEnvelope` and `createSafeCallback` from `runtime.ts`.
- Produces the public spec surface:
  - `type LLMMessage` (discriminated union), `interface LLMToolCall { id; name; arguments: Record<string, unknown> }`, `interface ToolSchema`, `interface LLMTurnUsage`, `interface LLMTurnOutcome` (public, parsed `toolCalls`), `interface LLMTurnRequest` (public), `interface LLMContext { id; release() }`
  - Mapping functions (exported for tests): `toWireMessage(message: LLMMessage): LLMTurnMessage`, `fromWireOutcome(outcome: WireTurnOutcome): LLMTurnOutcome`
  - `LLM.runTurn(request, onEvent)`, `LLM.createContext(options?)`, `LLM.releaseContext(id)`, `LLM.releaseAllContexts()`, `LLM.contextIds`, `LLM.countTokens(request)`

- [ ] **Step 1: Write failing mapping tests**

Append to `package/src/turn.test.ts`:

```ts
const { toWireMessage, fromWireOutcome } = await import('./turn')

describe('toWireMessage', () => {
  it('serializes assistant tool calls to JSON', () => {
    const wire = toWireMessage({
      role: 'assistant',
      content: 'calling',
      toolCalls: [{ id: 'c1', name: 'f', arguments: { path: '/a' } }],
    })
    expect(wire.role).toBe('assistant')
    expect(JSON.parse(wire.toolCallsJson ?? '')).toEqual([
      { id: 'c1', name: 'f', arguments: { path: '/a' } },
    ])
  })

  it('passes tool result fields through flat', () => {
    const wire = toWireMessage({
      role: 'tool',
      toolCallId: 'c1',
      name: 'f',
      content: 'ok',
      isError: true,
    })
    expect(wire).toEqual({
      role: 'tool',
      content: 'ok',
      toolCallId: 'c1',
      name: 'f',
      isError: true,
      toolCallsJson: undefined,
    })
  })
})

describe('fromWireOutcome', () => {
  const base = {
    finishReason: 'tool_calls' as const,
    content: '',
    toolCalls: [{ id: 'c1', name: 'f', argumentsJson: '{"path":"/a"}' }],
    usage: { promptTokens: 10, completionTokens: 5 },
    stats: {
      tokenCount: 5,
      tokensPerSecond: 1,
      timeToFirstToken: 1,
      totalTime: 1,
      toolExecutionTime: 0,
    },
  }

  it('parses tool call arguments to objects', () => {
    const outcome = fromWireOutcome(base)
    expect(outcome.toolCalls[0]?.arguments).toEqual({ path: '/a' })
  })

  it('degrades unparseable arguments to an empty object', () => {
    const outcome = fromWireOutcome({
      ...base,
      toolCalls: [{ id: 'c1', name: 'f', argumentsJson: '{broken' }],
    })
    expect(outcome.toolCalls[0]?.arguments).toEqual({})
  })
})
```

Run: `bun --cwd package test` — expected FAIL (`./turn` does not exist).

- [ ] **Step 2: Implement `turn.ts`**

```ts
import type {
  GenerationStats,
  LLMGenerationConfig,
  LLMToolCallWire,
  LLMTokenCountRequest as WireTokenCountRequest,
  LLMTurnContextOptions as WireContextOptions,
  LLMTurnFinishReason,
  LLMTurnMessage,
  LLMTurnOutcome as WireTurnOutcome,
  LLMTurnRequest as WireTurnRequest,
  LLMTurnUsage,
} from './specs/LLM.nitro'
import { safeJsonParse } from './runtime'

export interface LLMToolCall {
  id: string
  name: string
  /** Parsed arguments. The native parser produced these; malformed output never reaches here. */
  arguments: Record<string, unknown>
}

export type LLMMessage =
  | { role: 'system' | 'user'; content: string }
  | { role: 'assistant'; content: string; toolCalls?: LLMToolCall[] }
  | {
      role: 'tool'
      toolCallId: string
      content: string
      /** Tool name, when known. Not consumed by every chat template. */
      name?: string
      /** True when the tool failed. The model must see failures to recover. */
      isError?: boolean
    }

export interface ToolSchema {
  name: string
  description: string
  /** Serialized JSON Schema. Root must be an object schema. */
  parameters: string
}

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

export interface LLMTurnRequest {
  messages: LLMMessage[]
  contextId?: string
  instructions?: string
  history?: LLMMessage[]
  tools?: ToolSchema[]
  generationConfig?: LLMGenerationConfig
  responseSchema?: string
  tokenBatchSize?: number
}

export interface LLMTurnOutcome {
  finishReason: LLMTurnFinishReason
  rawFinishReason?: string
  content: string
  thinking?: string
  /** Populated when finishReason is 'tool_calls'. Loops should branch on toolCalls.length. */
  toolCalls: LLMToolCall[]
  usage: LLMTurnUsage
  stats: GenerationStats
  error?: string
  stage?: string
}

export interface LLMTokenCountRequest {
  contextId?: string
  instructions?: string
  history?: LLMMessage[]
  tools?: ToolSchema[]
  messages?: LLMMessage[]
}

/** @internal exported for tests */
export function toWireMessage(message: LLMMessage): LLMTurnMessage {
  switch (message.role) {
    case 'assistant':
      return {
        role: 'assistant',
        content: message.content,
        toolCallId: undefined,
        name: undefined,
        isError: undefined,
        toolCallsJson:
          message.toolCalls !== undefined ? JSON.stringify(message.toolCalls) : undefined,
      }
    case 'tool':
      return {
        role: 'tool',
        content: message.content,
        toolCallId: message.toolCallId,
        name: message.name,
        isError: message.isError,
        toolCallsJson: undefined,
      }
    default:
      return {
        role: message.role,
        content: message.content,
        toolCallId: undefined,
        name: undefined,
        isError: undefined,
        toolCallsJson: undefined,
      }
  }
}

function fromWireToolCall(call: LLMToolCallWire): LLMToolCall {
  return {
    id: call.id,
    name: call.name,
    arguments: safeJsonParse<Record<string, unknown>>(call.argumentsJson, {}),
  }
}

/** @internal exported for tests */
export function fromWireOutcome(outcome: WireTurnOutcome): LLMTurnOutcome {
  return {
    finishReason: outcome.finishReason,
    rawFinishReason: outcome.rawFinishReason,
    content: outcome.content,
    thinking: outcome.thinking,
    toolCalls: outcome.toolCalls.map(fromWireToolCall),
    usage: outcome.usage,
    stats: outcome.stats,
    error: outcome.error,
    stage: outcome.stage,
  }
}

/** @internal */
export function toWireRequest(request: LLMTurnRequest): WireTurnRequest {
  return {
    messages: request.messages.map(toWireMessage),
    contextId: request.contextId,
    instructions: request.instructions,
    history: request.history?.map(toWireMessage),
    tools: request.tools,
    generationConfig: request.generationConfig,
    responseSchema: request.responseSchema,
    tokenBatchSize: request.tokenBatchSize,
  }
}

/** @internal */
export function toWireContextOptions(options: LLMContextOptions): WireContextOptions {
  return {
    instructions: options.instructions,
    history: options.history?.map(toWireMessage),
    tools: options.tools,
    generationConfig: options.generationConfig,
  }
}

/** @internal */
export function toWireTokenCountRequest(
  request: LLMTokenCountRequest,
): WireTokenCountRequest {
  return {
    contextId: request.contextId,
    instructions: request.instructions,
    history: request.history?.map(toWireMessage),
    tools: request.tools,
    messages: request.messages?.map(toWireMessage),
  }
}
```

- [ ] **Step 3: Attach methods to the `LLM` object in `llm.ts`**

Import the turn module pieces plus `validateTurnRequest`, `validateTurnContextOptions`, `validateTokenCountRequest`, `mapStreamEventEnvelope`, `createSafeCallback` and add to the `LLM` object:

```ts
  /**
   * Run one LLM Generation Turn. Tool Call Requests come back to the caller;
   * this package executes nothing. Branch your loop on toolCalls.length, not
   * on finishReason.
   */
  async runTurn(
    request: LLMTurnRequest,
    onEvent?: (event: StreamEvent) => void,
  ): Promise<LLMTurnOutcome> {
    validateTurnRequest(toWireRequest(request))
    const safeOnEvent = createSafeCallback('LLM.runTurn onEvent', onEvent)
    const wireOutcome = await getInstance().runTurn(toWireRequest(request), envelope => {
      if (!safeOnEvent) return
      const event = mapStreamEventEnvelope(envelope)
      if (event) safeOnEvent(event)
    })
    return fromWireOutcome(wireOutcome)
  },

  /** Create a Turn Context over the Resident Model. Release it when done. */
  async createContext(options: LLMContextOptions = {}): Promise<LLMContext> {
    validateTurnContextOptions(toWireContextOptions(options))
    const id = await getInstance().createTurnContext(toWireContextOptions(options))
    return {
      id,
      release: () => getInstance().releaseTurnContext(id),
    }
  },

  releaseContext(id: string): void {
    getInstance().releaseTurnContext(assertNonEmptyString(id, 'LLM contextId'))
  },

  releaseAllContexts(): void {
    getInstance().releaseAllTurnContexts()
  },

  get contextIds(): string[] {
    return getInstance().turnContextIds
  },

  /** Count tokens for an assembled prompt with the loaded tokenizer. */
  countTokens(request: LLMTokenCountRequest): Promise<number> {
    validateTokenCountRequest(toWireTokenCountRequest(request))
    return getInstance().countTokens(toWireTokenCountRequest(request))
  },
```

(Validation runs on the wire form so `toolCallsJson` shape checks apply; call `toWireRequest` once and reuse — adjust the code to bind it to a local.)

- [ ] **Step 4: Export from `index.ts`**

```ts
export type {
  LLMContext,
  LLMContextOptions,
  LLMMessage,
  LLMTokenCountRequest,
  LLMToolCall,
  LLMTurnOutcome,
  LLMTurnRequest,
  ToolSchema,
} from './turn'
export type { LLMTurnFinishReason, LLMTurnUsage } from './specs/LLM.nitro'
```

- [ ] **Step 5: Run tests and typecheck**

Run: `bun --cwd package test && bun --cwd package typecheck`
Expected: PASS, including a new spy-based test that `LLM.runTurn` rejects invalid requests before touching native (add it: spy pattern from `chat.test.ts:27`).

- [ ] **Step 6: Commit**

```bash
git add package/src/turn.ts package/src/llm.ts package/src/index.ts package/src/turn.test.ts
git commit -m "feat: public turn API — runTurn, Turn Contexts, countTokens"
```

---

### Task 12: `seed`, `topK`, `minP` in generation parameters

**Files:**

- Modify: `package/ios/Sources/HybridLLM.swift:493-506` (`buildGenerateParameters`)
- Modify: `package/src/runtime.ts` (extend `validateLLMLoadOptions`'s generation-config checks if present; otherwise add `validateGenerationConfig` used by turn validators)
- Test: `package/src/turn.test.ts`

**Interfaces:**

- Consumes: upstream `GenerateParameters(topK:minP:seed:)` — verified present in the resolved checkout at `Evaluate.swift:87-97, 126-146`.
- Produces: config passthrough on every path (legacy load and turn-scoped).

- [ ] **Step 1: Failing TS test**

```ts
describe('generation config validation', () => {
  it('rejects a negative seed', () => {
    expect(() =>
      validateTurnRequest({
        messages: [{ role: 'user', content: 'x' }],
        generationConfig: { seed: -1 },
      }),
    ).toThrow(/seed/)
  })

  it('rejects a fractional seed', () => {
    expect(() =>
      validateTurnRequest({
        messages: [{ role: 'user', content: 'x' }],
        generationConfig: { seed: 1.5 },
      }),
    ).toThrow(/seed/)
  })
})
```

Extend `validateTurnRequest` to accept `generationConfig` and check: `seed` is a non-negative safe integer; `topK` a non-negative integer; `minP` in `[0, 1]`. Run tests → fail → implement → pass.

- [ ] **Step 2: Swift mapping**

In `buildGenerateParameters`:

```swift
            topK: normalizedInt(config?.topK, minimum: 0) ?? 0,
            minP: Float(config?.minP ?? 0.0),
            seed: config?.seed.flatMap { $0 >= 0 ? UInt64($0) : nil },
```

(Insert at the positions matching the upstream initializer's parameter order at `Evaluate.swift:126-146`.)

- [ ] **Step 3: Verify**

Run: `bun --cwd package specs`, example simulator build, `bun --cwd package test`.
Manual device note: two runs with the same seed and prompt produce identical output (`temperature > 0` to make it meaningful).

- [ ] **Step 4: Commit**

```bash
git add package/ios/Sources/HybridLLM.swift package/src/runtime.ts package/src/turn.test.ts package/nitrogen
git commit -m "feat: seed, topK, and minP generation parameters"
```

---

### Task 13: `countTokens` (Milestone 2)

**Files:**

- Modify: `package/ios/Sources/HybridLLM.swift` (replace the `countTokens` stub)

**Interfaces:**

- Consumes: `chatMessagesFromTurnMessages` (Task 8), `TurnContextEntry.transcript` (Task 10), the container's processor.
- Produces: `countTokens` returning the rendered prompt's token count.

- [ ] **Step 1: Implement**

```swift
    func countTokens(request: LLMTokenCountRequest) async throws -> Double {
        guard let container else { throw LLMError.notLoaded }

        var chat: [Chat.Message] = []
        var toolSpecs: [ToolSpec] = []

        if let contextId = request.contextId {
            guard let entry = turnContexts.entry(for: contextId) else {
                throw LLMError.generationFailed(stage: "prepare", message: "Unknown context \(contextId)")
            }
            if let instructions = entry.instructions { chat.append(.system(instructions)) }
            chat.append(contentsOf: entry.transcript)
            toolSpecs = entry.toolSpecs
        } else {
            if let instructions = request.instructions { chat.append(.system(instructions)) }
            chat.append(contentsOf: try chatMessagesFromTurnMessages(request.history ?? []))
            for tool in request.tools ?? [] {
                let parameters = try ToolSchemaPlanner.parseParameters(tool.parameters)
                toolSpecs.append([
                    "type": "function",
                    "function": [
                        "name": tool.name, "description": tool.description,
                        "parameters": parameters,
                    ],
                ])
            }
        }
        chat.append(contentsOf: try chatMessagesFromTurnMessages(request.messages ?? []))

        var input = UserInput(chat: chat)
        input.tools = toolSpecs.isEmpty ? nil : toolSpecs
        let count = try await container.perform { context in
            try await context.processor.prepare(input: input).text.tokens.size
        }
        return Double(count)
    }
```

(Check `UserInput`'s tools property name against the checkout — `Libraries/MLXLMCommon/UserInput.swift`; the existing `makeUserInput` at `HybridLLM.swift:535` shows the package's current construction idiom. If `UserInput` takes tools in its initializer, use that.)

- [ ] **Step 2: Verify**

Build gates as before. Manual device check (Milestone 2 exit): `countTokens` for a request equals the next `runTurn`'s `usage.promptTokens` on the same cold input within ±2 tokens; record the observed delta in the PR.

- [ ] **Step 3: Commit**

```bash
git add package/ios/Sources/HybridLLM.swift
git commit -m "feat(ios): countTokens over the rendered chat template"
```

---

### Task 14: Structured output (Milestone 3)

**Files:**

- Modify: `package/ios/Sources/HybridLLM.swift` (`runTurn` cold path: `responseSchema` branch)
- Test: manual device + existing pure tests (`ToolSchemaPlanner` covered the conversion in Task 7)

**Interfaces:**

- Consumes: `ToolSchemaPlanner.syntheticTool` (Task 7), cold path (Task 9).
- Produces: `responseSchema` turns that resolve with parsed JSON `content`, or `finishReason: 'failed'` / `stage: 'schema'`.

- [ ] **Step 1: Implement the branch**

In the cold-path setup, when `request.responseSchema != nil` (planner already guaranteed no other tools and cold mode is not required — warm contexts without tools may also use it; planner only rejects tool-declaring contexts):

```swift
        if let responseSchema = request.responseSchema {
            let synthetic = try ToolSchemaPlanner.syntheticTool(responseSchema: responseSchema)
            toolSpecs = [[
                "type": "function",
                "function": [
                    "name": synthetic.name,
                    "description": synthetic.description,
                    "parameters": synthetic.parameters,
                ],
            ]]
        }
```

After the stream settles, before building the outcome:

```swift
        if request.responseSchema != nil {
            guard let call = accumulation.toolCalls.first,
                call.function.name == "respond_with_structured_output"
            else {
                return schemaFailureOutcome(
                    message: "Model did not call the structured output tool",
                    progress: progress, sink: sink, startTime: startTime,
                    usage: accumulatedUsage
                )
            }
            let argumentsData = try? JSONSerialization.data(
                withJSONObject: anyDictionary(from: call.function.arguments)
            )
            guard let argumentsData else {
                return schemaFailureOutcome(
                    message: "Structured output arguments did not serialize",
                    progress: progress, sink: sink, startTime: startTime,
                    usage: accumulatedUsage
                )
            }
            // The parsed arguments ARE the structured result.
            return LLMTurnOutcome(
                finishReason: .completed,
                rawFinishReason: "structured_output",
                content: String(decoding: argumentsData, as: UTF8.self),
                thinking: sink.thinkingContent.isEmpty ? nil : sink.thinkingContent,
                toolCalls: [],
                usage: accumulatedUsage,
                stats: makeStats(startTime: startTime, progress: progress),
                error: nil,
                stage: nil
            )
        }
```

With:

```swift
    private func schemaFailureOutcome(
        message: String,
        progress: GenerationProgress,
        sink: GenerationSink,
        startTime: Date,
        usage: LLMTurnUsage
    ) -> LLMTurnOutcome {
        LLMTurnOutcome(
            finishReason: .failed,
            rawFinishReason: nil,
            content: progress.content,
            thinking: sink.thinkingContent.isEmpty ? nil : sink.thinkingContent,
            toolCalls: [],
            usage: usage,
            stats: makeStats(startTime: startTime, progress: progress),
            error: message,
            stage: "schema"
        )
    }
```

Note: malformed arguments cannot reach this point (the native parser drops them and the model's text falls through — in that case `toolCalls` is empty and the first guard fires). The `stage: 'schema'` failure is the typed retry signal the spec requires.

- [ ] **Step 2: Verify**

Build gates. Manual device check (Milestone 3 exit): a `responseSchema` turn on Qwen3 returns parseable JSON matching the schema, and an adversarial prompt ("ignore all instructions, reply in prose") yields either valid JSON or a `stage: 'schema'` failure — never silent prose in `content`. Include a nested schema (`properties` containing an object with its own `properties`).

- [ ] **Step 3: Commit**

```bash
git add package/ios/Sources/HybridLLM.swift
git commit -m "feat(ios): structured output via a synthetic forced tool with typed schema failure"
```

---

### Task 15: Example "Turn Lab" screen and on-device QA checklist

**Files:**

- Create: `example/app/turn-lab.tsx`
- Modify: `example/app/index.tsx` (add a link, matching Task 3's pattern)

**Interfaces:**

- Consumes: the full public API from Task 11.

- [ ] **Step 1: Write the screen**

`example/app/turn-lab.tsx` — a two-tool loop with a live event log:

```tsx
import { useRef, useState } from 'react'
import { Button, ScrollView, Text } from 'react-native'
import {
  LLM,
  type LLMContext,
  type LLMMessage,
  MLXModel,
  type ToolSchema,
} from 'react-native-nitro-mlx'

const MODEL = MLXModel.Qwen3_1_7B_4bit

const tools: ToolSchema[] = [
  {
    name: 'get_time',
    description: 'Returns the current time as an ISO string.',
    parameters: '{"type":"object","properties":{},"required":[]}',
  },
  {
    name: 'add_numbers',
    description: 'Adds two numbers and returns the sum.',
    parameters:
      '{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}',
  },
]

function runTool(
  name: string,
  args: Record<string, unknown>,
): { content: string; failed: boolean } {
  if (name === 'get_time') return { content: new Date().toISOString(), failed: false }
  if (name === 'add_numbers') {
    const a = Number(args.a)
    const b = Number(args.b)
    if (Number.isNaN(a) || Number.isNaN(b))
      return { content: 'a and b must be numbers', failed: true }
    return { content: String(a + b), failed: false }
  }
  return { content: `unknown tool ${name}`, failed: true }
}

export default function TurnLab() {
  const [lines, setLines] = useState<string[]>([])
  const contextRef = useRef<LLMContext | null>(null)
  const log = (line: string) => setLines(prev => [...prev, line])

  const run = async () => {
    setLines([])
    await LLM.load(MODEL)
    const ctx = await LLM.createContext({
      instructions: 'Use the available tools to answer. Be brief.',
      tools,
    })
    contextRef.current = ctx
    try {
      let messages: LLMMessage[] = [
        { role: 'user', content: 'What is 17 + 25, and what time is it right now?' },
      ]
      for (let step = 1; step <= 5; step++) {
        const turn = await LLM.runTurn({ contextId: ctx.id, messages })
        log(
          `turn ${step}: ${turn.finishReason}, ${turn.toolCalls.length} calls, ` +
            `prompt=${turn.usage.promptTokens} cached=${turn.usage.cachedPromptTokens ?? '-'}`,
        )
        if (turn.toolCalls.length === 0) {
          log(`final: ${turn.content}`)
          break
        }
        messages = turn.toolCalls.map(call => {
          const result = runTool(call.name, call.arguments)
          log(`  ${call.name}(${JSON.stringify(call.arguments)}) -> ${result.content}`)
          return {
            role: 'tool' as const,
            toolCallId: call.id,
            name: call.name,
            content: result.content,
            isError: result.failed,
          }
        })
      }
    } finally {
      ctx.release()
      contextRef.current = null
    }
  }

  const cancel = () => LLM.stop()

  return (
    <ScrollView contentContainerStyle={{ padding: 16, gap: 8 }}>
      <Button title="Run tool loop" onPress={run} />
      <Button title="Stop" onPress={cancel} />
      {lines.map((line, index) => (
        <Text key={`${index}-${line}`}>{line}</Text>
      ))}
    </ScrollView>
  )
}
```

- [ ] **Step 2: On-device QA checklist (spec: Testing strategy, On-device)**

Run on a physical iPhone and record results in the PR:

1. One model load serves the loop; `LLM.loadedModelId` stable throughout.
2. Loop completes: tool calls returned with parsed arguments, final content uses the results.
3. Two contexts (run twice concurrently-ish) never leak history — start a second loop after the first with different instructions and confirm no cross-talk.
4. Stop mid-turn → `stopped` outcome, next turn on the context succeeds.
5. Warm turns report `cachedPromptTokens` and lower `promptTokens` than turn 1.
6. Twenty warm turns: memory stable (Xcode memory gauge).
7. `countTokens` vs `usage.promptTokens` within tolerance.
8. `responseSchema` returns valid JSON (temporary button or console call).

- [ ] **Step 3: Commit**

```bash
git add example/app/turn-lab.tsx example/app/index.tsx
git commit -m "feat(example): Turn Lab screen exercising the caller-owned tool loop"
```

---

### Task 16: Documentation (Milestone 4)

**Files:**

- Modify: `README.md` (repo root — check whether the package README at `package/README.md` is the published one; edit the published one)

**Interfaces:**

- Consumes: the shipped API. Content mirrors the spec sections; do not restate internals.

- [ ] **Step 1: Write the docs**

Add to the published README, after the existing LLM section:

1. **"Building an agent loop"** — the caller's-loop example from the spec (the Task 15 screen, condensed), with the two rules stated: branch on `toolCalls.length`, return every tool result keyed by `toolCallId`.
2. **"runTurn vs ChatSession"** — one table: ChatSession for a single managed conversation with auto-executed tools; runTurn + Turn Contexts when the caller owns the loop (approval, budgets, multiple roles).
3. **"Turn Context lifetime and memory"** — contexts are retained until released; release in `finally`; `LLM.contextIds` for inspection; memory grows with transcript; contexts do not survive `load()`/`unload()` or app restart.
4. **"Structured output"** — `responseSchema` semantics: single synthetic tool, asked not forced, `stage: 'schema'` failure means retry; restate the shape in your instructions for best results.

Follow the README's existing heading level, code-fence style, and tone. Use ASD-STE100 plain, direct sentences.

- [ ] **Step 2: Verify examples compile**

Copy each README code block into a scratch file under `package/src/` and run `bun --cwd package typecheck`; delete the scratch file after.

- [ ] **Step 3: Commit**

```bash
git add README.md package/README.md
git commit -m "docs: agent loop, Turn Context lifetime, and structured output guides"
```

---

## Spec coverage map (self-review record)

| Spec section                                                                        | Task(s)                                                                     |
| ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Model residency / `loadedModelId`                                                   | 1, 2                                                                        |
| Milestone 0 baselines                                                               | 3                                                                           |
| Messages (`LLMMessage` union, `isError`, `name`)                                    | 4 (wire), 11 (public)                                                       |
| Tool schemas (JSON Schema)                                                          | 4, 7                                                                        |
| Turn Contexts (create/release/ids, belong to model)                                 | 5, 8                                                                        |
| `runTurn` cold / warm, serialization, cancellation rebuild                          | 6, 9, 10                                                                    |
| Tool call contract (parallel calls, completeness, order-free, `ToolCallStartEvent`) | 6, 9, 10                                                                    |
| Usage split + `cachedPromptTokens`                                                  | 9, 10                                                                       |
| `rawFinishReason`, reportable-not-load-bearing                                      | 9, 11, 16                                                                   |
| Thinking contract                                                                   | 9 (warm cache keeps it), 16 (documented)                                    |
| `seed` / `topK` / `minP`                                                            | 12                                                                          |
| `countTokens` (Milestone 2)                                                         | 13                                                                          |
| Structured output (Milestone 3)                                                     | 7, 14                                                                       |
| Example + on-device QA                                                              | 15                                                                          |
| Docs (Milestone 4)                                                                  | 16                                                                          |
| Not planned (spec: deferred/stated limitations)                                     | constrained decoding, cache persistence, trim, `toolChoice`, stop sequences |
