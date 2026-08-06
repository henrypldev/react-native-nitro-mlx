# MLX Stream Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Guarantee one terminal stream event per generation, descriptive staged errors, and a concurrency guard in the MLX reply path.

**Architecture:** The Nitro spec gains a `generation_error` event kind and a `stage` envelope field. Swift `LLMError` becomes descriptive with a failure stage. `HybridLLMCore` rejects overlapping generations, wraps each phase with a stage, and emits exactly one terminal event (`generation_end` or `generation_error`) from `streamWithEvents`.

**Tech Stack:** TypeScript (bun test), Nitro Modules (nitrogen codegen), Swift (MLX Swift LM), standalone `swiftc` test binaries.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-06-mlx-stream-reliability-design.md`.
- Stages are exactly: `prepare`, `generate`, `tool`, `history`.
- Concurrency policy: reject the second call with `LLMError.alreadyGenerating`. No queue.
- `stop()` / `load()` interrupt → `generation_end` with partial stats; promise resolves.
- Thrown error → `generation_error` event AND a rejected promise.
- Do NOT commit files under `docs/` (user preference).
- CocoaPods commands need `LANG=en_US.UTF-8`.
- MLX runtime needs a physical device; verification here is tests + compile, not on-device runs.
- TS commands run from `package/`: `bun test src/runtime.test.ts`, `bun typecheck`, `bun specs`.

---

### Task 1: TS event surface (`generation_error`)

**Files:**
- Modify: `package/src/specs/LLM.nitro.ts`
- Modify: `package/src/runtime.ts` (mapStreamEventEnvelope, ~line 188)
- Modify: `package/src/llm.ts` (streamWithEvents doc comment only)
- Test: `package/src/runtime.test.ts`

**Interfaces:**
- Produces: `GenerationErrorEvent { type: 'generation_error'; error: string; stage: string; stats: GenerationStats }`, `StreamEventKind` member `'generation_error'`, envelope field `stage?: string`. Task 4's Swift code relies on the regenerated `StreamEventEnvelope` having `stage` and `StreamEventKind.generationError`.

- [ ] **Step 1: Write the failing tests**

Add to `package/src/runtime.test.ts`, next to the existing `mapStreamEventEnvelope` tests (match the file's existing test style — `import { describe, expect, test } from 'bun:test'` is already there):

```ts
test('maps generation_error envelope', () => {
  const stats = {
    tokenCount: 12,
    tokensPerSecond: 0,
    timeToFirstToken: 0,
    totalTime: 4200,
    toolExecutionTime: 0,
  }
  expect(
    mapStreamEventEnvelope({
      kind: 'generation_error',
      error: 'Generation failed during generate: boom',
      stage: 'generate',
      stats,
    }),
  ).toEqual({
    type: 'generation_error',
    error: 'Generation failed during generate: boom',
    stage: 'generate',
    stats,
  })
})

test('maps generation_error envelope with missing optional fields', () => {
  const event = mapStreamEventEnvelope({ kind: 'generation_error' })
  expect(event).toEqual({
    type: 'generation_error',
    error: '',
    stage: '',
    stats: {
      tokenCount: 0,
      tokensPerSecond: 0,
      timeToFirstToken: 0,
      totalTime: 0,
      toolExecutionTime: 0,
    },
  })
})
```

If the file already defines an `EMPTY_STATS`-style fixture, reuse it instead of the inline object.

- [ ] **Step 2: Run tests to verify they fail**

Run: `bun --cwd package test src/runtime.test.ts`
Expected: FAIL — type error / `null` return for unknown kind `generation_error`.

- [ ] **Step 3: Implement the TS changes**

In `package/src/specs/LLM.nitro.ts`:

1. After `GenerationEndEvent` add:

```ts
export interface GenerationErrorEvent {
  type: 'generation_error'
  error: string
  stage: string
  stats: GenerationStats
}
```

2. Add `| GenerationErrorEvent` to the `StreamEvent` union.
3. Add `| 'generation_error'` to `StreamEventKind`.
4. Add `stage?: string` to `StreamEventEnvelope` (after `error?: string`).

In `package/src/runtime.ts`, in `mapStreamEventEnvelope`, before `default:` add:

```ts
    case 'generation_error':
      return {
        type: 'generation_error',
        error: envelope.error ?? '',
        stage: envelope.stage ?? '',
        stats: envelope.stats ?? EMPTY_STATS,
      }
```

In `package/src/llm.ts`, extend the `streamWithEvents` doc comment example switch with:

```ts
   *     case 'generation_error':
   *       showError(event.error) // event.stage, event.stats also available
   *       break
```

- [ ] **Step 4: Run tests and typecheck**

Run: `bun --cwd package test src/runtime.test.ts && bun --cwd package typecheck`
Expected: PASS, no type errors.

- [ ] **Step 5: Commit**

```bash
git add package/src/specs/LLM.nitro.ts package/src/runtime.ts package/src/llm.ts package/src/runtime.test.ts
git commit -m "feat: add generation_error stream event to TS surface"
```

---

### Task 2: Regenerate the Nitro bridge

**Files:**
- Modify (generated): `package/nitrogen/**`, `package/lib/**`

**Interfaces:**
- Consumes: Task 1's spec changes.
- Produces: Swift `StreamEventKind.generationError` enum case and `StreamEventEnvelope.stage: String?` used by Tasks 4–5.

- [ ] **Step 1: Run codegen + build**

Run: `bun --cwd package specs`
Expected: nitrogen regenerates without errors; build succeeds.

- [ ] **Step 2: Inspect the generated Swift envelope**

Run: `grep -rn "generationError\|stage" package/nitrogen/generated/ios/swift/StreamEventEnvelope.swift package/nitrogen/generated/ios/swift/StreamEventKind.swift`
Expected: `generationError` case exists; `stage` field exists. Note the exact initializer parameter order of `StreamEventEnvelope` for Task 4.

- [ ] **Step 3: Commit**

```bash
git add package/nitrogen
git commit -m "chore: regenerate nitrogen bridge for generation_error event"
```

(`lib/` is build output; add it only if the repo tracks it — check `git status`.)

---

### Task 3: Swift `LLMError` expansion + standalone test

**Files:**
- Modify: `package/ios/Sources/LLMError.swift`
- Create: `package/ios/Tests/LLMErrorTests.swift`
- Modify: `package/package.json` (add script)

**Interfaces:**
- Produces: `LLMError.alreadyGenerating`, `LLMError.generationFailed(stage:message:)`, `LLMError.failureStage: String?`, `enum GenerationStage: String { prepare, generate, tool, history }`. Task 4 consumes all of these.

- [ ] **Step 1: Write the failing test**

Create `package/ios/Tests/LLMErrorTests.swift` following the standalone executable pattern of `ManagedHistoryTrimPlannerSpyTests.swift` (top-level asserts, prints a success line at the end):

```swift
import Foundation

func expect(_ condition: Bool, _ message: String) {
    if !condition {
        print("FAIL: \(message)")
        exit(1)
    }
}

expect(
    LLMError.notLoaded.errorDescription == "No model is loaded. Call load() before generation.",
    "notLoaded message"
)
expect(
    LLMError.alreadyGenerating.errorDescription
        == "A generation is already in progress. Wait for it to finish or call stop().",
    "alreadyGenerating message"
)
expect(
    LLMError.generationFailed(stage: "generate", message: "boom").errorDescription
        == "Generation failed during generate: boom",
    "generationFailed message"
)
expect(LLMError.notLoaded.failureStage == nil, "notLoaded has no stage")
expect(
    LLMError.generationFailed(stage: "prepare", message: "x").failureStage == "prepare",
    "generationFailed stage"
)
expect(GenerationStage.prepare.rawValue == "prepare", "stage prepare")
expect(GenerationStage.generate.rawValue == "generate", "stage generate")
expect(GenerationStage.tool.rawValue == "tool", "stage tool")
expect(GenerationStage.history.rawValue == "history", "stage history")

print("LLMErrorTests: all assertions passed")
```

Add to `package/package.json` scripts, next to `test:ios-history-trim`:

```json
"test:ios-llm-error": "swiftc ios/Sources/LLMError.swift ios/Tests/LLMErrorTests.swift -o /tmp/LLMErrorTests && /tmp/LLMErrorTests",
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun --cwd package test:ios-llm-error`
Expected: FAIL to compile — `alreadyGenerating`, `generationFailed`, `failureStage`, `GenerationStage` do not exist.

- [ ] **Step 3: Implement `LLMError.swift`**

Replace the file content with:

```swift
import Foundation

public enum LLMError: Error, LocalizedError {
    case notLoaded
    case alreadyGenerating
    case generationFailed(stage: String, message: String)

    public var errorDescription: String? {
        switch self {
        case .notLoaded:
            return "No model is loaded. Call load() before generation."
        case .alreadyGenerating:
            return "A generation is already in progress. Wait for it to finish or call stop()."
        case .generationFailed(let stage, let message):
            return "Generation failed during \(stage): \(message)"
        }
    }

    /// The failure stage for `generationFailed`, nil otherwise.
    public var failureStage: String? {
        if case .generationFailed(let stage, _) = self {
            return stage
        }
        return nil
    }
}

/// Phases of a generation call, reported in `generation_error` events.
public enum GenerationStage: String {
    case prepare
    case generate
    case tool
    case history
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `bun --cwd package test:ios-llm-error && bun --cwd package test:ios-history-trim`
Expected: both print their success line.

- [ ] **Step 5: Commit**

```bash
git add package/ios/Sources/LLMError.swift package/ios/Tests/LLMErrorTests.swift package/package.json
git commit -m "feat(ios): descriptive staged LLMError with LocalizedError"
```

---

### Task 4: Emitter + `HybridLLM` guard, stages, terminal-event contract

**Files:**
- Modify: `package/ios/Sources/StreamEventEmitter.swift`
- Modify: `package/ios/Sources/HybridLLM.swift`

**Interfaces:**
- Consumes: `LLMError` cases + `GenerationStage` (Task 3), generated `StreamEventKind.generationError` + envelope `stage` (Task 2).
- Produces: runtime behavior only; no new public API beyond the emitted event.

- [ ] **Step 1: Add `emitGenerationError` to `StreamEventEmitter.swift`**

Add `stage: String? = nil` to the private `emit(...)` helper's parameters and pass it into the `StreamEventEnvelope` initializer (use the exact parameter order noted in Task 2 Step 2). Then add:

```swift
    func emitGenerationError(error: String, stage: String, stats: GenerationStats) {
        emit(.generationError, timestamp: nowMs(), error: error, stats: stats, stage: stage)
    }
```

(Adjust argument order at the call site of `emit` to match its signature.)

- [ ] **Step 2: Add the concurrency guard + stage helper to `HybridLLMCore`**

In `HybridLLM.swift`, inside `HybridLLMCore`, near `log(_:)`, add:

```swift
    private func ensureNotGenerating() throws {
        if currentTask != nil {
            throw LLMError.alreadyGenerating
        }
    }

    /// Wraps a generation phase so foreign errors surface as
    /// `LLMError.generationFailed` with the phase's stage. `LLMError` and
    /// cancellation pass through untouched.
    private func withStage<T>(
        _ stage: GenerationStage,
        _ body: () async throws -> T
    ) async throws -> T {
        do {
            return try await body()
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as LLMError {
            throw error
        } catch {
            throw LLMError.generationFailed(
                stage: stage.rawValue,
                message: error.localizedDescription
            )
        }
    }
```

- [ ] **Step 3: Guard all three entry points**

In `generate(prompt:)`, `stream(prompt:onToken:onToolCall:)`, and `streamWithEvents(prompt:onEvent:)`, directly after the existing `guard let container else { throw LLMError.notLoaded }`, add:

```swift
        try ensureNotGenerating()
```

- [ ] **Step 4: Stage-wrap `generate` and `stream`**

In both methods' task closures, wrap the phases (no events in these paths — rejection only):

- `try await trimManagedHistoryIfNeeded(...)` → `try await withStage(.prepare) { try await self.trimManagedHistoryIfNeeded(upcomingPrompt: prompt) }`
- `try await runManagedSession(...)` and `try await performGeneration(...)` → wrap in `withStage(.generate) { ... }`
- `try await finalizeManagedHistory(history)` → wrap in `withStage(.history) { ... }`

Keep all surrounding logic identical.

- [ ] **Step 5: Rework `streamWithEvents` for the terminal-event contract**

Restructure the task closure so counters live outside a `do/catch` and the catch emits exactly one terminal event:

```swift
        let task = Task<String, Error> { @MainActor in
            let startTime = Date()
            let emitter = StreamEventEmitter(callback: onEvent)
            emitter.emitGenerationStart()

            var generationTokenCount = 0
            var generationTimeMs: Double = 0
            var toolExecutionTime: Double = 0
            var firstTokenTime: Date?

            func partialStats() -> GenerationStats {
                makeStats(
                    startTime: startTime,
                    firstTokenTime: firstTokenTime,
                    generationTokenCount: generationTokenCount,
                    generationTimeMs: generationTimeMs,
                    toolExecutionTimeMs: toolExecutionTime
                )
            }

            do {
                if canUseManagedSession {
                    try await withStage(.prepare) {
                        try await self.trimManagedHistoryIfNeeded(upcomingPrompt: prompt)
                    }

                    let batcher = TokenBatcher(batchSize: tokenBatchSize) { token in
                        emitter.emitToken(token)
                    }
                    let result = try await withStage(.generate) {
                        try await self.runManagedSession(prompt: prompt, batcher: batcher)
                    }
                    generationTokenCount = result.generationTokenCount
                    generationTimeMs = result.generationTimeMs
                    firstTokenTime = result.firstTokenTime

                    var updatedHistory = messageHistory
                    updatedHistory.append(LLMMessage(role: "user", content: prompt))
                    updatedHistory.append(LLMMessage(role: "assistant", content: result.output))
                    try await withStage(.history) {
                        try await self.finalizeManagedHistory(updatedHistory)
                    }

                    let stats = partialStats()
                    lastStats = stats
                    emitter.emitGenerationEnd(content: result.output, stats: stats)
                    return result.output
                }

                if manageHistory {
                    try await withStage(.prepare) {
                        try await self.trimManagedHistoryIfNeeded(upcomingPrompt: prompt)
                    }
                }

                var history = messageHistory
                let tokenBatcher = TokenBatcher(batchSize: tokenBatchSize) { token in
                    emitter.emitToken(token)
                }
                let sink = EventGenerationSink(emitter: emitter, batcher: tokenBatcher)

                let result = try await withStage(.generate) {
                    try await self.performGeneration(
                        container: container,
                        history: &history,
                        prompt: prompt,
                        toolResults: nil,
                        depth: 0,
                        sink: sink,
                        onGenerationInfo: { tokens, time in
                            generationTokenCount += tokens
                            generationTimeMs += time
                        },
                        toolExecutionTime: &toolExecutionTime
                    )
                }
                firstTokenTime = sink.firstTokenTime

                tokenBatcher.flush()
                try await withStage(.history) {
                    try await self.finalizeManagedHistory(history)
                }

                let stats = partialStats()
                lastStats = stats
                emitter.emitGenerationEnd(content: result, stats: stats)

                log(
                    "StreamWithEvents complete - \(generationTokenCount) tokens, \(String(format: "%.1f", stats.tokensPerSecond)) tokens/s (tool execution: \(String(format: "%.0f", toolExecutionTime))ms)"
                )
                return result
            } catch is CancellationError {
                // stop() or a superseding load(): resolve with partial state.
                let stats = partialStats()
                lastStats = stats
                emitter.emitGenerationEnd(content: "", stats: stats)
                return ""
            } catch {
                let stats = partialStats()
                lastStats = stats
                emitter.emitGenerationError(
                    error: error.localizedDescription,
                    stage: (error as? LLMError)?.failureStage ?? GenerationStage.generate.rawValue,
                    stats: stats
                )
                throw error
            }
        }
```

Compiler note: `history` is captured `inout`-style by `performGeneration` — since Swift forbids capturing `inout` in an escaping closure, `withStage`'s body is non-escaping `() async throws -> T`, which is fine; if the compiler still rejects `&history` inside the closure, hoist the `performGeneration` call out and wrap only its error, e.g. assign via a local `do/catch` that rethrows `LLMError.generationFailed(stage: "generate", ...)`.

- [ ] **Step 6: Verify TS side still builds**

Run: `bun --cwd package typecheck && bun --cwd package test src/runtime.test.ts`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add package/ios/Sources/StreamEventEmitter.swift package/ios/Sources/HybridLLM.swift
git commit -m "feat(ios): concurrency guard, staged errors, terminal stream events"
```

---

### Task 5: Swift compile verification (example app)

**Files:**
- None modified — verification only.

**Interfaces:**
- Consumes: all previous tasks.

- [ ] **Step 1: Install pods**

Run: `LANG=en_US.UTF-8 bun --cwd package specs:pod`
Expected: pod install succeeds for `example/`.

- [ ] **Step 2: Build the example iOS app**

Discover the workspace/scheme, then build for the generic iOS platform (no signing):

```bash
cd example/ios
xcodebuild -list
xcodebuild -workspace <name>.xcworkspace -scheme <scheme> \
  -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO build
```

Expected: BUILD SUCCEEDED — this is the only place the Swift changes compile against MLX + the generated Nitro types.

- [ ] **Step 3: Run all package tests one final time**

Run: `bun --cwd package test src/runtime.test.ts && bun --cwd package test:ios-llm-error && bun --cwd package test:ios-history-trim && bun --cwd package typecheck`
Expected: all PASS.

No commit — nothing changed. On-device behavior (guard rejection, error event) needs a physical device and is verified in the cora app after release.
