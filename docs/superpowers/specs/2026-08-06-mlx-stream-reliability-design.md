# MLX Stream Reliability Design

Date: 2026-08-06
Branch: `fix/reliability-posthog-review`
Status: Approved

## Background

A PostHog review (July 7 to August 6, 2026, app version 2.1.0) showed that all
five recorded MLX replies failed. The failures covered Qwen and three Llama
models, which points to the common MLX reply path, not one model file. Failed
replies always recorded 0 seconds because the module gives no failure data.

This design hardens the reply path in `react-native-nitro-mlx`. App-side work
(RevenueCat, PostHog event wiring, run IDs) is out of scope and lives in the
cora app.

## Goals

1. Prevent overlapping generations on one MLX container.
2. Give the app a descriptive error with a failure stage for every failure.
3. Guarantee exactly one terminal stream event for every `streamWithEvents`
   call: success, error, or interrupt.

## Design

### 1. Concurrency guard

`generate`, `stream`, and `streamWithEvents` check `currentTask` at entry on
the MainActor. If a generation runs, the call throws
`LLMError.alreadyGenerating` at once. No queue. The app controls retry.

### 2. Error type

`LLMError` conforms to `LocalizedError` and gets these cases:

- `notLoaded` — no model is loaded.
- `alreadyGenerating` — a generation is already in progress.
- `generationFailed(stage: String, message: String)` — a generation failed.

Failure stages:

- `prepare` — history trim and input preparation.
- `generate` — the token stream loop.
- `tool` — tool execution.
- `history` — history finalization after generation.

JS receives the localized message through the rejected Nitro promise.

### 3. Terminal event contract

`streamWithEvents` always ends with exactly one terminal event:

- Success → `generation_end` with content and stats (unchanged).
- Thrown error → new `generation_error` event with the error message, the
  stage, and partial stats (real elapsed time, tokens generated so far). The
  promise still rejects with the same error.
- `stop()` or a new `load()` during generation → `generation_end` with the
  partial content and partial stats. The promise resolves with the partial
  text (current `stop()` behavior stays).

Spec changes in `specs/LLM.nitro.ts`:

- New `StreamEventKind` member: `generationError`.
- New optional envelope field: `stage: string`.
- New `StreamEvent` union member in TS:
  `{ type: 'generation_error'; error: string; stage: string; stats: GenerationStats }`.

`mapStreamEventEnvelope` in `runtime.ts` maps the new kind. `bun specs`
regenerates the Nitro bridge code.

### 4. Files

- `package/src/specs/LLM.nitro.ts` — event kind, envelope field, union member.
- `package/src/runtime.ts` — envelope mapping.
- `package/src/llm.ts` — docs for the new event.
- `package/ios/Sources/LLMError.swift` — new cases, `LocalizedError`.
- `package/ios/Sources/StreamEventEmitter.swift` — `emitGenerationError`.
- `package/ios/Sources/HybridLLM.swift` — guard, stage wrapping, catch block
  that emits the terminal event, interrupt handling.

### 5. Error handling flow

In `streamWithEvents`, the generation body is wrapped so that a thrown error
is caught once at the top level. The catch block:

1. Builds partial stats from the start time and token counts collected so far.
2. Emits `generation_error` with the message, stage, and partial stats.
3. Rethrows, so the promise rejects.

`stream` and `generate` get the same guard and stage wrapping, but they have
no event channel; they only reject with the descriptive error.

### 6. Testing

- TS (`runtime.test.ts`): mapping of the `generationError` envelope, including
  missing optional fields.
- Swift (`package/ios/Tests`): the concurrency guard rejects a second call;
  stage classification wraps errors with the correct stage; the terminal event
  fires exactly once on success, error, and cancel.
- Implementation follows test-driven development.

## Out of scope

- RevenueCat setup ordering (cora app).
- PostHog event wiring, run IDs, app start and model-load analytics (cora app).
- A `getLastFailure()` telemetry API.
