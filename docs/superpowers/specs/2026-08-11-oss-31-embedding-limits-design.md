# OSS-31: Enforce MLX embedding limits and correct padding masks

Date: 2026-08-11 · Ticket: [OSS-31](https://linear.app/henry-pl-llc/issue/OSS-31)

## Problem

`HybridEmbeddings.computeEmbeddings` has three defects:

1. **No sequence limit.** `maxSequenceLength` is reported but never enforced. One long
   input pads the whole batch to its length. This can cause large allocations or
   positional-limit failures in the model.
2. **No batch limit.** `embedBatch` accepts arrays of any size. The padded tensor is
   `batch × longest`, which is unbounded in both dimensions.
3. **Wrong padding and mask.** Padding uses `eosTokenId ?? 0`, and the attention mask is
   `padded .!= padToken`. This masks *genuine* EOS (or id-0) tokens inside real content,
   which corrupts pooled embeddings.

## Approaches considered

- **A. Validate inline in `computeEmbeddings`.** Smallest diff, but the logic stays inside
  an MLX-importing closure and cannot be unit-tested off-device (MLX needs a physical
  device).
- **B. Extract a pure `EmbeddingsBatchPlanner` (chosen).** All limit/truncation/padding
  decisions live in a Foundation-only file, tested with the existing `swiftc` test-script
  pattern (`TTSSpeed`, `ManagedHistoryTrimPlanner`). `computeEmbeddings` only converts the
  plan into MLX arrays.
- **C. TS-side token counting.** Rejected: the tokenizer lives on the native side; TS
  cannot count tokens. TS only enforces cheap shape limits (batch size, option types).

## Design

### Pure planner — `package/ios/Sources/EmbeddingsBatchPlanner.swift`

No MLX imports. Holds the (moved) `EmbeddingsError` enum, extended with
`inputTooLong(index:tokenCount:limit:)` and `batchTooLarge(count:limit:)`, both with
`LocalizedError` descriptions so JS receives readable messages.

```swift
struct EmbeddingsBatchPlan: Equatable {
  let rows: [[Int]]      // padded token rows, all count == paddedLength
  let lengths: [Int]     // true token counts, drives the masks
  let paddedLength: Int
}

struct EmbeddingsBatchPlanner {
  static let maxBatchSize = 64        // mirrored in TS
  static let minPaddedLength = 16     // Apple Silicon alignment (MLXEmbedders README)
  static let fallbackSequenceLimit = 8192  // used when config has no max_position_embeddings

  static func sequenceLimit(maxSequenceLength: Int) -> Int
  static func padTokenString(fromTokenizerConfig: [String: Any]) -> String?
  static func resolvePadToken(configured: Int?, eos: Int?) -> Int   // → 0 fallback
  static func plan(inputs: [[Int]], maxSequenceLength: Int,
                   padTokenId: Int, truncate: Bool) throws -> EmbeddingsBatchPlan
}
```

Rules:

- Empty batch or any empty row → `.emptyInput`.
- `inputs.count > maxBatchSize` → `.batchTooLarge`.
- Row longer than the sequence limit → `.inputTooLong`, unless `truncate` is true.
- Truncation keeps the first `limit - 1` tokens plus the row's original last token. This
  preserves a trailing special token ([SEP]/EOS) that last-token pooling depends on,
  matching Hugging Face truncation semantics.
- `paddedLength = max(minPaddedLength, longest row)`, bounded by the sequence limit
  (and `minPaddedLength` when the limit is smaller).

### Mask correction — `HybridEmbeddings.computeEmbeddings`

- Pad token: `tokenizer_config.json` `pad_token` (string or `{content:}` form, read at
  load) → `convertTokenToId` → `eosTokenId` → `0`.
- Attention/pooling mask is built from **lengths**, not token values:
  `MLXArray(0..<paddedLength) .< MLXArray(lengths).reshaped([batch, 1])`.
  Genuine EOS content is never masked, and the pad token value no longer affects output.

### API — explicit truncation option

`Embeddings.nitro.ts` gains `EmbeddingsEmbedOptions { truncate?: boolean }`;
`embed`/`embedBatch` accept it (nitrogen regenerated via `bun specs`). Default is
**reject** with a descriptive error; truncation happens only on explicit opt-in.
`embeddings.ts` passes the option through `embed`, `embedBatch`, `embedQuery`,
`embedDocument`. `runtime.ts` validates the option and enforces
`EMBEDDINGS_MAX_BATCH_SIZE = 64` before crossing the bridge (Swift re-checks).

## Testing

- **Swift** (`package/ios/Tests/EmbeddingsBatchPlannerTests.swift`, run via new
  `test:ios-embeddings-planner` swiftc script): boundary lengths (== limit ok, limit+1
  rejected/truncated), truncation tail-token preservation, mixed-length batches (padded
  rows + lengths), min-16 padding, missing pad-token fallbacks (config → eos → 0),
  pad-token config parsing (string and dict forms), batch-size boundary, empty rows,
  fallback limit when config reports 0.
- **TS** (`bun test`): option validation, batch-size rejection at 65, pass-through of
  `truncate`.
- On-device behavior (actual MLX forward pass) is not unit-testable in CI; the planner
  boundary keeps that surface minimal.

## Out of scope

Auto-chunking oversized batches, Android, streaming embeddings.
