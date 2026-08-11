import Foundation

private enum TestFailure: Error {
  case failed(String)
}

private func expect(_ condition: @autoclosure () -> Bool, _ message: String) throws {
  if !condition() {
    throw TestFailure.failed(message)
  }
}

private func expectError(
  _ expected: EmbeddingsError,
  _ message: String,
  _ body: () throws -> Void
) throws {
  do {
    try body()
    throw TestFailure.failed("\(message): no error was thrown")
  } catch let error as EmbeddingsError {
    try expect(error == expected, "\(message): expected \(expected), received \(error)")
  }
}

@main
struct EmbeddingsBatchPlannerTests {
  static func main() throws {
    // Mixed-length batch: pads to minPaddedLength, keeps true lengths.
    let mixed = try EmbeddingsBatchPlanner.plan(
      inputs: [[1, 2, 3], [4, 5]],
      maxSequenceLength: 512,
      padTokenId: 9,
      truncate: false
    )
    try expect(mixed.paddedLength == 16, "short batch pads to minPaddedLength")
    try expect(mixed.lengths == [3, 2], "true lengths are preserved")
    try expect(
      mixed.rows == [
        [1, 2, 3] + Array(repeating: 9, count: 13),
        [4, 5] + Array(repeating: 9, count: 14),
      ],
      "rows are padded with the pad token"
    )

    // A row that contains the pad-token id as real content keeps its full length.
    let eosContent = try EmbeddingsBatchPlanner.plan(
      inputs: [[1, 9, 2, 9], [3]],
      maxSequenceLength: 512,
      padTokenId: 9,
      truncate: false
    )
    try expect(
      eosContent.lengths == [4, 1],
      "genuine pad-valued tokens are not treated as padding"
    )

    // Boundary: exactly at the limit is accepted.
    let atLimit = try EmbeddingsBatchPlanner.plan(
      inputs: [Array(repeating: 1, count: 512)],
      maxSequenceLength: 512,
      padTokenId: 0,
      truncate: false
    )
    try expect(atLimit.paddedLength == 512, "input at the limit is accepted unpadded")
    try expect(atLimit.lengths == [512], "input at the limit keeps its length")

    // Boundary: one past the limit is rejected with a descriptive error.
    try expectError(
      .inputTooLong(index: 1, tokenCount: 513, limit: 512),
      "oversized input is rejected"
    ) {
      _ = try EmbeddingsBatchPlanner.plan(
        inputs: [[1], Array(repeating: 1, count: 513)],
        maxSequenceLength: 512,
        padTokenId: 0,
        truncate: false
      )
    }

    // Explicit truncation keeps the first limit-1 tokens plus the original last token.
    let truncated = try EmbeddingsBatchPlanner.plan(
      inputs: [Array(1...20)],
      maxSequenceLength: 16,
      padTokenId: 0,
      truncate: true
    )
    try expect(truncated.paddedLength == 16, "truncated batch pads to the limit floor")
    try expect(truncated.lengths == [16], "truncated row reports the limit length")
    try expect(truncated.rows[0].count == 16, "truncated row is padded to paddedLength")

    let truncatedTail = try EmbeddingsBatchPlanner.plan(
      inputs: [[10, 20, 30, 40, 50, 99]],
      maxSequenceLength: 4,
      padTokenId: 0,
      truncate: true
    )
    try expect(
      truncatedTail.rows[0] == [10, 20, 30, 99],
      "truncation preserves the closing token"
    )
    try expect(truncatedTail.lengths == [4], "truncated length equals the limit")

    // Truncation is a no-op for rows within the limit.
    let truncateNoop = try EmbeddingsBatchPlanner.plan(
      inputs: [[1, 2]],
      maxSequenceLength: 4,
      padTokenId: 0,
      truncate: true
    )
    try expect(truncateNoop.rows[0] == [1, 2, 0, 0], "short rows are not truncated")
    try expect(truncateNoop.lengths == [2], "short rows keep their true length")

    // A limit below minPaddedLength caps the padded length (positional safety).
    try expect(
      truncateNoop.paddedLength == 4,
      "padded length never exceeds the sequence limit"
    )

    // Batch-size boundary.
    let fullBatch = try EmbeddingsBatchPlanner.plan(
      inputs: Array(repeating: [1], count: EmbeddingsBatchPlanner.maxBatchSize),
      maxSequenceLength: 512,
      padTokenId: 0,
      truncate: false
    )
    try expect(
      fullBatch.lengths.count == EmbeddingsBatchPlanner.maxBatchSize,
      "a full batch is accepted"
    )
    try expectError(
      .batchTooLarge(
        count: EmbeddingsBatchPlanner.maxBatchSize + 1,
        limit: EmbeddingsBatchPlanner.maxBatchSize
      ),
      "an oversized batch is rejected"
    ) {
      _ = try EmbeddingsBatchPlanner.plan(
        inputs: Array(repeating: [1], count: EmbeddingsBatchPlanner.maxBatchSize + 1),
        maxSequenceLength: 512,
        padTokenId: 0,
        truncate: false
      )
    }

    // Empty batches and empty rows are rejected.
    try expectError(.emptyInput, "an empty batch is rejected") {
      _ = try EmbeddingsBatchPlanner.plan(
        inputs: [],
        maxSequenceLength: 512,
        padTokenId: 0,
        truncate: false
      )
    }
    try expectError(.emptyInput, "an empty row is rejected") {
      _ = try EmbeddingsBatchPlanner.plan(
        inputs: [[1], []],
        maxSequenceLength: 512,
        padTokenId: 0,
        truncate: false
      )
    }

    // Missing max_position_embeddings falls back to the safety limit.
    try expect(
      EmbeddingsBatchPlanner.sequenceLimit(maxSequenceLength: 0) ==
        EmbeddingsBatchPlanner.fallbackSequenceLimit,
      "unknown model limit falls back to the safety limit"
    )
    try expect(
      EmbeddingsBatchPlanner.sequenceLimit(maxSequenceLength: 512) == 512,
      "known model limit is used directly"
    )
    try expectError(
      .inputTooLong(
        index: 0,
        tokenCount: EmbeddingsBatchPlanner.fallbackSequenceLimit + 1,
        limit: EmbeddingsBatchPlanner.fallbackSequenceLimit
      ),
      "the safety limit is enforced when the model limit is unknown"
    ) {
      _ = try EmbeddingsBatchPlanner.plan(
        inputs: [
          Array(repeating: 1, count: EmbeddingsBatchPlanner.fallbackSequenceLimit + 1)
        ],
        maxSequenceLength: 0,
        padTokenId: 0,
        truncate: false
      )
    }

    // Pad-token resolution: configured id → eos id → 0.
    try expect(
      EmbeddingsBatchPlanner.resolvePadToken(configured: 5, eos: 2) == 5,
      "a configured pad token wins"
    )
    try expect(
      EmbeddingsBatchPlanner.resolvePadToken(configured: nil, eos: 2) == 2,
      "eos is the fallback pad token"
    )
    try expect(
      EmbeddingsBatchPlanner.resolvePadToken(configured: nil, eos: nil) == 0,
      "0 is the last-resort pad token"
    )

    // Pad-token string parsing from tokenizer_config.json shapes.
    try expect(
      EmbeddingsBatchPlanner.padTokenString(
        fromTokenizerConfig: ["pad_token": "<pad>"]) == "<pad>",
      "string pad_token form is parsed"
    )
    try expect(
      EmbeddingsBatchPlanner.padTokenString(
        fromTokenizerConfig: ["pad_token": ["content": "[PAD]"]]) == "[PAD]",
      "added-token dict pad_token form is parsed"
    )
    try expect(
      EmbeddingsBatchPlanner.padTokenString(fromTokenizerConfig: [:]) == nil,
      "missing pad_token yields nil"
    )

    print("EmbeddingsBatchPlannerTests passed")
  }
}
