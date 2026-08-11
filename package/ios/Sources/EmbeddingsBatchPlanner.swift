import Foundation

enum EmbeddingsError: Error, Equatable, LocalizedError {
  case notLoaded
  case emptyInput
  case inputTooLong(index: Int, tokenCount: Int, limit: Int)
  case batchTooLarge(count: Int, limit: Int)

  var errorDescription: String? {
    switch self {
    case .notLoaded:
      return "No embeddings model is loaded."
    case .emptyInput:
      return "Embeddings input must not be empty."
    case .inputTooLong(let index, let tokenCount, let limit):
      return
        "Input \(index) is \(tokenCount) tokens, but the model accepts at most \(limit). "
        + "Shorten the text or pass { truncate: true }."
    case .batchTooLarge(let count, let limit):
      return "Batch of \(count) texts exceeds the limit of \(limit). Split the batch."
    }
  }
}

struct EmbeddingsBatchPlan: Equatable {
  let rows: [[Int]]
  let lengths: [Int]
  let paddedLength: Int
}

struct EmbeddingsBatchPlanner {
  static let maxBatchSize = 64
  static let minPaddedLength = 16
  static let fallbackSequenceLimit = 8192

  /// Models converted without `max_position_embeddings` report 0; fall back to a
  /// safety cap so one input can never allocate unbounded padded tensors.
  static func sequenceLimit(maxSequenceLength: Int) -> Int {
    maxSequenceLength > 0 ? maxSequenceLength : fallbackSequenceLimit
  }

  /// `pad_token` in tokenizer_config.json is either a plain string or an
  /// added-token object with a `content` field.
  static func padTokenString(fromTokenizerConfig config: [String: Any]) -> String? {
    if let token = config["pad_token"] as? String {
      return token
    }
    if let added = config["pad_token"] as? [String: Any] {
      return added["content"] as? String
    }
    return nil
  }

  static func resolvePadToken(configured: Int?, eos: Int?) -> Int {
    configured ?? eos ?? 0
  }

  static func plan(
    inputs: [[Int]],
    maxSequenceLength: Int,
    padTokenId: Int,
    truncate: Bool
  ) throws -> EmbeddingsBatchPlan {
    guard !inputs.isEmpty, inputs.allSatisfy({ !$0.isEmpty }) else {
      throw EmbeddingsError.emptyInput
    }
    guard inputs.count <= maxBatchSize else {
      throw EmbeddingsError.batchTooLarge(count: inputs.count, limit: maxBatchSize)
    }

    let limit = sequenceLimit(maxSequenceLength: maxSequenceLength)
    let bounded = try inputs.enumerated().map { index, row -> [Int] in
      guard row.count > limit else { return row }
      guard truncate else {
        throw EmbeddingsError.inputTooLong(
          index: index, tokenCount: row.count, limit: limit)
      }
      // Keep the row's closing token (usually [SEP]/EOS) so last-token pooling
      // still sees it, matching Hugging Face truncation semantics.
      return row.prefix(limit - 1) + [row[row.count - 1]]
    }

    let longest = bounded.reduce(0) { max($0, $1.count) }
    // Pad to at least 16 for Apple Silicon alignment (MLXEmbedders README), but
    // never past the positional limit.
    let paddedLength = min(max(minPaddedLength, longest), limit)
    let rows = bounded.map { row in
      row + Array(repeating: padTokenId, count: paddedLength - row.count)
    }
    return EmbeddingsBatchPlan(
      rows: rows,
      lengths: bounded.map(\.count),
      paddedLength: paddedLength
    )
  }
}
