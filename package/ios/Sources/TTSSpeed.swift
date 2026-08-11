import Foundation

enum TTSSpeedError: Error, LocalizedError {
  case unsupported(Double)

  var errorDescription: String? {
    switch self {
    case .unsupported(let speed):
      return "TTS speed must be between \(TTSSpeed.minimum) and \(TTSSpeed.maximum). Received \(speed)."
    }
  }
}

struct TTSSpeed {
  static let minimum = 0.5
  static let maximum = 2.0

  let multiplier: Float

  init(_ value: Double?) throws {
    let value = value ?? 1.0
    guard value.isFinite, value >= Self.minimum, value <= Self.maximum else {
      throw TTSSpeedError.unsupported(value)
    }
    self.multiplier = Float(value)
  }

  func outputSampleCount(for inputSampleCount: Int) -> Int {
    guard inputSampleCount > 0 else { return 0 }
    return max(1, Int(Float(inputSampleCount) / multiplier))
  }

  func adjustSpeed(
    _ samples: [Float],
    outputSampleCount: Int? = nil
  ) -> [Float] {
    guard !samples.isEmpty else { return [] }

    let outputSampleCount = outputSampleCount
      ?? self.outputSampleCount(for: samples.count)
    guard outputSampleCount > 0 else { return [] }
    guard outputSampleCount != samples.count else { return samples }
    guard outputSampleCount > 1 else { return [samples[0]] }

    let inputLastIndex = samples.count - 1
    let positionScale = Float(inputLastIndex) / Float(outputSampleCount - 1)
    return (0..<outputSampleCount).map { outputIndex in
      let position = Float(outputIndex) * positionScale
      let leftIndex = Int(floor(position))
      let rightIndex = min(leftIndex + 1, inputLastIndex)
      let rightWeight = position - Float(leftIndex)
      let leftWeight = 1 - rightWeight
      return leftWeight * samples[leftIndex] + rightWeight * samples[rightIndex]
    }
  }
}

/// Carries fractional sample-count rounding across streamed chunks so streaming
/// and one-shot synthesis have the same total duration.
struct TTSStreamSpeedPlanner {
  private let speed: TTSSpeed
  private var totalInputSampleCount = 0
  private var totalOutputSampleCount = 0

  init(speed: TTSSpeed) {
    self.speed = speed
  }

  mutating func outputSampleCount(for inputSampleCount: Int) -> Int {
    guard inputSampleCount > 0 else { return 0 }

    totalInputSampleCount += inputSampleCount
    let targetTotal = speed.outputSampleCount(for: totalInputSampleCount)
    let chunkOutputSampleCount = targetTotal - totalOutputSampleCount
    totalOutputSampleCount = targetTotal
    return chunkOutputSampleCount
  }
}
