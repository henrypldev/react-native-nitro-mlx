import Foundation

private enum TestFailure: Error {
  case failed(String)
}

private func expect(_ condition: @autoclosure () -> Bool, _ message: String) throws {
  if !condition() {
    throw TestFailure.failed(message)
  }
}

private func expectSamples(
  _ actual: [Float],
  _ expected: [Float],
  _ message: String
) throws {
  try expect(actual.count == expected.count, "\(message) sample count")
  for (index, pair) in zip(actual, expected).enumerated() {
    try expect(
      abs(pair.0 - pair.1) < 0.0001,
      "\(message) sample \(index): expected \(pair.1), received \(pair.0)"
    )
  }
}

@main
struct TTSSpeedTests {
  static func main() throws {
    let sampleRate = 24_000
    let inputSampleCount = sampleRate
    let slow = try TTSSpeed(0.5)
    let normal = try TTSSpeed(1.0)
    let fast = try TTSSpeed(2.0)

    let slowDuration = Double(slow.outputSampleCount(for: inputSampleCount)) / Double(sampleRate)
    let normalDuration = Double(normal.outputSampleCount(for: inputSampleCount)) / Double(sampleRate)
    let fastDuration = Double(fast.outputSampleCount(for: inputSampleCount)) / Double(sampleRate)

    try expect(slowDuration == 2.0, "slow output duration")
    try expect(normalDuration == 1.0, "normal output duration")
    try expect(fastDuration == 0.5, "fast output duration")
    try expect(slowDuration > normalDuration, "slow must be longer than normal")
    try expect(normalDuration > fastDuration, "normal must be longer than fast")

    let inputSamples: [Float] = [0, 7, 14, 21]
    try expectSamples(
      slow.adjustSpeed(inputSamples),
      [0, 3, 6, 9, 12, 15, 18, 21],
      "slow interpolation"
    )
    try expectSamples(
      normal.adjustSpeed(inputSamples),
      inputSamples,
      "normal interpolation"
    )
    try expectSamples(
      fast.adjustSpeed(inputSamples),
      [0, 21],
      "fast interpolation"
    )

    let streamedInputChunks = [7_777, 8_888, 7_335]
    for speed in [slow, normal, fast] {
      var planner = TTSStreamSpeedPlanner(speed: speed)
      let streamedOutputSampleCount = streamedInputChunks.reduce(into: 0) { total, chunk in
        total += planner.outputSampleCount(for: chunk)
      }
      try expect(
        streamedOutputSampleCount == speed.outputSampleCount(for: inputSampleCount),
        "streaming and one-shot output durations must match at speed \(speed.multiplier)"
      )
    }

    for unsupportedSpeed in [0.0, 0.49, 2.01, Double.infinity, Double.nan] {
      do {
        _ = try TTSSpeed(unsupportedSpeed)
        throw TestFailure.failed("unsupported speed \(unsupportedSpeed) was accepted")
      } catch is TTSSpeedError {
        // Expected.
      }
    }
  }
}
