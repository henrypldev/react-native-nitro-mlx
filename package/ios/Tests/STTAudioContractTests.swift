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
  _ expected: STTAudioError,
  _ message: String,
  _ body: () throws -> Void
) throws {
  do {
    try body()
    throw TestFailure.failed("\(message): no error was thrown")
  } catch let error as STTAudioError {
    try expect(error == expected, "\(message): expected \(expected), received \(error)")
  }
}

private func bytes(_ ascii: String) -> [UInt8] {
  Array(ascii.utf8)
}

@main
struct STTAudioContractTests {
  static func main() throws {
    // Empty buffers are rejected.
    try expectError(.emptyAudio, "empty buffer is rejected") {
      try STTAudioContract.validate(byteCount: 0, prefix: [])
    }

    // Byte counts that are not Float32-aligned are rejected.
    try expectError(.misalignedAudio(byteCount: 6), "odd byte count is rejected") {
      try STTAudioContract.validate(byteCount: 6, prefix: [0, 1, 2, 3, 4, 5])
    }
    try expectError(.misalignedAudio(byteCount: 2), "Int16-sized tail is rejected") {
      try STTAudioContract.validate(byteCount: 2, prefix: [0, 1])
    }

    // Recognizable encoded containers are rejected with the format named.
    let containers: [(prefix: [UInt8], format: String)] = [
      (bytes("RIFF") + bytes("....WAVE"), "WAV (RIFF)"),
      (bytes("ID3") + [3, 0, 0, 0, 0], "MP3 (ID3)"),
      (bytes("fLaC") + [0, 0, 0, 0], "FLAC"),
      (bytes("OggS") + [0, 0, 0, 0], "Ogg"),
      (bytes("FORM") + bytes("....AIFF"), "AIFF (FORM)"),
      (bytes("caff") + [0, 1, 0, 0], "CAF"),
      ([0, 0, 0, 32] + bytes("ftypM4A "), "MP4/M4A"),
    ]
    for container in containers {
      try expect(
        STTAudioContract.detectEncodedFormat(prefix: container.prefix) == container.format,
        "\(container.format) magic bytes are detected"
      )
      try expectError(
        .encodedAudio(format: container.format),
        "\(container.format) container is rejected"
      ) {
        try STTAudioContract.validate(byteCount: 4096, prefix: container.prefix)
      }
    }

    // Raw Float32 content passes: zeros, and a prefix too short to sniff.
    try STTAudioContract.validate(byteCount: 4, prefix: [0, 0, 128, 63])
    try expect(
      STTAudioContract.detectEncodedFormat(prefix: [82, 73]) == nil,
      "a prefix shorter than any magic is not misdetected"
    )

    // MP3 frame sync (0xFF 0xEx) is NOT sniffed: those bytes occur in real Float32 data.
    try STTAudioContract.validate(byteCount: 8, prefix: [0xFF, 0xFB, 0x90, 0x00])

    // Sample-rate resolution: nil falls back to the model rate.
    let resolvedDefault = try STTAudioContract.resolveSampleRate(nil)
    try expect(
      resolvedDefault == STTAudioContract.modelSampleRate,
      "nil sample rate resolves to the model rate"
    )
    let resolved24k = try STTAudioContract.resolveSampleRate(24000)
    try expect(resolved24k == 24000, "a supported sample rate resolves to itself")
    let resolvedMin = try STTAudioContract.resolveSampleRate(8000)
    try expect(resolvedMin == 8000, "the minimum sample rate is accepted")
    let resolvedMax = try STTAudioContract.resolveSampleRate(48000)
    try expect(resolvedMax == 48000, "the maximum sample rate is accepted")

    // Out-of-range and non-finite rates are rejected.
    for bad in [7999.0, 48001.0, 0.0, -16000.0, Double.nan, Double.infinity] {
      try expectError(
        .unsupportedSampleRate(sampleRate: bad),
        "sample rate \(bad) is rejected"
      ) {
        _ = try STTAudioContract.resolveSampleRate(bad)
      }
    }

    // Resampling: identity when the rates match (same array back, no drift).
    let identity: [Float] = [0.1, -0.2, 0.3]
    try expect(
      STTAudioContract.resample(identity, from: 16000, to: 16000) == identity,
      "matching rates return the input unchanged"
    )

    // Empty input stays empty.
    try expect(
      STTAudioContract.resample([], from: 24000, to: 16000).isEmpty,
      "empty input resamples to empty output"
    )

    // Downsampling 24 kHz -> 16 kHz keeps 2/3 of the samples.
    let downIn = [Float](repeating: 0.5, count: 24000)
    let down = STTAudioContract.resample(downIn, from: 24000, to: 16000)
    try expect(down.count == 16000, "24 kHz to 16 kHz yields 2/3 the samples, got \(down.count)")
    try expect(
      down.allSatisfy { abs($0 - 0.5) < 1e-6 },
      "a constant signal stays constant after downsampling"
    )

    // Upsampling 8 kHz -> 16 kHz doubles the sample count.
    let upIn: [Float] = [0, 1, 0, -1]
    let up = STTAudioContract.resample(upIn, from: 8000, to: 16000)
    try expect(up.count == 8, "8 kHz to 16 kHz doubles the samples, got \(up.count)")
    // Linear interpolation: the midpoint between 0 and 1 is 0.5.
    try expect(abs(up[1] - 0.5) < 1e-6, "interpolated midpoints are linear, got \(up[1])")

    // A ramp survives resampling monotonically (no reordering or wraparound).
    let ramp = (0..<240).map { Float($0) }
    let rampDown = STTAudioContract.resample(ramp, from: 24000, to: 16000)
    try expect(rampDown.count == 160, "ramp downsample has the expected length")
    try expect(
      zip(rampDown, rampDown.dropFirst()).allSatisfy { $0 < $1 },
      "a strictly increasing ramp stays strictly increasing"
    )

    // Errors carry actionable descriptions for the JS side.
    try expect(
      STTAudioError.misalignedAudio(byteCount: 6).errorDescription?.contains("multiple of 4")
        == true,
      "misaligned error explains the 4-byte requirement"
    )
    try expect(
      STTAudioError.encodedAudio(format: "WAV (RIFF)").errorDescription?.contains("WAV (RIFF)")
        == true,
      "encoded error names the detected container"
    )
    try expect(
      STTAudioError.unsupportedSampleRate(sampleRate: 96000).errorDescription?.contains("96000")
        == true,
      "sample-rate error names the offending rate"
    )

    print("STTAudioContractTests passed")
  }
}
