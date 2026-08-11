import Foundation

enum STTAudioError: Error, LocalizedError, Equatable {
  case emptyAudio
  case misalignedAudio(byteCount: Int)
  case encodedAudio(format: String)
  case unsupportedSampleRate(sampleRate: Double)

  var errorDescription: String? {
    switch self {
    case .emptyAudio:
      return "STT audio buffer is empty."
    case .misalignedAudio(let byteCount):
      return
        "STT audio must be raw native-endian Float32 PCM; byte length \(byteCount) is not a multiple of 4."
    case .encodedAudio(let format):
      return
        "STT audio looks like an encoded \(format) container. Decode it to raw mono Float32 PCM before calling transcribe."
    case .unsupportedSampleRate(let sampleRate):
      return
        "STT sampleRate \(sampleRate) is unsupported. Provide a rate between \(Int(STTAudioContract.minSampleRate)) and \(Int(STTAudioContract.maxSampleRate)) Hz."
    }
  }

  static func == (lhs: STTAudioError, rhs: STTAudioError) -> Bool {
    switch (lhs, rhs) {
    case (.emptyAudio, .emptyAudio):
      return true
    case (.misalignedAudio(let l), .misalignedAudio(let r)):
      return l == r
    case (.encodedAudio(let l), .encodedAudio(let r)):
      return l == r
    case (.unsupportedSampleRate(let l), .unsupportedSampleRate(let r)):
      // NaN never equals itself; compare bit patterns so NaN cases match in tests.
      return l.bitPattern == r.bitPattern
    default:
      return false
    }
  }
}

/// Pure (Foundation-only) audio-contract rules for the STT bridge, mirrored by the
/// TS-side guards in `runtime.ts`. Kept free of MLX imports so the rules run under
/// plain `swiftc` unit tests off-device.
struct STTAudioContract {
  /// Qwen3ASR consumes 16 kHz mono Float32; mirrored in TS as `STT_SAMPLE_RATE`.
  static let modelSampleRate: Double = 16000
  static let minSampleRate: Double = 8000
  static let maxSampleRate: Double = 48000

  /// Magic-byte signatures of encoded containers callers commonly pass by mistake.
  /// Only unambiguous ASCII magics are sniffed; MP3 frame sync (0xFF 0xEx) is skipped
  /// because those bytes legitimately occur in raw Float32 sample data.
  private static let signatures: [(magic: [UInt8], offset: Int, format: String)] = [
    (Array("RIFF".utf8), 0, "WAV (RIFF)"),
    (Array("ID3".utf8), 0, "MP3 (ID3)"),
    (Array("fLaC".utf8), 0, "FLAC"),
    (Array("OggS".utf8), 0, "Ogg"),
    (Array("FORM".utf8), 0, "AIFF (FORM)"),
    (Array("caff".utf8), 0, "CAF"),
    (Array("ftyp".utf8), 4, "MP4/M4A"),
  ]

  static func detectEncodedFormat(prefix: [UInt8]) -> String? {
    for signature in signatures {
      let end = signature.offset + signature.magic.count
      guard prefix.count >= end else { continue }
      if Array(prefix[signature.offset..<end]) == signature.magic {
        return signature.format
      }
    }
    return nil
  }

  static func validate(byteCount: Int, prefix: [UInt8]) throws {
    guard byteCount > 0 else {
      throw STTAudioError.emptyAudio
    }
    guard byteCount % MemoryLayout<Float32>.size == 0 else {
      throw STTAudioError.misalignedAudio(byteCount: byteCount)
    }
    if let format = detectEncodedFormat(prefix: prefix) {
      throw STTAudioError.encodedAudio(format: format)
    }
  }

  static func resolveSampleRate(_ requested: Double?) throws -> Double {
    guard let requested else {
      return modelSampleRate
    }
    guard requested.isFinite, requested >= minSampleRate, requested <= maxSampleRate else {
      throw STTAudioError.unsupportedSampleRate(sampleRate: requested)
    }
    return requested
  }

  /// Linear-interpolation resampling for mono PCM. Adequate for speech ASR input;
  /// callers needing higher-fidelity SRC should resample upstream (e.g. AVFoundation).
  static func resample(_ samples: [Float], from source: Double, to target: Double) -> [Float] {
    guard source != target, !samples.isEmpty else {
      return samples
    }

    let ratio = source / target
    let outputCount = max(1, Int((Double(samples.count) * target / source).rounded()))
    var output = [Float]()
    output.reserveCapacity(outputCount)

    for index in 0..<outputCount {
      let position = Double(index) * ratio
      let lower = min(Int(position), samples.count - 1)
      let upper = min(lower + 1, samples.count - 1)
      let fraction = Float(position - Double(lower))
      output.append(samples[lower] + (samples[upper] - samples[lower]) * fraction)
    }

    return output
  }
}
