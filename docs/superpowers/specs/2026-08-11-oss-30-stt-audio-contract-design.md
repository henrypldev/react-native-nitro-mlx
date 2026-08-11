# OSS-30: Define and validate the MLX direct STT audio contract

Date: 2026-08-11 · Ticket: [OSS-30](https://linear.app/henry-pl-llc/issue/OSS-30)

## Problem

`STT.transcribe`/`transcribeStream` treat every nonempty `ArrayBuffer` as native-endian
mono Float32 at 16 kHz:

1. **No format contract.** WAV, MP3, Int16, and odd byte lengths are silently
   reinterpreted as Float32 samples and transcribed as garbage.
2. **No sample-rate contract.** The library's own TTS output is 24 kHz; feeding it back
   into STT plays it at 2/3 speed inside the model.
3. **Forced language.** `HybridSTT` hardcodes `language: "English"` even though
   Qwen3ASR auto-detects when `language` is `nil` and supports forcing other languages.
4. **Missing consumer docs.** `NSMicrophoneUsageDescription` is set in the example app
   but never documented for consumers of the library.

## Approaches considered

- **A. Document-only.** Narrowly document "raw PCM Float32 mono 16 kHz" and change
  nothing. Rejected: silent corruption stays silent.
- **B. Full audio descriptor (format/encoding/channels/rate) with decoding.** Rejected:
  decoding WAV/MP3 containers is a codec concern; iOS already ships AVFoundation for
  that. The library's contract stays raw PCM.
- **C. Explicit options + validation + resampling (chosen).** The buffer stays raw
  native-endian mono Float32. A new `STTTranscribeOptions` makes `sampleRate` and
  `language` explicit. Both layers reject misaligned buffers and recognizable encoded
  containers with actionable errors. Non-16-kHz PCM inside a supported range is
  linearly resampled before inference; rates outside the range are rejected.

## Design

### Pure contract — `package/ios/Sources/STTAudioContract.swift`

Foundation-only (no MLX import), following the `EmbeddingsBatchPlanner` pattern so the
logic is unit-testable off-device with the `swiftc` test-script pattern.

```swift
enum STTAudioError: Error, LocalizedError, Equatable {
  case emptyAudio
  case misalignedAudio(byteCount: Int)
  case encodedAudio(format: String)
  case unsupportedSampleRate(sampleRate: Double)
}

struct STTAudioContract {
  static let modelSampleRate: Double = 16000  // Qwen3ASR input rate, mirrored in TS
  static let minSampleRate: Double = 8000
  static let maxSampleRate: Double = 48000

  static func detectEncodedFormat(prefix: [UInt8]) -> String?
  static func validate(byteCount: Int, prefix: [UInt8]) throws
  static func resolveSampleRate(_ requested: Double?) throws -> Double
  static func resample(_ samples: [Float], from: Double, to: Double) -> [Float]
}
```

Rules:

- `byteCount == 0` → `.emptyAudio`; `byteCount % 4 != 0` → `.misalignedAudio` (raw
  Float32 requires 4-byte frames; Int16 uploads usually trip this or the peak check in
  consumers).
- Container sniffing rejects unambiguous magic bytes only: `RIFF` (WAV), `ID3` (MP3),
  `fLaC`, `OggS`, `FORM` (AIFF), `caff` (CAF), and `ftyp` at offset 4 (MP4/M4A). MP3
  frame-sync sniffing (`0xFF 0xEx`) is deliberately omitted — those bytes occur in
  legitimate Float32 noise.
- `resolveSampleRate(nil)` → 16000 (the documented default). Rates must be finite and
  within [8000, 48000]; anything else → `.unsupportedSampleRate`.
- `resample` is linear interpolation, mono only. Identity when source == target.
  Linear is adequate for speech ASR input; callers needing mastering-grade SRC can
  resample upstream with AVFoundation.

### `HybridSTT` wiring

- `transcribe`/`transcribeStream` validate byte count + prefix via the contract before
  binding memory, resample when `options.sampleRate != 16000`, and pass
  `options.language` through (`nil` → Qwen3ASR auto-detect).
- `startListening` gains `STTListeningOptions { language? }`; the stored language is
  applied by `transcribeBuffer`/`stopListening`. The mic path already captures
  16 kHz mono Float32, so no descriptor is needed there.
- The previous hardcoded `"English"` default becomes auto-detect everywhere.

### API — `STT.nitro.ts`

```ts
interface STTTranscribeOptions { sampleRate?: number; language?: string }
interface STTListeningOptions { language?: string }

transcribe(audio: ArrayBuffer, options?: STTTranscribeOptions): Promise<string>
transcribeStream(audio, onToken, options?: STTTranscribeOptions): Promise<string>
startListening(options?: STTListeningOptions): Promise<void>
```

Nitrogen regenerated via `bun specs`.

### TS-side guards — `runtime.ts`

Mirror the cheap checks before crossing the bridge (Swift re-checks):

- `validateSTTAudio`: ArrayBuffer type + nonempty (existing), 4-byte alignment,
  encoded-container magic-byte rejection with the detected format named in the error.
- `validateSTTTranscribeOptions` / `validateSTTListeningOptions`: `sampleRate` must be
  an integer in `[STT_MIN_SAMPLE_RATE, STT_MAX_SAMPLE_RATE]`; `language` must be a
  non-empty string when present.
- Exported constants: `STT_SAMPLE_RATE = 16000`, `STT_MIN_SAMPLE_RATE = 8000`,
  `STT_MAX_SAMPLE_RATE = 48000`.

### README

- New "Audio format" subsection under Speech-to-Text: raw PCM, Float32, mono,
  native-endian, 16 kHz default, `sampleRate` option resamples within 8–48 kHz,
  encoded containers rejected.
- `language` option documented (auto-detect default).
- Requirements section documents `NSMicrophoneUsageDescription` for live
  transcription (with the Expo `app.json` form used by the example app).

## Testing

- **Swift** (`package/ios/Tests/STTAudioContractTests.swift`, run via new
  `test:ios-stt-audio` swiftc script): empty/misaligned byte counts, each container
  magic, non-container prefixes accepted, short prefixes, sample-rate resolution
  (nil default, bounds, NaN/infinite/zero/negative), resample identity, 24 k→16 k
  length and content, 8 k→16 k upsample, constant-signal preservation, empty input.
- **TS** (`bun test`): alignment rejection, WAV/ID3/fLaC/OggS/ftyp rejection, valid
  Float32 acceptance, sampleRate bounds and non-integer rejection, language type
  checks, pass-through of options.
- On-device inference (multilingual output quality) is not unit-testable in CI; the
  contract boundary keeps that surface minimal.

## Out of scope

Container decoding (WAV/MP3 parsing), stereo downmix, Int16 auto-conversion, Android,
per-chunk streaming input.
