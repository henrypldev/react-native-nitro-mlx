import Foundation
import NitroModules
internal import MLX
internal import MLXAudioSTT
internal import MLXAudioCore

enum STTError: Error {
  case notLoaded
  case notListening
  case alreadyListening
}

class HybridSTT: HybridSTTSpec {
  private var model: Qwen3ASRModel?
  private var activeTask: Task<String, Error>?
  private var loadTask: Task<Void, Error>?
  private var captureManager: AudioCaptureManager?
  private var listeningLanguage: String?

  var isLoaded: Bool { model != nil }
  var isTranscribing: Bool { activeTask != nil }
  var isListening: Bool { captureManager?.isCapturing ?? false }
  var modelId: String = ""

  /// Must stay synchronous: the JS ArrayBuffer is only valid for the duration
  /// of the bridge call, so the copy cannot be deferred into a Task.
  private func samplesFromArrayBuffer(
    _ buffer: ArrayBuffer,
    sampleRate: Double?
  ) throws -> [Float] {
    let byteCount = buffer.size
    let rawPtr = UnsafeRawPointer(buffer.data)
    let prefix = [UInt8](UnsafeRawBufferPointer(start: rawPtr, count: min(byteCount, 12)))
    try STTAudioContract.validate(byteCount: byteCount, prefix: prefix)
    let sourceRate = try STTAudioContract.resolveSampleRate(sampleRate)

    let count = byteCount / MemoryLayout<Float>.size
    let floatPtr = rawPtr.bindMemory(to: Float.self, capacity: count)
    let samples = Array(UnsafeBufferPointer(start: floatPtr, count: count))
    return STTAudioContract.resample(
      samples,
      from: sourceRate,
      to: STTAudioContract.modelSampleRate
    )
  }

  func load(modelId: String, options: STTLoadOptions?) throws -> Promise<Void> {
    self.loadTask?.cancel()

    return Promise.async { [self] in
      let task = Task { @MainActor in
        self.activeTask?.cancel()
        self.activeTask = nil
        self.model = nil
        MLX.Memory.clearCache()

        let loadedModel = try await Qwen3ASRModel.fromPretrained(modelId)

        try Task.checkCancellation()

        self.model = loadedModel
        self.modelId = modelId

        options?.onProgress?(1.0)
      }

      self.loadTask = task
      try await task.value
    }
  }

  func transcribe(audio: ArrayBuffer, options: STTTranscribeOptions?) throws -> Promise<String> {
    guard let model else {
      throw STTError.notLoaded
    }

    let samples = try samplesFromArrayBuffer(audio, sampleRate: options?.sampleRate)
    let language = options?.language

    return Promise.async { [self] in
      let task = Task<String, Error> {
        let output = model.generate(audio: MLXArray(samples), language: language)
        return output.text
      }

      self.activeTask = task
      defer { self.activeTask = nil }

      return try await task.value
    }
  }

  func transcribeStream(
    audio: ArrayBuffer,
    onToken: @escaping (_ token: String) -> Void,
    options: STTTranscribeOptions?
  ) throws -> Promise<String> {
    guard let model else {
      throw STTError.notLoaded
    }

    let samples = try samplesFromArrayBuffer(audio, sampleRate: options?.sampleRate)
    let language = options?.language

    return Promise.async { [self] in
      let task = Task<String, Error> {
        let stream = model.generateStream(audio: MLXArray(samples), language: language)
        var finalText = ""

        for try await event in stream {
          if Task.isCancelled { break }

          switch event {
          case .token(let token):
            onToken(token)
          case .result(let output):
            finalText = output.text
          case .info:
            break
          }
        }

        return finalText
      }

      self.activeTask = task
      defer { self.activeTask = nil }

      return try await task.value
    }
  }

  func startListening(options: STTListeningOptions?) throws -> Promise<Void> {
    guard model != nil else {
      throw STTError.notLoaded
    }
    guard captureManager == nil || !captureManager!.isCapturing else {
      throw STTError.alreadyListening
    }

    listeningLanguage = options?.language

    return Promise.async { [self] in
      let manager = AudioCaptureManager()
      self.captureManager = manager
      try await manager.startCapturing()
    }
  }

  func transcribeBuffer() throws -> Promise<String> {
    guard let model else {
      throw STTError.notLoaded
    }
    guard let manager = captureManager, manager.isCapturing else {
      throw STTError.notListening
    }
    guard let audio = manager.snapshotAndClear() else {
      return Promise.resolved(withResult: "")
    }

    let language = listeningLanguage
    return Promise.async { [self] in
      let task = Task<String, Error> {
        let output = model.generate(audio: audio, language: language)
        return output.text
      }

      self.activeTask = task
      defer { self.activeTask = nil }

      let result = try await task.value
      MLX.Memory.clearCache()
      return result
    }
  }

  func stopListening() throws -> Promise<String> {
    guard let model else {
      throw STTError.notLoaded
    }
    guard let manager = captureManager, manager.isCapturing else {
      throw STTError.notListening
    }

    let audio = manager.stopCapturing()
    self.captureManager = nil
    let language = listeningLanguage
    listeningLanguage = nil

    return Promise.async { [self] in
      let task = Task<String, Error> {
        let output = model.generate(audio: audio, language: language)
        return output.text
      }

      self.activeTask = task
      defer { self.activeTask = nil }

      let result = try await task.value
      MLX.Memory.clearCache()
      return result
    }
  }

  func stop() throws {
    activeTask?.cancel()
    activeTask = nil
    if let manager = captureManager, manager.isCapturing {
      _ = manager.stopCapturing()
    }
    captureManager = nil
    listeningLanguage = nil
  }

  func unload() throws {
    loadTask?.cancel()
    loadTask = nil
    activeTask?.cancel()
    activeTask = nil
    if let manager = captureManager, manager.isCapturing {
      _ = manager.stopCapturing()
    }
    captureManager = nil
    listeningLanguage = nil
    model = nil
    modelId = ""
    Memory.clearCache()
  }
}
