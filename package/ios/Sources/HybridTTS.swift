import Foundation
import NitroModules
internal import MLX
internal import MLXAudioTTS
internal import MLXAudioCore

enum TTSError: Error {
  case notLoaded
}

class HybridTTS: HybridTTSSpec {
  private var model: SpeechGenerationModel?
  private var activeTask: Task<Any, Error>?
  private var loadTask: Task<Void, Error>?

  var isLoaded: Bool { model != nil }
  var isGenerating: Bool { activeTask != nil }
  var modelId: String = ""
  var sampleRate: Double {
    Double(model?.sampleRate ?? 24000)
  }

  private func mlxArrayToArrayBuffer(
    _ audio: MLXArray,
    speed: TTSSpeed,
    outputSampleCount: Int? = nil
  ) -> ArrayBuffer {
    let evaluated = audio.asType(.float32)
    MLX.eval(evaluated)
    let samples = speed.adjustSpeed(
      evaluated.asArray(Float.self),
      outputSampleCount: outputSampleCount
    )
    let byteSize = samples.count * MemoryLayout<Float>.size
    let buffer = ArrayBuffer.allocate(size: byteSize)
    samples.withUnsafeBytes { srcPtr in
      guard byteSize > 0, let source = srcPtr.baseAddress else { return }
      UnsafeMutableRawPointer(buffer.data).copyMemory(
        from: source,
        byteCount: byteSize
      )
    }
    return buffer
  }

  func load(modelId: String, options: TTSLoadOptions?) throws -> Promise<Void> {
    self.loadTask?.cancel()

    return Promise.async { [self] in
      let task = Task { @MainActor in
        self.activeTask?.cancel()
        self.activeTask = nil
        self.model = nil
        MLX.Memory.clearCache()

        let loadedModel = try await TTS.loadModel(modelRepo: modelId)

        try Task.checkCancellation()

        self.model = loadedModel
        self.modelId = modelId

        options?.onProgress?(1.0)
      }

      self.loadTask = task
      try await task.value
    }
  }

  func generate(
    text: String,
    options: TTSGenerateOptions?
  ) throws -> Promise<ArrayBuffer> {
    guard let model else {
      throw TTSError.notLoaded
    }
    let speed = try TTSSpeed(options?.speed)

    return Promise.async { [self] in
      let task = Task<Any, Error> {
        let audio = try await model.generate(
          text: text,
          voice: options?.voice,
          refAudio: nil,
          refText: nil,
          language: nil
        )
        return self.mlxArrayToArrayBuffer(audio, speed: speed) as Any
      }

      self.activeTask = task
      defer { self.activeTask = nil }

      return try await task.value as! ArrayBuffer
    }
  }

  func stream(
    text: String,
    onAudioChunk: @escaping (ArrayBuffer) -> Void,
    options: TTSGenerateOptions?
  ) throws -> Promise<Void> {
    guard let model else {
      throw TTSError.notLoaded
    }
    let speed = try TTSSpeed(options?.speed)

    return Promise.async { [self] in
      let task = Task<Any, Error> {
        var speedPlanner = TTSStreamSpeedPlanner(speed: speed)
        let stream = model.generateStream(
          text: text,
          voice: options?.voice,
          refAudio: nil,
          refText: nil,
          language: nil,
          generationParameters: model.defaultGenerationParameters
        )

        for try await event in stream {
          if Task.isCancelled { break }

          switch event {
          case .audio(let audio):
            let outputSampleCount = speedPlanner.outputSampleCount(
              for: audio.dim(0)
            )
            guard outputSampleCount > 0 else { continue }
            let buffer = self.mlxArrayToArrayBuffer(
              audio,
              speed: speed,
              outputSampleCount: outputSampleCount
            )
            onAudioChunk(buffer)
          case .progress(let value):
            options?.onProgress?(value)
          case .token, .info:
            break
          }
        }
        return () as Any
      }

      self.activeTask = task
      defer { self.activeTask = nil }

      _ = try await task.value
    }
  }

  func stop() throws {
    activeTask?.cancel()
    activeTask = nil
  }

  func unload() throws {
    loadTask?.cancel()
    loadTask = nil
    activeTask?.cancel()
    activeTask = nil
    model = nil
    modelId = ""
    Memory.clearCache()
  }
}
