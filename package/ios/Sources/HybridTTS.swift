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

  private func mlxArrayToArrayBuffer(_ audio: MLXArray) -> ArrayBuffer {
    let evaluated = audio.asType(.float32)
    MLX.eval(evaluated)
    let arrayData = evaluated.asData(access: .copy)
    let byteSize = arrayData.data.count
    let buffer = ArrayBuffer.allocate(size: byteSize)
    arrayData.data.withUnsafeBytes { srcPtr in
      UnsafeMutableRawPointer(buffer.data).copyMemory(
        from: srcPtr.baseAddress!,
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

        let loadedModel = try await TTSModelUtils.loadModel(modelRepo: modelId)

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

    return Promise.async { [self] in
      let task = Task<Any, Error> {
        let audio = try await model.generate(
          text: text,
          voice: options?.voice,
          refAudio: nil,
          refText: nil,
          language: nil
        )
        return self.mlxArrayToArrayBuffer(audio) as Any
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

    return Promise.async { [self] in
      let task = Task<Any, Error> {
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
            let buffer = self.mlxArrayToArrayBuffer(audio)
            onAudioChunk(buffer)
          case .token, .info, .progress:
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
