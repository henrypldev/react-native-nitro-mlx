import type { HybridObject } from 'react-native-nitro-modules'

export interface TTSLoadOptions {
  onProgress?: (progress: number) => void
}

export interface TTSGenerateOptions {
  voice?: string
  /**
   * Speech-rate multiplier. Supported range is 0.5 (slowest) through 2.0 (fastest).
   * Currently implemented with linear resampling, so pitch changes with speech rate.
   */
  speed?: number
  /**
   * Fractional generation progress (0-1). Only fires during `stream()`, and only for
   * models with a deterministic step count (e.g. diffusion denoise steps). `generate()`
   * is a one-shot call that emits no events, so this is never invoked there.
   */
  onProgress?: (progress: number) => void
}

export interface TTS extends HybridObject<{ ios: 'swift' }> {
  readonly isLoaded: boolean
  readonly isGenerating: boolean
  readonly modelId: string
  readonly sampleRate: number

  load(modelId: string, options?: TTSLoadOptions): Promise<void>
  generate(text: string, options?: TTSGenerateOptions): Promise<ArrayBuffer>
  stream(
    text: string,
    onAudioChunk: (audio: ArrayBuffer) => void,
    options?: TTSGenerateOptions,
  ): Promise<void>
  stop(): void
  unload(): void
}
