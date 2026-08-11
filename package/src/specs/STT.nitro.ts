import type { HybridObject } from 'react-native-nitro-modules'

export interface STTLoadOptions {
  onProgress?: (progress: number) => void
}

/** Options for transcribing raw native-endian mono Float32 PCM buffers. */
export interface STTTranscribeOptions {
  /**
   * Sample rate of the PCM in Hz (default 16000). Rates within 8000–48000 are
   * resampled natively before inference; others are rejected.
   */
  sampleRate?: number
  /** Spoken language (e.g. `'Spanish'`). Omit to auto-detect. */
  language?: string
}

export interface STTListeningOptions {
  /**
   * Spoken language applied to `transcribeBuffer`/`stopListening` results.
   * Omit to auto-detect.
   */
  language?: string
}

export interface STTTranscriptionInfo {
  promptTokens: number
  generationTokens: number
  tokensPerSecond: number
  prefillTime: number
  generateTime: number
}

export interface STT extends HybridObject<{ ios: 'swift' }> {
  readonly isLoaded: boolean
  readonly isTranscribing: boolean
  readonly isListening: boolean
  readonly modelId: string

  load(modelId: string, options?: STTLoadOptions): Promise<void>

  transcribe(audio: ArrayBuffer, options?: STTTranscribeOptions): Promise<string>
  transcribeStream(
    audio: ArrayBuffer,
    onToken: (token: string) => void,
    options?: STTTranscribeOptions,
  ): Promise<string>

  startListening(options?: STTListeningOptions): Promise<void>
  transcribeBuffer(): Promise<string>
  stopListening(): Promise<string>

  stop(): void
  unload(): void
}
