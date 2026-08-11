import type { HybridObject } from 'react-native-nitro-modules'

export interface STTLoadOptions {
  onProgress?: (progress: number) => void
}

/**
 * Audio contract for direct transcription. The buffer itself is always raw
 * native-endian mono Float32 PCM; these options make the remaining
 * characteristics explicit.
 */
export interface STTTranscribeOptions {
  /**
   * Sample rate of the provided PCM in Hz. Defaults to 16000 (the model's
   * input rate). Rates between 8000 and 48000 are resampled natively before
   * inference; anything else is rejected.
   */
  sampleRate?: number
  /**
   * Spoken language of the audio (e.g. `'English'`, `'Spanish'`). Omit to let
   * the model auto-detect the language.
   */
  language?: string
}

export interface STTListeningOptions {
  /**
   * Spoken language applied to `transcribeBuffer`/`stopListening` results.
   * Omit to let the model auto-detect the language.
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
