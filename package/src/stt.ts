import { NitroModules } from 'react-native-nitro-modules'
import {
  assertNonEmptyString,
  createSafeCallback,
  validateSTTAudio,
  validateSTTListeningOptions,
  validateSTTLoadOptions,
  validateSTTTranscribeOptions,
} from './runtime'
import type {
  STTListeningOptions,
  STTLoadOptions,
  STT as STTSpec,
  STTTranscribeOptions,
} from './specs/STT.nitro'

let instance: STTSpec | null = null

function getInstance(): STTSpec {
  if (!instance) {
    instance = NitroModules.createHybridObject<STTSpec>('STT')
  }
  if (!instance) {
    throw new Error('Failed to initialize the STT Nitro module.')
  }
  return instance
}

export const STT = {
  load(modelId: string, options?: STTLoadOptions): Promise<void> {
    return getInstance().load(
      assertNonEmptyString(modelId, 'STT modelId'),
      validateSTTLoadOptions(options),
    )
  },

  transcribe(audio: ArrayBuffer, options?: STTTranscribeOptions): Promise<string> {
    return getInstance().transcribe(
      validateSTTAudio(audio, 'STT audio'),
      validateSTTTranscribeOptions(options),
    )
  },

  transcribeStream(
    audio: ArrayBuffer,
    onToken: (token: string) => void,
    options?: STTTranscribeOptions,
  ): Promise<string> {
    return getInstance().transcribeStream(
      validateSTTAudio(audio, 'STT audio'),
      createSafeCallback('STT.transcribeStream onToken', onToken) ?? (() => {}),
      validateSTTTranscribeOptions(options),
    )
  },

  startListening(options?: STTListeningOptions): Promise<void> {
    return getInstance().startListening(validateSTTListeningOptions(options))
  },

  transcribeBuffer(): Promise<string> {
    return getInstance().transcribeBuffer()
  },

  stopListening(): Promise<string> {
    return getInstance().stopListening()
  },

  stop(): void {
    getInstance().stop()
  },

  unload(): void {
    getInstance().unload()
  },

  get isLoaded(): boolean {
    return getInstance().isLoaded
  },

  get isTranscribing(): boolean {
    return getInstance().isTranscribing
  },

  get isListening(): boolean {
    return getInstance().isListening
  },

  get modelId(): string {
    return getInstance().modelId
  },
}
