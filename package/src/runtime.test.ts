import { describe, expect, it } from 'bun:test'
import {
  assertArrayBuffer,
  assertBoolean,
  assertNonEmptyString,
  createSafeCallback,
  EMBEDDINGS_MAX_BATCH_SIZE,
  mapStreamEventEnvelope,
  STT_MAX_SAMPLE_RATE,
  STT_MIN_SAMPLE_RATE,
  STT_SAMPLE_RATE,
  safeJsonParse,
  TTS_MAX_SPEED,
  TTS_MIN_SPEED,
  validateEmbeddingsBatch,
  validateEmbeddingsEmbedOptions,
  validateLLMLoadOptions,
  validateSTTAudio,
  validateSTTListeningOptions,
  validateSTTTranscribeOptions,
  validateTTSGenerateOptions,
} from './runtime'

describe('runtime guards', () => {
  it('accepts valid primitive inputs', () => {
    expect(assertNonEmptyString('mlx-community/Qwen3-0.6B-4bit', 'modelId')).toBe(
      'mlx-community/Qwen3-0.6B-4bit',
    )
    expect(assertBoolean(true, 'debug')).toBe(true)
    expect(assertArrayBuffer(new ArrayBuffer(8), 'audio').byteLength).toBe(8)
  })

  it('rejects invalid primitive inputs', () => {
    expect(() => assertNonEmptyString('   ', 'modelId')).toThrow(
      'must be a non-empty string',
    )
    expect(() => assertBoolean('true', 'debug')).toThrow('must be a boolean')
    expect(() => assertArrayBuffer(new ArrayBuffer(0), 'audio')).toThrow(
      'must not be empty',
    )
  })

  it('wraps callbacks so user exceptions do not escape', () => {
    const originalConsoleError = console.error
    const errors: unknown[][] = []
    console.error = (...args: unknown[]) => {
      errors.push(args)
    }

    try {
      const callback = createSafeCallback('LLM.stream onToken', () => {
        throw new Error('boom')
      })

      expect(() => callback?.('token')).not.toThrow()
      expect(errors).toHaveLength(1)
      expect(String(errors[0]?.[0])).toContain('callback threw')
    } finally {
      console.error = originalConsoleError
    }
  })

  it('rejects duplicate tool names', () => {
    expect(() =>
      validateLLMLoadOptions({
        tools: [
          {
            name: 'weather',
            description: 'Weather tool',
            parameters: [],
            handler: async () => ({}),
          },
          {
            name: 'weather',
            description: 'Duplicate weather tool',
            parameters: [],
            handler: async () => ({}),
          },
        ],
      }),
    ).toThrow('tools must have unique names')
  })

  it('rejects invalid TTS generation options', () => {
    expect(() => validateTTSGenerateOptions({ speed: TTS_MIN_SPEED - 0.01 })).toThrow(
      'must be between 0.5 and 2',
    )
    expect(() => validateTTSGenerateOptions({ speed: TTS_MAX_SPEED + 0.01 })).toThrow(
      'must be between 0.5 and 2',
    )
    expect(() => validateTTSGenerateOptions({ speed: Number.POSITIVE_INFINITY })).toThrow(
      'must be between 0.5 and 2',
    )
    expect(() => validateTTSGenerateOptions({ speed: Number.NaN })).toThrow(
      'must be between 0.5 and 2',
    )
    expect(validateTTSGenerateOptions({ speed: TTS_MIN_SPEED })?.speed).toBe(
      TTS_MIN_SPEED,
    )
    expect(validateTTSGenerateOptions({ speed: 1 })?.speed).toBe(1)
    expect(validateTTSGenerateOptions({ speed: TTS_MAX_SPEED })?.speed).toBe(
      TTS_MAX_SPEED,
    )
    expect(() => validateTTSGenerateOptions({ voice: '   ' })).toThrow(
      'must be a non-empty string',
    )
  })

  it('falls back on malformed JSON payloads', () => {
    expect(safeJsonParse('{"ok":true}', { ok: false })).toEqual({ ok: true })
    expect(safeJsonParse('{bad json', { ok: false })).toEqual({ ok: false })
  })
})

describe('stream event envelope mapping', () => {
  it('expands each kind into its discriminated event', () => {
    expect(mapStreamEventEnvelope({ kind: 'token', token: 'hi' })).toEqual({
      type: 'token',
      token: 'hi',
    })
    expect(mapStreamEventEnvelope({ kind: 'thinking_chunk', chunk: 'why' })).toEqual({
      type: 'thinking_chunk',
      chunk: 'why',
    })
    expect(
      mapStreamEventEnvelope({
        kind: 'tool_call_start',
        id: '1',
        name: 'search',
        arguments: '{"q":"x"}',
      }),
    ).toEqual({
      type: 'tool_call_start',
      id: '1',
      name: 'search',
      arguments: '{"q":"x"}',
    })
    expect(mapStreamEventEnvelope({ kind: 'generation_start', timestamp: 42 })).toEqual({
      type: 'generation_start',
      timestamp: 42,
    })
  })

  it('carries stats through on generation_end', () => {
    const stats = {
      tokenCount: 10,
      tokensPerSecond: 5,
      timeToFirstToken: 100,
      totalTime: 2000,
      toolExecutionTime: 0,
    }
    expect(
      mapStreamEventEnvelope({ kind: 'generation_end', content: 'done', stats }),
    ).toEqual({ type: 'generation_end', content: 'done', stats })
  })

  it('emits generation_end with zeroed stats rather than dropping it', () => {
    expect(mapStreamEventEnvelope({ kind: 'generation_end', content: 'done' })).toEqual({
      type: 'generation_end',
      content: 'done',
      stats: {
        tokenCount: 0,
        tokensPerSecond: 0,
        timeToFirstToken: 0,
        totalTime: 0,
        toolExecutionTime: 0,
      },
    })
  })

  it('maps generation_error with error, stage, and partial stats', () => {
    const stats = {
      tokenCount: 12,
      tokensPerSecond: 0,
      timeToFirstToken: 0,
      totalTime: 4200,
      toolExecutionTime: 0,
    }
    expect(
      mapStreamEventEnvelope({
        kind: 'generation_error',
        error: 'Generation failed during generate: boom',
        stage: 'generate',
        stats,
      }),
    ).toEqual({
      type: 'generation_error',
      error: 'Generation failed during generate: boom',
      stage: 'generate',
      stats,
    })
  })

  it('emits generation_error with empty fields rather than dropping it', () => {
    expect(mapStreamEventEnvelope({ kind: 'generation_error' })).toEqual({
      type: 'generation_error',
      error: '',
      stage: '',
      stats: {
        tokenCount: 0,
        tokensPerSecond: 0,
        timeToFirstToken: 0,
        totalTime: 0,
        toolExecutionTime: 0,
      },
    })
  })

  it('returns null for an unrecognized kind', () => {
    expect(
      mapStreamEventEnvelope({
        kind: 'not_a_kind',
      } as unknown as Parameters<typeof mapStreamEventEnvelope>[0]),
    ).toBeNull()
  })
})

describe('embeddings guards', () => {
  it('passes through valid embed options', () => {
    expect(validateEmbeddingsEmbedOptions(undefined)).toBeUndefined()
    expect(validateEmbeddingsEmbedOptions({ truncate: true })).toEqual({
      truncate: true,
    })
    expect(validateEmbeddingsEmbedOptions({ truncate: false })).toEqual({
      truncate: false,
    })
    expect(validateEmbeddingsEmbedOptions({})).toEqual({})
  })

  it('rejects a non-boolean truncate option', () => {
    expect(() =>
      validateEmbeddingsEmbedOptions({
        truncate: 'yes',
      } as unknown as Parameters<typeof validateEmbeddingsEmbedOptions>[0]),
    ).toThrow('must be a boolean')
  })

  it('accepts a batch at the size limit', () => {
    const texts = Array.from({ length: EMBEDDINGS_MAX_BATCH_SIZE }, (_, i) => `t${i}`)
    expect(validateEmbeddingsBatch(texts)).toEqual(texts)
  })

  it('rejects a batch over the size limit', () => {
    const texts = Array.from({ length: EMBEDDINGS_MAX_BATCH_SIZE + 1 }, (_, i) => `t${i}`)
    expect(() => validateEmbeddingsBatch(texts)).toThrow(
      `at most ${EMBEDDINGS_MAX_BATCH_SIZE}`,
    )
  })

  it('rejects empty or non-array batches', () => {
    expect(() => validateEmbeddingsBatch([])).toThrow('non-empty array')
    expect(() => validateEmbeddingsBatch('hello' as unknown as string[])).toThrow(
      'non-empty array',
    )
  })

  it('rejects batches with empty items', () => {
    expect(() => validateEmbeddingsBatch(['ok', '  '])).toThrow('texts[1]')
  })
})

function audioWithHeader(ascii: string, offset = 0, byteLength = 64): ArrayBuffer {
  const buffer = new ArrayBuffer(byteLength)
  const view = new Uint8Array(buffer)
  for (let i = 0; i < ascii.length; i++) {
    view[offset + i] = ascii.charCodeAt(i)
  }
  return buffer
}

describe('STT audio contract', () => {
  it('exposes the model sample-rate constants', () => {
    expect(STT_SAMPLE_RATE).toBe(16000)
    expect(STT_MIN_SAMPLE_RATE).toBe(8000)
    expect(STT_MAX_SAMPLE_RATE).toBe(48000)
  })

  it('accepts a raw Float32 buffer', () => {
    const buffer = new Float32Array([0, 0.5, -0.5, 1]).buffer
    expect(validateSTTAudio(buffer, 'STT audio')).toBe(buffer)
  })

  it('rejects byte lengths that are not a multiple of 4', () => {
    expect(() => validateSTTAudio(new ArrayBuffer(6), 'STT audio')).toThrow(
      'multiple of 4',
    )
  })

  it('rejects recognizable encoded containers', () => {
    expect(() => validateSTTAudio(audioWithHeader('RIFF'), 'STT audio')).toThrow(
      'WAV (RIFF)',
    )
    expect(() => validateSTTAudio(audioWithHeader('ID3'), 'STT audio')).toThrow(
      'MP3 (ID3)',
    )
    expect(() => validateSTTAudio(audioWithHeader('fLaC'), 'STT audio')).toThrow('FLAC')
    expect(() => validateSTTAudio(audioWithHeader('OggS'), 'STT audio')).toThrow('Ogg')
    expect(() => validateSTTAudio(audioWithHeader('FORM'), 'STT audio')).toThrow('AIFF')
    expect(() => validateSTTAudio(audioWithHeader('caff'), 'STT audio')).toThrow('CAF')
    expect(() => validateSTTAudio(audioWithHeader('ftyp', 4), 'STT audio')).toThrow(
      'MP4/M4A',
    )
  })

  it('does not misdetect MP3 frame sync in raw sample data', () => {
    const buffer = new ArrayBuffer(8)
    new Uint8Array(buffer).set([0xff, 0xfb, 0x90, 0x00, 0, 0, 0, 0])
    expect(validateSTTAudio(buffer, 'STT audio')).toBe(buffer)
  })

  it('rejects empty and non-ArrayBuffer audio', () => {
    expect(() => validateSTTAudio(new ArrayBuffer(0), 'STT audio')).toThrow(
      'must not be empty',
    )
    expect(() => validateSTTAudio('audio', 'STT audio')).toThrow('must be an ArrayBuffer')
  })

  it('accepts transcribe options within the contract', () => {
    expect(validateSTTTranscribeOptions(undefined)).toBeUndefined()
    expect(validateSTTTranscribeOptions({})).toEqual({})
    expect(validateSTTTranscribeOptions({ sampleRate: 24000 })).toEqual({
      sampleRate: 24000,
    })
    expect(validateSTTTranscribeOptions({ language: 'Spanish' })).toEqual({
      language: 'Spanish',
    })
  })

  it('rejects out-of-range or non-integer sample rates', () => {
    expect(() =>
      validateSTTTranscribeOptions({ sampleRate: STT_MIN_SAMPLE_RATE - 1 }),
    ).toThrow('between 8000 and 48000')
    expect(() =>
      validateSTTTranscribeOptions({ sampleRate: STT_MAX_SAMPLE_RATE + 1 }),
    ).toThrow('between 8000 and 48000')
    expect(() => validateSTTTranscribeOptions({ sampleRate: Number.NaN })).toThrow(
      'integer',
    )
    expect(() => validateSTTTranscribeOptions({ sampleRate: 44100.5 })).toThrow('integer')
  })

  it('rejects empty languages', () => {
    expect(() => validateSTTTranscribeOptions({ language: '  ' })).toThrow(
      'non-empty string',
    )
    expect(() => validateSTTListeningOptions({ language: '' })).toThrow(
      'non-empty string',
    )
  })

  it('accepts listening options', () => {
    expect(validateSTTListeningOptions(undefined)).toBeUndefined()
    expect(validateSTTListeningOptions({ language: 'French' })).toEqual({
      language: 'French',
    })
  })
})
