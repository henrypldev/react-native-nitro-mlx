import type {
  EmbeddingsEmbedOptions,
  EmbeddingsLoadOptions,
} from './specs/Embeddings.nitro'
import type {
  LLMLoadOptions,
  StreamEvent,
  StreamEventEnvelope,
  ToolDefinition,
} from './specs/LLM.nitro'
import type {
  STTListeningOptions,
  STTLoadOptions,
  STTTranscribeOptions,
} from './specs/STT.nitro'
import type { TTSGenerateOptions, TTSLoadOptions } from './specs/TTS.nitro'

const ERROR_PREFIX = '[react-native-nitro-mlx]'
export const TTS_MIN_SPEED = 0.5
export const TTS_MAX_SPEED = 2
const runtimeConsole = (
  globalThis as { console?: { error?: (...args: unknown[]) => void } }
).console

function describeType(value: unknown): string {
  if (value === null) {
    return 'null'
  }
  if (value === undefined) {
    return 'undefined'
  }
  if (value instanceof ArrayBuffer) {
    return 'ArrayBuffer'
  }
  return typeof value
}

export function assertNonEmptyString(value: unknown, name: string): string {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new TypeError(`${ERROR_PREFIX} ${name} must be a non-empty string.`)
  }
  return value
}

export function assertArrayBuffer(value: unknown, name: string): ArrayBuffer {
  if (!(value instanceof ArrayBuffer)) {
    throw new TypeError(
      `${ERROR_PREFIX} ${name} must be an ArrayBuffer, received ${describeType(value)}.`,
    )
  }
  if (value.byteLength === 0) {
    throw new TypeError(`${ERROR_PREFIX} ${name} must not be empty.`)
  }
  return value
}

export function assertBoolean(value: unknown, name: string): boolean {
  if (typeof value !== 'boolean') {
    throw new TypeError(`${ERROR_PREFIX} ${name} must be a boolean.`)
  }
  return value
}

export function createSafeCallback<TArgs extends unknown[]>(
  name: string,
  callback?: ((...args: TArgs) => void) | null,
): ((...args: TArgs) => void) | undefined {
  if (callback == null) {
    return undefined
  }
  if (typeof callback !== 'function') {
    throw new TypeError(
      `${ERROR_PREFIX} ${name} must be a function, received ${describeType(callback)}.`,
    )
  }

  return (...args: TArgs) => {
    try {
      callback(...args)
    } catch (error) {
      runtimeConsole?.error?.(`${ERROR_PREFIX} ${name} callback threw.`, error)
    }
  }
}

function validateToolDefinitions(tools: ToolDefinition[]): ToolDefinition[] {
  const seenNames = new Set<string>()

  return tools.map((tool, index) => {
    const name = assertNonEmptyString(tool?.name, `tools[${index}].name`)
    if (seenNames.has(name)) {
      throw new TypeError(
        `${ERROR_PREFIX} tools must have unique names. Duplicate: '${name}'.`,
      )
    }
    seenNames.add(name)

    if (typeof tool.handler !== 'function') {
      throw new TypeError(`${ERROR_PREFIX} tools[${index}].handler must be a function.`)
    }

    return tool
  })
}

function validateLoadOptions<
  T extends { onProgress?: ((progress: number) => void) | null },
>(options: T | undefined, name: string): T | undefined {
  if (!options) {
    return undefined
  }

  return {
    ...options,
    onProgress: createSafeCallback(`${name}.load onProgress`, options.onProgress),
  }
}

export function validateLLMLoadOptions(
  options?: LLMLoadOptions,
): LLMLoadOptions | undefined {
  const validated = validateLoadOptions(options, 'LLM')
  if (!validated) {
    return undefined
  }

  if (
    validated.toolExecution !== undefined &&
    validated.toolExecution !== 'parallel' &&
    validated.toolExecution !== 'sequential'
  ) {
    throw new TypeError(
      `${ERROR_PREFIX} LLM toolExecution must be 'parallel' or 'sequential'.`,
    )
  }

  validateGenerationConfig(validated.generationConfig, 'LLM load generationConfig')

  return {
    ...validated,
    tools: validated.tools ? validateToolDefinitions(validated.tools) : validated.tools,
  }
}

export function validateModelDownloadCallback(
  callback?: ((progress: number) => void) | null,
): ((progress: number) => void) | undefined {
  return createSafeCallback('ModelManager.download onProgress', callback)
}

export function validateSTTLoadOptions(
  options?: STTLoadOptions,
): STTLoadOptions | undefined {
  return validateLoadOptions(options, 'STT')
}

/** Mirrors `STTAudioContract.modelSampleRate` on the native side. */
export const STT_SAMPLE_RATE = 16000
export const STT_MIN_SAMPLE_RATE = 8000
export const STT_MAX_SAMPLE_RATE = 48000

/**
 * Mirrors `STTAudioContract.signatures` on the native side. MP3 frame sync
 * (0xFF 0xEx) is deliberately absent — those bytes occur in raw Float32 data.
 */
const STT_ENCODED_SIGNATURES: ReadonlyArray<{
  magic: string
  offset: number
  format: string
}> = [
  { magic: 'RIFF', offset: 0, format: 'WAV (RIFF)' },
  { magic: 'ID3', offset: 0, format: 'MP3 (ID3)' },
  { magic: 'fLaC', offset: 0, format: 'FLAC' },
  { magic: 'OggS', offset: 0, format: 'Ogg' },
  { magic: 'FORM', offset: 0, format: 'AIFF (FORM)' },
  { magic: 'caff', offset: 0, format: 'CAF' },
  { magic: 'ftyp', offset: 4, format: 'MP4/M4A' },
]

function detectEncodedAudioFormat(buffer: ArrayBuffer): string | null {
  const view = new Uint8Array(buffer, 0, Math.min(buffer.byteLength, 12))
  for (const { magic, offset, format } of STT_ENCODED_SIGNATURES) {
    if (view.length < offset + magic.length) {
      continue
    }
    let matches = true
    for (let i = 0; i < magic.length; i++) {
      if (view[offset + i] !== magic.charCodeAt(i)) {
        matches = false
        break
      }
    }
    if (matches) {
      return format
    }
  }
  return null
}

export function validateSTTAudio(value: unknown, name: string): ArrayBuffer {
  const buffer = assertArrayBuffer(value, name)
  if (buffer.byteLength % 4 !== 0) {
    throw new TypeError(
      `${ERROR_PREFIX} ${name} must be raw native-endian mono Float32 PCM; byte length ${buffer.byteLength} is not a multiple of 4.`,
    )
  }
  const format = detectEncodedAudioFormat(buffer)
  if (format) {
    throw new TypeError(
      `${ERROR_PREFIX} ${name} looks like an encoded ${format} container. Decode it to raw mono Float32 PCM before transcribing.`,
    )
  }
  return buffer
}

function validateSTTLanguage(language: unknown): void {
  if (language !== undefined) {
    assertNonEmptyString(language, 'STT language')
  }
}

export function validateSTTTranscribeOptions(
  options?: STTTranscribeOptions,
): STTTranscribeOptions | undefined {
  if (!options) {
    return undefined
  }
  if (options.sampleRate !== undefined) {
    if (!Number.isInteger(options.sampleRate)) {
      throw new TypeError(`${ERROR_PREFIX} STT sampleRate must be an integer in Hz.`)
    }
    if (
      options.sampleRate < STT_MIN_SAMPLE_RATE ||
      options.sampleRate > STT_MAX_SAMPLE_RATE
    ) {
      throw new RangeError(
        `${ERROR_PREFIX} STT sampleRate must be between ${STT_MIN_SAMPLE_RATE} and ${STT_MAX_SAMPLE_RATE} Hz.`,
      )
    }
  }
  validateSTTLanguage(options.language)
  return options
}

export function validateSTTListeningOptions(
  options?: STTListeningOptions,
): STTListeningOptions | undefined {
  if (!options) {
    return undefined
  }
  validateSTTLanguage(options.language)
  return options
}

export function validateEmbeddingsLoadOptions(
  options?: EmbeddingsLoadOptions,
): EmbeddingsLoadOptions | undefined {
  return validateLoadOptions(options, 'Embeddings')
}

/** Mirrors `EmbeddingsBatchPlanner.maxBatchSize` on the native side. */
export const EMBEDDINGS_MAX_BATCH_SIZE = 64

export function validateEmbeddingsEmbedOptions(
  options?: EmbeddingsEmbedOptions,
): EmbeddingsEmbedOptions | undefined {
  if (!options) {
    return undefined
  }
  if (options.truncate !== undefined) {
    assertBoolean(options.truncate, 'Embeddings truncate')
  }
  return options
}

export function validateEmbeddingsBatch(texts: string[]): string[] {
  if (!Array.isArray(texts) || texts.length === 0) {
    throw new TypeError(
      `${ERROR_PREFIX} Embeddings.embedBatch requires a non-empty array.`,
    )
  }
  if (texts.length > EMBEDDINGS_MAX_BATCH_SIZE) {
    throw new RangeError(
      `${ERROR_PREFIX} Embeddings.embedBatch accepts at most ${EMBEDDINGS_MAX_BATCH_SIZE} texts per call, received ${texts.length}. Split the batch.`,
    )
  }
  return texts.map((t, i) => assertNonEmptyString(t, `Embeddings texts[${i}]`))
}

export function validateTTSLoadOptions(
  options?: TTSLoadOptions,
): TTSLoadOptions | undefined {
  return validateLoadOptions(options, 'TTS')
}

export function validateTTSGenerateOptions(
  options?: TTSGenerateOptions,
): TTSGenerateOptions | undefined {
  if (!options) {
    return undefined
  }

  if (options.voice !== undefined) {
    assertNonEmptyString(options.voice, 'TTS voice')
  }

  if (options.speed !== undefined) {
    if (
      !Number.isFinite(options.speed) ||
      options.speed < TTS_MIN_SPEED ||
      options.speed > TTS_MAX_SPEED
    ) {
      throw new RangeError(
        `${ERROR_PREFIX} TTS speed must be between ${TTS_MIN_SPEED} and ${TTS_MAX_SPEED}.`,
      )
    }
  }

  return {
    ...options,
    onProgress: createSafeCallback('TTS.stream onProgress', options.onProgress),
  }
}

/**
 * Expand the flat `StreamEventEnvelope` that crosses the bridge back into the
 * discriminated `StreamEvent` union that consumers switch on.
 *
 * The native side always populates the fields its `kind` implies, so the `??` fallbacks
 * are defensive only. A malformed terminal envelope is dropped because manufacturing
 * an outcome would hide a native transport defect.
 */
export function mapStreamEventEnvelope(
  envelope: StreamEventEnvelope,
): StreamEvent | null {
  switch (envelope.kind) {
    case 'generation_start':
      return { type: 'generation_start', timestamp: envelope.timestamp ?? 0 }
    case 'token':
      return { type: 'token', token: envelope.token ?? '' }
    case 'thinking_start':
      return { type: 'thinking_start', timestamp: envelope.timestamp ?? 0 }
    case 'thinking_chunk':
      return { type: 'thinking_chunk', chunk: envelope.chunk ?? '' }
    case 'thinking_end':
      return {
        type: 'thinking_end',
        content: envelope.content ?? '',
        timestamp: envelope.timestamp ?? 0,
      }
    case 'tool_call_start':
      return {
        type: 'tool_call_start',
        id: envelope.id ?? '',
        name: envelope.name ?? '',
        arguments: envelope.arguments ?? '',
      }
    case 'tool_call_executing':
      return { type: 'tool_call_executing', id: envelope.id ?? '' }
    case 'tool_call_completed':
      return {
        type: 'tool_call_completed',
        id: envelope.id ?? '',
        result: envelope.result ?? '',
      }
    case 'tool_call_failed':
      return {
        type: 'tool_call_failed',
        id: envelope.id ?? '',
        error: envelope.error ?? '',
      }
    case 'generation_outcome':
      return envelope.outcome
        ? { type: 'generation_outcome', outcome: envelope.outcome }
        : null
    default:
      return null
  }
}

export function safeJsonParse<T>(value: string, fallback: T): T {
  try {
    return JSON.parse(value) as T
  } catch {
    return fallback
  }
}

const TURN_ROLES = new Set(['system', 'user', 'assistant', 'tool'])

export interface TurnMessageLike {
  role: string
  content: string
  toolCallId?: string
  name?: string
  isError?: boolean
  toolCallsJson?: string
}

export function validateTurnMessages(
  value: unknown,
  name: string,
  options: { requireNonEmpty?: boolean },
): TurnMessageLike[] {
  if (!Array.isArray(value)) {
    throw new TypeError(`${ERROR_PREFIX} ${name} must be an array`)
  }
  if (options.requireNonEmpty && value.length === 0) {
    throw new TypeError(`${ERROR_PREFIX} ${name} must not be empty`)
  }
  return value.map((message, index) => {
    const label = `${name}[${index}]`
    const role = (message as TurnMessageLike)?.role
    if (typeof role !== 'string' || !TURN_ROLES.has(role)) {
      throw new TypeError(`${ERROR_PREFIX} ${label} has unknown role: ${String(role)}`)
    }
    if (typeof (message as TurnMessageLike).content !== 'string') {
      throw new TypeError(`${ERROR_PREFIX} ${label}.content must be a string`)
    }
    const m = message as TurnMessageLike
    if (role === 'tool' && (typeof m.toolCallId !== 'string' || m.toolCallId === '')) {
      throw new TypeError(
        `${ERROR_PREFIX} ${label} is a tool message and requires a non-empty toolCallId`,
      )
    }
    if (m.toolCallsJson !== undefined && role !== 'assistant') {
      throw new TypeError(
        `${ERROR_PREFIX} ${label} carries tool calls but only assistant messages may`,
      )
    }
    return m
  })
}

export interface ToolSchemaLike {
  name: string
  description: string
  parameters: string
}

export function validateToolSchemas(value: unknown, name: string): ToolSchemaLike[] {
  if (!Array.isArray(value)) {
    throw new TypeError(`${ERROR_PREFIX} ${name} must be an array`)
  }
  const seen = new Set<string>()
  return value.map((tool, index) => {
    const label = `${name}[${index}]`
    const t = tool as ToolSchemaLike
    assertNonEmptyString(t?.name, `${label}.name`)
    assertNonEmptyString(t?.description, `${label}.description`)
    assertNonEmptyString(t?.parameters, `${label}.parameters`)
    if (seen.has(t.name)) {
      throw new TypeError(`${ERROR_PREFIX} ${name} contains a duplicate tool name: ${t.name}`)
    }
    seen.add(t.name)
    let parsed: unknown
    try {
      parsed = JSON.parse(t.parameters)
    } catch {
      throw new TypeError(`${ERROR_PREFIX} ${label}.parameters is not valid JSON`)
    }
    if (
      typeof parsed !== 'object' ||
      parsed === null ||
      Array.isArray(parsed) ||
      (parsed as { type?: unknown }).type !== 'object'
    ) {
      throw new TypeError(
        `${ERROR_PREFIX} ${label}.parameters root must be an object schema ("type": "object")`,
      )
    }
    return t
  })
}

export interface GenerationConfigLike {
  seed?: unknown
  topK?: unknown
  minP?: unknown
}

function validateGenerationConfig(config: unknown, name: string): void {
  if (config === undefined) {
    return
  }
  if (typeof config !== 'object' || config === null || Array.isArray(config)) {
    throw new TypeError(`${ERROR_PREFIX} ${name} must be an object.`)
  }
  const { seed, topK, minP } = config as GenerationConfigLike
  if (seed !== undefined) {
    if (typeof seed !== 'number' || !Number.isSafeInteger(seed) || seed < 0) {
      throw new TypeError(
        `${ERROR_PREFIX} ${name}.seed must be a non-negative safe integer.`,
      )
    }
  }
  if (topK !== undefined) {
    if (typeof topK !== 'number' || !Number.isInteger(topK) || topK < 0) {
      throw new TypeError(`${ERROR_PREFIX} ${name}.topK must be a non-negative integer.`)
    }
  }
  if (minP !== undefined) {
    if (typeof minP !== 'number' || !Number.isFinite(minP) || minP < 0 || minP > 1) {
      throw new TypeError(`${ERROR_PREFIX} ${name}.minP must be between 0 and 1.`)
    }
  }
}

export interface TurnRequestLike {
  messages: unknown
  contextId?: string
  instructions?: string
  history?: unknown
  tools?: unknown
  generationConfig?: unknown
  responseSchema?: string
  tokenBatchSize?: number
}

export function validateTurnRequest(request: TurnRequestLike): void {
  validateTurnMessages(request.messages, 'runTurn messages', { requireNonEmpty: true })
  const hasTools = Array.isArray(request.tools) && request.tools.length > 0
  if (request.responseSchema !== undefined) {
    assertNonEmptyString(request.responseSchema, 'runTurn responseSchema')
    validateToolSchemas(
      [{ name: '__schema__', description: '-', parameters: request.responseSchema }],
      'runTurn responseSchema',
    )
    if (hasTools) {
      throw new TypeError(`${ERROR_PREFIX} runTurn responseSchema is exclusive with tools`)
    }
  }
  if (request.contextId !== undefined) {
    assertNonEmptyString(request.contextId, 'runTurn contextId')
    if (
      request.instructions !== undefined ||
      request.history !== undefined ||
      request.generationConfig !== undefined ||
      hasTools
    ) {
      throw new TypeError(
        `${ERROR_PREFIX} runTurn instructions, history, tools, and generationConfig are cold-turn fields; remove them or remove contextId`,
      )
    }
  }
  if (hasTools) {
    validateToolSchemas(request.tools, 'runTurn tools')
  }
  if (request.history !== undefined) {
    validateTurnMessages(request.history, 'runTurn history', {})
  }
  validateGenerationConfig(request.generationConfig, 'runTurn generationConfig')
}

export function validateTurnContextOptions(options: {
  instructions?: string
  history?: unknown
  tools?: unknown
  generationConfig?: unknown
}): void {
  if (options.instructions !== undefined) {
    assertNonEmptyString(options.instructions, 'createContext instructions')
  }
  if (options.history !== undefined) {
    validateTurnMessages(options.history, 'createContext history', {})
  }
  if (options.tools !== undefined) {
    validateToolSchemas(options.tools, 'createContext tools')
  }
  validateGenerationConfig(options.generationConfig, 'createContext generationConfig')
}

export function validateTokenCountRequest(request: {
  contextId?: string
  instructions?: string
  history?: unknown
  tools?: unknown
  messages?: unknown
}): void {
  if (request.contextId !== undefined) {
    assertNonEmptyString(request.contextId, 'countTokens contextId')
  }
  if (request.history !== undefined) {
    validateTurnMessages(request.history, 'countTokens history', {})
  }
  if (request.messages !== undefined) {
    validateTurnMessages(request.messages, 'countTokens messages', {})
  }
  if (request.tools !== undefined) {
    validateToolSchemas(request.tools, 'countTokens tools')
  }
}
