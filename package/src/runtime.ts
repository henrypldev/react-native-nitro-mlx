import { z } from 'zod'
import type { JsonValue } from './json'
import type {
  EmbeddingsEmbedOptions,
  EmbeddingsLoadOptions,
} from './specs/Embeddings.nitro'
import type {
  LLMGenerationConfig,
  LLMLoadOptions,
  LLMTokenCountRequest,
  LLMToolSchema,
  LLMTurnContextOptions,
  LLMTurnMessage,
  LLMTurnRequest,
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
// SAFETY: every supported RN runtime provides a global console, but the
// `globalThis` type cannot prove it; the optional access degrades to a no-op.
const runtimeConsole = (
  globalThis as { console?: { error?: (...args: unknown[]) => void } }
).console

const stringSchema = z.string()
const booleanSchema = z.boolean()
const arrayBufferSchema = z.instanceof(ArrayBuffer)
const functionSchema = z.instanceof(Function)

/**
 * Names the runtime representation of a value for error messages. Generic so
 * it stays callable from failed-guard branches where TS has narrowed to `never`.
 */
function describeType<T>(value: T): string {
  if (value === null) {
    return 'null'
  }
  if (value === undefined) {
    return 'undefined'
  }
  if (arrayBufferSchema.safeParse(value).success) {
    return 'ArrayBuffer'
  }
  return Object.prototype.toString.call(value).slice(8, -1).toLowerCase()
}

export function assertNonEmptyString(value: string, name: string): string {
  const parsed = stringSchema.safeParse(value)
  if (!parsed.success || parsed.data.trim().length === 0) {
    throw new TypeError(`${ERROR_PREFIX} ${name} must be a non-empty string.`)
  }
  return parsed.data
}

export function assertArrayBuffer(value: ArrayBuffer, name: string): ArrayBuffer {
  const parsed = arrayBufferSchema.safeParse(value)
  if (!parsed.success) {
    throw new TypeError(
      `${ERROR_PREFIX} ${name} must be an ArrayBuffer, received ${describeType(value)}.`,
    )
  }
  if (parsed.data.byteLength === 0) {
    throw new TypeError(`${ERROR_PREFIX} ${name} must not be empty.`)
  }
  return parsed.data
}

export function assertBoolean(value: boolean, name: string): boolean {
  if (!booleanSchema.safeParse(value).success) {
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
  if (!functionSchema.safeParse(callback).success) {
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

    if (!functionSchema.safeParse(tool.handler).success) {
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

export function validateSTTAudio(value: ArrayBuffer, name: string): ArrayBuffer {
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

function validateSTTLanguage(language: string | undefined): void {
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
    // SAFETY: the caller declares the payload contract via T and supplies a
    // fallback; a payload that violates T is the caller's documented risk.
    return JSON.parse(value) as T
  } catch {
    return fallback
  }
}

const turnMessageSchema = z.looseObject({
  role: z.enum(['system', 'user', 'assistant', 'tool']),
  content: z.string(),
  toolCallId: z.string().optional(),
  name: z.string().optional(),
  isError: z.boolean().optional(),
  toolCallsJson: z.string().optional(),
})

const roleProbeSchema = z.looseObject({ role: z.coerce.string() })

function turnMessageError(
  issues: z.core.$ZodIssue[],
  message: LLMTurnMessage,
  label: string,
): TypeError {
  const field = issues[0]?.path[0]
  if (field === 'content') {
    return new TypeError(`${ERROR_PREFIX} ${label}.content must be a string`)
  }
  if (field === undefined || field === 'role') {
    const probe = roleProbeSchema.safeParse(message)
    const role = probe.success ? probe.data.role : undefined
    return new TypeError(`${ERROR_PREFIX} ${label} has unknown role: ${String(role)}`)
  }
  return new TypeError(`${ERROR_PREFIX} ${label}.${String(field)} is invalid`)
}

export function validateTurnMessages(
  value: LLMTurnMessage[],
  name: string,
  options: { requireNonEmpty?: boolean },
): LLMTurnMessage[] {
  if (!Array.isArray(value)) {
    throw new TypeError(`${ERROR_PREFIX} ${name} must be an array`)
  }
  if (options.requireNonEmpty && value.length === 0) {
    throw new TypeError(`${ERROR_PREFIX} ${name} must not be empty`)
  }
  return value.map((message, index) => {
    const label = `${name}[${index}]`
    const parsed = turnMessageSchema.safeParse(message)
    if (!parsed.success) {
      throw turnMessageError(parsed.error.issues, message, label)
    }
    const m = parsed.data
    if (m.role === 'tool' && (m.toolCallId === undefined || m.toolCallId === '')) {
      throw new TypeError(
        `${ERROR_PREFIX} ${label} is a tool message and requires a non-empty toolCallId`,
      )
    }
    if (m.toolCallsJson !== undefined && m.role !== 'assistant') {
      throw new TypeError(
        `${ERROR_PREFIX} ${label} carries tool calls but only assistant messages may`,
      )
    }
    return message
  })
}

const objectSchemaRootSchema = z.looseObject({ type: z.literal('object') })

export function validateToolSchemas(
  value: LLMToolSchema[],
  name: string,
): LLMToolSchema[] {
  if (!Array.isArray(value)) {
    throw new TypeError(`${ERROR_PREFIX} ${name} must be an array`)
  }
  const seen = new Set<string>()
  return value.map((tool, index) => {
    const label = `${name}[${index}]`
    assertNonEmptyString(tool?.name, `${label}.name`)
    assertNonEmptyString(tool?.description, `${label}.description`)
    assertNonEmptyString(tool?.parameters, `${label}.parameters`)
    if (seen.has(tool.name)) {
      throw new TypeError(
        `${ERROR_PREFIX} ${name} contains a duplicate tool name: ${tool.name}`,
      )
    }
    seen.add(tool.name)
    let parsedParameters: JsonValue
    try {
      parsedParameters = JSON.parse(tool.parameters)
    } catch {
      throw new TypeError(`${ERROR_PREFIX} ${label}.parameters is not valid JSON`)
    }
    if (!objectSchemaRootSchema.safeParse(parsedParameters).success) {
      throw new TypeError(
        `${ERROR_PREFIX} ${label}.parameters root must be an object schema ("type": "object")`,
      )
    }
    return tool
  })
}

const generationTuningSchema = z.looseObject({
  seed: z
    .number()
    .refine(seed => Number.isSafeInteger(seed) && seed >= 0)
    .optional(),
  topK: z
    .number()
    .refine(topK => Number.isInteger(topK) && topK >= 0)
    .optional(),
  minP: z
    .number()
    .refine(minP => Number.isFinite(minP) && minP >= 0 && minP <= 1)
    .optional(),
})

function validateGenerationConfig(
  config: LLMGenerationConfig | undefined,
  name: string,
): void {
  if (config === undefined) {
    return
  }
  const parsed = generationTuningSchema.safeParse(config)
  if (parsed.success) {
    return
  }
  const field = parsed.error.issues[0]?.path[0]
  if (field === 'seed') {
    throw new TypeError(
      `${ERROR_PREFIX} ${name}.seed must be a non-negative safe integer.`,
    )
  }
  if (field === 'topK') {
    throw new TypeError(`${ERROR_PREFIX} ${name}.topK must be a non-negative integer.`)
  }
  if (field === 'minP') {
    throw new TypeError(`${ERROR_PREFIX} ${name}.minP must be between 0 and 1.`)
  }
  throw new TypeError(`${ERROR_PREFIX} ${name} must be an object.`)
}

export function validateTurnRequest(request: LLMTurnRequest): void {
  validateTurnMessages(request.messages, 'runTurn messages', { requireNonEmpty: true })
  const hasTools = Array.isArray(request.tools) && request.tools.length > 0
  if (request.responseSchema !== undefined) {
    assertNonEmptyString(request.responseSchema, 'runTurn responseSchema')
    validateToolSchemas(
      [{ name: '__schema__', description: '-', parameters: request.responseSchema }],
      'runTurn responseSchema',
    )
    if (hasTools) {
      throw new TypeError(
        `${ERROR_PREFIX} runTurn responseSchema is exclusive with tools`,
      )
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
  if (request.tools !== undefined && hasTools) {
    validateToolSchemas(request.tools, 'runTurn tools')
  }
  if (request.history !== undefined) {
    validateTurnMessages(request.history, 'runTurn history', {})
  }
  validateGenerationConfig(request.generationConfig, 'runTurn generationConfig')
}

export function validateTurnContextOptions(options: LLMTurnContextOptions): void {
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

export function validateTokenCountRequest(request: LLMTokenCountRequest): void {
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
