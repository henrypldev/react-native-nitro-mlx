import type { EmbeddingsLoadOptions } from './specs/Embeddings.nitro'
import type {
  GenerationStats,
  LLMLoadOptions,
  StreamEvent,
  StreamEventEnvelope,
  ToolDefinition,
} from './specs/LLM.nitro'
import type { STTLoadOptions } from './specs/STT.nitro'
import type { TTSGenerateOptions, TTSLoadOptions } from './specs/TTS.nitro'

const ERROR_PREFIX = '[react-native-nitro-mlx]'
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

export function validateEmbeddingsLoadOptions(
  options?: EmbeddingsLoadOptions,
): EmbeddingsLoadOptions | undefined {
  return validateLoadOptions(options, 'Embeddings')
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
    if (!Number.isFinite(options.speed) || options.speed <= 0) {
      throw new RangeError(`${ERROR_PREFIX} TTS speed must be a positive finite number.`)
    }
  }

  return {
    ...options,
    onProgress: createSafeCallback('TTS.stream onProgress', options.onProgress),
  }
}

const EMPTY_STATS: GenerationStats = {
  tokenCount: 0,
  tokensPerSecond: 0,
  timeToFirstToken: 0,
  totalTime: 0,
  toolExecutionTime: 0,
}

/**
 * Expand the flat `StreamEventEnvelope` that crosses the bridge back into the
 * discriminated `StreamEvent` union that consumers switch on.
 *
 * The native side always populates the fields its `kind` implies, so the `??` fallbacks
 * are defensive only. `generation_end` falls back to zeroed stats rather than being
 * dropped, because swallowing the terminal event would strand UI state mid-generation.
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
    case 'generation_end':
      return {
        type: 'generation_end',
        content: envelope.content ?? '',
        stats: envelope.stats ?? EMPTY_STATS,
      }
    case 'generation_error':
      return {
        type: 'generation_error',
        error: envelope.error ?? '',
        stage: envelope.stage ?? '',
        stats: envelope.stats ?? EMPTY_STATS,
      }
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
