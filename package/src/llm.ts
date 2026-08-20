import { NitroModules } from 'react-native-nitro-modules'
import type { JsonObject } from './json'
import {
  assertBoolean,
  assertNonEmptyString,
  createSafeCallback,
  mapStreamEventEnvelope,
  safeJsonParse,
  validateLLMLoadOptions,
  validateTokenCountRequest,
  validateTurnContextOptions,
  validateTurnRequest,
} from './runtime'
import type {
  LLMGenerationOutcome,
  LLMLoadOptions,
  LLM as LLMSpec,
  StreamEvent,
  StreamEventEnvelope,
} from './specs/LLM.nitro'
import type {
  LLMContext,
  LLMContextOptions,
  LLMTokenCountRequest,
  LLMTurnOutcome,
  LLMTurnRequest,
} from './turn'
import {
  fromWireOutcome,
  toWireContextOptions,
  toWireRequest,
  toWireTokenCountRequest,
} from './turn'

export type EventCallback = (event: StreamEvent) => void

let instance: LLMSpec | null = null

export type Message = {
  role: 'user' | 'assistant' | 'system' | 'tool'
  content: string
}

export type ToolCallInfo = {
  name: string
  arguments: JsonObject
}

export type ToolCallUpdate = {
  toolCall: ToolCallInfo
  allToolCalls: ToolCallInfo[]
}

function getInstance(): LLMSpec {
  if (!instance) {
    instance = NitroModules.createHybridObject<LLMSpec>('LLM')
  }
  if (!instance) {
    throw new Error('Failed to initialize the LLM Nitro module.')
  }
  return instance
}

/**
 * LLM text generation using MLX on Apple Silicon.
 *
 * @example
 * ```ts
 * import { LLM } from 'react-native-nitro-mlx'
 *
 * // Load a model
 * await LLM.load('mlx-community/Qwen3-0.6B-4bit', progress => {
 *   console.log(`Loading: ${(progress * 100).toFixed(0)}%`)
 * })
 *
 * // Stream a response
 * const outcome = await LLM.stream('Hello!', token => {
 *   process.stdout.write(token)
 * })
 *
 * console.log(`${outcome.stats.tokensPerSecond} tokens/sec`)
 * ```
 */
export const LLM = {
  /**
   * Load a model into memory. Downloads the model from HuggingFace if not already cached.
   * @param modelId - HuggingFace model ID (e.g., 'mlx-community/Qwen3-0.6B-4bit')
   * @param options - Callback invoked with loading progress (0-1)
   */
  load(modelId: string, options?: LLMLoadOptions): Promise<void> {
    return getInstance().load(
      assertNonEmptyString(modelId, 'LLM modelId'),
      validateLLMLoadOptions(options),
    )
  },

  /**
   * Generate a complete response for a prompt. Blocks until generation is complete.
   * For streaming responses, use `stream()` instead.
   * @param prompt - The input text to generate a response for
   * @returns The normalized terminal outcome
   */
  generate(prompt: string): Promise<LLMGenerationOutcome> {
    return getInstance().generate(assertNonEmptyString(prompt, 'LLM prompt'))
  },

  /**
   * Stream a response token by token with optional tool calling support.
   * Tools must be provided when loading the model via `load()` options.
   * Tools are automatically executed when the model calls them.
   * @param prompt - The input text to generate a response for
   * @param onToken - Callback invoked for each generated token
   * @param onToolCall - Optional callback invoked when a tool is called.
   *   Receives the current tool call and an accumulated array of all tool calls so far.
   * @returns The normalized terminal outcome
   */
  stream(
    prompt: string,
    onToken: (token: string) => void,
    onToolCall?: (update: ToolCallUpdate) => void,
  ): Promise<LLMGenerationOutcome> {
    const accumulatedToolCalls: ToolCallInfo[] = []
    const safeOnToken = createSafeCallback('LLM.stream onToken', onToken)
    const safeOnToolCall = createSafeCallback('LLM.stream onToolCall', onToolCall)

    return getInstance().stream(
      assertNonEmptyString(prompt, 'LLM prompt'),
      safeOnToken ?? (() => {}),
      safeOnToolCall
        ? (name: string, argsJson: string) => {
            const toolCall = {
              name,
              arguments: safeJsonParse<JsonObject>(argsJson, {}),
            }
            accumulatedToolCalls.push(toolCall)
            safeOnToolCall({
              toolCall,
              allToolCalls: [...accumulatedToolCalls],
            })
          }
        : undefined,
    )
  },

  /**
   * Stream with typed events for thinking blocks and tool calls.
   * Provides granular lifecycle events for UI updates.
   *
   * @param prompt - The input text
   * @param onEvent - Callback receiving typed StreamEvent objects
   * @returns Promise resolving to the normalized terminal outcome
   *
   * @example
   * ```ts
   * await LLM.streamWithEvents(prompt, (event) => {
   *   switch (event.type) {
   *     case 'token':
   *       appendToContent(event.token)
   *       break
   *     case 'thinking_start':
   *       showThinkingIndicator()
   *       break
   *     case 'thinking_chunk':
   *       appendToThinking(event.chunk)
   *       break
   *     case 'tool_call_start':
   *       showToolCallCard(event.name, event.arguments)
   *       break
   *     case 'generation_outcome':
   *       if (event.outcome.finishReason === 'failed') {
   *         showError(event.outcome.error)
   *       }
   *       break
   *   }
   * })
   * ```
   */
  streamWithEvents(
    prompt: string,
    onEvent: EventCallback,
  ): Promise<LLMGenerationOutcome> {
    const safeOnEvent = createSafeCallback('LLM.streamWithEvents onEvent', onEvent)

    return getInstance().streamWithEvents(
      assertNonEmptyString(prompt, 'LLM prompt'),
      (envelope: StreamEventEnvelope) => {
        const event = mapStreamEventEnvelope(envelope)
        if (event) {
          safeOnEvent?.(event)
        }
      },
    )
  },

  /**
   * Stop the current generation. Safe to call even if not generating.
   */
  stop(): void {
    getInstance().stop()
  },

  /**
   * Unload the current model and release memory.
   * Call this when you're done with the model to free up memory.
   */
  unload(): void {
    getInstance().unload()
  },

  /**
   * Get the message history if management is enabled.
   * @returns Array of messages in the history
   */
  getHistory(): Message[] {
    // SAFETY: the native history only ever stores the four chat roles, which
    // the wire type cannot express beyond `string`.
    return getInstance().getHistory() as Message[]
  },

  /**
   * Clear the message history.
   */
  clearHistory(): void {
    getInstance().clearHistory()
  },

  /**
   * Run one LLM Generation Turn. Tool Call Requests come back to the caller;
   * this package executes nothing. Branch your loop on toolCalls.length, not
   * on finishReason.
   */
  async runTurn(
    request: LLMTurnRequest,
    onEvent?: (event: StreamEvent) => void,
  ): Promise<LLMTurnOutcome> {
    const wireRequest = toWireRequest(request)
    validateTurnRequest(wireRequest)
    const safeOnEvent = createSafeCallback('LLM.runTurn onEvent', onEvent)
    const wireOutcome = await getInstance().runTurn(wireRequest, envelope => {
      if (!safeOnEvent) return
      const event = mapStreamEventEnvelope(envelope)
      if (event) safeOnEvent(event)
    })
    return fromWireOutcome(wireOutcome)
  },

  /** Create a Turn Context: retained instructions, transcript, and warm KV cache over the Resident Model. Release it when done. */
  async createContext(options: LLMContextOptions = {}): Promise<LLMContext> {
    const wireOptions = toWireContextOptions(options)
    validateTurnContextOptions(wireOptions)
    const id = await getInstance().createTurnContext(wireOptions)
    return {
      id,
      release: () => getInstance().releaseTurnContext(id),
    }
  },

  /** Release a Turn Context. Idempotent. */
  releaseContext(id: string): void {
    getInstance().releaseTurnContext(assertNonEmptyString(id, 'LLM contextId'))
  },

  /** Release every Turn Context. */
  releaseAllContexts(): void {
    getInstance().releaseAllTurnContexts()
  },

  /** IDs of all live Turn Contexts. */
  get contextIds(): string[] {
    return getInstance().turnContextIds
  },

  /** Count tokens for an assembled prompt with the loaded tokenizer and chat template. */
  countTokens(request: LLMTokenCountRequest): Promise<number> {
    const wireRequest = toWireTokenCountRequest(request)
    validateTokenCountRequest(wireRequest)
    return getInstance().countTokens(wireRequest)
  },

  /** Whether a model is currently loaded and ready for generation */
  get isLoaded(): boolean {
    return getInstance().isLoaded
  },

  /**
   * The id of the Resident Model, or null when nothing is loaded.
   * `load()` with this exact id performs no weight I/O.
   */
  get loadedModelId(): string | null {
    const llm = getInstance()
    return llm.isLoaded && llm.modelId !== '' ? llm.modelId : null
  },

  /** Whether text is currently being generated */
  get isGenerating(): boolean {
    return getInstance().isGenerating
  },

  /** The ID of the currently loaded model, or empty string if none */
  get modelId(): string {
    return getInstance().modelId
  },

  /** Enable debug logging to console */
  get debug(): boolean {
    return getInstance().debug
  },

  set debug(value: boolean) {
    getInstance().debug = assertBoolean(value, 'LLM.debug')
  },

  /**
   * System prompt used when loading the model.
   * Set this before calling `load()`. Changes require reloading the model.
   * @default "You are a helpful assistant."
   */
  get systemPrompt(): string {
    return getInstance().systemPrompt
  },

  set systemPrompt(value: string) {
    getInstance().systemPrompt = assertNonEmptyString(value, 'LLM systemPrompt')
  },
}
