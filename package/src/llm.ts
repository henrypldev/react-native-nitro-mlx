import { NitroModules } from 'react-native-nitro-modules'
import {
  assertBoolean,
  assertNonEmptyString,
  createSafeCallback,
  mapStreamEventEnvelope,
  safeJsonParse,
  validateLLMLoadOptions,
} from './runtime'
import type {
  GenerationStats,
  LLMLoadOptions,
  LLM as LLMSpec,
  StreamEvent,
  StreamEventEnvelope,
} from './specs/LLM.nitro'

export type EventCallback = (event: StreamEvent) => void

let instance: LLMSpec | null = null

export type Message = {
  role: 'user' | 'assistant' | 'system' | 'tool'
  content: string
}

export type ToolCallInfo = {
  name: string
  arguments: Record<string, unknown>
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
 * await LLM.stream('Hello!', token => {
 *   process.stdout.write(token)
 * })
 *
 * // Get generation stats
 * const stats = LLM.getLastGenerationStats()
 * console.log(`${stats.tokensPerSecond} tokens/sec`)
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
   * @returns The complete generated text
   */
  generate(prompt: string): Promise<string> {
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
   * @returns The complete generated text
   */
  stream(
    prompt: string,
    onToken: (token: string) => void,
    onToolCall?: (update: ToolCallUpdate) => void,
  ): Promise<string> {
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
              arguments: safeJsonParse<Record<string, unknown>>(argsJson, {}),
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
   * @returns Promise resolving to final content string (thinking content stripped)
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
   *   }
   * })
   * ```
   */
  streamWithEvents(prompt: string, onEvent: EventCallback): Promise<string> {
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
   * Get statistics from the last generation.
   * @returns Statistics including token count, tokens/sec (excluding tool execution), TTFT, total time, and tool execution time
   */
  getLastGenerationStats(): GenerationStats {
    return getInstance().getLastGenerationStats()
  },

  /**
   * Get the message history if management is enabled.
   * @returns Array of messages in the history
   */
  getHistory(): Message[] {
    return getInstance().getHistory() as Message[]
  },

  /**
   * Clear the message history.
   */
  clearHistory(): void {
    getInstance().clearHistory()
  },

  /** Whether a model is currently loaded and ready for generation */
  get isLoaded(): boolean {
    return getInstance().isLoaded
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
