import type { AnyMap, HybridObject } from 'react-native-nitro-modules'

/**
 * Statistics from the last text generation.
 */
export interface GenerationStats {
  tokenCount: number
  tokensPerSecond: number
  timeToFirstToken: number
  totalTime: number
  toolExecutionTime: number
}

export type LLMGenerationFinishReason =
  | 'completed'
  | 'stopped'
  | 'superseded'
  | 'unloaded'
  | 'failed'

/**
 * Terminal outcome shared by every generation entry point.
 * A started turn always resolves with one of these; only pre-turn validation rejects.
 */
export interface LLMGenerationOutcome {
  /** User-visible content with thinking tags removed. */
  content: string
  /** Model thinking content, when the model emitted thinking tags. */
  thinking?: string
  stats: GenerationStats
  finishReason: LLMGenerationFinishReason
  /** Localized failure message when finishReason is `failed`. */
  error?: string
  /** Failure stage: 'prepare' | 'generate' | 'tool' | 'history'. */
  stage?: string
}

export interface GenerationStartEvent {
  type: 'generation_start'
  timestamp: number
}

export interface TokenEvent {
  type: 'token'
  token: string
}

export interface ThinkingStartEvent {
  type: 'thinking_start'
  timestamp: number
}

export interface ThinkingChunkEvent {
  type: 'thinking_chunk'
  chunk: string
}

export interface ThinkingEndEvent {
  type: 'thinking_end'
  content: string
  timestamp: number
}

export interface ToolCallStartEvent {
  type: 'tool_call_start'
  id: string
  name: string
  arguments: string
}

export interface ToolCallExecutingEvent {
  type: 'tool_call_executing'
  id: string
}

export interface ToolCallCompletedEvent {
  type: 'tool_call_completed'
  id: string
  result: string
}

export interface ToolCallFailedEvent {
  type: 'tool_call_failed'
  id: string
  error: string
}

export interface GenerationOutcomeEvent {
  type: 'generation_outcome'
  outcome: LLMGenerationOutcome
}

export type StreamEvent =
  | GenerationStartEvent
  | TokenEvent
  | ThinkingStartEvent
  | ThinkingChunkEvent
  | ThinkingEndEvent
  | ToolCallStartEvent
  | ToolCallExecutingEvent
  | ToolCallCompletedEvent
  | ToolCallFailedEvent
  | GenerationOutcomeEvent

/**
 * Discriminant for `StreamEventEnvelope`.
 *
 * Nitro cannot represent a discriminated union of structs (`TokenEvent | ThinkingEvent | ...`)
 * — an inline string literal on a struct field is ambiguous between a string and a union
 * enum. A *named* literal union compiles to a native enum, so the events cross the bridge
 * as one envelope struct discriminated by this, and `llm.ts` maps it back to `StreamEvent`.
 */
export type StreamEventKind =
  | 'generation_start'
  | 'token'
  | 'thinking_start'
  | 'thinking_chunk'
  | 'thinking_end'
  | 'tool_call_start'
  | 'tool_call_executing'
  | 'tool_call_completed'
  | 'tool_call_failed'
  | 'generation_outcome'

/**
 * Flat wire representation of a `StreamEvent`. Which fields are populated depends on
 * `kind`; consumers should use the mapped `StreamEvent` union from `llm.ts` instead.
 * @internal
 */
export interface StreamEventEnvelope {
  kind: StreamEventKind
  timestamp?: number
  token?: string
  chunk?: string
  content?: string
  id?: string
  name?: string
  arguments?: string
  result?: string
  error?: string
  outcome?: LLMGenerationOutcome
}

export interface LLMMessage {
  role: string
  content: string
}

/**
 * Controls low-level token generation behavior.
 */
export interface LLMGenerationConfig {
  /** Maximum number of tokens to generate */
  maxTokens?: number
  /** Sliding-window KV cache size. When set, old cache entries are rotated out */
  maxKVSize?: number
  /** KV cache quantization bits. Use 4 or 8 to reduce cache memory usage */
  kvBits?: number
  /** KV cache quantization group size */
  kvGroupSize?: number
  /** Token index at which KV cache quantization begins */
  quantizedKVStart?: number
  /** Sampling temperature. Set to 0 for greedy decoding */
  temperature?: number
  /** Top-p / nucleus sampling threshold */
  topP?: number
  /** Penalty applied to recently repeated tokens */
  repetitionPenalty?: number
  /** Number of recent tokens considered for repetition penalty */
  repetitionContextSize?: number
  /** Prompt prefill chunk size used for long-context batching */
  prefillStepSize?: number
}

/**
 * Controls history trimming when managed chat history is enabled.
 */
export interface LLMContextConfig {
  /**
   * Maximum prompt token count to preserve when rebuilding managed history.
   * Additional context passed during `load()` is treated as pinned and preserved.
   */
  maxContextTokens?: number
  /** Number of most-recent history messages to preserve during trimming */
  keepLastMessages?: number
}

/**
 * Parameter definition for a tool.
 */
export interface ToolParameter {
  name: string
  type: string
  description: string
  required: boolean
}

/**
 * Tool definition that can be called by the model.
 */
export interface ToolDefinition {
  name: string
  description: string
  parameters: ToolParameter[]
  handler: (args: AnyMap) => Promise<AnyMap>
}

export type LLMToolExecution = 'parallel' | 'sequential'

/** Options for loading a model.
 */
export interface LLMLoadOptions {
  /** Callback invoked with loading progress (0-1) */
  onProgress?: (progress: number) => void
  /** Additional context to provide to the model */
  additionalContext?: LLMMessage[]
  /** Whether to automatically manage message history */
  manageHistory?: boolean
  /** Tools available for the model to call */
  tools?: ToolDefinition[]
  /** Default generation parameters applied to future requests */
  generationConfig?: LLMGenerationConfig
  /** Number of generated chunks to batch before crossing the JS bridge */
  tokenBatchSize?: number
  /** Context trimming behavior for managed chat history */
  contextConfig?: LLMContextConfig
  /** How tool calls emitted by the same model pass are executed. @default 'parallel' */
  toolExecution?: LLMToolExecution
}

/**
 * Low-level LLM interface for text generation using MLX.
 * @internal Use the `LLM` export from `react-native-nitro-mlx` instead.
 */
export interface LLM extends HybridObject<{ ios: 'swift' }> {
  /**
   * Load a model into memory. Downloads from HuggingFace if not already cached.
   * @param modelId - HuggingFace model ID (e.g., 'mlx-community/Qwen3-0.6B-4bit')
   * @param options - Callback invoked with loading progress (0-1)
   */
  load(modelId: string, options?: LLMLoadOptions): Promise<void>

  /**
   * Generate a complete response for a prompt.
   * @param prompt - The input text to generate a response for
   * @returns The normalized terminal outcome
   */
  generate(prompt: string): Promise<LLMGenerationOutcome>

  /**
   * Stream a response token by token with optional tool calling support.
   * Tools are automatically executed when the model calls them.
   * @param prompt - The input text to generate a response for
   * @param onToken - Callback invoked for each generated token
   * @param onToolCall - Optional callback invoked when a tool is called (for UI feedback)
   * @returns The normalized terminal outcome
   */
  stream(
    prompt: string,
    onToken: (token: string) => void,
    onToolCall?: (toolName: string, args: string) => void,
  ): Promise<LLMGenerationOutcome>

  streamWithEvents(
    prompt: string,
    onEvent: (event: StreamEventEnvelope) => void,
  ): Promise<LLMGenerationOutcome>

  /**
   * Stop the current generation.
   */
  stop(): void

  /**
   * Unload the current model and release memory.
   */
  unload(): void

  /**
   * Get the message history if management is enabled.
   * @returns Array of messages in the history
   */
  getHistory(): LLMMessage[]

  /**
   * Clear the message history.
   */
  clearHistory(): void

  /** Whether a model is currently loaded */
  readonly isLoaded: boolean
  /** Whether text is currently being generated */
  readonly isGenerating: boolean
  /** The ID of the currently loaded model */
  readonly modelId: string

  /** Enable debug logging */
  debug: boolean
  /** System prompt used when loading the model */
  systemPrompt: string
}

/**
 * Supported parameter types for tool definitions.
 * Used for type safety in createTool().
 */
export type ToolParameterType = 'string' | 'number' | 'boolean' | 'array' | 'object'
