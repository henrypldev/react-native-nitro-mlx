import { safeJsonParse } from './runtime'
import type {
  GenerationStats,
  LLMGenerationConfig,
  LLMToolCallWire,
  LLMTurnFinishReason,
  LLMTurnMessage,
  LLMTurnUsage,
  LLMTurnContextOptions as WireContextOptions,
  LLMTokenCountRequest as WireTokenCountRequest,
  LLMTurnOutcome as WireTurnOutcome,
  LLMTurnRequest as WireTurnRequest,
} from './specs/LLM.nitro'

export interface LLMToolCall {
  id: string
  name: string
  /** Parsed arguments. The native parser produced these; malformed output never reaches here. */
  arguments: Record<string, unknown>
}

export type LLMMessage =
  | { role: 'system' | 'user'; content: string }
  | { role: 'assistant'; content: string; toolCalls?: LLMToolCall[] }
  | {
      role: 'tool'
      toolCallId: string
      content: string
      /** Tool name, when known. Not consumed by every chat template. */
      name?: string
      /** True when the tool failed. The model must see failures to recover. */
      isError?: boolean
    }

export interface ToolSchema {
  name: string
  description: string
  /** Serialized JSON Schema. Root must be an object schema. */
  parameters: string
}

export interface LLMContextOptions {
  instructions?: string
  history?: LLMMessage[]
  tools?: ToolSchema[]
  generationConfig?: LLMGenerationConfig
}

export interface LLMContext {
  readonly id: string
  release(): void
}

export interface LLMTurnRequest {
  messages: LLMMessage[]
  contextId?: string
  instructions?: string
  history?: LLMMessage[]
  tools?: ToolSchema[]
  generationConfig?: LLMGenerationConfig
  responseSchema?: string
  tokenBatchSize?: number
}

export interface LLMTurnOutcome {
  finishReason: LLMTurnFinishReason
  rawFinishReason?: string
  content: string
  thinking?: string
  /** Populated when finishReason is 'tool_calls'. Loops should branch on toolCalls.length. */
  toolCalls: LLMToolCall[]
  /**
   * Best-effort on cancelled turns: a turn cancelled before its final
   * accounting arrives may report zero counts. The native layer folds in
   * counts when it can, but a cancellation can race ahead of that step.
   */
  usage: LLMTurnUsage
  stats: GenerationStats
  error?: string
  stage?: string
}

export interface LLMTokenCountRequest {
  contextId?: string
  instructions?: string
  history?: LLMMessage[]
  tools?: ToolSchema[]
  messages?: LLMMessage[]
}

/** @internal exported for tests */
export function toWireMessage(message: LLMMessage): LLMTurnMessage {
  switch (message.role) {
    case 'assistant':
      return {
        role: 'assistant',
        content: message.content,
        toolCallId: undefined,
        name: undefined,
        isError: undefined,
        toolCallsJson:
          message.toolCalls !== undefined ? JSON.stringify(message.toolCalls) : undefined,
      }
    case 'tool':
      return {
        role: 'tool',
        content: message.content,
        toolCallId: message.toolCallId,
        name: message.name,
        isError: message.isError,
        toolCallsJson: undefined,
      }
    default:
      return {
        role: message.role,
        content: message.content,
        toolCallId: undefined,
        name: undefined,
        isError: undefined,
        toolCallsJson: undefined,
      }
  }
}

function fromWireToolCall(call: LLMToolCallWire): LLMToolCall {
  return {
    id: call.id,
    name: call.name,
    arguments: safeJsonParse<Record<string, unknown>>(call.argumentsJson, {}),
  }
}

/** @internal exported for tests */
export function fromWireOutcome(outcome: WireTurnOutcome): LLMTurnOutcome {
  return {
    finishReason: outcome.finishReason,
    rawFinishReason: outcome.rawFinishReason,
    content: outcome.content,
    thinking: outcome.thinking,
    toolCalls: outcome.toolCalls.map(fromWireToolCall),
    usage: outcome.usage,
    stats: outcome.stats,
    error: outcome.error,
    stage: outcome.stage,
  }
}

/**
 * @internal `LLMToolSchema` (wire) and `ToolSchema` (public) are structurally
 * identical, so `tools` passes through unmapped below.
 */
export function toWireRequest(request: LLMTurnRequest): WireTurnRequest {
  return {
    messages: request.messages.map(toWireMessage),
    contextId: request.contextId,
    instructions: request.instructions,
    history: request.history?.map(toWireMessage),
    tools: request.tools,
    generationConfig: request.generationConfig,
    responseSchema: request.responseSchema,
    tokenBatchSize: request.tokenBatchSize,
  }
}

/** @internal */
export function toWireContextOptions(options: LLMContextOptions): WireContextOptions {
  return {
    instructions: options.instructions,
    history: options.history?.map(toWireMessage),
    tools: options.tools,
    generationConfig: options.generationConfig,
  }
}

/** @internal */
export function toWireTokenCountRequest(
  request: LLMTokenCountRequest,
): WireTokenCountRequest {
  return {
    contextId: request.contextId,
    instructions: request.instructions,
    history: request.history?.map(toWireMessage),
    tools: request.tools,
    messages: request.messages?.map(toWireMessage),
  }
}
