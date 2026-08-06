import { LLM } from './llm'
import { safeJsonParse } from './runtime'
import type {
  GenerationStats,
  LLMContextConfig,
  LLMGenerationConfig,
  LLMMessage,
  StreamEvent,
  ToolDefinition,
} from './specs/LLM.nitro'

const ERROR_PREFIX = '[react-native-nitro-mlx]'

/** Role of a chat message. */
export type ChatRole = 'system' | 'user' | 'assistant' | 'tool'

/** Lifecycle status of a tool call produced by the model. */
export type ChatToolCallStatus = 'pending' | 'executing' | 'completed' | 'failed'

/** Structured record describing a tool call made by the model. */
export interface ChatToolCall {
  id: string
  name: string
  arguments: Record<string, unknown>
  status: ChatToolCallStatus
  result?: unknown
  error?: string
  startedAt: number
  completedAt?: number
}

interface BaseChatMessageFields {
  id: string
  createdAt: number
}

export interface SystemChatMessage extends BaseChatMessageFields {
  role: 'system'
  content: string
}

export interface UserChatMessage extends BaseChatMessageFields {
  role: 'user'
  content: string
}

export interface AssistantChatMessage extends BaseChatMessageFields {
  role: 'assistant'
  content: string
  thinking?: string
  toolCalls?: ChatToolCall[]
  stats?: GenerationStats
  /** True while the message is still being streamed. */
  isStreaming?: boolean
  error?: string
}

export interface ToolChatMessage extends BaseChatMessageFields {
  role: 'tool'
  toolCallId: string
  name: string
  content: string
}

/** Discriminated union over all chat message roles. */
export type ChatMessage =
  | SystemChatMessage
  | UserChatMessage
  | AssistantChatMessage
  | ToolChatMessage

/** Message shape accepted when seeding or replacing history. `id` and `createdAt` are auto-filled. */
export type ChatMessageInit = { id?: string; createdAt?: number } & (
  | Omit<SystemChatMessage, 'id' | 'createdAt'>
  | Omit<UserChatMessage, 'id' | 'createdAt'>
  | Omit<AssistantChatMessage, 'id' | 'createdAt'>
  | Omit<ToolChatMessage, 'id' | 'createdAt'>
)

/** High-level state machine status surfaced to the UI. */
export type ChatSessionStatus =
  | 'idle'
  | 'loading'
  | 'streaming'
  | 'tool_calling'
  | 'done'
  | 'error'

export interface ChatSessionState {
  status: ChatSessionStatus
  isGenerating: boolean
  isLoaded: boolean
  modelId: string
  /** Partial assistant content accumulated during the current stream. */
  partialAssistantContent: string
  /** Partial assistant thinking content accumulated during the current thinking block. */
  partialAssistantThinking: string
  /** Tool calls that are currently in-flight for the active turn. */
  activeToolCalls: ChatToolCall[]
  lastError: Error | null
  lastStats: GenerationStats | null
}

export interface ChatSessionOptions {
  modelId: string
  systemPrompt?: string
  initialMessages?: ChatMessageInit[]
  tools?: ToolDefinition[]
  generationConfig?: LLMGenerationConfig
  contextConfig?: LLMContextConfig
  tokenBatchSize?: number
  /** Called on every state transition with the latest session snapshot. */
  onUpdate?: (state: ChatSessionState) => void
  /** Called when a new message is appended to history (user, assistant, or tool). */
  onMessage?: (message: ChatMessage) => void
  /** Called for each streamed token of assistant content. */
  onToken?: (token: string) => void
  /** Called on every tool-call lifecycle update (pending/executing/completed/failed). */
  onToolCall?: (toolCall: ChatToolCall) => void
  /** Called when generation or loading fails. */
  onError?: (error: Error) => void
}

export interface ChatLoadOptions {
  onProgress?: (progress: number) => void
}

export interface SendMessageOptions {
  /** Per-call token callback, invoked in addition to the session-level onToken. */
  onToken?: (token: string) => void
  /** Per-call tool-call callback, invoked in addition to the session-level onToolCall. */
  onToolCall?: (toolCall: ChatToolCall) => void
}

export type ChatSessionListener = (state: ChatSessionState) => void

function assertNonEmpty(value: unknown, name: string): string {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new TypeError(`${ERROR_PREFIX} ${name} must be a non-empty string.`)
  }
  return value
}

/**
 * High-level chat session built on top of the low-level `LLM` singleton.
 *
 * Maintains its own structured message history so the UI has a stable source
 * of truth, while delegating actual generation to the native MLX runtime.
 *
 * @remarks
 * The underlying `LLM` module is a singleton; only one session can actively
 * generate at a time. Creating multiple sessions against the same runtime is
 * allowed but callers must coordinate `load()` / `unload()` themselves.
 */
export class ChatSession {
  private readonly _options: ChatSessionOptions
  private readonly _listeners = new Set<ChatSessionListener>()
  private _messages: ChatMessage[] = []
  private _state: ChatSessionState
  private _systemPrompt: string | undefined
  private _isLoaded = false
  private _idCounter = 0

  constructor(options: ChatSessionOptions) {
    assertNonEmpty(options.modelId, 'ChatSession modelId')
    this._options = options
    this._systemPrompt = options.systemPrompt
    this._state = this._createInitialState()

    if (options.initialMessages?.length) {
      this._messages = options.initialMessages.map(m => this._normalizeMessage(m))
    }
  }

  /** Current messages in the session (copy — safe to mutate). */
  get messages(): ChatMessage[] {
    return this._messages.slice()
  }

  /** Current session state snapshot. */
  get state(): ChatSessionState {
    return {
      ...this._state,
      activeToolCalls: this._state.activeToolCalls.slice(),
    }
  }

  get status(): ChatSessionStatus {
    return this._state.status
  }

  get isGenerating(): boolean {
    return this._state.isGenerating
  }

  get isLoaded(): boolean {
    return this._isLoaded
  }

  get modelId(): string {
    return this._options.modelId
  }

  get systemPrompt(): string | undefined {
    return this._systemPrompt
  }

  /** Subscribe to state updates. Returns an unsubscribe function. */
  subscribe(listener: ChatSessionListener): () => void {
    this._listeners.add(listener)
    return () => {
      this._listeners.delete(listener)
    }
  }

  /**
   * Load the underlying MLX model and apply the session's system prompt,
   * tools, and any seeded initial messages as additional context.
   */
  async load(loadOptions?: ChatLoadOptions): Promise<void> {
    this._setState({ status: 'loading', lastError: null })

    if (this._systemPrompt !== undefined) {
      LLM.systemPrompt = this._systemPrompt
    }

    try {
      await LLM.load(this._options.modelId, {
        onProgress: loadOptions?.onProgress,
        manageHistory: true,
        additionalContext: this._buildAdditionalContext(),
        tools: this._options.tools,
        generationConfig: this._options.generationConfig,
        contextConfig: this._options.contextConfig,
        tokenBatchSize: this._options.tokenBatchSize,
      })
      this._isLoaded = true
      this._setState({ status: 'idle', isLoaded: true })
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error))
      this._handleError(err)
      throw err
    }
  }

  /** Unload the underlying model. Safe to call repeatedly. */
  unload(): void {
    try {
      LLM.unload()
    } catch {
      // ignore — best-effort cleanup
    }
    this._isLoaded = false
    this._setState({
      status: 'idle',
      isLoaded: false,
      isGenerating: false,
      activeToolCalls: [],
      partialAssistantContent: '',
      partialAssistantThinking: '',
    })
  }

  /**
   * Update the system prompt. Takes effect immediately on the native side,
   * but behavior for already-generated turns is undefined — prefer setting
   * before `load()` or after `reset()`.
   */
  setSystemPrompt(prompt: string): void {
    assertNonEmpty(prompt, 'systemPrompt')
    this._systemPrompt = prompt
    if (this._isLoaded) {
      LLM.systemPrompt = prompt
    }
  }

  /**
   * Replace the JS-side message history. Does not alter native history —
   * call `reset()` first (or reload the session) when strict alignment is
   * required.
   */
  setMessages(messages: ChatMessageInit[]): void {
    this._messages = messages.map(m => this._normalizeMessage(m))
    this._emitUpdate()
  }

  /**
   * Clear user/assistant/tool messages from both the JS and native history.
   * System messages seeded via `initialMessages` are preserved.
   */
  clearHistory(): void {
    this._messages = this._messages.filter(m => m.role === 'system')
    try {
      LLM.clearHistory()
    } catch {
      // ignore — native may not be loaded
    }
    this._setState({
      partialAssistantContent: '',
      partialAssistantThinking: '',
      activeToolCalls: [],
      lastError: null,
    })
  }

  /** Full reset: clears history, errors, and transient stream state. */
  reset(): void {
    this.clearHistory()
    this._setState({
      status: 'idle',
      lastStats: null,
    })
  }

  /** Remove a message by id. Returns `true` if a message was removed. */
  deleteMessage(id: string): boolean {
    const idx = this._messages.findIndex(m => m.id === id)
    if (idx < 0) {
      return false
    }
    this._messages.splice(idx, 1)
    this._emitUpdate()
    return true
  }

  /** Patch a message by id. Returns `true` if a message was updated. */
  updateMessage(id: string, patch: Partial<ChatMessage>): boolean {
    const idx = this._messages.findIndex(m => m.id === id)
    if (idx < 0) {
      return false
    }
    const current = this._messages[idx]
    if (!current) {
      return false
    }
    const next = {
      ...current,
      ...patch,
      id: current.id,
      role: current.role,
    } as ChatMessage
    this._messages[idx] = next
    this._emitUpdate()
    return true
  }

  /** Stop the current generation. No-op when idle. */
  stop(): void {
    try {
      LLM.stop()
    } catch {
      // ignore
    }
  }

  /**
   * Append a user message, stream a generation, and resolve with the final
   * assistant message. Throws if a generation is already in progress.
   */
  async sendMessage(
    content: string,
    options?: SendMessageOptions,
  ): Promise<AssistantChatMessage> {
    assertNonEmpty(content, 'sendMessage content')
    if (this._state.isGenerating) {
      throw new Error(`${ERROR_PREFIX} A generation is already in progress.`)
    }
    if (!this._isLoaded) {
      throw new Error(`${ERROR_PREFIX} Call load() before sendMessage().`)
    }

    const userMessage: UserChatMessage = {
      id: this._nextId('user'),
      role: 'user',
      content,
      createdAt: Date.now(),
    }
    this._messages.push(userMessage)
    try {
      this._options.onMessage?.(userMessage)
    } catch {
      // user callbacks shouldn't break the session
    }

    const assistantMessage: AssistantChatMessage = {
      id: this._nextId('assistant'),
      role: 'assistant',
      content: '',
      createdAt: Date.now(),
      isStreaming: true,
      toolCalls: [],
    }
    this._messages.push(assistantMessage)

    this._setState({
      status: 'streaming',
      isGenerating: true,
      partialAssistantContent: '',
      partialAssistantThinking: '',
      activeToolCalls: [],
      lastError: null,
    })

    const toolCallsById = new Map<string, ChatToolCall>()
    let thinkingBuffer = ''

    const emitToolCall = (toolCall: ChatToolCall) => {
      options?.onToolCall?.(toolCall)
      this._options.onToolCall?.(toolCall)
    }

    const handleEvent = (event: StreamEvent): void => {
      switch (event.type) {
        case 'generation_start':
          break
        case 'thinking_start':
          thinkingBuffer = ''
          this._setState({ partialAssistantThinking: '' })
          break
        case 'thinking_chunk':
          thinkingBuffer += event.chunk
          this._setState({ partialAssistantThinking: thinkingBuffer })
          break
        case 'thinking_end':
          assistantMessage.thinking = (assistantMessage.thinking ?? '') + event.content
          thinkingBuffer = ''
          this._setState({ partialAssistantThinking: '' })
          break
        case 'token':
          assistantMessage.content += event.token
          this._setState({ partialAssistantContent: assistantMessage.content })
          options?.onToken?.(event.token)
          this._options.onToken?.(event.token)
          break
        case 'tool_call_start': {
          const args = safeJsonParse<Record<string, unknown>>(event.arguments, {})
          const toolCall: ChatToolCall = {
            id: event.id,
            name: event.name,
            arguments: args,
            status: 'pending',
            startedAt: Date.now(),
          }
          toolCallsById.set(event.id, toolCall)
          assistantMessage.toolCalls = [...(assistantMessage.toolCalls ?? []), toolCall]
          this._setState({
            status: 'tool_calling',
            activeToolCalls: Array.from(toolCallsById.values()),
          })
          emitToolCall(toolCall)
          break
        }
        case 'tool_call_executing': {
          const toolCall = toolCallsById.get(event.id)
          if (!toolCall) break
          toolCall.status = 'executing'
          this._setState({ activeToolCalls: Array.from(toolCallsById.values()) })
          emitToolCall(toolCall)
          break
        }
        case 'tool_call_completed': {
          const toolCall = toolCallsById.get(event.id)
          if (!toolCall) break
          toolCall.status = 'completed'
          toolCall.result = safeJsonParse<unknown>(event.result, event.result)
          toolCall.completedAt = Date.now()
          this._pushToolMessage(toolCall)
          this._setState({
            status: 'streaming',
            activeToolCalls: Array.from(toolCallsById.values()),
          })
          emitToolCall(toolCall)
          break
        }
        case 'tool_call_failed': {
          const toolCall = toolCallsById.get(event.id)
          if (!toolCall) break
          toolCall.status = 'failed'
          toolCall.error = event.error
          toolCall.completedAt = Date.now()
          this._pushToolMessage(toolCall)
          this._setState({
            status: 'streaming',
            activeToolCalls: Array.from(toolCallsById.values()),
          })
          emitToolCall(toolCall)
          break
        }
        case 'generation_end':
          assistantMessage.content = event.content
          assistantMessage.stats = event.stats
          this._setState({ lastStats: event.stats })
          break
        case 'generation_error':
          assistantMessage.stats = event.stats
          this._setState({ lastStats: event.stats })
          break
      }
    }

    try {
      // LLM.streamWithEvents wraps the event callback in its own safe-callback.
      await LLM.streamWithEvents(content, handleEvent)
      assistantMessage.isStreaming = false
      this._setState({
        status: 'done',
        isGenerating: false,
        partialAssistantContent: '',
        partialAssistantThinking: '',
        activeToolCalls: [],
      })
      this._options.onMessage?.(assistantMessage)
      return assistantMessage
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error))
      assistantMessage.isStreaming = false
      assistantMessage.error = err.message
      this._handleError(err)
      throw err
    }
  }

  private _buildAdditionalContext(): LLMMessage[] | undefined {
    if (this._messages.length === 0) {
      return undefined
    }
    const ctx: LLMMessage[] = []
    for (const msg of this._messages) {
      if (msg.role === 'system') continue
      ctx.push({ role: msg.role, content: msg.content })
    }
    return ctx.length > 0 ? ctx : undefined
  }

  private _pushToolMessage(toolCall: ChatToolCall): void {
    let content: string
    if (toolCall.status === 'failed') {
      content = toolCall.error ?? 'unknown error'
    } else {
      try {
        content = JSON.stringify(toolCall.result ?? null)
      } catch {
        content = String(toolCall.result)
      }
    }
    const toolMessage: ToolChatMessage = {
      id: this._nextId('tool'),
      role: 'tool',
      toolCallId: toolCall.id,
      name: toolCall.name,
      content,
      createdAt: Date.now(),
    }
    this._messages.push(toolMessage)
    this._options.onMessage?.(toolMessage)
  }

  private _createInitialState(): ChatSessionState {
    return {
      status: 'idle',
      isGenerating: false,
      isLoaded: false,
      modelId: this._options.modelId,
      partialAssistantContent: '',
      partialAssistantThinking: '',
      activeToolCalls: [],
      lastError: null,
      lastStats: null,
    }
  }

  private _normalizeMessage(init: ChatMessageInit): ChatMessage {
    const id = init.id ?? this._nextId(init.role)
    const createdAt = init.createdAt ?? Date.now()
    return { ...init, id, createdAt } as ChatMessage
  }

  private _nextId(prefix: string): string {
    this._idCounter += 1
    return `${prefix}-${Date.now().toString(36)}-${this._idCounter.toString(36)}`
  }

  private _setState(patch: Partial<ChatSessionState>): void {
    this._state = { ...this._state, ...patch }
    this._emitUpdate()
  }

  private _emitUpdate(): void {
    const snapshot = this.state
    try {
      this._options.onUpdate?.(snapshot)
    } catch {
      // ignore — user callbacks shouldn't break the session
    }
    for (const listener of this._listeners) {
      try {
        listener(snapshot)
      } catch {
        // ignore
      }
    }
  }

  private _handleError(err: Error): void {
    this._setState({
      status: 'error',
      isGenerating: false,
      lastError: err,
      partialAssistantContent: '',
      partialAssistantThinking: '',
      activeToolCalls: [],
    })
    this._options.onError?.(err)
  }
}

/** Factory helper that returns a new {@link ChatSession}. */
export function createChatSession(options: ChatSessionOptions): ChatSession {
  return new ChatSession(options)
}
