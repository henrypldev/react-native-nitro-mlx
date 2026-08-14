export {
  type AssistantChatMessage,
  type ChatLoadOptions,
  type ChatMessage,
  type ChatMessageInit,
  type ChatRole,
  ChatSession,
  type ChatSessionListener,
  type ChatSessionOptions,
  type ChatSessionState,
  type ChatSessionStatus,
  type ChatToolCall,
  type ChatToolCallStatus,
  createChatSession,
  type SendMessageOptions,
  type SystemChatMessage,
  type ToolChatMessage,
  type UserChatMessage,
} from './chat'
export { cosineSimilarity, Embeddings } from './embeddings'
export {
  type EventCallback,
  LLM,
  type Message,
  type ToolCallInfo,
  type ToolCallUpdate,
} from './llm'
export { ModelManager } from './modelManager'
export {
  MLXModel,
  MLXModels,
  ModelFamily,
  type ModelInfo,
  ModelProvider,
  type ModelQuantization,
  type ModelType,
} from './models'
export type {
  Embeddings as EmbeddingsSpec,
  EmbeddingsEmbedOptions,
  EmbeddingsLoadOptions,
} from './specs/Embeddings.nitro'
export type {
  GenerationOutcomeEvent,
  GenerationStartEvent,
  GenerationStats,
  LLM as LLMSpec,
  LLMContextConfig,
  LLMGenerationConfig,
  LLMGenerationFinishReason,
  LLMGenerationOutcome,
  LLMLoadOptions,
  LLMToolExecution,
  StreamEvent,
  ThinkingChunkEvent,
  ThinkingEndEvent,
  ThinkingStartEvent,
  TokenEvent,
  ToolCallCompletedEvent,
  ToolCallExecutingEvent,
  ToolCallFailedEvent,
  ToolCallStartEvent,
  ToolDefinition,
  ToolParameter,
  ToolParameterType,
} from './specs/LLM.nitro'
export type { ModelManager as ModelManagerSpec } from './specs/ModelManager.nitro'
export type {
  STT as STTSpec,
  STTListeningOptions,
  STTLoadOptions,
  STTTranscribeOptions,
  STTTranscriptionInfo,
} from './specs/STT.nitro'
export type {
  TTS as TTSSpec,
  TTSGenerateOptions,
  TTSLoadOptions,
} from './specs/TTS.nitro'
export { STT } from './stt'
export { createTool, type TypeSafeToolDefinition } from './tool-utils'
export { TTS } from './tts'
