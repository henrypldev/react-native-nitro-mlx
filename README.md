# react-native-nitro-mlx

Run LLMs, Text-to-Speech, and Speech-to-Text on-device in React Native using [MLX Swift](https://github.com/ml-explore/mlx-swift).

## Requirements

- iOS 26.0+

## Installation

```bash
npm install react-native-nitro-mlx react-native-nitro-modules
```

Then run pod install:

```bash
cd ios && pod install
```

## Usage

### Download a Model

```typescript
import { ModelManager } from 'react-native-nitro-mlx'

await ModelManager.download('mlx-community/Qwen3-0.6B-4bit', (progress) => {
  console.log(`Download progress: ${(progress * 100).toFixed(1)}%`)
})
```

### Load and Generate

```typescript
import { LLM } from 'react-native-nitro-mlx'

await LLM.load('mlx-community/Qwen3-0.6B-4bit', {
  onProgress: (progress) => {
    console.log(`Loading: ${(progress * 100).toFixed(0)}%`)
  },
  manageHistory: true,
  generationConfig: {
    maxTokens: 1024,
    temperature: 0.7,
    topP: 0.9,
    prefillStepSize: 512,
  },
  tokenBatchSize: 8,
  contextConfig: {
    maxContextTokens: 4096,
    keepLastMessages: 6,
  },
})

const response = await LLM.generate('What is the capital of France?')
console.log(response)
```

### Load with Additional Context

You can provide conversation history or few-shot examples when loading the model:

```typescript
await LLM.load('mlx-community/Qwen3-0.6B-4bit', {
  onProgress: (progress) => {
    console.log(`Loading: ${(progress * 100).toFixed(0)}%`)
  },
  additionalContext: [
    { role: 'user', content: 'What is machine learning?' },
    { role: 'assistant', content: 'Machine learning is...' },
    { role: 'user', content: 'Can you explain neural networks?' }
  ]
})
```

### Streaming

```typescript
let response = ''
await LLM.stream('Tell me a story', (token) => {
  response += token
  console.log(response)
})
```

### Stop Generation

```typescript
LLM.stop()
```

### Chat Session (high-level API)

For a session-oriented experience that manages structured history, streaming
state, and tool-call metadata for you, use `createChatSession`:

```typescript
import { createChatSession, MLXModel } from 'react-native-nitro-mlx'

const chat = createChatSession({
  modelId: MLXModel.Qwen3_1_7B_4bit,
  systemPrompt: 'You are a helpful assistant.',
  tools: [weatherTool],
  onUpdate: state => {
    // state.status, state.partialAssistantContent, state.activeToolCalls, ...
  },
})

await chat.load({ onProgress: p => console.log(`${(p * 100).toFixed(0)}%`) })

const assistant = await chat.sendMessage('Plan a 3-day trip to Tokyo', {
  onToken: token => {
    // append token to UI
  },
  onToolCall: call => {
    // render tool-call card with call.status + call.arguments
  },
})

console.log(assistant.content)
console.log(chat.messages)            // full typed history
console.log(chat.state.status)        // 'done'
console.log(chat.state.lastStats)     // GenerationStats from the last turn

chat.reset()                          // clear history, keep system prompt
chat.unload()                         // release the model
```

`ChatSession` delegates to the same low-level `LLM` module, so the existing
`LLM.stream` / `LLM.streamWithEvents` APIs remain available for advanced use
cases.

#### ChatSessionOptions

| Option | Description |
|--------|-------------|
| `modelId` | HuggingFace model id to load |
| `systemPrompt` | System prompt applied on `load()` |
| `initialMessages` | Seed messages appended to JS history and forwarded as `additionalContext` (system-role entries stay in JS history only) |
| `tools` | Tool definitions available to the model |
| `generationConfig` | Default `LLMGenerationConfig` (temperature, top-p, max tokens, ...) |
| `contextConfig` | `LLMContextConfig` for managed-history trimming |
| `tokenBatchSize` | Tokens batched per JS bridge hop |
| `onUpdate` | Called on every state transition with the latest snapshot |
| `onMessage` | Called when a user/assistant/tool message is appended to history |
| `onToken` | Called for each streamed assistant token |
| `onToolCall` | Called on every tool-call lifecycle update |
| `onError` | Called when `load()` or `sendMessage()` fails |

#### ChatSession methods

| Method | Description |
|--------|-------------|
| `load(options?): Promise<void>` | Load the model, apply system prompt, tools, and initial messages |
| `sendMessage(text, options?): Promise<AssistantChatMessage>` | Append a user message, stream generation, resolve with the final assistant message |
| `stop(): void` | Abort the in-flight generation |
| `reset(): void` | Clear history + transient state; keeps system messages from `initialMessages` |
| `clearHistory(): void` | Clear user/assistant/tool messages from JS + native history |
| `setSystemPrompt(prompt): void` | Update the system prompt |
| `setMessages(messages): void` | Replace JS-side history |
| `deleteMessage(id): boolean` | Remove a message by id |
| `updateMessage(id, patch): boolean` | Patch a message by id |
| `subscribe(listener): () => void` | Subscribe to state updates; returns unsubscribe |
| `unload(): void` | Unload the model |

#### ChatSessionState

| Field | Description |
|-------|-------------|
| `status` | `'idle' \| 'loading' \| 'streaming' \| 'tool_calling' \| 'done' \| 'error'` |
| `isGenerating` | Whether a turn is in progress |
| `isLoaded` | Whether the model has been loaded |
| `partialAssistantContent` | Accumulated assistant content during streaming |
| `partialAssistantThinking` | Accumulated thinking content during the current thinking block |
| `activeToolCalls` | Tool calls currently in-flight for the active turn |
| `lastError` | Last error thrown by `load()` or `sendMessage()` |
| `lastStats` | Stats from the last completed generation |

### Text-to-Speech

```typescript
import { TTS, MLXModel } from 'react-native-nitro-mlx'

await TTS.load(MLXModel.PocketTTS, {
  onProgress: (progress) => {
    console.log(`Loading: ${(progress * 100).toFixed(0)}%`)
  }
})

const audioBuffer = await TTS.generate('Hello world!', {
  voice: 'alba',
  speed: 1.0
})

// Or stream audio chunks as they're generated
await TTS.stream('Hello world!', (chunk) => {
  // Process each audio chunk
}, { voice: 'alba' })
```

Available voices: `alba`, `azelma`, `cosette`, `eponine`, `fantine`, `javert`, `jean`, `marius`

### Speech-to-Text

```typescript
import { STT, MLXModel } from 'react-native-nitro-mlx'

await STT.load(MLXModel.GLM_ASR_Nano_4bit, {
  onProgress: (progress) => {
    console.log(`Loading: ${(progress * 100).toFixed(0)}%`)
  }
})

// Transcribe an audio buffer
const text = await STT.transcribe(audioBuffer)

// Or use live microphone transcription
await STT.startListening()
const partial = await STT.transcribeBuffer() // Get current transcript
const final = await STT.stopListening()      // Stop and get final transcript
```

## API

### LLM

| Method | Description |
|--------|-------------|
| `load(modelId: string, options?: LLMLoadOptions): Promise<void>` | Load a model into memory |
| `generate(prompt: string): Promise<string>` | Generate a complete response |
| `stream(prompt: string, onToken: (token: string) => void): Promise<string>` | Stream tokens as they're generated |
| `stop(): void` | Stop the current generation |

#### LLMLoadOptions

| Property | Type | Description |
|----------|------|-------------|
| `onProgress` | `(progress: number) => void` | Optional callback invoked with loading progress (0-1) |
| `additionalContext` | `LLMMessage[]` | Optional conversation history or few-shot examples to provide to the model |
| `manageHistory` | `boolean` | Enables managed chat history |
| `tools` | `ToolDefinition[]` | Tools the model may call while streaming |
| `generationConfig` | `LLMGenerationConfig` | Default generation parameters such as `maxTokens`, `temperature`, `topP`, KV cache config, and `prefillStepSize` |
| `tokenBatchSize` | `number` | Number of streamed chunks to batch before crossing the JS bridge |
| `contextConfig` | `LLMContextConfig` | Managed-history trimming settings such as `maxContextTokens` and `keepLastMessages` |

#### LLMMessage

| Property | Type | Description |
|----------|------|-------------|
| `role` | `'user' \| 'assistant' \| 'system'` | The role of the message sender |
| `content` | `string` | The message content |

| Property | Description |
|----------|-------------|
| `isLoaded: boolean` | Whether a model is loaded |
| `isGenerating: boolean` | Whether generation is in progress |
| `modelId: string` | The currently loaded model ID |
| `debug: boolean` | Enable debug logging |

### TTS

| Method | Description |
|--------|-------------|
| `load(modelId: string, options?: TTSLoadOptions): Promise<void>` | Load a TTS model into memory |
| `generate(text: string, options?: TTSGenerateOptions): Promise<ArrayBuffer>` | Generate audio from text |
| `stream(text: string, onAudioChunk: (audio: ArrayBuffer) => void, options?: TTSGenerateOptions): Promise<void>` | Stream audio chunks as they're generated |
| `stop(): void` | Stop the current generation |
| `unload(): void` | Unload the model and free memory |

#### TTSGenerateOptions

| Property | Type | Description |
|----------|------|-------------|
| `voice` | `string` | Voice to use (alba, azelma, cosette, eponine, fantine, javert, jean, marius) |
| `speed` | `number` | Speech speed multiplier (`0.5`–`2.0`, default `1.0`). Currently uses linear resampling, so pitch changes with speed. |

| Property | Description |
|----------|-------------|
| `isLoaded: boolean` | Whether a TTS model is loaded |
| `isGenerating: boolean` | Whether audio generation is in progress |
| `modelId: string` | The currently loaded model ID |
| `sampleRate: number` | Audio sample rate of the loaded model (e.g. 24000) |

### STT

| Method | Description |
|--------|-------------|
| `load(modelId: string, options?: STTLoadOptions): Promise<void>` | Load an STT model into memory |
| `transcribe(audio: ArrayBuffer): Promise<string>` | Transcribe an audio buffer |
| `transcribeStream(audio: ArrayBuffer, onToken: (token: string) => void): Promise<string>` | Stream transcription tokens as they're generated |
| `startListening(): Promise<void>` | Start capturing audio from the microphone |
| `transcribeBuffer(): Promise<string>` | Transcribe the current audio buffer while listening |
| `stopListening(): Promise<string>` | Stop listening and transcribe final audio |
| `stop(): void` | Stop the current transcription |
| `unload(): void` | Unload the model and free memory |

| Property | Description |
|----------|-------------|
| `isLoaded: boolean` | Whether an STT model is loaded |
| `isTranscribing: boolean` | Whether transcription is in progress |
| `isListening: boolean` | Whether the microphone is active |
| `modelId: string` | The currently loaded model ID |

### ModelManager

| Method | Description |
|--------|-------------|
| `download(modelId: string, onProgress: (progress: number) => void): Promise<string>` | Download a model from Hugging Face |
| `isDownloaded(modelId: string): Promise<boolean>` | Check if a model is downloaded |
| `getDownloadedModels(): Promise<string[]>` | Get list of downloaded models |
| `deleteModel(modelId: string): Promise<void>` | Delete a downloaded model |
| `getModelPath(modelId: string): Promise<string>` | Get the local path of a model |

| Property | Description |
|----------|-------------|
| `debug: boolean` | Enable debug logging |

## Supported Models

Any MLX-compatible model from Hugging Face should work. The package exports an `MLXModel` enum with pre-defined models for convenience that are more likely to run well on-device:

```typescript
import { MLXModel } from 'react-native-nitro-mlx'

await ModelManager.download(MLXModel.Llama_3_2_1B_Instruct_4bit, (progress) => {
  console.log(`Download progress: ${(progress * 100).toFixed(1)}%`)
})
```

### LLM Models

| Model | Enum Key | Hugging Face ID |
|-------|----------|-----------------|
| **Llama 3.2 (Meta)** | | |
| Llama 3.2 1B 4-bit | `Llama_3_2_1B_Instruct_4bit` | `mlx-community/Llama-3.2-1B-Instruct-4bit` |
| Llama 3.2 1B 8-bit | `Llama_3_2_1B_Instruct_8bit` | `mlx-community/Llama-3.2-1B-Instruct-8bit` |
| Llama 3.2 3B 4-bit | `Llama_3_2_3B_Instruct_4bit` | `mlx-community/Llama-3.2-3B-Instruct-4bit` |
| Llama 3.2 3B 8-bit | `Llama_3_2_3B_Instruct_8bit` | `mlx-community/Llama-3.2-3B-Instruct-8bit` |
| **Qwen 2.5 (Alibaba)** | | |
| Qwen 2.5 0.5B 4-bit | `Qwen2_5_0_5B_Instruct_4bit` | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` |
| Qwen 2.5 0.5B 8-bit | `Qwen2_5_0_5B_Instruct_8bit` | `mlx-community/Qwen2.5-0.5B-Instruct-8bit` |
| Qwen 2.5 1.5B 4-bit | `Qwen2_5_1_5B_Instruct_4bit` | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` |
| Qwen 2.5 1.5B 8-bit | `Qwen2_5_1_5B_Instruct_8bit` | `mlx-community/Qwen2.5-1.5B-Instruct-8bit` |
| Qwen 2.5 3B 4-bit | `Qwen2_5_3B_Instruct_4bit` | `mlx-community/Qwen2.5-3B-Instruct-4bit` |
| Qwen 2.5 3B 8-bit | `Qwen2_5_3B_Instruct_8bit` | `mlx-community/Qwen2.5-3B-Instruct-8bit` |
| **Qwen 3** | | |
| Qwen 3 1.7B 4-bit | `Qwen3_1_7B_4bit` | `mlx-community/Qwen3-1.7B-4bit` |
| Qwen 3 1.7B 8-bit | `Qwen3_1_7B_8bit` | `mlx-community/Qwen3-1.7B-8bit` |
| **Gemma 3 (Google)** | | |
| Gemma 3 1B 4-bit | `Gemma_3_1B_IT_4bit` | `mlx-community/gemma-3-1b-it-4bit` |
| Gemma 3 1B 8-bit | `Gemma_3_1B_IT_8bit` | `mlx-community/gemma-3-1b-it-8bit` |
| **Phi 3.5 Mini (Microsoft)** | | |
| Phi 3.5 Mini 4-bit | `Phi_3_5_Mini_Instruct_4bit` | `mlx-community/Phi-3.5-mini-instruct-4bit` |
| Phi 3.5 Mini 8-bit | `Phi_3_5_Mini_Instruct_8bit` | `mlx-community/Phi-3.5-mini-instruct-8bit` |
| **Phi 4 Mini (Microsoft)** | | |
| Phi 4 Mini 4-bit | `Phi_4_Mini_Instruct_4bit` | `mlx-community/Phi-4-mini-instruct-4bit` |
| Phi 4 Mini 8-bit | `Phi_4_Mini_Instruct_8bit` | `mlx-community/Phi-4-mini-instruct-8bit` |
| **SmolLM (HuggingFace)** | | |
| SmolLM 1.7B 4-bit | `SmolLM_1_7B_Instruct_4bit` | `mlx-community/SmolLM-1.7B-Instruct-4bit` |
| SmolLM 1.7B 8-bit | `SmolLM_1_7B_Instruct_8bit` | `mlx-community/SmolLM-1.7B-Instruct-8bit` |
| **SmolLM2 (HuggingFace)** | | |
| SmolLM2 1.7B 4-bit | `SmolLM2_1_7B_Instruct_4bit` | `mlx-community/SmolLM2-1.7B-Instruct-4bit` |
| SmolLM2 1.7B 8-bit | `SmolLM2_1_7B_Instruct_8bit` | `mlx-community/SmolLM2-1.7B-Instruct-8bit` |
| **OpenELM (Apple)** | | |
| OpenELM 1.1B 4-bit | `OpenELM_1_1B_4bit` | `mlx-community/OpenELM-1_1B-4bit` |
| OpenELM 1.1B 8-bit | `OpenELM_1_1B_8bit` | `mlx-community/OpenELM-1_1B-8bit` |
| OpenELM 3B 4-bit | `OpenELM_3B_4bit` | `mlx-community/OpenELM-3B-4bit` |
| OpenELM 3B 8-bit | `OpenELM_3B_8bit` | `mlx-community/OpenELM-3B-8bit` |

### TTS Models

| Model | Enum Key | Hugging Face ID |
|-------|----------|-----------------|
| **PocketTTS (Kyutai)** - 44.6M params | | |
| PocketTTS bf16 | `PocketTTS` | `mlx-community/pocket-tts` |
| PocketTTS 8-bit | `PocketTTS_8bit` | `mlx-community/pocket-tts-8bit` |
| PocketTTS 4-bit | `PocketTTS_4bit` | `mlx-community/pocket-tts-4bit` |

### STT Models

| Model | Enum Key | Hugging Face ID |
|-------|----------|-----------------|
| **GLM-ASR (Alibaba)** - 1B params | | |
| GLM-ASR Nano 4-bit | `GLM_ASR_Nano_4bit` | `mlx-community/GLM-ASR-Nano-2512-4bit` |

Browse more models at [huggingface.co/mlx-community](https://huggingface.co/mlx-community).

## License

MIT
