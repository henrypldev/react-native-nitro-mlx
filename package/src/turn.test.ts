import { describe, expect, it, mock } from 'bun:test'

const nativeState = {
  isLoaded: false,
  modelId: '',
}

mock.module('react-native-nitro-modules', () => ({
  NitroModules: {
    createHybridObject: () => nativeState,
  },
}))

const { LLM } = await import('./llm')

describe('LLM.loadedModelId', () => {
  it('is null when no model is loaded', () => {
    nativeState.isLoaded = false
    nativeState.modelId = ''
    expect(LLM.loadedModelId).toBeNull()
  })

  it('is the model id when loaded', () => {
    nativeState.isLoaded = true
    nativeState.modelId = 'mlx-community/Qwen3-1.7B-4bit'
    expect(LLM.loadedModelId).toBe('mlx-community/Qwen3-1.7B-4bit')
  })
})
