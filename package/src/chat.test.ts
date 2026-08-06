import { describe, expect, it, mock, spyOn } from 'bun:test'
import type { GenerationStats } from './specs/LLM.nitro'

mock.module('react-native-nitro-modules', () => {
  return {
    NitroModules: {
      createHybridObject: () => {
        throw new Error('Native module should be mocked in this test')
      },
    },
  }
})

const { ChatSession } = await import('./chat')
const { LLM } = await import('./llm')

describe('ChatSession terminal events', () => {
  it('preserves partial stats from generation_error', async () => {
    const stats: GenerationStats = {
      tokenCount: 3,
      tokensPerSecond: 6,
      timeToFirstToken: 120,
      totalTime: 500,
      toolExecutionTime: 0,
    }

    const loadSpy = spyOn(LLM, 'load').mockResolvedValue(undefined)
    const streamSpy = spyOn(LLM, 'streamWithEvents').mockImplementation(
      async (_prompt, onEvent) => {
        onEvent({ type: 'token', token: 'partial' })
        onEvent({
          type: 'generation_error',
          error: 'Generation failed during generate: boom',
          stage: 'generate',
          stats,
        })
        throw new Error('Generation failed during generate: boom')
      },
    )

    try {
      const chat = new ChatSession({ modelId: 'test/model' })
      await chat.load()

      await expect(chat.sendMessage('hello')).rejects.toThrow('boom')
      expect(chat.state.lastStats).toEqual(stats)
      expect(chat.messages.at(-1)).toMatchObject({
        role: 'assistant',
        content: 'partial',
        stats,
      })
    } finally {
      loadSpy.mockRestore()
      streamSpy.mockRestore()
    }
  })
})
