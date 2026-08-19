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

const { validateTurnMessages, validateToolSchemas, validateTurnRequest } = await import(
  './runtime'
)

describe('validateTurnMessages', () => {
  it('rejects an empty array when required', () => {
    expect(() => validateTurnMessages([], 'messages', { requireNonEmpty: true })).toThrow(
      /messages must not be empty/,
    )
  })

  it('rejects an unknown role', () => {
    expect(() =>
      validateTurnMessages([{ role: 'oracle', content: 'x' }], 'messages', {}),
    ).toThrow(/unknown role/)
  })

  it('rejects a tool message without toolCallId', () => {
    expect(() =>
      validateTurnMessages([{ role: 'tool', content: 'x' }], 'messages', {}),
    ).toThrow(/toolCallId/)
  })

  it('rejects toolCallsJson on a non-assistant message', () => {
    expect(() =>
      validateTurnMessages(
        [{ role: 'user', content: 'x', toolCallsJson: '[]' }],
        'messages',
        {},
      ),
    ).toThrow(/assistant/)
  })

  it('accepts a full loop shape', () => {
    expect(() =>
      validateTurnMessages(
        [
          { role: 'user', content: 'hi' },
          {
            role: 'assistant',
            content: '',
            toolCallsJson: '[{"id":"c1","name":"f","arguments":{}}]',
          },
          { role: 'tool', toolCallId: 'c1', content: 'ok', isError: false },
        ],
        'messages',
        {},
      ),
    ).not.toThrow()
  })
})

describe('validateToolSchemas', () => {
  it('rejects parameters that are not JSON', () => {
    expect(() =>
      validateToolSchemas(
        [{ name: 'f', description: 'd', parameters: '{oops' }],
        'tools',
      ),
    ).toThrow(/JSON/)
  })

  it('rejects a non-object root schema', () => {
    expect(() =>
      validateToolSchemas(
        [{ name: 'f', description: 'd', parameters: '{"type":"string"}' }],
        'tools',
      ),
    ).toThrow(/object schema/)
  })

  it('rejects duplicate tool names', () => {
    const tool = { name: 'f', description: 'd', parameters: '{"type":"object"}' }
    expect(() => validateToolSchemas([tool, tool], 'tools')).toThrow(/duplicate/)
  })
})

describe('validateTurnRequest', () => {
  const user = [{ role: 'user', content: 'hi' }]

  it('rejects responseSchema together with tools', () => {
    expect(() =>
      validateTurnRequest({
        messages: user,
        responseSchema: '{"type":"object"}',
        tools: [{ name: 'f', description: 'd', parameters: '{"type":"object"}' }],
      }),
    ).toThrow(/exclusive/)
  })

  it('rejects a responseSchema that is not JSON', () => {
    expect(() =>
      validateTurnRequest({ messages: user, responseSchema: '{oops' }),
    ).toThrow(/JSON/)
  })

  it('rejects a responseSchema with a non-object root', () => {
    expect(() =>
      validateTurnRequest({ messages: user, responseSchema: '{"type":"string"}' }),
    ).toThrow(/object schema/)
  })

  it('rejects cold-turn fields on a warm request', () => {
    expect(() =>
      validateTurnRequest({
        messages: user,
        contextId: 'ctx-1',
        instructions: 'be brief',
      }),
    ).toThrow(/contextId/)
  })

  it('rejects generationConfig on a warm request', () => {
    expect(() =>
      validateTurnRequest({
        messages: user,
        contextId: 'ctx-1',
        generationConfig: { temperature: 0.5 },
      }),
    ).toThrow(/contextId/)
  })

  it('accepts a minimal cold request', () => {
    expect(() => validateTurnRequest({ messages: user })).not.toThrow()
  })
})

const { toWireMessage, fromWireOutcome } = await import('./turn')

describe('toWireMessage', () => {
  it('serializes assistant tool calls to JSON', () => {
    const wire = toWireMessage({
      role: 'assistant',
      content: 'calling',
      toolCalls: [{ id: 'c1', name: 'f', arguments: { path: '/a' } }],
    })
    expect(wire.role).toBe('assistant')
    expect(JSON.parse(wire.toolCallsJson ?? '')).toEqual([
      { id: 'c1', name: 'f', arguments: { path: '/a' } },
    ])
  })

  it('passes tool result fields through flat', () => {
    const wire = toWireMessage({
      role: 'tool',
      toolCallId: 'c1',
      name: 'f',
      content: 'ok',
      isError: true,
    })
    expect(wire).toEqual({
      role: 'tool',
      content: 'ok',
      toolCallId: 'c1',
      name: 'f',
      isError: true,
      toolCallsJson: undefined,
    })
  })
})

describe('fromWireOutcome', () => {
  const base = {
    finishReason: 'tool_calls' as const,
    content: '',
    toolCalls: [{ id: 'c1', name: 'f', argumentsJson: '{"path":"/a"}' }],
    usage: { promptTokens: 10, completionTokens: 5 },
    stats: {
      tokenCount: 5,
      tokensPerSecond: 1,
      timeToFirstToken: 1,
      totalTime: 1,
      toolExecutionTime: 0,
    },
  }

  it('parses tool call arguments to objects', () => {
    const outcome = fromWireOutcome(base)
    expect(outcome.toolCalls[0]?.arguments).toEqual({ path: '/a' })
  })

  it('degrades unparseable arguments to an empty object', () => {
    const outcome = fromWireOutcome({
      ...base,
      toolCalls: [{ id: 'c1', name: 'f', argumentsJson: '{broken' }],
    })
    expect(outcome.toolCalls[0]?.arguments).toEqual({})
  })
})

describe('LLM.runTurn', () => {
  it('rejects an invalid request before touching native', async () => {
    const { LLM } = await import('./llm')
    await expect(LLM.runTurn({ messages: [] })).rejects.toThrow(
      /messages must not be empty/,
    )
  })
})
