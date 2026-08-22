import { LegendList, type LegendListRef } from '@legendapp/list/react-native'
import * as Crypto from 'expo-crypto'
import { router, useFocusEffect } from 'expo-router'
import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Animated,
  LayoutAnimation,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native'
import { KeyboardAvoidingView } from 'react-native-keyboard-controller'
import {
  type JsonObject,
  LLM,
  type LLMContext,
  type LLMMessage,
  type LLMToolCall,
  type LLMTurnOutcome,
  MLXModel,
  ModelManager,
  type StreamEvent,
  type ToolSchema,
} from 'react-native-nitro-mlx'
import { SafeAreaView } from 'react-native-safe-area-context'
import { z } from 'zod'
import { type Palette, usePalette } from '../../constants/theme'

const MODEL_ID = MLXModel.Qwen3_1_7B_4bit
const MODEL_LABEL = 'Qwen3 1.7B'
const DEFAULT_MAX_STEPS = 6

const SUGGESTIONS = [
  "What's on my calendar today?",
  'Set a timer for 5 minutes labeled tea.',
  'Message Alex the time of my first meeting today.',
]

const TODAY_EVENTS = [
  '09:30 Standup',
  '11:00 Design review with Alex',
  '15:00 1:1 with Sam',
]

const AGENT_INSTRUCTIONS =
  'You are a helpful personal assistant. Use the available tools to gather ' +
  'facts before answering, and chain calls when the goal requires it ' +
  '(e.g. check the calendar before messaging someone about it). If a tool ' +
  'returns an error, read the message and correct your arguments rather ' +
  'than repeating the same call. Once you have what you need, stop calling ' +
  'tools and give a short, direct final answer.'

const TOOL_SCHEMAS: ToolSchema[] = [
  {
    name: 'get_current_time',
    description: 'Get the current device time as a readable string. Takes no parameters.',
    parameters: JSON.stringify({
      type: 'object',
      properties: {},
      additionalProperties: false,
    }),
  },
  {
    name: 'check_calendar',
    description: "Get today's calendar events. Takes no parameters.",
    parameters: JSON.stringify({
      type: 'object',
      properties: {},
      additionalProperties: false,
    }),
  },
  {
    name: 'set_timer',
    description: 'Start a timer for a number of minutes, with an optional label.',
    parameters: JSON.stringify({
      type: 'object',
      properties: {
        minutes: {
          type: 'number',
          description: 'Timer duration in minutes. Must be greater than 0.',
        },
        label: {
          type: 'string',
          description: 'Optional label describing what the timer is for.',
        },
      },
      required: ['minutes'],
      additionalProperties: false,
    }),
  },
  {
    name: 'send_message',
    description: 'Send a text message to a person.',
    parameters: JSON.stringify({
      type: 'object',
      properties: {
        to: { type: 'string', description: 'Recipient name.' },
        body: { type: 'string', description: 'Message text.' },
      },
      required: ['to', 'body'],
      additionalProperties: false,
    }),
  },
]

const TOOL_LABELS = new Map([
  ['get_current_time', 'Check the time'],
  ['check_calendar', 'Check the calendar'],
  ['set_timer', 'Set a timer'],
  ['send_message', 'Send a message'],
])

// --- Mock "personal assistant" toolbox -------------------------------------
// Module-level state so a multi-step goal visibly accumulates effects across
// the agent's tool calls within one conversation.

type TimerEntry = { id: string; minutes: number; label: string; createdAt: number }
type SentMessageEntry = { id: string; to: string; body: string; createdAt: number }

let mockTimers: TimerEntry[] = []
let mockSentMessages: SentMessageEntry[] = []

function resetMockState(): void {
  mockTimers = []
  mockSentMessages = []
}

type ToolExecutionResult = { content: string; isError: boolean }

// Model-produced arguments are best-effort JSON: coerce what is salvageable
// and fall back to sentinels the per-tool validation below reports on.
const setTimerArgsSchema = z.object({
  minutes: z.coerce.number().catch(Number.NaN),
  label: z.string().catch('Timer'),
})

const sendMessageArgsSchema = z.object({
  to: z.string().catch(''),
  body: z.string().catch(''),
})

function executeTool(name: string, args: JsonObject): ToolExecutionResult {
  switch (name) {
    case 'get_current_time':
      return { content: new Date().toLocaleString(), isError: false }

    case 'check_calendar':
      return { content: TODAY_EVENTS.join('\n'), isError: false }

    case 'set_timer': {
      const parsed = setTimerArgsSchema.parse(args)
      if (!Number.isFinite(parsed.minutes) || parsed.minutes <= 0) {
        return {
          content: `Invalid duration: "minutes" must be a positive number, got ${JSON.stringify(args.minutes)}`,
          isError: true,
        }
      }
      const minutes = parsed.minutes
      const label = parsed.label.trim().length > 0 ? parsed.label.trim() : 'Timer'
      mockTimers = [
        ...mockTimers,
        { id: Crypto.randomUUID(), minutes, label, createdAt: Date.now() },
      ]
      return { content: `Timer set for ${minutes} minute(s): "${label}"`, isError: false }
    }

    case 'send_message': {
      const parsed = sendMessageArgsSchema.parse(args)
      const to = parsed.to.trim()
      const body = parsed.body.trim()
      if (!to || !body) {
        return {
          content: 'Invalid arguments: "to" and "body" must both be non-empty strings.',
          isError: true,
        }
      }
      mockSentMessages = [
        ...mockSentMessages,
        { id: Crypto.randomUUID(), to, body, createdAt: Date.now() },
      ]
      return { content: `Message sent to ${to}`, isError: false }
    }

    default:
      return { content: `Unknown tool: ${name}`, isError: true }
  }
}

// --- The mini agent harness --------------------------------------------------
// The library hands back turn primitives; this is the caller-owned loop that
// turns them into an agent. The screen owns the Turn Context so the
// conversation (and its warm KV cache) survives across user messages.

type AgentStep =
  | { kind: 'tool'; call: LLMToolCall; result: string; isError: boolean }
  | { kind: 'answer'; text: string; thinking?: string }
  | { kind: 'aborted'; reason: string } // stopped / failed / length / max-steps

/**
 * Builds the human-readable abort reason. On 'failed', both turn.stage and
 * turn.error must surface — the stage says where it broke (e.g. schema
 * validation), the error says why.
 */
function describeAbortReason(turn: LLMTurnOutcome): string {
  if (turn.finishReason === 'failed') {
    return turn.stage
      ? `failed (${turn.stage}): ${turn.error ?? 'unknown error'}`
      : (turn.error ?? 'failed')
  }
  return turn.error ?? turn.finishReason
}

async function runAgentTurns(
  contextId: string,
  goal: string,
  opts: {
    onStep: (step: AgentStep) => void
    onEvent: (event: StreamEvent) => void
    maxSteps?: number
  },
): Promise<void> {
  const maxSteps = opts.maxSteps ?? DEFAULT_MAX_STEPS
  let messages: LLMMessage[] = [{ role: 'user', content: goal }]
  let reachedMaxSteps = true

  for (let step = 0; step < maxSteps; step += 1) {
    const turn = await LLM.runTurn({ contextId, messages }, opts.onEvent)

    if (
      turn.finishReason === 'stopped' ||
      turn.finishReason === 'failed' ||
      turn.finishReason === 'length'
    ) {
      opts.onStep({ kind: 'aborted', reason: describeAbortReason(turn) })
      reachedMaxSteps = false
      break
    }

    if (turn.toolCalls.length === 0) {
      opts.onStep({ kind: 'answer', text: turn.content, thinking: turn.thinking })
      reachedMaxSteps = false
      break
    }

    const results: LLMMessage[] = []
    for (const call of turn.toolCalls) {
      const { content, isError } = executeTool(call.name, call.arguments)
      opts.onStep({ kind: 'tool', call, result: content, isError })
      results.push({
        role: 'tool',
        toolCallId: call.id,
        name: call.name,
        content,
        isError: isError || undefined,
      })
    }
    messages = results
  }

  if (reachedMaxSteps) {
    opts.onStep({ kind: 'aborted', reason: 'max-steps' })
  }
}

function noticeText(reason: string): string {
  if (reason === 'max-steps') return `Stopped after ${DEFAULT_MAX_STEPS} tool steps`
  if (reason === 'stopped') return 'Stopped'
  if (reason === 'length') return 'Stopped at the response length limit'
  return `Agent stopped: ${reason}`
}

// --- Screen ------------------------------------------------------------------

type Entry =
  | { id: string; kind: 'user'; text: string }
  | { id: string; kind: 'assistant'; text: string; thinking?: string }
  | {
      id: string
      kind: 'tool'
      name: string
      args: JsonObject
      result: string
      isError: boolean
    }
  | { id: string; kind: 'notice'; text: string }

type Row =
  | Entry
  | { id: 'streaming'; kind: 'streaming'; text: string }
  | { id: 'working'; kind: 'working'; label: string }

type ModelPhase = 'checking' | 'absent' | 'loading' | 'error' | 'ready'

function usePulse(): Animated.Value {
  const opacity = useRef(new Animated.Value(1)).current
  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(opacity, { toValue: 0.25, duration: 700, useNativeDriver: true }),
        Animated.timing(opacity, { toValue: 1, duration: 700, useNativeDriver: true }),
      ]),
    )
    loop.start()
    return () => loop.stop()
  }, [opacity])
  return opacity
}

const PulseDot = ({ color, size = 8 }: { color: string; size?: number }) => {
  const opacity = usePulse()
  return (
    <Animated.View
      style={{
        width: size,
        height: size,
        borderRadius: size / 2,
        backgroundColor: color,
        opacity,
      }}
    />
  )
}

const Rail = ({
  palette,
  hasPrev,
  hasNext,
  children,
}: {
  palette: Palette
  hasPrev: boolean
  hasNext: boolean
  children: ReactNode
}) => (
  <View style={styles.railCol}>
    <View
      style={[
        styles.railLine,
        styles.railLineTop,
        { backgroundColor: palette.hairline, opacity: hasPrev ? 1 : 0 },
      ]}
    />
    {children}
    <View
      style={[
        styles.railLine,
        styles.railLineBottom,
        { backgroundColor: palette.hairline, opacity: hasNext ? 1 : 0 },
      ]}
    />
  </View>
)

const ToolStepRow = ({
  entry,
  palette,
  hasPrev,
  hasNext,
}: {
  entry: Extract<Entry, { kind: 'tool' }>
  palette: Palette
  hasPrev: boolean
  hasNext: boolean
}) => {
  const [expanded, setExpanded] = useState(false)

  const toggleExpanded = () => {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut)
    setExpanded(!expanded)
  }

  const label =
    TOOL_LABELS.get(entry.name) ??
    entry.name.replace(/_/g, ' ').replace(/^\w/, c => c.toUpperCase())
  const markColor = entry.isError ? palette.error : palette.muted

  return (
    <View style={styles.stepRow}>
      <Rail palette={palette} hasPrev={hasPrev} hasNext={hasNext}>
        <View style={[styles.railNode, { borderColor: markColor }]}>
          <Text style={[styles.railNodeMark, { color: markColor }]}>
            {entry.isError ? '✕' : '✓'}
          </Text>
        </View>
      </Rail>
      <TouchableOpacity
        style={styles.stepBody}
        onPress={toggleExpanded}
        activeOpacity={0.6}
      >
        <Text
          style={[
            styles.stepLabel,
            { color: entry.isError ? palette.error : palette.ink },
          ]}
        >
          {label}
        </Text>
        {expanded && (
          <View style={[styles.stepDetail, { backgroundColor: palette.surface }]}>
            <Text style={[styles.stepDetailText, { color: palette.muted }]}>
              {JSON.stringify(entry.args, null, 2)}
            </Text>
            <Text style={[styles.stepDetailText, { color: palette.muted }]}>
              → {entry.result}
            </Text>
          </View>
        )}
      </TouchableOpacity>
    </View>
  )
}

const WorkingRow = ({
  label,
  palette,
  hasPrev,
}: {
  label: string
  palette: Palette
  hasPrev: boolean
}) => (
  <View style={styles.stepRow}>
    <Rail palette={palette} hasPrev={hasPrev} hasNext={false}>
      <View style={styles.railPulse}>
        <PulseDot color={palette.ember} size={9} />
      </View>
    </Rail>
    <Text style={[styles.workingLabel, { color: palette.muted }]}>{label}</Text>
  </View>
)

const AssistantRow = ({
  entry,
  palette,
}: {
  entry: Extract<Entry, { kind: 'assistant' }>
  palette: Palette
}) => {
  const [showThinking, setShowThinking] = useState(false)

  const toggleThinking = () => {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut)
    setShowThinking(!showThinking)
  }

  return (
    <View style={styles.assistantBlock}>
      {entry.thinking ? (
        <TouchableOpacity onPress={toggleThinking} activeOpacity={0.6}>
          <Text style={[styles.thinkingToggle, { color: palette.muted }]}>
            {showThinking ? 'Hide thinking' : 'Show thinking'}
          </Text>
          {showThinking && (
            <Text style={[styles.thinkingText, { color: palette.muted }]}>
              {entry.thinking}
            </Text>
          )}
        </TouchableOpacity>
      ) : null}
      {entry.text ? (
        <Text style={[styles.assistantText, { color: palette.ink }]}>{entry.text}</Text>
      ) : null}
    </View>
  )
}

const ModelChip = ({ palette, pulsing }: { palette: Palette; pulsing?: boolean }) => (
  <View style={[styles.modelChip, { borderColor: palette.hairline }]}>
    {pulsing ? (
      <PulseDot color={palette.ember} size={7} />
    ) : (
      <View style={[styles.chipDot, { backgroundColor: palette.ember }]} />
    )}
    <Text style={[styles.modelChipText, { color: palette.muted }]}>MLX</Text>
  </View>
)

const openDownloadModal = () => {
  router.push('/download-modal')
}

export default function AgentChatScreen() {
  const palette = usePalette()
  const [phase, setPhase] = useState<ModelPhase>('checking')
  const [loadProgress, setLoadProgress] = useState(0)
  const [loadError, setLoadError] = useState('')
  const [entries, setEntries] = useState<Entry[]>([])
  const [input, setInput] = useState('')
  const [isRunning, setIsRunning] = useState(false)
  const [streamingText, setStreamingText] = useState('')
  const [isThinking, setIsThinking] = useState(false)

  const listRef = useRef<LegendListRef>(null)
  const ctxRef = useRef<LLMContext | null>(null)
  const isFocusedRef = useRef(false)

  const prepareModel = useCallback(async () => {
    setPhase('checking')
    try {
      const downloaded = await ModelManager.isDownloaded(MODEL_ID)
      if (!isFocusedRef.current) return
      if (!downloaded) {
        setPhase('absent')
        return
      }
      setPhase('loading')
      setLoadProgress(0)
      await LLM.load(MODEL_ID, { onProgress: setLoadProgress })
      if (!isFocusedRef.current) return
      setPhase('ready')
    } catch (error) {
      if (!isFocusedRef.current) return
      setLoadError(error instanceof Error ? error.message : String(error))
      setPhase('error')
    }
  }, [])

  useFocusEffect(
    useCallback(() => {
      isFocusedRef.current = true
      prepareModel()
      return () => {
        isFocusedRef.current = false
        ctxRef.current?.release()
        ctxRef.current = null
        LLM.unload()
      }
    }, [prepareModel]),
  )

  const startNewChat = () => {
    if (isRunning) return
    ctxRef.current?.release()
    ctxRef.current = null
    resetMockState()
    setEntries([])
  }

  const send = useCallback(
    async (raw: string) => {
      const text = raw.trim()
      if (!text || isRunning || phase !== 'ready') return

      setEntries(prev => [...prev, { id: Crypto.randomUUID(), kind: 'user', text }])
      setInput('')
      setIsRunning(true)
      setStreamingText('')
      setIsThinking(false)

      try {
        if (!ctxRef.current) {
          ctxRef.current = await LLM.createContext({
            instructions: AGENT_INSTRUCTIONS,
            tools: TOOL_SCHEMAS,
            // Qwen3 thinks by default; the reasoning trace is the main source
            // of generated tokens and therefore of sustained GPU heat.
            generationConfig: { enableThinking: false },
          })
        }

        let entrySeq = 0
        await runAgentTurns(ctxRef.current.id, text, {
          onEvent: event => {
            if (event.type === 'token') {
              setIsThinking(false)
              setStreamingText(prev => prev + event.token)
            } else if (
              event.type === 'thinking_start' ||
              event.type === 'thinking_chunk'
            ) {
              setIsThinking(true)
            } else if (event.type === 'thinking_end') {
              setIsThinking(false)
            }
          },
          onStep: step => {
            setStreamingText('')
            setIsThinking(false)
            entrySeq += 1
            if (step.kind === 'tool') {
              setEntries(prev => [
                ...prev,
                {
                  id: `${step.call.id}-${entrySeq}`,
                  kind: 'tool',
                  name: step.call.name,
                  args: step.call.arguments,
                  result: step.result,
                  isError: step.isError,
                },
              ])
            } else if (step.kind === 'answer') {
              setEntries(prev => [
                ...prev,
                {
                  id: Crypto.randomUUID(),
                  kind: 'assistant',
                  text: step.text.trim(),
                  thinking: step.thinking?.trim() || undefined,
                },
              ])
            } else {
              setEntries(prev => [
                ...prev,
                {
                  id: Crypto.randomUUID(),
                  kind: 'notice',
                  text: noticeText(step.reason),
                },
              ])
            }
          },
        })
      } catch (error) {
        setEntries(prev => [
          ...prev,
          {
            id: Crypto.randomUUID(),
            kind: 'notice',
            text: error instanceof Error ? error.message : String(error),
          },
        ])
      } finally {
        setIsRunning(false)
        setStreamingText('')
        setIsThinking(false)
      }
    },
    [isRunning, phase],
  )

  const rows = useMemo<Row[]>(() => {
    if (!isRunning) return entries
    if (streamingText.trim()) {
      return [...entries, { id: 'streaming', kind: 'streaming', text: streamingText }]
    }
    return [
      ...entries,
      { id: 'working', kind: 'working', label: isThinking ? 'Thinking…' : 'Working…' },
    ]
  }, [entries, isRunning, streamingText, isThinking])

  const renderRow = useCallback(
    ({ item, index }: { item: Row; index: number }) => {
      const prevIsTool = rows[index - 1]?.kind === 'tool'
      const nextKind = rows[index + 1]?.kind
      const nextIsRail = nextKind === 'tool' || nextKind === 'working'

      switch (item.kind) {
        case 'user':
          return (
            <View
              style={[
                styles.userBubble,
                { backgroundColor: palette.ink, borderCurve: 'continuous' },
              ]}
            >
              <Text style={[styles.userText, { color: palette.onInk }]}>{item.text}</Text>
            </View>
          )
        case 'assistant':
          return <AssistantRow entry={item} palette={palette} />
        case 'tool':
          return (
            <ToolStepRow
              entry={item}
              palette={palette}
              hasPrev={prevIsTool}
              hasNext={nextIsRail}
            />
          )
        case 'notice':
          return (
            <Text style={[styles.noticeText, { color: palette.muted }]}>{item.text}</Text>
          )
        case 'streaming':
          return (
            <View style={styles.assistantBlock}>
              <Text style={[styles.assistantText, { color: palette.ink }]}>
                {item.text}
              </Text>
            </View>
          )
        case 'working':
          return <WorkingRow label={item.label} palette={palette} hasPrev={prevIsTool} />
      }
    },
    [rows, palette],
  )

  const statusLabel = {
    checking: 'checking…',
    absent: 'not installed',
    loading: `loading ${(loadProgress * 100).toFixed(0)}%`,
    error: 'load failed',
    ready: `${MODEL_LABEL} · on-device`,
  }[phase]

  const statusDotColor = {
    checking: palette.muted,
    absent: palette.muted,
    loading: palette.ember,
    error: palette.error,
    ready: palette.ember,
  }[phase]

  return (
    <SafeAreaView
      style={[styles.container, { backgroundColor: palette.bg }]}
      edges={['bottom', 'top']}
    >
      <KeyboardAvoidingView style={styles.container} behavior="padding">
        <View style={[styles.header, { borderBottomColor: palette.hairline }]}>
          <View>
            <Text style={[styles.headerTitle, { color: palette.ink }]}>MLX Chat</Text>
            <View style={styles.statusRow}>
              {phase === 'loading' ? (
                <PulseDot color={palette.ember} size={6} />
              ) : (
                <View style={[styles.statusDot, { backgroundColor: statusDotColor }]} />
              )}
              <Text style={[styles.statusText, { color: palette.muted }]}>
                {statusLabel}
              </Text>
            </View>
          </View>
          <TouchableOpacity
            accessibilityLabel="New chat"
            style={[
              styles.newChatButton,
              { borderColor: palette.hairline },
              (isRunning || entries.length === 0) && styles.newChatDisabled,
            ]}
            onPress={startNewChat}
            disabled={isRunning || entries.length === 0}
          >
            <Text style={[styles.newChatGlyph, { color: palette.ink }]}>+</Text>
          </TouchableOpacity>
        </View>

        {phase === 'ready' && rows.length > 0 ? (
          <LegendList<Row>
            ref={listRef}
            data={rows}
            keyExtractor={item => item.id}
            estimatedItemSize={80}
            renderItem={renderRow}
            contentContainerStyle={styles.threadContent}
            alignItemsAtEnd
            maintainScrollAtEnd
            maintainVisibleContentPosition
          />
        ) : (
          <View style={styles.stage}>
            {phase === 'ready' ? (
              <>
                <View style={styles.greetingBlock}>
                  <Text style={[styles.greeting, { color: palette.ink }]}>
                    Ready when you are.
                  </Text>
                  <Text style={[styles.greetingSub, { color: palette.muted }]}>
                    Ask for something that takes a few steps. The agent plans, calls
                    tools, and answers — all on this device.
                  </Text>
                </View>
                <View style={styles.suggestions}>
                  {SUGGESTIONS.map(suggestion => (
                    <TouchableOpacity
                      key={suggestion}
                      style={[
                        styles.suggestionCard,
                        { backgroundColor: palette.surface, borderCurve: 'continuous' },
                      ]}
                      onPress={() => send(suggestion)}
                      activeOpacity={0.7}
                    >
                      <Text style={[styles.suggestionText, { color: palette.ink }]}>
                        {suggestion}
                      </Text>
                      <Text style={[styles.suggestionChevron, { color: palette.muted }]}>
                        ›
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
              </>
            ) : (
              <View style={styles.gate}>
                <ModelChip palette={palette} pulsing={phase === 'loading'} />
                {phase === 'absent' && (
                  <>
                    <Text style={[styles.gateTitle, { color: palette.ink }]}>
                      Install the model
                    </Text>
                    <Text style={[styles.gateSub, { color: palette.muted }]}>
                      {MODEL_LABEL} (4-bit) downloads once and then runs offline.
                    </Text>
                    <TouchableOpacity
                      style={[
                        styles.primaryButton,
                        { backgroundColor: palette.ink, borderCurve: 'continuous' },
                      ]}
                      onPress={openDownloadModal}
                    >
                      <Text style={[styles.primaryButtonText, { color: palette.onInk }]}>
                        Download model
                      </Text>
                    </TouchableOpacity>
                    <Text style={[styles.gateModelId, { color: palette.muted }]}>
                      {MODEL_ID}
                    </Text>
                  </>
                )}
                {(phase === 'loading' || phase === 'checking') && (
                  <>
                    <Text style={[styles.gateTitle, { color: palette.ink }]}>
                      {phase === 'checking'
                        ? 'Checking the model'
                        : `Loading ${MODEL_LABEL}`}
                    </Text>
                    <View
                      style={[styles.progressTrack, { backgroundColor: palette.surface }]}
                    >
                      <View
                        style={[
                          styles.progressFill,
                          {
                            backgroundColor: palette.ember,
                            width: `${Math.round(loadProgress * 100)}%`,
                          },
                        ]}
                      />
                    </View>
                    <Text style={[styles.gateSub, { color: palette.muted }]}>
                      {phase === 'checking'
                        ? 'Looking for a local copy…'
                        : `${(loadProgress * 100).toFixed(0)}%`}
                    </Text>
                  </>
                )}
                {phase === 'error' && (
                  <>
                    <Text style={[styles.gateTitle, { color: palette.ink }]}>
                      The model did not load
                    </Text>
                    <Text style={[styles.gateSub, { color: palette.error }]}>
                      {loadError}
                    </Text>
                    <TouchableOpacity
                      style={[
                        styles.primaryButton,
                        { backgroundColor: palette.surface, borderCurve: 'continuous' },
                      ]}
                      onPress={prepareModel}
                    >
                      <Text style={[styles.primaryButtonText, { color: palette.ink }]}>
                        Try again
                      </Text>
                    </TouchableOpacity>
                  </>
                )}
              </View>
            )}
          </View>
        )}

        <View style={[styles.composer, { borderTopColor: palette.hairline }]}>
          <TextInput
            value={input}
            onChangeText={setInput}
            placeholder={phase === 'ready' ? 'Message' : 'Install the model to start'}
            placeholderTextColor={palette.muted}
            style={[
              styles.input,
              {
                backgroundColor: palette.surface,
                color: palette.ink,
                borderCurve: 'continuous',
              },
            ]}
            editable={phase === 'ready' && !isRunning}
            multiline
          />
          {isRunning ? (
            <TouchableOpacity
              accessibilityLabel="Stop"
              style={[styles.sendButton, { backgroundColor: palette.ink }]}
              onPress={() => LLM.stop()}
            >
              <View style={[styles.stopGlyph, { backgroundColor: palette.onInk }]} />
            </TouchableOpacity>
          ) : (
            <TouchableOpacity
              accessibilityLabel="Send"
              style={[
                styles.sendButton,
                { backgroundColor: palette.ink },
                (!input.trim() || phase !== 'ready') && styles.sendDisabled,
              ]}
              onPress={() => send(input)}
              disabled={!input.trim() || phase !== 'ready'}
            >
              <Text style={[styles.sendGlyph, { color: palette.onInk }]}>↑</Text>
            </TouchableOpacity>
          )}
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: 6,
    paddingBottom: 12,
    borderBottomWidth: 1,
  },
  headerTitle: {
    fontSize: 22,
    fontWeight: '700',
    letterSpacing: -0.4,
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginTop: 3,
  },
  statusDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  statusText: {
    fontSize: 12,
    fontWeight: '500',
  },
  newChatButton: {
    width: 34,
    height: 34,
    borderRadius: 17,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  newChatDisabled: {
    opacity: 0.35,
  },
  newChatGlyph: {
    fontSize: 20,
    fontWeight: '400',
    lineHeight: 22,
  },
  stage: {
    flex: 1,
  },
  greetingBlock: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: 24,
  },
  greeting: {
    fontSize: 30,
    fontWeight: '700',
    letterSpacing: -0.8,
  },
  greetingSub: {
    fontSize: 15,
    lineHeight: 21,
    marginTop: 10,
    maxWidth: 300,
  },
  suggestions: {
    paddingHorizontal: 16,
    paddingBottom: 8,
    gap: 8,
  },
  suggestionCard: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: 16,
    paddingHorizontal: 16,
    paddingVertical: 14,
    gap: 12,
  },
  suggestionText: {
    flex: 1,
    fontSize: 15,
  },
  suggestionChevron: {
    fontSize: 18,
    lineHeight: 20,
  },
  gate: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 32,
  },
  modelChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    borderWidth: 1,
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  chipDot: {
    width: 7,
    height: 7,
    borderRadius: 3.5,
  },
  modelChipText: {
    fontFamily: 'Menlo',
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 1,
  },
  gateTitle: {
    fontSize: 20,
    fontWeight: '700',
    letterSpacing: -0.3,
    marginTop: 20,
  },
  gateSub: {
    fontSize: 14,
    lineHeight: 20,
    marginTop: 8,
    marginBottom: 20,
    textAlign: 'center',
    maxWidth: 280,
  },
  gateModelId: {
    fontFamily: 'Menlo',
    fontSize: 11,
    marginTop: 16,
  },
  primaryButton: {
    borderRadius: 999,
    paddingHorizontal: 24,
    paddingVertical: 13,
  },
  primaryButtonText: {
    fontSize: 16,
    fontWeight: '600',
  },
  progressTrack: {
    width: 180,
    height: 4,
    borderRadius: 2,
    marginTop: 20,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 2,
  },
  threadContent: {
    paddingVertical: 12,
  },
  userBubble: {
    alignSelf: 'flex-end',
    borderRadius: 20,
    paddingHorizontal: 14,
    paddingVertical: 10,
    marginHorizontal: 16,
    marginVertical: 4,
    maxWidth: '78%',
  },
  userText: {
    fontSize: 16,
    lineHeight: 22,
  },
  assistantBlock: {
    paddingHorizontal: 20,
    paddingVertical: 8,
  },
  assistantText: {
    fontSize: 16,
    lineHeight: 24,
  },
  thinkingToggle: {
    fontSize: 13,
    fontWeight: '600',
    marginBottom: 6,
  },
  thinkingText: {
    fontSize: 13,
    lineHeight: 19,
    fontStyle: 'italic',
    marginBottom: 10,
  },
  noticeText: {
    fontSize: 13,
    paddingHorizontal: 20,
    paddingVertical: 6,
  },
  stepRow: {
    flexDirection: 'row',
    paddingHorizontal: 20,
  },
  railCol: {
    width: 24,
    alignItems: 'center',
  },
  railLine: {
    width: 1,
  },
  railLineTop: {
    height: 8,
  },
  railLineBottom: {
    flex: 1,
  },
  railNode: {
    width: 16,
    height: 16,
    borderRadius: 8,
    borderWidth: 1.5,
    alignItems: 'center',
    justifyContent: 'center',
  },
  railNodeMark: {
    fontSize: 8,
    fontWeight: '700',
    lineHeight: 10,
  },
  railPulse: {
    height: 16,
    width: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stepBody: {
    flex: 1,
    paddingLeft: 10,
    paddingTop: 6,
    paddingBottom: 8,
  },
  stepLabel: {
    fontSize: 14,
    fontWeight: '600',
    lineHeight: 18,
  },
  stepDetail: {
    borderRadius: 10,
    padding: 10,
    marginTop: 8,
    gap: 6,
  },
  stepDetailText: {
    fontFamily: 'Menlo',
    fontSize: 11,
    lineHeight: 16,
  },
  workingLabel: {
    fontSize: 14,
    paddingLeft: 10,
    paddingTop: 6,
  },
  composer: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 8,
    paddingHorizontal: 12,
    paddingTop: 8,
    paddingBottom: 4,
    borderTopWidth: 1,
  },
  input: {
    flex: 1,
    borderRadius: 22,
    paddingHorizontal: 16,
    paddingTop: 11,
    paddingBottom: 11,
    fontSize: 16,
    maxHeight: 120,
  },
  sendButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sendDisabled: {
    opacity: 0.3,
  },
  sendGlyph: {
    fontSize: 18,
    fontWeight: '700',
    lineHeight: 21,
  },
  stopGlyph: {
    width: 11,
    height: 11,
    borderRadius: 2.5,
  },
})
