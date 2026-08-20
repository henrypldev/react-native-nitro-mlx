import * as Crypto from 'expo-crypto'
import { router, useFocusEffect } from 'expo-router'
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  ActivityIndicator,
  LayoutAnimation,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  useColorScheme,
  View,
} from 'react-native'
import { KeyboardAvoidingView } from 'react-native-keyboard-controller'
import {
  LLM,
  type LLMMessage,
  type LLMToolCall,
  type LLMTurnOutcome,
  MLXModel,
  ModelManager,
  type StreamEvent,
  type ToolSchema,
} from 'react-native-nitro-mlx'
import { SafeAreaView } from 'react-native-safe-area-context'

const MODEL_ID = MLXModel.Qwen3_1_7B_4bit
const DEFAULT_MAX_STEPS = 6

const PRESET_GOALS = [
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

// --- Mock "personal assistant" toolbox -------------------------------------
// Module-level state so a multi-step goal visibly accumulates effects across
// the agent's tool calls, and the on-screen panel can mirror it.

type TimerEntry = { id: string; minutes: number; label: string; createdAt: number }
type SentMessageEntry = { id: string; to: string; body: string; createdAt: number }

let mockTimers: TimerEntry[] = []
let mockSentMessages: SentMessageEntry[] = []

function resetMockState(): void {
  mockTimers = []
  mockSentMessages = []
}

function executeTool(
  name: string,
  args: Record<string, unknown>,
): { content: string; isError: boolean } {
  switch (name) {
    case 'get_current_time':
      return { content: new Date().toLocaleString(), isError: false }

    case 'check_calendar':
      return { content: TODAY_EVENTS.join('\n'), isError: false }

    case 'set_timer': {
      const minutes = Number(args.minutes)
      if (!Number.isFinite(minutes) || minutes <= 0) {
        return {
          content: `Invalid duration: "minutes" must be a positive number, got ${JSON.stringify(args.minutes)}`,
          isError: true,
        }
      }
      const label =
        typeof args.label === 'string' && args.label.trim().length > 0
          ? args.label.trim()
          : 'Timer'
      mockTimers = [
        ...mockTimers,
        { id: Crypto.randomUUID(), minutes, label, createdAt: Date.now() },
      ]
      return { content: `Timer set for ${minutes} minute(s): "${label}"`, isError: false }
    }

    case 'send_message': {
      const to = typeof args.to === 'string' ? args.to.trim() : ''
      const body = typeof args.body === 'string' ? args.body.trim() : ''
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
// turns them into an agent.

type AgentStep =
  | { kind: 'tool'; call: LLMToolCall; result: string; isError: boolean }
  | { kind: 'answer'; text: string }
  | { kind: 'aborted'; reason: string } // stopped / failed / length / max-steps

async function runAgent(
  goal: string,
  opts: {
    onStep: (step: AgentStep, turn: LLMTurnOutcome) => void
    onToken?: (token: string) => void
    maxSteps?: number
  },
): Promise<void> {
  const maxSteps = opts.maxSteps ?? DEFAULT_MAX_STEPS
  const ctx = await LLM.createContext({
    instructions: AGENT_INSTRUCTIONS,
    tools: TOOL_SCHEMAS,
  })

  const handleEvent = (event: StreamEvent) => {
    if (event.type === 'token') {
      opts.onToken?.(event.token)
    }
  }

  try {
    let messages: LLMMessage[] = [{ role: 'user', content: goal }]
    let lastTurn: LLMTurnOutcome | null = null

    for (let step = 0; step < maxSteps; step += 1) {
      const turn = await LLM.runTurn({ contextId: ctx.id, messages }, handleEvent)
      lastTurn = turn

      if (
        turn.finishReason === 'stopped' ||
        turn.finishReason === 'failed' ||
        turn.finishReason === 'length'
      ) {
        opts.onStep({ kind: 'aborted', reason: turn.error ?? turn.finishReason }, turn)
        return
      }

      if (turn.toolCalls.length === 0) {
        opts.onStep({ kind: 'answer', text: turn.content }, turn)
        return
      }

      const results: LLMMessage[] = []
      for (const call of turn.toolCalls) {
        const { content, isError } = executeTool(call.name, call.arguments)
        opts.onStep({ kind: 'tool', call, result: content, isError }, turn)
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

    if (lastTurn) {
      opts.onStep({ kind: 'aborted', reason: 'max-steps' }, lastTurn)
    }
  } finally {
    ctx.release()
  }
}

// --- Screen -------------------------------------------------------------

type TranscriptEntry =
  | { id: string; kind: 'goal'; text: string }
  | {
      id: string
      kind: 'tool'
      name: string
      args: Record<string, unknown>
      result: string
      isError: boolean
    }
  | { id: string; kind: 'assistant'; text: string }
  | { id: string; kind: 'aborted'; reason: string }

type LastTurnStats = {
  promptTokens: number
  completionTokens: number
  cachedPromptTokens?: number
  tokensPerSecond: number
}

const ToolStepBlock = ({
  entry,
}: {
  entry: Extract<TranscriptEntry, { kind: 'tool' }>
}) => {
  const [expanded, setExpanded] = useState(false)
  const colorScheme = useColorScheme()

  const toggleExpanded = () => {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut)
    setExpanded(!expanded)
  }

  const toolDisplayName = entry.name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase())

  return (
    <TouchableOpacity
      onPress={toggleExpanded}
      style={[styles.toolCallBlock, entry.isError && styles.toolCallBlockError]}
    >
      <View style={styles.toolCallHeader}>
        <Text style={styles.toolCallIcon}>{entry.isError ? '⚠️' : '🔧'}</Text>
        <Text style={[styles.toolCallLabel, entry.isError && styles.toolCallLabelError]}>
          {toolDisplayName}
        </Text>
        <Text style={entry.isError ? styles.toolCallErrorMark : styles.toolCallComplete}>
          {entry.isError ? '✕' : '✓'}
        </Text>
      </View>
      {expanded && (
        <>
          <Text
            style={[
              styles.toolCallArgs,
              { color: colorScheme === 'dark' ? '#aaa' : '#666' },
            ]}
          >
            args: {JSON.stringify(entry.args, null, 2)}
          </Text>
          <Text
            style={[
              styles.toolCallArgs,
              { color: colorScheme === 'dark' ? '#aaa' : '#666' },
            ]}
          >
            result: {entry.result}
          </Text>
        </>
      )}
    </TouchableOpacity>
  )
}

const TranscriptItem = ({
  entry,
  textColor,
}: {
  entry: TranscriptEntry
  textColor: string
}) => {
  if (entry.kind === 'goal') {
    return (
      <View style={styles.userMessage}>
        <Text style={[styles.messageText, { color: 'white' }]}>{entry.text}</Text>
      </View>
    )
  }

  if (entry.kind === 'tool') {
    return <ToolStepBlock entry={entry} />
  }

  if (entry.kind === 'assistant') {
    return (
      <View style={styles.message}>
        <Text style={[styles.messageText, { color: textColor }]}>
          {entry.text.trim()}
        </Text>
      </View>
    )
  }

  return (
    <View style={styles.abortedBlock}>
      <Text style={styles.abortedText}>Agent stopped: {entry.reason}</Text>
    </View>
  )
}

const StatTile = ({ label, value }: { label: string; value: string }) => {
  const colorScheme = useColorScheme()
  const textColor = colorScheme === 'dark' ? 'white' : 'black'
  return (
    <View style={styles.statTile}>
      <Text style={[styles.statValue, { color: textColor }]}>{value}</Text>
      <Text
        style={[styles.statLabel, { color: colorScheme === 'dark' ? '#aaa' : '#666' }]}
      >
        {label}
      </Text>
    </View>
  )
}

export default function AgentLabScreen() {
  const [isChecking, setIsChecking] = useState(true)
  const [isDownloaded, setIsDownloaded] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [loadProgress, setLoadProgress] = useState(0)
  const [isReady, setIsReady] = useState(false)
  const [goalInput, setGoalInput] = useState('')
  const [isRunning, setIsRunning] = useState(false)
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([])
  const [streamingText, setStreamingText] = useState('')
  const [stepCount, setStepCount] = useState(0)
  const [lastTurnStats, setLastTurnStats] = useState<LastTurnStats | null>(null)
  const [timers, setTimers] = useState<TimerEntry[]>([])
  const [sentMessages, setSentMessages] = useState<SentMessageEntry[]>([])

  const colorScheme = useColorScheme()
  const textColor = colorScheme === 'dark' ? 'white' : 'black'
  const bgColor = colorScheme === 'dark' ? 'black' : 'white'
  const scrollRef = useRef<ScrollView>(null)
  const isLoadingRef = useRef(false)

  const checkDownloaded = useCallback(async () => {
    setIsChecking(true)
    try {
      const downloaded = await ModelManager.isDownloaded(MODEL_ID)
      setIsDownloaded(downloaded)
    } catch (error) {
      console.error('Error checking download:', error)
    } finally {
      setIsChecking(false)
    }
  }, [])

  useFocusEffect(
    useCallback(() => {
      checkDownloaded()
      return () => {
        LLM.unload()
        setIsReady(false)
        isLoadingRef.current = false
      }
    }, [checkDownloaded]),
  )

  useEffect(() => {
    if (!isDownloaded || isReady || isLoadingRef.current) return

    const loadModel = async () => {
      isLoadingRef.current = true
      setIsLoading(true)
      setLoadProgress(0)
      try {
        await LLM.load(MODEL_ID, { onProgress: setLoadProgress })
        setIsReady(true)
      } catch (error) {
        console.error('Error loading model:', error)
      } finally {
        setIsLoading(false)
        isLoadingRef.current = false
      }
    }

    loadModel()
  }, [isDownloaded, isReady])

  const openDownloadModal = () => {
    router.push('/download-modal')
  }

  const handleResetMockState = useCallback(() => {
    if (isRunning) return
    resetMockState()
    setTimers([])
    setSentMessages([])
  }, [isRunning])

  const startRun = useCallback(
    async (goalText: string) => {
      const trimmed = goalText.trim()
      if (!trimmed || isRunning || !isReady) return

      setTranscript(prev => [
        ...prev,
        { id: Crypto.randomUUID(), kind: 'goal', text: trimmed },
      ])
      setGoalInput('')
      setStreamingText('')
      setStepCount(0)
      setLastTurnStats(null)
      setIsRunning(true)

      try {
        let steps = 0
        await runAgent(trimmed, {
          maxSteps: DEFAULT_MAX_STEPS,
          onToken: token => setStreamingText(prev => prev + token),
          onStep: (step, turn) => {
            steps += 1
            setStepCount(steps)
            setLastTurnStats({
              promptTokens: turn.usage.promptTokens,
              completionTokens: turn.usage.completionTokens,
              cachedPromptTokens: turn.usage.cachedPromptTokens,
              tokensPerSecond: turn.stats.tokensPerSecond,
            })
            setTimers(mockTimers)
            setSentMessages(mockSentMessages)
            setStreamingText('')

            if (step.kind === 'tool') {
              setTranscript(prev => [
                ...prev,
                {
                  id: `${step.call.id}-${steps}`,
                  kind: 'tool',
                  name: step.call.name,
                  args: step.call.arguments,
                  result: step.result,
                  isError: step.isError,
                },
              ])
            } else if (step.kind === 'answer') {
              setTranscript(prev => [
                ...prev,
                { id: Crypto.randomUUID(), kind: 'assistant', text: step.text },
              ])
            } else {
              setTranscript(prev => [
                ...prev,
                { id: Crypto.randomUUID(), kind: 'aborted', reason: step.reason },
              ])
            }
          },
        })
      } catch (error) {
        setTranscript(prev => [
          ...prev,
          {
            id: Crypto.randomUUID(),
            kind: 'aborted',
            reason: error instanceof Error ? error.message : String(error),
          },
        ])
      } finally {
        setIsRunning(false)
        setStreamingText('')
      }
    },
    [isRunning, isReady],
  )

  if (isChecking) {
    return (
      <SafeAreaView style={[styles.centered, { backgroundColor: bgColor }]}>
        <ActivityIndicator size="large" />
        <Text style={[styles.statusText, { color: textColor }]}>Checking model...</Text>
      </SafeAreaView>
    )
  }

  if (!isDownloaded) {
    return (
      <SafeAreaView style={[styles.centered, { backgroundColor: bgColor }]}>
        <Text style={[styles.title, { color: textColor }]}>Agent Lab</Text>
        <Text style={[styles.subtitle, { color: textColor }]}>
          Download the model to get started
        </Text>
        <TouchableOpacity style={styles.downloadButton} onPress={openDownloadModal}>
          <Text style={styles.downloadButtonText}>Download Model</Text>
        </TouchableOpacity>
        <Text style={[styles.modelId, { color: textColor }]}>{MODEL_ID}</Text>
      </SafeAreaView>
    )
  }

  if (isLoading) {
    return (
      <SafeAreaView style={[styles.centered, { backgroundColor: bgColor }]}>
        <ActivityIndicator size="large" />
        <Text style={[styles.statusText, { color: textColor }]}>
          Loading model... {(loadProgress * 100).toFixed(0)}%
        </Text>
      </SafeAreaView>
    )
  }

  return (
    <SafeAreaView
      style={[styles.container, { backgroundColor: bgColor }]}
      edges={['bottom', 'top']}
    >
      <KeyboardAvoidingView
        style={[styles.container, { backgroundColor: bgColor }]}
        behavior="padding"
        keyboardVerticalOffset={Platform.select({ ios: 0, default: 0 })}
      >
        <View
          style={[
            styles.header,
            { borderBottomColor: colorScheme === 'dark' ? '#333' : '#eee' },
          ]}
        >
          <Text style={[styles.headerTitle, { color: textColor }]}>Agent Lab</Text>
          <Text
            style={[
              styles.headerSubtitle,
              { color: colorScheme === 'dark' ? '#aaa' : '#666' },
            ]}
          >
            A caller-owned loop over LLM.runTurn
          </Text>
        </View>

        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.chipsRow}
          style={styles.chipsRail}
        >
          {PRESET_GOALS.map(goal => (
            <TouchableOpacity
              key={goal}
              style={[styles.chip, isRunning && styles.chipDisabled]}
              onPress={() => startRun(goal)}
              disabled={isRunning}
            >
              <Text style={styles.chipText}>{goal}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>

        <ScrollView
          ref={scrollRef}
          style={styles.transcript}
          contentContainerStyle={styles.transcriptContent}
          onContentSizeChange={() => scrollRef.current?.scrollToEnd({ animated: true })}
        >
          {transcript.map(entry => (
            <TranscriptItem key={entry.id} entry={entry} textColor={textColor} />
          ))}
          {isRunning && streamingText.trim() ? (
            <View style={styles.message}>
              <Text style={[styles.messageText, { color: textColor }]}>
                {streamingText}
              </Text>
            </View>
          ) : null}
          {isRunning ? (
            <View style={styles.runningRow}>
              <ActivityIndicator size="small" color="#007AFF" />
              <Text style={[styles.runningText, { color: textColor }]}>
                Agent is working…
              </Text>
            </View>
          ) : null}
        </ScrollView>

        <View
          style={[
            styles.statsRow,
            { borderTopColor: colorScheme === 'dark' ? '#333' : '#eee' },
          ]}
        >
          <StatTile label="Steps" value={String(stepCount)} />
          <StatTile
            label="Prompt"
            value={lastTurnStats ? String(lastTurnStats.promptTokens) : '–'}
          />
          <StatTile
            label="Cached"
            value={
              lastTurnStats?.cachedPromptTokens != null
                ? String(lastTurnStats.cachedPromptTokens)
                : '–'
            }
          />
          <StatTile
            label="Completion"
            value={lastTurnStats ? String(lastTurnStats.completionTokens) : '–'}
          />
          <StatTile
            label="Tok/s"
            value={lastTurnStats ? lastTurnStats.tokensPerSecond.toFixed(1) : '–'}
          />
        </View>

        <View
          style={[
            styles.mockPanel,
            {
              backgroundColor: colorScheme === 'dark' ? '#0f172a' : '#eef4ff',
              borderTopColor: colorScheme === 'dark' ? '#1e293b' : '#dbe7ff',
            },
          ]}
        >
          <View style={styles.mockPanelHeader}>
            <Text style={[styles.mockPanelTitle, { color: textColor }]}>
              Mock Assistant State
            </Text>
            <TouchableOpacity onPress={handleResetMockState} disabled={isRunning}>
              <Text
                style={[
                  styles.mockPanelReset,
                  isRunning && styles.mockPanelResetDisabled,
                ]}
              >
                Reset
              </Text>
            </TouchableOpacity>
          </View>
          <Text style={[styles.mockPanelLine, { color: textColor }]}>
            Timers:{' '}
            {timers.length === 0
              ? 'none'
              : timers.map(t => `${t.minutes}m "${t.label}"`).join(', ')}
          </Text>
          <Text style={[styles.mockPanelLine, { color: textColor }]}>
            Messages:{' '}
            {sentMessages.length === 0
              ? 'none'
              : sentMessages.map(m => `to ${m.to}`).join(', ')}
          </Text>
        </View>

        <View style={styles.inputContainer}>
          <TextInput
            value={goalInput}
            onChangeText={setGoalInput}
            placeholder="Type a goal..."
            placeholderTextColor="#999"
            style={[styles.input, { color: textColor }]}
            editable={!isRunning}
            onSubmitEditing={() => startRun(goalInput)}
            returnKeyType="send"
          />
          {isRunning ? (
            <TouchableOpacity style={styles.stopButton} onPress={() => LLM.stop()}>
              <Text style={styles.stopButtonText}>Stop</Text>
            </TouchableOpacity>
          ) : (
            <TouchableOpacity
              style={[styles.runButton, !goalInput.trim() && styles.runButtonDisabled]}
              onPress={() => startRun(goalInput)}
              disabled={!goalInput.trim()}
            >
              <Text style={styles.runButtonText}>Run</Text>
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
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  header: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    gap: 4,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
  },
  headerSubtitle: {
    fontSize: 12,
  },
  chipsRail: {
    flexGrow: 0,
  },
  chipsRow: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    flexDirection: 'row',
    gap: 8,
  },
  chip: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 16,
    backgroundColor: '#007AFF15',
    borderWidth: 1,
    borderColor: '#007AFF',
    maxWidth: 220,
  },
  chipDisabled: {
    opacity: 0.4,
  },
  chipText: {
    color: '#007AFF',
    fontSize: 12,
    fontWeight: '600',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    opacity: 0.7,
    marginBottom: 24,
    textAlign: 'center',
  },
  statusText: {
    marginTop: 12,
    fontSize: 16,
  },
  modelId: {
    marginTop: 16,
    fontSize: 12,
    opacity: 0.5,
  },
  downloadButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 32,
    paddingVertical: 16,
    borderRadius: 12,
  },
  downloadButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: '600',
  },
  transcript: {
    flex: 1,
  },
  transcriptContent: {
    paddingVertical: 8,
  },
  messageText: {
    fontSize: 16,
    lineHeight: 22,
  },
  message: {
    padding: 16,
    paddingHorizontal: 20,
  },
  userMessage: {
    padding: 12,
    paddingHorizontal: 16,
    backgroundColor: '#007AFF',
    alignSelf: 'flex-end',
    borderRadius: 16,
    marginRight: 12,
    marginVertical: 4,
    maxWidth: '80%',
  },
  runningRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 20,
    paddingVertical: 8,
  },
  runningText: {
    fontSize: 13,
    fontStyle: 'italic',
    opacity: 0.7,
  },
  abortedBlock: {
    marginHorizontal: 16,
    marginVertical: 6,
    padding: 10,
    borderRadius: 8,
    backgroundColor: '#FF950020',
    borderLeftWidth: 3,
    borderLeftColor: '#FF9500',
  },
  abortedText: {
    color: '#FF9500',
    fontSize: 13,
    fontWeight: '600',
  },
  toolCallBlock: {
    backgroundColor: '#007AFF15',
    borderRadius: 8,
    padding: 10,
    marginHorizontal: 16,
    marginVertical: 4,
    borderLeftWidth: 3,
    borderLeftColor: '#007AFF',
  },
  toolCallBlockError: {
    backgroundColor: '#FF3B3015',
    borderLeftColor: '#FF3B30',
  },
  toolCallHeader: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  toolCallIcon: {
    fontSize: 14,
    marginRight: 6,
  },
  toolCallLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#007AFF',
    flex: 1,
  },
  toolCallLabelError: {
    color: '#FF3B30',
  },
  toolCallComplete: {
    color: '#34C759',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
  toolCallErrorMark: {
    color: '#FF3B30',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
  toolCallArgs: {
    fontSize: 11,
    fontFamily: 'Menlo',
    marginTop: 8,
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingVertical: 10,
    borderTopWidth: 1,
  },
  statTile: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: 15,
    fontWeight: '700',
  },
  statLabel: {
    fontSize: 10,
    marginTop: 2,
  },
  mockPanel: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderTopWidth: 1,
    gap: 4,
  },
  mockPanelHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  mockPanelTitle: {
    fontSize: 13,
    fontWeight: '700',
  },
  mockPanelReset: {
    color: '#FF3B30',
    fontSize: 12,
    fontWeight: '600',
  },
  mockPanelResetDisabled: {
    opacity: 0.4,
  },
  mockPanelLine: {
    fontSize: 12,
    opacity: 0.8,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    gap: 8,
  },
  input: {
    flex: 1,
    backgroundColor: '#c4c4c62f',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 10,
    fontSize: 16,
  },
  runButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
  },
  runButtonDisabled: {
    opacity: 0.5,
  },
  runButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  stopButton: {
    backgroundColor: '#FF3B30',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
  },
  stopButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
})
