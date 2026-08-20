import { useState } from 'react'
import { Button, ScrollView, Text } from 'react-native'
import { LLM, MLXModel } from 'react-native-nitro-mlx'

const MODEL = MLXModel.Qwen3_1_7B_4bit
const PROMPT = 'List three facts about the Moon.'

async function timed<T>(label: string, fn: () => Promise<T>): Promise<string> {
  const start = performance.now()
  await fn()
  return `${label}: ${(performance.now() - start).toFixed(0)} ms`
}

export default function Benchmark() {
  const [lines, setLines] = useState<string[]>([])
  const log = (line: string) => setLines(prev => [...prev, line])

  const run = async () => {
    setLines([])
    log(await timed('initial load', () => LLM.load(MODEL)))
    log(await timed('repeat load (same id, must be ~0)', () => LLM.load(MODEL)))
    log(`loadedModelId: ${LLM.loadedModelId}`)

    // Cold: unmanaged path builds a fresh session per turn.
    await LLM.load(MODEL)
    for (let i = 1; i <= 3; i++) {
      log(await timed(`cold turn ${i}`, () => LLM.generate(PROMPT)))
    }

    // Warm: managed path retains its session across turns.
    await LLM.load(MODEL, { manageHistory: true })
    for (let i = 1; i <= 3; i++) {
      log(await timed(`warm turn ${i} (history grows)`, () => LLM.generate(PROMPT)))
    }
    log('done — record these numbers in the PR')
  }

  return (
    <ScrollView contentContainerStyle={{ padding: 16, gap: 8 }}>
      <Button title="Run baseline" onPress={run} />
      {lines.map(line => (
        <Text key={line}>{line}</Text>
      ))}
    </ScrollView>
  )
}
