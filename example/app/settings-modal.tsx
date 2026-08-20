import { router } from 'expo-router'
import { useState } from 'react'
import {
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  useColorScheme,
  View,
} from 'react-native'
import { LLM } from 'react-native-nitro-mlx'
import { SafeAreaView } from 'react-native-safe-area-context'
import { type BenchmarkResult, useBenchmark } from '../components/benchmark-context'

type NumericBenchmarkKey = {
  [K in keyof BenchmarkResult]: BenchmarkResult[K] extends number ? K : never
}[keyof BenchmarkResult]

function average(results: BenchmarkResult[], key: NumericBenchmarkKey): number {
  if (results.length === 0) return 0
  const sum = results.reduce((acc, r) => acc + r[key], 0)
  return sum / results.length
}

export default function SettingsModal() {
  const colorScheme = useColorScheme()
  const textColor = colorScheme === 'dark' ? 'white' : 'black'
  const bgColor = colorScheme === 'dark' ? 'black' : 'white'
  const cardBg = colorScheme === 'dark' ? '#1c1c1e' : '#f2f2f7'
  const { results, clearResults } = useBenchmark()
  const [systemPrompt, setSystemPrompt] = useState(LLM.systemPrompt)

  const avgTps = average(results, 'tokensPerSecond')
  const avgTtft = average(results, 'timeToFirstToken')
  const avgTokens = average(results, 'totalTokens')
  const avgTime = average(results, 'totalTime')
  const avgToolTime = average(results, 'toolExecutionTime')

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: bgColor }]}>
      <View style={styles.header}>
        <Text style={[styles.title, { color: textColor }]}>Benchmarks</Text>
        <TouchableOpacity onPress={() => router.back()} style={styles.closeButton}>
          <Text style={styles.closeButtonText}>Done</Text>
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.content} contentContainerStyle={styles.contentContainer}>
        <View style={[styles.card, { backgroundColor: cardBg }]}>
          <View style={[styles.cardHeader, { borderBottomColor: '#007AFF' }]}>
            <Text style={[styles.cardTitle, { color: textColor }]}>MLX</Text>
            <Text style={[styles.runCount, { color: '#007AFF' }]}>
              {results.length} runs
            </Text>
          </View>

          {results.length === 0 ? (
            <Text style={[styles.noData, { color: textColor }]}>
              No benchmark data yet. Start chatting to collect metrics.
            </Text>
          ) : (
            <View style={styles.metricsGrid}>
              <View style={styles.metric}>
                <Text style={[styles.metricValue, { color: '#007AFF' }]}>
                  {avgTps.toFixed(1)}
                </Text>
                <Text style={[styles.metricLabel, { color: textColor }]}>tokens/sec</Text>
              </View>
              <View style={styles.metric}>
                <Text style={[styles.metricValue, { color: '#007AFF' }]}>
                  {avgTtft.toFixed(0)}ms
                </Text>
                <Text style={[styles.metricLabel, { color: textColor }]}>
                  time to first token
                </Text>
              </View>
              <View style={styles.metric}>
                <Text style={[styles.metricValue, { color: '#007AFF' }]}>
                  {avgTokens.toFixed(0)}
                </Text>
                <Text style={[styles.metricLabel, { color: textColor }]}>avg tokens</Text>
              </View>
              <View style={styles.metric}>
                <Text style={[styles.metricValue, { color: '#007AFF' }]}>
                  {(avgTime / 1000).toFixed(1)}s
                </Text>
                <Text style={[styles.metricLabel, { color: textColor }]}>avg time</Text>
              </View>
              {avgToolTime > 0 && (
                <View style={styles.metric}>
                  <Text style={[styles.metricValue, { color: '#FF9500' }]}>
                    {(avgToolTime / 1000).toFixed(1)}s
                  </Text>
                  <Text style={[styles.metricLabel, { color: textColor }]}>
                    tool execution
                  </Text>
                </View>
              )}
            </View>
          )}
        </View>

        {results.length > 0 && (
          <TouchableOpacity style={styles.clearButton} onPress={clearResults}>
            <Text style={styles.clearButtonText}>Clear Results</Text>
          </TouchableOpacity>
        )}

        <View style={[styles.card, { backgroundColor: cardBg }]}>
          <Text style={[styles.cardTitle, { color: textColor, marginBottom: 12 }]}>
            System Prompt
          </Text>
          <TextInput
            style={[styles.promptInput, { color: textColor }]}
            value={systemPrompt}
            onChangeText={text => {
              setSystemPrompt(text)
              LLM.systemPrompt = text
            }}
            placeholder="Enter system prompt..."
            placeholderTextColor="#999"
            multiline
            numberOfLines={4}
          />
        </View>
      </ScrollView>
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#3333',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  closeButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  closeButtonText: {
    color: '#007AFF',
    fontSize: 16,
    fontWeight: '600',
  },
  content: {
    flex: 1,
  },
  contentContainer: {
    padding: 16,
    gap: 16,
  },
  card: {
    borderRadius: 12,
    padding: 16,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingBottom: 12,
    marginBottom: 12,
    borderBottomWidth: 2,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '600',
  },
  runCount: {
    fontSize: 14,
    fontWeight: '500',
  },
  noData: {
    fontSize: 14,
    opacity: 0.6,
    textAlign: 'center',
    paddingVertical: 20,
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 16,
  },
  metric: {
    width: '45%',
  },
  metricValue: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  metricLabel: {
    fontSize: 12,
    opacity: 0.6,
    marginTop: 2,
  },
  clearButton: {
    backgroundColor: '#FF3B30',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 8,
  },
  clearButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  promptInput: {
    backgroundColor: '#8881',
    borderRadius: 8,
    padding: 12,
    fontSize: 14,
    minHeight: 100,
    textAlignVertical: 'top',
  },
  promptNote: {
    fontSize: 12,
    opacity: 0.6,
    marginTop: 8,
    fontStyle: 'italic',
  },
})
