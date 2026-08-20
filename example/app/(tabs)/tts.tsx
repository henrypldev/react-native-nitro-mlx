import { useFocusEffect } from 'expo-router'
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  ActivityIndicator,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  useColorScheme,
  View,
} from 'react-native'
import { AudioContext } from 'react-native-audio-api'
import { MLXModel, TTS } from 'react-native-nitro-mlx'
import { SafeAreaView } from 'react-native-safe-area-context'

const MODEL_ID = MLXModel.PocketTTS

const VOICES = [
  'alba',
  'azelma',
  'cosette',
  'eponine',
  'fantine',
  'javert',
  'jean',
  'marius',
] as const

type Status = 'idle' | 'loading' | 'ready' | 'generating' | 'playing'

export default function TTSScreen() {
  const [status, setStatus] = useState<Status>('idle')
  const [text, setText] = useState('')
  const [voice, setVoice] = useState<string>(VOICES[0])
  const colorScheme = useColorScheme()
  const textColor = colorScheme === 'dark' ? 'white' : 'black'
  const bgColor = colorScheme === 'dark' ? 'black' : 'white'
  const isLoadingRef = useRef(false)
  const audioCtxRef = useRef<AudioContext | null>(null)

  const loadModel = useCallback(async () => {
    if (TTS.isLoaded) {
      setStatus('ready')
      return
    }
    if (isLoadingRef.current) return
    isLoadingRef.current = true
    setStatus('loading')
    try {
      await TTS.load(MODEL_ID)
      setStatus('ready')
    } catch (error) {
      console.error('Error loading TTS model:', error)
      setStatus('idle')
    } finally {
      isLoadingRef.current = false
    }
  }, [])

  useFocusEffect(
    useCallback(() => {
      loadModel()
      return () => {
        TTS.unload()
        setStatus('idle')
        isLoadingRef.current = false
      }
    }, [loadModel]),
  )

  useEffect(() => {
    return () => {
      audioCtxRef.current?.close()
    }
  }, [])

  const handleGenerate = async () => {
    if (!text.trim() || status !== 'ready') return

    try {
      setStatus('generating')
      const pcm = await TTS.generate(text, { voice })

      setStatus('playing')
      const sampleRate = TTS.sampleRate
      const ctx = new AudioContext({ sampleRate })
      audioCtxRef.current = ctx
      const samples = new Float32Array(pcm)
      const buffer = ctx.createBuffer(1, samples.length, sampleRate)
      buffer.copyToChannel(samples, 0)
      const source = ctx.createBufferSource()
      source.buffer = buffer
      source.connect(ctx.destination)
      source.start()

      const duration = samples.length / sampleRate
      setTimeout(() => {
        setStatus('ready')
      }, duration * 1000)
    } catch (error) {
      console.error('TTS error:', error)
      setStatus('ready')
    }
  }

  const statusText = {
    idle: '',
    loading: 'Downloading & loading model...',
    ready: 'Ready',
    generating: 'Generating audio...',
    playing: 'Playing...',
  } satisfies Record<Status, string>

  const canGenerate = status === 'ready' && text.trim().length > 0

  if (status === 'idle' || status === 'loading') {
    return (
      <SafeAreaView style={[styles.centered, { backgroundColor: bgColor }]}>
        <ActivityIndicator size="large" />
        <Text style={[styles.statusLabel, { color: textColor }]}>
          {statusText[status] || 'Preparing...'}
        </Text>
      </SafeAreaView>
    )
  }

  return (
    <SafeAreaView
      style={[styles.container, { backgroundColor: bgColor }]}
      edges={['bottom', 'top']}
    >
      <View
        style={[
          styles.header,
          {
            borderBottomColor: colorScheme === 'dark' ? '#333' : '#eee',
          },
        ]}
      >
        <Text style={[styles.headerTitle, { color: textColor }]}>MLX TTS</Text>
        <Text style={[styles.statusBadge]}>{statusText[status]}</Text>
      </View>

      <View style={styles.content}>
        <TextInput
          value={text}
          onChangeText={setText}
          placeholder="Enter text to speak..."
          placeholderTextColor="#999"
          style={[styles.textInput, { color: textColor }]}
          multiline
          editable={status === 'ready'}
        />

        <View style={styles.voiceContainer}>
          <Text style={[styles.voiceLabel, { color: textColor }]}>Voice</Text>
          <View style={styles.voiceList}>
            {VOICES.map(v => (
              <TouchableOpacity
                key={v}
                style={[styles.voiceChip, v === voice && styles.voiceChipSelected]}
                onPress={() => setVoice(v)}
              >
                <Text
                  style={[
                    styles.voiceChipText,
                    v === voice && styles.voiceChipTextSelected,
                  ]}
                >
                  {v}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        <TouchableOpacity
          style={[styles.generateButton, !canGenerate && styles.generateButtonDisabled]}
          onPress={handleGenerate}
          disabled={!canGenerate}
        >
          {status === 'generating' || status === 'playing' ? (
            <ActivityIndicator color="white" />
          ) : (
            <Text style={styles.generateButtonText}>Generate & Play</Text>
          )}
        </TouchableOpacity>
      </View>
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
    padding: 16,
    borderBottomWidth: 1,
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
  },
  statusBadge: {
    fontSize: 13,
    color: '#007AFF',
    fontWeight: '500',
  },
  statusLabel: {
    marginTop: 12,
    fontSize: 16,
  },
  content: {
    flex: 1,
    padding: 20,
    gap: 16,
  },
  textInput: {
    backgroundColor: '#c4c4c62f',
    borderRadius: 12,
    padding: 16,
    fontSize: 16,
    minHeight: 120,
    textAlignVertical: 'top',
  },
  voiceContainer: {
    gap: 8,
  },
  voiceLabel: {
    fontSize: 14,
    fontWeight: '600',
  },
  voiceList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  voiceChip: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#c4c4c62f',
  },
  voiceChipSelected: {
    backgroundColor: '#007AFF',
  },
  voiceChipText: {
    fontSize: 13,
    fontWeight: '500',
    color: '#999',
  },
  voiceChipTextSelected: {
    color: 'white',
  },
  generateButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  generateButtonDisabled: {
    opacity: 0.5,
  },
  generateButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: '600',
  },
})
