import { useFocusEffect } from 'expo-router'
import { useCallback, useRef, useState } from 'react'
import {
  ActivityIndicator,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  useColorScheme,
  View,
} from 'react-native'
import { cosineSimilarity, Embeddings } from 'react-native-nitro-mlx'
import { SafeAreaView } from 'react-native-safe-area-context'

const MODEL_ID = 'mlx-community/bge-small-en-v1.5-4bit'

type Status = 'idle' | 'loading' | 'ready' | 'embedding'

export default function EmbeddingsScreen() {
  const [status, setStatus] = useState<Status>('idle')
  const [textA, setTextA] = useState('')
  const [textB, setTextB] = useState('')
  const [similarity, setSimilarity] = useState<number | null>(null)
  const colorScheme = useColorScheme()
  const textColor = colorScheme === 'dark' ? 'white' : 'black'
  const bgColor = colorScheme === 'dark' ? 'black' : 'white'
  const isLoadingRef = useRef(false)

  const loadModel = useCallback(async () => {
    if (Embeddings.isLoaded) {
      setStatus('ready')
      return
    }
    if (isLoadingRef.current) return
    isLoadingRef.current = true
    setStatus('loading')
    try {
      await Embeddings.load(MODEL_ID)
      setStatus('ready')
    } catch (error) {
      console.error('Error loading Embeddings model:', error)
      setStatus('idle')
    } finally {
      isLoadingRef.current = false
    }
  }, [])

  useFocusEffect(
    useCallback(() => {
      loadModel()
      return () => {
        Embeddings.unload()
        setStatus('idle')
        setSimilarity(null)
        isLoadingRef.current = false
      }
    }, [loadModel]),
  )

  const handleCompare = async () => {
    if (!textA.trim() || !textB.trim() || status !== 'ready') return
    try {
      setStatus('embedding')
      setSimilarity(null)
      const [a, b] = await Embeddings.embedBatch([textA, textB])
      setSimilarity(cosineSimilarity(a, b))
    } catch (error) {
      console.error('Embeddings error:', error)
    } finally {
      setStatus('ready')
    }
  }

  const canCompare =
    status === 'ready' && textA.trim().length > 0 && textB.trim().length > 0

  const statusText = {
    idle: '',
    loading: 'Downloading & loading model...',
    ready: 'Ready',
    embedding: 'Embedding...',
  } satisfies Record<Status, string>

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
          { borderBottomColor: colorScheme === 'dark' ? '#333' : '#eee' },
        ]}
      >
        <Text style={[styles.headerTitle, { color: textColor }]}>MLX Embeddings</Text>
        <Text style={styles.statusBadge}>{statusText[status]}</Text>
      </View>

      <View style={styles.content}>
        <Text style={[styles.modelLabel, { color: textColor }]}>
          {MODEL_ID} · dim {Embeddings.dimension || '—'}
        </Text>

        <TextInput
          value={textA}
          onChangeText={setTextA}
          placeholder="First text..."
          placeholderTextColor="#999"
          style={[styles.textInput, { color: textColor }]}
          multiline
          editable={status === 'ready'}
        />

        <TextInput
          value={textB}
          onChangeText={setTextB}
          placeholder="Second text..."
          placeholderTextColor="#999"
          style={[styles.textInput, { color: textColor }]}
          multiline
          editable={status === 'ready'}
        />

        <TouchableOpacity
          style={[styles.compareButton, !canCompare && styles.compareButtonDisabled]}
          onPress={handleCompare}
          disabled={!canCompare}
        >
          {status === 'embedding' ? (
            <ActivityIndicator color="white" />
          ) : (
            <Text style={styles.compareButtonText}>Compare Similarity</Text>
          )}
        </TouchableOpacity>

        {similarity !== null && (
          <View style={styles.resultCard}>
            <Text style={[styles.resultLabel, { color: textColor }]}>
              Cosine similarity
            </Text>
            <Text style={[styles.resultValue, { color: textColor }]}>
              {similarity.toFixed(4)}
            </Text>
          </View>
        )}
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
  modelLabel: {
    fontSize: 12,
    opacity: 0.6,
  },
  textInput: {
    backgroundColor: '#c4c4c62f',
    borderRadius: 12,
    padding: 16,
    fontSize: 16,
    minHeight: 80,
    textAlignVertical: 'top',
  },
  compareButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  compareButtonDisabled: {
    opacity: 0.5,
  },
  compareButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: '600',
  },
  resultCard: {
    backgroundColor: '#007AFF15',
    borderRadius: 12,
    padding: 20,
    alignItems: 'center',
    gap: 8,
  },
  resultLabel: {
    fontSize: 13,
    opacity: 0.7,
  },
  resultValue: {
    fontSize: 36,
    fontWeight: '700',
    fontFamily: 'Menlo',
  },
})
