import { router, useLocalSearchParams } from 'expo-router'
import { useEffect, useState } from 'react'
import { ActivityIndicator, StyleSheet, Text, useColorScheme, View } from 'react-native'
import { MLXModel, ModelManager } from 'react-native-nitro-mlx'

const DEFAULT_MODEL_ID = MLXModel.Qwen3_1_7B_4bit

export default function DownloadModal() {
  const { modelId } = useLocalSearchParams<{ modelId?: string }>()
  const MODEL_ID = Object.values(MLXModel).find(m => m === modelId) ?? DEFAULT_MODEL_ID
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState('Starting download...')
  const colorScheme = useColorScheme()
  const textColor = colorScheme === 'dark' ? 'white' : 'black'
  const bgColor = colorScheme === 'dark' ? 'black' : 'white'

  useEffect(() => {
    const downloadModel = async () => {
      try {
        setStatus('Downloading model...')
        await ModelManager.download(MODEL_ID, p => {
          // Byte-level progress fires very frequently during large downloads;
          // only re-render on visible changes (>=0.5%) or on completion.
          setProgress(prev => (p >= 1 || p - prev >= 0.005 ? p : prev))
        })
        setStatus('Download complete!')
        setTimeout(() => {
          router.back()
        }, 500)
      } catch (error) {
        setStatus(`Error: ${error}`)
      }
    }

    downloadModel()
  }, [MODEL_ID])

  return (
    <View style={[styles.container, { backgroundColor: bgColor }]}>
      <View style={styles.content}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={[styles.title, { color: textColor }]}>Downloading Model</Text>
        <Text style={[styles.modelName, { color: textColor }]}>{MODEL_ID}</Text>

        <View style={styles.progressContainer}>
          <View style={[styles.progressBar, { width: `${progress * 100}%` }]} />
        </View>

        <Text style={[styles.progressText, { color: textColor }]}>
          {(progress * 100).toFixed(1)}%
        </Text>

        <Text style={[styles.status, { color: textColor }]}>{status}</Text>
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  content: {
    alignItems: 'center',
    width: '100%',
    maxWidth: 300,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    marginTop: 20,
    marginBottom: 8,
  },
  modelName: {
    fontSize: 12,
    opacity: 0.7,
    marginBottom: 24,
  },
  progressContainer: {
    width: '100%',
    height: 8,
    backgroundColor: '#E0E0E0',
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#007AFF',
    borderRadius: 4,
  },
  progressText: {
    fontSize: 24,
    fontWeight: 'bold',
    marginTop: 16,
  },
  status: {
    fontSize: 14,
    marginTop: 8,
    opacity: 0.7,
  },
})
