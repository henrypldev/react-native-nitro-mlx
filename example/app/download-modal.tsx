import { router, useLocalSearchParams } from 'expo-router'
import { useEffect, useState } from 'react'
import { StyleSheet, Text, View } from 'react-native'
import { MLXModel, ModelManager } from 'react-native-nitro-mlx'
import { usePalette } from '../constants/theme'

const DEFAULT_MODEL_ID = MLXModel.Qwen3_1_7B_4bit

export default function DownloadModal() {
  const { modelId } = useLocalSearchParams<{ modelId?: string }>()
  const MODEL_ID = Object.values(MLXModel).find(m => m === modelId) ?? DEFAULT_MODEL_ID
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState('Starting download…')
  const palette = usePalette()

  useEffect(() => {
    const downloadModel = async () => {
      try {
        setStatus('Downloading…')
        await ModelManager.download(MODEL_ID, p => {
          // Byte-level progress fires very frequently during large downloads;
          // only re-render on visible changes (>=0.5%) or on completion.
          setProgress(prev => (p >= 1 || p - prev >= 0.005 ? p : prev))
        })
        setStatus('Done')
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
    <View style={[styles.container, { backgroundColor: palette.bg }]}>
      <View style={styles.content}>
        <Text style={[styles.progressText, { color: palette.ink }]}>
          {(progress * 100).toFixed(1)}%
        </Text>

        <View style={[styles.progressTrack, { backgroundColor: palette.surface }]}>
          <View
            style={[
              styles.progressFill,
              { backgroundColor: palette.ember, width: `${progress * 100}%` },
            ]}
          />
        </View>

        <Text style={[styles.status, { color: palette.muted }]}>{status}</Text>
        <Text style={[styles.modelName, { color: palette.muted }]}>{MODEL_ID}</Text>
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
  progressText: {
    fontSize: 40,
    fontWeight: '700',
    letterSpacing: -1,
    fontVariant: ['tabular-nums'],
    marginBottom: 24,
  },
  progressTrack: {
    width: '100%',
    height: 5,
    borderRadius: 2.5,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 2.5,
  },
  status: {
    fontSize: 14,
    marginTop: 16,
  },
  modelName: {
    fontFamily: 'Menlo',
    fontSize: 11,
    marginTop: 8,
  },
})
