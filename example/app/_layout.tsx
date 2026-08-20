import { Stack } from 'expo-router'
import 'react-native-reanimated'
import { KeyboardProvider } from 'react-native-keyboard-controller'
import 'expo-dev-client'
import { BenchmarkProvider } from '../components/benchmark-context'

export { ErrorBoundary } from 'expo-router'

export const unstable_settings = {
  initialRouteName: '(tabs)',
}

export default function RootLayout() {
  return (
    <BenchmarkProvider>
      <KeyboardProvider>
        <Stack>
          <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
          <Stack.Screen
            name="benchmark"
            options={{
              title: 'Baseline Benchmark',
              headerShown: true,
            }}
          />
          <Stack.Screen
            name="agent-lab"
            options={{
              title: 'Agent Lab',
              headerShown: true,
            }}
          />
          <Stack.Screen
            name="download-modal"
            options={{
              presentation: 'modal',
              headerShown: false,
            }}
          />
          <Stack.Screen
            name="settings-modal"
            options={{
              presentation: 'modal',
              headerShown: false,
            }}
          />
        </Stack>
      </KeyboardProvider>
    </BenchmarkProvider>
  )
}
