import { Stack } from 'expo-router'
import 'react-native-reanimated'
import { KeyboardProvider } from 'react-native-keyboard-controller'
import 'expo-dev-client'

export { ErrorBoundary } from 'expo-router'

export const unstable_settings = {
  initialRouteName: '(tabs)',
}

export default function RootLayout() {
  return (
    <KeyboardProvider>
      <Stack>
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        <Stack.Screen
          name="download-modal"
          options={{
            presentation: 'modal',
            headerShown: false,
          }}
        />
      </Stack>
    </KeyboardProvider>
  )
}
