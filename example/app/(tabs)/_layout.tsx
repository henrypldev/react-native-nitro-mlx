import { NativeTabs } from 'expo-router/unstable-native-tabs'
import { DynamicColorIOS } from 'react-native'

const adaptiveTintColor =
  process.env.EXPO_OS === 'ios'
    ? DynamicColorIOS({
        light: '#0f172a',
        dark: '#f8fafc',
      })
    : '#2563eb'

export default function TabsLayout() {
  return (
    <NativeTabs
      minimizeBehavior="onScrollDown"
      tintColor={adaptiveTintColor}
      labelStyle={{
        color: adaptiveTintColor,
      }}
    >
      <NativeTabs.Trigger name="index">
        <NativeTabs.Trigger.Icon
          sf={{ default: 'bubble.left', selected: 'bubble.left.fill' }}
          md="chat"
        />
        <NativeTabs.Trigger.Label>Chat</NativeTabs.Trigger.Label>
      </NativeTabs.Trigger>

      <NativeTabs.Trigger name="tts">
        <NativeTabs.Trigger.Icon
          sf={{ default: 'speaker', selected: 'speaker.wave.2.fill' }}
          md="record_voice_over"
        />
        <NativeTabs.Trigger.Label>TTS</NativeTabs.Trigger.Label>
      </NativeTabs.Trigger>

      <NativeTabs.Trigger name="stt">
        <NativeTabs.Trigger.Icon sf={{ default: 'mic', selected: 'mic.fill' }} md="mic" />
        <NativeTabs.Trigger.Label>STT</NativeTabs.Trigger.Label>
      </NativeTabs.Trigger>

      <NativeTabs.Trigger name="embeddings">
        <NativeTabs.Trigger.Icon
          sf={{
            default: 'point.3.filled.connected.trianglepath.dotted',
            selected: 'point.3.connected.trianglepath.dotted',
          }}
          md="hub"
        />
        <NativeTabs.Trigger.Label>Embed</NativeTabs.Trigger.Label>
      </NativeTabs.Trigger>
    </NativeTabs>
  )
}
