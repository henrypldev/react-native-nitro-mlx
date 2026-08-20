import { useColorScheme } from 'react-native'

export type Palette = {
  bg: string
  surface: string
  hairline: string
  ink: string
  onInk: string
  muted: string
  ember: string
  error: string
}

const light: Palette = {
  bg: '#FFFFFF',
  surface: '#F2F4F7',
  hairline: '#E6EAF0',
  ink: '#0F172A',
  onInk: '#FFFFFF',
  muted: '#64748B',
  ember: '#D9480F',
  error: '#DC2626',
}

const dark: Palette = {
  bg: '#0B0D10',
  surface: '#171B21',
  hairline: '#262C35',
  ink: '#F5F7FA',
  onInk: '#0B0D10',
  muted: '#8B94A3',
  ember: '#FF8A4C',
  error: '#F87171',
}

export function usePalette(): Palette {
  return useColorScheme() === 'dark' ? dark : light
}
