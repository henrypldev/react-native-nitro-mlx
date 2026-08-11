import type { HybridObject } from 'react-native-nitro-modules'

/**
 * Options for loading an embeddings model.
 */
export interface EmbeddingsLoadOptions {
  /** Callback invoked with loading progress (0-1) */
  onProgress?: (progress: number) => void
}

/**
 * Options for embedding calls.
 */
export interface EmbeddingsEmbedOptions {
  /**
   * When true, inputs longer than `maxSequenceLength` tokens are truncated
   * (the closing special token is preserved). When false or omitted, oversized
   * inputs are rejected with an error.
   */
  truncate?: boolean
}

/**
 * Low-level embeddings interface using MLX.
 * @internal Use the `Embeddings` export from `react-native-nitro-mlx` instead.
 */
export interface Embeddings extends HybridObject<{ ios: 'swift' }> {
  /**
   * Load an embedding model into memory. Downloads from HuggingFace if not already cached.
   * @param modelId - HuggingFace model ID (e.g., 'mlx-community/bge-small-en-v1.5-4bit')
   * @param options - Callback invoked with loading progress (0-1)
   */
  load(modelId: string, options?: EmbeddingsLoadOptions): Promise<void>

  /**
   * Unload the current model and release memory.
   */
  unload(): void

  /**
   * Embed a single text and return the vector as Float32 bytes.
   * @param text - Input text
   * @param options - Truncation behavior for oversized input
   * @returns Float32 embedding bytes (length = `dimension` floats)
   */
  embed(text: string, options?: EmbeddingsEmbedOptions): Promise<ArrayBuffer>

  /**
   * Embed a batch of texts in a single forward pass (padded to the longest).
   * @param texts - Array of input texts (at most 64 per call)
   * @param options - Truncation behavior for oversized input
   * @returns Array of Float32 embedding buffers, one per input
   */
  embedBatch(texts: string[], options?: EmbeddingsEmbedOptions): Promise<ArrayBuffer[]>

  /** Whether a model is currently loaded */
  readonly isLoaded: boolean
  /** Output embedding dimension (e.g., 384 for bge-small) */
  readonly dimension: number
  /** Maximum supported sequence length (tokens) */
  readonly maxSequenceLength: number
}
