import { NitroModules } from 'react-native-nitro-modules'
import {
  assertNonEmptyString,
  validateEmbeddingsBatch,
  validateEmbeddingsEmbedOptions,
  validateEmbeddingsLoadOptions,
} from './runtime'
import type {
  EmbeddingsEmbedOptions,
  EmbeddingsLoadOptions,
  Embeddings as EmbeddingsSpec,
} from './specs/Embeddings.nitro'

let instance: EmbeddingsSpec | null = null
let loadedModelId: string | null = null

function getInstance(): EmbeddingsSpec {
  if (!instance) {
    instance = NitroModules.createHybridObject<EmbeddingsSpec>('Embeddings')
  }
  if (!instance) {
    throw new Error('Failed to initialize the Embeddings Nitro module.')
  }
  return instance
}

type PrefixFamily = {
  test: (modelId: string) => boolean
  query: (text: string) => string
  document: (text: string) => string
}

// Asymmetric retrieval models need task prefixes. Symmetric models (BGE, MiniLM)
// behave the same for queries and documents — no prefix.
const PREFIX_FAMILIES: PrefixFamily[] = [
  {
    test: id => /multilingual-e5|(^|\/)e5[-_]/i.test(id),
    query: t => `query: ${t}`,
    document: t => `passage: ${t}`,
  },
  {
    test: id => /nomic-embed-text/i.test(id),
    query: t => `search_query: ${t}`,
    document: t => `search_document: ${t}`,
  },
  {
    test: id => /qwen3-embedding/i.test(id),
    query: t => `Instruct: Given a query, retrieve relevant documents\nQuery: ${t}`,
    document: t => t,
  },
]

function prefixFamily(modelId: string | null): PrefixFamily | null {
  if (!modelId) return null
  return PREFIX_FAMILIES.find(f => f.test(modelId)) ?? null
}

function toFloat32(buffer: ArrayBuffer): Float32Array {
  return new Float32Array(buffer)
}

/**
 * Cosine similarity between two embedding vectors.
 * @returns Value in [-1, 1]. Returns 0 if either vector is all zeros.
 */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error(
      `[react-native-nitro-mlx] cosineSimilarity: length mismatch (${a.length} vs ${b.length}).`,
    )
  }
  let dot = 0
  let normA = 0
  let normB = 0
  for (let i = 0; i < a.length; i++) {
    // biome-ignore lint/style/noNonNullAssertion: indices within length are guaranteed numbers for Float32Array
    const x = a[i]!
    // biome-ignore lint/style/noNonNullAssertion: indices within length are guaranteed numbers for Float32Array
    const y = b[i]!
    dot += x * y
    normA += x * x
    normB += y * y
  }
  if (normA === 0 || normB === 0) return 0
  return dot / (Math.sqrt(normA) * Math.sqrt(normB))
}

/**
 * Text embeddings using MLX on Apple Silicon.
 *
 * @example
 * ```ts
 * import { Embeddings, cosineSimilarity } from 'react-native-nitro-mlx'
 *
 * await Embeddings.load('mlx-community/bge-small-en-v1.5-4bit')
 * const a = await Embeddings.embed('cats are cute')
 * const b = await Embeddings.embed('kittens are adorable')
 * console.log(cosineSimilarity(a, b))
 * ```
 */
export const Embeddings = {
  /**
   * Load an embedding model into memory. Downloads from HuggingFace if not already cached.
   * @param modelId - HuggingFace model ID (e.g., 'mlx-community/bge-small-en-v1.5-4bit')
   * @param options - Callback invoked with loading progress (0-1)
   */
  async load(modelId: string, options?: EmbeddingsLoadOptions): Promise<void> {
    const validated = assertNonEmptyString(modelId, 'Embeddings modelId')
    await getInstance().load(validated, validateEmbeddingsLoadOptions(options))
    loadedModelId = validated
  },

  /**
   * Unload the current model and release memory.
   */
  unload(): void {
    getInstance().unload()
    loadedModelId = null
  },

  /**
   * Embed a single text and return the vector as a `Float32Array`.
   * Input longer than `maxSequenceLength` tokens is rejected unless
   * `options.truncate` is true.
   */
  async embed(text: string, options?: EmbeddingsEmbedOptions): Promise<Float32Array> {
    const buffer = await getInstance().embed(
      assertNonEmptyString(text, 'Embeddings text'),
      validateEmbeddingsEmbedOptions(options),
    )
    return toFloat32(buffer)
  },

  /**
   * Embed a batch of texts in a single padded forward pass. Accepts at most 64
   * texts per call; input longer than `maxSequenceLength` tokens is rejected
   * unless `options.truncate` is true.
   */
  async embedBatch(
    texts: string[],
    options?: EmbeddingsEmbedOptions,
  ): Promise<Float32Array[]> {
    const buffers = await getInstance().embedBatch(
      validateEmbeddingsBatch(texts),
      validateEmbeddingsEmbedOptions(options),
    )
    return buffers.map(toFloat32)
  },

  /**
   * Embed a text as a retrieval *query*. For asymmetric models (E5, Nomic, Qwen3)
   * this auto-prepends the canonical query prefix. For symmetric models (BGE,
   * MiniLM) this is identical to `embed()`.
   */
  async embedQuery(
    text: string,
    options?: EmbeddingsEmbedOptions,
  ): Promise<Float32Array> {
    const validated = assertNonEmptyString(text, 'Embeddings text')
    const family = prefixFamily(loadedModelId)
    const prefixed = family ? family.query(validated) : validated
    const buffer = await getInstance().embed(
      prefixed,
      validateEmbeddingsEmbedOptions(options),
    )
    return toFloat32(buffer)
  },

  /**
   * Embed a text as a retrieval *document* / passage. For asymmetric models
   * (E5, Nomic) this auto-prepends the canonical document prefix. For
   * symmetric models (BGE, MiniLM) this is identical to `embed()`.
   */
  async embedDocument(
    text: string,
    options?: EmbeddingsEmbedOptions,
  ): Promise<Float32Array> {
    const validated = assertNonEmptyString(text, 'Embeddings text')
    const family = prefixFamily(loadedModelId)
    const prefixed = family ? family.document(validated) : validated
    const buffer = await getInstance().embed(
      prefixed,
      validateEmbeddingsEmbedOptions(options),
    )
    return toFloat32(buffer)
  },

  /** Whether a model is currently loaded */
  get isLoaded(): boolean {
    return getInstance().isLoaded
  },

  /** Output embedding dimension (e.g., 384 for bge-small) */
  get dimension(): number {
    return getInstance().dimension
  },

  /** Maximum supported sequence length (tokens) */
  get maxSequenceLength(): number {
    return getInstance().maxSequenceLength
  },
}
