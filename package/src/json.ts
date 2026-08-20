/**
 * A value produced by `JSON.parse`. Narrower than Nitro's `AnyMap` value type
 * (no `Int64`), so it stays assignable to `AnyMap` when crossing the bridge.
 */
export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue }

/** A JSON object as produced by `JSON.parse` on an object payload. */
export type JsonObject = { [key: string]: JsonValue }
