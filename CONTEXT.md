# On-device MLX Inference

This context describes the language used for model inference exposed by react-native-nitro-mlx.

## Language

**LLM Generation Turn**:
A single prompt and all of its tool-call continuations, ending with one LLM Generation Outcome and one history decision. Cancellation completes the turn promptly with the content generated so far; late tool results do not resume it.
_Avoid_: Model invocation, generation pass

**LLM Generation Outcome**:
The terminal record of an LLM Generation Turn, containing user-visible content, optional thinking content, generation statistics, finish reason, and failure details when applicable. Once a turn begins, completion, cancellation, and runtime failure all produce this same kind of outcome; only rejection before the turn begins produces no outcome.
_Avoid_: Generation result, terminal event
