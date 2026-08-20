# On-device MLX Inference

This context describes the language used for model inference exposed by react-native-nitro-mlx.

## Language

**Resident Model**:
The one set of model weights held in memory and shared by every Turn Context and every legacy entry point. Loading a different model ID replaces it.
_Avoid_: Model instance, engine

**Turn Context**:
Named, retained state for a sequence of LLM Generation Turns — instructions, accumulated transcript, and warm KV cache — held over the Resident Model. A Turn Context never owns model weights.
_Avoid_: Session, conversation, agent

**LLM Generation Turn**:
One bounded model pass, ending with final content or with Tool Call Requests returned to the caller. Legacy entry points instead execute tool calls and continue generating inside the same turn. Cancellation completes the turn promptly with the content generated so far; late tool results do not resume it.
_Avoid_: Model invocation, generation pass

**Tool Call Request**:
A model-emitted request to run a named tool with JSON arguments, returned to the caller as part of an LLM Generation Outcome instead of being executed by this package.
_Avoid_: Tool invocation, function call

**LLM Generation Outcome**:
The terminal record of an LLM Generation Turn, containing user-visible content, optional thinking content, any Tool Call Requests, generation statistics, finish reason, and failure details when applicable. Once a turn begins, completion, cancellation, and runtime failure all produce this same kind of outcome; only rejection before the turn begins produces no outcome.
_Avoid_: Generation result, terminal event
