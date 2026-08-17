# Expose turn contexts over one resident model

This package holds one Resident Model and runs one LLM Generation Turn at a time. Callers that need several independent conversations or agent roles create Turn Contexts, which hold instructions, transcript, and a warm KV cache over that single model, instead of loading a model for each role. The package returns Tool Call Requests to the caller rather than running a tool loop itself, because tool policy, approval, budgets, and persistence are product decisions that belong to the consumer application.

## Consequences

Agent orchestration is out of scope for this package; consumers build it on these primitives. A Turn Context costs memory in proportion to its transcript, so the consumer creates and releases contexts and the package never evicts one that the caller still holds. `LLM.load` with an already-loaded model ID no longer reads weights from disk, which makes a change of instructions, tools, or history cheap. Serialized turns trade parallel generation latency for bounded memory, predictable thermal behavior, and simple cancellation.
