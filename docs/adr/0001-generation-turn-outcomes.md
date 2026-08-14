# Resolve started LLM generation turns as outcomes

Once an LLM Generation Turn starts, it resolves with one normalized LLM Generation Outcome for completion, cancellation, or runtime failure; only validation before the turn starts rejects. This makes partial content and statistics reliable across complete-text, token-stream, and event-stream delivery, while requiring callers to inspect the finish reason instead of treating runtime failures as rejected promises.

## Consequences

The public interface makes one breaking change: terminal events and method return values carry the same outcome, and `getLastGenerationStats()` is removed. Stopped turns commit their partial assistant content; superseded, unloaded, and failed turns do not commit. The native turn module restores its managed MLX session after any outcome that cannot safely retain the mutated KV cache.
