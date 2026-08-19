import Foundation

/// Owns the single active generation task and prevents stale completions from
/// clearing a newer task. All access is serialized by `HybridLLMCore`'s actor.
///
/// The task's result type is erased on entry so one slot excludes every kind of
/// generation: legacy prompts resolve `LLMGenerationOutcome`, turns resolve
/// `LLMTurnOutcome`, and only one of them may run at a time.
@MainActor
final class GenerationTaskController {
    private struct ActiveGeneration {
        let id: UInt64
        let cancel: () -> Void
        let awaitCompletion: () async -> Void
    }

    private var activeGeneration: ActiveGeneration?
    private var nextID: UInt64 = 0
    private(set) var cancellationReason: GenerationCancellationReason?

    var isActive: Bool {
        activeGeneration != nil
    }

    func ensureIdle() throws {
        guard activeGeneration == nil else {
            throw LLMError.alreadyGenerating
        }
    }

    func begin<Value>(_ task: Task<Value, Never>) throws -> UInt64 {
        try ensureIdle()
        nextID &+= 1
        let id = nextID
        cancellationReason = nil
        activeGeneration = ActiveGeneration(
            id: id,
            cancel: { task.cancel() },
            awaitCompletion: { _ = await task.value }
        )
        return id
    }

    func finish(id: UInt64) {
        guard activeGeneration?.id == id else { return }
        activeGeneration = nil
        cancellationReason = nil
    }

    func cancel(reason: GenerationCancellationReason) {
        cancellationReason = reason
        activeGeneration?.cancel()
    }

    func cancelAndWait(reason: GenerationCancellationReason) async {
        guard let activeGeneration else { return }
        cancellationReason = reason
        activeGeneration.cancel()
        await activeGeneration.awaitCompletion()
        finish(id: activeGeneration.id)
    }
}

enum GenerationCancellationReason: Equatable {
    case stopped
    case superseded
    case unloaded
}
