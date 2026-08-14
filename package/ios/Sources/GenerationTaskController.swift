import Foundation

/// Owns the single active generation task and prevents stale completions from
/// clearing a newer task. All access is serialized by `HybridLLMCore`'s actor.
@MainActor
final class GenerationTaskController<Value> {
    private struct ActiveGeneration {
        let id: UInt64
        let task: Task<Value, Never>
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

    func begin(_ task: Task<Value, Never>) throws -> UInt64 {
        try ensureIdle()
        nextID &+= 1
        let id = nextID
        cancellationReason = nil
        activeGeneration = ActiveGeneration(id: id, task: task)
        return id
    }

    func finish(id: UInt64) {
        guard activeGeneration?.id == id else { return }
        activeGeneration = nil
        cancellationReason = nil
    }

    func cancel(reason: GenerationCancellationReason) {
        cancellationReason = reason
        activeGeneration?.task.cancel()
    }

    func cancelAndWait(reason: GenerationCancellationReason) async {
        guard let activeGeneration else { return }
        cancellationReason = reason
        activeGeneration.task.cancel()
        _ = await activeGeneration.task.value
        finish(id: activeGeneration.id)
    }
}

enum GenerationCancellationReason: Equatable {
    case stopped
    case superseded
    case unloaded
}
