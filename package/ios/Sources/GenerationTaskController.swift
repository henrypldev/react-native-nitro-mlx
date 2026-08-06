import Foundation

/// Owns the single active generation task and prevents stale completions from
/// clearing a newer task. All access is serialized by `HybridLLMCore`'s actor.
@MainActor
final class GenerationTaskController {
    private struct ActiveGeneration {
        let id: UInt64
        let task: Task<String, Error>
    }

    private var activeGeneration: ActiveGeneration?
    private var nextID: UInt64 = 0

    var isActive: Bool {
        activeGeneration != nil
    }

    func ensureIdle() throws {
        guard activeGeneration == nil else {
            throw LLMError.alreadyGenerating
        }
    }

    func begin(_ task: Task<String, Error>) throws -> UInt64 {
        try ensureIdle()
        nextID &+= 1
        let id = nextID
        activeGeneration = ActiveGeneration(id: id, task: task)
        return id
    }

    func finish(id: UInt64) {
        guard activeGeneration?.id == id else { return }
        activeGeneration = nil
    }

    func cancel() {
        activeGeneration?.task.cancel()
    }

    func cancelAndWait() async {
        guard let activeGeneration else { return }
        activeGeneration.task.cancel()
        _ = await activeGeneration.task.result
        finish(id: activeGeneration.id)
    }
}
