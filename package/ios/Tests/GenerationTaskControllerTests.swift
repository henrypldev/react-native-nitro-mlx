import Foundation

enum GenerationTaskControllerTestFailure: Error, CustomStringConvertible {
    case failed(String)

    var description: String {
        switch self {
        case .failed(let message):
            return message
        }
    }
}

func expectGenerationTaskController(
    _ condition: @autoclosure () -> Bool,
    _ message: String
) throws {
    if !condition() {
        throw GenerationTaskControllerTestFailure.failed(message)
    }
}

@main
struct GenerationTaskControllerTests {
    @MainActor
    static func main() async throws {
        try await cancellationKeepsTheSlotOccupiedUntilCompletion()
        try staleCompletionCannotClearANewerTask()
        print("GenerationTaskControllerTests passed")
    }

    @MainActor
    private static func cancellationKeepsTheSlotOccupiedUntilCompletion() async throws {
        let controller = GenerationTaskController<String>()
        let task = Task<String, Never> { @MainActor in
            while !Task.isCancelled {
                await Task.yield()
            }
            return "partial"
        }

        _ = try controller.begin(task)
        controller.cancel(reason: .stopped)

        try expectGenerationTaskController(
            controller.cancellationReason == .stopped,
            "the turn must retain its public cancellation reason"
        )

        try expectGenerationTaskController(
            controller.isActive,
            "cancellation must not make the generation slot idle"
        )

        do {
            try controller.ensureIdle()
            throw GenerationTaskControllerTestFailure.failed(
                "a second generation should be rejected until cancellation completes"
            )
        } catch LLMError.alreadyGenerating {
            // Expected.
        }

        await controller.cancelAndWait(reason: .stopped)
        try expectGenerationTaskController(
            !controller.isActive,
            "the slot should clear after the cancelled task completes"
        )
    }

    @MainActor
    private static func staleCompletionCannotClearANewerTask() throws {
        let controller = GenerationTaskController<String>()
        let firstTask = Task<String, Never> { "first" }
        let firstID = try controller.begin(firstTask)
        controller.finish(id: firstID)

        let secondTask = Task<String, Never> { "second" }
        let secondID = try controller.begin(secondTask)
        controller.finish(id: firstID)

        try expectGenerationTaskController(
            controller.isActive,
            "a stale completion must not clear the newer task"
        )

        controller.finish(id: secondID)
        try expectGenerationTaskController(
            !controller.isActive,
            "the owning completion should clear its task"
        )
    }
}
