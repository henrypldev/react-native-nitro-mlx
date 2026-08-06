import Foundation

// Standalone executable test, run via `bun test:ios-llm-error`.

enum TestFailure: Error, CustomStringConvertible {
    case failed(String)

    var description: String {
        switch self {
        case .failed(let message):
            return message
        }
    }
}

func expect(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() {
        throw TestFailure.failed(message)
    }
}

@main
struct LLMErrorTests {
    static func main() throws {
        try errorDescriptions()
        try failureStages()
        try generationStageRawValues()
        print("LLMErrorTests passed")
    }

    private static func errorDescriptions() throws {
        try expect(
            LLMError.notLoaded.errorDescription
                == "No model is loaded. Call load() before generation.",
            "notLoaded message"
        )
        try expect(
            LLMError.alreadyGenerating.errorDescription
                == "A generation is already in progress. Wait for it to finish or call stop().",
            "alreadyGenerating message"
        )
        try expect(
            LLMError.generationFailed(stage: "generate", message: "boom").errorDescription
                == "Generation failed during generate: boom",
            "generationFailed message"
        )
        try expect(
            LLMError.generationFailed(stage: "generate", message: "boom").localizedDescription
                == "Generation failed during generate: boom",
            "localizedDescription reaches the JS-facing message"
        )
    }

    private static func failureStages() throws {
        try expect(LLMError.notLoaded.failureStage == nil, "notLoaded has no stage")
        try expect(
            LLMError.alreadyGenerating.failureStage == nil,
            "alreadyGenerating has no stage"
        )
        try expect(
            LLMError.generationFailed(stage: "prepare", message: "x").failureStage == "prepare",
            "generationFailed stage"
        )
    }

    private static func generationStageRawValues() throws {
        try expect(GenerationStage.prepare.rawValue == "prepare", "stage prepare")
        try expect(GenerationStage.generate.rawValue == "generate", "stage generate")
        try expect(GenerationStage.tool.rawValue == "tool", "stage tool")
        try expect(GenerationStage.history.rawValue == "history", "stage history")
    }
}
