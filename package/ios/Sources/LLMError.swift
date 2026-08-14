import Foundation

public enum LLMError: Error, LocalizedError {
    case notLoaded
    case alreadyGenerating
    case generationFailed(stage: String, message: String)

    public var errorDescription: String? {
        switch self {
        case .notLoaded:
            return "No model is loaded. Call load() before generation."
        case .alreadyGenerating:
            return "A generation is already in progress. Wait for it to finish or call stop()."
        case .generationFailed(let stage, let message):
            return "Generation failed during \(stage): \(message)"
        }
    }

    /// The failure stage for `generationFailed`, nil otherwise.
    public var failureStage: String? {
        if case .generationFailed(let stage, _) = self {
            return stage
        }
        return nil
    }
}

/// Phases of a generation call, reported by failed generation outcomes.
public enum GenerationStage: String {
    case prepare
    case generate
    case tool
    case history
}
