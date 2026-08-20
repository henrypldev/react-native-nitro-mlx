import Foundation

/// Decision for `LLM.load`: reuse the resident container or read weights again.
enum ModelLoadAction: Equatable {
    case loadContainer
    case reuseContainer
}

enum ModelLoadPlan {
    /// `loadedModelId` uses `nil`/empty to mean "nothing loaded"; the resident
    /// container is reused only when the requested id matches it exactly.
    static func action(
        requestedModelId: String,
        loadedModelId: String?,
        hasContainer: Bool
    ) -> ModelLoadAction {
        guard hasContainer, let loaded = loadedModelId, !loaded.isEmpty,
            loaded == requestedModelId
        else {
            return .loadContainer
        }
        return .reuseContainer
    }
}
