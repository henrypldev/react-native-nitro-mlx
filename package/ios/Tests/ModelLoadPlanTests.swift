import Foundation

@main
struct ModelLoadPlanTests {
    static func main() {
        var failures = 0
        func expect(_ condition: Bool, _ message: String) {
            if !condition { failures += 1; print("FAIL: \(message)") } else { print("PASS: \(message)") }
        }

        // Same model, container resident -> reuse
        expect(
            ModelLoadPlan.action(requestedModelId: "m/a", loadedModelId: "m/a", hasContainer: true) == .reuseContainer,
            "same model with container reuses"
        )
        // Different model -> load
        expect(
            ModelLoadPlan.action(requestedModelId: "m/b", loadedModelId: "m/a", hasContainer: true) == .loadContainer,
            "different model loads"
        )
        // No container (never loaded, or unloaded) -> load
        expect(
            ModelLoadPlan.action(requestedModelId: "m/a", loadedModelId: nil, hasContainer: false) == .loadContainer,
            "no container loads"
        )
        // Same id string but container gone (unload happened) -> load
        expect(
            ModelLoadPlan.action(requestedModelId: "m/a", loadedModelId: "m/a", hasContainer: false) == .loadContainer,
            "same id without container loads"
        )
        // Empty loaded id treated as not loaded
        expect(
            ModelLoadPlan.action(requestedModelId: "m/a", loadedModelId: "", hasContainer: true) == .loadContainer,
            "empty loaded id loads"
        )

        if failures > 0 { exit(1) }
        print("All ModelLoadPlan tests passed")
    }
}
