import Foundation

@main
struct TurnContextRegistryTests {
    static func main() {
        var failures = 0
        func expect(_ condition: Bool, _ message: String) {
            if !condition { failures += 1; print("FAIL: \(message)") } else { print("PASS: \(message)") }
        }

        var registry = TurnContextRegistry<String>()

        let a = registry.insert("session-a")
        let b = registry.insert("session-b")
        expect(a != b, "ids are unique")
        expect(a.hasPrefix("ctx-"), "ids carry the ctx- prefix")
        expect(registry.count == 2, "count tracks inserts")
        expect(registry.ids == [a, b].sorted(), "ids are sorted")
        expect(registry.entry(for: a) == "session-a", "lookup returns the entry")
        expect(registry.entry(for: "ctx-999") == nil, "unknown id returns nil")

        registry.update(a, with: "session-a2")
        expect(registry.entry(for: a) == "session-a2", "update replaces the entry")

        expect(registry.release(a) == "session-a2", "release returns the entry")
        expect(registry.release(a) == nil, "double release returns nil (idempotent)")
        expect(registry.count == 1, "release shrinks count")

        let drained = registry.releaseAll()
        expect(drained == ["session-b"], "releaseAll returns remaining entries")
        expect(registry.count == 0, "releaseAll empties the registry")

        if failures > 0 { exit(1) }
        print("All TurnContextRegistry tests passed")
    }
}
