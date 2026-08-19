import Foundation

/// Bookkeeping for Turn Contexts. Generic over the entry type so this file
/// stays Foundation-only and testable with the standalone swiftc pattern;
/// HybridLLM instantiates it with an entry holding a ChatSession.
struct TurnContextRegistry<Entry> {
    private var entries: [String: Entry] = [:]
    private var nextId: UInt64 = 0

    var count: Int { entries.count }
    var ids: [String] { entries.keys.sorted() }

    mutating func insert(_ entry: Entry) -> String {
        nextId += 1
        let id = "ctx-\(nextId)"
        entries[id] = entry
        return id
    }

    func entry(for id: String) -> Entry? {
        entries[id]
    }

    mutating func update(_ id: String, with entry: Entry) {
        guard entries[id] != nil else { return }
        entries[id] = entry
    }

    @discardableResult
    mutating func release(_ id: String) -> Entry? {
        entries.removeValue(forKey: id)
    }

    mutating func releaseAll() -> [Entry] {
        let sortedIds = ids
        let released = sortedIds.compactMap { entries[$0] }
        entries.removeAll()
        return released
    }
}
