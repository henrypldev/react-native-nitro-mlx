import Foundation
import NitroModules

/// Emits generation lifecycle events to JS as native `StreamEventEnvelope` structs.
///
/// Nitro cannot express a discriminated union of structs, so all event shapes share one
/// envelope discriminated by `StreamEventKind`; `llm.ts` maps it back to the `StreamEvent`
/// union. Fields not relevant to a given kind stay `nil`.
struct StreamEventEmitter {
    private let callback: (StreamEventEnvelope) -> Void

    init(callback: @escaping (StreamEventEnvelope) -> Void) {
        self.callback = callback
    }

    private func nowMs() -> Double {
        Date().timeIntervalSince1970 * 1000
    }

    private func emit(
        _ kind: StreamEventKind,
        timestamp: Double? = nil,
        token: String? = nil,
        chunk: String? = nil,
        content: String? = nil,
        id: String? = nil,
        name: String? = nil,
        arguments: String? = nil,
        result: String? = nil,
        error: String? = nil,
        stats: GenerationStats? = nil
    ) {
        callback(
            StreamEventEnvelope(
                kind: kind,
                timestamp: timestamp,
                token: token,
                chunk: chunk,
                content: content,
                id: id,
                name: name,
                arguments: arguments,
                result: result,
                error: error,
                stats: stats
            )
        )
    }

    func emitGenerationStart() {
        emit(.generationStart, timestamp: nowMs())
    }

    func emitToken(_ token: String) {
        emit(.token, token: token)
    }

    func emitThinkingStart() {
        emit(.thinkingStart, timestamp: nowMs())
    }

    func emitThinkingChunk(_ chunk: String) {
        emit(.thinkingChunk, chunk: chunk)
    }

    func emitThinkingEnd(_ content: String) {
        emit(.thinkingEnd, timestamp: nowMs(), content: content)
    }

    func emitToolCallStart(id: String, name: String, arguments: String) {
        emit(.toolCallStart, id: id, name: name, arguments: arguments)
    }

    func emitToolCallExecuting(id: String) {
        emit(.toolCallExecuting, id: id)
    }

    func emitToolCallCompleted(id: String, result: String) {
        emit(.toolCallCompleted, id: id, result: result)
    }

    func emitToolCallFailed(id: String, error: String) {
        emit(.toolCallFailed, id: id, error: error)
    }

    func emitGenerationEnd(content: String, stats: GenerationStats) {
        emit(.generationEnd, content: content, stats: stats)
    }
}
