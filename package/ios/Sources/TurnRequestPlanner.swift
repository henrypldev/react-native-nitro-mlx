import Foundation

/// Foundation-only mirror of the wire turn message, so validation is
/// testable without MLX or Nitro. HybridLLM maps the generated type into this.
struct TurnMessagePlan {
    let role: String
    let content: String
    let toolCallId: String?
    let name: String?
    let isError: Bool?
    let toolCallsJson: String?
}

enum TurnPlanError: Error, Equatable {
    case emptyMessages
    case unknownRole(String)
    case missingToolCallId
    case toolCallsOnNonAssistant
    case unknownContext(String)
    case coldFieldsOnWarmTurn
    case schemaExclusiveWithTools
    case incompleteToolResults(missing: [String])
    case unknownToolCallId(String)
    case duplicateToolCallId(String)
}

enum TurnMode: Equatable {
    case cold
    case warm
}

struct TurnPlan: Equatable {
    let mode: TurnMode
    let providedToolCallIds: [String]
}

enum TurnRequestPlanner {
    private static let roles: Set<String> = ["system", "user", "assistant", "tool"]

    static func plan(
        messages: [TurnMessagePlan],
        contextId: String?,
        contextKnown: Bool,
        contextHasTools: Bool,
        pendingToolCallIds: [String],
        hasColdFields: Bool,
        requestHasTools: Bool,
        hasResponseSchema: Bool
    ) throws -> TurnPlan {
        guard !messages.isEmpty else { throw TurnPlanError.emptyMessages }

        var providedToolCallIds: [String] = []
        for message in messages {
            guard roles.contains(message.role) else {
                throw TurnPlanError.unknownRole(message.role)
            }
            if message.role == "tool" {
                guard let id = message.toolCallId, !id.isEmpty else {
                    throw TurnPlanError.missingToolCallId
                }
                providedToolCallIds.append(id)
            }
            if message.toolCallsJson != nil, message.role != "assistant" {
                throw TurnPlanError.toolCallsOnNonAssistant
            }
        }

        let mode: TurnMode
        if let contextId {
            guard contextKnown else { throw TurnPlanError.unknownContext(contextId) }
            guard !hasColdFields else { throw TurnPlanError.coldFieldsOnWarmTurn }
            mode = .warm

            // Every Tool Call Request from the previous turn needs exactly one
            // result, matched by id; order is free (spec: Tool call contract).
            var seenProvided: Set<String> = []
            for id in providedToolCallIds {
                if !seenProvided.insert(id).inserted {
                    throw TurnPlanError.duplicateToolCallId(id)
                }
            }

            let pending = Set(pendingToolCallIds)
            let provided = Set(providedToolCallIds)
            if let stray = provided.subtracting(pending).sorted().first {
                throw TurnPlanError.unknownToolCallId(stray)
            }
            let missing = pending.subtracting(provided).sorted()
            if !missing.isEmpty {
                throw TurnPlanError.incompleteToolResults(missing: missing)
            }
        } else {
            mode = .cold
        }

        if hasResponseSchema {
            let toolsInPlay = requestHasTools || (mode == .warm && contextHasTools)
            if toolsInPlay { throw TurnPlanError.schemaExclusiveWithTools }
        }

        return TurnPlan(mode: mode, providedToolCallIds: providedToolCallIds)
    }
}
