import Foundation

@main
struct TurnRequestPlannerTests {
    static func main() {
        var failures = 0
        func expect(_ condition: Bool, _ message: String) {
            if !condition { failures += 1; print("FAIL: \(message)") } else { print("PASS: \(message)") }
        }
        func expectThrows(_ expected: TurnPlanError, _ message: String, _ body: () throws -> Void) {
            do {
                try body()
                failures += 1
                print("FAIL: \(message) — no error thrown")
            } catch let error as TurnPlanError {
                expect(error == expected, "\(message) — got \(error)")
            } catch {
                failures += 1
                print("FAIL: \(message) — wrong error type \(error)")
            }
        }

        func msg(
            _ role: String, _ content: String = "x",
            toolCallId: String? = nil, toolCallsJson: String? = nil
        ) -> TurnMessagePlan {
            TurnMessagePlan(
                role: role, content: content, toolCallId: toolCallId,
                name: nil, isError: nil, toolCallsJson: toolCallsJson
            )
        }

        // Empty messages reject
        expectThrows(.emptyMessages, "empty messages reject") {
            _ = try TurnRequestPlanner.plan(
                messages: [], contextId: nil, contextKnown: false, contextHasTools: false,
                pendingToolCallIds: [], hasColdFields: false,
                requestHasTools: false, hasResponseSchema: false
            )
        }

        // Unknown role rejects
        expectThrows(.unknownRole("oracle"), "unknown role rejects") {
            _ = try TurnRequestPlanner.plan(
                messages: [msg("oracle")], contextId: nil, contextKnown: false, contextHasTools: false,
                pendingToolCallIds: [], hasColdFields: false,
                requestHasTools: false, hasResponseSchema: false
            )
        }

        // Tool message without id rejects
        expectThrows(.missingToolCallId, "tool message without id rejects") {
            _ = try TurnRequestPlanner.plan(
                messages: [msg("tool")], contextId: nil, contextKnown: false, contextHasTools: false,
                pendingToolCallIds: [], hasColdFields: false,
                requestHasTools: false, hasResponseSchema: false
            )
        }

        // Unknown context rejects
        expectThrows(.unknownContext("ctx-9"), "unknown context rejects") {
            _ = try TurnRequestPlanner.plan(
                messages: [msg("user")], contextId: "ctx-9", contextKnown: false, contextHasTools: false,
                pendingToolCallIds: [], hasColdFields: false,
                requestHasTools: false, hasResponseSchema: false
            )
        }

        // Cold fields on a warm request reject
        expectThrows(.coldFieldsOnWarmTurn, "cold fields on warm request reject") {
            _ = try TurnRequestPlanner.plan(
                messages: [msg("user")], contextId: "ctx-1", contextKnown: true, contextHasTools: false,
                pendingToolCallIds: [], hasColdFields: true,
                requestHasTools: false, hasResponseSchema: false
            )
        }

        // responseSchema vs request tools rejects
        expectThrows(.schemaExclusiveWithTools, "schema with request tools rejects") {
            _ = try TurnRequestPlanner.plan(
                messages: [msg("user")], contextId: nil, contextKnown: false, contextHasTools: false,
                pendingToolCallIds: [], hasColdFields: false,
                requestHasTools: true, hasResponseSchema: true
            )
        }

        // responseSchema vs context tools rejects
        expectThrows(.schemaExclusiveWithTools, "schema with context tools rejects") {
            _ = try TurnRequestPlanner.plan(
                messages: [msg("user")], contextId: "ctx-1", contextKnown: true, contextHasTools: true,
                pendingToolCallIds: [], hasColdFields: false,
                requestHasTools: false, hasResponseSchema: true
            )
        }

        // Pending tool calls: fewer results reject
        expectThrows(.incompleteToolResults(missing: ["c2"]), "missing tool result rejects") {
            _ = try TurnRequestPlanner.plan(
                messages: [msg("tool", toolCallId: "c1")],
                contextId: "ctx-1", contextKnown: true, contextHasTools: true,
                pendingToolCallIds: ["c1", "c2"], hasColdFields: false,
                requestHasTools: false, hasResponseSchema: false
            )
        }

        // Pending tool calls: unknown id rejects
        expectThrows(.unknownToolCallId("cX"), "unknown tool result id rejects") {
            _ = try TurnRequestPlanner.plan(
                messages: [msg("tool", toolCallId: "cX")],
                contextId: "ctx-1", contextKnown: true, contextHasTools: true,
                pendingToolCallIds: ["c1"], hasColdFields: false,
                requestHasTools: false, hasResponseSchema: false
            )
        }

        // Pending tool calls: duplicate result id rejects
        expectThrows(.duplicateToolCallId("c1"), "duplicate tool result id rejects") {
            _ = try TurnRequestPlanner.plan(
                messages: [msg("tool", toolCallId: "c1"), msg("tool", toolCallId: "c1")],
                contextId: "ctx-1", contextKnown: true, contextHasTools: true,
                pendingToolCallIds: ["c1"], hasColdFields: false,
                requestHasTools: false, hasResponseSchema: false
            )
        }

        // Pending tool calls: duplicate among an otherwise complete set rejects
        expectThrows(.duplicateToolCallId("c2"), "duplicate among complete set rejects") {
            _ = try TurnRequestPlanner.plan(
                messages: [msg("tool", toolCallId: "c2"), msg("tool", toolCallId: "c1"), msg("tool", toolCallId: "c2")],
                contextId: "ctx-1", contextKnown: true, contextHasTools: true,
                pendingToolCallIds: ["c1", "c2"], hasColdFields: false,
                requestHasTools: false, hasResponseSchema: false
            )
        }

        // Pending tool calls: complete set in any order passes, mode is warm
        do {
            let plan = try TurnRequestPlanner.plan(
                messages: [msg("tool", toolCallId: "c2"), msg("tool", toolCallId: "c1")],
                contextId: "ctx-1", contextKnown: true, contextHasTools: true,
                pendingToolCallIds: ["c1", "c2"], hasColdFields: false,
                requestHasTools: false, hasResponseSchema: false
            )
            expect(plan.mode == .warm, "warm plan for context request")
            expect(plan.providedToolCallIds.sorted() == ["c1", "c2"], "provided ids collected")
        } catch {
            failures += 1
            print("FAIL: complete tool results should pass — \(error)")
        }

        // Minimal cold request passes
        do {
            let plan = try TurnRequestPlanner.plan(
                messages: [msg("user")], contextId: nil, contextKnown: false, contextHasTools: false,
                pendingToolCallIds: [], hasColdFields: true,
                requestHasTools: true, hasResponseSchema: false
            )
            expect(plan.mode == .cold, "cold plan without context")
        } catch {
            failures += 1
            print("FAIL: minimal cold request should pass — \(error)")
        }

        if failures > 0 { exit(1) }
        print("All TurnRequestPlanner tests passed")
    }
}
