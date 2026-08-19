import Foundation

@main
struct ToolSchemaPlannerTests {
    static func main() {
        var failures = 0
        func expect(_ condition: Bool, _ message: String) {
            if !condition { failures += 1; print("FAIL: \(message)") } else { print("PASS: \(message)") }
        }

        // Valid object schema parses
        do {
            let schema = try ToolSchemaPlanner.parseParameters(
                #"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}"#
            )
            expect(schema["type"] as? String == "object", "valid schema parses")
            expect((schema["properties"] as? [String: Any])?["path"] != nil, "nested properties survive")
        } catch {
            failures += 1
            print("FAIL: valid schema should parse — \(error)")
        }

        // Invalid JSON throws
        do {
            _ = try ToolSchemaPlanner.parseParameters("{oops")
            failures += 1
            print("FAIL: invalid JSON should throw")
        } catch ToolSchemaError.invalidJSON {
            print("PASS: invalid JSON throws")
        } catch {
            failures += 1
            print("FAIL: wrong error for invalid JSON — \(error)")
        }

        // Non-object root throws
        do {
            _ = try ToolSchemaPlanner.parseParameters(#"{"type":"string"}"#)
            failures += 1
            print("FAIL: non-object root should throw")
        } catch ToolSchemaError.rootNotObjectSchema {
            print("PASS: non-object root throws")
        } catch {
            failures += 1
            print("FAIL: wrong error for non-object root — \(error)")
        }

        // Synthetic tool carries the schema and the fixed name
        do {
            let tool = try ToolSchemaPlanner.syntheticTool(
                responseSchema: #"{"type":"object","properties":{"answer":{"type":"string"}}}"#
            )
            expect(tool.name == "respond_with_structured_output", "synthetic tool name is fixed")
            expect(tool.description.contains("only"), "description instructs exclusive use")
            expect(tool.parameters["type"] as? String == "object", "synthetic tool carries the schema")
        } catch {
            failures += 1
            print("FAIL: synthetic tool should build — \(error)")
        }

        if failures > 0 { exit(1) }
        print("All ToolSchemaPlanner tests passed")
    }
}
