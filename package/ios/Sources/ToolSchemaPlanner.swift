import Foundation

enum ToolSchemaError: Error, Equatable {
    case invalidJSON
    case rootNotObjectSchema
}

enum ToolSchemaPlanner {
    /// Name of the single synthetic tool offered on a `responseSchema` turn.
    /// Shared with the caller that inspects the model's tool call so the two
    /// sides cannot drift.
    static let structuredOutputToolName = "respond_with_structured_output"

    /// Parses a serialized JSON Schema and enforces the wire contract:
    /// the root must be an object schema ("type": "object").
    static func parseParameters(_ json: String) throws -> [String: Any] {
        guard let data = json.data(using: .utf8),
            let parsed = try? JSONSerialization.jsonObject(with: data),
            let dictionary = parsed as? [String: Any]
        else {
            throw ToolSchemaError.invalidJSON
        }
        guard dictionary["type"] as? String == "object" else {
            throw ToolSchemaError.rootNotObjectSchema
        }
        return dictionary
    }

    /// Builds the single synthetic tool used for structured output. The model
    /// is asked, not forced, to call it (no tool_choice exists upstream), so
    /// the description restates the shape expectation.
    static func syntheticTool(
        responseSchema: String
    ) throws -> (name: String, description: String, parameters: [String: Any]) {
        let parameters = try parseParameters(responseSchema)
        return (
            name: structuredOutputToolName,
            description:
                "Return your final answer by calling this tool. It is the only way to respond: "
                + "call it exactly once, with arguments that match its parameter schema.",
            parameters: parameters
        )
    }
}
