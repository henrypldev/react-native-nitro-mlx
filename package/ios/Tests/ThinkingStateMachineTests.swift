import Foundation

enum ThinkingStateMachineTestFailure: Error {
    case mismatch(String)
}

@main
struct ThinkingStateMachineTests {
    static func main() throws {
        var machine = ThinkingStateMachine()
        var visible = ""
        var thinking = ""

        for chunk in ["before<th", "ink>private", " thought</thi", "nk>after"] {
            for output in machine.process(token: chunk) {
                switch output {
                case .token(let token):
                    visible += token
                case .thinkingChunk(let chunk):
                    thinking += chunk
                case .thinkingStart, .thinkingEnd:
                    break
                }
            }
        }

        guard visible == "beforeafter" else {
            throw ThinkingStateMachineTestFailure.mismatch(
                "expected visible content without thinking tags, got \(visible)"
            )
        }
        guard thinking == "private thought" else {
            throw ThinkingStateMachineTestFailure.mismatch(
                "expected separately accumulated thinking, got \(thinking)"
            )
        }

        print("ThinkingStateMachineTests passed")
    }
}
