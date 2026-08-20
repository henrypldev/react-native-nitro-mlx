# Agent-harness inference conventions

Date: 2026-08-17 · Status: Research · Validates `docs/superpowers/specs/2026-08-17-turn-scoped-inference-primitives-design.md`

## Question and scope

The spec proposes that `react-native-nitro-mlx` expose Turn Contexts, a `runTurn` call that returns
Tool Call Requests instead of executing them, token counting, and structured output as a forced
synthetic tool call — and that it ship no agent loop. This document asks how established inference
SDKs and agent frameworks shape that same layer, whether the spec follows or deviates from the
consensus, and, where it deviates, whether the on-device constraint justifies it. Scope is the seam
between one model call and the loop around it. It is not a review of the whole package.

All sources are primary: vendor documentation, protocol schemas, and source code. Every source was
read on 2026-08-17 unless noted. Versions and commits are recorded because this layer moves fast.

## What is conventional

Eight sources agree on the shape of this layer far more than they disagree.

**The model call returns tool calls; something above it executes them.** This holds without
exception. Anthropic's Messages API stops with `stop_reason: "tool_use"` and returns `tool_use`
blocks for the caller to run
([tool-use/overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)).
`llama-server` returns `finish_reason: "tool_calls"` and never executes a tool on the inference path;
its `/tools` endpoint is a separate opt-in surface whose own README says "Please do NOT use this
endpoint in a downstream application" (`tools/server/README.md:1636-1640`). Ollama returns
`tool_calls` from `/api/chat` and runs nothing; its agent loop lives in `agent/`, imported only from
`cmd/`, never from `server/`. In the OpenAI Agents SDK, `grep -rn "on_invoke_tool" src/agents/models/`
returns zero matches across all 19 files; execution happens in
`run_internal/tool_execution.py`, reached only from the Runner. In the Vercel AI SDK, a provider's
`doGenerate` returns `{content, finishReason, usage}` and has no execution path at all.

**The boundary the spec draws is the boundary the field draws.** Anthropic states it as a product
decision: use the **Agent SDK** for "Building an agent without implementing the tool loop yourself",
and the **Client SDK** for "Calling the API directly and implementing the tool loop yourself"
([agent-sdk/overview](https://code.claude.com/docs/en/agent-sdk/overview)). Two libraries, two
layers. No inference SDK examined ships a loop in the same object as the model call.

**Retained server-side context is optional, additive, and never the only path.** OpenAI has three
mutually exclusive state strategies — manual replay, `previous_response_id`, and the Conversations
API — and is explicit that the retained one is not a cost saving: "Even when using
`previous_response_id`, all previous input tokens for responses in the chain are billed as input
tokens in the API"
([conversation-state](https://developers.openai.com/api/docs/guides/conversation-state)). Anthropic
has no conversation handle at all; prompt caching is a byte-exact prefix match over a prompt the
client still resends in full
([prompt-caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)).
`llama-server` and Ollama are the same: stateless requests, implicit longest-common-prefix reuse.

**On-device, the analogue of a Turn Context exists but is unnamed.** `llama-server` holds one model
and N slots over it (`-np`, auto-resolving to 4 with a unified KV). It picks a slot by longest common
prefix with a similarity floor (`tools/server/server-context.cpp:1487-1600`, `-sps` default `0.10`),
and `cache_prompt` now defaults to `true` (`tools/server/server-task.h:53`). But there is no named
conversation: `id_slot` is a routing hint, and any slot can be reclaimed by LRU. The only explicit
state primitive is file-based: `POST /slots/{id}?action=save|restore` behind `--slot-save-path`,
which calls `llama_state_seq_save_file` (`tools/server/server-context.cpp:2481-2483`) and writes raw
KV tensors to disk (~14.3 MB per 1745 tokens, per the README). Ollama has no session concept and
inherits llama.cpp's prefix cache by running `llama-server` as a subprocess with `cache_prompt: true`
(`llm/llama_server.go:1569`).

**Tool call identity, argument shape, and result routing are near-universal, with one live split.**
Every source correlates a result to a call by an id. Every source keys a tool result message or block
by that id. The split is whether arguments cross the wire as a raw JSON string or a parsed object —
see the comparison below.

**Constrained decoding is the guarantee; a forced tool call is the portable fallback.** Anthropic and
OpenAI both guarantee schema conformance and both apply the same machinery to tool arguments.
Anthropic: structured outputs "guarantee schema-compliant responses through constrained decoding",
implemented with "constrained sampling with compiled grammar artifacts"; `strict: true` on a tool is
the same feature, described as "grammar-constrained sampling"
([structured-outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs),
[strict-tool-use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use)).
OpenAI: "Under the hood, strict mode works by leveraging our structured outputs feature"
([function-calling](https://developers.openai.com/api/docs/guides/function-calling)). `llama-server`
converts a JSON schema to GBNF and masks logits to `-INFINITY` for every rejected token
(`src/llama-grammar.cpp:1353-1393`), and applies the same machinery to tool-call syntax with lazy
triggers (`common/chat.cpp:1139-1160`). Where a grammar engine is absent, the forced tool call is
what frameworks reach for — and it is the *default*, not a legacy path. See "Structured output"
below.

## Point-by-point comparison

| Spec decision | Convention | Verdict |
| --- | --- | --- |
| Return tool calls to the caller, execute none | Universal | **Follows.** No source deviates. |
| Named retained context with a warm KV cache | Optional everywhere; explicitly named only by OpenAI Conversations | **Follows, and is better-suited on-device.** |
| One resident model, serialized turns | llama-server serves N slots concurrently over one model | **Deviates. Justified** by memory and thermals on a phone. |
| Tool calls as `{id, name, arguments: string}` | Split: OpenAI/Vercel use a string; Anthropic/Ollama/MCP/LangChain use an object | **Defensible, but wrong here.** See below. |
| Tool results as a `tool` role message keyed by call id | Universal in shape; Anthropic uses a content block instead of a role | **Follows.** |
| `completed / tool_calls / length / stopped / unloaded / superseded / failed` | Provider vocabularies differ; normalizers keep a raw passthrough | **Follows, with one gap.** |
| Token counting before a turn | Server endpoints now exist at Anthropic, OpenAI, and llama.cpp | **Follows.** |
| Structured output via a forced tool call | The default in Pydantic AI; the fallback wherever no grammar engine exists | **Follows.** |
| No agent loop in the library | Universal | **Follows.** |

### Returning tool calls instead of executing them

Unambiguously conventional. The strongest structural evidence is the OpenAI Agents SDK, whose seam is
three-phase rather than two: `Model.get_response(...) -> ModelResponse` is pure transport
(`src/agents/models/interface.py:67-100` @ v0.21.1); `process_model_response` walks
`response.output` and builds a deferred plan of `ToolRunFunction` records without invoking anything
(`src/agents/run_internal/turn_resolution.py:3401-3421`); and `execute_tools_and_side_effects`
performs approvals, guardrails, hooks, and invocation. Approvals live in phase three, not in the
model layer — exactly the separation the spec's motivation section argues for.

The Vercel AI SDK supplies the closest precedent for `runTurn` specifically: **a tool with no
`execute` function is returned to the caller and terminates the loop.** `isExecutableTool` is a
`typeof tool.execute === 'function'` check (`packages/provider-utils/src/types/executable-tool.ts:13-17`);
`executeToolCall` returns `undefined` for anything that fails it
(`packages/ai/src/generate-text/execute-tool-call.ts:89-91`); and the loop contract states it
normatively — "A tool calling loop continues until one of the following conditions is met: The model
returns a finish reason other than `tool-calls`; **A tool without an execute function is called**; A
tool call needs approval; One of the provided stop conditions returns `true`"
(`packages/ai/src/generate-text/stop-condition.ts:8-12`). The docs build a deliberate idiom on it:
"A tool that has no `execute` function acts as a termination signal… The final answer is available in
`result.staticToolCalls`, which contains tool calls that weren't executed"
([loop-control](https://ai-sdk.dev/docs/agents/loop-control)). That is `runTurn`'s semantics, in a
mainstream SDK, as a supported feature rather than an escape hatch.

Two nuances worth carrying. The Agents SDK's `ModelResponse` also carries provider-executed results
for hosted tools, and its `ProcessedResponse` comments the split — "Handoffs, functions and computer
actions need local processing / Hosted tools have already run"
(`src/agents/run_internal/run_steps.py:134-135`). The Vercel provider spec encodes the same
distinction as a field: `LanguageModelV4ToolCall.providerExecuted?: boolean`, "Whether the tool call
will be executed by the provider. If this flag is not set or is false, the tool call will be executed
by the client". This package will have both kinds simultaneously — the legacy path executes tools
natively, `runTurn` does not — and nothing in the proposed outcome type says which.

Separately, note what does *not* cross the seam. In Vercel, Pydantic AI, and LangChain, the tool
definition handed to the model carries **no callable** — Vercel's `LanguageModelV4FunctionTool` is
`{type, name, description?, inputSchema, inputExamples?, strict?, providerOptions?}` and its own
comment says "this is **not** the user-facing tool definition"; Pydantic AI's `ToolDefinition` is
`{name, parameters_json_schema, description, …}` (`pydantic_ai/tools.py:544-556`). The OpenAI Agents
SDK is the exception, passing the whole `FunctionTool` object into `get_response` and relying on
discipline rather than typing. The spec's `ToolSchema` — `ToolDefinition` without `handler` — is the
majority design and the structurally safer one.

### Named retained context and the warm KV cache

The spec is right that a warm context is the correct primitive once the caller owns the loop, and it
is right that this is not in tension with caller control. `ChatSession.respond(to messages:)` is
documented for precisely this: "Use this to continue an existing session with non-user roles, such as
one or more tool results, while preserving the session's KV cache", with the accompanying note
"Initializing a new session from history must prefill that history once. Reuse the same session with
this method for subsequent tool or agent turns to avoid repeatedly pre-filling the accumulated
transcript" (`ChatSession.swift:448-456`).

What is worth internalising is *how* that cache is a transcript. `streamMap` clears its message array
immediately after templating (`ChatSession.swift:641`); the session's history exists only as KV
tensors, and the assistant's generated tokens land there because generation feeds them back. There is
no `[Chat.Message]` transcript to inspect after the first pass. This makes the risk register's "the
KV cache is not retained as expected" the correct thing to measure first, and it also means the
cancelled-turn recovery question in Open Question 2 is a real question, not a formality.

Against convention, a named handle is unusual but not unprecedented — OpenAI's Conversations API is
"a long-running object with its own durable identifier", and its items are exempt from the 30-day TTL
that applies to plain responses. The difference is that OpenAI's handle is a convenience over a
stateless path that still works, whereas here the handle is the only way to get a warm cache. That is
justified: the network APIs can afford to reprefill because prefill happens on a datacentre GPU. On a
phone, prefill over a growing transcript is the dominant cost of a tool loop, and llama.cpp's
existence proof — the entire slot and prefix-cache machinery — is a direct concession to the same
pressure.

### One resident model, serialized turns

This deviates. `llama-server` runs N slots concurrently over one model and treats that as the normal
case. But the constraint is real: MLX generation on a phone is memory- and thermally-bound, and ADR
0002 already records the trade ("Serialized turns trade parallel generation latency for bounded
memory, predictable thermal behavior, and simple cancellation"). Nothing found here argues against
it. Note that Ollama, which is the closest consumer-facing analogue, ships `OLLAMA_NUM_PARALLEL`
defaulting to **1** (`envconfig/config.go:275`) — the serialized default is what a single-user
deployment actually gets.

### Tool call arguments: raw string or parsed object

This is where the spec is wrong, and the reason is local rather than conventional.

The field is genuinely split. Raw string: OpenAI Chat Completions
(`ChatCompletionMessageFunctionToolCall.function.arguments: string`), OpenAI Responses
(`FunctionToolCall.arguments` — "A JSON string of the arguments"), the Vercel AI SDK provider layer
(`LanguageModelV4ToolCall.input: string`, "Stringified JSON object with the tool call arguments"),
LangChain's streaming chunk (`ToolCallChunk.args: str | None`, "as a JSON-parseable string"), and
`llama-server`'s OpenAI-compatible output. Parsed object: Anthropic (`tool_use.input` is an object),
Ollama (`ToolCallFunctionArguments` is an ordered map, `api/types.go:221-234`), MCP
(`CallToolRequestParams.arguments?: { [key: string]: unknown }`), and LangChain's settled
`ToolCall.args: dict[str, Any]`.

The pattern behind the split is consistent: **the string form is the streaming/provider form, and the
object form is the settled/consumer form.** Anthropic says it outright — "the deltas are *partial
JSON strings*, whereas the final `tool_use.input` is always an *object*"
([streaming](https://platform.claude.com/docs/en/build-with-claude/streaming)). LangChain models the
same distinction as two types. Vercel keeps the string at the provider seam and parses *and
schema-validates* at the core seam in one step, throwing `InvalidToolInputError` on failure
(`packages/ai/src/generate-text/parse-tool-call.ts:191-225`). Pydantic AI is the one framework that
refuses to choose — `args: str | dict[str, Any] | None` with `args_as_dict()` / `args_as_json_str()`
accessors, because "the exact bytes the model produced (key order, whitespace) matter for prompt
caching" (`pydantic_ai/messages.py:2160-2164`, `:2235`).

`runTurn` is a settled, non-streaming outcome. By that rule it should carry an object.

One counter-consideration, which turns out not to apply here: every framework preserves the raw
string on the *failure* path — Vercel's `InvalidToolInputError.toolInput`, LangChain's
`InvalidToolCall.args: str | None` alongside `error`, Pydantic AI's `{'INVALID_JSON': '<raw>'}`
degradation. That is the strongest argument for a string. But it is unavailable here for the same
reason the "as emitted by the model" claim is false: `ToolCallProcessor` parses first, so a
malformed call never becomes a `ToolCall` and never reaches the caller in any form. The package
cannot offer the failure path the string exists to serve. It should say so explicitly instead.

More decisively, the string is not achievable as described. The spec's JSDoc says "Raw JSON text as
emitted by the model. The caller parses and validates it." That is false against
`mlx-swift-lm 3.31.4`: `ToolCall.Function.arguments` is `[String: JSONValue]`
(`Tool/ToolCall.swift:12`), parsed by `ToolCallProcessor` before it is ever yielded. A malformed
tool call never becomes a `ToolCall` at all — it is dropped or falls through to text. So a raw
`arguments: string` would be a **re-serialization** of an already-parsed dictionary, with key order
and formatting the package chose, not the model. The two properties the raw-string convention exists
to preserve — fidelity to what the model emitted, and the caller's ability to see and handle
malformed output — are both unavailable. What remains is a lossy round trip that every caller
immediately reverses with `JSON.parse`.

The precedent for a string in this package (`ToolCallStartEvent.arguments`,
`package/src/specs/LLM.nitro.ts:64-69`) is a *streaming* event, which is exactly where convention puts
the string. It does not carry over to the terminal outcome.

### Tool results keyed by call id

The spec's `LLMMessage { role: 'tool', toolCallId, content }` matches the field convention. OpenAI's
`ChatCompletionRequestToolMessage` requires `role`, `content`, and `tool_call_id`. LangChain's
`ToolMessage` carries `content` and `tool_call_id`. `Chat.Message.tool(_ content:, id:)` renders to
`tool_call_id` in the templated dictionary (`Chat.swift:87-89`, `:153-154`). Anthropic is the one
structural outlier and says so: "Unlike APIs that separate tool use or use special roles like `tool`
or `function`, the Claude API integrates tools directly into the `user` and `assistant` message
structure" ([handle-tool-calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)).
Since the upstream chat templates here take `tool_call_id`, the spec's choice is correct.

Two fields the convention has that the spec drops:

- **A tool name on the result.** Ollama documents `tool_name` on a tool message — "add the name of
  the tool that was executed to inform the model of the result" (`docs/api.md:506`). The package's own
  `ToolChatMessage` already has `name` (`package/src/chat.ts:60-65`). `Chat.Message` upstream does
  not carry one, so this is a real limitation rather than an omission the package can fix alone, but
  it should be stated.
- **An error flag.** MCP, LangChain, Anthropic, Vercel, and Pydantic AI all distinguish a failed tool
  result from a successful one *in band*: `CallToolResult.isError`; `ToolMessage.status: "success" |
  "error"`; `tool_result.is_error`; Vercel's typed `ToolResultOutput` variants (`text`, `json`,
  `error-text`, `error-json`, `execution-denied`, `content`); and Pydantic AI's four-value
  `ToolReturnPart.outcome: 'success' | 'failed' | 'denied' | 'interrupted'`. MCP's schema states the
  rationale directly: "Any errors that originate from the tool SHOULD be reported inside the result
  object, with `isError` set to true, *not* as an MCP protocol-level error response. Otherwise, the
  LLM would not be able to see that an error occurred and self-correct." Pydantic AI's docstring adds
  the distinction that matters most for an approval-gated harness: "Only `'failed'` is mapped to a
  provider's native error channel… A denial is a deliberate policy decision rather than a runtime
  error, while an interruption means no result was produced" (`pydantic_ai/messages.py:1340-1343`).
  A harness that feeds a denied tool back as an ordinary string loses that signal.

### Finish reason vocabulary

The spec's seven values map cleanly onto the field. `completed`/`tool_calls`/`length` are the
standard triple (OpenAI `stop`/`tool_calls`/`length`; Vercel
`stop`/`tool-calls`/`length`; `llama-server` `stop`/`tool_calls`; Ollama `stop`/`length`).
`stopped`/`unloaded`/`superseded` are lifecycle reasons this package genuinely has and the network
APIs do not; `failed` corresponds to OpenAI Responses' `status: "failed"`. `GenerateStopReason` at
`Evaluate.swift:1956-1965` supplies `stop`, `length`, and `cancelled`, so `length` is representable.

Three observations.

First, the field has learned to keep a **raw provider value alongside the normalized one**. The
Vercel AI SDK changed `finishReason` from a bare union into
`{ unified: 'stop'|'length'|'content-filter'|'tool-calls'|'error'|'other', raw: string | undefined }`
in the V3/V4 provider spec — changelog entry `feat: expose raw finish reason` — and surfaces both at
the core level as `StepResult.finishReason` and `StepResult.rawFinishReason`. A caller here has no
way to distinguish a turn that stopped on EOS from one that stopped on an extra EOS token
(`extraEOSTokens`), and no way to see why a `failed` turn failed beyond a localized string.

Second, **finish reason is telemetry, not control flow, in most of the field.** Of the four
frameworks, only Vercel routes on it. The OpenAI Agents SDK has no stop-reason field on
`ModelResponse` at all — `finish_reason` appears only inside its ChatCompletions adapters and is
consumed and discarded there, which is arguably why one loop works across Responses, ChatCompletions,
LiteLLM, and any-llm. LangChain never normalized it: `grep -rn 'finish_reason' langchain_core/`
returns only two comments, and providers write different keys (`finish_reason` at OpenAI,
`stop_reason` at Anthropic) into an untyped `response_metadata: dict[Any, Any]`. LangGraph's
`tools_condition` is twelve lines and routes purely on `len(ai_message.tool_calls) > 0`
(`langgraph/prebuilt/tool_node.py:1657-1659`). Pydantic AI is explicit in its own docstring: the
`FinishReason` alias is "mostly normalized to OpenTelemetry semantic convention values… Whether the
agent should automatically continue is determined by `ModelResponse.state`, not by this field"
(`pydantic_ai/messages.py:113-124`). This does not argue against the spec's enum — it argues that the
enum should be treated as reportable rather than load-bearing, with `toolCalls` being the thing a
loop actually branches on.

Third, OpenAI's Responses API and Pydantic AI both split the axis the spec merges. Responses has **no
status meaning "the model wants a tool"** — a tool-calling turn is `status: "completed"`, and the
caller inspects `output[]` for `type: "function_call"` items; `status` covers lifecycle
(`completed`, `failed`, `in_progress`, `cancelled`, `queued`, `incomplete`) while
`incomplete_details.reason` covers truncation and filtering. Pydantic AI does the same with
`finish_reason` beside `ModelResponseState: 'complete' | 'incomplete' | 'suspended' | 'interrupted'`,
where `'suspended'` is first-class handling for a mid-turn provider pause (Anthropic `pause_turn`,
OpenAI background mode). Folding `tool_calls` into the same enum as `unloaded` and `superseded` mixes
those axes. It is workable here because the spec also populates `toolCalls`, but the two-field
pattern — lifecycle plus reason — is where the field has moved.

### Token counting before a turn

Squarely conventional, and newly so on all three comparable surfaces.
Anthropic: `POST /v1/messages/count_tokens`, accepting "the same structured list of inputs for
creating a message, including support for system prompts, tools, images, and PDFs", returning
`input_tokens`, free, with separate rate limits, and explicitly "an **estimate**"
([token-counting](https://platform.claude.com/docs/en/build-with-claude/token-counting)).
OpenAI now has `POST /v1/responses/input_tokens`, which "returns the exact count the model will
receive", including formatting tokens, and whose guide states plainly that local tokenizers cannot
count tools and schemas correctly. `llama-server` exposes `/apply-template` plus `/tokenize`, and also
`POST /v1/chat/completions/input_tokens` and `/v1/messages/count_tokens`.

Putting it on the model object rather than in a utility is also conventional: Pydantic AI's abstract
`Model` carries an optional `count_tokens(...)` method alongside `request(...)`
(`pydantic_ai/models/__init__.py:518`). The spec's design is implementable exactly as written:
`applyChatTemplate(messages:tools:additionalContext:)` is on the `Tokenizer` protocol
(`Tokenizer.swift:16-20`), so the count includes tool schemas and template scaffolding rather than
bare text. Milestone 2's exit criterion (agreement with the real prefill count within a stated
tolerance) matches Anthropic's own "estimate" hedge and is the right bar.

### Structured output via a forced tool call

The spec is more conventional here than it gives itself credit for.

The clearest evidence is Pydantic AI, which supports three output modes and makes the forced tool
call the **default**: "This is the default as it's supported by virtually all models and has been
shown to work very well." Native Output — the provider's own constrained decoding — is documented as
"not supported by all models, and sometimes comes with restrictions". Prompted Output, which injects
the schema into instructions and parses the reply, is "often the least reliable"
([pydantic.dev/docs/ai/core-concepts/output](https://pydantic.dev/docs/ai/core-concepts/output/)).
That is exactly the ranking the spec assumes, from a framework that has to work across every backend.
The default is set in one place (`pydantic_ai/profiles/__init__.py:224`,
`'default_structured_output_mode': 'tool'`) and no model profile overrides it.

For balance: the Vercel AI SDK defaults the other way, driving structured output through the
provider's native `responseFormat: { type: 'json', schema }`
(`packages/provider/src/language-model/v4/language-model-v4-call-options.ts:66-83`). That is the
right default when every supported provider has a grammar engine. It is not this package's situation.

OpenAI likewise does **not** frame the two as old and new. Its guide splits them by purpose: "If you
are connecting the model to tools, functions, data, etc. in your system, then you should use function
calling — If you want to structure the model's output when it responds to the user, then you should
use a structured `text.format`." The thing OpenAI deprecates is **JSON mode**
(`{"type": "json_object"}`), not the forced tool call.

Anthropic has moved further: its tool-use overview no longer contains the old "JSON mode via a single
tool" advice, and points instead at `strict: true`. But the successor is not "stop using tools" — it
is "keep the tool and grammar-constrain its arguments". The two approaches converge once a grammar
engine exists.

That is the honest framing for this spec: **the forced tool call is the correct choice for a backend
with no grammar engine, and the thing that would replace it is not a different API shape but a
constrained decoder behind the same shape.** `mlx-swift-lm 3.31.4` has no grammar support — no
`tool_choice`, no `json_schema`, no `grammar`, and the only `LogitProcessor` conformances are the
penalty processors (`Evaluate.swift:389`, `:423`, `:452`, `:482`). `mlx-lm` (Python) is the same:
`logits_processors` exists as a hook, `make_logits_processors` builds only a logit-bias and a
repetition penalty, and grepping `generate.py` and `sample_utils.py` for `grammar`, `xgrammar`,
`outlines`, `json_schema`, and `bitmask` returns zero hits. Its `tool_parsers/` directory parses tool
calls out of generated text — the opposite of constraining them.

Two caveats the spec should absorb.

First, "requires the model to call it" overstates what is available. There is no `tool_choice` in
`mlx-swift-lm`, and `additionalContext` feeds arbitrary template variables, not a forcing mechanism.
The synthetic tool can be *offered as the only tool* and asked for in the prompt; it cannot be
*required*. Every source that offers real forcing does so as a first-class request parameter
(`tool_choice: "required"` or a named tool at OpenAI; `{"type":"tool","name":...}` at Anthropic;
`{type:'required'}` / `{type:'tool',toolName}` in Vercel). llama.cpp's grammar for tool calls is
*non-lazy* exactly when `tool_choice: required` is set (`common/chat.cpp:1139-1160`) — forcing and
constraining are the same act there.

Second, no source claims constrained decoding guarantees a *correct* answer, only a well-formed one.
Google states it most plainly: "While output is syntactically correct JSON, always validate values in
your application"
([ai.google.dev/gemini-api/docs/structured-output](https://ai.google.dev/gemini-api/docs/structured-output)).
OpenAI: "Structured Outputs can still contain mistakes." So the spec's typed `stage: 'schema'`
failure is the right primitive whether or not a grammar arrives later — a caller needs a retry path
regardless.

### The library/harness boundary

Comparable libraries stop where this one stops, and several stop earlier. `llama-server` stops at
returning `tool_calls` and actively warns against using its execution endpoint downstream. Ollama's
HTTP server stops there too; its agent loop, approval prompter, `MaxToolRounds` guard and compactor
all live in `agent/`, reachable only from `cmd/`. Anthropic ships the loop as a *separate product*.

The loop budget in particular is uniformly a harness concern, never a model-call parameter: OpenAI
Agents `max_turns` defaults to 10 on the Runner (`run_config.py:44`); Pydantic AI's
`UsageLimits.request_limit` defaults to 50 and is checked before each request
(`pydantic_ai/usage.py:418-440`, `:492-496`); LangGraph's prebuilt agent carries
`remaining_steps: RemainingSteps = 25` in graph *state* and substitutes a canned message on
exhaustion (`chat_agent_executor.py:74`, `:684-691`); Vercel's `generateText` defaults to
`stopWhen: isStepCount(1)` — no loop at all — while its `ToolLoopAgent` class defaults to 20
(`packages/ai/src/agent/tool-loop-agent.ts:132`). The spec's decision to queue nothing and count
nothing is consistent with all of them.

The one thing worth noting is that most of these stop *further back* than this package will: they
have no equivalent of a Turn Context, so their "primitive" is smaller. That makes the spec's
vocabulary defence (`CONTEXT.md` no longer defines Agent, Handoff, or Artifact; ADR 0002 records the
line) more load-bearing than it might look, because Turn Contexts move the boundary closer to the
harness than any comparable library sits.

## Deviations, and whether they are justified

**Justified.**

- *One resident model, serialized turns.* Deviates from llama.cpp's N-slot concurrency. ADR 0002
  records the reason and nothing found contradicts it. Ollama's own default of one parallel slot is
  supporting evidence.
- *A named context handle as the only warm path.* Deviates from the stateless-plus-prefix-cache norm.
  On-device prefill cost justifies it, and llama.cpp's slot machinery is the same concession made
  under a different name.
- *No agent loop.* Not a deviation at all.
- *Structured output via a forced tool call.* Not a deviation. It is Pydantic AI's default and the
  documented fallback wherever no grammar engine exists.

**Mistakes.**

- *`arguments` as a raw JSON string, documented as "as emitted by the model".* The claim is false
  against `mlx-swift-lm`, which parses before yielding (`Tool/ToolCall.swift:12`). The string would be
  a re-serialization, so it delivers neither fidelity nor visibility into malformed output, and every
  caller reverses it immediately. The settled form should be an object; the string belongs on the
  streaming event, where the package already has it.
- *`LLMMessage` cannot express an assistant message with tool calls.* `role`, `content`, and an
  optional `toolCallId` cannot represent the assistant turn that produced the calls. Two things follow.
  The `history` field on a cold `LLMTurnRequest` cannot seed a transcript that contains a tool
  exchange at all. And the package's own `AssistantChatMessage` already has `toolCalls`
  (`package/src/chat.ts:49-58`), so `runTurn` regresses against `ChatSession`.
- *The "cold turns cannot be continued" limitation is overstated.* The spec says the caller "would
  have to rebuild that assistant message itself — in a format that is chat-template-specific per model
  family." That is not what upstream requires. `Chat.Message.assistant(_ content:, toolCalls:)` exists
  (`Chat.swift:67-76`), and `MessageGenerator.addToolMetadata` renders it into a model-agnostic
  `tool_calls` array that the model's own Jinja template consumes (`Chat.swift:137-158`). Template
  specificity is handled by the template. A cold turn *can* be faithfully continued; the gap is that
  `LLMMessage` has nowhere to put the tool calls. Fixing the type removes the limitation.
- *`ToolSchema` cannot express a real tool schema.* `ToolParameter` is flat —
  `{ name, type: string, description, required }` (`package/src/specs/LLM.nitro.ts:191-196`) — with no
  nesting, no `items`, no `enum`, no `properties`. The Zod bridge is lossy: a nested object becomes
  `type: 'object'` with no shape, an array becomes `type: 'array'` with no `items`, and an enum
  becomes `'string'` (`package/src/tool-utils.ts:15-56`). Every convention source uses full JSON
  Schema — MCP requires `type: "object"` at the root and, since revision `2026-07-28`, permits any
  JSON Schema 2020-12 keyword; Anthropic uses `input_schema`; OpenAI uses `parameters`. Upstream
  already supports more than the bridge exposes: `ToolParameterType` is an `indirect enum` with
  `.array(elementType:)` and `.object(properties:)`, and `ToolParameter` carries `extraProperties`
  (`Tool/ToolParameter.swift:3-42`), and `Tool(schema:handler:)` accepts a raw JSON Schema dictionary.
  This also breaks Milestone 3: `responseSchema` is a serialized JSON Schema string, so schema-to-
  synthetic-tool conversion has to reach past `ToolSchema` for anything nested — which means the
  package would support richer schemas for structured output than for tools.
- *`responseSchema` is described as "required" of the model.* No forcing mechanism exists in
  `mlx-swift-lm`. The spec should say the synthetic tool is the only tool offered and the model is
  asked to call it.

**Not wrong, but worth correcting.**

- The spec cites `ChatSession.swift:757` for the `toolDispatch: nil` path. In 3.31.4 (commit
  `bd4b743`, 2026-06-29) the condition is at `:760`, inside `streamMap` at `:567`. The behaviour is as
  described: `if let toolCall = item.toolCall, toolDispatch != nil` collects for dispatch, and
  otherwise the item falls through to `transform(item)` and reaches the `streamDetails` caller.
  `HybridLLM.swift:756` and `:945` both check out.

## Concrete recommendations for the spec

Ordered by how much they change the interface.

**1. Make `LLMMessage` a discriminated union that can carry tool calls.** This is the largest gap and
it unblocks three other items.

```ts
export type LLMMessage =
  | { role: 'system' | 'user'; content: string }
  | { role: 'assistant'; content: string; toolCalls?: LLMToolCall[] }
  | { role: 'tool'; toolCallId: string; name?: string; content: string; isError?: boolean }
```

Evidence: `Chat.Message.assistant(_, toolCalls:)` and `addToolMetadata` already render this
(`Chat.swift:67-76`, `:137-158`); OpenAI's `ChatCompletionRequestToolMessage` and LangChain's
`ToolMessage` are the same shape; the package's own `ChatMessage` union already has it
(`package/src/chat.ts:34-73`). Adding `isError` follows MCP's stated rationale — a tool failure the
model cannot see is a failure it cannot recover from. If a boolean feels too thin for an
approval-gated harness, Pydantic AI's `outcome: 'success' | 'failed' | 'denied' | 'interrupted'` is
the richer precedent, and it maps only `'failed'` to a provider error channel. Then delete the "cold
turns cannot be continued" limitation, or narrow it to the cases the template genuinely cannot
express.

**2. Return parsed arguments, and keep the raw string only where it is real.**

```ts
export interface LLMToolCall {
  id: string
  name: string
  /** Parsed arguments. The native parser produced these; malformed output never reaches here. */
  arguments: AnyMap
}
```

Evidence: `ToolCall.Function.arguments: [String: JSONValue]` (`Tool/ToolCall.swift:12`); the
settled-object / streaming-string split at Anthropic
([streaming](https://platform.claude.com/docs/en/build-with-claude/streaming)) and LangChain
(`ToolCall.args: dict` vs `ToolCallChunk.args: str`); MCP's `arguments?: { [key: string]: unknown }`;
Ollama's ordered map (`api/types.go:221-234`). If Nitro cannot carry a map cleanly, a string is an
acceptable *wire* compromise — but the JSDoc must stop claiming it is what the model emitted, and the
TypeScript wrapper should parse it before returning, as it already does for `StreamEventEnvelope`.
Keep `id` non-optional: `ToolCallProcessor` always assigns one (`Tool/ToolCallProcessor.swift:453-468`,
`Tool/ToolCallFormat.swift:139-148`), and Pydantic AI does the same on purpose — `tool_call_id` has a
`default_factory` so "in case the tool call id is not provided by the model, Pydantic AI will generate
a random one" (`pydantic_ai/messages.py:2166-2170`), which makes the correlation invariant total
rather than conditional.

**3. Take a JSON Schema for tools, not a flat parameter list.** Add
`ToolSchema { name, description, parameters: string /* serialized JSON Schema */ }`, or at minimum
extend `ToolParameter` with `items`, `properties`, and `enum`. Evidence: MCP `2026-07-28` allows any
JSON Schema 2020-12 keyword in `inputSchema`; Anthropic `input_schema`; OpenAI `parameters`; and
upstream already models nesting (`Tool/ToolParameter.swift:3-35`) and accepts a raw schema dictionary
(`Tool/Tool.swift:60-63`). Without this, `responseSchema` and `tools` support different schema
languages in the same package, and Milestone 3's conversion step has no target type.

**4. Add usage accounting to the turn outcome.** `GenerationStats` today is
`{ tokenCount, tokensPerSecond, timeToFirstToken, totalTime, toolExecutionTime }`
(`package/src/specs/LLM.nitro.ts:6-12`) — no prompt/completion split, so a harness cannot attribute
context growth to prefill versus generation, and `countTokens` cannot be checked against a real turn
(which Milestone 2 requires). Every comparison source splits them: OpenAI
`prompt_tokens`/`completion_tokens` with `prompt_tokens_details.cached_tokens`; Anthropic
`cache_creation_input_tokens`/`cache_read_input_tokens`; Vercel
`{inputTokens: {total, noCache, cacheRead, cacheWrite}, outputTokens: {total, text, reasoning}}`.
`GenerateCompletionInfo` already carries `promptTokenCount` and `generationTokenCount`
(`Evaluate.swift:1971-1975`); expose both, plus a cached-prefix count if it can be derived, since that
is the number that proves a Turn Context is warm.

**5. Keep a raw finish reason alongside the normalized one, and treat the enum as reportable rather
than load-bearing.** Add `rawFinishReason?: string` (or the Vercel shape, `{ unified, raw }`).
Evidence: `LanguageModelV4FinishReason` in `vercel/ai` @ `a0b1ffc` was changed to exactly this so
normalization is not lossy, and the core surfaces both as `StepResult.finishReason` and
`StepResult.rawFinishReason`. Document that a caller's loop should branch on `toolCalls.length`, not
on `finishReason` — that is what three of the four frameworks examined actually do. Also consider
whether `tool_calls` belongs in the same enum as `unloaded` and `superseded`; OpenAI Responses
separates lifecycle `status` from `incomplete_details.reason`, and Pydantic AI separates
`finish_reason` from `ModelResponseState`, for this reason.

**6. Add streaming events for tool-call arguments.** `runTurn` today surfaces tool calls only in the
terminal outcome, so a harness cannot render "calling `search_files`…" before the turn ends. Every
streaming source emits them progressively: Anthropic `input_json_delta` with `partial_json`; OpenAI
`ChatCompletionMessageToolCallChunk` keyed by `index`, and `response.function_call_arguments.delta`
on Responses; Vercel `tool-input-start` / `tool-input-delta` / `tool-input-end`. The package already
has `ToolCallStartEvent { id, name, arguments: string }` — reuse it (with `arguments` accumulating)
rather than inventing a new shape, and note that `ToolCallProcessor` buffers until a complete call is
parsed, so per-token deltas may not be achievable without upstream changes. State that explicitly if
so.

**7. Say what the package does about parallel tool calls.** The spec's `toolCalls` is an array and
the example loop uses `Promise.all`, but nothing states whether several calls in one pass are
expected, whether results must be returned in order, or whether a caller may return a subset.
Convention is firm: OpenAI's `parallel_tool_calls` defaults to `true`; Anthropic requires all
`tool_result` blocks in a single user message and warns that "Tool result blocks must immediately
follow their corresponding tool use blocks in the message history"
([handle-tool-calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)).
The package already has `LLMToolExecution = 'parallel' | 'sequential'` on the legacy path. State the
`runTurn` contract, and state what happens if the caller returns fewer results than there were calls.

**8. Add `toolChoice` to `LLMTurnRequest`, or say why it is absent.** `'auto' | 'none' | 'required' |
{ name }` is first-class at OpenAI, Anthropic, and Vercel, and it is how `responseSchema` would be
implemented properly. `mlx-swift-lm 3.31.4` has no such parameter — grepping `Libraries/` for
`tool_choice`, `toolChoice`, `json_schema`, `response_format`, and `grammar` returns nothing — so
"deferred, pending upstream support" is a fine answer. Silence is not, because it makes the
"requires the model to call it" sentence in the structured-output section read as a capability.

**9. Fill in the generation-config gaps a harness needs.** `LLMGenerationConfig` has `maxTokens` and
`temperature`/`topP` but no `seed`, `topK`, `minP`, or stop sequences
(`package/src/specs/LLM.nitro.ts:152-173`). Upstream `GenerateParameters` gained an optional `seed`
in the checked-out commit ("Add optional seed to GenerateParameters for reproducible sampling",
`#377`), and already has `topK` and `minP`. Reproducibility matters for a harness that replays a
failed turn; OpenAI ships `seed` (Beta, "Determinism is not guaranteed") and llama.cpp ships it per
request. Stop sequences have no upstream equivalent beyond `extraEOSTokens` on the model config —
worth stating as a known gap rather than leaving unaddressed.

**10. Decide and document the thinking-content contract across turns.** `LLMTurnOutcome.thinking` is
returned, but nothing says whether it stays in the Turn Context's KV cache on the next turn, or
whether a caller replaying `history` on a cold turn should include it. This is a live correctness
question elsewhere: OpenAI "highly recommend[s] you pass back any reasoning items returned with the
last function call… pass back all reasoning items, function call items, and function call output
items, since the last `user` message"
([reasoning](https://developers.openai.com/api/docs/guides/reasoning)); Anthropic requires thinking
blocks be echoed back unchanged on the same model. Here the answer is probably "the KV cache keeps
whatever the model generated, including thinking tokens" — which is a *different* answer from the
network APIs and has a memory cost per context that Open Question 1 should measure separately.

**11. Note the two upstream primitives the spec lists as unavailable but which exist.**
`saveCache(to:)` / `loadPromptCache(url:)` are already cited. Also present:
`canTrimPromptCache` / `trimPromptCache(_:numTokens:)` (`KVCache.swift:1862`, `:1871`), which is the
mechanism a harness would need to trim a Turn Context that outgrows its window without discarding it.
Worth listing under stated limitations as "available upstream, not exposed", so the boundary is a
choice on the record rather than an assumed absence.

**12. Record the constrained-decoder path concretely, since it is closer than the spec assumes.**
The spec defers a real grammar decoder as "a project of its own". It is a smaller project than that
implies, because the raw material may not need to be written. XGrammar (mlc-ai/xgrammar, v0.2.5,
2026-07-22, Apache-2.0) ships a root `Package.swift` declaring an `XGrammar` SwiftPM library product
for macOS 14 / iOS 17 that compiles the C++ core directly, excluding `python` and `cpp/tvm_ffi` —
so it links with no Python and no CMake step. Its matcher API is exactly the shape a
`LogitProcessor` needs: `FillNextTokenBitmask(DLTensor*)`, `AcceptToken(int32_t)`, `Rollback(int)`
(`include/xgrammar/matcher.h:97`, `:116`, `:159`). Its README names it "the default structured
generation backend for … vLLM, SGLang, TensorRT-LLM, and MLC-LLM". llguidance
(guidance-ai/llguidance, v1.0.0) is the alternative, with a C header at `parser/llguidance.h` and a
claimed "~50μs of CPU time per token (for 128k tokenizer)" — easier to bridge from Swift, but it
requires vendoring a Rust static library.

Neither has been evaluated on-device here, and the XGrammar paper (arXiv 2411.15100) was read only at
abstract level, so no performance figure should be quoted from this document. The point is narrower:
"a project of its own" should become "vendor XGrammar behind the existing `LogitProcessor` protocol",
which is a different size of decision and worth recording as such.

## What could not be verified

- The XGrammar paper (arXiv 2411.15100) and the Outlines paper (arXiv 2307.09702) were read at
  abstract level only. No technique detail or performance number from either should be treated as
  verified. The XGrammar abstract states "up to 100x" over unnamed existing solutions; the Outlines
  abstract says only that the approach "adds little overhead". The frequently-cited O(1)-per-token
  claim for Outlines was **not** confirmed from the paper.
- Whether vLLM and SGLang apply guided decoding to *tool arguments* as well as to response format was
  not confirmed from their docs. llama.cpp demonstrably does (`common/chat.cpp:1139-1160`), and both
  Anthropic and OpenAI state it for their own APIs.
- Apple's Foundation Models guided generation was not investigated at all. Nothing here should be
  read as evidence that it can or cannot constrain a user-supplied MLX model.
- OpenAI's current documentation contains no prose sentence requiring that the assistant `tool_calls`
  message be replayed in Chat Completions. The requirement is implied by `tool_call_id` being required,
  and is stated explicitly only for the Responses API. No such sentence is quoted here.
- llguidance's README claims it powers OpenAI's Structured Outputs. That is a claim by llguidance and
  was not confirmed against OpenAI.
- Per-Turn-Context memory cost, cancelled-cache recoverability, and cold-versus-warm turn time on this
  package's baseline device are measurements, not literature. Milestones 0 and 1 remain the only way
  to settle them.
- The LangChain and LangGraph checkouts read here were depth-1, so no history or blame claim is made
  about them. No enumerated `Literal` of `finish_reason` values exists anywhere in `langchain-openai`;
  the value sets are provider API contracts, not repository facts.
- Claims about the OpenAI Agents SDK are scoped to the `Runner` text loop. `src/agents/realtime/`,
  `voice/`, and `sandbox/` contain their own loops that were not examined.

## Sources

Read 2026-08-17 unless stated.

**Anthropic** (platform.claude.com; note `docs.anthropic.com` and `docs.claude.com` now 30x-redirect here)
- https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
- https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls
- https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use
- https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- https://platform.claude.com/docs/en/build-with-claude/streaming
- https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons
- https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- https://platform.claude.com/docs/en/build-with-claude/token-counting
- https://code.claude.com/docs/en/agent-sdk/overview

**OpenAI** (developers.openai.com; `platform.openai.com/docs/*` 301-redirects here). OpenAPI spec
`openai/openai-openapi` v2.3.0 @ `2186421`, 2026-08-15. Agents SDK @ tag `v0.21.1`, 2026-08-16.
- https://developers.openai.com/api/docs/guides/function-calling
- https://developers.openai.com/api/docs/guides/structured-outputs
- https://developers.openai.com/api/docs/guides/conversation-state
- https://developers.openai.com/api/docs/guides/reasoning
- https://developers.openai.com/api/docs/guides/token-counting
- https://developers.openai.com/api/reference/resources/chat.md
- `openai/openai-agents-python` `src/agents/models/interface.py:67-100`, `:102-135`, `:138-161`;
  `src/agents/items.py:712-745`; `src/agents/usage.py:195-229`; `src/agents/run_config.py:44`;
  `src/agents/run.py:939`, `:1296-1321`; `src/agents/run_internal/turn_resolution.py:3401-3421`;
  `src/agents/run_internal/run_steps.py:134-135`; `src/agents/tool.py:455`, `:2118-2129`

**llama.cpp** — `ggml-org/llama.cpp` master @ `01818e49`, 2026-08-17
- `tools/server/server-context.cpp:1487-1600` (slot selection), `:2481-2483` (`llama_state_seq_save_file`),
  `:3122-3196` (prefix reuse), `:3134-3192` (`--cache-reuse`), `:4534-4562` (`/slots` dispatch)
- `tools/server/server-task.h:53` (`cache_prompt = true`), `:607-628` (RAM prompt cache)
- `tools/server/server-task.cpp:426`, `:468` (`finish_reason: "tool_calls"`)
- `tools/server/server-common.cpp:1140-1147` (`--jinja` gate), `:1162-1176` (`response_format`)
- `tools/server/server-schema.cpp:251-280` (`json_schema` → GBNF)
- `src/llama-grammar.cpp:1353-1393` (logit masking); `common/chat.cpp:1139-1160` (lazy tool grammars)
- `common/sampling.cpp:594-660`; `common/arg.cpp:2534-2542`, `:3518-3532`, `:3559-3566`
- `tools/server/README.md` lines 199-237, 494-580, 670-731, 1141-1190, 1533-1560, 1636-1640

**Ollama** — `ollama/ollama` main @ `d67ad834`, 2026-08-15, tag `v0.32.14`
- `api/types.go:197-207` (message), `:221-234` + `:298-303` (arguments as an object), `:1243-1270`
- `llm/server.go:249-267` (`DoneReason`); `server/routes.go:417`, `:488`, `:2495`, `:2633`
- `llm/llama_server.go:1569`, `:1583-1598`, `:2155`, `:2308-2336`; `envconfig/config.go:126-144`, `:275`
- `server/prompt.go:23-53`; `docs/api.md:59`, `:499-506`, `:517-525`, `:906-911`, `:1085-1099`
- `agent/session.go:18-50`, `agent/approval.go` (CLI-only agent loop)

**MCP** — revision `2026-07-28` (`/specification/latest` 307-redirects to it)
- `schema/2026-07-28/schema.ts` — `Tool` (:1973), `CallToolRequestParams` (:1863), `CallToolResult` (:1809),
  `ToolAnnotations` (:1912), `ContentBlock` (:2305), `RequestId` (:261), deprecated `ToolUseContent` /
  `ToolResultContent` (~:2400)
- `docs/specification/2026-07-28/server/tools.mdx`; `.../basic/index.mdx`; `.../changelog.mdx`

**Agent frameworks**
- Vercel AI SDK — `vercel/ai` @ `a0b1ffc`, 2026-08-17; published `ai@7.0.66`,
  `@ai-sdk/provider@4.0.7`. The provider spec is versioned with the major (`ai@5 → V2`, `@6 → V3`,
  `@7 → V4`) and all three ship side by side.
  `packages/provider/src/language-model/v4/` — `language-model-v4.ts:8-61`,
  `language-model-v4-finish-reason.ts:8-33`, `language-model-v4-tool-call.ts:6-41`,
  `language-model-v4-tool-choice.ts`, `language-model-v4-usage.ts:6-58`,
  `language-model-v4-function-tool.ts:8-53`, `language-model-v4-call-options.ts:66-83`, `:96`,
  `language-model-v4-stream-part.ts:52-96`, `language-model-v4-prompt.ts:26-245`,
  `language-model-v4-generate-result.ts`
  `packages/ai/src/` — `generate-text/stop-condition.ts:8-19`, `generate-text/execute-tool-call.ts:89-91`,
  `generate-text/parse-tool-call.ts:189-225`, `generate-text/step-result.ts:240`, `:245`,
  `generate-text/generate-text.ts:249`, `:1434-1444`, `agent/tool-loop-agent.ts:132`,
  `types/language-model.ts:75-106`
  `packages/provider-utils/src/types/executable-tool.ts:13-17`,
  `packages/provider-utils/src/types/content-part.ts:218-305`
  Docs: https://ai-sdk.dev/docs/agents/loop-control
- OpenAI Agents SDK — see the OpenAI block above; additionally
  `src/agents/run_internal/tool_execution.py:2049-2065`, `run_internal/run_loop.py:317-324`, `:1148`,
  `run_internal/oai_conversation.py:518-598`, `src/agents/tool.py:1829-1853`, `:2612-2622`,
  `src/agents/tool_context.py:51`, `src/agents/models/chatcmpl_converter.py:242-249`,
  `src/agents/models/openai_responses.py:967-968`, `:2195-2214`, `src/agents/items.py:76-86`,
  `:306-313`, `:467-473`, `:693-709`, `:925-929`
- Pydantic AI — 2.31.0, repo @ `b3cdbc96`, 2026-08-17;
  `pydantic_ai_slim/pydantic_ai/models/__init__.py:174-238`, `:280-289`, `:505-565`, `:633`;
  `pydantic_ai/messages.py:113-138`, `:1292-1344`, `:2160-2170`, `:2209`, `:2235`, `:2523-2600`,
  `:3554-3559`; `pydantic_ai/tools.py:544-556`; `pydantic_ai/output.py:43`, `:49`;
  `pydantic_ai/profiles/__init__.py:224`; `pydantic_ai/usage.py:95-99`, `:418-440`, `:492-496`;
  `pydantic_ai/_agent_graph.py:1106`, `:1376-1415`, `:1816-1841`; `pydantic_ai/models/openai.py:800`,
  `:3138-3142`; docs https://pydantic.dev/docs/ai/core-concepts/output/ (`ai.pydantic.dev`
  301-redirects here)
- LangChain / LangGraph — `langchain-core 1.5.6`, `langgraph 1.2.11`, `langgraph-prebuilt 1.1.0`;
  `libs/core/langchain_core/messages/tool.py:67-88`, `:206-239`, `:261-300`, `:349-412`, `:415-418`;
  `messages/ai.py:104-176`, `:521`, `:557`, `:652-731`; `messages/content.py:336-369`;
  `language_models/chat_models.py:2198-2250`, `:2355-2361`; `tools/base.py:1135`, `:1266`,
  `:1370-1372`, `:1422-1428`; `runnables/config.py:171`
  `libs/prebuilt/langgraph/prebuilt/tool_node.py:383-391`, `:622`, `:820-823`, `:954-958`,
  `:1266-1277`, `:1582-1659`; `prebuilt/chat_agent_executor.py:74`, `:278`, `:644`, `:661-693`,
  `:684-691`, `:831-859`, `:990`; `libs/langgraph/langgraph/_internal/_config.py:32`;
  `langgraph/graph/message.py:215-234`;
  `libs/partners/openai/langchain_openai/chat_models/base.py:1063`, `:1456-1457`, `:1864-1867`,
  `:4290-4313`; `libs/partners/anthropic/langchain_anthropic/chat_models.py:1913-1934`

**Structured output engines**
- XGrammar — `mlc-ai/xgrammar` v0.2.5 (2026-07-22); root `Package.swift`;
  `include/xgrammar/matcher.h:22`, `:32`, `:67`, `:97`, `:116`, `:159`, `:226`; README; arXiv 2411.15100 (abstract only)
- llguidance — `guidance-ai/llguidance` v1.0.0; README; `parser/llguidance.h`
- Outlines — arXiv 2307.09702 (abstract only)
- Google Gemini — https://ai.google.dev/gemini-api/docs/structured-output

**mlx-swift-lm** — 3.31.4, commit `bd4b743`, 2026-06-29 (checkout at
`example/ios/build/SourcePackages/checkouts/mlx-swift-lm`)
- `Libraries/MLXLMCommon/ChatSession.swift:145` (class), `:161` (`toolDispatch`), `:448-456`
  (`respond(to messages:)` docs), `:567-641` (`streamMap`, `messages.removeAll()`), `:760-783`
  (tool-call bypass and dispatch), `:823` (`clear`), `:833` (`synchronize`), `:861` (`saveCache`)
- `Libraries/MLXLMCommon/Chat.swift:67-76` (`.assistant(_, toolCalls:)`), `:87-96` (`.tool`, roles),
  `:137-158` (`addToolMetadata`)
- `Libraries/MLXLMCommon/Tool/ToolCall.swift:5-39`; `Tool/Tool.swift:5-63`;
  `Tool/ToolParameter.swift:3-42`; `Tool/ToolCallProcessor.swift:449-468`; `Tool/ToolCallFormat.swift:139-148`
- `Libraries/MLXLMCommon/Evaluate.swift:34-44` (`LogitProcessor`), `:389`/`:423`/`:452`/`:482`
  (only penalty conformances), `:1956-1965` (`GenerateStopReason`), `:1970-2004`
  (`GenerateCompletionInfo`), `:2052-2094` (`Generation`)
- `Libraries/MLXLMCommon/Tokenizer.swift:16-20` (`applyChatTemplate`)
- `Libraries/MLXLMCommon/KVCache.swift:1592`, `:1637`, `:1862`, `:1871`
- `skills/mlx-swift-lm/references/tool-calling.md`, `kv-cache.md`, `tokenizer-chat.md`
- mlx-lm (Python) — `ml-explore/mlx-lm` README; `mlx_lm/generate.py:304`, `:472`, `:1112`;
  `mlx_lm/sample_utils.py:72-126`

**This repository**
- `package/src/specs/LLM.nitro.ts:6-12`, `:14-19`, `:64-69`, `:144-147`, `:152-173`, `:191-206`
- `package/src/chat.ts:34-73`; `package/src/tool-utils.ts:15-56`
- `package/ios/Sources/HybridLLM.swift:756`, `:938-975`
</content>
</invoke>
