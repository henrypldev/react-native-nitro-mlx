import Foundation
import NitroModules
internal import MLX
internal import MLXLLM
internal import MLXLMCommon
internal import Tokenizers

private enum MainActorSync {
    static func read<T>(_ body: @escaping @MainActor () -> T) -> T {
        if Thread.isMainThread {
            return MainActor.assumeIsolated(body)
        }

        let semaphore = DispatchSemaphore(value: 0)
        var result: T!

        Task { @MainActor in
            result = body()
            semaphore.signal()
        }

        semaphore.wait()
        return result!
    }

    static func write(_ body: @escaping @MainActor () -> Void) {
        if Thread.isMainThread {
            MainActor.assumeIsolated(body)
            return
        }

        let semaphore = DispatchSemaphore(value: 0)

        Task { @MainActor in
            body()
            semaphore.signal()
        }

        semaphore.wait()
    }

    static func run(_ body: @escaping @MainActor () throws -> Void) throws {
        if Thread.isMainThread {
            try MainActor.assumeIsolated(body)
            return
        }

        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<Void, Error>!

        Task { @MainActor in
            result = Result {
                try body()
            }
            semaphore.signal()
        }

        semaphore.wait()
        try result.get()
    }
}

class HybridLLM: HybridLLMSpec {
    private let core: HybridLLMCore

    override init() {
        core = MainActorSync.read {
            HybridLLMCore()
        }
    }

    var isLoaded: Bool {
        MainActorSync.read { self.core.isLoaded }
    }

    var isGenerating: Bool {
        MainActorSync.read { self.core.isGenerating }
    }

    var modelId: String {
        MainActorSync.read { self.core.modelId }
    }

    var turnContextIds: [String] {
        MainActorSync.read { self.core.turnContextIds }
    }

    var debug: Bool {
        get {
            MainActorSync.read { self.core.debug }
        }
        set {
            MainActorSync.write { self.core.debug = newValue }
        }
    }

    var systemPrompt: String {
        get {
            MainActorSync.read { self.core.systemPrompt }
        }
        set {
            MainActorSync.write { self.core.systemPrompt = newValue }
        }
    }

    func load(modelId: String, options: LLMLoadOptions?) throws -> Promise<Void> {
        Promise.async { [core] in
            try await core.load(modelId: modelId, options: options)
        }
    }

    func generate(prompt: String) throws -> Promise<LLMGenerationOutcome> {
        Promise.async { [core] in
            try await core.generate(prompt: prompt)
        }
    }

    func stream(
        prompt: String,
        onToken: @escaping (String) -> Void,
        onToolCall: ((String, String) -> Void)?
    ) throws -> Promise<LLMGenerationOutcome> {
        Promise.async { [core] in
            try await core.stream(prompt: prompt, onToken: onToken, onToolCall: onToolCall)
        }
    }

    func streamWithEvents(
        prompt: String,
        onEvent: @escaping (StreamEventEnvelope) -> Void
    ) throws -> Promise<LLMGenerationOutcome> {
        Promise.async { [core] in
            try await core.streamWithEvents(prompt: prompt, onEvent: onEvent)
        }
    }

    func createTurnContext(options: LLMTurnContextOptions?) throws -> Promise<String> {
        Promise.async { [core] in
            try await core.createTurnContext(options: options)
        }
    }

    func releaseTurnContext(id: String) throws {
        try MainActorSync.run { self.core.releaseTurnContext(id: id) }
    }

    func releaseAllTurnContexts() throws {
        try MainActorSync.run { self.core.releaseAllTurnContexts() }
    }

    func runTurn(
        request: LLMTurnRequest,
        onEvent: @escaping (StreamEventEnvelope) -> Void
    ) throws -> Promise<LLMTurnOutcome> {
        Promise.async { [core] in
            try await core.runTurn(request: request, onEvent: onEvent)
        }
    }

    func countTokens(request: LLMTokenCountRequest) throws -> Promise<Double> {
        Promise.async { [core] in
            try await core.countTokens(request: request)
        }
    }

    func stop() throws {
        try MainActorSync.run { self.core.stop() }
    }

    func unload() throws {
        try MainActorSync.run { self.core.unload() }
    }

    func getHistory() throws -> [LLMMessage] {
        MainActorSync.read { self.core.getHistory() }
    }

    func clearHistory() throws {
        try MainActorSync.run { self.core.clearHistory() }
    }
}

@MainActor
private final class HybridLLMCore {
    private final class TokenBatcher {
        private let batchSize: Int
        private let emit: (String) -> Void
        private var pending: [String] = []

        init(batchSize: Int, emit: @escaping (String) -> Void) {
            self.batchSize = max(1, batchSize)
            self.emit = emit
        }

        func append(_ chunk: String) {
            guard !chunk.isEmpty else { return }

            pending.append(chunk)
            if pending.count >= batchSize {
                flush()
            }
        }

        func flush() {
            guard !pending.isEmpty else { return }
            emit(pending.joined())
            pending.removeAll(keepingCapacity: true)
        }
    }

    private protocol GenerationSink: AnyObject {
        var firstTokenTime: Date? { get }
        var thinkingContent: String { get }
        func ingest(chunk: String) -> String
        func flush()
        func finalizeStream() -> String
        func registerToolCall(name: String, arguments: String, modelID: String?) -> String
        func willExecuteTool(id: String)
        func didCompleteTool(id: String, result: String)
        func didFailTool(id: String, error: String)
        func willContinueAfterTools()
    }

    private final class StringGenerationSink: GenerationSink {
        private let batcher: TokenBatcher
        private let onToolCall: (String, String) -> Void
        private var thinkingMachine = ThinkingStateMachine()
        private(set) var firstTokenTime: Date?
        private(set) var thinkingContent = ""

        init(batcher: TokenBatcher, onToolCall: @escaping (String, String) -> Void) {
            self.batcher = batcher
            self.onToolCall = onToolCall
        }

        func ingest(chunk: String) -> String {
            var result = ""
            for output in thinkingMachine.process(token: chunk) {
                switch output {
                case .token(let token):
                    if !token.isEmpty && firstTokenTime == nil {
                        firstTokenTime = Date()
                    }
                    batcher.append(token)
                    result += token
                case .thinkingStart:
                    batcher.flush()
                case .thinkingChunk(let thinking):
                    thinkingContent += thinking
                case .thinkingEnd:
                    break
                }
            }
            return result
        }

        func flush() {
            batcher.flush()
        }

        func finalizeStream() -> String {
            var result = ""
            for output in thinkingMachine.flush() {
                switch output {
                case .token(let token):
                    if !token.isEmpty && firstTokenTime == nil {
                        firstTokenTime = Date()
                    }
                    batcher.append(token)
                    result += token
                case .thinkingStart:
                    batcher.flush()
                case .thinkingChunk(let thinking):
                    thinkingContent += thinking
                case .thinkingEnd:
                    break
                }
            }
            batcher.flush()
            return result
        }

        func registerToolCall(name: String, arguments: String, modelID: String?) -> String {
            onToolCall(name, arguments)
            return modelID ?? UUID().uuidString
        }

        func willExecuteTool(id: String) {}
        func didCompleteTool(id: String, result: String) {}
        func didFailTool(id: String, error: String) {}

        func willContinueAfterTools() {
            batcher.flush()
        }
    }

    private final class EventGenerationSink: GenerationSink {
        private let emitter: StreamEventEmitter
        private let batcher: TokenBatcher
        private var thinkingMachine = ThinkingStateMachine()
        private(set) var firstTokenTime: Date?
        private(set) var thinkingContent = ""

        init(emitter: StreamEventEmitter, batcher: TokenBatcher) {
            self.emitter = emitter
            self.batcher = batcher
        }

        func ingest(chunk: String) -> String {
            var result = ""
            for out in thinkingMachine.process(token: chunk) {
                result += emit(out)
            }
            return result
        }

        func flush() {
            batcher.flush()
        }

        func finalizeStream() -> String {
            var result = ""
            for out in thinkingMachine.flush() {
                result += emit(out)
            }
            batcher.flush()
            return result
        }

        func registerToolCall(name: String, arguments: String, modelID: String?) -> String {
            let id = modelID ?? UUID().uuidString
            emitter.emitToolCallStart(id: id, name: name, arguments: arguments)
            return id
        }

        func willExecuteTool(id: String) {
            emitter.emitToolCallExecuting(id: id)
        }

        func didCompleteTool(id: String, result: String) {
            emitter.emitToolCallCompleted(id: id, result: result)
        }

        func didFailTool(id: String, error: String) {
            emitter.emitToolCallFailed(id: id, error: error)
        }

        func willContinueAfterTools() {}

        private func emit(_ output: ThinkingStateMachine.Output) -> String {
            switch output {
            case .token(let token):
                if !token.isEmpty && firstTokenTime == nil {
                    firstTokenTime = Date()
                }
                batcher.append(token)
                return token
            case .thinkingStart:
                batcher.flush()
                emitter.emitThinkingStart()
            case .thinkingChunk(let chunk):
                batcher.flush()
                thinkingContent += chunk
                emitter.emitThinkingChunk(chunk)
            case .thinkingEnd(let content):
                batcher.flush()
                emitter.emitThinkingEnd(content)
            }
            return ""
        }
    }

    private final class GenerationProgress {
        var content = ""
        var generationTokenCount = 0
        var generationTimeMs: Double = 0
        var toolExecutionTimeMs: Double = 0
        var firstTokenTime: Date?

        func recordContent(_ value: String, firstTokenTime: Date?) {
            guard !value.isEmpty else { return }
            content += value
            if self.firstTokenTime == nil {
                self.firstTokenTime = firstTokenTime ?? Date()
            }
        }

        func recordGenerationInfo(tokens: Int, timeMs: Double) {
            generationTokenCount += tokens
            generationTimeMs += timeMs
        }
    }

    /// One Turn Context: a retained ChatSession plus the bookkeeping the
    /// session cannot expose. `transcript` mirrors the session's messages so
    /// the context can be rebuilt after a cancelled turn and counted by
    /// countTokens; `pendingToolCallIds` enforces the tool-result contract.
    struct TurnContextEntry {
        var session: ChatSession
        let instructions: String?
        let toolSpecs: [ToolSpec]
        let toolNames: Set<String>
        let parameters: GenerateParameters
        var transcript: [Chat.Message]
        var pendingToolCallIds: [String]
        var needsRebuild: Bool
        /// Running total of tokens already encoded in the warm KV cache: the
        /// sum of prompt and completion tokens of every committed turn. An
        /// estimate — exact only while no trimming has occurred (spec: open
        /// question 4). Reset to 0 on a rebuild, whose fresh session re-encodes
        /// the whole transcript.
        var promptTokensSeen: Int
    }

    private var turnContexts = TurnContextRegistry<TurnContextEntry>()

    private var session: ChatSession?
    private let generationTasks = GenerationTaskController()
    private var container: ModelContainer?
    private var modelFactory: any ModelFactory = LLMModelFactory.shared
    private let tokenizerLoader: any TokenizerLoader = LocalTokenizerLoader()
    private var manageHistory: Bool = false
    private var seedMessages: [LLMMessage] = []
    private var messageHistory: [LLMMessage] = []
    private var structuredHistory: [Chat.Message] = []
    private var loadTask: Task<Void, Error>?

    private var tools: [ToolDefinition] = []
    private var toolSchemas: [ToolSpec] = []
    private var generationParameters: GenerateParameters = GenerateParameters()
    private var tokenBatchSize: Int = 4
    private var toolExecution: LLMToolExecution = .parallel
    private var contextConfig: LLMContextConfig?
    private var acceptsGeneration = false
    private var pendingUnload = false

    var isLoaded: Bool { acceptsGeneration && container != nil }
    var isGenerating: Bool { generationTasks.isActive }
    var modelId: String = ""
    var debug: Bool = false
    var systemPrompt: String = "You are a helpful assistant."

    private let maxToolCallDepth = 10
    private let defaultKeepLastMessages = 4

    private var canUseManagedSession: Bool {
        manageHistory && container != nil
    }

    private func log(_ message: String) {
        if debug {
            print("[MLXReactNative.HybridLLM] \(message)")
        }
    }

    private func ensureNotGenerating() throws {
        try generationTasks.ensureIdle()
    }

    /// Wraps a generation phase so foreign errors surface as
    /// `LLMError.generationFailed` with the phase's stage. `LLMError` and
    /// cancellation pass through untouched.
    private func withStage<T>(
        _ stage: GenerationStage,
        _ body: () async throws -> T
    ) async throws -> T {
        do {
            return try await body()
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as LLMError {
            throw error
        } catch {
            throw LLMError.generationFailed(
                stage: stage.rawValue,
                message: error.localizedDescription
            )
        }
    }

    private func getMemoryUsage() -> String {
        var taskInfo = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result: kern_return_t = withUnsafeMutablePointer(to: &taskInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    $0,
                    &count
                )
            }
        }

        if result == KERN_SUCCESS {
            let usedMB = Float(taskInfo.resident_size) / 1024.0 / 1024.0
            return String(format: "%.1f MB", usedMB)
        } else {
            return "unknown"
        }
    }

    private func getGPUMemoryUsage() -> String {
        let snapshot = Memory.snapshot()
        let allocatedMB = Float(snapshot.activeMemory) / 1024.0 / 1024.0
        let cacheMB = Float(snapshot.cacheMemory) / 1024.0 / 1024.0
        let peakMB = Float(snapshot.peakMemory) / 1024.0 / 1024.0
        return String(
            format: "Allocated: %.1f MB, Cache: %.1f MB, Peak: %.1f MB",
            allocatedMB,
            cacheMB,
            peakMB
        )
    }

    private func buildToolSchema(from tool: ToolDefinition) -> ToolSpec {
        var properties: [String: [String: Any]] = [:]
        var required: [String] = []

        for param in tool.parameters {
            properties[param.name] = [
                "type": param.type,
                "description": param.description,
            ]
            if param.required {
                required.append(param.name)
            }
        }

        return [
            "type": "function",
            "function": [
                "name": tool.name,
                "description": tool.description,
                "parameters": [
                    "type": "object",
                    "properties": properties,
                    "required": required,
                ],
            ],
        ] as ToolSpec
    }

    private func normalizedInt(_ value: Double?, minimum: Int = 0) -> Int? {
        guard let value else { return nil }
        return max(minimum, Int(value))
    }

    /// `UInt64(Double)` traps on values that are non-finite, negative, or too
    /// large to represent exactly — guard against all three before converting.
    private func normalizedSeed(_ value: Double?) -> UInt64? {
        guard let value,
            value.isFinite,
            value >= 0,
            value.rounded() == value,
            value <= 9_007_199_254_740_991 // 2^53, Double's exact-integer ceiling
        else {
            return nil
        }
        return UInt64(value)
    }

    private func buildGenerateParameters(from config: LLMGenerationConfig?) -> GenerateParameters {
        GenerateParameters(
            maxTokens: normalizedInt(config?.maxTokens, minimum: 1),
            maxKVSize: normalizedInt(config?.maxKVSize, minimum: 1),
            kvBits: normalizedInt(config?.kvBits, minimum: 1),
            kvGroupSize: normalizedInt(config?.kvGroupSize, minimum: 1) ?? 64,
            quantizedKVStart: normalizedInt(config?.quantizedKVStart, minimum: 0) ?? 0,
            temperature: Float(config?.temperature ?? 0.6),
            topP: Float(config?.topP ?? 1.0),
            topK: normalizedInt(config?.topK, minimum: 0) ?? 0,
            minP: Float(config?.minP ?? 0.0),
            repetitionPenalty: config?.repetitionPenalty.map(Float.init),
            repetitionContextSize: normalizedInt(config?.repetitionContextSize, minimum: 0) ?? 20,
            prefillStepSize: normalizedInt(config?.prefillStepSize, minimum: 1) ?? 512,
            seed: normalizedSeed(config?.seed)
        )
    }

    private func configuredToolSchemas() -> [ToolSpec]? {
        toolSchemas.isEmpty ? nil : toolSchemas
    }

    private func combinedHistory(with history: [LLMMessage]) -> [LLMMessage] {
        seedMessages + history
    }

    private func chatMessages(from history: [LLMMessage]) -> [Chat.Message] {
        history.compactMap { message in
            switch message.role {
            case "user":
                return .user(message.content)
            case "assistant":
                return .assistant(message.content)
            case "system":
                return .system(message.content)
            case "tool":
                return .tool(message.content)
            default:
                return nil
            }
        }
    }

    private func makeUserInput(history: [LLMMessage], prompt: String?) -> UserInput {
        var chat: [Chat.Message] = []

        if !systemPrompt.isEmpty {
            chat.append(.system(systemPrompt))
        }

        chat.append(contentsOf: chatMessages(from: combinedHistory(with: history)))

        if let prompt {
            chat.append(.user(prompt))
        }

        return UserInput(chat: chat, tools: configuredToolSchemas())
    }

    private func rebuildManagedSession() {
        guard canUseManagedSession, let container else {
            session = nil
            return
        }

        let history = chatMessages(from: seedMessages) + structuredHistory

        if history.isEmpty {
            session = ChatSession(
                container,
                instructions: systemPrompt,
                generateParameters: generationParameters,
                tools: configuredToolSchemas()
            )
        } else {
            session = ChatSession(
                container,
                instructions: systemPrompt,
                history: history,
                generateParameters: generationParameters,
                tools: configuredToolSchemas()
            )
        }
    }

    private func ensureManagedSession() throws -> ChatSession {
        guard canUseManagedSession else {
            throw LLMError.notLoaded
        }

        if session == nil {
            rebuildManagedSession()
        }

        guard let session else {
            throw LLMError.notLoaded
        }

        return session
    }

    private func trimManagedHistoryIfNeeded(upcomingPrompt: String? = nil) async throws {
        guard manageHistory, let container else { return }

        let maxContextTokens = normalizedInt(contextConfig?.maxContextTokens, minimum: 1)
        guard let maxContextTokens else { return }

        let keepLastMessages = normalizedInt(
            contextConfig?.keepLastMessages,
            minimum: 0
        ) ?? defaultKeepLastMessages

        var tokenizationPasses = 0

        func tokenCount(for history: [LLMMessage]) async throws -> Int {
            tokenizationPasses += 1
            let input = try await container.prepare(
                input: makeUserInput(history: history, prompt: upcomingPrompt)
            )
            return input.text.tokens.size
        }

        let originalHistory = messageHistory
        let initialTokenCount = try await tokenCount(for: originalHistory)

        guard initialTokenCount > maxContextTokens else { return }

        let maxRemovableMessages = max(0, originalHistory.count - keepLastMessages)
        guard maxRemovableMessages > 0 else {
            log(
                "Context remains above the configured limit (\(maxContextTokens) tokens); pinned and recent messages were preserved"
            )
            return
        }

        guard let trimPlan = try await ManagedHistoryTrimPlanner.plan(
            initialTokenCount: initialTokenCount,
            maxContextTokens: maxContextTokens,
            maxRemovableMessages: maxRemovableMessages,
            tokenCountAfterRemoving: { removalCount in
                try await tokenCount(
                    for: Array(originalHistory.dropFirst(removalCount))
                )
            }
        ) else {
            return
        }

        let removedCount = trimPlan.removalCount
        let trimmedHistory = Array(originalHistory.dropFirst(removedCount))

        messageHistory = trimmedHistory
        structuredHistory = Array(structuredHistory.dropFirst(removedCount))
        log(
            "Trimmed \(removedCount) message(s) from managed history to stay within \(maxContextTokens) prompt tokens after \(tokenizationPasses) tokenization pass(es)"
        )
        rebuildManagedSession()

        if !trimPlan.fitsBudget {
            log(
                "Context still exceeds \(maxContextTokens) tokens after trimming because preserved messages alone are larger than the budget"
            )
        }
    }

    private func finalizeManagedHistory(
        _ history: [LLMMessage],
        structured: [Chat.Message]
    ) async throws {
        guard manageHistory else { return }
        let previouslyCommittedHistory = messageHistory
        let previouslyCommittedStructuredHistory = structuredHistory
        messageHistory = history
        structuredHistory = structured
        do {
            try await trimManagedHistoryIfNeeded()
        } catch {
            messageHistory = previouslyCommittedHistory
            structuredHistory = previouslyCommittedStructuredHistory
            rebuildManagedSession()
            throw error
        }
    }

    private func makeStats(
        startTime: Date,
        firstTokenTime: Date?,
        generationTokenCount: Int,
        generationTimeMs: Double,
        toolExecutionTimeMs: Double
    ) -> GenerationStats {
        let endTime = Date()
        let totalTime = endTime.timeIntervalSince(startTime) * 1000
        let timeToFirstToken = (firstTokenTime ?? endTime).timeIntervalSince(startTime) * 1000
        let tokensPerSecond = generationTimeMs > 0
            ? Double(generationTokenCount) / (generationTimeMs / 1000)
            : 0

        return GenerationStats(
            tokenCount: Double(generationTokenCount),
            tokensPerSecond: tokensPerSecond,
            timeToFirstToken: timeToFirstToken,
            totalTime: totalTime,
            toolExecutionTime: toolExecutionTimeMs
        )
    }

    private func makeStats(startTime: Date, progress: GenerationProgress) -> GenerationStats {
        makeStats(
            startTime: startTime,
            firstTokenTime: progress.firstTokenTime,
            generationTokenCount: progress.generationTokenCount,
            generationTimeMs: progress.generationTimeMs,
            toolExecutionTimeMs: progress.toolExecutionTimeMs
        )
    }

    private func executeToolCall(
        tool: ToolDefinition,
        argsDict: [String: Any]
    ) async throws -> String {
        let argsAnyMap = dictionaryToAnyMap(argsDict)
        let outerPromise = tool.handler(argsAnyMap)
        let innerPromise = try await outerPromise.await()
        let resultAnyMap = try await innerPromise.await()
        let resultDict = anyMapToDictionary(resultAnyMap)
        return dictionaryToJson(resultDict)
    }

    /// Resets everything `load()` reconfigures, but not model residency.
    /// Shared by the full reset and the same-model reuse path.
    private func resetTurnConfiguration() {
        session = nil
        tools = []
        toolSchemas = []
        seedMessages = []
        messageHistory = []
        structuredHistory = []
        manageHistory = false
        generationParameters = GenerateParameters()
        tokenBatchSize = 4
        toolExecution = .parallel
        contextConfig = nil
    }

    private func resetModelState() {
        releaseAllTurnContexts()
        resetTurnConfiguration()
        container = nil
        modelId = ""
        Memory.clearCache()
    }

    private func finishGeneration(id: UInt64) {
        generationTasks.finish(id: id)
        if pendingUnload && !generationTasks.isActive {
            completeUnload()
        }
    }

    private func completeUnload() {
        let memoryBefore = getMemoryUsage()
        let gpuBefore = getGPUMemoryUsage()
        log("Before unload - Host: \(memoryBefore), GPU: \(gpuBefore)")

        resetModelState()
        pendingUnload = false

        let memoryAfter = getMemoryUsage()
        let gpuAfter = getGPUMemoryUsage()
        log("After unload - Host: \(memoryAfter), GPU: \(gpuAfter)")
    }

    func load(modelId: String, options: LLMLoadOptions?) async throws {
        loadTask?.cancel()
        acceptsGeneration = false
        pendingUnload = false

        let task = Task { @MainActor in
            await generationTasks.cancelAndWait(reason: .superseded)
            try Task.checkCancellation()

            let action = ModelLoadPlan.action(
                requestedModelId: modelId,
                loadedModelId: self.modelId,
                hasContainer: container != nil
            )

            let loadedContainer: ModelContainer
            switch action {
            case .reuseContainer:
                log("Reusing resident container for \(modelId)")
                releaseAllTurnContexts()
                resetTurnConfiguration()
                options?.onProgress?(1.0)
                loadedContainer = container!
            case .loadContainer:
                resetModelState()

                let memoryAfterCleanup = getMemoryUsage()
                let gpuAfterCleanup = getGPUMemoryUsage()
                log("After cleanup - Host: \(memoryAfterCleanup), GPU: \(gpuAfterCleanup)")

                if !(await ModelDownloader.shared.isDownloaded(modelId: modelId)) {
                    log("Model not cached, downloading before load: \(modelId)")
                    _ = try await ModelDownloader.shared.download(
                        modelId: modelId,
                        progressCallback: { fraction in
                            options?.onProgress?(fraction)
                        }
                    )
                }

                let modelDir = await ModelDownloader.shared.getModelDirectory(modelId: modelId)
                log("Loading from directory: \(modelDir.path)")

                loadedContainer = try await modelFactory.loadContainer(
                    from: modelDir,
                    using: tokenizerLoader
                )

                try Task.checkCancellation()

                let memoryAfterContainer = getMemoryUsage()
                let gpuAfterContainer = getGPUMemoryUsage()
                log("Model loaded - Host: \(memoryAfterContainer), GPU: \(gpuAfterContainer)")
            }

            if let jsTools = options?.tools {
                tools = jsTools
                toolSchemas = jsTools.map { buildToolSchema(from: $0) }
                log("Loaded \(tools.count) tools: \(tools.map(\.name))")
            }

            generationParameters = buildGenerateParameters(from: options?.generationConfig)
            tokenBatchSize = normalizedInt(options?.tokenBatchSize, minimum: 1) ?? 4
            toolExecution = options?.toolExecution ?? .parallel
            contextConfig = options?.contextConfig

            self.container = loadedContainer
            self.modelId = modelId
            manageHistory = options?.manageHistory ?? false
            seedMessages = options?.additionalContext ?? []
            messageHistory = []
            structuredHistory = []

            if manageHistory {
                log("History management enabled with \(seedMessages.count) seed messages")
            }

            rebuildManagedSession()
            acceptsGeneration = true
        }

        loadTask = task
        try await task.value
    }

    func generate(prompt: String) async throws -> LLMGenerationOutcome {
        let batcher = TokenBatcher(batchSize: tokenBatchSize, emit: { _ in })
        return try await beginTurn(
            prompt: prompt,
            sink: StringGenerationSink(batcher: batcher, onToolCall: { _, _ in })
        )
    }

    func stream(
        prompt: String,
        onToken: @escaping (String) -> Void,
        onToolCall: ((String, String) -> Void)?
    ) async throws -> LLMGenerationOutcome {
        let batcher = TokenBatcher(batchSize: tokenBatchSize, emit: onToken)
        return try await beginTurn(
            prompt: prompt,
            sink: StringGenerationSink(
                batcher: batcher,
                onToolCall: onToolCall ?? { _, _ in }
            )
        )
    }

    func streamWithEvents(
        prompt: String,
        onEvent: @escaping (StreamEventEnvelope) -> Void
    ) async throws -> LLMGenerationOutcome {
        let emitter = StreamEventEmitter(callback: onEvent)
        let batcher = TokenBatcher(batchSize: tokenBatchSize) { token in
            emitter.emitToken(token)
        }
        let outcome = try await beginTurn(
            prompt: prompt,
            sink: EventGenerationSink(emitter: emitter, batcher: batcher),
            onStarted: emitter.emitGenerationStart,
            onFinished: emitter.emitGenerationOutcome
        )
        return outcome
    }

    private struct PendingToolCall {
        let id: String
        let modelID: String?
        let modelCall: ToolCall
        let tool: ToolDefinition?
        let name: String
        let args: [String: Any]
    }

    private struct ToolExecutionBatch {
        let results: [String?]
        let wasCancelled: Bool
    }

    private func beginTurn(
        prompt: String,
        sink: GenerationSink,
        onStarted: (() -> Void)? = nil,
        onFinished: ((LLMGenerationOutcome) -> Void)? = nil
    ) async throws -> LLMGenerationOutcome {
        guard acceptsGeneration, container != nil else {
            throw LLMError.notLoaded
        }
        try ensureNotGenerating()

        let task = Task<LLMGenerationOutcome, Never> { @MainActor in
            let outcome = await self.performTurn(prompt: prompt, sink: sink)
            onFinished?(outcome)
            return outcome
        }
        let generationID = try generationTasks.begin(task)
        onStarted?()
        defer { finishGeneration(id: generationID) }
        return await task.value
    }

    private func performTurn(
        prompt: String,
        sink: GenerationSink
    ) async -> LLMGenerationOutcome {
        let startTime = Date()
        let progress = GenerationProgress()
        var turnHistory = messageHistory
        var turnStructuredHistory = structuredHistory
        var currentPassContent = ""
        var didAppendPrompt = false

        func outcome(
            _ finishReason: LLMGenerationFinishReason,
            error: String? = nil,
            stage: String? = nil
        ) -> LLMGenerationOutcome {
            LLMGenerationOutcome(
                content: progress.content,
                thinking: sink.thinkingContent.isEmpty ? nil : sink.thinkingContent,
                stats: makeStats(startTime: startTime, progress: progress),
                finishReason: finishReason,
                error: error,
                stage: stage
            )
        }

        do {
            if manageHistory {
                try await withStage(.prepare) {
                    try await self.trimManagedHistoryIfNeeded(upcomingPrompt: prompt)
                }
                turnHistory = messageHistory
                turnStructuredHistory = structuredHistory
            }

            guard let turnSession = try makeTurnSession() else {
                throw LLMError.notLoaded
            }

            var inputMessages: [Chat.Message] = [.user(prompt)]
            turnHistory.append(LLMMessage(role: "user", content: prompt))
            turnStructuredHistory.append(.user(prompt))
            didAppendPrompt = true
            var depth = 0

            while true {
                try Task.checkCancellation()
                currentPassContent = ""
                let pendingCalls = try await withStage(.generate) {
                    try await self.runSessionPass(
                        session: turnSession,
                        inputMessages: inputMessages,
                        sink: sink,
                        progress: progress,
                        passContent: &currentPassContent
                    )
                }

                if pendingCalls.isEmpty {
                    if !currentPassContent.isEmpty {
                        turnHistory.append(
                            LLMMessage(role: "assistant", content: currentPassContent)
                        )
                        turnStructuredHistory.append(.assistant(currentPassContent))
                    }
                    currentPassContent = ""
                    break
                }

                guard depth < maxToolCallDepth else {
                    throw LLMError.generationFailed(
                        stage: GenerationStage.tool.rawValue,
                        message: "Maximum tool continuation depth reached"
                    )
                }

                let batch = await executeToolCalls(
                    pendingCalls,
                    sink: sink,
                    progress: progress
                )
                let completedCalls = zip(pendingCalls, batch.results).compactMap { call, result in
                    result == nil ? nil : call
                }
                if !currentPassContent.isEmpty || !completedCalls.isEmpty {
                    turnHistory.append(
                        LLMMessage(role: "assistant", content: currentPassContent)
                    )
                    turnStructuredHistory.append(
                        .assistant(
                            currentPassContent,
                            toolCalls: completedCalls.isEmpty
                                ? nil
                                : completedCalls.map(\.modelCall)
                        )
                    )
                    currentPassContent = ""
                }
                for (call, result) in zip(pendingCalls, batch.results) {
                    guard let result else { continue }
                    turnHistory.append(LLMMessage(role: "tool", content: result))
                    turnStructuredHistory.append(.tool(result, id: call.modelID))
                }
                if batch.wasCancelled {
                    throw CancellationError()
                }

                inputMessages = zip(pendingCalls, batch.results).compactMap { call, result in
                    result.map { .tool($0, id: call.modelID) }
                }
                sink.willContinueAfterTools()
                depth += 1
            }

            if manageHistory {
                try await withStage(.history) {
                    try await self.finalizeManagedHistory(
                        turnHistory,
                        structured: turnStructuredHistory
                    )
                }
            }
            return outcome(.completed)
        } catch is CancellationError {
            let reason = generationTasks.cancellationReason ?? .stopped
            if !currentPassContent.isEmpty {
                turnHistory.append(LLMMessage(role: "assistant", content: currentPassContent))
                turnStructuredHistory.append(.assistant(currentPassContent))
            }

            switch reason {
            case .stopped:
                if manageHistory {
                    if !didAppendPrompt {
                        turnHistory.append(LLMMessage(role: "user", content: prompt))
                        turnStructuredHistory.append(.user(prompt))
                    }
                    messageHistory = turnHistory
                    structuredHistory = turnStructuredHistory
                    rebuildManagedSession()
                }
                return outcome(.stopped)
            case .superseded:
                if manageHistory { rebuildManagedSession() }
                return outcome(.superseded)
            case .unloaded:
                session = nil
                return outcome(.unloaded)
            }
        } catch {
            if manageHistory { rebuildManagedSession() }
            return outcome(
                .failed,
                error: error.localizedDescription,
                stage: (error as? LLMError)?.failureStage
                    ?? GenerationStage.generate.rawValue
            )
        }
    }

    private func makeTurnSession() throws -> ChatSession? {
        guard let container else { return nil }
        if manageHistory {
            return try ensureManagedSession()
        }
        return ChatSession(
            container,
            instructions: systemPrompt,
            generateParameters: generationParameters,
            tools: configuredToolSchemas()
        )
    }

    private func runSessionPass(
        session: ChatSession,
        inputMessages: [Chat.Message],
        sink: GenerationSink,
        progress: GenerationProgress,
        passContent: inout String
    ) async throws -> [PendingToolCall] {
        var pendingCalls: [PendingToolCall] = []
        var didFinalize = false
        defer {
            if !didFinalize {
                let suffix = sink.finalizeStream()
                passContent += suffix
                progress.recordContent(suffix, firstTokenTime: sink.firstTokenTime)
            }
        }

        for try await generation in session.streamDetails(to: inputMessages) {
            try Task.checkCancellation()
            switch generation {
            case .chunk(let text):
                let content = sink.ingest(chunk: text)
                passContent += content
                progress.recordContent(content, firstTokenTime: sink.firstTokenTime)
            case .toolCall(let toolCall):
                sink.flush()
                let name = toolCall.function.name
                let args = convertToolCallArguments(toolCall.function.arguments)
                let id = sink.registerToolCall(
                    name: name,
                    arguments: dictionaryToJson(args),
                    modelID: toolCall.id
                )
                pendingCalls.append(
                    PendingToolCall(
                        id: id,
                        modelID: toolCall.id,
                        modelCall: toolCall,
                        tool: tools.first(where: { $0.name == name }),
                        name: name,
                        args: args
                    )
                )
            case .info(let info):
                sink.flush()
                progress.recordGenerationInfo(
                    tokens: info.generationTokenCount,
                    timeMs: info.generateTime * 1000
                )
            }
        }

        let suffix = sink.finalizeStream()
        didFinalize = true
        passContent += suffix
        progress.recordContent(suffix, firstTokenTime: sink.firstTokenTime)
        try Task.checkCancellation()
        return pendingCalls
    }

    private func executeToolCalls(
        _ calls: [PendingToolCall],
        sink: GenerationSink,
        progress: GenerationProgress
    ) async -> ToolExecutionBatch {
        let startedAt = Date()
        var results = Array<String?>(repeating: nil, count: calls.count)

        if toolExecution == .sequential {
            for (index, call) in calls.enumerated() {
                if Task.isCancelled { break }
                sink.willExecuteTool(id: call.id)
                guard let result = await executeToolUntilCancelled(call) else { break }
                results[index] = serializedToolResult(result)
                emitToolResult(result, call: call, sink: sink)
            }
        } else {
            for call in calls {
                sink.willExecuteTool(id: call.id)
            }
            let (completions, continuation) = AsyncStream.makeStream(
                of: (Int, Result<String, Error>).self
            )
            let tasks = calls.enumerated().map { index, call in
                Task { @MainActor [self] in
                    continuation.yield((index, await executeTool(call)))
                }
            }

            var completed = 0
            await withTaskCancellationHandler {
                for await (index, result) in completions {
                    if Task.isCancelled { break }
                    completed += 1
                    results[index] = serializedToolResult(result)
                    emitToolResult(result, call: calls[index], sink: sink)
                    if completed == calls.count { break }
                }
            } onCancel: {
                continuation.finish()
            }
            continuation.finish()
            tasks.forEach { $0.cancel() }
        }

        progress.toolExecutionTimeMs += Date().timeIntervalSince(startedAt) * 1000
        return ToolExecutionBatch(results: results, wasCancelled: Task.isCancelled)
    }

    private func executeToolUntilCancelled(
        _ call: PendingToolCall
    ) async -> Result<String, Error>? {
        let (completions, continuation) = AsyncStream.makeStream(
            of: Result<String, Error>.self
        )
        let task = Task { @MainActor [self] in
            continuation.yield(await executeTool(call))
            continuation.finish()
        }
        let result: Result<String, Error>? = await withTaskCancellationHandler {
            var iterator = completions.makeAsyncIterator()
            return await iterator.next()
        } onCancel: {
            continuation.finish()
        }
        task.cancel()
        return result
    }

    private func executeTool(_ call: PendingToolCall) async -> Result<String, Error> {
        guard let tool = call.tool else {
            return .failure(
                LLMError.generationFailed(
                    stage: GenerationStage.tool.rawValue,
                    message: "Unknown tool: \(call.name)"
                )
            )
        }
        do {
            return .success(try await executeToolCall(tool: tool, argsDict: call.args))
        } catch {
            return .failure(error)
        }
    }

    private func emitToolResult(
        _ result: Result<String, Error>,
        call: PendingToolCall,
        sink: GenerationSink
    ) {
        switch result {
        case .success(let value):
            sink.didCompleteTool(id: call.id, result: value)
        case .failure(let error):
            sink.didFailTool(id: call.id, error: error.localizedDescription)
        }
    }

    private func serializedToolResult(_ result: Result<String, Error>) -> String {
        switch result {
        case .success(let value):
            return value
        case .failure(let error):
            return dictionaryToJson(["error": error.localizedDescription])
        }
    }

    private func convertToolCallArguments(_ arguments: [String: JSONValue]) -> [String: Any] {
        var result: [String: Any] = [:]
        for (key, value) in arguments {
            result[key] = value.anyValue
        }
        return result
    }

    private func dictionaryToAnyMap(_ dict: [String: Any]) -> AnyMap {
        let anyMap = AnyMap()
        for (key, value) in dict {
            switch value {
            case let stringValue as String:
                anyMap.setString(key: key, value: stringValue)
            case let doubleValue as Double:
                anyMap.setDouble(key: key, value: doubleValue)
            case let intValue as Int:
                anyMap.setDouble(key: key, value: Double(intValue))
            case let boolValue as Bool:
                anyMap.setBoolean(key: key, value: boolValue)
            default:
                anyMap.setString(key: key, value: String(describing: value))
            }
        }
        return anyMap
    }

    private func anyMapToDictionary(_ anyMap: AnyMap) -> [String: Any] {
        var dict: [String: Any] = [:]
        for key in anyMap.getAllKeys() {
            if anyMap.isString(key: key) {
                dict[key] = anyMap.getString(key: key)
            } else if anyMap.isDouble(key: key) {
                dict[key] = anyMap.getDouble(key: key)
            } else if anyMap.isBool(key: key) {
                dict[key] = anyMap.getBoolean(key: key)
            }
        }
        return dict
    }

    var turnContextIds: [String] { turnContexts.ids }

    func createTurnContext(options: LLMTurnContextOptions?) throws -> String {
        guard let container else { throw LLMError.notLoaded }

        var toolSpecs: [ToolSpec] = []
        var toolNames: Set<String> = []
        for tool in options?.tools ?? [] {
            toolSpecs.append(try toolSpec(from: tool))
            toolNames.insert(tool.name)
        }

        let history = try chatMessagesFromTurnMessages(options?.history ?? [])
        let parameters = buildGenerateParameters(from: options?.generationConfig)
        let session = ChatSession(
            container,
            instructions: options?.instructions,
            history: history,
            generateParameters: parameters,
            tools: toolSpecs.isEmpty ? nil : toolSpecs
        )

        return turnContexts.insert(
            TurnContextEntry(
                session: session,
                instructions: options?.instructions,
                toolSpecs: toolSpecs,
                toolNames: toolNames,
                parameters: parameters,
                transcript: history,
                pendingToolCallIds: [],
                needsRebuild: false,
                promptTokensSeen: 0
            )
        )
    }

    func releaseTurnContext(id: String) {
        guard let entry = turnContexts.release(id) else { return }
        synchronizeDroppedSession(entry.session)
    }

    func releaseAllTurnContexts() {
        for entry in turnContexts.releaseAll() {
            synchronizeDroppedSession(entry.session)
        }
    }

    /// The session must settle its KV cache before it is dropped, but the paths
    /// that drop one are synchronous. The task holds the last reference, so the
    /// wait happens off the call and no async cache work outlives the reference.
    private func synchronizeDroppedSession(_ session: ChatSession) {
        Task { @MainActor in
            await session.synchronize()
        }
    }

    /// Everything a turn accumulates from one model pass. A reference type so
    /// the cancellation path can still report the counts folded before the
    /// stream was torn down, like `GenerationProgress`.
    private final class TurnAccumulation {
        /// The upstream calls, kept alongside their wire form because a warm
        /// context mirrors them into its transcript as `.assistant(_:toolCalls:)`.
        var toolCalls: [ToolCall] = []
        /// Built as tool calls arrive, so these ids are the ids already emitted
        /// on `tool_call_start`. Tool results are correlated against them, so
        /// they must be computed exactly once.
        var wireToolCalls: [LLMToolCallWire] = []
        var promptTokens = 0
        var completionTokens = 0
        var stopReason: GenerateStopReason?

        /// The calls carrying the ids the caller was given: the model may omit
        /// an id, in which case the wire form holds a minted one. A transcript
        /// mirroring the raw calls would render assistant tool calls with no id
        /// beside tool results carrying one, which pairing templates cannot
        /// resolve. `toolCalls` and `wireToolCalls` grow in lockstep, one
        /// append each per `.toolCall` element.
        var identifiedToolCalls: [ToolCall] {
            zip(toolCalls, wireToolCalls).map { call, wire in
                ToolCall(function: call.function, id: wire.id)
            }
        }

        /// Upstream sets `.length` exactly when the iterator hit its token
        /// limit (Evaluate.swift:1891-1897) and always reports a stop reason on
        /// a non-cancelled pass, so no token-count heuristic is needed — one
        /// would mislabel a model that emits EOS on the last allowed token.
        var stoppedAtLength: Bool {
            if case .some(.length) = stopReason { return true }
            return false
        }
    }

    /// What a `responseSchema` turn resolved to, read off the accumulation
    /// the same pass already folded — never a distinct model round-trip.
    private enum StructuredOutputResult {
        case success(content: String)
        case failure(message: String)
    }

    /// Inspects the one tool call a schema turn's stream produced, if any.
    /// A model that answers in prose instead never reaches `.toolCall` in
    /// `collectTurn`, so `accumulation.toolCalls` is empty — the same
    /// failure as a call to any other name.
    private func structuredOutputResult(from accumulation: TurnAccumulation) -> StructuredOutputResult {
        guard let call = accumulation.toolCalls.first,
            call.function.name == ToolSchemaPlanner.structuredOutputToolName
        else {
            return .failure(message: "Model did not call the structured output tool")
        }
        let arguments = convertToolCallArguments(call.function.arguments)
        guard let argumentsData = try? JSONSerialization.data(withJSONObject: arguments) else {
            return .failure(message: "Structured output arguments did not serialize")
        }
        return .success(content: String(decoding: argumentsData, as: UTF8.self))
    }

    /// Runs one Generation Turn. Tool Call Requests come back to the caller
    /// instead of being executed here.
    ///
    /// Preflight rejects; a started turn always resolves with an outcome
    /// (docs/adr/0001-generation-turn-outcomes.md), so every check that can
    /// fail runs before `generationTasks.begin`.
    func runTurn(
        request: LLMTurnRequest,
        onEvent: @escaping (StreamEventEnvelope) -> Void
    ) async throws -> LLMTurnOutcome {
        guard acceptsGeneration, let container else {
            throw LLMError.notLoaded
        }
        try ensureNotGenerating()

        let plan = try planTurn(for: request)

        // Mapped before the switch, because the warm branch mutates the
        // registry and every rejection belongs in preflight.
        let inputMessages = try chatMessagesFromTurnMessages(request.messages)

        // Built once so both branches below offer the identical spec; the
        // planner already guarantees no other tools are in play whenever
        // this is non-nil (schemaExclusiveWithTools).
        let schemaToolSpec: ToolSpec? = try request.responseSchema.map {
            try syntheticToolSpec(responseSchema: $0)
        }

        let session: ChatSession
        let warm: WarmTurnCommit?
        // Restores a warm session's tools after the pass, win or lose: the
        // session is retained in the registry, so a schema turn's borrowed
        // tool must not leak into the next turn on the same context.
        var restoreWarmTools: (() -> Void)?
        switch plan.mode {
        case .cold:
            session = ChatSession(
                container,
                instructions: request.instructions,
                history: try chatMessagesFromTurnMessages(request.history ?? []),
                generateParameters: buildGenerateParameters(from: request.generationConfig),
                tools: try schemaToolSpec.map { [$0] } ?? turnToolSpecs(from: request.tools ?? [])
            )
            warm = nil
        case .warm:
            // The planner proved both of these; failing here would mean the
            // context was released between planning and now.
            guard let contextId = request.contextId,
                let entry = turnContexts.entry(for: contextId)
            else {
                throw LLMError.generationFailed(
                    stage: GenerationStage.prepare.rawValue,
                    message: "unknown Turn Context \(request.contextId ?? "")"
                )
            }
            let prepared = rebuiltIfNeeded(entry, id: contextId, container: container)
            session = prepared.session
            warm = WarmTurnCommit(
                contextId: contextId,
                cachedPromptTokens: prepared.promptTokensSeen
            )
            // `ChatSession.tools` is a plain `var`, so a warm context can
            // take a per-turn tool despite being fixed at creation for the
            // ordinary tool-calling path. The planner only reaches here with
            // a schema when the context itself declared no tools
            // (schemaExclusiveWithTools), so `previousTools` is always nil —
            // captured anyway so a future coupling change can't leak a
            // stale list into the next turn.
            if let schemaToolSpec {
                let previousTools = session.tools
                session.tools = [schemaToolSpec]
                restoreWarmTools = { session.tools = previousTools }
            }
        }

        let emitter = StreamEventEmitter(callback: onEvent)
        let batcher = TokenBatcher(
            batchSize: normalizedInt(request.tokenBatchSize, minimum: 1) ?? tokenBatchSize
        ) { token in
            emitter.emitToken(token)
        }
        let sink = EventGenerationSink(emitter: emitter, batcher: batcher)

        let task = Task<LLMTurnOutcome, Never> { @MainActor in
            await self.performTurn(
                session: session,
                inputMessages: inputMessages,
                sink: sink,
                warm: warm,
                expectsStructuredOutput: schemaToolSpec != nil
            )
        }
        let generationID = try generationTasks.begin(task)
        emitter.emitGenerationStart()
        defer {
            finishGeneration(id: generationID)
            restoreWarmTools?()
        }
        return await task.value
    }

    /// Validates the request and decides cold versus warm. Planner rejections
    /// become readable `prepare`-stage failures.
    private func planTurn(for request: LLMTurnRequest) throws -> TurnPlan {
        let entry = request.contextId.flatMap { turnContexts.entry(for: $0) }
        let requestHasTools = !(request.tools ?? []).isEmpty
        do {
            return try TurnRequestPlanner.plan(
                messages: request.messages.map(turnMessagePlan),
                contextId: request.contextId,
                contextKnown: entry != nil,
                contextHasTools: !(entry?.toolNames.isEmpty ?? true),
                pendingToolCallIds: entry?.pendingToolCallIds ?? [],
                hasColdFields: request.instructions != nil
                    || request.history != nil
                    || request.generationConfig != nil
                    || requestHasTools,
                requestHasTools: requestHasTools,
                hasResponseSchema: request.responseSchema != nil
            )
        } catch let error as TurnPlanError {
            throw LLMError.generationFailed(
                stage: GenerationStage.prepare.rawValue,
                message: planFailureMessage(error)
            )
        }
    }

    private func turnMessagePlan(_ message: LLMTurnMessage) -> TurnMessagePlan {
        TurnMessagePlan(
            role: message.role,
            content: message.content,
            toolCallId: message.toolCallId,
            name: message.name,
            isError: message.isError,
            toolCallsJson: message.toolCallsJson
        )
    }

    private func planFailureMessage(_ error: TurnPlanError) -> String {
        switch error {
        case .emptyMessages:
            return "messages must not be empty"
        case .unknownRole(let role):
            return "unknown message role \(role)"
        case .missingToolCallId:
            return "tool messages need a toolCallId"
        case .toolCallsOnNonAssistant:
            return "only assistant messages may carry tool calls"
        case .unknownContext(let id):
            return "unknown Turn Context \(id)"
        case .coldFieldsOnWarmTurn:
            return
                "instructions, history, tools, and generationConfig are cold-turn fields; "
                + "remove them or remove contextId"
        case .schemaExclusiveWithTools:
            return "responseSchema is exclusive with tools"
        case .incompleteToolResults(let missing):
            return "missing tool results for \(missing.joined(separator: ", "))"
        case .unknownToolCallId(let id):
            return "tool result \(id) does not match a pending Tool Call Request"
        case .duplicateToolCallId(let id):
            return "duplicate tool result for \(id)"
        }
    }

    /// nil rather than an empty array, because upstream treats an empty tool
    /// list as "tools are in play" when it renders the chat template.
    private func turnToolSpecs(from tools: [LLMToolSchema]) throws -> [ToolSpec]? {
        guard !tools.isEmpty else { return nil }
        return try tools.map { tool in
            do {
                return try toolSpec(from: tool)
            } catch let error as ToolSchemaError {
                throw LLMError.generationFailed(
                    stage: GenerationStage.prepare.rawValue,
                    message: "tool \(tool.name): \(toolSchemaFailureMessage(error))"
                )
            }
        }
    }

    /// Projects the single synthetic tool (Task 7) onto the same
    /// `Sendable`-typed `ToolSpec` shape `toolSpec(from:)` builds for a
    /// declared tool — `ToolSchemaPlanner.syntheticTool` only knows
    /// `[String: Any]`, which upstream's `tools` array cannot hold directly.
    private func syntheticToolSpec(responseSchema: String) throws -> ToolSpec {
        do {
            let synthetic = try ToolSchemaPlanner.syntheticTool(responseSchema: responseSchema)
            let function: [String: any Sendable] = [
                "name": synthetic.name,
                "description": synthetic.description,
                "parameters": sendableJSON(jsonValue(from: synthetic.parameters)),
            ]
            return ["type": "function", "function": function]
        } catch let error as ToolSchemaError {
            throw LLMError.generationFailed(
                stage: GenerationStage.prepare.rawValue,
                message: "responseSchema: \(toolSchemaFailureMessage(error))"
            )
        }
    }

    private func toolSchemaFailureMessage(_ error: ToolSchemaError) -> String {
        switch error {
        case .invalidJSON:
            return "parameters is not valid JSON"
        case .rootNotObjectSchema:
            return "parameters must be a JSON Schema whose root type is object"
        }
    }

    /// What a warm turn commits against its Turn Context when the pass ends.
    /// Captured before the pass, so releasing the context mid-turn changes the
    /// bookkeeping but never what the outcome reports.
    private struct WarmTurnCommit {
        let contextId: String
        /// `promptTokensSeen` as of the turn's start — what the KV cache
        /// already held, reported as the outcome's `cachedPromptTokens`.
        let cachedPromptTokens: Int
    }

    /// A turn that did not commit left the session's KV cache holding a prompt
    /// the transcript never recorded, so the next turn generates from a fresh
    /// session built out of the mirror. The rebuilt session has encoded
    /// nothing yet, hence `promptTokensSeen` returns to zero.
    private func rebuiltIfNeeded(
        _ entry: TurnContextEntry,
        id: String,
        container: ModelContainer
    ) -> TurnContextEntry {
        guard entry.needsRebuild else { return entry }

        // The likeliest drop point for a session with in-flight cache work:
        // `needsRebuild` is set by a cancelled or failed turn.
        synchronizeDroppedSession(entry.session)

        var rebuilt = entry
        rebuilt.session = ChatSession(
            container,
            instructions: entry.instructions,
            history: entry.transcript,
            generateParameters: entry.parameters,
            tools: entry.toolSpecs.isEmpty ? nil : entry.toolSpecs
        )
        rebuilt.needsRebuild = false
        rebuilt.promptTokensSeen = 0
        turnContexts.update(id, with: rebuilt)
        return rebuilt
    }

    /// Mirrors what the session appended, so a rebuild and countTokens see the
    /// transcript the KV cache encodes. A context released mid-turn is not
    /// resurrected: `update` ignores unknown ids.
    private func commitWarmTurn(
        _ warm: WarmTurnCommit,
        inputMessages: [Chat.Message],
        content: String,
        accumulation: TurnAccumulation
    ) {
        guard var entry = turnContexts.entry(for: warm.contextId) else { return }

        entry.transcript.append(contentsOf: inputMessages)
        let toolCalls = accumulation.identifiedToolCalls
        if !content.isEmpty || !toolCalls.isEmpty {
            entry.transcript.append(
                .assistant(content, toolCalls: toolCalls.isEmpty ? nil : toolCalls)
            )
        }
        // The same array the outcome carries, so a caller can only answer ids
        // it was actually given (Task 9: ids are minted once, at registration).
        entry.pendingToolCallIds = accumulation.wireToolCalls.map(\.id)
        entry.promptTokensSeen += accumulation.promptTokens + accumulation.completionTokens
        turnContexts.update(warm.contextId, with: entry)
    }

    /// Leaves `transcript` and `pendingToolCallIds` as they were before the
    /// turn: nothing was committed, so the caller owes no tool results and the
    /// partial content is not part of the conversation.
    private func markWarmTurnForRebuild(_ warm: WarmTurnCommit) {
        guard var entry = turnContexts.entry(for: warm.contextId) else { return }
        entry.needsRebuild = true
        turnContexts.update(warm.contextId, with: entry)
    }

    /// One model pass, over a throwaway session (cold) or a Turn Context's
    /// retained session (warm). Never throws: a started turn reports
    /// cancellation and failure as an outcome.
    private func performTurn(
        session: ChatSession,
        inputMessages: [Chat.Message],
        sink: GenerationSink,
        warm: WarmTurnCommit?,
        expectsStructuredOutput: Bool = false
    ) async -> LLMTurnOutcome {
        let startTime = Date()
        let progress = GenerationProgress()
        let accumulation = TurnAccumulation()

        func outcome(
            _ finishReason: LLMTurnFinishReason,
            rawFinishReason: String? = nil,
            content: String? = nil,
            toolCalls: [LLMToolCallWire] = [],
            error: String? = nil,
            stage: String? = nil
        ) -> LLMTurnOutcome {
            LLMTurnOutcome(
                finishReason: finishReason,
                rawFinishReason: rawFinishReason,
                content: content ?? progress.content,
                thinking: sink.thinkingContent.isEmpty ? nil : sink.thinkingContent,
                toolCalls: toolCalls,
                usage: LLMTurnUsage(
                    promptTokens: Double(accumulation.promptTokens),
                    completionTokens: Double(accumulation.completionTokens),
                    // A warm turn's promptTokens counts only the messages this
                    // pass appended; this is what the cache already held.
                    // Nothing is cached on a cold turn.
                    cachedPromptTokens: warm.map { Double($0.cachedPromptTokens) }
                ),
                stats: makeStats(startTime: startTime, progress: progress),
                error: error,
                stage: stage
            )
        }

        do {
            try await collectTurn(
                session: session,
                inputMessages: inputMessages,
                sink: sink,
                progress: progress,
                into: accumulation,
                suppressToolCallEvents: expectsStructuredOutput
            )

            if expectsStructuredOutput {
                switch structuredOutputResult(from: accumulation) {
                case .success(let content):
                    // The synthetic tool call is a request/response
                    // implementation detail, never a Tool Call Request the
                    // caller learns about (toolCalls is always [] on a
                    // schema outcome) — clearing it before the warm mirror
                    // sees it keeps `pendingToolCallIds` in sync with what
                    // the caller actually received. Otherwise a committed
                    // id nobody was told about would deadlock every later
                    // turn on this Turn Context (`incompleteToolResults`).
                    // The mirrored content is the JSON itself, so the
                    // transcript reads like the model answered directly.
                    accumulation.toolCalls = []
                    accumulation.wireToolCalls = []
                    if let warm {
                        commitWarmTurn(
                            warm,
                            inputMessages: inputMessages,
                            content: content,
                            accumulation: accumulation
                        )
                    }
                    return outcome(.completed, rawFinishReason: "structured_output", content: content)
                case .failure(let message):
                    // Nothing commits on a schema failure, matching the
                    // cancellation/error paths below: the live session's KV
                    // cache still advanced (the model's wrong-tool-call or
                    // prose reply actually happened), but the mirror must
                    // not silently drop that exchange (a bare user turn with
                    // no assistant reply) nor invent one the caller was
                    // never shown. Marking the context for rebuild means the
                    // next warm turn reconstructs cleanly from the
                    // last-committed mirror instead of drifting from the
                    // live cache.
                    if let warm { markWarmTurnForRebuild(warm) }
                    return outcome(.failed, error: message, stage: "schema")
                }
            }

            if let warm {
                commitWarmTurn(
                    warm,
                    inputMessages: inputMessages,
                    content: progress.content,
                    accumulation: accumulation
                )
            }

            if !accumulation.wireToolCalls.isEmpty {
                return outcome(.toolCalls, toolCalls: accumulation.wireToolCalls)
            }
            if accumulation.stoppedAtLength {
                return outcome(.length, rawFinishReason: "maxTokens")
            }
            return outcome(.completed)
        } catch is CancellationError {
            if let warm { markWarmTurnForRebuild(warm) }
            // Tool calls parsed before the cancellation are dropped: the caller
            // must not answer requests no context is waiting for.
            switch generationTasks.cancellationReason ?? .stopped {
            case .stopped:
                return outcome(.stopped, rawFinishReason: "stopped")
            case .superseded:
                return outcome(.superseded, rawFinishReason: "superseded")
            case .unloaded:
                return outcome(.unloaded, rawFinishReason: "unloaded")
            }
        } catch {
            // Same reasoning as cancellation: nothing was committed, so the
            // session's cache no longer matches the mirror.
            if let warm { markWarmTurnForRebuild(warm) }
            return outcome(
                .failed,
                error: error.localizedDescription,
                stage: (error as? LLMError)?.failureStage
                    ?? GenerationStage.generate.rawValue
            )
        }
    }

    /// Folds one `streamDetails` pass into `accumulation`, routing tokens
    /// through the same sink legacy turns use so thinking tags, token batching,
    /// and the `tool_call_start` envelope behave identically.
    ///
    /// `suppressToolCallEvents` keeps a schema turn's synthetic tool call (and
    /// anything else the model calls while only that tool was offered) off the
    /// wire entirely: the id is still minted and folded into `accumulation`
    /// exactly as usual, so `structuredOutputResult` and the warm mirror keep
    /// working — only `sink.registerToolCall`'s `tool_call_start` emission
    /// (and everything downstream of it, e.g. a caller auto-executing it) is
    /// skipped. The outcome always reports `toolCalls: []` for a schema turn,
    /// so the caller must never see this call as a Tool Call Request.
    private func collectTurn(
        session: ChatSession,
        inputMessages: [Chat.Message],
        sink: GenerationSink,
        progress: GenerationProgress,
        into accumulation: TurnAccumulation,
        suppressToolCallEvents: Bool = false
    ) async throws {
        var didFinalize = false
        defer {
            if !didFinalize {
                progress.recordContent(
                    sink.finalizeStream(),
                    firstTokenTime: sink.firstTokenTime
                )
            }
        }

        for try await generation in session.streamDetails(to: inputMessages) {
            switch generation {
            case .chunk(let text):
                progress.recordContent(
                    sink.ingest(chunk: text),
                    firstTokenTime: sink.firstTokenTime
                )
            case .toolCall(let toolCall):
                sink.flush()
                let name = toolCall.function.name
                let argumentsJson = dictionaryToJson(
                    convertToolCallArguments(toolCall.function.arguments)
                )
                // A schema turn mints the same id `registerToolCall` would
                // have (model id if present, else a fresh UUID) without
                // calling it, so nothing reaches the bridge.
                let id = suppressToolCallEvents
                    ? (toolCall.id ?? UUID().uuidString)
                    : sink.registerToolCall(
                        name: name,
                        arguments: argumentsJson,
                        modelID: toolCall.id
                    )
                accumulation.toolCalls.append(toolCall)
                accumulation.wireToolCalls.append(
                    LLMToolCallWire(id: id, name: name, argumentsJson: argumentsJson)
                )
            case .info(let info):
                sink.flush()
                accumulation.promptTokens += info.promptTokenCount
                accumulation.completionTokens += info.generationTokenCount
                accumulation.stopReason = info.stopReason
                progress.recordGenerationInfo(
                    tokens: info.generationTokenCount,
                    timeMs: info.generateTime * 1000
                )
            }
            // Checked after the element is folded, never before: upstream emits
            // `.info` as the last element of a pass even when that pass was
            // cancelled (Evaluate.swift:1905-1917), and a cancelled turn must
            // still report the tokens it spent.
            try Task.checkCancellation()
        }

        progress.recordContent(sink.finalizeStream(), firstTokenTime: sink.firstTokenTime)
        didFinalize = true
        try Task.checkCancellation()
    }

    /// Renders the same chat template `runTurn` would send and counts its
    /// tokens. Uses `container.prepare(input:)` rather than a
    /// `container.perform` closure: it only takes the container's mutex
    /// long enough to read out the `UserInputProcessor` (see
    /// `ModelContainer.prepare` upstream), so a `countTokens` call does not
    /// serialize behind an in-flight `runTurn`'s prefill/decode — both may
    /// run concurrently. Mirrors the pattern already used by
    /// `trimManagedHistoryIfNeeded` above.
    func countTokens(request: LLMTokenCountRequest) async throws -> Double {
        guard let container else { throw LLMError.notLoaded }

        var chat: [Chat.Message] = []
        var toolSpecs: [ToolSpec]?

        if let contextId = request.contextId {
            guard let entry = turnContexts.entry(for: contextId) else {
                throw LLMError.generationFailed(
                    stage: GenerationStage.prepare.rawValue,
                    message: "Unknown context \(contextId)"
                )
            }
            if let instructions = entry.instructions { chat.append(.system(instructions)) }
            chat.append(contentsOf: entry.transcript)
            toolSpecs = entry.toolSpecs.isEmpty ? nil : entry.toolSpecs
        } else {
            if let instructions = request.instructions { chat.append(.system(instructions)) }
            chat.append(contentsOf: try chatMessagesFromTurnMessages(request.history ?? []))
            toolSpecs = try turnToolSpecs(from: request.tools ?? [])
        }
        chat.append(contentsOf: try chatMessagesFromTurnMessages(request.messages ?? []))

        let input = UserInput(chat: chat, tools: toolSpecs)
        let prepared = try await container.prepare(input: input)
        return Double(prepared.text.tokens.size)
    }

    private func toolSpec(from tool: LLMToolSchema) throws -> ToolSpec {
        let parameters = try ToolSchemaPlanner.parseParameters(tool.parameters)
        let function: [String: any Sendable] = [
            "name": tool.name,
            "description": tool.description,
            "parameters": sendableJSON(jsonValue(from: parameters)),
        ]
        return ["type": "function", "function": function]
    }

    /// Maps wire turn messages to upstream chat messages. Assistant tool calls
    /// ride in toolCallsJson as [{id, name, arguments}]; upstream renders them
    /// model-agnostically via addToolMetadata (Chat.swift:137-158).
    private func chatMessagesFromTurnMessages(
        _ messages: [LLMTurnMessage]
    ) throws -> [Chat.Message] {
        try messages.map { message in
            switch message.role {
            case "system":
                return .system(message.content)
            case "user":
                return .user(message.content)
            case "assistant":
                guard let json = message.toolCallsJson, !json.isEmpty else {
                    return .assistant(message.content)
                }
                return .assistant(message.content, toolCalls: try parseWireToolCalls(json))
            case "tool":
                // The model has no other signal that a tool result is a
                // failure (`isError` never crosses the chat template on its
                // own) — this is the one place every route (cold turn,
                // warm turn, Turn Context creation, `countTokens`) renders a
                // tool message, so prefixing here is what actually makes
                // the wire contract's "the model must see failures" true.
                let content =
                    message.isError == true ? "Error: \(message.content)" : message.content
                return .tool(content, id: message.toolCallId)
            default:
                throw LLMError.generationFailed(
                    stage: GenerationStage.prepare.rawValue,
                    message: "Unknown role \(message.role)"
                )
            }
        }
    }

    private func parseWireToolCalls(_ json: String) throws -> [ToolCall] {
        guard let data = json.data(using: .utf8),
            let array = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        else {
            throw LLMError.generationFailed(
                stage: GenerationStage.prepare.rawValue,
                message: "toolCallsJson is not valid JSON"
            )
        }
        return array.map { entry in
            ToolCall(
                function: .init(
                    name: entry["name"] as? String ?? "",
                    arguments: (entry["arguments"] as? [String: Any] ?? [:])
                        .mapValues { jsonValue(from: $0) }
                ),
                id: entry["id"] as? String
            )
        }
    }

    /// `JSONSerialization` erases Bool and Int to the same `NSNumber` class, so
    /// `value as? Bool` also matches 0 and 1. The CoreFoundation type id is the
    /// only reliable discriminator, which is why this does not reuse
    /// `JSONValue.from(_:)`.
    private func jsonValue(from value: Any) -> JSONValue {
        switch value {
        case let number as NSNumber:
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                return .bool(number.boolValue)
            }
            return CFNumberIsFloatType(number as CFNumber)
                ? .double(number.doubleValue)
                : .int(number.intValue)
        case let string as String:
            return .string(string)
        case let array as [Any]:
            return .array(array.map { jsonValue(from: $0) })
        case let dictionary as [String: Any]:
            return .object(dictionary.mapValues { jsonValue(from: $0) })
        default:
            return .null
        }
    }

    /// Projects a parsed JSON value onto the `Sendable`-typed values `ToolSpec`
    /// requires. Mirrors the upstream `JSONValue.sendableValue`, which is
    /// module-internal.
    private func sendableJSON(_ value: JSONValue) -> any Sendable {
        switch value {
        case .null:
            return NSNull()
        case .bool(let value):
            return value
        case .int(let value):
            return value
        case .double(let value):
            return value
        case .string(let value):
            return value
        case .array(let values):
            return values.map { sendableJSON($0) }
        case .object(let values):
            return values.mapValues { sendableJSON($0) }
        }
    }

    func stop() {
        generationTasks.cancel(reason: .stopped)
    }

    func unload() {
        loadTask?.cancel()
        loadTask = nil
        acceptsGeneration = false
        pendingUnload = true
        generationTasks.cancel(reason: .unloaded)

        if !generationTasks.isActive {
            completeUnload()
        }
    }

    func getHistory() -> [LLMMessage] {
        combinedHistory(with: messageHistory)
    }

    func clearHistory() {
        messageHistory = []
        structuredHistory = []
        rebuildManagedSession()
        log("Message history cleared")
    }
}

/// Loads a Hugging Face tokenizer from a local directory and bridges it to
/// `MLXLMCommon.Tokenizer`. The mlx-swift-lm 3.x API requires an explicit
/// `TokenizerLoader`; this mirrors the expansion of `#huggingFaceTokenizerLoader()`.
private struct LocalTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let upstream = try await Tokenizers.AutoTokenizer.from(modelFolder: directory)
        return TokenizerBridge(upstream)
    }
}

private struct TokenizerBridge: MLXLMCommon.Tokenizer {
    private let upstream: any Tokenizers.Tokenizer

    init(_ upstream: any Tokenizers.Tokenizer) {
        self.upstream = upstream
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    // swift-transformers uses `decode(tokens:)` instead of `decode(tokenIds:)`.
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        upstream.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        upstream.convertIdToToken(id)
    }

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(
                messages: messages, tools: tools, additionalContext: additionalContext)
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}
