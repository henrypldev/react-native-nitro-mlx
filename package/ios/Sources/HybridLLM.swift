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

    func generate(prompt: String) throws -> Promise<String> {
        Promise.async { [core] in
            try await core.generate(prompt: prompt)
        }
    }

    func stream(
        prompt: String,
        onToken: @escaping (String) -> Void,
        onToolCall: ((String, String) -> Void)?
    ) throws -> Promise<String> {
        Promise.async { [core] in
            try await core.stream(prompt: prompt, onToken: onToken, onToolCall: onToolCall)
        }
    }

    func streamWithEvents(
        prompt: String,
        onEvent: @escaping (StreamEventEnvelope) -> Void
    ) throws -> Promise<String> {
        Promise.async { [core] in
            try await core.streamWithEvents(prompt: prompt, onEvent: onEvent)
        }
    }

    func stop() throws {
        try MainActorSync.run { self.core.stop() }
    }

    func unload() throws {
        try MainActorSync.run { self.core.unload() }
    }

    func getLastGenerationStats() throws -> GenerationStats {
        MainActorSync.read { self.core.getLastGenerationStats() }
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
        func ingest(chunk: String) -> String
        func flush()
        func finalizeStream() -> String
        func registerToolCall(name: String, arguments: String) -> String
        func willExecuteTool(id: String)
        func didCompleteTool(id: String, result: String)
        func didFailTool(id: String, error: String)
        func willContinueAfterTools()
    }

    private final class StringGenerationSink: GenerationSink {
        private let batcher: TokenBatcher
        private let onToolCall: (String, String) -> Void
        private(set) var firstTokenTime: Date?

        init(batcher: TokenBatcher, onToolCall: @escaping (String, String) -> Void) {
            self.batcher = batcher
            self.onToolCall = onToolCall
        }

        func ingest(chunk: String) -> String {
            if !chunk.isEmpty && firstTokenTime == nil {
                firstTokenTime = Date()
            }
            batcher.append(chunk)
            return chunk
        }

        func flush() {
            batcher.flush()
        }

        func finalizeStream() -> String { "" }

        func registerToolCall(name: String, arguments: String) -> String {
            onToolCall(name, arguments)
            return UUID().uuidString
        }

        func willExecuteTool(id: String) {}
        func didCompleteTool(id: String, result: String) {}
        func didFailTool(id: String, error: String) {}

        func willContinueAfterTools() {
            batcher.flush()
            if firstTokenTime == nil {
                firstTokenTime = Date()
            }
            batcher.append("\u{200B}")
        }
    }

    private final class EventGenerationSink: GenerationSink {
        private let emitter: StreamEventEmitter
        private let batcher: TokenBatcher
        private var thinkingMachine = ThinkingStateMachine()
        private(set) var firstTokenTime: Date?

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

        func registerToolCall(name: String, arguments: String) -> String {
            let id = UUID().uuidString
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
                emitter.emitThinkingChunk(chunk)
            case .thinkingEnd(let content):
                batcher.flush()
                emitter.emitThinkingEnd(content)
            }
            return ""
        }
    }

    private struct ManagedSessionResult {
        let output: String
        let generationTokenCount: Int
        let generationTimeMs: Double
        let firstTokenTime: Date?
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

    private var session: ChatSession?
    private let generationTasks = GenerationTaskController()
    private var container: ModelContainer?
    private var lastStats: GenerationStats = GenerationStats(
        tokenCount: 0,
        tokensPerSecond: 0,
        timeToFirstToken: 0,
        totalTime: 0,
        toolExecutionTime: 0
    )
    private var modelFactory: any ModelFactory = LLMModelFactory.shared
    private let tokenizerLoader: any TokenizerLoader = LocalTokenizerLoader()
    private var manageHistory: Bool = false
    private var seedMessages: [LLMMessage] = []
    private var messageHistory: [LLMMessage] = []
    private var loadTask: Task<Void, Error>?

    private var tools: [ToolDefinition] = []
    private var toolSchemas: [ToolSpec] = []
    private var generationParameters: GenerateParameters = GenerateParameters()
    private var tokenBatchSize: Int = 4
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
        manageHistory && toolSchemas.isEmpty && container != nil
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

    private func buildGenerateParameters(from config: LLMGenerationConfig?) -> GenerateParameters {
        GenerateParameters(
            maxTokens: normalizedInt(config?.maxTokens, minimum: 1),
            maxKVSize: normalizedInt(config?.maxKVSize, minimum: 1),
            kvBits: normalizedInt(config?.kvBits, minimum: 1),
            kvGroupSize: normalizedInt(config?.kvGroupSize, minimum: 1) ?? 64,
            quantizedKVStart: normalizedInt(config?.quantizedKVStart, minimum: 0) ?? 0,
            temperature: Float(config?.temperature ?? 0.6),
            topP: Float(config?.topP ?? 1.0),
            repetitionPenalty: config?.repetitionPenalty.map(Float.init),
            repetitionContextSize: normalizedInt(config?.repetitionContextSize, minimum: 0) ?? 20,
            prefillStepSize: normalizedInt(config?.prefillStepSize, minimum: 1) ?? 512
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

        let history = chatMessages(from: combinedHistory(with: messageHistory))

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

    private func finalizeManagedHistory(_ history: [LLMMessage]) async throws {
        guard manageHistory else { return }
        messageHistory = history
        try await trimManagedHistoryIfNeeded()
        if canUseManagedSession {
            rebuildManagedSession()
        }
    }

    private func buildChatMessages(
        history: [LLMMessage],
        prompt: String,
        toolResults: [String]?,
        depth: Int
    ) -> [Chat.Message] {
        var chat: [Chat.Message] = []

        if !systemPrompt.isEmpty {
            chat.append(.system(systemPrompt))
        }

        chat.append(contentsOf: chatMessages(from: combinedHistory(with: history)))

        if depth == 0 {
            chat.append(.user(prompt))
        }

        if let toolResults {
            for result in toolResults {
                chat.append(.tool(result))
            }
        }

        return chat
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

    private func runManagedSession(
        prompt: String,
        batcher: TokenBatcher?,
        progress: GenerationProgress? = nil
    ) async throws -> ManagedSessionResult {
        let session = try ensureManagedSession()

        var output = ""
        var firstTokenTime: Date?
        var generationTokenCount = 0
        var generationTimeMs: Double = 0

        defer {
            batcher?.flush()
        }

        for try await generation in session.streamDetails(to: prompt, images: [], videos: []) {
            if Task.isCancelled { break }

            switch generation {
            case .chunk(let text):
                if firstTokenTime == nil {
                    firstTokenTime = Date()
                }

                output += text
                batcher?.append(text)
                progress?.recordContent(text, firstTokenTime: firstTokenTime)

            case .info(let info):
                generationTokenCount += info.generationTokenCount
                generationTimeMs += info.generateTime * 1000
                progress?.recordGenerationInfo(
                    tokens: info.generationTokenCount,
                    timeMs: info.generateTime * 1000
                )
                log(
                    "Generation info: \(info.generationTokenCount) tokens, \(String(format: "%.1f", info.tokensPerSecond)) tokens/s"
                )

            case .toolCall:
                break
            }
        }

        return ManagedSessionResult(
            output: output,
            generationTokenCount: generationTokenCount,
            generationTimeMs: generationTimeMs,
            firstTokenTime: firstTokenTime
        )
    }

    private func resetModelState() {
        session = nil
        container = nil
        tools = []
        toolSchemas = []
        seedMessages = []
        messageHistory = []
        manageHistory = false
        generationParameters = GenerateParameters()
        tokenBatchSize = 4
        contextConfig = nil
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
            await generationTasks.cancelAndWait()
            try Task.checkCancellation()
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

            let loadedContainer = try await modelFactory.loadContainer(
                from: modelDir,
                using: tokenizerLoader
            )

            try Task.checkCancellation()

            let memoryAfterContainer = getMemoryUsage()
            let gpuAfterContainer = getGPUMemoryUsage()
            log("Model loaded - Host: \(memoryAfterContainer), GPU: \(gpuAfterContainer)")

            if let jsTools = options?.tools {
                tools = jsTools
                toolSchemas = jsTools.map { buildToolSchema(from: $0) }
                log("Loaded \(tools.count) tools: \(tools.map(\.name))")
            }

            generationParameters = buildGenerateParameters(from: options?.generationConfig)
            tokenBatchSize = normalizedInt(options?.tokenBatchSize, minimum: 1) ?? 4
            contextConfig = options?.contextConfig

            self.container = loadedContainer
            self.modelId = modelId
            manageHistory = options?.manageHistory ?? false
            seedMessages = options?.additionalContext ?? []
            messageHistory = []

            if manageHistory {
                log("History management enabled with \(seedMessages.count) seed messages")
            }

            rebuildManagedSession()
            acceptsGeneration = true
        }

        loadTask = task
        try await task.value
    }

    func generate(prompt: String) async throws -> String {
        guard acceptsGeneration, let container else {
            throw LLMError.notLoaded
        }
        try ensureNotGenerating()

        let task = Task<String, Error> { @MainActor in
                let startTime = Date()

                if canUseManagedSession {
                    try await withStage(.prepare) {
                        try await self.trimManagedHistoryIfNeeded(upcomingPrompt: prompt)
                    }

                    let result = try await withStage(.generate) {
                        try await self.runManagedSession(prompt: prompt, batcher: nil)
                    }

                    var updatedHistory = messageHistory
                    updatedHistory.append(LLMMessage(role: "user", content: prompt))
                    updatedHistory.append(LLMMessage(role: "assistant", content: result.output))
                    try await withStage(.history) {
                        try await self.finalizeManagedHistory(updatedHistory)
                    }

                    let stats = makeStats(
                        startTime: startTime,
                        firstTokenTime: result.firstTokenTime,
                        generationTokenCount: result.generationTokenCount,
                        generationTimeMs: result.generationTimeMs,
                        toolExecutionTimeMs: 0
                    )
                    lastStats = stats
                    return result.output
                }

                if manageHistory {
                    try await withStage(.prepare) {
                        try await self.trimManagedHistoryIfNeeded(upcomingPrompt: prompt)
                    }
                }

                var history = messageHistory
                var generationTokenCount = 0
                var generationTimeMs: Double = 0
                var toolExecutionTime: Double = 0
                let batcher = TokenBatcher(batchSize: tokenBatchSize, emit: { _ in })
                let sink = StringGenerationSink(batcher: batcher, onToolCall: { _, _ in })

                let result: String
                do {
                    result = try await performGeneration(
                        container: container,
                        history: &history,
                        prompt: prompt,
                        toolResults: nil,
                        depth: 0,
                        sink: sink,
                        onGenerationInfo: { tokens, time in
                            generationTokenCount += tokens
                            generationTimeMs += time
                        },
                        toolExecutionTime: &toolExecutionTime
                    )
                } catch is CancellationError {
                    throw CancellationError()
                } catch let error as LLMError {
                    throw error
                } catch {
                    throw LLMError.generationFailed(
                        stage: GenerationStage.generate.rawValue,
                        message: error.localizedDescription
                    )
                }

                try await withStage(.history) {
                    try await self.finalizeManagedHistory(history)
                }

                lastStats = makeStats(
                    startTime: startTime,
                    firstTokenTime: sink.firstTokenTime,
                    generationTokenCount: generationTokenCount,
                    generationTimeMs: generationTimeMs,
                    toolExecutionTimeMs: toolExecutionTime
                )

                return result
            }

        let generationID = try generationTasks.begin(task)
        defer { finishGeneration(id: generationID) }
        return try await task.value
    }

    func stream(
        prompt: String,
        onToken: @escaping (String) -> Void,
        onToolCall: ((String, String) -> Void)?
    ) async throws -> String {
        guard acceptsGeneration, let container else {
            throw LLMError.notLoaded
        }
        try ensureNotGenerating()

        let task = Task<String, Error> { @MainActor in
                let startTime = Date()
                let batcher = TokenBatcher(batchSize: tokenBatchSize, emit: onToken)

                if canUseManagedSession {
                    try await withStage(.prepare) {
                        try await self.trimManagedHistoryIfNeeded(upcomingPrompt: prompt)
                    }

                    let result = try await withStage(.generate) {
                        try await self.runManagedSession(prompt: prompt, batcher: batcher)
                    }

                    var updatedHistory = messageHistory
                    updatedHistory.append(LLMMessage(role: "user", content: prompt))
                    updatedHistory.append(LLMMessage(role: "assistant", content: result.output))
                    try await withStage(.history) {
                        try await self.finalizeManagedHistory(updatedHistory)
                    }

                    let stats = makeStats(
                        startTime: startTime,
                        firstTokenTime: result.firstTokenTime,
                        generationTokenCount: result.generationTokenCount,
                        generationTimeMs: result.generationTimeMs,
                        toolExecutionTimeMs: 0
                    )
                    lastStats = stats
                    return result.output
                }

                if manageHistory {
                    try await withStage(.prepare) {
                        try await self.trimManagedHistoryIfNeeded(upcomingPrompt: prompt)
                    }
                }

                var history = messageHistory
                var generationTokenCount = 0
                var generationTimeMs: Double = 0
                var toolExecutionTime: Double = 0
                let sink = StringGenerationSink(
                    batcher: batcher,
                    onToolCall: onToolCall ?? { _, _ in }
                )

                let result: String
                do {
                    result = try await performGeneration(
                        container: container,
                        history: &history,
                        prompt: prompt,
                        toolResults: nil,
                        depth: 0,
                        sink: sink,
                        onGenerationInfo: { tokens, time in
                            generationTokenCount += tokens
                            generationTimeMs += time
                        },
                        toolExecutionTime: &toolExecutionTime
                    )
                } catch is CancellationError {
                    throw CancellationError()
                } catch let error as LLMError {
                    throw error
                } catch {
                    throw LLMError.generationFailed(
                        stage: GenerationStage.generate.rawValue,
                        message: error.localizedDescription
                    )
                }

                batcher.flush()
                try await withStage(.history) {
                    try await self.finalizeManagedHistory(history)
                }

                let stats = makeStats(
                    startTime: startTime,
                    firstTokenTime: sink.firstTokenTime,
                    generationTokenCount: generationTokenCount,
                    generationTimeMs: generationTimeMs,
                    toolExecutionTimeMs: toolExecutionTime
                )
                lastStats = stats

                log(
                    "Stream complete - \(generationTokenCount) tokens, \(String(format: "%.1f", stats.tokensPerSecond)) tokens/s"
                )
                return result
            }

        let generationID = try generationTasks.begin(task)
        defer { finishGeneration(id: generationID) }
        return try await task.value
    }

    func streamWithEvents(
        prompt: String,
        onEvent: @escaping (StreamEventEnvelope) -> Void
    ) async throws -> String {
        guard acceptsGeneration, let container else {
            throw LLMError.notLoaded
        }
        try ensureNotGenerating()

        let task = Task<String, Error> { @MainActor in
            let startTime = Date()
            let emitter = StreamEventEmitter(callback: onEvent)
            emitter.emitGenerationStart()

            let progress = GenerationProgress()
            var toolExecutionTime: Double = 0

            @MainActor func partialStats() -> GenerationStats {
                makeStats(startTime: startTime, progress: progress)
            }

            do {
                if canUseManagedSession {
                    try await withStage(.prepare) {
                        try await self.trimManagedHistoryIfNeeded(upcomingPrompt: prompt)
                    }

                    let batcher = TokenBatcher(batchSize: tokenBatchSize) { token in
                        emitter.emitToken(token)
                    }
                    let result = try await withStage(.generate) {
                        try await self.runManagedSession(
                            prompt: prompt,
                            batcher: batcher,
                            progress: progress
                        )
                    }

                    var updatedHistory = messageHistory
                    updatedHistory.append(LLMMessage(role: "user", content: prompt))
                    updatedHistory.append(LLMMessage(role: "assistant", content: result.output))
                    try await withStage(.history) {
                        try await self.finalizeManagedHistory(updatedHistory)
                    }

                    let stats = partialStats()
                    lastStats = stats
                    emitter.emitGenerationEnd(content: result.output, stats: stats)
                    return result.output
                }

                if manageHistory {
                    try await withStage(.prepare) {
                        try await self.trimManagedHistoryIfNeeded(upcomingPrompt: prompt)
                    }
                }

                var history = messageHistory
                let tokenBatcher = TokenBatcher(batchSize: tokenBatchSize) { token in
                    emitter.emitToken(token)
                }
                let sink = EventGenerationSink(emitter: emitter, batcher: tokenBatcher)

                let result: String
                do {
                    result = try await performGeneration(
                        container: container,
                        history: &history,
                        prompt: prompt,
                        toolResults: nil,
                        depth: 0,
                        sink: sink,
                        onGenerationInfo: { _, _ in },
                        toolExecutionTime: &toolExecutionTime,
                        progress: progress
                    )
                } catch is CancellationError {
                    throw CancellationError()
                } catch let error as LLMError {
                    throw error
                } catch {
                    throw LLMError.generationFailed(
                        stage: GenerationStage.generate.rawValue,
                        message: error.localizedDescription
                    )
                }
                tokenBatcher.flush()
                try await withStage(.history) {
                    try await self.finalizeManagedHistory(history)
                }

                let stats = partialStats()
                lastStats = stats
                emitter.emitGenerationEnd(content: result, stats: stats)

                log(
                    "StreamWithEvents complete - \(progress.generationTokenCount) tokens, \(String(format: "%.1f", stats.tokensPerSecond)) tokens/s (tool execution: \(String(format: "%.0f", progress.toolExecutionTimeMs))ms)"
                )
                return result
            } catch is CancellationError {
                // stop() or a superseding load(): resolve with partial state.
                let stats = partialStats()
                lastStats = stats
                emitter.emitGenerationEnd(content: progress.content, stats: stats)
                return progress.content
            } catch {
                let stats = partialStats()
                lastStats = stats
                emitter.emitGenerationError(
                    error: error.localizedDescription,
                    stage: (error as? LLMError)?.failureStage
                        ?? GenerationStage.generate.rawValue,
                    stats: stats
                )
                throw error
            }
        }

        let generationID = try generationTasks.begin(task)
        defer { finishGeneration(id: generationID) }
        return try await task.value
    }

    private func performGeneration(
        container: ModelContainer,
        history: inout [LLMMessage],
        prompt: String,
        toolResults: [String]?,
        depth: Int,
        sink: GenerationSink,
        onGenerationInfo: @escaping (Int, Double) -> Void,
        toolExecutionTime: inout Double,
        progress: GenerationProgress? = nil
    ) async throws -> String {
        if depth >= maxToolCallDepth {
            log("Max tool call depth reached (\(maxToolCallDepth))")
            return ""
        }

        var output = ""
        var pendingToolCalls: [(id: String, tool: ToolDefinition, args: [String: Any])] = []
        var didFinalizeStream = false

        defer {
            if !didFinalizeStream {
                let suffix = sink.finalizeStream()
                progress?.recordContent(suffix, firstTokenTime: sink.firstTokenTime)
            }
        }

        let chat = buildChatMessages(
            history: history,
            prompt: prompt,
            toolResults: toolResults,
            depth: depth
        )
        let userInput = UserInput(chat: chat, tools: configuredToolSchemas())
        let lmInput = try await withStage(.prepare) {
            try await container.prepare(input: userInput)
        }
        let parameters = generationParameters

        let stream = try await container.generate(
            input: lmInput,
            parameters: parameters
        )

        for await generation in stream {
            if Task.isCancelled { break }

            switch generation {
            case .chunk(let text):
                let content = sink.ingest(chunk: text)
                output += content
                progress?.recordContent(content, firstTokenTime: sink.firstTokenTime)

            case .toolCall(let toolCall):
                sink.flush()
                log("Tool call detected: \(toolCall.function.name)")

                guard let tool = tools.first(where: { $0.name == toolCall.function.name }) else {
                    log("Unknown tool: \(toolCall.function.name)")
                    continue
                }

                let argsDict = convertToolCallArguments(toolCall.function.arguments)
                let argsJson = dictionaryToJson(argsDict)
                let id = sink.registerToolCall(name: toolCall.function.name, arguments: argsJson)

                pendingToolCalls.append((id: id, tool: tool, args: argsDict))

            case .info(let info):
                sink.flush()
                log(
                    "Generation info: \(info.generationTokenCount) tokens, \(String(format: "%.1f", info.tokensPerSecond)) tokens/s"
                )
                let generationTime = info.tokensPerSecond > 0
                    ? Double(info.generationTokenCount) / info.tokensPerSecond * 1000
                    : 0
                onGenerationInfo(info.generationTokenCount, generationTime)
                progress?.recordGenerationInfo(
                    tokens: info.generationTokenCount,
                    timeMs: generationTime
                )
            }
        }

        let suffix = sink.finalizeStream()
        didFinalizeStream = true
        output += suffix
        progress?.recordContent(suffix, firstTokenTime: sink.firstTokenTime)

        if Task.isCancelled {
            return output
        }

        if !pendingToolCalls.isEmpty {
            log("Executing \(pendingToolCalls.count) tool call(s)")
            let toolStartTime = Date()

            for call in pendingToolCalls {
                sink.willExecuteTool(id: call.id)
            }

            let allToolResults: [String] = await withTaskGroup(of: (Int, String).self) { group in
                for (index, call) in pendingToolCalls.enumerated() {
                    group.addTask { [self] in
                        do {
                            let resultJson = try await executeToolCall(
                                tool: call.tool,
                                argsDict: call.args
                            )
                            await log("Tool result for \(call.tool.name): \(resultJson.prefix(100))...")
                            sink.didCompleteTool(id: call.id, result: resultJson)
                            return (index, resultJson)
                        } catch {
                            await log("Tool execution error for \(call.tool.name): \(error)")
                            sink.didFailTool(id: call.id, error: error.localizedDescription)
                            return (index, "{\"error\": \"Tool execution failed\"}")
                        }
                    }
                }

                var results = Array(repeating: "", count: pendingToolCalls.count)
                for await (index, result) in group {
                    results[index] = result
                }
                return results
            }

            let elapsedToolTime = Date().timeIntervalSince(toolStartTime) * 1000
            toolExecutionTime += elapsedToolTime
            progress?.toolExecutionTimeMs += elapsedToolTime

            if Task.isCancelled {
                return output
            }

            if depth == 0 {
                history.append(LLMMessage(role: "user", content: prompt))
            }
            if !output.isEmpty {
                history.append(LLMMessage(role: "assistant", content: output))
            }
            for result in allToolResults {
                history.append(LLMMessage(role: "tool", content: result))
            }

            sink.willContinueAfterTools()

            let continuation = try await performGeneration(
                container: container,
                history: &history,
                prompt: prompt,
                toolResults: allToolResults,
                depth: depth + 1,
                sink: sink,
                onGenerationInfo: onGenerationInfo,
                toolExecutionTime: &toolExecutionTime,
                progress: progress
            )

            return output + continuation
        }

        if manageHistory {
            if depth == 0 {
                history.append(LLMMessage(role: "user", content: prompt))
            }
            if !output.isEmpty {
                history.append(LLMMessage(role: "assistant", content: output))
            }
        }

        return output
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

    func stop() {
        generationTasks.cancel()
    }

    func unload() {
        loadTask?.cancel()
        loadTask = nil
        acceptsGeneration = false
        pendingUnload = true
        generationTasks.cancel()

        if !generationTasks.isActive {
            completeUnload()
        }
    }

    func getLastGenerationStats() -> GenerationStats {
        lastStats
    }

    func getHistory() -> [LLMMessage] {
        combinedHistory(with: messageHistory)
    }

    func clearHistory() {
        messageHistory = []
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
