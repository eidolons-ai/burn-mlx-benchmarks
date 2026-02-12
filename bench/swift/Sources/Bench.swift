/// MLX Swift benchmark for Qwen3-0.6B decode throughput.
///
/// Uses manual forward passes with eval() (synchronous) for accurate per-token
/// timing. Does NOT use TokenIterator's built-in async pipelining.
///
/// NOTE: This is the most uncertain component of the benchmark harness.
/// The mlx-swift-lm API may differ from what's assumed here. If it doesn't
/// compile, check the actual API at https://github.com/ml-explore/mlx-swift-lm
/// and adapt accordingly.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers

// MARK: - JSON Schema Types

struct PromptsFile: Codable {
    let config: BenchConfig
    let warmup: PromptEntry
    let prompts: [PromptEntry]
}

struct BenchConfig: Codable {
    let max_new_tokens: Int
    let timed_iterations: Int
    let warmup_iterations: Int
    let sleep_between_runs_secs: Int
    let temperature: Double
}

struct PromptEntry: Codable {
    let id: String
    let system_prompt: String
    let user_message: String
    let label: String?
}

struct RunResult: Codable {
    let prompt_id: String
    let iteration: Int
    let token_ids: [Int]
    let per_token_latencies_ms: [Double]
    let prefill_time_secs: Double
    let decode_time_secs: Double
    let total_time_secs: Double
    let tokens_generated: Int
    let prompt_tokens: Int
    let decode_tps: Double
    let prefill_tps: Double
}

struct BenchResults: Codable {
    let framework: String
    let precision: String
    var runs: [RunResult]
    var peak_rss_mb: Double
}

// MARK: - Helpers

func peakRSSMB() -> Double {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    let kr = withUnsafeMutablePointer(to: &info) { ptr in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), intPtr, &count)
        }
    }
    if kr == KERN_SUCCESS {
        return Double(info.resident_size_max) / (1024 * 1024)
    }
    return 0
}

func log(_ msg: String) {
    FileHandle.standardError.write(Data((msg + "\n").utf8))
}

func applyChatTemplate(tokenizer: any Tokenizer, systemPrompt: String, userMessage: String) throws -> [Int] {
    let messages: [[String: String]] = [
        ["role": "system", "content": systemPrompt],
        ["role": "user", "content": userMessage],
    ]
    return try tokenizer.applyChatTemplate(messages: messages)
}

// MARK: - Generation

struct GenerationResult {
    var tokenIds: [Int]
    var perTokenLatenciesMs: [Double]
    var prefillTimeSecs: Double
    var decodeTimeSecs: Double
    var totalTimeSecs: Double
    var tokensGenerated: Int
    var promptTokenCount: Int
}

func generateTokens(
    model: any LanguageModel,
    tokenizer: any Tokenizer,
    promptTokens: [Int],
    maxNewTokens: Int
) -> GenerationResult {
    let eosTokenId = tokenizer.unknownTokenId ?? 151645 // fallback Qwen3 EOS
    let eosId = tokenizer.eosTokenId ?? eosTokenId

    let cache = model.newCache(parameters: GenerateParameters(temperature: 0))
    let promptArray = MLXArray(promptTokens)

    // --- Prefill ---
    let prefillStart = Date.timeIntervalSinceReferenceDate

    // Forward pass with full prompt. Add batch dimension: [seqLen] → [1, seqLen]
    let inputText = LMInput.Text(tokens: promptArray.expandedDimensions(axis: 0))
    let prefillOutput = model(inputText, cache: cache, state: nil)
    eval(prefillOutput.logits)

    let prefillEnd = Date.timeIntervalSinceReferenceDate
    let prefillTime = prefillEnd - prefillStart

    // Extract first token: argmax of last position's logits
    // logits shape: [1, seqLen, vocabSize]
    let seqLen = prefillOutput.logits.dim(1)
    let lastLogits = prefillOutput.logits[0, seqLen - 1]
    let firstTokenArray = argMax(lastLogits)
    eval(firstTokenArray)
    let firstTokenId = firstTokenArray.item(Int.self)

    var tokenIds = [firstTokenId]
    var perTokenLatenciesMs: [Double] = []
    var currentToken = firstTokenId

    if currentToken == eosId {
        return GenerationResult(
            tokenIds: tokenIds,
            perTokenLatenciesMs: perTokenLatenciesMs,
            prefillTimeSecs: prefillTime,
            decodeTimeSecs: 0,
            totalTimeSecs: prefillTime,
            tokensGenerated: 1,
            promptTokenCount: promptTokens.count
        )
    }

    // --- Decode loop ---
    let decodeStart = Date.timeIntervalSinceReferenceDate

    for _ in 0..<(maxNewTokens - 1) {
        let t0 = Date.timeIntervalSinceReferenceDate

        // Single token input: [1] → [1, 1] with batch dimension
        let tokenInput = LMInput.Text(tokens: MLXArray([currentToken]).expandedDimensions(axis: 0))
        let result = model(tokenInput, cache: cache, state: nil)
        eval(result.logits)

        let nextToken = argMax(result.logits[0, 0]) // [1, 1, vocabSize] → scalar
        eval(nextToken)

        let t1 = Date.timeIntervalSinceReferenceDate

        currentToken = nextToken.item(Int.self)
        tokenIds.append(currentToken)
        perTokenLatenciesMs.append((t1 - t0) * 1000.0)

        if currentToken == eosId { break }
    }

    let decodeEnd = Date.timeIntervalSinceReferenceDate
    let decodeTime = decodeEnd - decodeStart
    let totalTime = prefillTime + decodeTime

    return GenerationResult(
        tokenIds: tokenIds,
        perTokenLatenciesMs: perTokenLatenciesMs,
        prefillTimeSecs: prefillTime,
        decodeTimeSecs: decodeTime,
        totalTimeSecs: totalTime,
        tokensGenerated: tokenIds.count,
        promptTokenCount: promptTokens.count
    )
}

// MARK: - Main

func runBenchmark() async throws {
    // Parse CLI args
    let args = ProcessInfo.processInfo.arguments
    var modelPath: String?
    var promptsFile: String?
    var outputPath: String?

    var i = 1
    while i < args.count {
        switch args[i] {
        case "--model-path":
            i += 1; modelPath = args[i]
        case "--prompts-file":
            i += 1; promptsFile = args[i]
        case "--output":
            i += 1; outputPath = args[i]
        default:
            break
        }
        i += 1
    }

    guard let modelPath, let promptsFile else {
        log("Usage: mlx-swift-bench --model-path <path> --prompts-file <path> [--output <path>]")
        Foundation.exit(1)
    }

    // Load prompts
    let promptsData = try Data(contentsOf: URL(fileURLWithPath: promptsFile))
    let prompts = try JSONDecoder().decode(PromptsFile.self, from: promptsData)
    let config = prompts.config

    // Load model
    log("Loading model from \(modelPath)...")
    let modelDir = URL(filePath: modelPath)
    let configuration = ModelConfiguration(directory: modelDir)
    let context = try await LLMModelFactory.shared.load(configuration: configuration)
    let model = context.model
    let tokenizer = context.tokenizer
    log("Model loaded.")

    // Warmup
    let warmupTokens = try applyChatTemplate(
        tokenizer: tokenizer,
        systemPrompt: prompts.warmup.system_prompt,
        userMessage: prompts.warmup.user_message
    )
    log("Running \(config.warmup_iterations) warmup iterations...")
    for i in 0..<config.warmup_iterations {
        _ = generateTokens(model: model, tokenizer: tokenizer, promptTokens: warmupTokens, maxNewTokens: 16)
        log("  warmup \(i + 1)/\(config.warmup_iterations) done")
    }

    // Timed runs
    var results = BenchResults(
        framework: "mlx-swift",
        precision: "float16",
        runs: [],
        peak_rss_mb: 0
    )

    let totalRuns = prompts.prompts.count * config.timed_iterations
    var runNum = 0

    for promptInfo in prompts.prompts {
        let promptTokens = try applyChatTemplate(
            tokenizer: tokenizer,
            systemPrompt: promptInfo.system_prompt,
            userMessage: promptInfo.user_message
        )
        log("\nPrompt '\(promptInfo.id)': \(promptTokens.count) input tokens")

        for iteration in 0..<config.timed_iterations {
            runNum += 1
            if runNum > 1 {
                try await Task.sleep(nanoseconds: UInt64(config.sleep_between_runs_secs) * 1_000_000_000)
            }

            FileHandle.standardError.write(
                Data("  iteration \(iteration + 1)/\(config.timed_iterations) (run \(runNum)/\(totalRuns))...".utf8)
            )

            let result = generateTokens(
                model: model,
                tokenizer: tokenizer,
                promptTokens: promptTokens,
                maxNewTokens: config.max_new_tokens
            )

            let decodeTokens = max(result.tokensGenerated - 1, 0)
            let decodeTps = result.decodeTimeSecs > 0
                ? Double(decodeTokens) / result.decodeTimeSecs : 0
            let prefillTps = result.prefillTimeSecs > 0
                ? Double(result.promptTokenCount) / result.prefillTimeSecs : 0

            log(" \(result.tokensGenerated) tokens, decode \(String(format: "%.1f", decodeTps)) tok/s, prefill \(String(format: "%.1f", prefillTps)) tok/s")

            results.runs.append(RunResult(
                prompt_id: promptInfo.id,
                iteration: iteration,
                token_ids: result.tokenIds,
                per_token_latencies_ms: result.perTokenLatenciesMs,
                prefill_time_secs: result.prefillTimeSecs,
                decode_time_secs: result.decodeTimeSecs,
                total_time_secs: result.totalTimeSecs,
                tokens_generated: result.tokensGenerated,
                prompt_tokens: result.promptTokenCount,
                decode_tps: decodeTps,
                prefill_tps: prefillTps
            ))
        }
    }

    results.peak_rss_mb = peakRSSMB()

    // Output JSON
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let jsonData = try encoder.encode(results)

    if let outputPath {
        try jsonData.write(to: URL(fileURLWithPath: outputPath))
        log("\nResults written to \(outputPath)")
    } else {
        FileHandle.standardOutput.write(jsonData)
        FileHandle.standardOutput.write(Data("\n".utf8))
    }
}

@main
struct SwiftBench {
    static func main() async throws {
        try await runBenchmark()
    }
}
