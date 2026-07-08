//! ONNX-imported Qwen3-0.6B benchmark.
//!
//! The model is generated at build time by `burn-onnx` from a cacheless opset-16
//! ONNX graph (`forward(input_ids) -> logits`, no KV cache). Because there is no
//! KV cache, autoregressive decode would be O(n^2) re-prefill, so this binary
//! does NOT report a decode-throughput comparison. It measures:
//!   * TTFT / single-forward **prefill** throughput over each prompt, and
//!   * **token-id parity** (a short greedy decode via re-prefill) so that
//!     `compare.py verify` can check it against the hand-coded model.
//! It also does a one-shot numeric check against the ONNX Runtime reference
//! logits produced by `get_qwen_onnx.py`.

extern crate alloc;

use bench_common::{get_peak_rss_mb, write_results, BenchResults, PromptsFile, RunResult};
use burn::module::{Initializer, Param};
use burn::prelude::*;
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
use burn_store::{ModuleSnapshot, PytorchStore};
use clap::Parser;
use qwen3_burn::tokenizer::Qwen3Tokenizer;
use std::time::Instant;

// WEIGHTS_PATH, TEST_DATA_PATH, SEQ_LENGTH, VOCAB_SIZE + `qwen_model::Model`.
include!(concat!(env!("OUT_DIR"), "/model_info.rs"));
use qwen_model::Model;

/// Number of tokens greedily decoded (via re-prefill) purely to populate
/// `token_ids` for cross-framework parity. Kept small because each step
/// recomputes the whole sequence (no KV cache).
const PARITY_TOKENS: usize = 24;

const NOTE: &str = "ONNX-imported cacheless graph (no KV cache): TTFT/prefill + \
token-id parity only; decode throughput is not measured.";

#[derive(Parser)]
#[command(name = "qwen-onnx", about = "ONNX-imported Burn benchmark for Qwen3-0.6B")]
struct Args {
    /// Path to model directory (for tokenizer.json)
    #[arg(long)]
    model_path: String,

    /// Path to prompts.json
    #[arg(long)]
    prompts_file: String,

    /// Output JSON file (default: stdout)
    #[arg(long)]
    output: Option<String>,
}

/// Reference test vector saved by `get_qwen_onnx.py` (ONNX Runtime logits).
#[derive(Module, Debug)]
struct TestData<B: Backend> {
    input_ids: Param<Tensor<B, 2, Int>>,
    logits: Param<Tensor<B, 3>>,
}

impl<B: Backend> TestData<B> {
    fn zeros(device: &B::Device) -> Self {
        use burn::module::ParamId;
        Self {
            input_ids: Param::initialized(
                ParamId::new(),
                Tensor::zeros([1, SEQ_LENGTH], device),
            ),
            logits: Initializer::Zeros.init([1, SEQ_LENGTH, VOCAB_SIZE], device),
        }
    }
}

/// Run a single full-sequence forward pass.
fn forward_once<B: Backend>(model: &Model<B>, ids: &[i32], device: &B::Device) -> Tensor<B, 3> {
    let n = ids.len();
    let input = Tensor::<B, 2, Int>::from_data(TensorData::new(ids.to_vec(), [1, n]), device);
    model.forward(input)
}

/// Greedy argmax over the last sequence position.
fn argmax_last<B: Backend>(logits: &Tensor<B, 3>) -> u32 {
    let [_, n, _] = logits.dims();
    let last = logits.clone().slice([0..1, n - 1..n]); // [1, 1, vocab]
    let idx = last.argmax(2); // [1, 1, 1] Int
    idx.into_scalar().elem::<i64>() as u32
}

/// One-shot numeric sanity check vs the ONNX Runtime reference logits.
fn verify_against_reference<B: Backend>(model: &Model<B>, device: &B::Device) {
    if !std::path::Path::new(TEST_DATA_PATH).exists() {
        eprintln!("Reference test data not found at {TEST_DATA_PATH}; skipping numeric check.");
        return;
    }
    let mut test_data = TestData::<B>::zeros(device);
    let mut store = PytorchStore::from_file(TEST_DATA_PATH);
    if let Err(e) = test_data.load_from(&mut store) {
        eprintln!("Could not load reference test data: {e:?}; skipping numeric check.");
        return;
    }
    let input_ids = test_data.input_ids.val();
    let reference = test_data.logits.val();
    let output = model.forward(input_ids);
    let abs = (output - reference).abs();
    let max_diff = abs.clone().max().into_scalar().elem::<f64>();
    let mean_diff = abs.mean().into_scalar().elem::<f64>();
    let ok = max_diff <= 1e-2 && mean_diff <= 1e-3;
    eprintln!(
        "Numeric check vs ONNX Runtime: max_diff={max_diff:.3e}, mean_diff={mean_diff:.3e} -> {}",
        if ok { "PASS" } else { "WARN (exceeds PR#181 tolerance)" }
    );
}

fn run_bench<B: Backend>(args: Args, device: B::Device, framework: &str) {
    let prompts_data = PromptsFile::load(&args.prompts_file);
    let config = &prompts_data.config;

    eprintln!("Loading ONNX-imported model ({framework})...");
    let model = Model::<B>::from_file(WEIGHTS_PATH, &device);
    eprintln!("Model loaded.");

    verify_against_reference::<B>(&model, &device);

    let tokenizer = Qwen3Tokenizer::new(&format!("{}/tokenizer.json", args.model_path))
        .expect("Failed to load tokenizer");

    // Warmup (compiles shaders, allocates buffers).
    let warmup_text = tokenizer.apply_chat_template(
        &prompts_data.warmup.system_prompt,
        &prompts_data.warmup.user_message,
    );
    let warmup_ids: Vec<i32> = tokenizer.encode(&warmup_text).iter().map(|&t| t as i32).collect();
    eprintln!("Running {} warmup iterations...", config.warmup_iterations);
    for _ in 0..config.warmup_iterations {
        let logits = forward_once(&model, &warmup_ids, &device);
        let _ = logits.into_data(); // force sync
    }

    let mut runs = Vec::new();
    let total_runs = prompts_data.prompts.len() * config.timed_iterations;
    let mut run_num = 0;

    for prompt_info in &prompts_data.prompts {
        let prompt_text =
            tokenizer.apply_chat_template(&prompt_info.system_prompt, &prompt_info.user_message);
        let prompt_ids: Vec<i32> = tokenizer.encode(&prompt_text).iter().map(|&t| t as i32).collect();
        let prompt_tokens = prompt_ids.len();
        eprintln!("\nPrompt '{}': {} input tokens", prompt_info.id, prompt_tokens);

        // Greedy parity tokens (deterministic) — computed once, reused per run.
        let mut seq = prompt_ids.clone();
        let mut token_ids: Vec<u32> = Vec::with_capacity(PARITY_TOKENS);
        for _ in 0..PARITY_TOKENS {
            let logits = forward_once(&model, &seq, &device);
            let next = argmax_last(&logits);
            token_ids.push(next);
            seq.push(next as i32);
        }

        for iteration in 0..config.timed_iterations {
            run_num += 1;
            if run_num > 1 {
                std::thread::sleep(std::time::Duration::from_secs(config.sleep_between_runs_secs));
            }
            eprint!(
                "  iteration {}/{} (run {}/{})...",
                iteration + 1,
                config.timed_iterations,
                run_num,
                total_runs
            );

            // Time a single full-sequence prefill forward, synced via readback.
            let start = Instant::now();
            let logits = forward_once(&model, &prompt_ids, &device);
            let _ = logits.into_data();
            let prefill_secs = start.elapsed().as_secs_f64();
            let prefill_tps = if prefill_secs > 0.0 {
                prompt_tokens as f64 / prefill_secs
            } else {
                0.0
            };

            eprintln!(" prefill {:.3}s ({:.0} tok/s)", prefill_secs, prefill_tps);

            runs.push(RunResult {
                prompt_id: prompt_info.id.clone(),
                iteration,
                token_ids: token_ids.clone(),
                per_token_latencies_ms: Vec::new(),
                ttft_secs: prefill_secs,
                decode_time_secs: 0.0,
                total_time_secs: prefill_secs,
                tokens_generated: token_ids.len(),
                prompt_tokens,
                decode_tps: 0.0,
                prefill_tps,
            });
        }
    }

    let results = BenchResults {
        framework: framework.to_string(),
        precision: "float32".to_string(),
        runs,
        peak_rss_mb: get_peak_rss_mb(),
        decode_measured: false,
        notes: Some(NOTE.to_string()),
    };
    write_results(&results, args.output.clone().as_deref());
}

fn main() {
    let args = Args::parse();

    #[cfg(feature = "wgpu")]
    {
        use burn::backend::wgpu::WgpuDevice;
        use burn::backend::Wgpu;
        type B = Wgpu<half::f16, i32>;
        run_bench::<B>(args, WgpuDevice::DefaultDevice, "burn-onnx-wgpu");
    }

    #[cfg(feature = "metal")]
    {
        use burn::backend::wgpu::WgpuDevice;
        use burn::backend::Metal;
        type B = Metal<half::f16, i32>;
        run_bench::<B>(args, WgpuDevice::DefaultDevice, "burn-onnx-metal");
    }

    #[cfg(feature = "mlx")]
    {
        use burn_mlx::{Mlx, MlxDevice};
        // The ONNX graph is exported in f32 (use_cache=False, eager attn), and
        // its weights load as f32, so the backend float type must be f32.
        type B = Mlx<f32>;
        run_bench::<B>(args, MlxDevice::Gpu, "burn-onnx-mlx");
    }

    #[cfg(feature = "flex")]
    {
        use burn::backend::flex::FlexDevice;
        use burn::backend::Flex;
        type B = Flex;
        run_bench::<B>(args, FlexDevice, "burn-onnx-flex");
    }

    #[cfg(not(any(feature = "wgpu", feature = "metal", feature = "mlx", feature = "flex")))]
    {
        let _ = args;
        eprintln!("No backend enabled. Build with --features wgpu|metal|mlx|flex");
        std::process::exit(1);
    }
}
