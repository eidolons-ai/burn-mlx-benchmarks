use burn::prelude::Backend;
use clap::Parser;
use qwen3_burn::model::{GenerationEvent, GenerationParams, QuantizationMode, Qwen3};
use qwen3_burn::sampling::Sampler;
use qwen3_burn::tokenizer::Qwen3Tokenizer;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::ops::ControlFlow;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "burn-bench", about = "Burn benchmark for Qwen3-0.6B")]
struct Args {
    /// Path to model directory (config.json, model.safetensors, tokenizer.json)
    #[arg(long)]
    model_path: String,

    /// Path to prompts.json
    #[arg(long)]
    prompts_file: String,

    /// Output JSON file (default: stdout)
    #[arg(long)]
    output: Option<String>,
}

#[derive(Deserialize)]
struct PromptsFile {
    config: BenchConfig,
    warmup: PromptEntry,
    prompts: Vec<PromptEntry>,
}

#[derive(Deserialize)]
struct BenchConfig {
    max_new_tokens: usize,
    timed_iterations: usize,
    warmup_iterations: usize,
    sleep_between_runs_secs: u64,
    #[allow(dead_code)]
    temperature: f64,
}

#[derive(Deserialize)]
struct PromptEntry {
    id: String,
    system_prompt: String,
    user_message: String,
    #[allow(dead_code)]
    label: Option<String>,
}

#[derive(Serialize)]
struct BenchResults {
    framework: String,
    precision: String,
    runs: Vec<RunResult>,
    peak_rss_mb: f64,
}

#[derive(Serialize)]
struct RunResult {
    prompt_id: String,
    iteration: usize,
    token_ids: Vec<u32>,
    per_token_latencies_ms: Vec<f64>,
    prefill_time_secs: f64,
    decode_time_secs: f64,
    total_time_secs: f64,
    tokens_generated: usize,
    prompt_tokens: usize,
    decode_tps: f64,
    prefill_tps: f64,
}

/// Get peak RSS in MB via mach_task_info (macOS).
fn get_peak_rss_mb() -> f64 {
    use std::mem;

    #[repr(C)]
    #[allow(non_camel_case_types)]
    struct mach_task_basic_info {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: [u32; 2],
        system_time: [u32; 2],
        policy: i32,
        suspend_count: i32,
    }

    extern "C" {
        fn mach_task_self() -> u32;
        fn task_info(
            target_task: u32,
            flavor: u32,
            task_info_out: *mut mach_task_basic_info,
            task_info_count: *mut u32,
        ) -> i32;
    }

    const MACH_TASK_BASIC_INFO: u32 = 20;

    unsafe {
        let mut info: mach_task_basic_info = mem::zeroed();
        let mut count = (mem::size_of::<mach_task_basic_info>() / mem::size_of::<u32>()) as u32;
        let kr = task_info(
            mach_task_self(),
            MACH_TASK_BASIC_INFO,
            &mut info as *mut _,
            &mut count,
        );
        if kr == 0 {
            info.resident_size_max as f64 / (1024.0 * 1024.0)
        } else {
            0.0
        }
    }
}

fn run_generation<B: Backend>(
    model: &mut Qwen3<B>,
    tokenizer: &Qwen3Tokenizer,
    prompt_text: &str,
    max_new_tokens: usize,
) -> (Vec<u32>, Vec<f64>, f64, f64, f64, usize, usize) {
    let mut token_ids: Vec<u32> = Vec::new();
    let mut token_timestamps: Vec<Instant> = Vec::new();
    let mut prefill_time_secs: f64 = 0.0;
    let mut total_time_secs: f64 = 0.0;
    let mut tokens_generated: usize = 0;

    let prompt_tokens = tokenizer.encode(prompt_text).len();

    let mut sampler = Sampler::Argmax;

    let _output = model.generate_streaming(
        tokenizer,
        GenerationParams {
            prompt: prompt_text,
            max_new_tokens,
            temperature: 0.0,
            sampler: &mut sampler,
            prefill_chunk_size: None,
        },
        |event| {
            match event {
                GenerationEvent::Token {
                    token_id,
                    tokens_generated: _,
                    ..
                } => {
                    token_timestamps.push(Instant::now());
                    token_ids.push(token_id);
                }
                GenerationEvent::Done {
                    tokens_generated: tg,
                    total_time_secs: tt,
                    prefill_time_secs: pt,
                    ..
                } => {
                    tokens_generated = tg;
                    total_time_secs = tt;
                    prefill_time_secs = pt;
                }
                GenerationEvent::PrefillProgress { .. } => {}
            }
            ControlFlow::Continue(())
        },
    );

    // Compute per-token latencies from timestamp deltas
    let mut per_token_latencies_ms = Vec::new();
    for i in 1..token_timestamps.len() {
        let delta = token_timestamps[i].duration_since(token_timestamps[i - 1]);
        per_token_latencies_ms.push(delta.as_secs_f64() * 1000.0);
    }

    let decode_time_secs = total_time_secs - prefill_time_secs;

    (
        token_ids,
        per_token_latencies_ms,
        prefill_time_secs,
        decode_time_secs,
        total_time_secs,
        tokens_generated,
        prompt_tokens,
    )
}

fn run_bench<B: Backend>(args: Args, device: burn::prelude::Device<B>, framework: &str, precision: &str) {
    // Load prompts
    let prompts_json = fs::read_to_string(&args.prompts_file).expect("Failed to read prompts file");
    let prompts_data: PromptsFile =
        serde_json::from_str(&prompts_json).expect("Failed to parse prompts JSON");

    let config = &prompts_data.config;

    // Load model
    eprintln!("Loading model from {} ({})...", args.model_path, framework);

    let mut model = Qwen3::<B>::from_pretrained(
        &args.model_path,
        4096,
        QuantizationMode::None,
        &device,
    )
    .expect("Failed to load model");

    let tokenizer_path = format!("{}/tokenizer.json", args.model_path);
    let tokenizer =
        Qwen3Tokenizer::new(&tokenizer_path).expect("Failed to load tokenizer");

    eprintln!("Model loaded.");

    // Warmup
    let warmup_text = tokenizer.apply_chat_template(
        &prompts_data.warmup.system_prompt,
        &prompts_data.warmup.user_message,
    );
    eprintln!(
        "Running {} warmup iterations...",
        config.warmup_iterations
    );
    for i in 0..config.warmup_iterations {
        let _ = run_generation(&mut model, &tokenizer, &warmup_text, 16);
        eprintln!("  warmup {}/{} done", i + 1, config.warmup_iterations);
    }

    // Timed runs
    let mut runs = Vec::new();
    let total_runs = prompts_data.prompts.len() * config.timed_iterations;
    let mut run_num = 0;

    for prompt_info in &prompts_data.prompts {
        let prompt_text = tokenizer.apply_chat_template(
            &prompt_info.system_prompt,
            &prompt_info.user_message,
        );
        let prompt_token_count = tokenizer.encode(&prompt_text).len();
        eprintln!(
            "\nPrompt '{}': {} input tokens",
            prompt_info.id, prompt_token_count
        );

        for iteration in 0..config.timed_iterations {
            run_num += 1;
            if run_num > 1 {
                std::thread::sleep(std::time::Duration::from_secs(
                    config.sleep_between_runs_secs,
                ));
            }

            eprint!(
                "  iteration {}/{} (run {}/{})...",
                iteration + 1,
                config.timed_iterations,
                run_num,
                total_runs
            );

            let (token_ids, per_token_latencies_ms, prefill_time, decode_time, total_time, toks_gen, prompt_toks) =
                run_generation(&mut model, &tokenizer, &prompt_text, config.max_new_tokens);

            let decode_tokens = if toks_gen > 0 { toks_gen - 1 } else { 0 };
            let decode_tps = if decode_time > 0.0 {
                decode_tokens as f64 / decode_time
            } else {
                0.0
            };
            let prefill_tps = if prefill_time > 0.0 {
                prompt_toks as f64 / prefill_time
            } else {
                0.0
            };

            eprintln!(
                " {} tokens, decode {:.1} tok/s, prefill {:.1} tok/s",
                toks_gen, decode_tps, prefill_tps
            );

            runs.push(RunResult {
                prompt_id: prompt_info.id.clone(),
                iteration,
                token_ids,
                per_token_latencies_ms,
                prefill_time_secs: prefill_time,
                decode_time_secs: decode_time,
                total_time_secs: total_time,
                tokens_generated: toks_gen,
                prompt_tokens: prompt_toks,
                decode_tps,
                prefill_tps,
            });
        }
    }

    let results = BenchResults {
        framework: framework.to_string(),
        precision: precision.to_string(),
        runs,
        peak_rss_mb: get_peak_rss_mb(),
    };

    let output_json = serde_json::to_string_pretty(&results).expect("Failed to serialize results");

    if let Some(output_path) = &args.output {
        fs::write(output_path, &output_json).expect("Failed to write output file");
        eprintln!("\nResults written to {}", output_path);
    } else {
        let stdout = std::io::stdout();
        let mut handle = stdout.lock();
        handle
            .write_all(output_json.as_bytes())
            .expect("Failed to write to stdout");
        handle.write_all(b"\n").ok();
    }
}

fn main() {
    let args = Args::parse();

    #[cfg(feature = "wgpu")]
    {
        use burn::backend::wgpu::WgpuDevice;
        use burn::backend::Wgpu;
        type B = Wgpu<half::f16, i32>;
        let device = WgpuDevice::DefaultDevice;
        run_bench::<B>(args, device, "burn-wgpu", "float16");
    }

    #[cfg(feature = "mlx")]
    {
        use burn_mlx::{Mlx, MlxDevice};
        let device = MlxDevice::Gpu;
        run_bench::<Mlx>(args, device, "burn-mlx", "float32");
    }

    #[cfg(not(any(feature = "wgpu", feature = "mlx")))]
    {
        let _ = args;
        eprintln!("No backend enabled. Build with --features wgpu or --features mlx");
        std::process::exit(1);
    }
}
