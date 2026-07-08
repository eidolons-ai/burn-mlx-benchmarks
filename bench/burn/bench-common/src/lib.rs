//! Shared benchmark schema and helpers for the Burn-based Qwen3 benchmarks.
//!
//! Both `qwen-handcoded` and `qwen-onnx` depend on this crate so that they
//! emit a byte-identical JSON shape consumable by `bench/compare.py`. Keeping
//! it backend-agnostic (no `burn` dependency) means the schema can never drift
//! between the two binaries.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;

/// Top-level prompts/config file (`bench/prompts.json`).
#[derive(Deserialize)]
pub struct PromptsFile {
    pub config: BenchConfig,
    pub warmup: PromptEntry,
    pub prompts: Vec<PromptEntry>,
}

#[derive(Deserialize)]
pub struct BenchConfig {
    pub max_new_tokens: usize,
    pub timed_iterations: usize,
    pub warmup_iterations: usize,
    pub sleep_between_runs_secs: u64,
    #[allow(dead_code)]
    pub temperature: f64,
}

#[derive(Deserialize)]
pub struct PromptEntry {
    pub id: String,
    pub system_prompt: String,
    pub user_message: String,
    #[allow(dead_code)]
    pub label: Option<String>,
}

#[derive(Serialize)]
pub struct BenchResults {
    pub framework: String,
    pub precision: String,
    pub runs: Vec<RunResult>,
    pub peak_rss_mb: f64,
    /// Whether this framework produces a meaningful autoregressive *decode*
    /// throughput. The ONNX-imported graph is cacheless (no KV cache), so it
    /// only reports TTFT / prefill + token-id parity; `compare.py` skips it in
    /// the decode tables. Always `true` for the hand-coded frameworks. JSON from
    /// the Python/Swift benches omits the field, which `compare.py` reads as
    /// `true`.
    pub decode_measured: bool,
    /// Free-form methodology note surfaced in the report (e.g. the no-KV-cache
    /// caveat for the ONNX case). `None` for the hand-coded/MLX frameworks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

#[derive(Serialize)]
pub struct RunResult {
    pub prompt_id: String,
    pub iteration: usize,
    pub token_ids: Vec<u32>,
    pub per_token_latencies_ms: Vec<f64>,
    pub ttft_secs: f64,
    pub decode_time_secs: f64,
    pub total_time_secs: f64,
    pub tokens_generated: usize,
    pub prompt_tokens: usize,
    pub decode_tps: f64,
    pub prefill_tps: f64,
}

impl PromptsFile {
    /// Load and parse a prompts.json file.
    pub fn load(path: &str) -> Self {
        let json = fs::read_to_string(path).expect("Failed to read prompts file");
        serde_json::from_str(&json).expect("Failed to parse prompts JSON")
    }
}

/// Serialize results to the given path, or to stdout when `output` is `None`.
pub fn write_results(results: &BenchResults, output: Option<&str>) {
    let json = serde_json::to_string_pretty(results).expect("Failed to serialize results");
    match output {
        Some(path) => {
            fs::write(path, &json).expect("Failed to write output file");
            eprintln!("\nResults written to {}", path);
        }
        None => {
            let stdout = std::io::stdout();
            let mut handle = stdout.lock();
            handle.write_all(json.as_bytes()).expect("Failed to write to stdout");
            handle.write_all(b"\n").ok();
        }
    }
}

/// Get peak resident set size in MB via `mach_task_info` (macOS).
pub fn get_peak_rss_mb() -> f64 {
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
