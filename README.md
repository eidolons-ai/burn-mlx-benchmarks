# Qwen3-0.6B Benchmark: Burn vs MLX

Reproducible benchmark comparing decode throughput (tokens/sec) of Qwen3-0.6B across three inference implementations on Apple Silicon:

| Framework | Language | GPU Backend |
|-----------|----------|-------------|
| **burn-wgpu** | Rust | WGPU/Metal |
| **burn-mlx** | Rust | MLX/Metal (via [burn-mlx](https://github.com/eidolons-ai/burn-mlx/tree/burn-0-20)) |
| **mlx-lm** | Python | MLX/Metal |
| **mlx-swift** | Swift | MLX/Metal |

All four run the same prompts with greedy decoding (temperature=0) and produce a shared JSON result schema for direct comparison.

## Prerequisites

- macOS 14+ on Apple Silicon (M1/M2/M3/M4)
- [Rust toolchain](https://rustup.rs/) (for Burn benchmark)
- Python 3.12+ and [uv](https://docs.astral.sh/uv/) (for MLX benchmark)
- Swift 5.9+ / Xcode 15+ (for Swift benchmark)
- Qwen3-0.6B model weights in SafeTensors format

### Model weights

The benchmark expects a directory containing `config.json`, `tokenizer.json`, and `model.safetensors`. By default it looks at `./models/Qwen3-0.6B/` relative to this repo. Override with the `MODEL_PATH` environment variable:

```bash
export MODEL_PATH=/path/to/Qwen3-0.6B
```

## Quick start

```bash
# 1. Build everything (venv, Rust release binary, Swift release binary)
./bench/setup.sh

# 2. Run all benchmarks and generate the comparison report
./bench/run_all.sh
```

Results are written to `results/<timestamp>/` containing JSON data, a markdown report, and charts.

## Repository layout

```
bench/
  prompts.json        # Test prompts and benchmark config
  hw_info.py          # Hardware detection (chip, GPU cores, memory)
  mlx_bench.py        # MLX Python benchmark
  compare.py          # Analysis: token verification + report generation
  setup.sh            # One-time build/install
  run_all.sh          # Run all benchmarks end-to-end
  burn/               # Rust/Burn benchmark (Cargo project)
    Cargo.toml
    src/main.rs
  swift/              # Swift/MLX benchmark (SPM project)
    Package.swift
    Sources/Bench.swift
results/              # Created by run_all.sh
  <timestamp>/
    hw_info.json
    mlx_results.json
    burn_wgpu_results.json
    burn_mlx_results.json
    swift_results.json
    verify.txt
    report.md
    decode_tps_bar.png
    latency_cdf.png
```

## Running individual benchmarks

Each benchmark can be run standalone. They all accept the same arguments:

```bash
# MLX Python
source bench/.venv/bin/activate
python bench/mlx_bench.py \
  --model-path /path/to/Qwen3-0.6B \
  --prompts-file bench/prompts.json \
  --output mlx_results.json

# Burn/WGPU (build first with: cd bench/burn && cargo build --release --features wgpu)
bench/burn/target/release/burn-bench-wgpu \
  --model-path /path/to/Qwen3-0.6B \
  --prompts-file bench/prompts.json \
  --output burn_wgpu_results.json

# Burn/MLX (build first with: cd bench/burn && cargo build --release --features mlx)
bench/burn/target/release/burn-bench-mlx \
  --model-path /path/to/Qwen3-0.6B \
  --prompts-file bench/prompts.json \
  --output burn_mlx_results.json

# Swift (build first with: cd bench/swift && swift build -c release)
$(cd bench/swift && swift build -c release --show-bin-path)/mlx-swift-bench \
  --model-path /path/to/Qwen3-0.6B \
  --prompts-file bench/prompts.json \
  --output swift_results.json
```

Omit `--output` to print JSON to stdout.

## Comparing results

```bash
# Check token consistency across frameworks
python bench/compare.py verify mlx_results.json burn_results.json

# Generate markdown report and charts
python bench/compare.py report \
  --hw-info hw_info.json \
  --output-dir ./report \
  mlx_results.json burn_results.json swift_results.json
```

## Benchmark configuration

Edit `bench/prompts.json` to change:

| Setting | Default | Description |
|---------|---------|-------------|
| `max_new_tokens` | 256 | Tokens to generate per prompt |
| `timed_iterations` | 10 | Measurement iterations per prompt |
| `warmup_iterations` | 2 | Warmup runs (results discarded) |
| `sleep_between_runs_secs` | 5 | Cooldown between runs to reduce thermal throttle noise |
| `temperature` | 0.0 | Greedy decoding (deterministic) |

Three test prompts are included: **short** (~20 input tokens), **medium** (~100), and **long** (~500).

## Measurement methodology

**Precision**: Burn/WGPU uses `Wgpu<f16, i32>`. Burn/MLX uses f32 at the Burn interface level (MLX handles f16 internally). MLX Python weights are explicitly cast from bf16 to f16 for storage parity with the WGPU benchmark.

**Timing**: Each framework uses synchronous GPU evaluation to ensure timestamps reflect actual GPU work, not just command submission:

- **Burn (WGPU & MLX)**: `sample_token()` calls `logits.into_data()` which forces GPU-to-CPU sync. Timestamp deltas between successive `Token` callbacks measure real per-token wall time.
- **MLX Python**: `mx.eval()` after each forward pass forces synchronous GPU completion before recording the timestamp.
- **MLX Swift**: `eval()` (from the MLX module) provides the same synchronous guarantee.

**Isolation**: Each framework runs as a separate process, ensuring full GPU memory release between runs.

**Sampling**: Temperature 0 with argmax — deterministic output for reproducibility. Minor token divergence after ~10 tokens is expected due to f16 accumulation differences.
