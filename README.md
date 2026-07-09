# Qwen3-0.6B Benchmark: Burn vs MLX

Reproducible benchmark comparing decode throughput (tokens/sec) of Qwen3-0.6B across several
inference implementations on Apple Silicon, built on **Burn 0.21**.

| Case | Language | Backend | Notes |
|------|----------|---------|-------|
| **burn-metal** | Rust | Metal (CubeCL/MSL) | Burn's recommended Apple-Silicon GPU path |
| **burn-wgpu** | Rust | WGPU (WGSL) | Cross-platform GPU path |
| **burn-mlx** | Rust | MLX/Metal (via [burn-mlx](https://github.com/eidolons-ai/burn-mlx)) | Third-party MLX backend |
| **burn-flex** | Rust | Flex (pure-Rust CPU, `apple-amx`) | New 0.21 CPU backend (replaces ndarray) |
| **burn-onnx** | Rust | MLX/Metal | **ONNX-imported** model (see [ONNX case](#the-onnx-case)) |
| **mlx-lm** | Python | MLX/Metal | Reference implementation |
| **mlx-swift** | Swift | MLX/Metal | Reference implementation |

The Rust cases all decode the hand-coded [qwen3-burn](https://github.com/eidolons-ai/qwen3-burn)
model, **except** `burn-onnx`, which runs a model auto-generated from an ONNX export via
[`burn-onnx`](https://github.com/tracel-ai/burn-onnx). All cases run the same prompts with greedy
decoding (temperature=0) and emit a shared JSON schema for direct comparison.

This repo backs [tracel-ai/burn#4512](https://github.com/tracel-ai/burn/issues/4512).

## What changed in the 0.21 update

- Bumped from Burn **0.20 → 0.21** (requires **Rust ≥ 1.92**). This required porting the
  `qwen3-burn` model and the third-party `burn-mlx` backend to 0.21 (both on their `burn-0-21`
  branches). See [Dependencies](#dependencies--local-development).
- Adopted `burn::backend::Metal` (CubeCL MSL passthrough) as the recommended GPU path, alongside
  the existing WGPU case.
- Added a **`burn-flex`** CPU case (`apple-amx` GEMM) — the new pure-Rust CPU backend that replaces
  ndarray, as suggested in the issue thread.
- Added a **`burn-onnx`** case that auto-generates the Qwen3 model from ONNX and compares it against
  the hand-coded implementation.

## The ONNX case

`burn-onnx` imports Qwen3-0.6B from an ONNX graph using `burn-onnx`'s `ModelGen` codegen (mirroring
[burn-onnx PR #181](https://github.com/tracel-ai/burn-onnx/pull/181)). Two properties drive how it
is benchmarked:

- **No KV cache.** The only Qwen3 ONNX graph `burn-onnx` imports cleanly is exported with
  `use_cache=False` (a stateless `input_ids → logits` forward). Autoregressive decode would be
  O(n²) re-prefill, so **the ONNX case reports TTFT / single-forward prefill and token-id
  correctness only — never a decode-throughput comparison.** It is excluded from the decode tables
  and charts; `compare.py` reads its `decode_measured: false` flag.
- **f32, MLX GPU.** The graph is exported and imported in **f32**. It runs on the **MLX** backend,
  giving a clean same-backend comparison against the hand-coded `burn-mlx` case (auto-generated vs
  hand-written on the same GPU backend). Making the graph run on MLX required implementing
  `gather_nd` in the burn-mlx fork (the one op the graph used that the backend lacked) plus a
  u32→i32 index-readback fix. `--features flex` also works as a pure-Rust CPU fallback. The cubecl
  Metal/WGPU backends still cannot *load* the graph — its bool constant is persisted as
  `Bool(Native)`, which cubecl's `bool_from_data` rejects.

Correctness is checked two ways: a one-shot numeric diff of the imported model's logits against the
ONNX Runtime reference (tolerance `max<1e-2, mean<1e-3`, as in PR #181), and cross-framework
**token-id parity** via `compare.py verify`.

## Prerequisites

- macOS 14+ on Apple Silicon (M1/M2/M3/M4)
- [Rust toolchain](https://rustup.rs/) **≥ 1.92** (for Burn 0.21)
- Python 3.12+ and [uv](https://docs.astral.sh/uv/) (for the MLX benchmark and the ONNX export;
  the ONNX export pulls `torch`/`transformers`/`onnx` into an ephemeral uv environment)
- Swift 5.9+ / Xcode 15+ (for the Swift benchmark)
- Qwen3-0.6B model weights in SafeTensors format

### Model weights

The benchmark expects a directory containing `config.json`, `tokenizer.json`, and
`model.safetensors`. By default it looks at `./models/Qwen3-0.6B/`. Override with `MODEL_PATH`:

```bash
export MODEL_PATH=/path/to/Qwen3-0.6B
```

## Quick start

```bash
# 1. Build everything (venv, ONNX export, Rust release binaries, Swift binary)
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
  get_qwen_onnx.py    # Exports Qwen3-0.6B to a cacheless opset-16 ONNX graph
  compare.py          # Analysis: token verification + report generation
  setup.sh            # One-time build/install
  run_all.sh          # Run all benchmarks end-to-end
  burn/               # Cargo workspace (Burn 0.21)
    bench-common/     # Shared JSON schema + helpers (backend-agnostic)
    qwen-handcoded/   # Hand-coded qwen3-burn benchmark (wgpu|metal|mlx|flex)
    qwen-onnx/        # ONNX-imported benchmark (build.rs runs burn-onnx ModelGen)
  swift/              # Swift/MLX benchmark (SPM project)
results/              # Created by run_all.sh
```

## Dependencies / local development

The `burn-0-21` ports of the two upstream repos are consumed as git dependencies (both on their
`burn-0-21` branches):

- [`qwen3-burn`](https://github.com/eidolons-ai/qwen3-burn) — hand-coded Qwen3 model
- [`burn-mlx`](https://github.com/eidolons-ai/burn-mlx) — third-party MLX backend (this repo's ONNX
  work added `gather_nd` + a u32 index-readback fix here)

To hack on either locally, add a `[patch."https://github.com/eidolons-ai/<repo>.git"]` entry
pointing at a local checkout in `bench/burn/Cargo.toml`.

## Running individual benchmarks

The hand-coded binary is built once per backend feature and copied to a per-backend name by
`setup.sh` (`qwen-handcoded-{wgpu,metal,mlx,flex}`); the ONNX binary is `qwen-onnx-mlx`. Each
accepts the same arguments:

```bash
# MLX Python
source bench/.venv/bin/activate
python bench/mlx_bench.py --model-path "$MODEL_PATH" --prompts-file bench/prompts.json --output mlx_results.json

# Burn/Metal (hand-coded)
bench/burn/target/release/qwen-handcoded-metal --model-path "$MODEL_PATH" --prompts-file bench/prompts.json --output burn_metal_results.json

# Burn/ONNX (MLX GPU)
bench/burn/target/release/qwen-onnx-mlx --model-path "$MODEL_PATH" --prompts-file bench/prompts.json --output burn_onnx_results.json

# Swift
$(cd bench/swift && swift build -c release --show-bin-path)/mlx-swift-bench --model-path "$MODEL_PATH" --prompts-file bench/prompts.json --output swift_results.json
```

To build a single case manually:

```bash
cd bench/burn
cargo build --release -p qwen-handcoded --features metal   # or wgpu | mlx | flex
cargo build --release -p qwen-onnx      --features mlx      # requires ./bench/get_qwen_onnx.py first
```

## Comparing results

```bash
# Check token consistency across frameworks
python bench/compare.py verify mlx_results.json burn_metal_results.json burn_onnx_results.json

# Generate markdown report and charts
python bench/compare.py report --hw-info hw_info.json --output-dir ./report *_results.json
```

## Benchmark configuration

Edit `bench/prompts.json`:

| Setting | Default | Description |
|---------|---------|-------------|
| `max_new_tokens` | 256 | Tokens to generate per prompt (hand-coded cases) |
| `timed_iterations` | 10 | Measurement iterations per prompt |
| `warmup_iterations` | 2 | Warmup runs (results discarded) |
| `sleep_between_runs_secs` | 5 | Cooldown between runs to reduce thermal throttle noise |
| `temperature` | 0.0 | Greedy decoding (deterministic) |

Three test prompts are included: **short** (~20 input tokens), **medium** (~100), **long** (~500).

## Measurement methodology

**Precision**: `burn-wgpu`/`burn-metal`/`burn-mlx` run f16; `burn-flex` runs f32 (Flex only
implements the default `Flex<f32>`); `burn-onnx` runs f32 (as exported). MLX Python weights are cast
to f16 for parity with the GPU cases.

**Timing**: Each framework forces synchronous GPU/CPU evaluation so timestamps reflect real work.
The hand-coded Burn cases read back logits (`into_data()`) on each sampled token; the ONNX case
reads back the prefill logits to time a single full-sequence forward. MLX Python/Swift call
`eval()`.

**Decode vs prefill**: decode throughput is only meaningful for KV-cached models, so the decode
tables/charts include the hand-coded and MLX cases only. The ONNX case (cacheless) appears in the
TTFT and **Prefill Throughput** tables and in the token-parity check.

**Isolation**: each framework runs as a separate process for clean GPU memory release.

**Sampling**: temperature 0 / argmax. Minor token divergence after ~10 tokens is expected across
frameworks due to f16 vs f32 accumulation differences.
