#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_DIR="$SCRIPT_DIR"

# Default model path
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/Qwen3-0.6B}"
PROMPTS_FILE="$BENCH_DIR/prompts.json"

# Results directory with timestamp
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$REPO_ROOT/results/$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

echo "=== Qwen3-0.6B Benchmark Suite ==="
echo "Model:   $MODEL_PATH"
echo "Results: $RESULTS_DIR"
echo ""

# Activate venv for Python scripts
VENV_DIR="$BENCH_DIR/.venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "WARNING: No venv found at $VENV_DIR. Run setup.sh first."
    echo "Continuing with system Python..."
fi

# --- Hardware info ---
echo ">>> Collecting hardware info..."
python3 "$BENCH_DIR/hw_info.py" > "$RESULTS_DIR/hw_info.json"
echo "    Saved to hw_info.json"
echo ""

# --- MLX Python benchmark ---
echo ">>> Running MLX Python benchmark..."
python3 "$BENCH_DIR/mlx_bench.py" \
    --model-path "$MODEL_PATH" \
    --prompts-file "$PROMPTS_FILE" \
    --output "$RESULTS_DIR/mlx_results.json"
echo "    Saved to mlx_results.json"
echo ""

# --- Burn/WGPU benchmark ---
BURN_WGPU_BIN="$BENCH_DIR/burn/target/release/burn-bench-wgpu"
if [ -f "$BURN_WGPU_BIN" ]; then
    echo ">>> Running Burn/WGPU benchmark..."
    "$BURN_WGPU_BIN" \
        --model-path "$MODEL_PATH" \
        --prompts-file "$PROMPTS_FILE" \
        --output "$RESULTS_DIR/burn_wgpu_results.json"
    echo "    Saved to burn_wgpu_results.json"
    echo ""
else
    echo ">>> SKIPPING Burn/WGPU benchmark (binary not found). Run setup.sh first."
    echo ""
fi

# --- Burn/MLX benchmark ---
BURN_MLX_BIN="$BENCH_DIR/burn/target/release/burn-bench-mlx"
if [ -f "$BURN_MLX_BIN" ]; then
    echo ">>> Running Burn/MLX benchmark..."
    "$BURN_MLX_BIN" \
        --model-path "$MODEL_PATH" \
        --prompts-file "$PROMPTS_FILE" \
        --output "$RESULTS_DIR/burn_mlx_results.json"
    echo "    Saved to burn_mlx_results.json"
    echo ""
else
    echo ">>> SKIPPING Burn/MLX benchmark (binary not found). Run setup.sh first."
    echo ""
fi

# --- MLX Swift benchmark ---
SWIFT_BIN="$BENCH_DIR/swift/.build/xcode/Build/Products/Release/mlx-swift-bench"
if [ -f "$SWIFT_BIN" ]; then
    echo ">>> Running MLX Swift benchmark..."
    "$SWIFT_BIN" \
        --model-path "$MODEL_PATH" \
        --prompts-file "$PROMPTS_FILE" \
        --output "$RESULTS_DIR/swift_results.json"
    echo "    Saved to swift_results.json"
    echo ""
else
    echo ">>> SKIPPING MLX Swift benchmark (binary not found). Run setup.sh first."
    echo ""
fi

# --- Collect available result files ---
RESULT_FILES=()
for f in mlx_results.json burn_wgpu_results.json burn_mlx_results.json swift_results.json; do
    if [ -f "$RESULTS_DIR/$f" ]; then
        RESULT_FILES+=("$RESULTS_DIR/$f")
    fi
done

if [ ${#RESULT_FILES[@]} -lt 1 ]; then
    echo "ERROR: No result files found. Something went wrong."
    exit 1
fi

# --- Token verification ---
if [ ${#RESULT_FILES[@]} -ge 2 ]; then
    echo ">>> Verifying token consistency..."
    python3 "$BENCH_DIR/compare.py" verify "${RESULT_FILES[@]}" | tee "$RESULTS_DIR/verify.txt"
    echo ""
fi

# --- Generate report ---
echo ">>> Generating report..."
python3 "$BENCH_DIR/compare.py" report \
    --output-dir "$RESULTS_DIR" \
    --hw-info "$RESULTS_DIR/hw_info.json" \
    "${RESULT_FILES[@]}"
echo ""

echo "=== Benchmark complete ==="
echo "Results directory: $RESULTS_DIR"
echo ""
echo "Files:"
ls -la "$RESULTS_DIR/"
