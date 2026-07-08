#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_DIR="$SCRIPT_DIR"
REL="$BENCH_DIR/burn/target/release"

MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/Qwen3-0.6B}"
PROMPTS_FILE="$BENCH_DIR/prompts.json"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$REPO_ROOT/results/$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

echo "=== Qwen3-0.6B Benchmark Suite ==="
echo "Model:   $MODEL_PATH"
echo "Results: $RESULTS_DIR"
echo ""

VENV_DIR="$BENCH_DIR/.venv"
if [ -x "$VENV_DIR/bin/python3" ]; then
    PYTHON="$VENV_DIR/bin/python3"
else
    echo "WARNING: No venv found at $VENV_DIR. Run setup.sh first."
    PYTHON="python3"
fi

# --- Hardware info ---
echo ">>> Collecting hardware info..."
"$PYTHON" "$BENCH_DIR/hw_info.py" > "$RESULTS_DIR/hw_info.json"
echo "    Saved to hw_info.json"
echo ""

# --- MLX Python benchmark ---
echo ">>> Running MLX Python benchmark..."
"$PYTHON" "$BENCH_DIR/mlx_bench.py" \
    --model-path "$MODEL_PATH" \
    --prompts-file "$PROMPTS_FILE" \
    --output "$RESULTS_DIR/mlx_results.json"
echo "    Saved to mlx_results.json"
echo ""

# --- Burn benchmarks (hand-coded, one binary per backend) ---
run_bin() {
    local bin="$1" out="$2" label="$3"
    if [ -f "$bin" ]; then
        echo ">>> Running $label benchmark..."
        "$bin" \
            --model-path "$MODEL_PATH" \
            --prompts-file "$PROMPTS_FILE" \
            --output "$RESULTS_DIR/$out"
        echo "    Saved to $out"
        echo ""
    else
        echo ">>> SKIPPING $label (binary not found: $bin). Run setup.sh first."
        echo ""
    fi
}

run_bin "$REL/qwen-handcoded-wgpu"  burn_wgpu_results.json  "Burn/WGPU (hand-coded)"
run_bin "$REL/qwen-handcoded-metal" burn_metal_results.json "Burn/Metal (hand-coded)"
run_bin "$REL/qwen-handcoded-mlx"   burn_mlx_results.json   "Burn/MLX (hand-coded)"
run_bin "$REL/qwen-handcoded-flex"  burn_flex_results.json  "Burn/Flex CPU (hand-coded)"

# ONNX-imported model (Flex CPU backend). Cacheless graph: TTFT/prefill +
# token parity only (see report methodology note).
run_bin "$REL/qwen-onnx-flex"       burn_onnx_results.json  "Burn/ONNX (Flex CPU)"

# --- MLX Swift benchmark ---
SWIFT_BIN="$BENCH_DIR/swift/.build/xcode/Build/Products/Release/mlx-swift-bench"
run_bin "$SWIFT_BIN" swift_results.json "MLX Swift"

# --- Collect available result files ---
RESULT_FILES=()
for f in mlx_results.json burn_wgpu_results.json burn_metal_results.json \
         burn_mlx_results.json burn_flex_results.json burn_onnx_results.json \
         swift_results.json; do
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
    "$PYTHON" "$BENCH_DIR/compare.py" verify "${RESULT_FILES[@]}" | tee "$RESULTS_DIR/verify.txt"
    echo ""
fi

# --- Generate report ---
echo ">>> Generating report..."
"$PYTHON" "$BENCH_DIR/compare.py" report \
    --output-dir "$RESULTS_DIR" \
    --hw-info "$RESULTS_DIR/hw_info.json" \
    "${RESULT_FILES[@]}"
echo ""

echo "=== Benchmark complete ==="
echo "Results directory: $RESULTS_DIR"
echo ""
echo "Files:"
ls -la "$RESULTS_DIR/"
