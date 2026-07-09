#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_DIR="$SCRIPT_DIR"
WS="$BENCH_DIR/burn"           # cargo workspace
REL="$WS/target/release"

# Default model path
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/Qwen3-0.6B}"

echo "=== Qwen3-0.6B Benchmark Setup ==="
echo "Bench dir:  $BENCH_DIR"
echo "Model path: $MODEL_PATH"
echo ""

# --- Verify model weights ---
echo "Checking model weights..."
for f in config.json tokenizer.json; do
    if [ ! -f "$MODEL_PATH/$f" ]; then
        echo "ERROR: Missing $MODEL_PATH/$f"
        echo "Set MODEL_PATH to the directory containing Qwen3-0.6B weights."
        exit 1
    fi
done
if ! ls "$MODEL_PATH"/*.safetensors 1>/dev/null 2>&1; then
    echo "ERROR: No .safetensors files found in $MODEL_PATH"
    exit 1
fi
echo "  Model weights found."
echo ""

# --- Python venv (MLX Python bench) ---
echo "Setting up Python virtual environment..."
VENV_DIR="$BENCH_DIR/.venv"
if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found. Install it: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi
if [ ! -d "$VENV_DIR" ]; then
    echo "  Creating venv with uv..."
    uv venv --python 3.12 "$VENV_DIR" 2>/dev/null \
        || uv venv --python 3.13 "$VENV_DIR" 2>/dev/null \
        || uv venv "$VENV_DIR"
fi
echo "  Installing Python dependencies..."
VIRTUAL_ENV="$VENV_DIR" uv pip install --quiet mlx-lm matplotlib numpy
echo "  Python venv ready: $("$VENV_DIR/bin/python3" --version), mlx $("$VENV_DIR/bin/python3" -c 'import mlx.core; print(mlx.core.__version__)' 2>/dev/null || echo '?')"
echo ""

# --- Build hand-coded Burn benchmark (one binary per backend feature) ---
# The workspace builds a single `qwen-handcoded` binary; we build it once per
# backend feature and copy it to a per-backend name.
build_handcoded() {
    local feat="$1" name="$2"
    echo "Building qwen-handcoded (--features $feat)..."
    (cd "$WS" && cargo build --release -p qwen-handcoded --features "$feat" 2>&1 | tail -1)
    if [ -f "$REL/qwen-handcoded" ]; then
        cp -f "$REL/qwen-handcoded" "$REL/qwen-handcoded-$name"
        echo "  qwen-handcoded-$name: OK"
    else
        echo "  WARNING: qwen-handcoded ($feat) build may have failed."
    fi
}

build_handcoded wgpu wgpu
build_handcoded metal metal
build_handcoded mlx mlx
build_handcoded flex flex

# MLX needs its Metal shader library colocated with the binary.
METALLIB="$(find "$WS/target/release/build" -path '*mlx-sys-burn-*/out/build/lib/mlx.metallib' -print -quit 2>/dev/null)"
if [ -n "$METALLIB" ]; then
    cp -f "$METALLIB" "$REL/mlx.metallib"
    echo "  mlx.metallib colocated."
fi
echo ""

# --- ONNX case: export the cacheless graph, then build (Flex CPU backend) ---
# The ONNX-imported model runs on the Flex backend (see README methodology:
# cubecl Metal/wgpu cannot load its Native-bool constant, and the MLX backend
# lacks gather_nd). Flex is a first-class pure-Rust CPU backend in burn 0.21.
ONNX_ARTIFACT="$WS/qwen-onnx/artifacts/qwen3-0_6b_opset16.onnx"
if [ ! -f "$ONNX_ARTIFACT" ]; then
    echo "Exporting Qwen3-0.6B to ONNX (this downloads torch/transformers via uv)..."
    MODEL_PATH="$MODEL_PATH" "$BENCH_DIR/get_qwen_onnx.py"
else
    echo "ONNX artifact already present: $ONNX_ARTIFACT"
fi

echo "Building qwen-onnx (--features flex)..."
(cd "$WS" && cargo build --release -p qwen-onnx --features flex 2>&1 | tail -1)
if [ -f "$REL/qwen-onnx" ]; then
    cp -f "$REL/qwen-onnx" "$REL/qwen-onnx-flex"
    echo "  qwen-onnx-flex: OK"
else
    echo "  WARNING: qwen-onnx build may have failed."
fi
echo ""

# --- Build Swift benchmark ---
echo "Building Swift benchmark (release via xcodebuild)..."
SWIFT_BUILD_DIR="$BENCH_DIR/swift/.build/xcode"
(cd "$BENCH_DIR/swift" && xcodebuild -scheme mlx-swift-bench -configuration Release \
    -destination 'platform=macOS' -derivedDataPath .build/xcode build 2>&1 | \
    grep -E '(BUILD|error:)' | tail -5)
SWIFT_BIN="$SWIFT_BUILD_DIR/Build/Products/Release/mlx-swift-bench"
if [ -f "$SWIFT_BIN" ]; then
    echo "  Swift binary: $SWIFT_BIN"
else
    echo "  WARNING: Swift build may have failed. Check output above."
fi
echo ""

echo "=== Setup complete ==="
echo ""
echo "Run benchmarks with: ./bench/run_all.sh"
