#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_DIR="$SCRIPT_DIR"

# Default model path (sibling qwen3-burn repo)
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
# Check for any safetensors file
if ! ls "$MODEL_PATH"/*.safetensors 1>/dev/null 2>&1; then
    echo "ERROR: No .safetensors files found in $MODEL_PATH"
    exit 1
fi
echo "  Model weights found."
echo ""

# --- Python venv ---
echo "Setting up Python virtual environment..."
VENV_DIR="$BENCH_DIR/.venv"
if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found. Install it: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi
if [ ! -d "$VENV_DIR" ]; then
    # Prefer python 3.12 for mlx compatibility
    echo "  Creating venv with uv..."
    uv venv --python 3.12 "$VENV_DIR" 2>/dev/null \
        || uv venv --python 3.13 "$VENV_DIR" 2>/dev/null \
        || uv venv "$VENV_DIR"
fi
echo "  Installing Python dependencies..."
VIRTUAL_ENV="$VENV_DIR" uv pip install --quiet mlx-lm matplotlib numpy
source "$VENV_DIR/bin/activate"
echo "  Python venv ready: $(python --version), mlx $(python -c 'import mlx.core; print(mlx.core.__version__)')"
deactivate
echo ""

# --- Build Burn benchmarks ---
echo "Building Burn/WGPU benchmark (release)..."
(cd "$BENCH_DIR/burn" && cargo build --release --features wgpu 2>&1 | tail -1)
cp -f "$BENCH_DIR/burn/target/release/burn-bench" "$BENCH_DIR/burn/target/release/burn-bench-wgpu" 2>/dev/null || true
if [ -f "$BENCH_DIR/burn/target/release/burn-bench-wgpu" ]; then
    echo "  burn-bench-wgpu: OK"
else
    echo "  WARNING: Burn/WGPU build may have failed."
fi

echo "Building Burn/Metal benchmark (release)..."
(cd "$BENCH_DIR/burn" && cargo build --release --features metal 2>&1 | tail -1)
cp -f "$BENCH_DIR/burn/target/release/burn-bench" "$BENCH_DIR/burn/target/release/burn-bench-metal" 2>/dev/null || true
if [ -f "$BENCH_DIR/burn/target/release/burn-bench-metal" ]; then
    echo "  burn-bench-metal: OK"
else
    echo "  WARNING: Burn/Metal build may have failed."
fi

echo "Building Burn/MLX benchmark (release)..."
(cd "$BENCH_DIR/burn" && cargo build --release --features mlx 2>&1 | tail -1)
cp -f "$BENCH_DIR/burn/target/release/burn-bench" "$BENCH_DIR/burn/target/release/burn-bench-mlx" 2>/dev/null || true
# MLX needs its Metal shader library (mlx.metallib) colocated with the binary
METALLIB="$(find "$BENCH_DIR/burn/target/release/build" -path '*/mlx-sys-burn-*/out/build/lib/mlx.metallib' -print -quit 2>/dev/null)"
if [ -n "$METALLIB" ]; then
    cp -f "$METALLIB" "$BENCH_DIR/burn/target/release/mlx.metallib"
fi
if [ -f "$BENCH_DIR/burn/target/release/burn-bench-mlx" ]; then
    echo "  burn-bench-mlx: OK"
else
    echo "  WARNING: Burn/MLX build may have failed."
fi
echo ""

# --- Build Swift benchmark ---
# Must use xcodebuild (not swift build) because Metal shaders require Xcode's
# build system to compile the .metallib that MLX needs at runtime.
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
