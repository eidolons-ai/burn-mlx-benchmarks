#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "onnx>=1.17.0",
#   "onnxruntime>=1.18.0",
#   "transformers>=4.51.0",
#   "numpy",
#   "torch",
#   "onnxscript",
# ]
# ///

"""Export Qwen3-0.6B to a cacheless opset-16 ONNX graph for the Burn ONNX case.

This mirrors tracel-ai/burn-onnx PR #181's `crates/model-checks/qwen/get_model.py`:
the model is exported with `use_cache=False` and `attn_implementation="eager"`, so
the graph is a stateless `input_ids -> logits` forward (NO KV cache). It is the only
Qwen3 graph that burn-onnx is known to import cleanly. Autoregressive decode with it
is therefore O(n^2) re-prefill; the benchmark only uses it for TTFT/prefill + token
parity, never for a decode-throughput comparison.

It also emits a `*_test_data.pt` with reference logits (computed via ONNX Runtime) so
the Rust side can assert numeric parity within the same tolerance as PR #181.
"""

import argparse
import sys
from pathlib import Path

SEQ_LENGTH = 32  # fixed sequence length used for the reference test vector
SAFE_NAME = "qwen3-0_6b"


def export_onnx(model_path: str, original_path: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig

    print(f"Loading model from {model_path} ...")
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.eval()
    print(f"  Loaded (vocab_size={config.vocab_size})")

    class LogitsWrapper(torch.nn.Module):
        def __init__(self, causal_lm):
            super().__init__()
            self.causal_lm = causal_lm

        def forward(self, input_ids):
            return self.causal_lm(input_ids).logits

    wrapper = LogitsWrapper(model).eval()

    print("Exporting to ONNX (opset 16, use_cache=False) ...")
    dummy = torch.randint(0, config.vocab_size, (1, SEQ_LENGTH))
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(original_path),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "logits": {0: "batch_size", 1: "sequence"},
            },
            opset_version=16,
            do_constant_folding=True,
        )
    if not original_path.exists():
        raise FileNotFoundError(f"Failed to create ONNX file at {original_path}")
    print(f"  Wrote {original_path}")


def generate_test_data(processed_path: Path, test_data_path: Path, vocab_size: int):
    import numpy as np
    import onnxruntime as ort
    import torch

    print("Generating reference test data via ONNX Runtime ...")
    np.random.seed(42)
    input_ids = np.random.randint(0, vocab_size, size=(1, SEQ_LENGTH), dtype=np.int64)
    session = ort.InferenceSession(str(processed_path))
    logits = session.run(None, {"input_ids": input_ids})[0]
    torch.save(
        {"input_ids": torch.from_numpy(input_ids), "logits": torch.from_numpy(logits)},
        test_data_path,
    )
    print(f"  Wrote {test_data_path} (logits shape {logits.shape})")


def main():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Export Qwen3-0.6B to cacheless ONNX")
    parser.add_argument(
        "--model-path",
        default=str(repo_root / "models" / "Qwen3-0.6B"),
        help="Local HF model directory (or HF repo id) to export",
    )
    parser.add_argument(
        "--out-dir",
        default=str(repo_root / "bench" / "burn" / "qwen-onnx" / "artifacts"),
        help="Directory to write the ONNX graph + reference test data",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    original_path = out_dir / f"{SAFE_NAME}.onnx"
    processed_path = out_dir / f"{SAFE_NAME}_opset16.onnx"
    test_data_path = out_dir / f"{SAFE_NAME}_test_data.pt"

    if processed_path.exists() and test_data_path.exists():
        print(f"Already exported:\n  {processed_path}\n  {test_data_path}\nNothing to do.")
        return

    from transformers import AutoConfig

    vocab_size = AutoConfig.from_pretrained(args.model_path).vocab_size

    if not processed_path.exists():
        if not original_path.exists():
            export_onnx(args.model_path, original_path)
        print("Applying shape inference (file-based, >2GB safe) ...")
        import onnx.shape_inference

        onnx.shape_inference.infer_shapes_path(str(original_path), str(processed_path))
        print(f"  Wrote {processed_path}")
        if original_path.exists():
            original_path.unlink()

    if not test_data_path.exists():
        generate_test_data(processed_path, test_data_path, vocab_size)

    print("\nDone.")
    print(f"  ONNX:      {processed_path}")
    print(f"  Test data: {test_data_path}")


if __name__ == "__main__":
    sys.exit(main())
