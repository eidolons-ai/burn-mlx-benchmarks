#!/usr/bin/env python3
"""MLX-LM benchmark for Qwen3-0.6B decode throughput.

Uses a manual generation loop with mx.eval() for accurate per-token timing.
Does NOT use mlx_lm.generate() or generate_step() since those use
mx.async_eval() for pipelining, which makes per-token timing inaccurate.
"""

import argparse
import json
import os
import resource
import sys
import time

import mlx.core as mx
import mlx_lm


def get_peak_rss_mb() -> float:
    """Get peak resident set size in MB (macOS)."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss / (1024 * 1024)


def apply_chat_template(tokenizer, system_prompt: str, user_message: str) -> list[int]:
    """Apply Qwen3 chat template and return token IDs.

    Uses the same template as qwen3-burn's Qwen3Tokenizer::apply_chat_template().
    """
    text = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return tokenizer.encode(text)


def generate_tokens(
    model, tokenizer, prompt_tokens: list[int], max_new_tokens: int
) -> dict:
    """Run one generation pass with per-token timing.

    Returns dict with token_ids, per_token_latencies_ms, prefill_time_secs,
    decode_time_secs, total_time_secs, tokens_generated.
    """
    from mlx_lm.models.cache import make_prompt_cache

    eos_token_id = tokenizer.eos_token_id

    # Set up KV cache
    cache = make_prompt_cache(model)

    prompt_array = mx.array(prompt_tokens)[None]  # [1, seq_len]

    # --- Prefill ---
    t_prefill_start = time.perf_counter()
    logits = model(prompt_array, cache=cache)
    mx.eval(logits)  # Force GPU sync
    t_prefill_end = time.perf_counter()

    prefill_time = t_prefill_end - t_prefill_start

    # First token from prefill logits
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)
    first_token_id = token.item()

    token_ids = [first_token_id]
    per_token_latencies_ms: list[float] = []

    if first_token_id == eos_token_id:
        return {
            "token_ids": token_ids,
            "per_token_latencies_ms": per_token_latencies_ms,
            "ttft_secs": prefill_time,
            "decode_time_secs": 0.0,
            "total_time_secs": prefill_time,
            "tokens_generated": 1,
            "prompt_tokens": len(prompt_tokens),
        }

    # --- Decode loop ---
    t_decode_start = time.perf_counter()

    for _ in range(max_new_tokens - 1):
        t0 = time.perf_counter()
        logits = model(mx.array([[first_token_id if len(token_ids) == 1 else token_ids[-1]]]), cache=cache)
        mx.eval(logits)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        t1 = time.perf_counter()

        tok_id = token.item()
        per_token_latencies_ms.append((t1 - t0) * 1000.0)
        token_ids.append(tok_id)

        if tok_id == eos_token_id:
            break

    t_decode_end = time.perf_counter()
    decode_time = t_decode_end - t_decode_start
    total_time = prefill_time + decode_time

    return {
        "token_ids": token_ids,
        "per_token_latencies_ms": per_token_latencies_ms,
        "ttft_secs": prefill_time,
        "decode_time_secs": decode_time,
        "total_time_secs": total_time,
        "tokens_generated": len(token_ids),
        "prompt_tokens": len(prompt_tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX-LM Qwen3 benchmark")
    parser.add_argument("--model-path", required=True, help="Path to model directory")
    parser.add_argument("--prompts-file", required=True, help="Path to prompts.json")
    parser.add_argument("--output", default=None, help="Output JSON file (default: stdout)")
    args = parser.parse_args()

    # Load prompts config
    with open(args.prompts_file) as f:
        prompts_data = json.load(f)

    config = prompts_data["config"]
    max_new_tokens = config["max_new_tokens"]
    timed_iterations = config["timed_iterations"]
    warmup_iterations = config["warmup_iterations"]
    sleep_secs = config["sleep_between_runs_secs"]

    # Load model
    print(f"Loading model from {args.model_path}...", file=sys.stderr)
    model, tokenizer = mlx_lm.load(args.model_path)

    # Cast bf16 weights to f16 for parity with Burn
    model.apply(
        lambda x: x.astype(mx.float16) if hasattr(x, "dtype") and x.dtype == mx.bfloat16 else x
    )
    mx.eval(model.parameters())
    print("Model loaded and cast to f16.", file=sys.stderr)

    # Warmup
    warmup_prompt = prompts_data["warmup"]
    warmup_tokens = apply_chat_template(
        tokenizer, warmup_prompt["system_prompt"], warmup_prompt["user_message"]
    )
    print(f"Running {warmup_iterations} warmup iterations...", file=sys.stderr)
    for i in range(warmup_iterations):
        _ = generate_tokens(model, tokenizer, warmup_tokens, max_new_tokens=16)
        print(f"  warmup {i + 1}/{warmup_iterations} done", file=sys.stderr)

    # Timed runs
    results = {
        "framework": "mlx-lm",
        "mlx_version": mx.__version__,
        "precision": "float16",
        "runs": [],
    }

    prompts = prompts_data["prompts"]
    total_runs = len(prompts) * timed_iterations
    run_num = 0

    for prompt_info in prompts:
        prompt_tokens = apply_chat_template(
            tokenizer, prompt_info["system_prompt"], prompt_info["user_message"]
        )
        print(
            f"\nPrompt '{prompt_info['id']}': {len(prompt_tokens)} input tokens",
            file=sys.stderr,
        )

        for iteration in range(timed_iterations):
            run_num += 1
            if run_num > 1:
                time.sleep(sleep_secs)

            print(
                f"  iteration {iteration + 1}/{timed_iterations} (run {run_num}/{total_runs})...",
                file=sys.stderr,
                end="",
                flush=True,
            )

            result = generate_tokens(model, tokenizer, prompt_tokens, max_new_tokens)

            # Compute decode TPS
            decode_tokens = result["tokens_generated"] - 1  # exclude first token from prefill
            decode_tps = (
                decode_tokens / result["decode_time_secs"]
                if result["decode_time_secs"] > 0
                else 0.0
            )
            prefill_tps = (
                result["prompt_tokens"] / result["ttft_secs"]
                if result["ttft_secs"] > 0
                else 0.0
            )

            run_result = {
                "prompt_id": prompt_info["id"],
                "iteration": iteration,
                "token_ids": result["token_ids"],
                "per_token_latencies_ms": result["per_token_latencies_ms"],
                "ttft_secs": result["ttft_secs"],
                "decode_time_secs": result["decode_time_secs"],
                "total_time_secs": result["total_time_secs"],
                "tokens_generated": result["tokens_generated"],
                "prompt_tokens": result["prompt_tokens"],
                "decode_tps": decode_tps,
                "prefill_tps": prefill_tps,
            }
            results["runs"].append(run_result)

            print(
                f" {result['tokens_generated']} tokens, "
                f"decode {decode_tps:.1f} tok/s, "
                f"prefill {prefill_tps:.1f} tok/s",
                file=sys.stderr,
            )

    results["peak_rss_mb"] = get_peak_rss_mb()

    # Output
    output_json = json.dumps(results, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
            f.write("\n")
        print(f"\nResults written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
