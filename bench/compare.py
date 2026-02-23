#!/usr/bin/env python3
"""Compare benchmark results across frameworks.

Subcommands:
  verify  - Compare token IDs across frameworks for each prompt
  report  - Generate summary statistics, markdown table, and charts
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# verify subcommand
# ---------------------------------------------------------------------------

def cmd_verify(args):
    """Compare output token_ids across frameworks for each prompt."""
    result_files = args.results
    all_results = {}
    for path in result_files:
        data = load_results(path)
        fw = data["framework"]
        all_results[fw] = data

    if len(all_results) < 2:
        print("Need at least 2 result files to compare.", file=sys.stderr)
        sys.exit(1)

    # Group runs by prompt_id, take first iteration for comparison
    frameworks = list(all_results.keys())
    prompt_ids = set()
    for fw, data in all_results.items():
        for run in data["runs"]:
            prompt_ids.add(run["prompt_id"])

    prompt_ids = sorted(prompt_ids)
    all_match = True

    for prompt_id in prompt_ids:
        print(f"\n--- Prompt: {prompt_id} ---")
        fw_tokens = {}
        for fw, data in all_results.items():
            # Use first iteration (iteration 0)
            for run in data["runs"]:
                if run["prompt_id"] == prompt_id and run["iteration"] == 0:
                    fw_tokens[fw] = run["token_ids"]
                    break

        if len(fw_tokens) < 2:
            print(f"  Only {len(fw_tokens)} framework(s) have results for this prompt.")
            continue

        fw_list = list(fw_tokens.keys())
        ref_fw = fw_list[0]
        ref_tokens = fw_tokens[ref_fw]

        for other_fw in fw_list[1:]:
            other_tokens = fw_tokens[other_fw]
            min_len = min(len(ref_tokens), len(other_tokens))

            divergence_point = None
            for i in range(min_len):
                if ref_tokens[i] != other_tokens[i]:
                    divergence_point = i
                    break

            if divergence_point is None and len(ref_tokens) == len(other_tokens):
                print(f"  {ref_fw} vs {other_fw}: MATCH ({len(ref_tokens)} tokens)")
            elif divergence_point is None:
                print(
                    f"  {ref_fw} vs {other_fw}: Match for first {min_len} tokens, "
                    f"but lengths differ ({len(ref_tokens)} vs {len(other_tokens)})"
                )
            else:
                all_match = False
                print(
                    f"  {ref_fw} vs {other_fw}: DIVERGE at token {divergence_point} "
                    f"({ref_tokens[divergence_point]} vs {other_tokens[divergence_point]}), "
                    f"first {divergence_point} tokens match"
                )

    if all_match:
        print("\nAll frameworks produced identical output tokens.")
    else:
        print(
            "\nSome frameworks diverged. This is expected with f16 precision — "
            "small floating-point differences accumulate over generation steps."
        )


# ---------------------------------------------------------------------------
# report subcommand
# ---------------------------------------------------------------------------

def cmd_report(args):
    """Generate summary statistics, markdown, and charts."""
    result_files = args.results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for path in result_files:
        data = load_results(path)
        fw = data["framework"]
        all_results[fw] = data

    # Load hardware info if available
    hw_info = None
    if args.hw_info:
        hw_info = load_results(args.hw_info)

    # Aggregate stats per framework per prompt
    stats = {}  # framework -> prompt_id -> {decode_tps: [...], ttft_secs: [...], ...}
    for fw, data in all_results.items():
        stats[fw] = defaultdict(lambda: defaultdict(list))
        for run in data["runs"]:
            pid = run["prompt_id"]
            stats[fw][pid]["decode_tps"].append(run["decode_tps"])
            # Support both new ttft_secs and legacy prefill_time_secs
            ttft = run.get("ttft_secs", run.get("prefill_time_secs", 0))
            stats[fw][pid]["ttft_secs"].append(ttft)
            stats[fw][pid]["total_time_secs"].append(run["total_time_secs"])
            stats[fw][pid]["tokens_generated"].append(run["tokens_generated"])
            stats[fw][pid]["per_token_latencies_ms"].extend(
                run.get("per_token_latencies_ms", [])
            )

    frameworks = sorted(all_results.keys())
    prompt_ids = set()
    for fw in frameworks:
        prompt_ids.update(stats[fw].keys())
    prompt_ids = sorted(prompt_ids)

    # --- Markdown report ---
    md_lines = ["# Qwen3-0.6B Benchmark: Burn vs MLX", ""]

    if hw_info:
        md_lines.append("## Hardware")
        md_lines.append("")
        md_lines.append(f"- **Chip**: {hw_info.get('chip', 'unknown')}")
        md_lines.append(f"- **GPU Cores**: {hw_info.get('gpu_cores', 'unknown')}")
        md_lines.append(f"- **Memory**: {hw_info.get('memory_gb', 'unknown')} GB")
        md_lines.append(f"- **macOS**: {hw_info.get('macos_version', 'unknown')}")
        md_lines.append("")

    md_lines.append("## Decode Throughput (tokens/sec)")
    md_lines.append("")

    # Build table
    header = "| Framework | Prompt |  Mean | Stddev |   Min |   Max |"
    sep = "|-----------|--------|------:|-------:|------:|------:|"
    md_lines.append(header)
    md_lines.append(sep)

    for fw in frameworks:
        for pid in prompt_ids:
            vals = stats[fw][pid]["decode_tps"]
            if not vals:
                continue
            arr = np.array(vals)
            md_lines.append(
                f"| {fw:9s} | {pid:6s} | {arr.mean():5.1f} | {arr.std():6.2f} | {arr.min():5.1f} | {arr.max():5.1f} |"
            )

    md_lines.append("")

    # Overall decode TPS per framework (across all prompts)
    md_lines.append("## Overall Decode Throughput")
    md_lines.append("")
    md_lines.append("| Framework | Mean tok/s | Stddev | Samples |")
    md_lines.append("|-----------|----------:|-------:|--------:|")

    overall_decode = {}
    for fw in frameworks:
        all_vals = []
        for pid in prompt_ids:
            all_vals.extend(stats[fw][pid]["decode_tps"])
        if all_vals:
            arr = np.array(all_vals)
            overall_decode[fw] = arr
            md_lines.append(
                f"| {fw:9s} | {arr.mean():9.1f} | {arr.std():6.2f} | {len(arr):7d} |"
            )

    md_lines.append("")

    # Speedup ratios
    if len(overall_decode) >= 2:
        md_lines.append("## Speedup Ratios")
        md_lines.append("")
        fw_list = sorted(overall_decode.keys(), key=lambda f: overall_decode[f].mean(), reverse=True)
        fastest = fw_list[0]
        for fw in fw_list[1:]:
            ratio = overall_decode[fastest].mean() / overall_decode[fw].mean()
            md_lines.append(f"- **{fastest}** is **{ratio:.1f}x** faster than **{fw}**")
        md_lines.append("")

    # Time to first token
    md_lines.append("## Time to First Token (seconds)")
    md_lines.append("")
    md_lines.append("| Framework | Prompt |   Mean | Stddev |")
    md_lines.append("|-----------|--------|-------:|-------:|")
    for fw in frameworks:
        for pid in prompt_ids:
            vals = stats[fw][pid]["ttft_secs"]
            if not vals:
                continue
            arr = np.array(vals)
            md_lines.append(
                f"| {fw:9s} | {pid:6s} | {arr.mean():6.3f} | {arr.std():6.4f} |"
            )
    md_lines.append("")

    # Peak RSS
    md_lines.append("## Memory Usage")
    md_lines.append("")
    for fw in frameworks:
        rss = all_results[fw].get("peak_rss_mb", "N/A")
        if isinstance(rss, (int, float)):
            md_lines.append(f"- **{fw}**: {rss:.0f} MB peak RSS")
        else:
            md_lines.append(f"- **{fw}**: {rss}")
    md_lines.append("")

    # Correctness summary
    md_lines.append("## Token Correctness")
    md_lines.append("")
    md_lines.append("Run `compare.py verify` for detailed token comparison.")
    md_lines.append("")

    md_content = "\n".join(md_lines)
    md_path = output_dir / "report.md"
    md_path.write_text(md_content)
    print(f"Markdown report: {md_path}")
    print()
    print(md_content)

    # --- Charts ---
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Bar chart: decode tok/s by framework
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(frameworks))
        means = []
        stds = []
        for fw in frameworks:
            all_vals = []
            for pid in prompt_ids:
                all_vals.extend(stats[fw][pid]["decode_tps"])
            arr = np.array(all_vals) if all_vals else np.array([0])
            means.append(arr.mean())
            stds.append(arr.std())

        palette = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=palette[:len(frameworks)])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(frameworks)
        ax.set_ylabel("Decode Throughput (tokens/sec)")
        ax.set_title("Qwen3-0.6B Decode Throughput by Framework")
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{mean:.1f}", ha="center", va="bottom", fontweight="bold")
        fig.tight_layout()
        bar_path = output_dir / "decode_tps_bar.png"
        fig.savefig(bar_path, dpi=150)
        plt.close(fig)
        print(f"Bar chart: {bar_path}")

        # CDF chart: per-token latency
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
        for i, fw in enumerate(frameworks):
            all_latencies = []
            for pid in prompt_ids:
                all_latencies.extend(stats[fw][pid]["per_token_latencies_ms"])
            if not all_latencies:
                continue
            arr = np.sort(all_latencies)
            cdf = np.arange(1, len(arr) + 1) / len(arr)
            color = colors[i % len(colors)]
            ax.plot(arr, cdf, label=fw, color=color, linewidth=2)

        ax.set_xlabel("Per-Token Latency (ms)")
        ax.set_ylabel("CDF")
        ax.set_title("Per-Token Decode Latency CDF")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        cdf_path = output_dir / "latency_cdf.png"
        fig.savefig(cdf_path, dpi=150)
        plt.close(fig)
        print(f"CDF chart: {cdf_path}")

    except ImportError:
        print("matplotlib not installed — skipping charts.", file=sys.stderr)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # verify
    p_verify = subparsers.add_parser("verify", help="Compare token IDs across frameworks")
    p_verify.add_argument("results", nargs="+", help="Result JSON files to compare")

    # report
    p_report = subparsers.add_parser("report", help="Generate summary report and charts")
    p_report.add_argument("results", nargs="+", help="Result JSON files")
    p_report.add_argument("--output-dir", default=".", help="Directory for output files")
    p_report.add_argument("--hw-info", default=None, help="Hardware info JSON file")

    args = parser.parse_args()

    if args.command == "verify":
        cmd_verify(args)
    elif args.command == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
