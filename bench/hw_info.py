#!/usr/bin/env python3
"""Collect Apple Silicon hardware info and framework versions."""

import json
import subprocess
import sys


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def sysctl(key: str) -> str:
    return run(["sysctl", "-n", key])


def main() -> None:
    info: dict = {}

    # Chip / CPU
    info["chip"] = sysctl("machdep.cpu.brand_string")
    info["p_cores"] = int(sysctl("hw.perflevel0.physicalcpu") or "0")
    info["e_cores"] = int(sysctl("hw.perflevel1.physicalcpu") or "0")
    info["cpu_cores_total"] = int(sysctl("hw.physicalcpu") or "0")

    # GPU
    gpu_line = run(
        ["system_profiler", "SPDisplaysDataType", "-detailLevel", "basic"]
    )
    gpu_cores = ""
    for line in gpu_line.splitlines():
        if "Total Number of Cores" in line:
            gpu_cores = line.split(":")[-1].strip()
            break
    info["gpu_cores"] = int(gpu_cores) if gpu_cores.isdigit() else gpu_cores

    # Memory
    mem_bytes = int(sysctl("hw.memsize") or "0")
    info["memory_gb"] = round(mem_bytes / (1024**3), 1)

    # macOS
    info["macos_version"] = run(["sw_vers", "-productVersion"])
    info["macos_build"] = run(["sw_vers", "-buildVersion"])
    info["kernel"] = sysctl("kern.osrelease")

    # Framework versions
    # MLX (Python)
    try:
        import mlx.core  # type: ignore

        info["mlx_version"] = mlx.core.__version__
    except Exception:
        info["mlx_version"] = run(
            [sys.executable, "-c", "import mlx.core; print(mlx.core.__version__)"]
        ) or "not installed"

    # Rust / Cargo
    cargo_v = run(["cargo", "--version"])
    info["cargo_version"] = cargo_v.split()[-1] if cargo_v else "not installed"

    # Swift
    swift_v = run(["swift", "--version"])
    for part in swift_v.split("\n"):
        if "Swift version" in part:
            info["swift_version"] = part.strip()
            break
    else:
        info["swift_version"] = swift_v or "not installed"

    json.dump(info, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
