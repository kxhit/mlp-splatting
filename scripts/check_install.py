#!/usr/bin/env python3
"""Fail-fast environment check before a CUDA run."""

from __future__ import annotations

import shutil
import subprocess

import torch


def main() -> int:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise SystemExit("nvcc was not found on PATH; install/load a CUDA toolkit")
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch cannot see a CUDA GPU")
    nvcc_version = subprocess.run(
        [nvcc, "--version"], check=True, capture_output=True, text=True
    ).stdout.strip().splitlines()[-1]
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"nvcc: {nvcc_version}")
    for index in range(torch.cuda.device_count()):
        print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
    print("Environment check passed. The renderer will compile on first use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
