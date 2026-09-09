#!/usr/bin/env python3
"""Backward-compatible Replica launcher; prefer the root ``train.py``."""

from __future__ import annotations

import os
import sys


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from mip360 import run as base_run
    base_run.main([*sys.argv[1:], "--dataset", "replica"])


if __name__ == "__main__":
    main()
