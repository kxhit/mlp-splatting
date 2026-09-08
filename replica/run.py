#!/usr/bin/env python3
"""Replica entrypoint that reuses mip360/run.py with a Replica-specific loader.

All CLI arguments are forwarded unchanged (including semantic options such as
`--feature_slots`), because argparse is defined in `mip360/run.py`.
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    if __package__:
        from . import replica_loader as replica_loader_module
    else:  # Support ``python replica/run.py``.
        import replica_loader as replica_loader_module

    from mip360 import run as base_run
    from mip360 import train as base_train

    base_train.mip360Dataset = replica_loader_module.mip360Dataset
    base_run.main()


if __name__ == "__main__":
    main()
