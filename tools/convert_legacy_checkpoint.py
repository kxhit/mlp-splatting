#!/usr/bin/env python3
"""Normalize and convert an MLP Splatting checkpoint without loading CUDA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mip360.checkpoints import convert_checkpoint  # noqa: E402


def _default_output(source: Path, target_slots: int) -> Path:
    suffix = source.suffix or ".pt"
    return source.with_name(f"{source.stem}.slots{target_slots}{suffix}")


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location="cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Canonicalize module/_orig_mod prefixes and legacy mlps.lin* keys, "
            "then losslessly convert directional feature slots."
        )
    )
    parser.add_argument("checkpoint", type=Path, nargs="?", help="trusted input .pt checkpoint")
    parser.add_argument("--input", dest="input_checkpoint", type=Path, help="named form of the input path")
    parser.add_argument("-o", "--output", type=Path, help="new output path")
    parser.add_argument(
        "--target-feature-slots",
        type=int,
        choices=(4, 5),
        default=4,
        help="output feature layout (default: 4 direct directional slots)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.checkpoint is not None and args.input_checkpoint is not None:
        raise ValueError("provide the checkpoint either positionally or with --input, not both")
    source_arg = args.input_checkpoint if args.input_checkpoint is not None else args.checkpoint
    if source_arg is None:
        raise ValueError("an input checkpoint is required")
    source = source_arg.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {source}")
    output = (args.output or _default_output(source, args.target_feature_slots)).expanduser().resolve()
    if output == source:
        raise FileExistsError("output must differ from input; in-place conversion is not supported")
    output.parent.mkdir(parents=True, exist_ok=True)

    raw = _load(source)
    converted, before, after = convert_checkpoint(
        raw, target_feature_slots=args.target_feature_slots
    )

    try:
        with output.open("xb") as handle:
            torch.save(converted, handle)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing output: {output}") from exc
    except BaseException:
        output.unlink(missing_ok=True)
        raise

    print(
        f"Converted {source} -> {output}\n"
        f"  primitives: {after.num_primitives}\n"
        f"  MLP: PE={after.pe_frequencies}, hidden={after.hidden_dim}\n"
        f"  features: {before.feature_slots}x{before.feature_dim} -> "
        f"{after.feature_slots}x{after.feature_dim}"
    )
    if isinstance(raw, dict) and "optimizer_state_dict" in raw and "optimizer_state_dict" not in converted:
        print("  optimizer: removed (slot-shaped moments are incompatible; resume with a fresh optimizer)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
