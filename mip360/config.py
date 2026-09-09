"""Small, strict JSON experiment-config loader.

Configs contain only portable experiment choices.  Scene, checkpoint, and output
paths stay on the command line so a checked-in recipe never captures one
machine's directory layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = 1
_TOP_LEVEL_KEYS = {
    "schema_version",
    "name",
    "description",
    "provenance",
    "training",
    "evaluation",
}
_PATH_KEYS = {
    "data_dir",
    "output_dir",
    "resume_ckpt",
    "scene_ckpt",
    "save_json",
}


def _action_by_dest(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {
        action.dest: action
        for action in parser._actions  # argparse has no public action iterator.
        if action.dest not in {argparse.SUPPRESS, "help"}
    }


def _validated_value(
    parser: argparse.ArgumentParser,
    action: argparse.Action,
    key: str,
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(action.default, bool):
        if not isinstance(value, bool):
            parser.error(f"config key {key!r} must be true or false")
        converted = value
    elif action.type is not None:
        try:
            converted = action.type(value)
        except (TypeError, ValueError) as exc:
            parser.error(f"invalid value for config key {key!r}: {exc}")
    else:
        converted = value

    if action.choices is not None and converted not in action.choices:
        choices = ", ".join(map(str, action.choices))
        parser.error(f"config key {key!r} must be one of: {choices}")
    return converted


def load_config_defaults(
    parser: argparse.ArgumentParser,
    config_path: str | Path,
    section: str,
) -> dict[str, Any]:
    """Load and validate one command section from a versioned JSON config."""
    path = Path(config_path).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        parser.error(f"config file not found: {path}")
    except json.JSONDecodeError as exc:
        parser.error(f"invalid JSON config {path}: {exc}")

    if not isinstance(payload, dict):
        parser.error(f"config root must be an object: {path}")
    unknown_top = sorted(set(payload) - _TOP_LEVEL_KEYS)
    if unknown_top:
        parser.error(f"unknown top-level config keys: {', '.join(unknown_top)}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        parser.error(
            f"config schema_version must be {SCHEMA_VERSION}, "
            f"got {payload.get('schema_version')!r}"
        )

    values = payload.get(section)
    if not isinstance(values, dict):
        parser.error(f"config must contain an object named {section!r}: {path}")

    actions = _action_by_dest(parser)
    unknown = sorted(set(values) - set(actions))
    if unknown:
        parser.error(f"unknown {section} config keys: {', '.join(unknown)}")
    captured_paths = sorted(set(values) & _PATH_KEYS)
    if captured_paths:
        parser.error(
            "portable configs cannot contain machine-specific path keys: "
            + ", ".join(captured_paths)
        )

    return {
        key: _validated_value(parser, actions[key], key, value)
        for key, value in values.items()
    }


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None,
    section: str,
) -> argparse.Namespace:
    """Apply config values as defaults, preserving CLI-over-config precedence."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config")
    known, _ = pre_parser.parse_known_args(argv)
    if known.config:
        parser.set_defaults(**load_config_defaults(parser, known.config, section))
    args = parser.parse_args(argv)
    if args.config:
        args.config = str(Path(args.config).expanduser().resolve())
    return args
