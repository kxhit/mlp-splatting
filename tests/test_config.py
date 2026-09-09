import json
from pathlib import Path

import pytest

from mip360.config import parse_args_with_config
from mip360.eval_rgb_psnr import build_parser as build_eval_parser
from mip360.run import build_parser as build_train_parser


ROOT = Path(__file__).resolve().parents[1]
MAIN_CONFIG = ROOT / "configs" / "replica" / "room0_lseg.json"
FAST_CONFIG = ROOT / "configs" / "replica" / "room0_lseg_fast.json"


def test_main_config_matches_verified_room0_release():
    train_args = parse_args_with_config(
        build_train_parser(),
        ["--config", str(MAIN_CONFIG), "--data", "scene", "--output", "run"],
        section="training",
    )
    assert train_args.dataset == "replica"
    assert train_args.max_points_per_leaf == 10
    assert train_args.num_epochs == 60_000
    assert train_args.tile_order == "tilecenter"
    assert train_args.tile_size == 8
    assert train_args.feature_type == "lseg"
    assert train_args.feature_slots == 5
    assert train_args.expected_views == 70
    assert train_args.expected_primitives == 2398

    eval_args = parse_args_with_config(
        build_eval_parser(),
        [
            "--config", str(MAIN_CONFIG),
            "--data", "scene",
            "--checkpoint", "scene.pt",
        ],
        section="evaluation",
    )
    assert eval_args.dataset_loader == "replica"
    assert eval_args.max_points_per_leaf == 10
    assert eval_args.expected_views == 10
    assert eval_args.tile_order == "tilecenter"
    assert eval_args.tile_size == 8
    assert eval_args.score_threshold == pytest.approx(5e-3)


def test_fast_config_drives_train_and_eval_defaults():
    train_args = parse_args_with_config(
        build_train_parser(),
        ["--config", str(FAST_CONFIG), "--data", "scene", "--output", "run"],
        section="training",
    )
    assert train_args.dataset == "replica"
    assert train_args.max_points_per_leaf == 20
    assert train_args.num_epochs == 60_000
    assert train_args.tile_size == 8
    assert train_args.tile_mosaic_k_views == 4
    assert train_args.feature_type == "lseg"
    assert train_args.feature_slots == 5
    assert train_args.expected_views == 70
    assert train_args.expected_primitives == 1344

    eval_args = parse_args_with_config(
        build_eval_parser(),
        [
            "--config", str(FAST_CONFIG),
            "--data", "scene",
            "--checkpoint", "scene.pt",
        ],
        section="evaluation",
    )
    assert eval_args.dataset_loader == "replica"
    assert eval_args.max_points_per_leaf == 20
    assert eval_args.expected_views == 10
    assert eval_args.tile_order == "tilecenter"
    assert eval_args.score_threshold == pytest.approx(3e-3)


def test_cli_overrides_config():
    args = parse_args_with_config(
        build_train_parser(),
        [
            "--config", str(FAST_CONFIG),
            "--data", "scene",
            "--output", "run",
            "--iterations", "7",
            "--max-points-per-leaf", "10",
        ],
        section="training",
    )
    assert args.num_epochs == 7
    assert args.max_points_per_leaf == 10


@pytest.mark.parametrize(
    "training",
    [
        {"not_a_real_option": 1},
        {"tile_order": "random"},
        {"tile_mosaic": "yes"},
        {"data_dir": "/machine/specific/path"},
    ],
)
def test_invalid_config_fails_early(tmp_path, training):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"schema_version": 1, "training": training}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        parse_args_with_config(
            build_train_parser(),
            ["--config", str(path), "--data", "scene"],
            section="training",
        )
