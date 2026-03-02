import torch
import os
import sys


def _strip_prefixes(state_dict: dict) -> dict:
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        if nk.startswith("module."):
            nk = nk[len("module."):]
        new_sd[nk] = v
    return new_sd


def infer_model_dims(state_dict: dict) -> tuple[int, int, int, int]:
    # returns (num_primitives, hidden_dim, num_frequencies, feature_dim_low)
    if "positions" in state_dict:
        num_primitives = state_dict["positions"].shape[0]
    elif "mlps.lin1.weight" in state_dict:
        num_primitives = state_dict["mlps.lin1.weight"].shape[0]
    elif "W1" in state_dict:
        num_primitives = state_dict["W1"].shape[0]
    else:
        raise KeyError("Cannot infer num_primitives from checkpoint state_dict")

    if "mlps.lin1.weight" in state_dict and "mlps.lin2.weight" in state_dict:
        # vmodel/cuda layout:
        # - mlps.lin1.weight: (N, H, Denc)
        # - mlps.lin2.weight: (N, H, H)
        h = state_dict["mlps.lin2.weight"].shape[2]
        denc = state_dict["mlps.lin1.weight"].shape[2]
    elif "W1" in state_dict and "W2" in state_dict:
        # tile_cuda/splat_tile_cuda layout:
        # - W1: (N, H, Denc)
        # - W2: (N, H, H)
        h = state_dict["W2"].shape[1]
        denc = state_dict["W1"].shape[2]
    else:
        raise KeyError(
            "Cannot infer hidden_dim/encoding dim from checkpoint; "
            "expected either mlps.lin*.weight or W* tensors"
        )

    # encoded_dim = 6 * num_frequencies * 2
    assert denc % (6 * 2) == 0, "Encoded dim not divisible by 12"
    num_f = denc // 12
    feature_dim_low = 0
    if "feat_attr" in state_dict:
        feat_attr = state_dict["feat_attr"]
        if torch.is_tensor(feat_attr) and feat_attr.ndim == 2:
            feature_dim_low = int(feat_attr.shape[1])
    return num_primitives, h, num_f, feature_dim_low


def build_scene_from_ckpt(ckpt_path: str, device: torch.device, engine: str | None = None):
    # Import model robustly whether run as package or script, with engine switch
    if engine is None:
        engine = os.environ.get("MLPSP_ENGINE", "legacy")
    engine = str(engine).lower()

    # Resolve imports based on engine
    SceneCls = None
    if engine == "cuda":
        try:
            from ..cumodel import NeuralPrimitiveScene as SceneImpl  # type: ignore
        except Exception:
            try:
                from cumodel import NeuralPrimitiveScene as SceneImpl  # type: ignore
            except Exception:
                parent = os.path.dirname(os.path.dirname(__file__))
                if parent not in sys.path:
                    sys.path.append(parent)
                from cumodel import NeuralPrimitiveScene as SceneImpl  # type: ignore
        # SceneCls = SceneImpl
    elif engine == "tile_cuda":
        try:
            from ..cumodel import NeuralPrimitiveSceneTileCUDA as SceneImpl  # type: ignore
        except Exception:
            try:
                from cumodel import NeuralPrimitiveSceneTileCUDA as SceneImpl  # type: ignore
            except Exception:
                parent = os.path.dirname(os.path.dirname(__file__))
                if parent not in sys.path:
                    sys.path.append(parent)
                from cumodel import NeuralPrimitiveSceneTileCUDA as SceneImpl  # type: ignore
        # SceneCls = SceneImpl
    elif engine == "splat_tile_cuda":
        try:
            from ..cumodel import NeuralPrimitiveSceneSplatTileCUDA as SceneImpl  # type: ignore
        except Exception:
            try:
                from cumodel import NeuralPrimitiveSceneSplatTileCUDA as SceneImpl  # type: ignore
            except Exception:
                parent = os.path.dirname(os.path.dirname(__file__))
                if parent not in sys.path:
                    sys.path.append(parent)
                from cumodel import NeuralPrimitiveSceneSplatTileCUDA as SceneImpl  # type: ignore
    else:
        try:
            from ..vmodel import NeuralPrimitiveScene as SceneImpl  # type: ignore
        except Exception:
            try:
                from vmodel import NeuralPrimitiveScene as SceneImpl  # type: ignore
            except Exception:
                parent = os.path.dirname(os.path.dirname(__file__))
                if parent not in sys.path:
                    sys.path.append(parent)
                from vmodel import NeuralPrimitiveScene as SceneImpl  # type: ignore
    SceneCls = SceneImpl

    raw_sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw_sd, dict) and "scene_state_dict" in raw_sd:
        raw_sd = raw_sd["scene_state_dict"]
    sd = _strip_prefixes(raw_sd)

    # Remap legacy MLP weights to tile engine parameter names if needed
    def _remap_for_tile(sd_in: dict) -> dict:
        if not ("mlps.lin1.weight" in sd_in and "mlps.lin2.weight" in sd_in and "mlps.lin3.weight" in sd_in):
            return sd_in
        out = {}
        # carry over common tensors
        for k in ("positions", "log_scales"):
            if k in sd_in:
                out[k] = sd_in[k]
        # map mlps -> W*, b*
        out["W1"] = sd_in["mlps.lin1.weight"]
        out["b1"] = sd_in["mlps.lin1.bias"]
        out["W2"] = sd_in["mlps.lin2.weight"]
        out["b2"] = sd_in["mlps.lin2.bias"]
        out["W3"] = sd_in["mlps.lin3.weight"]
        out["b3"] = sd_in["mlps.lin3.bias"]
        return out

    N, hidden_dim, num_f, feature_dim_low = infer_model_dims(sd)

    print(f"[viewer] N: {N}, hidden_dim: {hidden_dim}, num_f: {num_f}, feature_dim_low: {feature_dim_low}")

    # Try to preserve model's topk if present via buffer heuristics (none saved), fallback to default 128
    # scene = SceneCls(
    scene_kwargs = dict(
        num_primitives=N,
        hidden_dim=hidden_dim,
        pe_frequencies=num_f,
        # topk=N,
        # topk=64,
        use_integrated_pe=True,
    )
    if SceneCls.__name__ == "NeuralPrimitiveSceneTileCUDA":
        scene_kwargs["feature_dim_low"] = int(feature_dim_low)
    scene = SceneCls(**scene_kwargs).to(device)

    def remove_stat_buffers(scene):
        REMOVE = [
            "stat_sum_w",
            "stat_max_w",
            "stat_hits",
            "stat_contrib_count",
            "stat_sum_werr",
            "alpha_accum"
        ]

        for name in REMOVE:
            if hasattr(scene, name):
                delattr(scene, name)   # ✅ 正确删除 buffer
    remove_stat_buffers(scene)


    # Apply remap for tile engine to avoid unexpected/missing keys
    if SceneCls.__name__ == "NeuralPrimitiveSceneTileCUDA" or SceneCls.__name__ == "NeuralPrimitiveSceneSplatTileCUDA":
        sd = _remap_for_tile(sd)

    missing, unexpected = scene.load_state_dict(sd, strict=False)
    # Ignore benign buffer differences for tile engine
    if SceneCls.__name__ == "NeuralPrimitiveSceneTileCUDA" or SceneCls.__name__ == "NeuralPrimitiveSceneSplatTileCUDA":
        missing = [m for m in missing if m not in ("alpha_accum", "encoder.freq_bands")]
    if len(missing) > 0:
        print("[viewer] missing keys:", missing)
    if len(unexpected) > 0:
        print("[viewer] unexpected keys:", unexpected)

    scene.eval()
    return scene

