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


def infer_model_dims(state_dict: dict) -> tuple[int, int, int]:
    # returns (num_primitives, hidden_dim, num_frequencies)
    if "positions" in state_dict:
        num_primitives = state_dict["positions"].shape[0]
    else:
        num_primitives = state_dict["mlps.lin1.weight"].shape[0]

    # lin2: (N, 4, H)
    h = state_dict["mlps.lin2.weight"].shape[2]
    # lin1: (N, H, Denc)
    denc = state_dict["mlps.lin1.weight"].shape[2]
    # encoded_dim = 6 * num_frequencies * 2
    assert denc % (6 * 2) == 0, "Encoded dim not divisible by 12"
    num_f = denc // 12
    return num_primitives, h, num_f


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

    N, hidden_dim, num_f = infer_model_dims(sd)

    print(f"[viewer] N: {N}, hidden_dim: {hidden_dim}, num_f: {num_f}")

    # Try to preserve model's topk if present via buffer heuristics (none saved), fallback to default 128
    # scene = SceneCls(
    scene = SceneCls(
        num_primitives=N,
        hidden_dim=hidden_dim,
        pe_frequencies=num_f,
        # topk=N,
        # topk=64,
        use_integrated_pe=True,
    ).to(device)

    # Apply remap for tile engine to avoid unexpected/missing keys
    if SceneCls.__name__ == "NeuralPrimitiveSceneTileCUDA":
        sd = _remap_for_tile(sd)

    missing, unexpected = scene.load_state_dict(sd, strict=False)
    # Ignore benign buffer differences for tile engine
    if SceneCls.__name__ == "NeuralPrimitiveSceneTileCUDA":
        missing = [m for m in missing if m not in ("alpha_accum", "encoder.freq_bands")]
    if len(missing) > 0:
        print("[viewer] missing keys:", missing)
    if len(unexpected) > 0:
        print("[viewer] unexpected keys:", unexpected)

    scene.eval()
    return scene


