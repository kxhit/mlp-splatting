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


def build_scene_from_ckpt(ckpt_path: str, device: torch.device):
    # Import model robustly whether run as package or script
    try:
        from ..vmodel import NeuralPrimitiveScene  # type: ignore
    except Exception:
        try:
            from vmodel import NeuralPrimitiveScene  # type: ignore
        except Exception:
            parent = os.path.dirname(os.path.dirname(__file__))
            if parent not in sys.path:
                sys.path.append(parent)
            from vmodel import NeuralPrimitiveScene  # type: ignore

    raw_sd = torch.load(ckpt_path, map_location="cpu")
    sd = _strip_prefixes(raw_sd)

    N, hidden_dim, num_f = infer_model_dims(sd)

    print(f"[viewer] N: {N}, hidden_dim: {hidden_dim}, num_f: {num_f}")

    # Try to preserve model's topk if present via buffer heuristics (none saved), fallback to default 128
    scene = NeuralPrimitiveScene(
        num_primitives=N,
        hidden_dim=hidden_dim,
        pe_frequencies=num_f,
        # topk=N,
        # topk=64,
        use_integrated_pe=True,
    ).to(device)

    missing, unexpected = scene.load_state_dict(sd, strict=False)
    if len(missing) > 0:
        print("[viewer] missing keys:", missing)
    if len(unexpected) > 0:
        print("[viewer] unexpected keys:", unexpected)

    scene.eval()
    return scene


