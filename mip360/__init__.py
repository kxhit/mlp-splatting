"""Feature-aware, tile-based MLP Splatting renderer."""

from __future__ import annotations

__all__ = ["NeuralPrimitiveSceneTileCUDA"]


def __getattr__(name: str):
    # Keep package import light; the CUDA extension itself is compiled lazily on
    # first renderer use.
    if name == "NeuralPrimitiveSceneTileCUDA":
        from .cumodel import NeuralPrimitiveSceneTileCUDA

        return NeuralPrimitiveSceneTileCUDA
    raise AttributeError(name)
