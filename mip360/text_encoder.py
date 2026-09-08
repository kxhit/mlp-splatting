import torch


class CLIPTextEncoder:
    """Lightweight CLIP text-encoding wrapper for viewer-side query embedding."""

    def __init__(self, device: torch.device, model_name: str = "ViT-B/32", download_root: str = "/tmp/tmp_clip"):
        self.device = device
        self.model_name = model_name
        self.download_root = download_root
        self._clip = None
        self._model = None

    def _lazy_init(self):
        if self._model is not None:
            return
        try:
            import clip  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Failed to import clip. Install dependency (e.g. pip install git+https://github.com/openai/CLIP.git)."
            ) from exc

        self._clip = clip
        model, _ = clip.load(self.model_name, device=str(self.device), download_root=self.download_root)
        self._model = model.float().eval()

    @torch.no_grad()
    def encode_texts(self, texts) -> torch.Tensor:
        self._lazy_init()
        assert self._clip is not None
        assert self._model is not None

        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, (list, tuple)):
            raise TypeError("texts must be a string or a list/tuple of strings.")
        clean_texts = [str(t) for t in texts if str(t).strip() != ""]
        if len(clean_texts) == 0:
            raise ValueError("texts must include at least one non-empty prompt.")

        tokens = self._clip.tokenize(clean_texts).to(self.device)
        feats = self._model.encode_text(tokens).float()
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        return self.encode_texts([text])[0]
