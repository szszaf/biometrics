"""Runtime ECAPA: preprocess SpeechBrain + backbone z douczonych wag."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier

from voice_auth.audio_preprocess import build_tta_batch_from_wav, load_wav_mono_16k_from_bytes
from voice_auth.config import VOICE_EMBEDDING_DIM

logger = logging.getLogger(__name__)


class VoiceEmbeddingEngine:
    """Ładuje ``speechbrain/spkrec-ecapa-voxceleb`` (preprocess) i wagi ``embedding_model`` z checkpointu."""

    def __init__(
        self,
        *,
        weights_path: Path,
        speechbrain_savedir: Path,
        device: torch.device,
    ):
        self.device = device
        self.weights_path = Path(weights_path)
        self.speechbrain_savedir = Path(speechbrain_savedir)
        logger.info("Voice: ładuję SpeechBrain ECAPA z %s", self.speechbrain_savedir)
        spk_module = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(self.speechbrain_savedir),
            run_opts={"device": str(device)},
        )
        self.compute_features = spk_module.mods.compute_features
        self.mean_var_norm = spk_module.mods.mean_var_norm
        self.embedding_model = spk_module.mods.embedding_model.to(device)
        try:
            state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(self.weights_path, map_location="cpu")
        state = {k: v.float() if getattr(v, "dtype", None) == torch.float16 else v for k, v in state.items()}
        self.embedding_model.load_state_dict(state, strict=True)
        self.embedding_model.eval()
        self._spk_module = spk_module

    @torch.no_grad()
    def _forward_embedding(self, wavs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """``wavs`` (B, T) na ``self.device`` — L2-normalized (B, D)."""
        feats = self.compute_features(wavs)
        feats = self.mean_var_norm(feats, lengths)
        emb = self.embedding_model(feats)
        emb = emb.squeeze(1)
        return F.normalize(emb, p=2, dim=-1, eps=1e-12)

    @torch.no_grad()
    def embed_from_bytes(self, data: bytes) -> np.ndarray:
        wav = load_wav_mono_16k_from_bytes(data)
        batch, lengths = build_tta_batch_from_wav(wav)
        batch = batch.to(self.device)
        lengths = lengths.to(self.device)
        parts = self._forward_embedding(batch, lengths)
        mean_emb = F.normalize(parts.mean(dim=0), dim=0, eps=1e-12)
        out = mean_emb.cpu().numpy().astype(np.float32)
        if out.shape != (VOICE_EMBEDDING_DIM,):
            raise ValueError(f"unexpected embedding shape {out.shape}")
        return out

    def close(self) -> None:
        self.embedding_model = None
        self._spk_module = None
