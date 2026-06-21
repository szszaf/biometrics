"""Ładowanie audio z bytes i multi-crop jak w notebooku ewaluacji."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from voice_auth.config import (
    AUTH_MAX_AUDIO_SECONDS,
    CROP_SECONDS,
    ENROLLMENT_NOISE_RATIOS,
    MAX_READ_SECONDS,
    NUM_TTA_CROPS,
    TARGET_SAMPLE_RATE,
)


def load_wav_mono_16k_from_bytes(data: bytes) -> torch.Tensor:
    """Zwraca 1-D float tensor mono @ ``TARGET_SAMPLE_RATE`` (CPU)."""
    wav, sr = torchaudio.load(io.BytesIO(data))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if sr != TARGET_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SAMPLE_RATE)
    return wav


def load_wav_mono_16k_from_path(path: Path | str) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    if sr != TARGET_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SAMPLE_RATE)
    return wav


def _cap_auth_length(wav: torch.Tensor) -> torch.Tensor:
    max_len = int(AUTH_MAX_AUDIO_SECONDS * TARGET_SAMPLE_RATE)
    if wav.shape[0] <= max_len:
        return wav
    start = max(0, (wav.shape[0] - max_len) // 2)
    return wav[start : start + max_len]


def _trim_for_crops(wav: torch.Tensor) -> torch.Tensor:
    max_len = int(MAX_READ_SECONDS * TARGET_SAMPLE_RATE)
    if wav.shape[0] > max_len:
        start = max(0, (wav.shape[0] - max_len) // 2)
        return wav[start : start + max_len]
    return wav


def _get_crops(wav: torch.Tensor, n_crops: int, crop_len: int) -> list[torch.Tensor]:
    wav = _trim_for_crops(wav)
    t = wav.shape[0]
    if t <= crop_len:
        padded = F.pad(wav, (0, crop_len - t))
        return [padded] * n_crops
    if n_crops == 1:
        start = max(0, (t - crop_len) // 2)
        return [wav[start : start + crop_len]]
    starts = np.linspace(0, t - crop_len, n_crops).astype(int)
    return [wav[s : s + crop_len] for s in starts]


def add_noise_by_amplitude_torch(signal: torch.Tensor, noise_ratio: float) -> torch.Tensor:
    signal_rms = torch.sqrt(torch.mean(signal**2))
    noise_std = noise_ratio * signal_rms
    noise = torch.randn_like(signal) * noise_std
    return signal + noise


def build_tta_batch_from_wav(wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``wav`` mono float CPU → padded batch (B, T) na urządzeniu docelowym w engine."""
    fixed = _build_fixed_crops(wav)
    return _batch_from_crops(fixed)


def build_enrollment_tta_batch_from_wav(wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Enrollment batch with clean crops and noisy amplitude augmentations."""
    fixed = _build_fixed_crops(wav)
    augmented: list[torch.Tensor] = []
    for crop in fixed:
        augmented.append(crop)
        augmented.extend(add_noise_by_amplitude_torch(crop, ratio) for ratio in ENROLLMENT_NOISE_RATIOS)
    return _batch_from_crops(augmented)


def _build_fixed_crops(wav: torch.Tensor) -> list[torch.Tensor]:
    wav = _cap_auth_length(wav)
    crop_len = int(CROP_SECONDS * TARGET_SAMPLE_RATE)
    crops = _get_crops(wav, NUM_TTA_CROPS, crop_len)
    fixed: list[torch.Tensor] = []
    for c in crops:
        if c.shape[0] < crop_len:
            c = F.pad(c, (0, crop_len - c.shape[0]))
        elif c.shape[0] > crop_len:
            c = c[:crop_len]
        fixed.append(c)
    return fixed


def _batch_from_crops(crops: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    padded = torch.stack(crops)
    lengths = torch.ones(padded.shape[0], dtype=torch.float32)
    return padded, lengths
