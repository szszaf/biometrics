import pytest
import torch

from voice_auth.audio_preprocess import (
    add_noise_by_amplitude_torch,
    build_enrollment_tta_batch_from_wav,
)
from voice_auth.config import CROP_SECONDS, ENROLLMENT_NOISE_RATIOS, NUM_TTA_CROPS, TARGET_SAMPLE_RATE


def test_add_noise_by_amplitude_torch_scales_noise_by_signal_rms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal = torch.tensor([3.0, 4.0])
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.ones_like(value))

    noisy = add_noise_by_amplitude_torch(signal, 0.5)

    signal_rms = torch.sqrt(torch.mean(signal**2))
    assert torch.allclose(noisy, signal + 0.5 * signal_rms)


def test_enrollment_batch_adds_three_noisy_versions_per_crop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crop_len = int(CROP_SECONDS * TARGET_SAMPLE_RATE)
    wav = torch.ones(crop_len)
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.ones_like(value))

    batch, lengths = build_enrollment_tta_batch_from_wav(wav)

    variants_per_crop = 1 + len(ENROLLMENT_NOISE_RATIOS)
    assert batch.shape == (NUM_TTA_CROPS * variants_per_crop, crop_len)
    assert lengths.shape == (NUM_TTA_CROPS * variants_per_crop,)
    for crop_idx in range(NUM_TTA_CROPS):
        start = crop_idx * variants_per_crop
        assert torch.allclose(batch[start], torch.ones(crop_len))
        assert torch.allclose(batch[start + 1], torch.full((crop_len,), 1.5))
        assert torch.allclose(batch[start + 2], torch.full((crop_len,), 1.75))
        assert torch.allclose(batch[start + 3], torch.full((crop_len,), 2.0))
