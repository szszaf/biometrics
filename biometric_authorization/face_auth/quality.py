from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from PIL import Image

from face_auth.config import (
    FACE_LOW_BLUR_SCORE,
    FACE_LOW_CONTRAST,
    FACE_LOW_SOURCE_SIZE,
    FACE_MAX_BRIGHTNESS,
    FACE_MIN_BLUR_SCORE,
    FACE_MIN_BRIGHTNESS,
    FACE_MIN_CONTRAST,
    FACE_MIN_SOURCE_SIZE,
)

FaceQualityLevel = Literal["clean", "low_quality", "reject"]


@dataclass(frozen=True)
class FaceQualityReport:
    width: int
    height: int
    face_aligned: bool
    blur_score: float
    brightness_mean: float
    contrast_std: float
    estimated_quality: FaceQualityLevel
    warnings: tuple[str, ...]


def assess_face_image_quality(
    pil_image: Image.Image,
    aligned_image: Image.Image | None,
) -> FaceQualityReport:
    width, height = pil_image.size
    target_image = aligned_image if aligned_image is not None else pil_image
    gray = _to_gray_array(target_image)
    blur_score = _laplacian_variance(gray)
    brightness_mean = float(gray.mean())
    contrast_std = float(gray.std())
    warnings = _quality_warnings(
        width=width,
        height=height,
        face_aligned=aligned_image is not None,
        blur_score=blur_score,
        brightness_mean=brightness_mean,
        contrast_std=contrast_std,
    )
    quality = _estimate_quality(warnings)
    return FaceQualityReport(
        width=width,
        height=height,
        face_aligned=aligned_image is not None,
        blur_score=blur_score,
        brightness_mean=brightness_mean,
        contrast_std=contrast_std,
        estimated_quality=quality,
        warnings=tuple(warnings),
    )


def _to_gray_array(pil_image: Image.Image) -> np.ndarray:
    rgb = np.asarray(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _quality_warnings(
    *,
    width: int,
    height: int,
    face_aligned: bool,
    blur_score: float,
    brightness_mean: float,
    contrast_std: float,
) -> list[str]:
    warnings: list[str] = []
    min_side = min(width, height)
    if not face_aligned:
        warnings.append("face_not_detected")
    if min_side < FACE_MIN_SOURCE_SIZE:
        warnings.append("too_small")
    elif min_side < FACE_LOW_SOURCE_SIZE:
        warnings.append("low_resolution")
    if blur_score < FACE_MIN_BLUR_SCORE:
        warnings.append("too_blurred")
    elif blur_score < FACE_LOW_BLUR_SCORE:
        warnings.append("blurred")
    if brightness_mean < FACE_MIN_BRIGHTNESS:
        warnings.append("too_dark")
    elif brightness_mean > FACE_MAX_BRIGHTNESS:
        warnings.append("too_bright")
    if contrast_std < FACE_MIN_CONTRAST:
        warnings.append("too_low_contrast")
    elif contrast_std < FACE_LOW_CONTRAST:
        warnings.append("low_contrast")
    return warnings


def _estimate_quality(warnings: list[str]) -> FaceQualityLevel:
    reject_warnings = {
        "face_not_detected",
        "too_small",
        "too_blurred",
        "too_dark",
        "too_bright",
        "too_low_contrast",
    }
    if any(warning in reject_warnings for warning in warnings):
        return "reject"
    if warnings:
        return "low_quality"
    return "clean"
