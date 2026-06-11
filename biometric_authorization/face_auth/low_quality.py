from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFilter

from face_auth.config import INPUT_SIZE, LOW_QUALITY_UPSCALE_SIZE


def enhance_low_quality_face(pil_image: Image.Image) -> Image.Image:
    image = _resize_for_enhancement(pil_image.convert("RGB"))
    image = _apply_clahe(image)
    image = _denoise(image)
    return image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=85, threshold=3))


def make_low_quality_variants(pil_image: Image.Image) -> list[Image.Image]:
    base = pil_image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.LANCZOS)
    clahe = _apply_clahe(base)
    denoised = _denoise(base)
    sharpened = base.filter(ImageFilter.UnsharpMask(radius=1.0, percent=110, threshold=3))
    combined = _apply_clahe(denoised).filter(
        ImageFilter.UnsharpMask(radius=1.0, percent=85, threshold=3)
    )
    return [base, clahe, denoised, sharpened, combined]


def _resize_for_enhancement(pil_image: Image.Image) -> Image.Image:
    width, height = pil_image.size
    max_side = max(width, height)
    if max_side >= LOW_QUALITY_UPSCALE_SIZE:
        return pil_image
    scale = LOW_QUALITY_UPSCALE_SIZE / max(1, max_side)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return pil_image.resize(new_size, Image.Resampling.LANCZOS)


def _apply_clahe(pil_image: Image.Image) -> Image.Image:
    rgb = np.asarray(pil_image.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_rgb, mode="RGB")


def _denoise(pil_image: Image.Image) -> Image.Image:
    rgb = np.asarray(pil_image.convert("RGB"))
    denoised = cv2.fastNlMeansDenoisingColored(rgb, None, 3, 3, 7, 21)
    return Image.fromarray(denoised, mode="RGB")
