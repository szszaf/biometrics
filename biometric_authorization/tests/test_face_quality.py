from PIL import Image

from face_auth.config import INPUT_SIZE
from face_auth.low_quality import make_low_quality_variants
from face_auth.quality import assess_face_image_quality


def test_assess_face_image_quality_returns_clean_when_image_has_texture() -> None:
    image = _textured_image(220, 220)

    report = assess_face_image_quality(image, image.resize((INPUT_SIZE, INPUT_SIZE)))

    assert report.estimated_quality == "clean"
    assert report.face_aligned is True
    assert report.warnings == ()


def test_assess_face_image_quality_rejects_too_small_image() -> None:
    image = _textured_image(24, 24)

    report = assess_face_image_quality(image, image)

    assert report.estimated_quality == "reject"
    assert "too_small" in report.warnings


def test_make_low_quality_variants_returns_arcface_sized_images() -> None:
    image = _textured_image(48, 48)

    variants = make_low_quality_variants(image)

    assert len(variants) >= 3
    assert all(variant.size == (INPUT_SIZE, INPUT_SIZE) for variant in variants)


def _textured_image(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            value = (x * 7 + y * 11) % 256
            pixels[x, y] = (value, 255 - value, (x * 13 + y * 3) % 256)
    return image
