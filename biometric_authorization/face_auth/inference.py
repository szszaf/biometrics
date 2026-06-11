import io
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

from face_auth.config import EMBEDDING_DIM, INPUT_SIZE
from face_auth.low_quality import enhance_low_quality_face, make_low_quality_variants
from face_auth.quality import FaceQualityReport, assess_face_image_quality

if TYPE_CHECKING:
    from face_auth.align import FaceLandmarkerAligner

PreprocessingMode = Literal["standard", "low_quality_robust"]


@dataclass(frozen=True)
class QualityAwareFaceEmbedding:
    embedding: torch.Tensor
    quality: FaceQualityReport
    preprocessing_mode: PreprocessingMode


class FaceQualityRejectedError(Exception):
    def __init__(self, quality: FaceQualityReport):
        self.quality = quality

    def __str__(self) -> str:
        reasons = ", ".join(self.quality.warnings) if self.quality.warnings else "unknown"
        return f"Probka twarzy ma zbyt niska jakosc do bezpiecznego uwierzytelnienia ({reasons})"


def default_preprocess():
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def celeba_cropped_notebook_preprocess():
    """Jak ``image_crop`` / notebook: ``ToTensor`` + ``Normalize`` — bez ``Resize`` (wejście 112×112)."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


@torch.no_grad()
def embedding_from_pil(
    model,
    device,
    pil_image: Image.Image,
    transform=None,
    face_aligner: "FaceLandmarkerAligner | None" = None,
):
    pil_image = ImageOps.exif_transpose(pil_image)
    if face_aligner is not None:
        aligned = face_aligner.align_pil(pil_image)
        if aligned is None:
            raise ValueError("Nie wykryto twarzy lub nie udało się wyrównać obrazu (MediaPipe)")
        pil_image = aligned
        if transform is None:
            transform = celeba_cropped_notebook_preprocess()
    elif transform is None:
        transform = default_preprocess()
    tensor = transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
    emb = model(tensor)
    emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().squeeze(0)


@torch.no_grad()
def quality_aware_embedding_from_pil(
    model,
    device,
    pil_image: Image.Image,
    transform=None,
    face_aligner: "FaceLandmarkerAligner | None" = None,
) -> QualityAwareFaceEmbedding:
    pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
    aligned = None
    if face_aligner is not None:
        aligned = face_aligner.align_pil(pil_image)
        if aligned is None:
            enhanced = enhance_low_quality_face(pil_image)
            aligned = face_aligner.align_pil(enhanced)

    quality = assess_face_image_quality(pil_image, aligned)
    if quality.estimated_quality == "reject":
        raise FaceQualityRejectedError(quality)

    if aligned is None:
        embedding = embedding_from_pil(
            model,
            device,
            pil_image,
            transform=transform,
            face_aligner=None,
        )
        return QualityAwareFaceEmbedding(
            embedding=embedding,
            quality=quality,
            preprocessing_mode="standard",
        )

    if quality.estimated_quality == "clean":
        embedding = embedding_from_pil(
            model,
            device,
            aligned,
            transform=transform or celeba_cropped_notebook_preprocess(),
            face_aligner=None,
        )
        return QualityAwareFaceEmbedding(
            embedding=embedding,
            quality=quality,
            preprocessing_mode="standard",
        )

    variants = make_low_quality_variants(aligned)
    parts = [
        embedding_from_pil(
            model,
            device,
            variant,
            transform=transform or celeba_cropped_notebook_preprocess(),
            face_aligner=None,
        )
        for variant in variants
    ]
    stacked = torch.stack(parts)
    embedding = F.normalize(stacked.mean(dim=0), dim=0, eps=1e-12)
    return QualityAwareFaceEmbedding(
        embedding=embedding,
        quality=quality,
        preprocessing_mode="low_quality_robust",
    )


@torch.no_grad()
def embedding_from_bytes(
    model,
    device,
    data: bytes,
    transform=None,
    face_aligner: "FaceLandmarkerAligner | None" = None,
):
    pil = Image.open(io.BytesIO(data))
    return embedding_from_pil(
        model, device, pil, transform=transform, face_aligner=face_aligner
    )


@torch.no_grad()
def quality_aware_embedding_from_bytes(
    model,
    device,
    data: bytes,
    transform=None,
    face_aligner: "FaceLandmarkerAligner | None" = None,
) -> QualityAwareFaceEmbedding:
    pil = Image.open(io.BytesIO(data))
    return quality_aware_embedding_from_pil(
        model, device, pil, transform=transform, face_aligner=face_aligner
    )


@torch.no_grad()
def embedding_from_path(
    model,
    device,
    path: Path | str,
    transform=None,
    face_aligner: "FaceLandmarkerAligner | None" = None,
):
    pil = Image.open(path)
    return embedding_from_pil(
        model, device, pil, transform=transform, face_aligner=face_aligner
    )


def cosine_similarity(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
    return F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()


def embedding_to_numpy(emb: torch.Tensor) -> np.ndarray:
    return emb.numpy().astype(np.float32)


def numpy_to_embedding(arr: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(arr.astype(np.float32))
    return F.normalize(t, p=2, dim=0)


@torch.no_grad()
def average_embedding_from_bytes_list(
    model,
    device,
    blobs: list[bytes],
    face_aligner: "FaceLandmarkerAligner | None" = None,
):
    """Średnia embeddingów z wielu zdjęć (po normalizacji L2 każdej klatki), ponowna normalizacja L2."""
    if not blobs:
        raise ValueError("Potrzebna co najmniej jedna klatka")
    parts = [
        embedding_from_bytes(model, device, data, face_aligner=face_aligner) for data in blobs
    ]
    stacked = torch.stack(parts)
    return F.normalize(stacked.mean(dim=0), dim=0, eps=1e-12)
