import io
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

from face_auth.config import EMBEDDING_DIM, INPUT_SIZE

if TYPE_CHECKING:
    from face_auth.align import FaceLandmarkerAligner


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
