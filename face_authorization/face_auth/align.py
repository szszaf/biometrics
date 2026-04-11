"""Wyrównanie twarzy pod ArcFace za pomocą MediaPipe Face Landmarker (Tasks API)."""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from skimage.transform import SimilarityTransform

# Szablon 5 punktów w przestrzeni docelowej 112×112 (jak w InsightFace / typowy pipeline ArcFace)
_ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# Indeksy topologii Face Mesh (zgodne z modelem Face Landmarker; pierwsze ~468 jak w klasycznym mesh)
_PERSON_LEFT_EYE = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
]
_PERSON_RIGHT_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
]
_NOSE_TIP = 4
_PERSON_MOUTH_LEFT = 291
_PERSON_MOUTH_RIGHT = 61


def _mean_lm_xy(
    landmarks: list,
    indices: list[int],
    w: float,
    h: float,
) -> np.ndarray:
    pts = []
    for i in indices:
        lm = landmarks[i]
        pts.append([lm.x * w, lm.y * h])
    return np.mean(pts, axis=0).astype(np.float32)


def _single_lm_xy(landmarks: list, idx: int, w: float, h: float) -> np.ndarray:
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def landmarks_to_arcface_src(
    landmarks: list,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Zwraca 5×2 punkty źródłowe: lewe oko, prawe oko, nos, kącik ust (lewy), kącik ust (prawy) — perspektywa osoby."""
    w, h = float(image_width), float(image_height)
    left_eye = _mean_lm_xy(landmarks, _PERSON_LEFT_EYE, w, h)
    right_eye = _mean_lm_xy(landmarks, _PERSON_RIGHT_EYE, w, h)
    nose = _single_lm_xy(landmarks, _NOSE_TIP, w, h)
    mouth_l = _single_lm_xy(landmarks, _PERSON_MOUTH_LEFT, w, h)
    mouth_r = _single_lm_xy(landmarks, _PERSON_MOUTH_RIGHT, w, h)
    return np.stack([left_eye, right_eye, nose, mouth_l, mouth_r], axis=0)


def warp_similarity_to_arcface(rgb: np.ndarray, src_five: np.ndarray) -> np.ndarray | None:
    """RGB uint8 (H,W,3) -> wycinek 112×112 RGB — ta sama logika co ``align_face`` w ``image_crop.ipynb``.

    ``SimilarityTransform().estimate`` + ``cv2.warpAffine(..., borderValue=0.0)``.
    Przy ``estimate`` zwracającym ``False`` (skimage) zwracamy ``None``.
    """
    tform = SimilarityTransform()
    ok = tform.estimate(src_five, _ARCFACE_DST)
    if ok is False:
        return None
    m = tform.params[0:2, :]
    return cv2.warpAffine(rgb, m, (112, 112), borderValue=0.0)


class FaceLandmarkerAligner:
    """Otwiera model `.task`, wykrywa twarz i zwraca wyrównany obraz 112×112."""

    def __init__(self, model_path: Path):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Brak modelu Face Landmarker: {model_path}")

        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

        base_options = BaseOptions(model_asset_path=str(model_path))
        options = FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._lock = threading.Lock()

    def close(self) -> None:
        self._landmarker.close()

    def align_rgb_to_pil(self, rgb: np.ndarray) -> Image.Image | None:
        """rgb: uint8, shape (H,W,3), RGB. Zwraca PIL RGB 112×112 lub None."""
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Oczekiwano obrazu RGB (H,W,3) uint8")
        h, w = rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with self._lock:
            result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        lms = result.face_landmarks[0]
        max_idx = max(
            *_PERSON_LEFT_EYE,
            *_PERSON_RIGHT_EYE,
            _NOSE_TIP,
            _PERSON_MOUTH_LEFT,
            _PERSON_MOUTH_RIGHT,
        )
        if len(lms) <= max_idx:
            return None
        src = landmarks_to_arcface_src(lms, w, h)
        warped = warp_similarity_to_arcface(rgb, src)
        if warped is None:
            return None
        return Image.fromarray(warped, mode="RGB")

    def align_pil(self, image: Image.Image) -> Image.Image | None:
        rgb = np.asarray(image.convert("RGB"))
        return self.align_rgb_to_pil(rgb)
