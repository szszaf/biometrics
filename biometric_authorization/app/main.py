import io
import json
import logging
import os
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from face_auth.align import FaceLandmarkerAligner
from face_auth.config import DEFAULT_FACE_LANDMARKER_MODEL, DEFAULT_WEIGHTS, ENROLL_DB_PATH
from face_auth.inference import (
    FaceQualityRejectedError,
    average_embedding_from_bytes_list,
    cosine_similarity,
    embedding_from_bytes,
    embedding_to_numpy,
    numpy_to_embedding,
    quality_aware_embedding_from_bytes,
)
from face_auth.low_quality import enhance_low_quality_face
from face_auth.model import load_backbone
from face_auth.quality import FaceQualityReport, assess_face_image_quality
from face_auth.store import EnrollmentStore
from voice_auth.config import (
    DEFAULT_SPEECHBRAIN_ECAPA_SAVEDIR,
    DEFAULT_VOICE_WEIGHTS,
    ENROLL_DB_VOICE_PATH,
    VOICE_EMBEDDING_DIM,
)
from voice_auth.engine import VoiceEmbeddingEngine

logger = logging.getLogger(__name__)

Modality = Literal["face", "voice"]


class FaceQualityResponse(BaseModel):
    width: int
    height: int
    face_aligned: bool
    blur_score: float
    brightness_mean: float
    contrast_std: float
    estimated_quality: str
    warnings: list[str] = Field(default_factory=list)


def _env_flag(name: str, *, default: bool = True) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return default


class VerifyResponse(BaseModel):
    accepted: bool
    similarity: float
    threshold: float
    user_id: str
    modality: str
    quality: FaceQualityResponse | None = None
    preprocessing_mode: str | None = None
    quality_warnings: list[str] = Field(default_factory=list)


class IdentifyHit(BaseModel):
    user_id: str
    similarity: float


class IdentifyResponse(BaseModel):
    results: list[IdentifyHit]
    modality: str
    quality: FaceQualityResponse | None = None
    preprocessing_mode: str | None = None
    quality_warnings: list[str] = Field(default_factory=list)


class UserSummary(BaseModel):
    user_id: str
    sample_count: int = 1
    enrolled_at: str | None = None
    modality: str = "face"


class CapabilitiesResponse(BaseModel):
    modalities: list[str] = Field(description="Aktywne metody (co najmniej jedna)")
    face: dict
    voice: dict


class AdminSummaryResponse(BaseModel):
    enrolled_users_face: int
    enrolled_users_voice: int
    device: str
    face_weights: str | None = None
    face_landmarker: str | None = None
    voice_weights: str | None = None
    speechbrain_ecapa_dir: str | None = None
    note: str


class HealthResponse(BaseModel):
    status: str
    modalities: list[str]
    device: str
    enrolled_users_face: int
    enrolled_users_voice: int
    face_weights: str | None = None
    face_landmarker: str | None = None
    voice_weights: str | None = None


class ExperimentStatusResponse(BaseModel):
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    output_path: str
    returncode: int | None = None
    message: str | None = None
    progress: dict | None = None


def _weights_path() -> Path:
    p = os.environ.get("ARCFACE_WEIGHTS", "").strip()
    return Path(p) if p else DEFAULT_WEIGHTS


def _db_path_face() -> Path:
    p = os.environ.get("ENROLL_DB_PATH", "").strip()
    return Path(p) if p else ENROLL_DB_PATH


def _db_path_voice() -> Path:
    p = os.environ.get("ENROLL_DB_VOICE_PATH", "").strip()
    return Path(p) if p else ENROLL_DB_VOICE_PATH


def _face_landmarker_path() -> Path:
    p = os.environ.get("MEDIAPIPE_FACE_MODEL", "").strip()
    return Path(p) if p else DEFAULT_FACE_LANDMARKER_MODEL


def _voice_weights_path() -> Path:
    p = os.environ.get("VOICE_WEIGHTS", "").strip()
    return Path(p) if p else DEFAULT_VOICE_WEIGHTS


def _speechbrain_ecapa_dir() -> Path:
    p = os.environ.get("SPEECHBRAIN_ECAPA_SAVEDIR", "").strip()
    return Path(p) if p else DEFAULT_SPEECHBRAIN_ECAPA_SAVEDIR


def _face_assets_available() -> bool:
    w, lm = _weights_path(), _face_landmarker_path()
    return w.is_file() and lm.is_file()


def _voice_weights_available() -> bool:
    return _voice_weights_path().is_file()


def _parse_modality(raw: str | None) -> Modality | None:
    if raw is None or raw.strip() == "":
        return None
    m = raw.strip().lower()
    if m not in ("face", "voice"):
        raise HTTPException(status_code=400, detail="modality musi być „face” lub „voice”")
    return m  # type: ignore[return-value]


def _resolve_modality(state, requested: str | None) -> Modality:
    m = _parse_modality(requested)
    if m == "face":
        if not state.face_enabled:
            raise HTTPException(status_code=503, detail="Modality twarzy jest wyłączona (brak plików modeli).")
        return "face"
    if m == "voice":
        if not state.voice_enabled:
            raise HTTPException(status_code=503, detail="Modality głosu jest wyłączona (brak wag lub SpeechBrain).")
        return "voice"
    if state.face_enabled:
        return "face"
    if state.voice_enabled:
        return "voice"
    raise RuntimeError("Brak aktywnej modality biometrycznej")


def _face_quality_response(report: FaceQualityReport) -> FaceQualityResponse:
    return FaceQualityResponse(
        width=report.width,
        height=report.height,
        face_aligned=report.face_aligned,
        blur_score=report.blur_score,
        brightness_mean=report.brightness_mean,
        contrast_std=report.contrast_std,
        estimated_quality=report.estimated_quality,
        warnings=list(report.warnings),
    )


def _face_quality_detail(report: FaceQualityReport) -> dict:
    return {
        "message": str(FaceQualityRejectedError(report)),
        "quality": _face_quality_response(report).model_dump(),
        "quality_warnings": list(report.warnings),
    }


def _assess_face_quality_from_bytes(state, data: bytes) -> FaceQualityResponse:
    pil_image = Image.open(io.BytesIO(data))
    pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
    aligned = state.face_aligner.align_pil(pil_image) if state.face_aligner else None
    if aligned is None and state.face_aligner is not None:
        enhanced = enhance_low_quality_face(pil_image)
        aligned = state.face_aligner.align_pil(enhanced)
    report = assess_face_image_quality(pil_image, aligned)
    return _face_quality_response(report)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _experiment_status_response(state) -> ExperimentStatusResponse:
    with state.low_res_experiment_lock:
        raw = dict(state.low_res_experiment)
    progress = None
    if LOW_RES_EXPERIMENT_OUTPUT.is_file():
        try:
            with open(LOW_RES_EXPERIMENT_OUTPUT, "r", encoding="utf-8") as handle:
                latest = json.load(handle)
            progress = latest.get("progress")
        except Exception:
            progress = None
    return ExperimentStatusResponse(
        status=str(raw.get("status") or "idle"),
        started_at=raw.get("started_at"),
        finished_at=raw.get("finished_at"),
        output_path=str(raw.get("output_path") or LOW_RES_EXPERIMENT_OUTPUT),
        returncode=raw.get("returncode"),
        message=raw.get("message"),
        progress=progress,
    )


def _read_low_res_experiment_result() -> dict:
    if not LOW_RES_EXPERIMENT_OUTPUT.is_file():
        raise HTTPException(status_code=404, detail="Brak zapisanego wyniku eksperymentu low-res.")
    with open(LOW_RES_EXPERIMENT_OUTPUT, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_low_res_experiment_job(state, threshold: float) -> None:
    app_dir = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(app_dir) if not existing_pythonpath else f"{app_dir}{os.pathsep}{existing_pythonpath}"
    )
    command = [
        sys.executable,
        str(LOW_RES_EXPERIMENT_SCRIPT),
        "--output",
        str(LOW_RES_EXPERIMENT_OUTPUT),
        "--threshold",
        str(threshold),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(app_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=None,
            check=False,
        )
    except Exception as e:
        with state.low_res_experiment_lock:
            state.low_res_experiment.update(
                {
                    "status": "failed",
                    "finished_at": _utc_now(),
                    "returncode": None,
                    "message": str(e),
                }
            )
        return

    message = completed.stderr.strip() or completed.stdout.strip() or None
    with state.low_res_experiment_lock:
        state.low_res_experiment.update(
            {
                "status": "done" if completed.returncode == 0 else "failed",
                "finished_at": _utc_now(),
                "returncode": completed.returncode,
                "message": message,
            }
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.device = device
    app.state.low_res_experiment_lock = threading.Lock()
    app.state.low_res_experiment = {
        "status": "idle",
        "started_at": None,
        "finished_at": None,
        "returncode": None,
        "message": None,
        "output_path": str(LOW_RES_EXPERIMENT_OUTPUT),
    }

    app.state.face_enabled = _face_assets_available()
    app.state.voice_enabled = False

    if app.state.face_enabled:
        weights = _weights_path()
        lm_path = _face_landmarker_path()
        model, dev = load_backbone(weights)
        app.state.model = model
        app.state.device = dev
        app.state.weights_path = weights
        app.state.face_landmarker_path = lm_path
        app.state.face_aligner = FaceLandmarkerAligner(lm_path)
        app.state.face_store = EnrollmentStore(_db_path_face(), embedding_dim=512)
        logger.info("Twarz: włączona (%s)", weights)
    else:
        app.state.model = None
        app.state.weights_path = None
        app.state.face_landmarker_path = None
        app.state.face_aligner = None
        app.state.face_store = None
        logger.warning(
            "Twarz: wyłączona — brak %s lub %s",
            _weights_path(),
            _face_landmarker_path(),
        )

    app.state.voice_engine = None
    app.state.voice_store = None
    if _voice_weights_available():
        try:
            engine = VoiceEmbeddingEngine(
                weights_path=_voice_weights_path(),
                speechbrain_savedir=_speechbrain_ecapa_dir(),
                device=app.state.device,
            )
            app.state.voice_engine = engine
            app.state.voice_store = EnrollmentStore(_db_path_voice(), embedding_dim=VOICE_EMBEDDING_DIM)
            app.state.voice_enabled = True
            logger.info("Głos: włączony (%s)", _voice_weights_path())
        except Exception:
            logger.exception("Głos: nie udało się załadować modelu — modality wyłączona")
            app.state.voice_engine = None
            app.state.voice_store = None
            app.state.voice_enabled = False
    else:
        logger.warning("Głos: wyłączony — brak pliku wag %s", _voice_weights_path())

    if not app.state.face_enabled and not app.state.voice_enabled:
        raise RuntimeError(
            "Brak co najmniej jednej modality: dodaj wagi ArcFace + face_landmarker.task (twarz) "
            "lub ecapa_cv_pl_best.pth + cache SpeechBrain (głos). Zobacz README."
        )

    if app.state.face_enabled and _env_flag("SEED_AUTO", default=True):
        try:
            from face_auth.seed import run_auto_seed

            run_auto_seed(
                app.state.face_store,
                app.state.model,
                app.state.device,
                app.state.face_aligner,
            )
        except Exception:
            logger.exception("Automatyczny seed profili CelebA nie powiódł się")

    if app.state.voice_enabled and _env_flag("SEED_AUTO", default=True):
        try:
            from voice_auth.seed import run_voice_auto_seed

            run_voice_auto_seed(app.state.voice_store, app.state.voice_engine)
        except Exception:
            logger.exception("Automatyczny seed profili głosu (Common Voice) nie powiódł się")

    yield

    if app.state.face_aligner is not None:
        app.state.face_aligner.close()
    if app.state.face_store is not None:
        app.state.face_store.close()
    if app.state.voice_store is not None:
        app.state.voice_store.close()
    if app.state.voice_engine is not None:
        app.state.voice_engine.close()


app = FastAPI(
    title="Biometric authorization API",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "health", "description": "Stan usługi i dostępne modality"},
        {"name": "users", "description": "Rejestracja (twarz / głos)"},
        {"name": "authentication", "description": "Weryfikacja, identyfikacja, porównanie 1:1"},
        {"name": "admin", "description": "Podsumowanie operacyjne"},
    ],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
LOW_RES_EXPERIMENT_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "evaluate_low_res_face.py"
LOW_RES_EXPERIMENT_OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "system"
    / "experiments"
    / "low_res_face_latest.json"
)


@app.get("/")
def serve_gui():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/capabilities", response_model=CapabilitiesResponse, tags=["health"])
def capabilities(request: Request):
    st = request.app.state
    modalities: list[str] = []
    if st.face_enabled:
        modalities.append("face")
    if st.voice_enabled:
        modalities.append("voice")
    return CapabilitiesResponse(
        modalities=modalities,
        face={
            "enabled": st.face_enabled,
            "weights": str(st.weights_path) if st.face_enabled else None,
            "landmarker": str(st.face_landmarker_path) if st.face_enabled else None,
        },
        voice={
            "enabled": st.voice_enabled,
            "weights": str(_voice_weights_path()) if st.voice_enabled else None,
            "speechbrain_savedir": str(_speechbrain_ecapa_dir()) if st.voice_enabled else None,
        },
    )


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health(request: Request):
    st = request.app.state
    modalities: list[str] = []
    if st.face_enabled:
        modalities.append("face")
    if st.voice_enabled:
        modalities.append("voice")
    n_face = len(st.face_store.list_user_ids()) if st.face_store else 0
    n_voice = len(st.voice_store.list_user_ids()) if st.voice_store else 0
    return HealthResponse(
        status="ok",
        modalities=modalities,
        device=str(st.device),
        enrolled_users_face=n_face,
        enrolled_users_voice=n_voice,
        face_weights=str(st.weights_path) if st.face_enabled else None,
        face_landmarker=str(st.face_landmarker_path) if st.face_enabled else None,
        voice_weights=str(_voice_weights_path()) if st.voice_enabled else None,
    )


@app.post("/face/quality", response_model=FaceQualityResponse, tags=["authentication"])
async def face_quality(
    request: Request,
    image: Annotated[UploadFile | None, File()] = None,
):
    st = request.app.state
    if not st.face_enabled:
        raise HTTPException(status_code=503, detail="Ocena jakości twarzy wymaga wag ArcFace i MediaPipe.")
    if image is None:
        raise HTTPException(status_code=400, detail="Prześlij pole „image”.")
    data = await image.read()
    try:
        return _assess_face_quality_from_bytes(st, data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Nie udało się ocenić obrazu: {e}") from e


@app.get("/users", response_model=list[UserSummary], tags=["users"])
def list_users(request: Request, modality: str | None = Query(None, description="face (domyślnie jeśli włączone) lub voice")):
    st = request.app.state
    m = _resolve_modality(st, modality)
    store = st.face_store if m == "face" else st.voice_store
    assert store is not None
    return [
        UserSummary(user_id=u, sample_count=n, enrolled_at=ts, modality=m)
        for u, n, ts in store.list_users_info()
    ]


@app.post("/users/{user_id}/enroll", status_code=201, tags=["users"])
async def enroll(
    request: Request,
    user_id: str,
    modality: str | None = Query(None),
    image: Annotated[UploadFile | None, File()] = None,
    audio: Annotated[UploadFile | None, File()] = None,
):
    st = request.app.state
    m = _resolve_modality(st, modality)
    if not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id nie może być pusty")
    uid = user_id.strip()

    if m == "face":
        if image is None:
            raise HTTPException(status_code=400, detail="Dla twarzy prześlij pole „image”.")
        data = await image.read()
        try:
            emb = embedding_from_bytes(
                st.model, st.device, data, face_aligner=st.face_aligner
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Nie udało się wczytać obrazu: {e}") from e
        vec = embedding_to_numpy(emb)
        st.face_store.upsert(uid, vec, sample_count=1)
        return {"user_id": uid, "status": "enrolled", "sample_count": 1, "modality": "face"}

    if audio is None:
        raise HTTPException(status_code=400, detail="Dla głosu prześlij pole „audio”.")
    raw = await audio.read()
    try:
        vec = st.voice_engine.embed_enrollment_from_bytes(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Nie udało się przetworzyć nagrania: {e}") from e
    st.voice_store.upsert(uid, vec, sample_count=1)
    return {"user_id": uid, "status": "enrolled", "sample_count": 1, "modality": "voice"}


@app.post("/users/{user_id}/enroll_multi", status_code=201, tags=["users"])
async def enroll_multi(
    request: Request,
    user_id: str,
    modality: str | None = Query(None),
    images: Annotated[list[UploadFile] | None, File()] = None,
    audios: Annotated[list[UploadFile] | None, File()] = None,
):
    st = request.app.state
    m = _resolve_modality(st, modality)
    uid = user_id.strip()
    if not uid:
        raise HTTPException(status_code=400, detail="user_id nie może być pusty")

    if m == "face":
        img_list = images or []
        if len(img_list) < 3:
            raise HTTPException(status_code=400, detail="Minimum 3 klatki z kamery")
        if len(img_list) > 12:
            raise HTTPException(status_code=400, detail="Maksimum 12 klatek")
        blobs: list[bytes] = [await img.read() for img in img_list]
        try:
            emb = average_embedding_from_bytes_list(
                st.model,
                st.device,
                blobs,
                face_aligner=st.face_aligner,
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Przetwarzanie klatek nie powiodło się: {e}") from e
        vec = embedding_to_numpy(emb)
        st.face_store.upsert(uid, vec, sample_count=len(blobs))
        return {"user_id": uid, "status": "enrolled", "sample_count": len(blobs), "modality": "face"}

    audio_list = audios or []
    if len(audio_list) < 3:
        raise HTTPException(status_code=400, detail="Minimum 3 próbek audio")
    if len(audio_list) > 12:
        raise HTTPException(status_code=400, detail="Maksimum 12 próbek audio")
    parts: list[np.ndarray] = []
    for a in audio_list:
        raw = await a.read()
        try:
            parts.append(st.voice_engine.embed_enrollment_from_bytes(raw))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Przetwarzanie audio nie powiodło się: {e}") from e
    stacked = torch.from_numpy(np.stack(parts, axis=0).astype(np.float32))
    mean_emb = F.normalize(stacked.mean(dim=0), dim=0, eps=1e-12)
    vec = mean_emb.numpy().astype(np.float32)
    st.voice_store.upsert(uid, vec, sample_count=len(audio_list))
    return {"user_id": uid, "status": "enrolled", "sample_count": len(audio_list), "modality": "voice"}


@app.get("/admin/summary", response_model=AdminSummaryResponse, tags=["admin"])
def admin_summary(request: Request):
    st = request.app.state
    n_face = len(st.face_store.list_user_ids()) if st.face_store else 0
    n_voice = len(st.voice_store.list_user_ids()) if st.voice_store else 0
    return AdminSummaryResponse(
        enrolled_users_face=n_face,
        enrolled_users_voice=n_voice,
        device=str(st.device),
        face_weights=str(st.weights_path) if st.face_enabled else None,
        face_landmarker=str(st.face_landmarker_path) if st.face_enabled else None,
        voice_weights=str(_voice_weights_path()) if st.voice_enabled else None,
        speechbrain_ecapa_dir=str(_speechbrain_ecapa_dir()) if st.voice_enabled else None,
        note="W query parametru „modality” podaj face lub voice przy listach / enroll / verify / identify. "
        "FAR/FRR — raport z notebooków.",
    )


@app.post("/admin/experiments/low-res/run", response_model=ExperimentStatusResponse, tags=["admin"])
def run_low_res_experiment(
    request: Request,
    threshold: float = Query(0.16, ge=-1.0, le=1.0),
):
    st = request.app.state
    if not st.face_enabled:
        raise HTTPException(status_code=503, detail="Eksperyment low-res wymaga włączonej modality twarzy.")
    if not LOW_RES_EXPERIMENT_SCRIPT.is_file():
        raise HTTPException(status_code=500, detail="Brak skryptu eksperymentu low-res.")
    with st.low_res_experiment_lock:
        if st.low_res_experiment.get("status") == "running":
            should_start = False
        else:
            st.low_res_experiment = {
                "status": "running",
                "started_at": _utc_now(),
                "finished_at": None,
                "returncode": None,
                "message": "Eksperyment low-res został uruchomiony.",
                "output_path": str(LOW_RES_EXPERIMENT_OUTPUT),
            }
            should_start = True
    if not should_start:
        return _experiment_status_response(st)
    worker = threading.Thread(
        target=_run_low_res_experiment_job,
        args=(st, threshold),
        daemon=True,
    )
    worker.start()
    return _experiment_status_response(st)


@app.get("/admin/experiments/low-res/status", response_model=ExperimentStatusResponse, tags=["admin"])
def low_res_experiment_status(request: Request):
    return _experiment_status_response(request.app.state)


@app.get("/admin/experiments/low-res/latest", response_model=dict, tags=["admin"])
def latest_low_res_experiment():
    return _read_low_res_experiment_result()


@app.delete("/users/{user_id}", status_code=204, response_class=Response, tags=["users"])
def remove_user(request: Request, user_id: str, modality: str | None = Query(None)):
    st = request.app.state
    m = _resolve_modality(st, modality)
    store = st.face_store if m == "face" else st.voice_store
    assert store is not None
    if not store.delete(user_id):
        raise HTTPException(status_code=404, detail="Nie znaleziono użytkownika")
    return Response(status_code=204)


@app.post("/verify", response_model=VerifyResponse, tags=["authentication"])
async def verify(
    request: Request,
    user_id: str = Form(...),
    threshold: float = Query(0.16, ge=-1.0, le=1.0),
    modality: str | None = Query(None),
    image: Annotated[UploadFile | None, File()] = None,
    audio: Annotated[UploadFile | None, File()] = None,
):
    st = request.app.state
    m = _resolve_modality(st, modality)
    if m == "face":
        if image is None:
            raise HTTPException(status_code=400, detail="Dla twarzy prześlij pole „image”.")
        stored = st.face_store.get(user_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="Użytkownik nie jest zarejestrowany (twarz)")
        data = await image.read()
        try:
            face_result = quality_aware_embedding_from_bytes(
                st.model, st.device, data, face_aligner=st.face_aligner
            )
        except FaceQualityRejectedError as e:
            raise HTTPException(status_code=400, detail=_face_quality_detail(e.quality)) from e
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Nie udało się wczytać obrazu: {e}") from e
        probe = face_result.embedding
        ref = numpy_to_embedding(stored)
        sim = cosine_similarity(probe, ref)
        return VerifyResponse(
            accepted=sim >= threshold,
            similarity=sim,
            threshold=threshold,
            user_id=user_id,
            modality="face",
            quality=_face_quality_response(face_result.quality),
            preprocessing_mode=face_result.preprocessing_mode,
            quality_warnings=list(face_result.quality.warnings),
        )

    if audio is None:
        raise HTTPException(status_code=400, detail="Dla głosu prześlij pole „audio”.")
    stored = st.voice_store.get(user_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Użytkownik nie jest zarejestrowany (głos)")
    raw = await audio.read()
    try:
        probe_np = st.voice_engine.embed_from_bytes(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Nie udało się przetworzyć nagrania: {e}") from e
    probe = numpy_to_embedding(probe_np)
    ref = numpy_to_embedding(stored)
    sim = cosine_similarity(probe, ref)
    return VerifyResponse(
        accepted=sim >= threshold,
        similarity=sim,
        threshold=threshold,
        user_id=user_id,
        modality="voice",
    )


@app.post("/identify", response_model=IdentifyResponse, tags=["authentication"])
async def identify(
    request: Request,
    top_k: int = Query(5, ge=1, le=500),
    modality: str | None = Query(None),
    image: Annotated[UploadFile | None, File()] = None,
    audio: Annotated[UploadFile | None, File()] = None,
):
    st = request.app.state
    m = _resolve_modality(st, modality)
    if m == "face":
        if image is None:
            raise HTTPException(status_code=400, detail="Dla twarzy prześlij pole „image”.")
        rows = st.face_store.all_embeddings()
        if not rows:
            raise HTTPException(status_code=404, detail="Brak zarejestrowanych użytkowników (twarz)")
        data = await image.read()
        try:
            face_result = quality_aware_embedding_from_bytes(
                st.model, st.device, data, face_aligner=st.face_aligner
            )
        except FaceQualityRejectedError as e:
            raise HTTPException(status_code=400, detail=_face_quality_detail(e.quality)) from e
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Nie udało się wczytać obrazu: {e}") from e
        probe = face_result.embedding
        scored: list[tuple[str, float]] = []
        for uid, arr in rows:
            ref = numpy_to_embedding(arr)
            scored.append((uid, cosine_similarity(probe, ref)))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        return IdentifyResponse(
            results=[IdentifyHit(user_id=u, similarity=s) for u, s in top],
            modality="face",
            quality=_face_quality_response(face_result.quality),
            preprocessing_mode=face_result.preprocessing_mode,
            quality_warnings=list(face_result.quality.warnings),
        )

    if audio is None:
        raise HTTPException(status_code=400, detail="Dla głosu prześlij pole „audio”.")
    rows = st.voice_store.all_embeddings()
    if not rows:
        raise HTTPException(status_code=404, detail="Brak zarejestrowanych użytkowników (głos)")
    raw = await audio.read()
    try:
        probe_np = st.voice_engine.embed_from_bytes(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Nie udało się przetworzyć nagrania: {e}") from e
    probe = numpy_to_embedding(probe_np)
    scored = [(uid, cosine_similarity(probe, numpy_to_embedding(arr))) for uid, arr in rows]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]
    return IdentifyResponse(
        results=[IdentifyHit(user_id=u, similarity=s) for u, s in top],
        modality="voice",
    )


@app.post("/compare", response_model=dict, tags=["authentication"])
async def compare(
    request: Request,
    threshold: float = Query(0.16, ge=-1.0, le=1.0),
    image_a: Annotated[UploadFile | None, File()] = None,
    image_b: Annotated[UploadFile | None, File()] = None,
):
    """Weryfikacja 1:1 twarzą (dwa obrazy) — wymaga włączonej modality twarzy."""
    st = request.app.state
    if not st.face_enabled:
        raise HTTPException(status_code=503, detail="Porównanie obrazów wymaga wag ArcFace i MediaPipe.")
    if image_a is None or image_b is None:
        raise HTTPException(status_code=400, detail="Prześlij image_a i image_b.")
    try:
        da = await image_a.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Nie udało się wczytać pierwszego obrazu: {e}") from e
    try:
        db = await image_b.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Nie udało się wczytać drugiego obrazu: {e}") from e
    try:
        r1 = quality_aware_embedding_from_bytes(
            st.model, st.device, da, face_aligner=st.face_aligner
        )
    except FaceQualityRejectedError as e:
        raise HTTPException(status_code=400, detail=_face_quality_detail(e.quality)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Nie udało się wczytać pierwszego obrazu: {e}") from e
    try:
        r2 = quality_aware_embedding_from_bytes(
            st.model, st.device, db, face_aligner=st.face_aligner
        )
    except FaceQualityRejectedError as e:
        raise HTTPException(status_code=400, detail=_face_quality_detail(e.quality)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Nie udało się wczytać drugiego obrazu: {e}") from e
    sim = cosine_similarity(r1.embedding, r2.embedding)
    return {
        "same_person_guess": sim >= threshold,
        "similarity": sim,
        "threshold": threshold,
        "modality": "face",
        "quality_a": _face_quality_response(r1.quality).model_dump(),
        "quality_b": _face_quality_response(r2.quality).model_dump(),
        "preprocessing_mode_a": r1.preprocessing_mode,
        "preprocessing_mode_b": r2.preprocessing_mode,
        "quality_warnings": sorted(set(r1.quality.warnings + r2.quality.warnings)),
    }


@app.post("/compare_voice", response_model=dict, tags=["authentication"])
async def compare_voice(
    request: Request,
    threshold: float = Query(0.35, ge=-1.0, le=1.0),
    audio_a: UploadFile = File(...),
    audio_b: UploadFile = File(...),
):
    """Weryfikacja 1:1 głosem — wymaga włączonej modality głosu."""
    st = request.app.state
    if not st.voice_enabled:
        raise HTTPException(status_code=503, detail="Porównanie głosu wymaga wag ECAPA.")
    try:
        a1, a2 = await audio_a.read(), await audio_b.read()
        e1 = numpy_to_embedding(st.voice_engine.embed_from_bytes(a1))
        e2 = numpy_to_embedding(st.voice_engine.embed_from_bytes(a2))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Nie udało się przetworzyć audio: {e}") from e
    sim = cosine_similarity(e1, e2)
    return {"same_speaker_guess": sim >= threshold, "similarity": sim, "threshold": threshold, "modality": "voice"}
