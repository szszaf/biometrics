import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from face_auth.align import FaceLandmarkerAligner
from face_auth.config import DEFAULT_FACE_LANDMARKER_MODEL, DEFAULT_WEIGHTS, ENROLL_DB_PATH
from face_auth.inference import (
    average_embedding_from_bytes_list,
    cosine_similarity,
    embedding_from_bytes,
    embedding_to_numpy,
    numpy_to_embedding,
)
from face_auth.model import load_backbone
from face_auth.store import EnrollmentStore

logger = logging.getLogger(__name__)


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


class IdentifyHit(BaseModel):
    user_id: str
    similarity: float


class IdentifyResponse(BaseModel):
    results: list[IdentifyHit]


class UserSummary(BaseModel):
    user_id: str
    sample_count: int = 1
    enrolled_at: str | None = None


class AdminSummaryResponse(BaseModel):
    enrolled_users: int
    weights: str
    device: str
    face_landmarker: str
    note: str


class HealthResponse(BaseModel):
    status: str
    weights: str
    device: str
    enrolled_users: int
    face_landmarker: str


def _weights_path() -> Path:
    p = os.environ.get("ARCFACE_WEIGHTS", "").strip()
    return Path(p) if p else DEFAULT_WEIGHTS


def _db_path() -> Path:
    p = os.environ.get("ENROLL_DB_PATH", "").strip()
    return Path(p) if p else ENROLL_DB_PATH


def _face_landmarker_path() -> Path:
    p = os.environ.get("MEDIAPIPE_FACE_MODEL", "").strip()
    return Path(p) if p else DEFAULT_FACE_LANDMARKER_MODEL


@asynccontextmanager
async def lifespan(app: FastAPI):
    weights = _weights_path()
    if not weights.is_file():
        raise RuntimeError(f"Brak pliku wag: {weights} (ustaw ARCFACE_WEIGHTS lub dodaj results/*.pth)")
    lm_path = _face_landmarker_path()
    if not lm_path.is_file():
        raise RuntimeError(
            f"Brak modelu MediaPipe Face Landmarker: {lm_path} "
            "(pobierz face_landmarker.task lub ustaw MEDIAPIPE_FACE_MODEL)"
        )
    model, device = load_backbone(weights)
    app.state.model = model
    app.state.device = device
    app.state.weights_path = weights
    app.state.face_landmarker_path = lm_path
    app.state.face_aligner = FaceLandmarkerAligner(lm_path)
    app.state.store = EnrollmentStore(_db_path())
    if _env_flag("SEED_AUTO", default=True):
        try:
            from face_auth.seed import run_auto_seed

            run_auto_seed(
                app.state.store,
                app.state.model,
                app.state.device,
                app.state.face_aligner,
            )
        except Exception:
            logger.exception("Automatyczny seed profili CelebA nie powiódł się")
    yield
    app.state.face_aligner.close()
    app.state.store.close()


app = FastAPI(
    title="Face authorization API",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "health", "description": "Liveness and model paths"},
        {"name": "users", "description": "Enrollment and user records"},
        {"name": "authentication", "description": "Verify, identify, 1:1 compare"},
        {"name": "admin", "description": "Operational summary"},
    ],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/")
def serve_gui():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    store: EnrollmentStore = app.state.store
    return HealthResponse(
        status="ok",
        weights=str(app.state.weights_path),
        device=str(app.state.device),
        enrolled_users=len(store.list_user_ids()),
        face_landmarker=str(app.state.face_landmarker_path),
    )


@app.get("/users", response_model=list[UserSummary], tags=["users"])
def list_users():
    store: EnrollmentStore = app.state.store
    return [
        UserSummary(user_id=u, sample_count=n, enrolled_at=ts)
        for u, n, ts in store.list_users_info()
    ]


@app.post("/users/{user_id}/enroll", status_code=201, tags=["users"])
async def enroll(user_id: str, image: UploadFile = File(...)):
    if not user_id.strip():
        raise HTTPException(400, "user_id nie może być pusty")
    data = await image.read()
    try:
        emb = embedding_from_bytes(
            app.state.model, app.state.device, data, face_aligner=app.state.face_aligner
        )
    except Exception as e:
        raise HTTPException(400, f"Nie udało się wczytać obrazu: {e}") from e
    vec = embedding_to_numpy(emb)
    app.state.store.upsert(user_id.strip(), vec, sample_count=1)
    return {"user_id": user_id.strip(), "status": "enrolled", "sample_count": 1}


@app.post("/users/{user_id}/enroll_multi", status_code=201, tags=["users"])
async def enroll_multi(
    user_id: str,
    images: list[UploadFile] = File(),
):
    uid = user_id.strip()
    if not uid:
        raise HTTPException(400, "user_id nie może być pusty")
    if len(images) < 3:
        raise HTTPException(400, "Minimum 3 klatki z kamery")
    if len(images) > 12:
        raise HTTPException(400, "Maksimum 12 klatek")
    blobs: list[bytes] = []
    for img in images:
        blobs.append(await img.read())
    try:
        emb = average_embedding_from_bytes_list(
            app.state.model,
            app.state.device,
            blobs,
            face_aligner=app.state.face_aligner,
        )
    except Exception as e:
        raise HTTPException(400, f"Przetwarzanie klatek nie powiodło się: {e}") from e
    vec = embedding_to_numpy(emb)
    app.state.store.upsert(uid, vec, sample_count=len(blobs))
    return {"user_id": uid, "status": "enrolled", "sample_count": len(blobs)}


@app.get("/admin/summary", response_model=AdminSummaryResponse, tags=["admin"])
def admin_summary():
    store: EnrollmentStore = app.state.store
    return AdminSummaryResponse(
        enrolled_users=len(store.list_user_ids()),
        weights=str(app.state.weights_path),
        device=str(app.state.device),
        face_landmarker=str(app.state.face_landmarker_path),
        note="FAR/FRR, ROC i eksperymenty szumu — eksport z notebooków lub osobny batch; UI pokazuje stan bazy i próg.",
    )


@app.delete("/users/{user_id}", status_code=204, response_class=Response, tags=["users"])
def remove_user(user_id: str):
    if not app.state.store.delete(user_id):
        raise HTTPException(404, "Nie znaleziono użytkownika")
    return Response(status_code=204)


@app.post("/verify", response_model=VerifyResponse, tags=["authentication"])
async def verify(
    user_id: str = Form(...),
    threshold: float = Query(0.45, ge=-1.0, le=1.0),
    image: UploadFile = File(...),
):
    stored = app.state.store.get(user_id)
    if stored is None:
        raise HTTPException(404, "Użytkownik nie jest zarejestrowany")
    data = await image.read()
    try:
        probe = embedding_from_bytes(
            app.state.model, app.state.device, data, face_aligner=app.state.face_aligner
        )
    except Exception as e:
        raise HTTPException(400, f"Nie udało się wczytać obrazu: {e}") from e
    ref = numpy_to_embedding(stored)
    sim = cosine_similarity(probe, ref)
    return VerifyResponse(
        accepted=sim >= threshold,
        similarity=sim,
        threshold=threshold,
        user_id=user_id,
    )


@app.post("/identify", response_model=IdentifyResponse, tags=["authentication"])
async def identify(
    top_k: int = Query(5, ge=1, le=500),
    image: UploadFile = File(...),
):
    rows = app.state.store.all_embeddings()
    if not rows:
        raise HTTPException(404, "Brak zarejestrowanych użytkowników")
    data = await image.read()
    try:
        probe = embedding_from_bytes(
            app.state.model, app.state.device, data, face_aligner=app.state.face_aligner
        )
    except Exception as e:
        raise HTTPException(400, f"Nie udało się wczytać obrazu: {e}") from e

    scored: list[tuple[str, float]] = []
    for uid, arr in rows:
        ref = numpy_to_embedding(arr)
        scored.append((uid, cosine_similarity(probe, ref)))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]
    return IdentifyResponse(results=[IdentifyHit(user_id=u, similarity=s) for u, s in top])


@app.post("/compare", response_model=dict, tags=["authentication"])
async def compare(
    threshold: float = Query(0.45, ge=-1.0, le=1.0),
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...),
):
    """Weryfikacja 1:1 bez bazy — dwa uploady (np. eksperymenty)."""
    try:
        da, db = await image_a.read(), await image_b.read()
        e1 = embedding_from_bytes(
            app.state.model, app.state.device, da, face_aligner=app.state.face_aligner
        )
        e2 = embedding_from_bytes(
            app.state.model, app.state.device, db, face_aligner=app.state.face_aligner
        )
    except Exception as e:
        raise HTTPException(400, f"Nie udało się wczytać obrazów: {e}") from e
    sim = cosine_similarity(e1, e2)
    return {"same_person_guess": sim >= threshold, "similarity": sim, "threshold": threshold}
