from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_WEIGHTS = BASE_DIR / "results" / "arcface_celeba_best.pth"
DEFAULT_FACE_LANDMARKER_MODEL = BASE_DIR / "models" / "face_landmarker.task"
ENROLL_DB_PATH = BASE_DIR / "data" / "system" / "enrollments.db"

EMBEDDING_DIM = 512
INPUT_SIZE = 112
