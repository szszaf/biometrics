from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_VOICE_WEIGHTS = BASE_DIR / "results" / "ecapa_cv_pl_best.pth"
DEFAULT_SPEECHBRAIN_ECAPA_SAVEDIR = BASE_DIR / "models" / "speechbrain_ecapa"
ENROLL_DB_VOICE_PATH = BASE_DIR / "data" / "system" / "enrollments_voice.db"

VOICE_EMBEDDING_DIM = 192
TARGET_SAMPLE_RATE = 16_000
AUTH_MAX_AUDIO_SECONDS = 60.0
CROP_SECONDS = 4.0
MAX_READ_SECONDS = 8.0
NUM_TTA_CROPS = 3


def common_voice_pl_root() -> Path | None:
    """Katalog ``.../pl`` Common Voice PL: domyślna wersja korpusu lub pierwszy ``cv-corpus-*/pl``."""
    data = BASE_DIR / "data"
    preferred = data / "cv-corpus-25.0-2026-03-09" / "pl"
    if (preferred / "clips").is_dir() and (preferred / "validated.tsv").is_file():
        return preferred
    for p in sorted(data.glob("cv-corpus-*/pl")):
        if (p / "clips").is_dir() and (p / "validated.tsv").is_file():
            return p
    return None
