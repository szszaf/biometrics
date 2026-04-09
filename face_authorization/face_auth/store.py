import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from face_auth.config import EMBEDDING_DIM, ENROLL_DB_PATH


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # FastAPI uruchamia sync handlery w wielu wątkach — jedno połączenie + lock zamiast check_same_thread.
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS enrollments (
            user_id TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL,
            sample_count INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    conn.commit()
    cols = {row[1] for row in conn.execute("PRAGMA table_info(enrollments)").fetchall()}
    if "sample_count" not in cols:
        conn.execute("ALTER TABLE enrollments ADD COLUMN sample_count INTEGER NOT NULL DEFAULT 1")
        conn.commit()
    return conn


class EnrollmentStore:
    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path) if db_path else ENROLL_DB_PATH
        self._conn = _connect(self.db_path)
        self._lock = threading.Lock()

    def close(self):
        with self._lock:
            self._conn.close()

    def upsert(self, user_id: str, embedding: np.ndarray, sample_count: int = 1) -> None:
        if embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(f"embedding must be ({EMBEDDING_DIM},), got {embedding.shape}")
        if sample_count < 1:
            raise ValueError("sample_count must be >= 1")
        blob = embedding.astype(np.float32).tobytes()
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO enrollments (user_id, embedding, created_at, sample_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    embedding = excluded.embedding,
                    created_at = excluded.created_at,
                    sample_count = excluded.sample_count
                """,
                (user_id, blob, now, sample_count),
            )
            self._conn.commit()

    def delete(self, user_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM enrollments WHERE user_id = ?", (user_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def get(self, user_id: str) -> np.ndarray | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT embedding FROM enrollments WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32).copy()

    def list_user_ids(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT user_id FROM enrollments ORDER BY user_id").fetchall()
        return [r[0] for r in rows]

    def list_users_info(self) -> list[tuple[str, int, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT user_id, sample_count, created_at FROM enrollments ORDER BY user_id"
            ).fetchall()
        return [(str(r[0]), int(r[1]), str(r[2])) for r in rows]

    def all_embeddings(self) -> list[tuple[str, np.ndarray]]:
        with self._lock:
            rows = self._conn.execute("SELECT user_id, embedding FROM enrollments").fetchall()
        return [(uid, np.frombuffer(blob, dtype=np.float32).copy()) for uid, blob in rows]
