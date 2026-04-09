"""Automatyczne wdrożenie profili z CelebA (split test, rozłączny z treningiem)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from face_auth.config import BASE_DIR
from face_auth.inference import embedding_from_path, embedding_to_numpy

logger = logging.getLogger(__name__)


def _load_identity_to_cropped_files(
    identity_file: Path,
    cropped_dir: Path,
) -> dict[int, list[Path]]:
    """Mapa: celeb_identity_id -> istniejące ścieżki do *_cropped.jpg."""
    by_id: dict[int, list[Path]] = {}
    if not identity_file.is_file() or not cropped_dir.is_dir():
        return by_id
    with open(identity_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            orig_name, id_str = parts[0], parts[1]
            try:
                celeb_id = int(id_str)
            except ValueError:
                continue
            cropped_name = Path(orig_name).stem + "_cropped.jpg"
            p = cropped_dir / cropped_name
            if p.is_file():
                by_id.setdefault(celeb_id, []).append(p)
    for k in by_id:
        by_id[k].sort()
    return by_id


def _read_split_ids(split_file: Path) -> list[int]:
    ids: list[int] = []
    if not split_file.is_file():
        return ids
    with open(split_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.append(int(line))
            except ValueError:
                continue
    return ids


def run_auto_seed(
    store,
    model,
    device,
    face_aligner,
    *,
    target_count: int | None = None,
    split_name: str | None = None,
) -> int:
    """
    Uzupełnia bazę do ``target_count`` użytkowników z tożsamości ze wskazanego splitu.
    Używa jednego zdjęcia na tożsamość (pierwsze dostępne po sortowaniu).
    Zwraca liczbę nowo zapisanych rekordów w tej sesji.
    """
    if target_count is None:
        target_count = int(os.environ.get("SEED_ENROLLED_COUNT", "80"))
    if target_count <= 0:
        return 0

    split_name = split_name or os.environ.get("SEED_SPLIT", "test").strip().lower()
    split_file = BASE_DIR / "data" / "split" / f"{split_name}_split.txt"
    identity_file = BASE_DIR / "data" / "celeba_metadata" / "identity_CelebA.txt"
    cropped_dir = BASE_DIR / "data" / "img_align_celeba_cropped" / "cropped"

    existing = set(store.list_user_ids())
    if len(existing) >= target_count:
        msg = f"[seed] Już jest {len(existing)} użytkowników (cel {target_count}) — pomijam."
        print(msg, flush=True)
        logger.info(msg)
        return 0

    print(
        f"[seed] Start: cel {target_count} profili, split={split_name}, "
        f"dane pod {BASE_DIR / 'data'}",
        flush=True,
    )

    by_id = _load_identity_to_cropped_files(identity_file, cropped_dir)
    if not by_id:
        msg = (
            f"[seed] Brak mapowania obrazów — sprawdź czy zamontowano:\n"
            f"  - {identity_file}\n"
            f"  - {cropped_dir}"
        )
        print(msg, flush=True)
        logger.warning(msg)
        return 0

    order = _read_split_ids(split_file)
    if not order:
        msg = f"[seed] Brak lub pusty split: {split_file}"
        print(msg, flush=True)
        logger.warning(msg)
        return 0

    added = 0
    for celeb_id in order:
        if len(existing) >= target_count:
            break
        uid = str(celeb_id)
        if uid in existing:
            continue
        paths = by_id.get(celeb_id)
        if not paths:
            continue
        img_path = paths[0]
        try:
            emb = embedding_from_path(
                model, device, img_path, face_aligner=face_aligner
            )
            vec = embedding_to_numpy(emb)
            store.upsert(uid, vec, sample_count=1)
            existing.add(uid)
            added += 1
            if added % 10 == 0:
                print(f"[seed] Zapisano {len(existing)} / {target_count}…", flush=True)
        except Exception as e:
            logger.warning("Seed: pomijam %s (%s): %s", uid, img_path, e)

    total = len(store.list_user_ids())
    summary = (
        f"[seed] Gotowe: +{added} nowych, łącznie {total} użytkowników w bazie (cel {target_count})."
    )
    print(summary, flush=True)
    logger.info(summary)
    if total < target_count:
        print(
            f"[seed] UWAGA: mniej niż {target_count} profili — "
            "dopnij brakujące zdjęcia / split lub zwiększ zbiór testowy.",
            flush=True,
        )
    return added
