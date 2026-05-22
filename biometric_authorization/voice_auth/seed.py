"""Automatyczne wgrywanie profili głosu z Common Voice PL (split z ``data/voice_split/``)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from voice_auth.config import BASE_DIR, common_voice_pl_root

logger = logging.getLogger(__name__)


def _read_split_client_ids(split_file: Path) -> list[str]:
    ids: list[str] = []
    if not split_file.is_file():
        return ids
    with open(split_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def _build_client_to_sorted_clips(validated_tsv: Path) -> dict[str, list[Path]]:
    """Mapa ``client_id`` → posortowane ścieżki do plików w ``clips/`` (tylko istniejące)."""
    clips_dir = validated_tsv.parent / "clips"
    by_client: dict[str, list[Path]] = {}
    if not validated_tsv.is_file() or not clips_dir.is_dir():
        return by_client
    with open(validated_tsv, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
        if "client_id" not in header or "path" not in header:
            logger.warning("[seed-voice] Nieoczekiwany nagłówek validated.tsv: %s", validated_tsv)
        for line in f:
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) < 2:
                continue
            cid, rel = parts[0], parts[1]
            if not cid or not rel:
                continue
            p = clips_dir / rel
            if p.is_file():
                by_client.setdefault(cid, []).append(p)
    for k in by_client:
        by_client[k].sort(key=lambda x: x.name)
    return by_client


def run_voice_auto_seed(
    store,
    voice_engine,
    *,
    target_count: int | None = None,
    split_name: str | None = None,
    validated_name: str | None = None,
) -> int:
    """
    Uzupełnia bazę głosu do ``target_count`` użytkowników (``client_id`` ze splitu).
    Jedna walidowana próbka audio na mówcę (pierwsza po sortowaniu ścieżek), jak jedno zdjęcie w seedzie twarzy.

    ``SEED_AUTO`` — włączany z ``app.main`` (ten moduł nie sprawdza flagi).
    ``SEED_VOICE_ENROLLED_COUNT`` — nadpisuje liczbę docelową; inaczej ``SEED_ENROLLED_COUNT``, domyślnie 80.
    ``SEED_VOICE_SPLIT`` — np. ``test`` → ``data/voice_split/test_split.txt`` (domyślnie ``test``).
    ``SEED_VOICE_VALIDATED_TSV`` — domyślnie ``validated.tsv`` w katalogu ``pl`` korpusu.
    """
    if target_count is None:
        raw = os.environ.get("SEED_VOICE_ENROLLED_COUNT", "").strip()
        if raw:
            target_count = int(raw)
        else:
            target_count = int(os.environ.get("SEED_ENROLLED_COUNT", "80"))
    if target_count <= 0:
        return 0

    split_name = split_name or os.environ.get("SEED_VOICE_SPLIT", "test").strip().lower()
    validated_name = (validated_name or os.environ.get("SEED_VOICE_VALIDATED_TSV", "validated.tsv")).strip()

    pl_root = common_voice_pl_root()
    split_file = BASE_DIR / "data" / "voice_split" / f"{split_name}_split.txt"

    existing = set(store.list_user_ids())
    if len(existing) >= target_count:
        msg = f"[seed-voice] Już jest {len(existing)} użytkowników (cel {target_count}) — pomijam."
        print(msg, flush=True)
        logger.info(msg)
        return 0

    if pl_root is None:
        msg = (
            f"[seed-voice] Brak rozpakowanego korpusu Common Voice PL pod {BASE_DIR / 'data'} "
            f"(oczekiwane: …/cv-corpus-*/pl z validated.tsv i clips/)."
        )
        print(msg, flush=True)
        logger.warning(msg)
        return 0

    validated_tsv = pl_root / validated_name
    if not validated_tsv.is_file():
        msg = f"[seed-voice] Brak pliku {validated_tsv}"
        print(msg, flush=True)
        logger.warning(msg)
        return 0

    print(
        f"[seed-voice] Start: cel {target_count} profili, split={split_name}, "
        f"korpus={pl_root}, validated={validated_name}",
        flush=True,
    )

    by_client = _build_client_to_sorted_clips(validated_tsv)
    if not by_client:
        msg = f"[seed-voice] Brak wpisów w {validated_tsv} z istniejącymi plikami w clips/"
        print(msg, flush=True)
        logger.warning(msg)
        return 0

    order = _read_split_client_ids(split_file)
    if not order:
        msg = f"[seed-voice] Brak lub pusty split: {split_file}"
        print(msg, flush=True)
        logger.warning(msg)
        return 0

    added = 0
    for cid in order:
        if len(existing) >= target_count:
            break
        if cid in existing:
            continue
        paths = by_client.get(cid)
        if not paths:
            continue
        clip = paths[0]
        try:
            data = clip.read_bytes()
            vec = voice_engine.embed_from_bytes(data)
            store.upsert(cid, vec, sample_count=1)
            existing.add(cid)
            added += 1
            if added % 10 == 0:
                print(f"[seed-voice] Zapisano {len(existing)} / {target_count}…", flush=True)
        except Exception as e:
            logger.warning("[seed-voice] Pomijam %s (%s): %s", cid, clip, e)

    total = len(store.list_user_ids())
    summary = f"[seed-voice] Gotowe: +{added} nowych, łącznie {total} użytkowników (głos), cel {target_count}."
    print(summary, flush=True)
    logger.info(summary)
    if total < target_count:
        print(
            f"[seed-voice] UWAGA: mniej niż {target_count} profili — sprawdź split / korpus / zgodność client_id.",
            flush=True,
        )
    return added
