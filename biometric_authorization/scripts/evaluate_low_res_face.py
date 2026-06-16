from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Sequence

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter

from face_auth.config import DEFAULT_WEIGHTS, INPUT_SIZE
from face_auth.inference import (
    celeba_cropped_notebook_preprocess,
    cosine_similarity,
    embedding_from_pil,
)
from face_auth.low_quality import make_low_quality_variants
from face_auth.model import load_backbone

DEFAULT_OUTPUT = BASE_DIR / "data" / "system" / "experiments" / "low_res_face_latest.json"
DEFAULT_IDENTITY_FILE = BASE_DIR / "data" / "celeba_metadata" / "identity_CelebA.txt"
DEFAULT_SPLIT_FILE = BASE_DIR / "data" / "split" / "test_split.txt"
DEFAULT_CROPPED_DIR = BASE_DIR / "data" / "img_align_celeba_cropped" / "cropped"


@dataclass(frozen=True)
class ExperimentMetrics:
    samples: int
    genuine_attempts: int
    impostor_attempts: int
    frr: float
    far: float


@dataclass(frozen=True)
class PersonImages:
    user_id: str
    paths: tuple[Path, ...]


@dataclass
class ExperimentProgress:
    output_path: Path
    started_at: str
    started_monotonic: float
    total: int
    last_write_completed: int = -1

    def update(self, *, stage: str, completed: int, force: bool = False) -> None:
        if not force and completed - self.last_write_completed < 25 and completed < self.total:
            return
        self.last_write_completed = completed
        _write_json(
            self.output_path,
            {
                "status": "running",
                "started_at": self.started_at,
                "finished_at": None,
                "message": f"Running stage: {stage}.",
                "progress": _progress_payload(
                    stage=stage,
                    completed=completed,
                    total=self.total,
                    started_monotonic=self.started_monotonic,
                ),
            },
        )


def main() -> int:
    args = _parse_args()
    output_path = Path(args.output)
    started_at = _utc_now()
    started_monotonic = monotonic()
    total = args.clean_samples + args.low_res_samples
    _write_json(
        output_path,
        {
            "status": "running",
            "started_at": started_at,
            "finished_at": None,
            "message": "Low-res face experiment is running.",
            "progress": _progress_payload(
                stage="initializing",
                completed=0,
                total=total,
                started_monotonic=started_monotonic,
            ),
        },
    )

    try:
        progress = ExperimentProgress(
            output_path=output_path,
            started_at=started_at,
            started_monotonic=started_monotonic,
            total=total,
        )
        result = run_experiment(args, started_at=started_at, progress=progress)
    except Exception as exc:
        _write_json(
            output_path,
            {
                "status": "failed",
                "started_at": started_at,
                "finished_at": _utc_now(),
                "message": str(exc),
                "progress": _progress_payload(
                    stage="failed",
                    completed=0,
                    total=total,
                    started_monotonic=started_monotonic,
                ),
            },
        )
        return 1

    _write_json(output_path, result)
    return 0 if result["status"] == "done" else 1


def run_experiment(
    args: argparse.Namespace,
    *,
    started_at: str,
    progress: ExperimentProgress,
) -> dict:
    output_path = Path(args.output)
    weights_path = Path(args.weights)
    identity_file = Path(args.identity_file)
    split_file = Path(args.split_file)
    cropped_dir = Path(args.cropped_dir)
    _require_file(weights_path, "ArcFace weights")
    _require_file(identity_file, "CelebA identity metadata")
    _require_file(split_file, "CelebA split file")
    _require_dir(cropped_dir, "CelebA cropped images")
    progress.update(stage="loading_people", completed=0, force=True)

    people = _load_people(
        identity_file=identity_file,
        split_file=split_file,
        cropped_dir=cropped_dir,
        max_users=args.users,
    )
    if len(people) < 2:
        raise RuntimeError("Need at least two CelebA identities with two cropped images each.")

    progress.update(stage="loading_model", completed=0, force=True)
    model, device = load_backbone(weights_path)
    transform = celeba_cropped_notebook_preprocess()
    progress.update(stage="building_references", completed=0, force=True)
    references = _build_references(model, device, transform, people)

    clean_metrics = _evaluate_clean(
        model=model,
        device=device,
        transform=transform,
        people=people,
        references=references,
        threshold=args.threshold,
        target_samples=args.clean_samples,
        progress=progress,
    )
    standard_metrics, robust_metrics = _evaluate_low_res(
        model=model,
        device=device,
        transform=transform,
        people=people,
        references=references,
        threshold=args.threshold,
        target_samples=args.low_res_samples,
        base_completed=args.clean_samples,
        progress=progress,
    )
    frr_delta_pp = (robust_metrics.frr - clean_metrics.frr) * 100.0
    passes_metric = frr_delta_pp <= 5.0
    passes_sample_counts = (
        clean_metrics.samples >= args.clean_samples
        and robust_metrics.samples >= args.low_res_samples
        and len(people) >= args.users
    )

    return {
        "status": "done",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "output_path": str(output_path),
        "threshold": args.threshold,
        "requested": {
            "users": args.users,
            "clean_samples": args.clean_samples,
            "low_res_samples": args.low_res_samples,
            "low_res_sizes": [64, 80],
            "rejected_sizes": [8, 12, 16, 24, 32, 48],
        },
        "actual_users": len(people),
        "clean": asdict(clean_metrics),
        "low_res_standard": asdict(standard_metrics),
        "low_res_robust": asdict(robust_metrics),
        "frr_delta_pp": frr_delta_pp,
        "progress": _progress_payload(
            stage="done",
            completed=args.clean_samples + args.low_res_samples,
            total=args.clean_samples + args.low_res_samples,
            started_monotonic=progress.started_monotonic,
        ),
        "passes_p3_metric_requirement": passes_metric,
        "passes_p3_sample_count_requirement": passes_sample_counts,
        "passes_p3_requirement": passes_metric and passes_sample_counts,
        "method": "CelebA clean crops + mild synthetic CCTV degradations (64x64, 80x80); smaller crops are treated as quality rejects.",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate low-res face robustness for P3.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--identity-file", default=str(DEFAULT_IDENTITY_FILE))
    parser.add_argument("--split-file", default=str(DEFAULT_SPLIT_FILE))
    parser.add_argument("--cropped-dir", default=str(DEFAULT_CROPPED_DIR))
    parser.add_argument("--threshold", type=float, default=0.16)
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--clean-samples", type=int, default=1000)
    parser.add_argument("--low-res-samples", type=int, default=5000)
    return parser.parse_args()


def _load_people(
    *,
    identity_file: Path,
    split_file: Path,
    cropped_dir: Path,
    max_users: int,
) -> list[PersonImages]:
    split_ids = _read_split_ids(split_file)
    by_id = _load_identity_to_cropped_files(identity_file, cropped_dir)
    people: list[PersonImages] = []
    for celeb_id in split_ids:
        paths = by_id.get(celeb_id, [])
        if len(paths) < 2:
            continue
        people.append(PersonImages(user_id=str(celeb_id), paths=tuple(paths)))
        if len(people) >= max_users:
            break
    return people


def _build_references(
    model,
    device,
    transform,
    people: Sequence[PersonImages],
) -> dict[str, torch.Tensor]:
    references: dict[str, torch.Tensor] = {}
    for person in people:
        image = Image.open(person.paths[0]).convert("RGB")
        references[person.user_id] = embedding_from_pil(
            model,
            device,
            image,
            transform=transform,
            face_aligner=None,
        )
    return references


def _evaluate_clean(
    *,
    model,
    device,
    transform,
    people: Sequence[PersonImages],
    references: dict[str, torch.Tensor],
    threshold: float,
    target_samples: int,
    progress: ExperimentProgress,
) -> ExperimentMetrics:
    false_rejects = 0
    false_accepts = 0
    progress.update(stage="clean_samples", completed=0, force=True)
    for attempt_idx in range(target_samples):
        person = people[attempt_idx % len(people)]
        probe_path = _probe_path(person, attempt_idx)
        probe_image = Image.open(probe_path).convert("RGB")
        probe = embedding_from_pil(
            model,
            device,
            probe_image,
            transform=transform,
            face_aligner=None,
        )
        if cosine_similarity(probe, references[person.user_id]) < threshold:
            false_rejects += 1
        impostor_user_id = people[(attempt_idx + 1) % len(people)].user_id
        if cosine_similarity(probe, references[impostor_user_id]) >= threshold:
            false_accepts += 1
        progress.update(stage="clean_samples", completed=attempt_idx + 1)
    progress.update(stage="clean_samples", completed=target_samples, force=True)
    return _metrics(target_samples, false_rejects, false_accepts)


def _evaluate_low_res(
    *,
    model,
    device,
    transform,
    people: Sequence[PersonImages],
    references: dict[str, torch.Tensor],
    threshold: float,
    target_samples: int,
    base_completed: int,
    progress: ExperimentProgress,
) -> tuple[ExperimentMetrics, ExperimentMetrics]:
    standard_false_rejects = 0
    standard_false_accepts = 0
    robust_false_rejects = 0
    robust_false_accepts = 0
    progress.update(stage="low_res_samples", completed=base_completed, force=True)
    for attempt_idx in range(target_samples):
        person = people[attempt_idx % len(people)]
        probe_path = _probe_path(person, attempt_idx)
        image = Image.open(probe_path).convert("RGB")
        degraded = _degrade_image(image, attempt_idx)
        standard_probe = embedding_from_pil(
            model,
            device,
            degraded,
            transform=transform,
            face_aligner=None,
        )
        robust_probe = _robust_embedding_from_cropped_image(
            model,
            device,
            transform,
            degraded,
        )
        reference = references[person.user_id]
        impostor_reference = references[people[(attempt_idx + 1) % len(people)].user_id]
        if cosine_similarity(standard_probe, reference) < threshold:
            standard_false_rejects += 1
        if cosine_similarity(standard_probe, impostor_reference) >= threshold:
            standard_false_accepts += 1
        if cosine_similarity(robust_probe, reference) < threshold:
            robust_false_rejects += 1
        if cosine_similarity(robust_probe, impostor_reference) >= threshold:
            robust_false_accepts += 1
        progress.update(stage="low_res_samples", completed=base_completed + attempt_idx + 1)
    progress.update(stage="low_res_samples", completed=base_completed + target_samples, force=True)
    return (
        _metrics(target_samples, standard_false_rejects, standard_false_accepts),
        _metrics(target_samples, robust_false_rejects, robust_false_accepts),
    )


def _read_split_ids(split_file: Path) -> list[int]:
    ids: list[int] = []
    with open(split_file, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            value = line.strip()
            if not value:
                continue
            try:
                ids.append(int(value))
            except ValueError:
                continue
    return ids


def _load_identity_to_cropped_files(
    identity_file: Path,
    cropped_dir: Path,
) -> dict[int, list[Path]]:
    by_id: dict[int, list[Path]] = {}
    with open(identity_file, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                celeb_id = int(parts[1])
            except ValueError:
                continue
            cropped_name = Path(parts[0]).stem + "_cropped.jpg"
            cropped_path = cropped_dir / cropped_name
            if cropped_path.is_file():
                by_id.setdefault(celeb_id, []).append(cropped_path)
    for paths in by_id.values():
        paths.sort()
    return by_id


def _probe_path(person: PersonImages, attempt_idx: int) -> Path:
    probe_paths = person.paths[1:]
    return probe_paths[attempt_idx % len(probe_paths)]


def _robust_embedding_from_cropped_image(
    model,
    device,
    transform,
    image: Image.Image,
) -> torch.Tensor:
    parts = [
        embedding_from_pil(
            model,
            device,
            variant,
            transform=transform,
            face_aligner=None,
        )
        for variant in make_low_quality_variants(image)
    ]
    return F.normalize(torch.stack(parts).mean(dim=0), dim=0, eps=1e-12)


def _degrade_image(image: Image.Image, attempt_idx: int) -> Image.Image:
    sizes = (64, 80)
    small_size = sizes[attempt_idx % len(sizes)]
    degraded = image.resize((small_size, small_size), Image.Resampling.BILINEAR)
    degraded = degraded.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    if attempt_idx % 2 == 0:
        degraded = degraded.filter(ImageFilter.GaussianBlur(radius=0.7 + (attempt_idx % 3) * 0.25))
    if attempt_idx % 3 == 0:
        degraded = ImageEnhance.Contrast(degraded).enhance(0.55)
    if attempt_idx % 5 == 0:
        degraded = ImageEnhance.Brightness(degraded).enhance(0.72)
    arr = np.asarray(degraded).astype(np.int16)
    noise_scale = 4 + (attempt_idx % 4) * 3
    rng = np.random.default_rng(attempt_idx)
    noise = rng.integers(-noise_scale, noise_scale + 1, size=arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _metrics(samples: int, false_rejects: int, false_accepts: int) -> ExperimentMetrics:
    if samples <= 0:
        return ExperimentMetrics(
            samples=0,
            genuine_attempts=0,
            impostor_attempts=0,
            frr=math.nan,
            far=math.nan,
        )
    return ExperimentMetrics(
        samples=samples,
        genuine_attempts=samples,
        impostor_attempts=samples,
        frr=false_rejects / samples,
        far=false_accepts / samples,
    )


def _progress_payload(
    *,
    stage: str,
    completed: int,
    total: int,
    started_monotonic: float,
) -> dict:
    elapsed_seconds = max(0.0, monotonic() - started_monotonic)
    percent = (100.0 * completed / total) if total > 0 else 0.0
    eta_seconds = None
    if completed > 0 and completed < total:
        rate = completed / max(elapsed_seconds, 1e-6)
        eta_seconds = (total - completed) / rate
    return {
        "stage": stage,
        "completed": completed,
        "total": total,
        "percent": percent,
        "elapsed_seconds": elapsed_seconds,
        "eta_seconds": eta_seconds,
    }


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} not found: {path}")


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise RuntimeError(f"{label} not found: {path}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
