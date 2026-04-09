# Biometrics — face authorization

Code and Docker live under `**face_authorization/**` — all paths below are relative to that folder (run every command from there).

## Before `docker compose up --build`


| What               | Path (under `face_authorization/`)                         | When                                                                                                                                                 |
| ------------------ | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| ArcFace weights    | `results/arcface_celeba_best.pth`                          | **Always** — Dockerfile `COPY results`; `ARCFACE_WEIGHTS` in compose points here. Other filename → put `.pth` in `results/` and set env accordingly. |
| SQLite enrollments | `data/system/` (directory, can be empty)                   | **Always** — mounted for the DB; create with `mkdir -p data/system` if missing.                                                                      |
| CelebA map         | `data/celeba_metadata/identity_CelebA.txt`                 | **If seed on** — optional if `SEED_AUTO=0`.                                                                                                          |
| ID list            | `data/split/test_split.txt`                                | **If seed on** — or `valid_split.txt` + `SEED_SPLIT=valid`.                                                                                          |
| Cropped faces      | `data/img_align_celeba_cropped/cropped/` (`*_cropped.jpg`) | **If seed on** — big tree; seed picks first file per ID.                                                                                             |


**Not on you:** `face_landmarker.task` is downloaded during `**docker build`**. App source is already in the repo under `face_authorization/face_auth` and `face_authorization/app`.

**No seed / no CelebA:** set `SEED_AUTO=0` in `docker-compose.yml` — then only weights + `data/system/` matter.

## Docker

```bash
cd face_authorization
docker compose up --build
```

- UI: [127.0.0.1:8000](http://127.0.0.1:8000/) · [API docs](http://127.0.0.1:8000/docs)
- Compose bind-mounts `face_auth/`, `app/`, `data/system`, and (for seed) the three `data/...` trees — **uvicorn `--reload`**. Stop: `Ctrl+C` / `docker compose down`.

## Auto-seed (optional)

If `SEED_AUTO` is on (default), startup runs `face_auth/seed.py`: reads `data/split/{SEED_SPLIT}_split.txt` (default `test`), **one** sorted `*_cropped.jpg` per ID, same pipeline as API, `**sample_count=1`** (not `enroll_multi`). Target `**SEED_ENROLLED_COUNT`** (default **80**); skips existing users. Split `test`/`valid` vs training on `train`. Missing CelebA paths → app runs; check `[seed]` logs. Off: `SEED_AUTO=0`.

## Env

`ARCFACE_WEIGHTS`, `MEDIAPIPE_FACE_MODEL`, `ENROLL_DB_PATH`, `SEED_`* — `docker-compose.yml`, `app/main.py`.