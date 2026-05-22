# Biometrics — **BioDesk** (twarz + głos)

Serwis FastAPI z prostym UI (HTML/CSS/JS) do **rejestracji**, **weryfikacji** (1:1 z podanym identyfikatorem) i **identyfikacji** (1:N) — modality **twarz** (ArcFace + MediaPipe) i/lub **głos** (ECAPA-TDNN + preprocess SpeechBrain).

Kod i Docker: katalog **`biometric_authorization/`** — polecenia uruchamiaj z tego katalogu.

## Wymagania systemowe (P2 — „System”)

Zgodnie z treścią projektu (Biometria, rozpoznawanie mówcy):

| Wymaganie | Realizacja w BioDesk |
|-----------|----------------------|
| Dodawanie użytkowników | API + UI: rejestracja twarzy i/lub głosu (`/users/.../enroll`, `enroll_multi`) |
| Weryfikacja (ID + próbka) | `POST /verify` z `modality=face` lub `voice` |
| Identyfikacja (tylko próbka) | `POST /identify` z `modality=face` lub `voice` |
| Próbki niewidziane przez model na treningu | CelebA: split `data/split/`; głos: `data/voice_split/` + korpus `data/cv-corpus-*/pl/`; seed przy `SEED_AUTO=1` używa wyłącznie wybranego splitu (np. `test`) |
| Profile wielu osób (np. ≥ 100) | API bez limitu; auto-seed: CelebA (`SEED_ENROLLED_COUNT`) i głos z CV (`SEED_VOICE_ENROLLED_COUNT` / `SEED_ENROLLED_COUNT`); dalsze profile przez API |
| Czas rejestracji (wytyczna ~≤ 20 min) | UI zbiera krótkie ujęcia/nagrania; pełna sesja zależy od operatora |
| Próbka uwierzytelnienia ≤ ~60 s (audio) | Serwer przycina analizowany fragment (`voice_auth/config.py`: `AUTH_MAX_AUDIO_SECONDS`) |

Eksperymenty raportowe z PDF (szum, kodeki itd.) są poza zakresem tego repozytorium — przygotuj je w notebookach / osobnych skryptach.

## Zasada: co najmniej jedna modality

- Jeśli **brak** `results/arcface_celeba_best.pth` **lub** `models/face_landmarker.task` → **twarz wyłączona**.
- Jeśli **brak** `results/ecapa_cv_pl_best.pth` **lub** nie da się załadować SpeechBrain ECAPA → **głos wyłączony**.
- Aplikacja wystartuje tylko gdy **co najmniej jedna** modality jest dostępna (`GET /capabilities`).

## Przed `docker compose up --build`

| Zasób | Ścieżka (względem `biometric_authorization/`) | Kiedy |
|--------|-----------------------------------------------|--------|
| ArcFace (twarz) | `results/arcface_celeba_best.pth` | Jeśli modality twarz ma być włączona |
| MediaPipe Face Landmarker | `models/face_landmarker.task` | Pobierany w **Dockerfile** do `models/` |
| ECAPA głos | `results/ecapa_cv_pl_best.pth` | Jeśli modality głosu ma być włączona |
| Cache SpeechBrain ECAPA | `models/speechbrain_ecapa/` | Prepobierany w **Dockerfile** (hub `speechbrain/spkrec-ecapa-voxceleb`) |
| SQLite twarz | `data/system/enrollments.db` (domyślnie; nadpisz `ENROLL_DB_PATH`) | Zawsze (katalog montowany) |
| SQLite głos | `data/system/enrollments_voice.db` (`ENROLL_DB_VOICE_PATH`) | Gdy używasz głosu |
| Common Voice PL (surowe nagrania) | `data/cv-corpus-25.0-2026-03-09/pl/` (`clips/`, `validated.tsv`, …) | **Auto-seed głosu** (`SEED_AUTO=1`): wraz z `data/voice_split/`; bez korpusu seed głosu się pomija |

**Auto-seed przy starcie** (`SEED_AUTO=1`, domyślnie w `docker-compose.yml`):

- **Twarz (CelebA):** `SEED_SPLIT` (np. `test`), `SEED_ENROLLED_COUNT`, `SEED_NOTEBOOK_PIPELINE` — jak wcześniej; wymaga zamontowanych metadanych i cropów.
- **Głos (Common Voice PL):** wymaga `data/cv-corpus-*/pl/` z `validated.tsv` i `clips/`, oraz `data/voice_split/{split}_split.txt`. Zmienne: `SEED_VOICE_SPLIT` (domyślnie `test`), `SEED_VOICE_ENROLLED_COUNT` (gdy puste — używane jest `SEED_ENROLLED_COUNT`), `SEED_VOICE_VALIDATED_TSV` (domyślnie `validated.tsv`). Bez korpusu lub splitu w logach pojawi się `[seed-voice]` i seed się pominie.

**Wyłączenie obu seedów:** `SEED_AUTO=0`.

**Docker build — brak lokalnego `ecapa_cv_pl_best.pth`:** możesz podać `VOICE_WEIGHTS_URL` (np. w `docker-compose.yml` → `build.args` albo `docker build --build-arg VOICE_WEIGHTS_URL=https://…/ecapa_cv_pl_best.pth`). Krok `RUN` w **Dockerfile** pobierze plik tylko wtedy, gdy po `COPY results` plik nadal nie istnieje.

## Docker

```bash
cd biometric_authorization
docker compose up --build
```

- UI: [http://127.0.0.1:8000/](http://127.0.0.1:8000/) · [OpenAPI](http://127.0.0.1:8000/docs)
- Montowane: `face_auth/`, `voice_auth/`, `app/`, `data/system`, zbiory pod seed itd. (jak w `docker-compose.yml`).

## API — parametr `modality`

Dla `GET /users`, `POST .../enroll`, `enroll_multi`, `POST /verify`, `POST /identify`, `DELETE /users/{id}`:

- `?modality=face` lub `?modality=voice`
- Domyślnie wybierana jest pierwsza dostępna: **face**, jeśli włączona, w przeciwnym razie **voice**.

Dodatkowo: `GET /capabilities`, `POST /compare` (tylko twarz), `POST /compare_voice` (tylko głos).

## Notebook treningowy (głos)

`biometric_authorization/notebooks/ecapa_finetune.ipynb` — douczanie ECAPA na Common Voice PL (oryginał przeniesiony z projektu).

## Struktura katalogów

```
biometric_authorization/
  app/                 # FastAPI + static (UI)
  face_auth/           # ArcFace, MediaPipe, seed CelebA, baza twarzy
  voice_auth/          # ECAPA inference, konfiguracja audio
  data/
    system/            # SQLite
    split/             # CelebA ID lists
    voice_split/       # listy mówców CV PL (train/valid/test)
    cv-corpus-25.0-2026-03-09/pl/   # Common Voice PL po rozpakowaniu tar.gz (lokalnie)
  models/              # face_landmarker.task, speechbrain_ecapa/
  results/             # *.pth (lokalnie; w .gitignore)
  notebooks/
```
