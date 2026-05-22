FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    libegl1 \
    libgl1 \
    libgles2-mesa \
    libglib2.0-0 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p models \
    && curl -fsSL -o models/face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

COPY requirements.txt .
# CPU wheels — mniejszy obraz i brak zależności od GPU w kontenerze (inferencja na CPU wystarczy).
RUN pip install --no-cache-dir \
    torch==2.5.1+cpu torchvision==0.20.1+cpu torchaudio==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

RUN python -c "\
from speechbrain.inference.speaker import EncoderClassifier;\
EncoderClassifier.from_hparams(\
    source='speechbrain/spkrec-ecapa-voxceleb',\
    savedir='/app/models/speechbrain_ecapa',\
    run_opts={'device': 'cpu'},\
);\
print('SpeechBrain ECAPA cache OK')"

COPY face_auth ./face_auth
COPY voice_auth ./voice_auth
COPY app ./app
COPY results ./results
ARG VOICE_WEIGHTS_URL=""
RUN set -e; \
    if [ ! -f results/ecapa_cv_pl_best.pth ] && [ -n "$VOICE_WEIGHTS_URL" ]; then \
      echo "Pobieranie wag głosu (VOICE_WEIGHTS_URL)..."; \
      curl -fsSL "$VOICE_WEIGHTS_URL" -o results/ecapa_cv_pl_best.pth; \
    fi

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
