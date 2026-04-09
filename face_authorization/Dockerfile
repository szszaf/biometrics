FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libegl1 \
    libgl1 \
    libgles2-mesa \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p models \
    && curl -fsSL -o models/face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY face_auth ./face_auth
COPY app ./app
COPY results ./results

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
