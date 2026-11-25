# ECG Anomaly Detection API
# Build: docker build -t ecg-api .
# Run:   docker run -d --name ecg-api -p 8000:8000 ecg-api   ->  http://localhost:8000/docs
FROM python:3.11-slim

# libgomp1: OpenMP runtime needed by LightGBM
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # serve the committed model file; no MLflow registry and no training in the container
    MODEL_SOURCE=file \
    AUTO_TRAIN=false \
    PORT=8000

WORKDIR /app

# dependencies first, so code changes don't reinstall them
COPY requirements.txt .
RUN pip install -r requirements.txt

# source code and the trained model
COPY src/ ./src/
COPY models/ ./models/

# run as a non-root user
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# python:3.11-slim has no curl, so the healthcheck uses Python
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=4)" || exit 1

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
