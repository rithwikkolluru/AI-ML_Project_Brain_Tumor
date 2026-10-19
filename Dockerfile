# Build Brain Tumor AI Clinical Workstation for Render
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=10000 \
    # Limit TF memory growth so it doesn't OOM on Render's 512MB
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    OMP_NUM_THREADS=2

# System dependencies (OpenCV headless needs libGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements first (layer cache)
COPY requirements-render.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-render.txt

# Copy application source
COPY backend/  /app/backend/
COPY frontend/ /app/frontend/
COPY data/     /app/data/
COPY main.py   /app/

# Copy trained model weights (39 MB – tracked in git via LFS-friendly size)
COPY brain_tumor_model.keras /app/

# Runtime directory for temp uploads
RUN mkdir -p /app/uploads_temp && chmod 777 /app/uploads_temp

EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:${PORT}/api/status || exit 1

# Use $PORT so Render's dynamic port injection works
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
