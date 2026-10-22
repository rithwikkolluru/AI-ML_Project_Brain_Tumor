# Build Brain Tumor AI Clinical Workstation for Render
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=10000 \
    # Limit TF memory so it doesn't OOM on Render's 512MB RAM
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    # Use only 1 thread for inter/intra ops on the constrained container
    OMP_NUM_THREADS=2 \
    TF_NUM_INTRAOP_THREADS=1 \
    TF_NUM_INTEROP_THREADS=1 \
    # Disable TF XLA JIT compilation (saves ~100MB RAM on startup)
    TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false" \
    # Tell Keras not to use the legacy TF1-style session
    KERAS_BACKEND=tensorflow

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

# Copy trained model weights — use the lighter phase1 model (20MB) as primary on Render
# to avoid OOM on free tier (512MB). Fall back to full model if present.
COPY best_model_phase1.keras /app/
COPY brain_tumor_model.keras /app/

# Runtime directory for temp uploads
RUN mkdir -p /app/uploads_temp && chmod 777 /app/uploads_temp

EXPOSE 10000

# Health check — give the server 90s to load TF model before first check
HEALTHCHECK --interval=30s --timeout=20s --start-period=90s --retries=4 \
  CMD curl -f http://localhost:${PORT}/api/status || exit 1

# Use $PORT so Render's dynamic port injection works
# Use --timeout 120 to avoid premature worker restart during TF model load
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 120"]
