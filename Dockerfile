# Build Brain Tumor AI Clinical Workstation for Render
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

# Install system dependencies (GL and image processing libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements-render.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-render.txt

# Copy application files
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY data/ /app/data/
COPY brain_tumor_model.keras /app/
COPY main.py /app/

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:${PORT}/api/status || exit 1

# Start ASGI server using Render's dynamic $PORT
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
