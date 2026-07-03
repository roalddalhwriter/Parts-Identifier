# Use an official lightweight Python image
FROM python:3.10-slim

# Set system environment variables to optimize Python/Torch inside Docker
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/app/model_cache \
    TRANSFORMERS_CACHE=/app/model_cache

WORKDIR /app

# Install system dependencies needed for basic image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application files
COPY main.py .
COPY static/ ./static/

# Pre-download DINOv2 during the build phase so the container starts instantly
RUN python -c "from transformers import AutoImageProcessor, AutoModel; AutoImageProcessor.from_pretrained('facebook/dinov2-base'); AutoModel.from_pretrained('facebook/dinov2-base')"

# Cloud Run passes the port via the PORT environment variable
EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]