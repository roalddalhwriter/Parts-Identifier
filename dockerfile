FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Install minimal system dependencies required for basic imaging
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your backend code files
COPY main.py .

# Pre-download the DINOv2 model during the container build phase.
# This prevents Cloud Run from timing out on cold starts.
RUN python -c "from transformers import AutoImageProcessor, AutoModel; AutoImageProcessor.from_pretrained('facebook/dinov2-base'); AutoModel.from_pretrained('facebook/dinov2-base')"

# Expose port 8080
EXPOSE 8080

# Fire up uvicorn using the same environment structure as your old API
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]