# Kokoro-82M TTS server for DGX Spark (GB10, aarch64, sm_121).
#
# The NGC PyTorch 25.04 container is the first tag with Blackwell support.
# If your host driver requires CUDA 13 userspace, bump to 25.07-py3 or newer.
#
# Build:
#   docker build -t kokoro-tts-server:latest .
# Run (with GPU):
#   docker run --rm --gpus all -p 8880:8880 kokoro-tts-server:latest
FROM nvcr.io/nvidia/pytorch:25.04-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng \
        ffmpeg \
        libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download model weights + all voice packs at build time so the image ships
# fully self-contained. Comment out if you prefer runtime download.
RUN ./scripts/download_model.sh /app/models/kokoro-82m

EXPOSE 8880
ENV PYTHONUNBUFFERED=1 \
    KOKORO_MODEL_DIR=/app/models/kokoro-82m \
    KOKORO_HOST=0.0.0.0 \
    KOKORO_PORT=8880

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8880"]
