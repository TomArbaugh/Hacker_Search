# Docker image for Hugging Face Spaces (free CPU tier).
# Bakes the embedding model AND the prebuilt Chroma index into the image, so a
# cold start is just "boot + load from local disk into RAM" — no downloads,
# no embedding at runtime.
FROM python:3.11-slim

# HF Spaces runs containers as a non-root user with uid 1000. Set up its home
# and point all model caches there so the baked model persists in the image.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/huggingface \
    PORT=7860

WORKDIR /app

# Install deps first so this layer is cached across code changes.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app + the prebuilt index (data/chroma must exist locally first).
COPY --chown=user . .

USER user

# Pre-download the embedding model into the image cache at build time.
RUN python -c "import config; from sentence_transformers import SentenceTransformer; SentenceTransformer(config.EMBED_MODEL)"

EXPOSE 7860

# 1 worker keeps RAM low (the model loads once); threads handle light concurrency.
# --preload loads the app before forking so we see import errors in startup logs.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "8", \
     "--timeout", "120", "--preload", "app:app"]
