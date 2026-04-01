# ── xenium-registration-v2 ───────────────────────────────────────────────────
# Streamlit app: Xenium image registration + h5ad export
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL org.opencontainers.image.title="xenium-registration" \
      org.opencontainers.image.description="Xenium image registration and h5ad export Streamlit app" \
      org.opencontainers.image.version="2.0"

# System deps for image codecs and zarr
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libjpeg-dev \
        libpng-dev \
        libwebp-dev \
        libzstd-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY xenium_app/ xenium_app/
COPY scripts/   scripts/
COPY tests/     tests/
COPY app.py     .

# Streamlit config — headless, non-interactive
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# Mount Xenium bundle data at /data (read-only recommended)
# docker run -v /path/to/xenium_bundles:/data -v /path/to/output:/output ...
VOLUME ["/data", "/output"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
