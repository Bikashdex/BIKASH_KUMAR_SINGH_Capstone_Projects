FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgdcm-tools curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
# Wheelhouse first (tools/fetch_api_wheels.py fills it). Docker Desktop's build proxy crawled at 16 kB/s
# on the 195 MB torch CPU wheel versus 3 MB/s from the host. torch is installed from the wheelhouse with
# --no-index on its own line: as long as an index is reachable, pip prefers the index copy of the same
# version and re-downloads those 195 MB over the proxy, which is exactly what this avoids.
COPY wheelhouse/ /wheelhouse/
# Both installs are --no-index. With an index reachable, pip asks it about every requirement before
# falling back to --find-links, and DNS inside the build container is unreliable: a ~30 s offline install
# became 20 min of "Failed to establish a new connection ... fastapi" retries. torch gets its own layer
# so that a requirements change does not pay the 195 MB wheel install again.
RUN pip install --no-cache-dir --no-index --find-links=/wheelhouse torch==2.4.1 torchvision==0.19.1
RUN pip install --no-cache-dir --no-index --find-links=/wheelhouse -r requirements-api.txt

COPY src/ src/
COPY configs/ configs/

# /data backs the audit volume. It must exist and belong to appuser in the image: a fresh named volume inherits
# the image directory's ownership, otherwise the volume is root:root and sqlite cannot create audit.db in it.
RUN useradd --create-home --uid 10001 appuser && mkdir -p /data && chown -R appuser:appuser /data
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
