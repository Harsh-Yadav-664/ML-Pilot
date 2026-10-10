# MLPilot in one image: the API and the built web UI (issue #68).
#
#   docker build -f docker/Dockerfile -t mlpilot .
#   docker run --rm -p 127.0.0.1:8000:8000 -v mlpilot-data:/data mlpilot
#
# The log prints the address to open, with the access token after the "#". Always publish the
# port on 127.0.0.1 as above: inside the container the server listens on all interfaces, which is
# what lets Docker forward the port, so the host side of -p is what keeps it off the network.
# The state (metadata database, snapshots, models, token) is in /data.

# -- 1. the web UI, built for the same origin as the API --------------------------------------
FROM node:22-slim AS ui
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
# An empty token overrides any frontend/.env.local that slipped into the context.
RUN VITE_API_BASE_URL=/api/v1 VITE_MLPILOT_TOKEN= npm run build

# -- 2. the wheel: the same `pip install mlpilot` a user gets ----------------------------------
FROM python:3.13-slim AS wheel
RUN pip install --no-cache-dir build
WORKDIR /src/backend
COPY backend/ ./
COPY --from=ui /src/frontend/dist/index.html mlpilot/ui/index.html
RUN python -m build --wheel --outdir /wheels

# -- 3. the runtime ---------------------------------------------------------------------------
FROM python:3.13-slim
# LightGBM needs the OpenMP runtime, which the slim image lacks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=wheel /wheels/*.whl /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels
RUN useradd --create-home --uid 10001 mlpilot && mkdir /data && chown mlpilot /data
USER mlpilot
ENV MLPILOT_HOME=/data \
    PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
CMD ["mlpilot", "ui", "--host", "0.0.0.0", "--port", "8000", "--no-open"]
