FROM python:3.12-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
     && rm -rf /var/lib/apt/lists/* \
     && python -m pip --cert /etc/ssl/certs/ca-certificates.crt install --no-cache-dir --upgrade "pip>=26.2.2" "certifi>=2026.10.19"

WORKDIR /build
RUN python -m pip install --no-cache-dir build
COPY pyproject.toml build_hook.py README.md LICENSE ./
COPY src ./src
COPY skills ./skills
ARG GNOMON_BUILD_COMMIT
RUN GNOMON_BUILD_COMMIT="$GNOMON_BUILD_COMMIT" python -m build --wheel

FROM python:3.12-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
     && rm -rf /var/lib/apt/lists/* \
     && python -m pip --cert /etc/ssl/certs/ca-certificates.crt install --no-cache-dir --upgrade "pip>=26.2.2" "certifi>=2026.10.19"

LABEL org.opencontainers.image.source="https://github.com/TensorLink-AI/Gnomon"
LABEL org.opencontainers.image.description="Evidence-backed local forecasting runtime"

RUN groupadd --system gnomon && useradd --system --gid gnomon --create-home gnomon
COPY --from=builder /build/dist/*.whl /tmp/
RUN python -m pip install --no-cache-dir /tmp/*.whl && rm -f /tmp/*.whl

USER gnomon
WORKDIR /data
ENTRYPOINT ["gnomon"]
CMD ["capabilities"]
