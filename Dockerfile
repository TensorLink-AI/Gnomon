FROM python:3.12-slim AS builder

WORKDIR /build
RUN python -m pip install --no-cache-dir build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m build --wheel

FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.source="https://github.com/TensorLink-AI/Gnomon"
LABEL org.opencontainers.image.description="Evidence-backed local forecasting runtime"

RUN groupadd --system gnomon && useradd --system --gid gnomon --create-home gnomon
COPY --from=builder /build/dist/*.whl /tmp/
RUN python -m pip install --no-cache-dir /tmp/*.whl && rm -f /tmp/*.whl

USER gnomon
WORKDIR /data
ENTRYPOINT ["gnomon"]
CMD ["capabilities"]

