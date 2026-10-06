FROM python:3.13-slim AS build
ARG GNOMON_BUILD_COMMIT
ENV GNOMON_BUILD_COMMIT=${GNOMON_BUILD_COMMIT}
WORKDIR /build/core
COPY pyproject.toml README.md LICENSE build_hook.py ./
COPY src ./src
COPY skills ./skills
COPY integrations ./integrations
COPY services/hosted /build/hosted
RUN python -m pip wheel --no-cache-dir --wheel-dir /wheels . /build/hosted

FROM python:3.13-slim
ARG GNOMON_BUILD_COMMIT
LABEL org.opencontainers.image.source="https://github.com/TensorLink-AI/Gnomon" \
      org.opencontainers.image.revision="${GNOMON_BUILD_COMMIT}"
COPY --from=build /wheels /wheels
RUN python -m pip install --no-cache-dir --no-index --find-links=/wheels gnomon-hosted==0.1.0b1 \
    && rm -rf /wheels \
    && useradd --uid 10001 --create-home gnomon \
    && mkdir /data && chown gnomon:gnomon /data
USER 10001:10001
WORKDIR /data
EXPOSE 8765
ENTRYPOINT ["gnomon-hosted", "--root", "/data"]
CMD ["serve", "--host", "0.0.0.0"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/ready', timeout=3)"
