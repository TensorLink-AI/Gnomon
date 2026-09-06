# Build separately; this image is never a dependency of the core Gnomon wheel.
# The default base is immutable; overrides must also be trusted immutable references.
ARG BASE_IMAGE=python@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a
FROM ${BASE_IMAGE}
COPY requirements.txt /opt/ordinary-requirements.txt
RUN python -m pip install --no-cache-dir --require-hashes --only-binary=:all: \
    --index-url https://pypi.org/simple -r /opt/ordinary-requirements.txt \
    && python -m pip check
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    NUMBA_CACHE_DIR=/tmp/numba PYTHONDONTWRITEBYTECODE=1
USER 65534:65534
WORKDIR /tmp
ENTRYPOINT ["python"]
