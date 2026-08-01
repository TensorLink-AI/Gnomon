"""Fetching and caching benchmark corpora.

Datasets are downloaded once into a content-addressed cache and read from
there afterwards, so a run is reproducible and offline-repeatable. Every
loader also accepts a local path, which is the supported route for air-gapped
machines and for pinning a revision by hand.

Parquet needs ``pyarrow``. That is a genuine dependency of the *benchmark*,
not of Aion, and it stays in the ``bench`` extra.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .contracts import BenchError

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
CACHE_ENV = "AIONBENCH_CACHE"
DEFAULT_CACHE = Path.home() / ".cache" / "aionbench"


def cache_dir(override: str | os.PathLike[str] | None = None) -> Path:
    directory = Path(override or os.environ.get(CACHE_ENV) or DEFAULT_CACHE).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def hf_url(repo_id: str, filename: str, *, repo_type: str = "datasets", revision: str = "main") -> str:
    quoted = "/".join(urllib.parse.quote(part, safe="") for part in filename.split("/"))
    return f"{HF_ENDPOINT}/{repo_type}/{repo_id}/resolve/{urllib.parse.quote(revision, safe='')}/{quoted}"


def fetch(
    url: str,
    *,
    cache: str | os.PathLike[str] | None = None,
    filename: str | None = None,
    force: bool = False,
    timeout: int = 300,
) -> Path:
    """Download ``url`` into the cache once; return the local path."""
    import hashlib

    directory = cache_dir(cache) / "downloads"
    directory.mkdir(parents=True, exist_ok=True)
    stem = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    suffix = Path(urllib.parse.urlparse(url).path).suffix
    target = directory / (filename or f"{stem}{suffix}")

    if target.is_file() and target.stat().st_size > 0 and not force:
        return target

    headers = {"User-Agent": "AionBench/0.1"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token and HF_ENDPOINT in url:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, headers=headers)
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, open(partial, "wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
    except urllib.error.HTTPError as exc:
        partial.unlink(missing_ok=True)
        raise BenchError(
            "DATASET_DOWNLOAD_FAILED",
            f"HTTP {exc.code} fetching {url}. "
            "Gated datasets need HF_TOKEN set; offline machines can pass a local path instead.",
            {"url": url, "status": exc.code},
        ) from exc
    except urllib.error.URLError as exc:
        partial.unlink(missing_ok=True)
        raise BenchError(
            "DATASET_UNREACHABLE",
            f"Could not reach {url}: {exc.reason}. Download the file elsewhere "
            "and point the suite at it with --data-path.",
            {"url": url},
        ) from exc

    partial.replace(target)
    return target


def read_parquet(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read a parquet file into plain dicts."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise BenchError(
            "PYARROW_REQUIRED",
            "Reading parquet benchmark files needs pyarrow. "
            "Install the bench extra: pip install -e '.[bench]'",
        ) from exc

    table = pq.read_table(str(path))
    return table.to_pylist()


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise BenchError(
                    "INVALID_JSONL",
                    f"{path}:{number} is not valid JSON: {exc}",
                    {"path": str(path), "line": number},
                ) from exc
    return rows


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: str | os.PathLike[str]) -> list[dict[str, str]]:
    import csv

    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_table(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Dispatch on extension: parquet, jsonl, json, or csv."""
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return read_parquet(path)
    if suffix in (".jsonl", ".ndjson"):
        return read_jsonl(path)
    if suffix == ".json":
        payload = read_json(path)
        if isinstance(payload, list):
            return [dict(row) for row in payload]
        for key in ("data", "rows", "questions", "items"):
            if isinstance(payload.get(key), list):
                return [dict(row) for row in payload[key]]
        raise BenchError(
            "UNRECOGNISED_JSON_LAYOUT",
            f"{path} is a JSON object with no obvious row array.",
            {"keys": sorted(payload)[:20]},
        )
    if suffix in (".csv", ".tsv"):
        return read_csv(path)
    raise BenchError(
        "UNSUPPORTED_DATASET_FORMAT",
        f"Cannot read {path}: unsupported extension {suffix!r}.",
        {"supported": [".parquet", ".jsonl", ".json", ".csv"]},
    )


def load_pickle(path: str | os.PathLike[str], *, allowed: bool) -> Any:
    """Unpickle a dataset asset, but only with explicit consent.

    TSAIA ships its executor variables, contexts, constraints, and ground
    truth as pickles. Unpickling executes arbitrary code by construction, so
    this refuses unless the caller has opted in for this run.
    """
    if not allowed:
        raise BenchError(
            "PICKLE_NOT_ALLOWED",
            f"{path} is a pickle, and unpickling runs arbitrary code. "
            "Pass --allow-pickle to load this suite's assets, ideally in a container.",
            {"path": str(path)},
        )
    import pickle

    with open(path, "rb") as handle:
        return pickle.load(handle)
