"""Governed boundary for LLM-extracted point-in-time covariate tables.

The model is useful for finding rows in messy documents and normalising them
into a table. It is not a source of observations. Every admitted value and
time therefore remains tied to a verbatim source span, while ``known_at`` is
owned by the host that supplied the document. Gnomon's existing fold-safe
covariate ablation remains the only mechanism that grants forecast influence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from typing import Any

from .covariates import covariates_from_rows, parse_mapping

CONTRACT_VERSION = "0.1"
MAX_TABLES = 8
MAX_ROWS = 500
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_NUMBER = re.compile(
    r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

COVARIATE_TABLES_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "type": {"type": "string"},
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "document_index": {"type": "integer"},
                        "timestamp": {"type": "string"},
                        "source_time_span": {"type": "string"},
                        "value": {"type": "number"},
                        "evidence_quote": {"type": "string"},
                    },
                    "required": [
                        "document_index", "timestamp", "source_time_span",
                        "value", "evidence_quote",
                    ],
                },
            },
        },
        "required": ["name", "type", "rows"],
    },
}


def _aware_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _time_is_cited(normalised: datetime, source_time: str) -> bool:
    """Accept an exact aware instant or its unambiguous ISO calendar date."""
    text = source_time.strip()
    if not text:
        return False
    parsed = _aware_timestamp(text)
    if parsed is not None:
        return parsed == normalised
    try:
        parsed_date = datetime.fromisoformat(text)
        return (parsed_date.date() == normalised.date()
                and normalised.hour == normalised.minute
                == normalised.second == normalised.microsecond == 0)
    except ValueError:
        return False


def _value_is_cited(value: float, quote: str, source_time: str) -> bool:
    # A date component is not evidence for a covariate value (e.g. value 27
    # must not match the day in 2026-08-27).
    value_text = quote.replace(source_time, " ").replace(",", "")
    for token in _NUMBER.findall(value_text):
        try:
            cited = float(token)
        except ValueError:
            continue
        if math.isclose(cited, value, rel_tol=1e-12, abs_tol=1e-12):
            return True
    return False


def validate_llm_covariate_tables(
    raw: Any,
    *,
    documents: list[str],
    known_at: str | None,
    as_of: str,
    document_known_at: list[str | None] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Validate and seal LLM-extracted tables for inline Gnomon use.

    ``known_at`` and ``as_of`` are host-owned. Model-supplied availability
    timestamps are deliberately ignored, preventing retrospective context
    from entering historical folds. The returned rows have already passed the
    same loader used by Gnomon's public inline-covariate channel.
    """
    rejections: list[str] = []
    boundary = _aware_timestamp(as_of)
    fallback_known = _aware_timestamp(known_at) if known_at is not None else None
    if boundary is None:
        raise ValueError("as_of must be timezone-aware ISO-8601")
    supplied = document_known_at or [None] * len(documents)
    if len(supplied) != len(documents):
        raise ValueError("document_known_at must align one-for-one with documents")
    known_by_document: list[datetime | None] = []
    for item in supplied:
        parsed = _aware_timestamp(item) if item is not None else fallback_known
        if parsed is not None and parsed > boundary:
            raise ValueError("document known_at cannot be later than as_of")
        known_by_document.append(parsed)
    proposals = raw if isinstance(raw, list) else []
    if raw not in (None, []) and not isinstance(raw, list):
        rejections.append("covariate_tables is not an array")

    tables: list[dict[str, Any]] = []
    for table_index, proposal in enumerate(proposals[:MAX_TABLES]):
        label = f"covariate table {table_index + 1}"
        if not isinstance(proposal, dict):
            rejections.append(f"{label} is not an object")
            continue
        name = str(proposal.get("name") or "").strip()
        value_type = str(proposal.get("type") or "").strip()
        if not _NAME.fullmatch(name):
            rejections.append(f"{label} has an invalid name")
            continue
        mapping = [{"name": name, "type": value_type,
                    "availability": "future_known"}]
        try:
            parse_mapping(mapping)
        except Exception as error:
            rejections.append(f"{label} has invalid mapping: {error}")
            continue
        clean_rows: list[dict[str, Any]] = []
        for row_index, row in enumerate((proposal.get("rows") or [])[:MAX_ROWS]):
            row_label = f"{label} row {row_index + 1}"
            if not isinstance(row, dict):
                rejections.append(f"{row_label} is not an object")
                continue
            try:
                document_index = int(row.get("document_index", -1))
            except (TypeError, ValueError):
                document_index = -1
            if not 0 <= document_index < len(documents):
                rejections.append(f"{row_label} has invalid document_index")
                continue
            host_known = known_by_document[document_index]
            if host_known is None:
                rejections.append(f"{row_label} source has no host-owned known_at")
                continue
            quote = str(row.get("evidence_quote") or "").strip()
            source_time = str(row.get("source_time_span") or "").strip()
            document = documents[document_index]
            if not quote or quote not in document:
                rejections.append(f"{row_label} lacks a verbatim evidence_quote")
                continue
            if source_time not in quote:
                rejections.append(f"{row_label} source_time_span is not quoted")
                continue
            timestamp = _aware_timestamp(row.get("timestamp"))
            if timestamp is None or not _time_is_cited(timestamp, source_time):
                rejections.append(f"{row_label} timestamp is not supported by its quote")
                continue
            try:
                value = float(row["value"])
            except (KeyError, TypeError, ValueError):
                value = math.nan
            if not math.isfinite(value) or not _value_is_cited(
                    value, quote, source_time):
                rejections.append(f"{row_label} value is not numeric and verbatim-cited")
                continue
            clean_rows.append({
                "timestamp": timestamp.isoformat(),
                "known_at": host_known.isoformat(),
                name: value,
                "provenance": {
                    "class": "llm_extracted_host_verified",
                    "document_index": document_index,
                    "evidence_quote": quote,
                    "source_time_span": source_time,
                },
            })
        if not clean_rows:
            rejections.append(f"{label} has no admissible rows")
            continue
        # Exercise the real public loader now; provenance is retained in the
        # receipt but omitted from the inline row passed to the numeric loader.
        inline = [{key: value for key, value in row.items()
                   if key != "provenance"} for row in clean_rows]
        try:
            dataset = covariates_from_rows(inline, mapping)
        except Exception as error:
            rejections.append(f"{label} failed covariate validation: {error}")
            continue
        tables.append({
            "name": name,
            "mapping": mapping[0],
            "rows": clean_rows,
            "dataset_fingerprint": dataset.fingerprint,
            "provenance_class": "llm_extracted_host_verified",
            "forecast_influence": "requires_fold_safe_ablation",
        })

    body = {
        "version": CONTRACT_VERSION,
        "known_at_by_document": [
            item.isoformat() if item is not None else None
            for item in known_by_document
        ],
        "as_of": boundary.isoformat(),
        "tables": tables,
        "llm_can_propose": True,
        "llm_can_admit": False,
        "future_observations_exposed": False,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    body["seal_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return body, rejections


def inline_covariate_arguments(receipt: dict[str, Any]) -> dict[str, Any]:
    """Flatten a single validated table into public forecast arguments.

    Multiple tables can have different time coverage and vintage sets; silent
    merging would make their provenance ambiguous. Callers must combine them
    explicitly in a later registry design, so v0.1 binds exactly one table.
    """
    tables = list(receipt.get("tables") or [])
    if not tables:
        return {}
    if len(tables) != 1:
        return {}
    table = tables[0]
    rows = [{key: value for key, value in row.items() if key != "provenance"}
            for row in table["rows"]]
    return {"covariates": rows, "covariate_mapping": [table["mapping"]]}
