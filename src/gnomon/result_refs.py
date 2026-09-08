"""Bounded session result receipts, distinct from the durable temporal ledger.

Limits bound encoded responses and retained temporary bytes, not trusted provider
execution or parser peak memory. Retrieval never invokes a provider or opens a
caller-selected path. JSON text pages concatenate exactly; pointers select values
without requiring an agent to retrieve an entire large result.
"""

from collections import OrderedDict
from dataclasses import dataclass, asdict
import hashlib
import json
import re
from pathlib import Path
import tempfile
from uuid import uuid4

from .contracts import GnomonError


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class ResultLimits:
    max_response_bytes: int = 8192
    max_result_bytes: int = 16 * 1024 * 1024
    max_retained_bytes: int = 64 * 1024 * 1024
    max_results: int = 16

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("result limits must be positive integers")
        if self.max_response_bytes < 2048:
            raise ValueError("max_response_bytes must be at least 2048")
        if not self.max_response_bytes <= self.max_result_bytes <= self.max_retained_bytes:
            raise ValueError("response, individual-result and total retention limits must be ordered")


class ResultReferences:
    def __init__(self, limits: ResultLimits):
        self.limits = limits
        self._directory = None
        self._entries = OrderedDict()
        self._bytes = 0
        self._closed = False

    def put(self, value):
        if self._closed:
            raise GnomonError("SESSION_CLOSED", "Result storage is closed.")
        if self._directory is None:
            self._directory = tempfile.TemporaryDirectory(prefix="gnomon-results-")
        ref = "result_" + uuid4().hex
        path = Path(self._directory.name) / ref
        size, digest = 0, hashlib.sha256()
        try:
            with path.open("xb") as handle:
                path.chmod(0o600)
                encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
                for fragment in encoder.iterencode(value):
                    chunk = fragment.encode("utf-8")
                    size += len(chunk)
                    if size > self.limits.max_result_bytes:
                        raise GnomonError("RESULT_RETENTION_LIMIT", "Operation completed, but its result exceeds the operator retention limit.",
                            {"operation_completed": True, "result_retained": False,
                             "max_result_bytes": self.limits.max_result_bytes,
                             "retry_may_repeat_work": True,
                             **{key: value[key] for key in ("execution_id", "study_id", "recorded")
                                if isinstance(value, dict) and key in value
                                and len(str(value[key])) <= 160}})
                    handle.write(chunk)
                    digest.update(chunk)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        while self._entries and (len(self._entries) >= self.limits.max_results
                                 or self._bytes + size > self.limits.max_retained_bytes):
            _, (old_path, old_size, _) = self._entries.popitem(last=False)
            old_path.unlink()
            self._bytes -= old_size
        self._entries[ref] = (path, size, digest.hexdigest())
        self._bytes += size
        return ref

    def text(self, ref):
        if not isinstance(ref, str) or ref not in self._entries:
            raise GnomonError("RESULT_NOT_RETAINED", "Result reference is unknown, expired or belongs to another session.")
        path, size, digest = self._entries[ref]
        try:
            with path.open("rb") as handle:
                raw = handle.read(self.limits.max_result_bytes + 1)
        except OSError:
            raise GnomonError("RESULT_INTEGRITY", "Retained result is unavailable.") from None
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
            raise GnomonError("RESULT_INTEGRITY", "Retained result failed its integrity check.")
        self._entries.move_to_end(ref)
        return raw.decode("utf-8"), digest

    def value(self, ref):
        return json.loads(self.text(ref)[0])

    def project(self, value):
        size = 0
        for fragment in json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":")).iterencode(value):
            size += len(fragment.encode("utf-8"))
            if size > self.limits.max_response_bytes:
                break
        else:
            return value
        ref = self.put(value)
        summary = {key: value[key] for key in (
            "status", "execution_id", "study_id", "data_ref", "provider", "revision", "evidence", "recorded",
            "series_id", "unit", "horizon", "statistic", "value", "source_as_of", "recorded_as_of")
            if key in value and type(value[key]) in (str, int, float, bool, type(None))
            and len(encode(value[key]).encode("utf-8")) <= 160}
        # A projected answer contains no partial forecast presented as complete.
        projected = {"schema_version": "1", "status": "unscored" if value.get("status") == "unscored" else "result_available", "partial": True,
                     "result_ref": ref, "summary": summary, "action_authorized": False,
                     "retention": "session_lru", "full_result": {"tool": "gnomon_read", "arguments": {"result_ref": ref}}}
        if "error" in value:
            projected["status"] = "error"
            projected["error"] = {"code": "FULL_ERROR_RETAINED", "message": "Read the retained result for complete error details.",
                                  "retryable": False, "repair_options": [{"action": "read_retained_error",
                                      "description": "Use the full_result read call before changing the request or repeating an operation."}]}
        if len(encode(projected).encode("utf-8")) > self.limits.max_response_bytes:
            projected["summary"] = {}
        return projected

    def read(self, result_ref, *, pointer="", offset=0, max_chars=4096):
        if type(offset) is not int or offset < 0 or type(max_chars) is not int or not 1 <= max_chars <= 4096:
            raise GnomonError("INVALID_ARGUMENTS", "offset must be nonnegative and max_chars must be between 1 and 4096.")
        if not isinstance(pointer, str) or len(pointer) > 1024 or (pointer and not pointer.startswith("/")):
            raise GnomonError("INVALID_ARGUMENTS", "pointer must be an empty or slash-prefixed JSON pointer, at most 1024 characters.")
        text, digest = self.text(result_ref)
        if pointer:
            value = json.loads(text)
            for token in pointer[1:].split("/"):
                # Reject malformed escapes rather than silently selecting another key.
                if re.search(r"~(?![01])", token):
                    raise GnomonError("INVALID_ARGUMENTS", "Malformed JSON pointer escape.")
                key = token.replace("~1", "/").replace("~0", "~")
                try:
                    if isinstance(value, list):
                        if not key.isascii() or not key.isdigit() or (len(key) > 1 and key[0] == "0"):
                            raise KeyError(key)
                        value = value[int(key)]
                    elif isinstance(value, dict):
                        value = value[key]
                    else:
                        raise KeyError(key)
                except (KeyError, IndexError, ValueError):
                    raise GnomonError("RESULT_POINTER_NOT_FOUND", "No value exists at that result pointer.") from None
            text = encode(value)
        if offset > len(text):
            raise GnomonError("INVALID_ARGUMENTS", "offset exceeds the selected JSON text length.")

        def page(end):
            return {"schema_version": "1", "status": "ok", "result_ref": result_ref, "pointer": pointer,
                    "encoding": "json_text", "offset_unit": "unicode_codepoints", "offset": offset,
                    "next_offset": end if end < len(text) else None, "total_chars": len(text),
                    "root_sha256": digest, "text": text[offset:end]}

        low, high = offset, min(len(text), offset + max_chars)
        while low < high:
            middle = (low + high + 1) // 2
            if len(encode(page(middle)).encode("utf-8")) <= self.limits.max_response_bytes:
                low = middle
            else:
                high = middle - 1
        if low == offset and offset < len(text):
            raise GnomonError("RESULT_PAGE_LIMIT", "The pointer leaves insufficient space for a result page; use a shorter pointer.")
        result = page(low)
        if len(encode(result).encode("utf-8")) > self.limits.max_response_bytes:
            raise GnomonError("RESULT_PAGE_LIMIT", "Result page metadata exceeds the response limit.")
        return result

    def close(self):
        self._entries.clear()
        self._bytes = 0
        if self._directory is not None:
            self._directory.cleanup()
        self._closed = True
