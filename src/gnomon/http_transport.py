"""Bounded JSON HTTP transport for operator-configured inference services."""

from __future__ import annotations

import ipaddress
import json
import math
import os
import re
import socket
import time
from urllib import error, parse, request

from .forecast_adapter import ForecastAdapterError


class InferenceHTTPError(ForecastAdapterError):
    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward service credentials or user observations to a redirect.
        return None


class JSONTransport:
    """No implicit POST retry: forecasting can incur cost and create state.

    GETs may retry transient errors. Request and response sizes, socket timeouts
    and retry counts are bounded. Exceptions omit bodies, tokens and URLs.
    URL/auth configuration belongs to the operator, not an agent tool argument.
    """

    def __init__(self, base_url: str, *, token_env: str | None = None,
                 auth_header: str = "Authorization", auth_prefix: str = "Bearer ",
                 timeout: float = 30.0, get_retries: int = 1,
                 max_bytes: int = 8 * 1024 * 1024, allow_http: bool = False):
        parsed = parse.urlsplit(base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ForecastAdapterError("base_url must be an HTTP(S) URL without credentials, query or fragment")
        try:
            local = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            local = parsed.hostname.lower() == "localhost"
        if parsed.scheme == "http" and not (local or allow_http):
            raise ForecastAdapterError("non-loopback HTTP requires explicit allow_http=True")
        if token_env and parsed.scheme == "http" and not local:
            raise ForecastAdapterError("service credentials require HTTPS outside loopback")
        if token_env is not None and (not isinstance(token_env, str) or not token_env):
            raise ForecastAdapterError("token_env must name a nonempty environment variable")
        if (not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", auth_header)
                or any(ch in auth_prefix for ch in "\r\n")
                or auth_header.lower() in {"host", "content-length", "content-type"}):
            raise ForecastAdapterError("invalid authentication header configuration")
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ForecastAdapterError("timeout must be finite and positive")
        if type(get_retries) is not int or not 0 <= get_retries <= 3:
            raise ForecastAdapterError("get_retries must be between 0 and 3")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ForecastAdapterError("max_bytes must be a positive integer")
        self.base_url = base_url.rstrip("/")
        self.token_env, self.auth_header, self.auth_prefix = token_env, auth_header, auth_prefix
        self.timeout, self.get_retries, self.max_bytes = timeout, get_retries, max_bytes
        self._opener = request.build_opener(_NoRedirect())

    def call(self, path: str, payload: dict | None = None) -> dict:
        if not re.fullmatch(r"/[a-zA-Z0-9_/-]*", path) or ".." in path or "//" in path:
            raise ForecastAdapterError("service path must be a fixed relative path")
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.token_env:
            token = os.environ.get(self.token_env)
            if not token or any(ch in token for ch in "\r\n"):
                raise InferenceHTTPError("service authentication environment variable is missing or invalid")
            headers[self.auth_header] = self.auth_prefix + token
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        if body is not None and len(body) > self.max_bytes:
            raise InferenceHTTPError("service request exceeds byte limit")
        method = "GET" if body is None else "POST"
        attempts = self.get_retries + 1 if method == "GET" else 1
        for attempt in range(attempts):
            try:
                req = request.Request(self.base_url + path, data=body, headers=headers, method=method)
                with self._opener.open(req, timeout=self.timeout) as response:
                    raw = response.read(self.max_bytes + 1)
                if len(raw) > self.max_bytes:
                    raise InferenceHTTPError("service response exceeds byte limit")
                try:
                    value = json.loads(raw)
                except (ValueError, UnicodeError):
                    raise InferenceHTTPError("service returned invalid JSON") from None
                if not isinstance(value, dict):
                    raise InferenceHTTPError("service response must be a JSON object")
                return value
            except error.HTTPError as exc:
                status = exc.code
                exc.close()
                if status not in {429, 502, 503, 504} or attempt == attempts - 1:
                    raise InferenceHTTPError(f"service returned HTTP {status}", status=status) from None
            except (error.URLError, socket.timeout, TimeoutError, OSError):
                if attempt == attempts - 1:
                    raise InferenceHTTPError("service connection failed or timed out") from None
            time.sleep(min(0.1 * 2 ** attempt, 1.0))
        raise InferenceHTTPError("service attempts exhausted")
