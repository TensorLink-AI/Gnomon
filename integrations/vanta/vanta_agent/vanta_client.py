"""Client for the Vanta miner REST server (vanta-network/vanta_api/miner_rest_server.py).

The miner process runs the server on :8088 and relays each order to validators;
`/api/submit-order` is synchronous (typically 20-60 s). The server honours a
client-supplied `order_uuid`, so the agent derives it deterministically from the
decision: after a crash or timeout, `/api/order-status/<uuid>` tells whether that
exact order was processed, and the same order is never sent under a new id.
"""
from dataclasses import dataclass
import json
import os
import urllib.error
import urllib.request
import uuid

# Fixed namespace so order ids are reproducible across restarts and machines.
ORDER_NAMESPACE = uuid.UUID("6f1b8f2e-3c2a-5d84-9a51-7e0c4a1d2b90")


def order_uuid(client_order_id: str, index: int) -> str:
    return str(uuid.uuid5(ORDER_NAMESPACE, f"{client_order_id}:{index}"))


@dataclass(frozen=True)
class Order:
    pair: str
    order_type: str  # LONG | SHORT | FLAT
    leverage: float | None  # None for FLAT

    def payload(self, uuid_: str) -> dict:
        body = {"execution_type": "MARKET", "trade_pair": self.pair,
                "order_type": self.order_type, "order_uuid": uuid_}
        if self.order_type != "FLAT":
            body["leverage"] = round(self.leverage, 6)
        return body


class VantaError(RuntimeError):
    """The outcome of a submission is known to be a rejection."""


class VantaUncertain(RuntimeError):
    """The outcome is unknown (timeout, connection loss): reconcile before retrying."""


class VantaClient:
    def __init__(self, url, *, api_key, timeout=120.0, opener=urllib.request.urlopen):
        self.url, self.api_key, self.timeout, self.opener = url.rstrip("/"), api_key, timeout, opener

    @classmethod
    def from_config(cls, config, opener=urllib.request.urlopen):
        key = os.environ.get(config.vanta.api_key_env)
        if not key:
            raise RuntimeError(f"Set {config.vanta.api_key_env} to the key in vanta_api/api_keys.json")
        return cls(config.vanta.url, api_key=key, timeout=config.vanta.timeout_seconds, opener=opener)

    def _call(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.url + path, data=data, method=method, headers={
            "Content-Type": "application/json", "Authorization": self.api_key})
        try:
            with self.opener(request, timeout=self.timeout) as response:
                return response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            try:
                return error.code, json.loads(error.read() or b"{}")
            except ValueError:
                return error.code, {"error": str(error)}

    def health(self) -> bool:
        try:
            status, _ = self._call("GET", "/api/health")
            return status == 200
        except OSError:
            return False

    def submit(self, order: Order, uuid_: str) -> dict:
        try:
            status, body = self._call("POST", "/api/submit-order", order.payload(uuid_))
        except (OSError, TimeoutError) as error:
            raise VantaUncertain(f"{order.pair} {uuid_}: {error}") from error
        if status == 200 and body.get("success"):
            return body
        if status >= 500:
            raise VantaUncertain(f"{order.pair} {uuid_}: HTTP {status} {body}")
        raise VantaError(f"{order.pair} {uuid_}: HTTP {status} {body}")

    def status(self, uuid_: str) -> str:
        """completed | failed | not_found (raises VantaUncertain if unreachable)."""
        try:
            code, body = self._call("GET", f"/api/order-status/{uuid_}")
        except (OSError, TimeoutError) as error:
            raise VantaUncertain(str(error)) from error
        if code not in (200, 404):
            raise VantaUncertain(f"order-status HTTP {code} {body}")
        return body.get("status", "not_found")
