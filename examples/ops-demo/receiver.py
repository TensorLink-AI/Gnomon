"""Tiny demo telemetry exporter and webhook receiver; never for production."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time


STARTED = time.time()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/metrics":
            self.send_error(404)
            return
        # Rising queue with a small periodic component; crosses 130 after the
        # monitor has enough history to evaluate a short forecast.
        elapsed = time.time() - STARTED
        value = 100 + elapsed * .8 + (int(elapsed) % 5)
        body = f"# TYPE demo_queue_depth gauge\ndemo_queue_depth {value:.3f}\n".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/events":
            self.send_error(404)
            return
        size = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(size))
        destination = Path("/events/events.jsonl")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self.send_response(204)
        self.end_headers()

    def log_message(self, *_):
        return


ThreadingHTTPServer(("0.0.0.0", 9187), Handler).serve_forever()
