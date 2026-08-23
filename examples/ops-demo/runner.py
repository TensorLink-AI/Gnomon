"""One cron-shaped demo monitor execution after Prometheus has a history."""

from datetime import datetime, timedelta, timezone
import subprocess
import time
import urllib.parse
import urllib.request


for _ in range(90):
    try:
        urllib.request.urlopen("http://prometheus:9090/-/ready", timeout=2).close()
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("Prometheus did not become ready")

# Accumulate enough one-second samples for separated evaluation folds.
time.sleep(45)
end = datetime.now(timezone.utc)
start = end - timedelta(seconds=42)
query = urllib.parse.urlencode({
    "query": "demo_queue_depth", "start": start.timestamp(),
    "end": end.timestamp(), "step": "1s",
})
source = "prom://prometheus:9090/api/v1/query_range?" + query
command = [
    "gnomon", "monitor", "run", source,
    "--time", "timestamp", "--target", "value", "--series", "series",
    "--frequency", "1s", "--horizon", "5", "--threshold", "130",
    "--alert-cost", "1", "--miss-cost", "20", "--output", "/output",
    "--state", "/state/events.json", "--webhook", "http://telemetry:9187/events",
    "--prometheus-expression", "demo_queue_depth",
    "--prometheus-rule-output", "/output/gnomon-rule.yml",
]
raise SystemExit(subprocess.run(command, check=False).returncode)
