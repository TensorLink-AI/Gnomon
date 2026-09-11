"""examples/agent_cheating_demo.sh runs offline, quickly, and shows the rescoring guarantees."""

import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "examples" / "agent_cheating_demo.sh"


def test_demo_runs_offline_in_under_thirty_seconds(tmp_path):
    env = {**os.environ, "PATH": os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")]),
           "GNOMON_DEMO_WORK": str(tmp_path / "work")}
    for name in ("EPHEMERIS_BASE_URL", "EPHEMERIS_API_TOKEN"):
        env.pop(name, None)
    started = time.monotonic()
    completed = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, env=env, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert time.monotonic() - started < 30
    out = completed.stdout
    assert "excluded_by_source_cutoff: 49" in out
    assert "status: complete  replay: recorded" in out
    assert "rows_added: 1  revisions_created: 1" in out
    assert "provider_calls: 0  predictions_reused_exactly: True  original_unchanged: True" in out
    assert "changed actuals: 1 fold(s)" in out
    assert (tmp_path / "work" / "compare.json").is_file()


def test_demo_needs_no_network_or_credentials():
    text = SCRIPT.read_text()
    assert "EPHEMERIS" not in text and "http" not in text
    assert os.access(SCRIPT, os.X_OK)
