"""The route-with-gnomon skill's code runs as written."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re

SKILL = Path(__file__).resolve().parents[1] / "skills/route-with-gnomon"


def _python_blocks(path):
    return re.findall(r"```python\n(.*?)\n```", path.read_text(), re.S)


def test_skill_setup_block_builds_a_working_memory_router(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (block,) = _python_blocks(SKILL / "SKILL.md")
    scope = {"__name__": "skill"}
    exec(compile(block, "route-with-gnomon/SKILL.md", "exec"), scope)
    session = scope["session"]
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    history = [50.0 + (i % 7) for i in range(70)]
    request = {"history": history, "horizon": 7, "series_id": "store-1", "unit": "units", "season": 7,
               "timestamps": [(start + timedelta(days=i)).isoformat() for i in range(70)],
               "future_timestamps": [(start + timedelta(days=70 + i)).isoformat() for i in range(7)]}
    routing = session.forecast("router/stores", request)["routing"]
    assert routing["served_provider"] == "seasonal_naive" and routing["reason"] == "insufficient_evidence"
    assert "effective_n" in routing and routing["shadow_execution_ids"]
    session.close()
    scope["base"].close()


def test_skill_replay_reference_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (block,) = _python_blocks(SKILL / "references/replay.md")
    exec(compile(block, "route-with-gnomon/references/replay.md", "exec"), {"__name__": "replay"})
    out = dict(line.split(" ", 1) for line in capsys.readouterr().out.strip().splitlines())
    assert set(out) == {"pooled", "memory"} and "'memory':" in out["memory"]
