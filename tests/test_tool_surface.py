"""Tool-surface gating and token discipline.

The MCP surface competes for model attention and context tokens; these
tests pin the subtraction: deprecated and near-duplicate tools stay off
the default surface (returning only under GNOMON_V02_COMPAT=1), and no
default response may balloon past the size budget.
"""

from __future__ import annotations


def _names(monkeypatch, compat: bool = False) -> list[str]:
    from gnomon.toolspec import visible_tools

    if compat:
        monkeypatch.setenv("GNOMON_V02_COMPAT", "1")
    else:
        monkeypatch.delenv("GNOMON_V02_COMPAT", raising=False)
    return [tool["name"] for tool in visible_tools()]


# --- Fix 1: the deprecated decision pair is opt-in -------------------------

def test_deprecated_decision_pair_hidden_by_default(monkeypatch) -> None:
    from gnomon.toolspec import runner_for

    names = _names(monkeypatch)
    assert "gnomon_record_decision" not in names
    assert "gnomon_resolve_decision" not in names
    assert runner_for("gnomon_record_decision") is None
    assert runner_for("gnomon_resolve_decision") is None


def test_v02_compat_restores_decision_pair(monkeypatch, tmp_path) -> None:
    from gnomon.toolspec import runner_for
    from gnomon.tracking import TrackingStore

    names = _names(monkeypatch, compat=True)
    assert "gnomon_record_decision" in names
    assert "gnomon_resolve_decision" in names

    # They do not just appear — they still function end to end.
    monkeypatch.setenv("GNOMON_REGISTRY_PATH", str(tmp_path / "registry.db"))
    TrackingStore().register(
        forecast_id="fc_compat", project="compat", selected_model="drift",
        support="supported", artifact_path=str(tmp_path),
    )
    record = runner_for("gnomon_record_decision")({
        "decision_id": "d1", "project": "compat", "forecast_id": "fc_compat",
        "action": "scale_up", "expected_outcome": "load absorbed",
    })
    assert record["status"] == "ok"
    resolved = runner_for("gnomon_resolve_decision")({
        "decision_id": "d1", "actual_outcome": "load absorbed",
        "correct": True,
    })
    assert resolved["status"] == "ok"
    assert resolved["decision"]["correct"] is True


def test_capabilities_reports_compat_state(monkeypatch) -> None:
    from gnomon.runtime import capabilities

    monkeypatch.delenv("GNOMON_V02_COMPAT", raising=False)
    assert capabilities()["compat"]["v02_tools"] is False
    monkeypatch.setenv("GNOMON_V02_COMPAT", "1")
    assert capabilities()["compat"]["v02_tools"] is True
