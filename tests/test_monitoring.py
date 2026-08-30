import json

from gnomon.monitoring import (
    default_state_path, deliver_events, firing_events, prometheus_rule,
)


def test_default_state_uses_writable_artifact_directory(tmp_path):
    assert default_state_path(tmp_path / "out") == tmp_path / "out" / ".monitor-events.json"


def test_event_delivery_records_before_side_effect_and_is_idempotent(tmp_path, monkeypatch):
    payload = {"monitor_id": "m1", "created_at": "2026-01-01T00:00:00Z",
               "triggers": [{"series": "api", "armed": True,
                             "first_alert_step": 2,
                             "first_alert_timestamp": "2026-01-02",
                             "trigger": {"threshold": 10}}]}
    events = firing_events(payload, "/artifact/m1")
    calls = []

    class Response:
        status = 204
        def __enter__(self): return self
        def __exit__(self, *_): return False

    monkeypatch.setattr("urllib.request.urlopen",
                        lambda request, timeout: calls.append(request) or Response())
    state = tmp_path / "state.json"
    first = deliver_events(events, state_path=state, webhook="https://hooks.example/x")
    second = deliver_events(events, state_path=state, webhook="https://hooks.example/x")
    assert len(first["delivered"]) == 1 and len(calls) == 1
    assert second["already_recorded"] == first["delivered"]
    persisted = json.loads(state.read_text())
    assert persisted["events"][events[0]["event_id"]]["status"] == "delivered"


def test_prometheus_rule_is_dependency_free_block_yaml(tmp_path):
    path = prometheus_rule(monitor_id="monitor-1", expression="rate(http_requests_total[5m])",
                           threshold=12.5, output=tmp_path / "rule.yml")
    rendered = path.read_text()
    assert rendered.startswith("groups:\n")
    assert '      - alert: "Gnomon_monitor_1"' in rendered
    assert '        expr: "(rate(http_requests_total[5m])) > 12.5"' in rendered
    assert 'monitor_id: "monitor-1"' in rendered


def test_failed_delivery_retries_and_can_resume(tmp_path, monkeypatch):
    event = {"event_id": "e1", "kind": "test"}
    attempts = []
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *_args, **_kwargs: attempts.append(1) or (_ for _ in ()).throw(OSError("down")))
    state = tmp_path / "state.json"
    failed = deliver_events([event], state_path=state,
                            webhook="https://hooks.example/x", max_attempts=2)
    assert failed["delivery_failed"] == ["e1"] and len(attempts) == 2

    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *_): return False
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    resumed = deliver_events([event], state_path=state,
                             webhook="https://hooks.example/x")
    assert resumed["delivered"] == ["e1"]
