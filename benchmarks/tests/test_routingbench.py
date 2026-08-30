from benchmarks.routingbench.run import summarize


def test_summary_fails_closed_on_incomplete_stream():
    rows = {
        "stable_gain": {"complete": False, "decisions": [],
                        "relative_improvement_vs_champion": 0,
                        "routed_error": 0, "always_champion_error": 0,
                        "challenger_routes": 0},
        "gain_then_drift": {"complete": False, "decisions": [],
                            "routed_error": 0, "always_champion_error": 0},
        "mixed_control": {"complete": False, "decisions": [],
                          "routed_error": 0, "always_champion_error": 0,
                          "challenger_routes": 0},
        "regime_and_replay_probes": {
            "complete": False,
            "subdaily": {"recommendation": "last_value", "paired_outcomes": 0},
            "daily_weekly": {"recommendation": "last_value", "paired_outcomes": 0},
            "future_replay_equal": False,
            "unpinned": {"recommendation": "last_value"},
            "deterministic_replay_equal": False,
        },
    }
    report = summarize(rows)
    assert report["gates"]["completion"] is False
    assert report["decision_ready"] is False
