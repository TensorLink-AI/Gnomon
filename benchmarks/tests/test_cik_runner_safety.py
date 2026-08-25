import json

from benchmarks.cik.run_cik import _load_checkpoint


def test_resume_retries_provider_and_process_failures_but_keeps_model_results(tmp_path):
    payload = {
        "valid": {"name": "Task", "row": {"seed": 1, "score": 0.2}},
        "provider": {"name": "Task", "row": {
            "seed": 2, "error": "OpenRouter returned HTTP 403: daily limit"}},
        "timeout": {"name": "Task", "row": {
            "seed": 3, "error": "case_timeout_after_900s"}},
        "model": {"name": "Task", "row": {
            "seed": 4, "error": "could not parse any valid forecast"}},
    }
    (tmp_path / "case-checkpoint.json").write_text(json.dumps(payload))
    loaded = _load_checkpoint(tmp_path)
    assert set(loaded) == {"valid", "model"}
