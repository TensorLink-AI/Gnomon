import pytest
from gnomon import GnomonSession
from gnomon.models import BASELINES


@pytest.mark.parametrize("name,expected,season", [
    ("last_value", [4, 4, 4], 1),
    ("historical_mean", [3, 3, 3], 1),
    ("seasonal_naive", [2, 4, 2], 2),
])
def test_reference_baselines_have_exact_outputs(name, expected, season):
    with GnomonSession.from_config() as session:
        result = session.call("gnomon_forecast", {"provider": name,
            "request": {"history": [6, 0, 2, 4], "horizon": 3, "season": season}})
        assert list(result["result"]["point"]) == expected
        assert result["evidence"] == "inference_only"
        assert result["action_authorized"] is False


def test_only_three_reference_baselines_are_builtin():
    assert BASELINES == {"last_value", "seasonal_naive", "historical_mean"}
