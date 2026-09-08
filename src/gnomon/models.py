"""Three dependency-free reference baselines; users own other model software."""

from statistics import mean
from .forecast_adapter import ForecastAdapterError

def last_value(history: list[float], horizon: int, season: int) -> list[float]:
    return [history[-1]] * horizon


def seasonal_naive(history: list[float], horizon: int, season: int) -> list[float]:
    if len(history) < season:
        raise ForecastAdapterError(
            f"insufficient seasonal history: season={season} requires {season} observations; observed={len(history)}. "
            "Supply more observed history; repair modes cannot create seasonal evidence.",
            details={"required_history": season, "observed_history": len(history), "season": season,
                     "horizon": horizon, "guidance": "Keep the intended season and supply at least season observed values."})
    return [history[-season + (index % season)] for index in range(horizon)]


def historical_mean(history: list[float], horizon: int, season: int) -> list[float]:
    """Repeat the expanding-prefix level as an assumption-light baseline."""
    if not history:
        raise ValueError("insufficient history")
    return [mean(history)] * horizon


MODELS = {"last_value": last_value, "seasonal_naive": seasonal_naive,
          "historical_mean": historical_mean}
BASELINES = frozenset(MODELS)


def predict(name: str, history: list[float], horizon: int, season: int) -> list[float]:
    if name not in MODELS:
        raise ValueError(f"Unknown baseline {name!r}; register your model with InferenceEngine.")
    return MODELS[name](history, horizon, season)
