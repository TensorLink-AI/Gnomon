"""Run with python -m gnomon.examples.custom_provider. No external dependencies."""
from gnomon import ForecastResult, InferenceEngine


def main():
    with InferenceEngine(cache_size=8) as engine:
        engine.register('custom', lambda request: ForecastResult(
            point=(request.history[-1],) * request.horizon,
            series_id=request.series_id, unit=request.unit,
            timestamps=request.future_timestamps), revision='custom-v1', deterministic=True)
        request = dict(history=[1, 2, 3], horizon=2, unit='widgets', series_id='sales',
                       future_timestamps=['2026-01-04T00:00:00Z', '2026-01-05T00:00:00Z'])
        first, second = engine.forecast('custom', request), engine.forecast('custom', request)
        assert not first.cache_hit and second.cache_hit
        assert second.result.point == (3, 3) and second.result.unit == 'widgets'
        print(second.to_dict())


if __name__ == '__main__':
    main()
