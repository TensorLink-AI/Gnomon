"""Validate only the output contract. This script does not forecast or score."""
import json
import math
from pathlib import Path
import sys


def validate(value, horizon):
    if not isinstance(value, dict):
        return 'Expected an object with a point array.'
    points = value.get('point')
    if not isinstance(points, list) or len(points) != horizon:
        return f'point must contain exactly {horizon} forecast values.'
    if any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 for p in points):
        return 'Every point must be finite, numeric and nonnegative.'
    return None


if __name__ == '__main__':
    try:
        task = json.loads(Path('task.json').read_text())
        value = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else 'forecast.json').read_text())
        problem = validate(value, task['horizon'])
    except (OSError, ValueError) as exc:
        problem = f'{type(exc).__name__}: {exc}'
    print(json.dumps({'valid': problem is None, 'cause': problem,
                      'scope': 'Output structure only; no accuracy claim.'}))
    sys.exit(2 if problem else 0)
