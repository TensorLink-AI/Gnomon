"""Pure reserved-panel construction, exercised only with synthetic input rows.

No file/network access or CLI. The future operational caller must authenticate
the original manifest and pass the final access gate BEFORE loading sales rows.
Structural validation here is not authorization to access reserved data.
"""
from collections import Counter
from datetime import timedelta

from . import m5_ml_adapter as shared


def validate_panel(manifest):
    """Validate the declared 8+24 disjoint panel, without consuming sales rows."""
    splits = manifest['splits']
    expected = {'development': (8, 2, 4), 'reserved': (24, 8, 3)}
    for split, (count, stores, per_store) in expected.items():
        selected = splits[split]
        if len(selected) != count:
            raise ValueError('Wrong fixed panel size: '+split)
        for item in selected:
            for field in ('store_id', 'item_id', 'series_id', 'initial_history_sha256'):
                if not isinstance(item.get(field), str) or not item[field]:
                    raise ValueError('Missing explicit panel identity or selection hash: '+field)
            digest = item['initial_history_sha256']
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Invalid selection-prefix hash')
            if item['series_id'] == '__default__':
                raise ValueError('Explicit series identity required')
        sizes = Counter(item['store_id'] for item in selected)
        if len(sizes) != stores or set(sizes.values()) != {per_store}:
            raise ValueError('Wrong fixed store/item balance: '+split)
        for field in ('item_id', 'series_id'):
            if len({item[field] for item in selected}) != count:
                raise ValueError('Duplicate panel identity: '+field)
    for field in ('store_id', 'item_id', 'series_id'):
        if {i[field] for i in splits['development']} & {i[field] for i in splits['reserved']}:
            raise ValueError('Development/reserved overlap: '+field)
    return {(item['store_id'], item['item_id']): dict(item) for item in splits['reserved']}


def build_reserved_jobs(source_rows, calendar, manifest):
    """Construct 624 host tasks from already-loaded rows using the shared builder.

    No reselection, padding, repaired history, altered origins, model calls, or
    mutation of input objects. A list/tuple is required deliberately: this helper
    must not be handed a lazy archive reader that performs hidden file access.
    The caller, not this structural validator, owns the final access gate.
    """
    identities = validate_panel(manifest)
    if not isinstance(source_rows, (list, tuple)):
        raise ValueError('Already-loaded rows required; no lazy archive readers')
    days = [calendar[f'd_{i}'] for i in range(shared.FIRST, shared.LAST+1)]
    if any(b-a != timedelta(days=1) for a, b in zip(days, days[1:])):
        raise ValueError('Calendar must be complete and consecutive')
    stamps = [shared.period_end(day) for day in days]
    result = {}
    for row in source_rows:
        identity = row['store_id'], row['item_id']
        if identity not in identities:
            continue
        metadata = identities[identity]; series = metadata['series_id']
        if series in result:
            raise ValueError('Duplicate selected source row: '+series)
        try:
            raw = [row[f'd_{i}'] for i in range(shared.FIRST, shared.LAST+1)]
            if any(isinstance(value, bool) for value in raw):
                raise ValueError('Boolean sales observation')
            values = [float(value) for value in raw]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Invalid or missing selected observations; no replacement: '+series) from exc
        result[series] = shared.build_series_jobs(metadata, values, stamps)
    if set(result) != {item['series_id'] for item in identities.values()}:
        raise ValueError('Missing selected source series; no replacement')
    if sum(map(len, result.values())) != 624:
        raise ValueError('Incomplete fixed panel task grid')
    return {series: result[series] for series in sorted(result)}
