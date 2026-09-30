"""Portable argument discovery without weakening canonical validation."""
from copy import deepcopy
from .forecast_adapter import ForecastAdapterError


def _union(specs):
    if all(s == specs[0] for s in specs):
        return deepcopy(specs[0])
    result = {}
    types = []
    for s in specs:
        t = s.get('type')
        if 'const' in s and isinstance(s['const'], str):
            t = 'string'
        for v in t if isinstance(t, list) else [t]:
            if v and v not in types:
                types.append(v)
    if types and all('type' in s or 'const' in s for s in specs):
        result['type'] = types[0] if len(types) == 1 else types
    if all('const' in s or 'enum' in s for s in specs):
        result['enum'] = list(dict.fromkeys(v for s in specs for v in s.get('enum', [s.get('const')])))
    if any('properties' in s for s in specs):
        keys = dict.fromkeys(k for s in specs for k in s.get('properties', {}))
        result['properties'] = {k: _union([s['properties'][k] for s in specs if k in s.get('properties', {})]) for k in keys}
    if all('items' in s for s in specs):
        result['items'] = _union([s['items'] for s in specs])
    result['description'] = 'Requirements vary by operation; retrieve the exact variant through gnomon_capabilities(schema_tool=..., schema_variant=...).'
    return result


def portable_schema(schema):
    """Retain exact branches, adding a permissive union visible to simple hosts."""
    result = deepcopy(schema)
    branches = schema.get('oneOf')
    if not branches:
        return result
    union = _union(branches)
    result['properties'] = union.get('properties', {})
    result['required'] = sorted(set.intersection(*(set(b.get('required', [])) for b in branches)))
    result['additionalProperties'] = False
    variants = []
    for i, b in enumerate(branches):
        label = b.get('properties', {}).get('operation', {}).get('const', str(i))
        variants.append(f"{label}: requires {', '.join(b.get('required', []))}")
    result['description'] = schema.get('description', '') + '\nVariants: ' + '; '.join(variants) + '. Retrieve exact schema through gnomon_capabilities with schema_tool and optional schema_variant.'
    return result


def describe_schema(tools, name, variant=None):
    if not isinstance(name, str) or (variant is not None and not isinstance(variant, str)):
        raise ForecastAdapterError("schema_tool and schema_variant must be strings")
    tool = next((t for t in tools if t['name'] == name), None)
    if tool is None:
        raise ForecastAdapterError('schema_tool must name an exposed tool')
    schema = tool['inputSchema']
    variants = {str(b.get('properties', {}).get('operation', {}).get('const', i)): b
                for i, b in enumerate(schema.get('oneOf', []))}
    if variant is not None and variant not in variants:
        raise ForecastAdapterError('Unknown schema_variant; omit it to list available variants')
    return {'schema_version': '1', 'status': 'ok', 'tool': name,
            'variants': list(variants), 'schema': deepcopy(variants[variant] if variant is not None else schema)}
