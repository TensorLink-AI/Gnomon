"""Prefer mature evidence from this series; borrow other series when sparse."""
from datetime import datetime
from .lifetime_context import retrieve as contextual


def retrieve(current,episodes):
    if not isinstance(current.get('series_id'),str) or not current['series_id']:raise ValueError('Exact current series identity required')
    result=contextual(current,episodes)
    if result['ready']:
        result['selected']=sorted(result['candidates'],key=lambda r:(r['series_id']!=current['series_id'],r['distance'],r['origin'],r['series_id']))[:16]
    recent=set(sorted({datetime.fromisoformat(r['origin']) for r in result['candidates']})[-8:])
    result['selected_outside_recent_eight']=sum(datetime.fromisoformat(r['origin']) not in recent for r in result['selected'])
    result.update(retrieval_policy='same_series_then_context_distance',same_series_candidates=sum(r['series_id']==current['series_id'] for r in result['candidates']),
        selected_same_series=sum(r['series_id']==current['series_id'] for r in result['selected']))
    return result
