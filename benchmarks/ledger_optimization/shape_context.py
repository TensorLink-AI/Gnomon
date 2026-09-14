"""Observed calendar-shape descriptors for causally filtered evidence retrieval."""
from datetime import datetime,timedelta
import hashlib,json
import numpy as np
from .lifetime_context import retrieve as coarse


def profile(history,origin):
    at=datetime.fromisoformat(origin);values=np.asarray(history,dtype=float)
    if at.tzinfo is None or values.shape!=(730,) or not np.isfinite(values).all() or (values<0).any():raise ValueError('730finite nonnegative observed hours and aware origin required')
    log=np.log1p(values[-672:]);center=float(log.mean());scale=max(.1,float(log.std()));labels=[at-timedelta(hours=672-i) for i in range(672)]
    hourly=[float(log[[t.hour==h for t in labels]].mean()) for h in range(24)]
    weekly=[float(log[[t.weekday()==d for t in labels]].mean()) for d in range(7)]
    vector=[(v-center)/scale for v in hourly+weekly]
    return {'vector':vector,'center':center,'scale':scale,'observations':672,'hour_counts':[28]*24,'weekday_counts':[96]*7,
        'history_start':labels[0].isoformat(),'history_end':labels[-1].isoformat(),'origin':origin,
        'input_sha256':hashlib.sha256(json.dumps({'history':history,'origin':origin},separators=(',',':')).encode()).hexdigest()}


def retrieve(current,episodes,get_profile):
    # Coarse retrieval performs all maturity/domain validation before profile access.
    result=coarse(current,episodes);q=get_profile(current);candidates=result['candidates'];result.update(retrieval_policy='coarse_plus_observed_calendar_shape',
        query_shape=q,shape_location=None,shape_scale=None,shape_dimensions=31,coarse_dimensions=12)
    if candidates:
        p=[get_profile(r) for r in candidates];x=np.asarray([r['vector'] for r in p]);scale=np.maximum(.1,x.std(axis=0));distances=np.sum(((x-q['vector'])/scale)**2,axis=1)
        result.update(shape_location=x.mean(axis=0).tolist(),shape_scale=scale.tolist())
        result['candidates']=[{**c,'coarse_distance':c['distance'],'shape_distance':float(d),'distance':c['distance']/12+float(d)/31,'shape_input_sha256':v['input_sha256']}
            for c,d,v in zip(candidates,distances,p,strict=True)]
        if result['ready']:result['selected']=sorted(result['candidates'],key=lambda r:(r['distance'],r['origin'],r['series_id']))[:16]
        recent=set(sorted({r['origin'] for r in result['candidates']})[-8:]);result['selected_outside_recent_eight']=sum(r['origin'] not in recent for r in result['selected'])
    return result
