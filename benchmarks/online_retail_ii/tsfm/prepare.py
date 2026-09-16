"""Augment one chronological task using origin-frozen API calls and matured pairs."""
import json,shutil
from datetime import date,timedelta
from pathlib import Path
import numpy as np
from benchmarks.online_retail_ii.data import dump,sha
from benchmarks.online_retail_ii.full_span.agent import load_case
from benchmarks.online_retail_ii.full_span.models import CANDIDATES as BUILTINS,forecast,metrics
from .client import PROVIDERS,POLICIES,parse_forecast,digest


def prepare_case(source,output,client,revision,store):
    source,output=Path(source),Path(output);task,history=load_case(source)
    if output.exists():raise ValueError('Case already prepared; no silent API rerun')
    output.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,output)
    history_count=len(history);task['candidates']=[*BUILTINS,*PROVIDERS]
    task['tsfm_semantics']={'modes':POLICIES,'point':'API median clipped at zero',
        'training_cutoff':'unknown','scope':'retrospective service augmentation, not a historical availability claim',
        'api_forecasts_reused_across_seeds':True,'feedback_enabled':False}
    for provider in PROVIDERS:
        label=task['case_id']+'-'+provider
        try:
            raw=client.forecast(history.value,label,provider)
            value=parse_forecast(raw,14);available=True;error=None
        except (ValueError,KeyError,TypeError) as exc:
            # No retry and no missing case: a failed TSFM control gets the same
            # disclosed seasonal-naive fallback, but cannot be selected as TSFM.
            value=forecast('seasonal_naive_7',history.value,14,date.fromisoformat(task['origin']))
            value.update(api_meta={});available=False;error=type(exc).__name__
        receipt=json.loads((client.output/label/'receipt.json').read_text())
        frozen={**value,'available':available,'provider':provider,'revision':revision[provider],
            'request_fingerprint':task['request_fingerprint'],'point_sha256':digest(value['point']),
            'api_response_sha256':receipt['response_sha256'],'fallback_used':not available,'error':error,
            'api_seconds':receipt['seconds'],'api_receipt_label':label,'api_request_sha256':receipt['request_sha256']}
        dump(output/(provider+'-forecast.json'),frozen)
        store[task['series_id'],task['origin'],provider]=frozen
        task['model_availability'][provider]={'available':available,'observed':history_count,'required':14,
            'reason':None if available else 'remote_request_failed_or_invalid_response'}
    cv=json.loads((source/'current-cv.json').read_text())
    for provider in PROVIDERS:
        folds=[]
        for offset in (42,28,14):
            if history_count-offset<14:continue
            origin=str(date.fromisoformat(task['origin'])-timedelta(days=offset))
            past=store[task['series_id'],origin,provider]
            target=history.value.to_numpy()[-offset:][:14]
            score=metrics(past['point'],target,history.value.to_numpy()[:-offset])
            folds.append({'origin':origin,'rmsle':score['rmsle'],'fallback_used':past['fallback_used'],
                          'error':past['error'],'api_response_sha256':past['api_response_sha256']})
        eligible=bool(folds) and task['model_availability'][provider]['available'] and not any(f['fallback_used'] for f in folds)
        cv['candidates'][provider]={'folds':folds,'mean_rmsle':float(np.mean([f['rmsle'] for f in folds])) if folds else None,
            'eligible':eligible,'fold_count':len(folds),'model_availability':task['model_availability'][provider]}
    cv['ranking']=sorted((p for p,v in cv['candidates'].items() if v['eligible']),key=lambda p:(cv['candidates'][p]['mean_rmsle'],task['candidates'].index(p)))
    matured=json.loads((source/'matured-outcomes.json').read_text())
    for past in matured['records']:
        if past['target_end']>task['origin']:raise ValueError('Immature actuals')
        past['provider_revisions']={p:'online-retail-ii-full-span-v3/'+past['models_sha256'] for p in BUILTINS}
        for provider in PROVIDERS:
            value=store[task['series_id'],past['origin'],provider]
            past['predictions'][provider]=value['point'];past['fallbacks'][provider]=value['fallback_used']
            past['availability'][provider]={'available':value['available']}
            score=metrics(value['point'],past['actual'],history.value)
            past['scores'][provider]=score['rmsle'] if value['available'] else None
            past['provider_revisions'][provider]=value['revision']
        past['external_api_training_cutoff']='unknown'
    dump(output/'task.json',task);dump(output/'current-cv.json',cv);dump(output/'matured-outcomes.json',matured)
    return task
