"""Temporal and comparison invariants for the CPU evaluation, without network or paid calls."""
import copy

import pytest

np=pytest.importorskip('numpy')

from benchmarks.cpu_eval.agent import cases, prepare
from benchmarks.cpu_eval.analyze import block_interval, policy, analyze
from benchmarks.cpu_eval.common import PROTOCOL, digest, identity, read, write
from benchmarks.cpu_eval.data import forward_fill, origins
from benchmarks.cpu_eval.models import forecast
from benchmarks.cpu_eval.report import report
from benchmarks.cpu_eval.run import validate_manifest
from benchmarks.workflow.schema import Case


def test_origins_and_causal_missing_fill():
    assert origins(20,4,2,3,8)==[12,14,16]
    assert origins(10,4,2,3,8)==[]
    assert forward_fill([None,3,None,100])==[0.,3.,3.,100.]
    assert forward_fill([None,3,None])==[0.,3.,3.]


def test_write_is_immutable(tmp_path):
    path=tmp_path/'record.json'
    write(path,{'a':1})
    write(path,{'a':1})
    with pytest.raises(ValueError,match='replace'):
        write(path,{'a':2})


def test_manifest_tampering():
    manifest={'mode':'pilot','values':[1,2]}
    manifest['sha256']=digest(manifest)
    validate_manifest(manifest)
    manifest['values'][1]=9
    with pytest.raises(ValueError,match='hash'):
        validate_manifest(manifest)


def test_bounded_changes_distance_only():
    m={'protocol':read(PROTOCOL),'spec':{'season':7},'series':[{'id':'x'}]}
    current,bounded=policy(m,'current'),policy(m,'bounded')
    bounded['memory']['distance']='robust'
    assert current==bounded
    assert policy(m,'fase')['memory']['retention']=='fase'
    assert not policy(m,'no_memory').get('memory')


def test_block_interval_preserves_paired_signal():
    assert block_interval([[1]*20,[3]*20],5,100,0)==[2.,2.]
    assert block_interval([[1]*2],5,100,0) is None


def test_synthetic_agent_cases_and_control(tmp_path):
    cohort=cases()
    assert len(cohort)==12 and len({c['id'] for c in cohort})==12
    assert sum(bool(c.get('episode')) for c in cohort)==3
    for c in cohort:
        Case.from_dict(c)
    prepare(tmp_path)
    providers=read(tmp_path/'providers.example.json')
    assert 'execution_options' not in providers['backends']['lean']['options']
    assert '--allow-model-requests' not in read(tmp_path/'experiment.example.json')['command']
    assert read(tmp_path/'cohort.json')['scored_episodes']==108


def test_baselines():
    assert list(forecast('seasonal_naive',[1,2,3,4],5,2,0))==[3,4,3,4,3]
    assert list(forecast('drift',[1,2,3],2,1,0))==[4,5]


def fixture_run(root, monkeypatch):
    from datetime import datetime,timedelta,timezone
    protocol=copy.deepcopy(read(PROTOCOL))
    protocol.update(models=['last_value','seasonal_naive'], validation_origins=8,warmup_origins=8,
                    pilot_score_origins=2, max_history=32)
    stamps=[(datetime(2020,1,1,tzinfo=timezone.utc)+timedelta(days=i)).isoformat() for i in range(60)]
    values=list(range(60))
    positions=list(range(30,48))
    phases=['validation']*8+['warmup']*8+['score']*2
    manifest={'mode':'pilot','protocol':protocol,'spec':{'id':'fixture','horizon':1,'season':1,'stride':1},
              'series':[{'id':'x','values':values,'timestamps':stamps,'origins':positions,'phases':phases,'mase_scale':1.}]}
    manifest['sha256']=digest(manifest)
    write(root/'manifest.json',manifest)
    pinned={'runtime':identity(),'manifest_sha256':manifest['sha256']}
    write(root/'identity.json',pinned)
    for i,phase in zip(positions,phases):
        for name in protocol['models']:
            plan={'series':'x','origin_index':i,'phase':phase,'model':name}
            stem=root/'calls'/digest(['x',i,name])
            write(str(stem)+'.request.json',plan)
            # Validation prefers last_value; the holdout prefers seasonal_naive.
            point=values[i] + (1 if phase!='validation' and name=='last_value' else 2 if phase=='validation' and name=='seasonal_naive' else 0)
            write(str(stem)+'.result.json',{'request_sha256':digest(plan),'status':'ok','point':[point]})
    receipts={p.name:digest(read(p)) for p in (root/'calls').glob('*.json')}
    write(root/'COMPLETE.json',{'identity':digest(pinned),'receipts':receipts,'dispatch_seconds':0.})
    return manifest


def test_fixed_model_never_selected_using_test_outcomes(tmp_path,monkeypatch):
    fixture_run(tmp_path,monkeypatch)
    summary=analyze(tmp_path)
    assert summary['fixed_validation_model']=='last_value'
    assert summary['scores_mase']['fixed_validation']==1.
    assert summary['expected_scored_folds']==2
    rows=read(tmp_path/'decisions.json')
    assert len(rows)==12  # six arms on the same two scored origins, no warmup scores


def test_receipt_tampering_is_rejected(tmp_path,monkeypatch):
    fixture_run(tmp_path,monkeypatch)
    target=next((tmp_path/'calls').glob('*.result.json'))
    target.write_text('{}')
    with pytest.raises(ValueError,match='Receipt'):
        analyze(tmp_path)


def test_missing_configuration_cannot_make_macro_score(tmp_path):
    result=report(tmp_path,'evaluation')
    assert not result['complete'] and not result['macro_mase']
    assert len(result['missing_configurations'])==6


def test_worker_timeout_is_bounded_and_next_call_can_recover():
    from benchmarks.cpu_eval.run import CPUWorker
    request={'name':'last_value','history':[1.,2.,3.],'horizon':2,'season':1,'seed':0}
    with CPUWorker() as cpu:
        assert cpu.predict(request,0)['status']=='timeout'
        assert cpu.process is None
        result=cpu.predict(request,10)
        assert result['status']=='ok' and result['point']==[3.,3.]


def test_scored_stage_cannot_launch_without_successful_pilot(tmp_path):
    from benchmarks.cpu_eval.suite import execute
    with pytest.raises(ValueError,match='requires --pilot'):
        execute(tmp_path/'out',tmp_path/'cache','evaluation')
    assert not (tmp_path/'out').exists()


def test_prepared_pilot_series_are_reserved_from_score(tmp_path):
    pa=pytest.importorskip('pyarrow')
    from datetime import datetime
    from benchmarks.cpu_eval.data import prepare as prepare_data
    proto=copy.deepcopy(read(PROTOCOL))
    proto.update(pilot_series=1,series_limit=2,score_origins=8)
    spec={'id':'electricity/D','season':1,'horizon':2,'stride':1}
    data=pa.Table.from_pylist([{'item_id':str(i),'target':list(range(100)),
                             'start':datetime(2020,1,1),'freq':'D'} for i in range(3)])
    path=tmp_path/'source.arrow'
    with pa.OSFile(str(path),'wb') as sink:
        with pa.ipc.new_stream(sink,data.schema) as writer:
            writer.write_table(data)
    pilot=prepare_data(proto,spec,path,{},'pilot')
    scored=prepare_data(proto,spec,path,{},'evaluation')
    assert not {s['id'] for s in pilot['series']} & {s['id'] for s in scored['series']}


def test_fixed_selection_scale_cannot_see_later_series_history():
    from benchmarks.cpu_eval.analyze import validation_scale
    series={'timestamps':['2026-01-01T00:00:00Z','2026-01-02T00:00:00Z','2026-01-03T00:00:00Z'],
            'values':[1.,3.,100.]}
    cutoff='2026-01-02T00:00:00Z'
    assert validation_scale(series,cutoff,1)==2.
    series['values'][2]=-1000000.
    assert validation_scale(series,cutoff,1)==2.
