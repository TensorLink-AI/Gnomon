"""Execution, checkpoint, budget, concurrency and visibility regression checks."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from .run import HERE, OTHER, ARMS, prepare, records, environment, assess, runtime_inventory
from .transport import dump, sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    root=parser.parse_args().output.resolve();root.mkdir(parents=True,exist_ok=False)
    frozen={p.name:sha(p) for p in HERE.iterdir() if p.is_file()}
    initial_inventory=runtime_inventory()
    dump(root/'initial-runtime-inventory.json',initial_inventory)
    jobs=next(iter(json.loads((OTHER/'setup/recovered-task-source/host-jobs.json').read_text()).values()))
    job=jobs[0];checks=[];results={};base_config={'model':'ridge','window':180,'lags':14,'alpha':10}
    for arm in ARMS:
        work=root/arm;work.mkdir();home=work/'home';home.mkdir()
        python=OTHER/('plain-venv' if arm=='plain' else 'gnomon-venv')/'bin/python'
        prepare(work,job,[],arm);env=environment(home,work,python)

        def invoke(*args,expect=0,where=work):
            result=subprocess.run([str(python),'lab.py',*args],cwd=where,env=env,capture_output=True,text=True)
            with (root/'commands.jsonl').open('a') as f:f.write(json.dumps({'arm':arm,'cwd':str(where),'argv':args,'code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})+'\n')
            assert result.returncode==expect,(args,result.stdout,result.stderr)
            return json.loads(result.stdout)

        def grade():
            protected={n:sha(work/n) for n in ('task.json','history.csv','future.csv','previous_runs.json','lab.py','core.py','numerical.py','backend.json','maturation.py')}
            return assess(work,job,protected,b'')

        def clone(name):
            dest=root/(arm+'-'+name);shutil.copytree(work,dest);return dest

        def charge(where, target):
            existing=sum(r['event']=='attempt' and r['task_origin']==job['origin'] for r in records(where))
            with (where/'experiments.jsonl').open('a') as f:
                for i in range(target-existing):
                    f.write(json.dumps({'event':'attempt','task_origin':job['origin'],
                                        'attempt_id':'synthetic-budget-test-'+str(i),'test_fixture':True})+'\n')

        rejected=invoke('backtest','--config',json.dumps(base_config),expect=2)
        assert rejected['numerical_calls_started']==0
        started=invoke('start');assert started['numerical_calls_started']==4
        baseline=(work/'checkpoint.json').read_bytes()
        a=grade();assert a['valid'] and a['baseline_checkpoint_only'] and not a['workflow_complete']
        assert invoke('start')['numerical_calls_started']==0
        invoke('backtest','--config',json.dumps(base_config))
        assert (work/'checkpoint.json').read_bytes()==baseline
        assert not grade()['workflow_complete'], 'Testing alone is not explicit selection after comparison'
        assert invoke('backtest','--config',json.dumps(base_config))['numerical_calls_started']==0
        checks.append(arm+': baseline-first, idempotence, explicit post-comparison selection required')

        # With exactly one numerical slot left, a whole batch is rejected and final fit succeeds.
        scarce=clone('scarce');charge(scarce,59)
        before=(scarce/'experiments.jsonl').read_bytes()
        denied=invoke('backtest','--config',json.dumps({**base_config,'alpha':12}),expect=2,where=scarce)
        assert denied['numerical_calls_started']==0 and (scarce/'experiments.jsonl').read_bytes()==before
        assert (scarce/'checkpoint.json').read_bytes()==baseline
        last=invoke('commit','--config',json.dumps(base_config),where=scarce)
        assert last['numerical_calls_started']==1 and last['budget']['numerical_remaining']==0
        assert invoke('commit','--config',json.dumps(base_config),where=scarce)['numerical_calls_started']==0
        checks.append(arm+': atomic batch refusal, final-fit reserve, zero-fit reselection at exhausted budget')

        invoke('commit','--config',json.dumps(base_config))
        selected=(work/'checkpoint.json').read_bytes();a=grade();assert a['workflow_complete']
        results[arm]=[r for r in records(work) if r['event']=='result']
        assert len(results[arm])==8
        denied=invoke('commit','--execution-id','not-an-execution',expect=2)
        assert denied['numerical_calls_started']==0 and (work/'checkpoint.json').read_bytes()==selected
        invoke('commit','--config','{"model":"ridge","lags":999}',expect=2)
        assert (work/'checkpoint.json').read_bytes()==selected
        checks.append(arm+': valid ML commit, unknown/invalid selection leaves checkpoint intact')

        # Reserve agent requests as well as the numerical final fit. Phase rejection
        # starts no work; an explicit final commit and exact retrieval still work.
        selection=clone('selection-phase')
        import time
        dump(selection/'agent-budget.json',{'forwarded_requests':13,'remaining_requests':3,
                                            'deadline_epoch':time.time()+480})
        before=(selection/'experiments.jsonl').read_bytes()
        denied=invoke('backtest','--config',json.dumps({**base_config,'alpha':14}),expect=2,where=selection)
        assert denied['error']['code']=='SELECTION_PHASE_RESERVED' and denied['numerical_calls_started']==0
        assert (selection/'experiments.jsonl').read_bytes()==before
        assert invoke('backtest','--config',json.dumps(base_config),where=selection)['numerical_calls_started']==0
        assert invoke('commit','--config',json.dumps(base_config),where=selection)['result']['selection_after_comparison']
        dump(selection/'agent-budget.json',{'forwarded_requests':2,'remaining_requests':14,
                                            'deadline_epoch':time.time()+89})
        assert invoke('backtest','--config',json.dumps({**base_config,'alpha':15}),expect=2,where=selection)['error']['code']=='SELECTION_PHASE_RESERVED'
        checks.append(arm+': protected selection requests/time reject new exploration but allow explicit commit')

        # Two simultaneous callers have room for only one complete batch plus final reserve.
        concurrent=clone('concurrent');charge(concurrent,55)
        calls=[subprocess.Popen([str(python),'lab.py','backtest','--config',json.dumps({**base_config,'alpha':alpha})],
                               cwd=concurrent,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
               for alpha in (11,12)]
        responses=[p.communicate(timeout=90) for p in calls]
        assert sorted(p.returncode for p in calls)==[0,2],responses
        assert sum(r['event']=='attempt' for r in records(concurrent))==58
        assert (concurrent/'checkpoint.json').read_bytes()==selected
        checks.append(arm+': concurrent callers cannot over-admit folds or overwrite checkpoint')

        # The public final ID must resolve uniquely, even when the provider/config is identical.
        ambiguous=clone('ambiguous')
        code='import core; core.execute('+repr(base_config)+',730,"forecast")'
        subprocess.run([str(python),'-c',code],cwd=ambiguous,env=env,check=True,capture_output=True)
        rejected=invoke('commit','--config',json.dumps(base_config),expect=2,where=ambiguous)
        assert rejected['error']['code']=='AMBIGUOUS_EXECUTIONS'
        ids=rejected['error']['details']['available_execution_ids'];assert len(ids)==2
        invoke('commit','--execution-id',ids[1],where=ambiguous)
        checks.append(arm+': duplicate configuration executions require explicit execution_id')

        if arm=='plain':
            # Failed provider attempt is charged, but cannot replace the valid checkpoint.
            failure=clone('provider-failure')
            invoke('backtest','--config',json.dumps({**base_config,'alpha':13}),where=failure)
            code='import sys,core,lab\ncore.predict=lambda *a: (_ for _ in ()).throw(RuntimeError("synthetic fit failure"))\nsys.argv=["lab.py","commit","--config",'+repr(json.dumps({**base_config,'alpha':13}))+']\nlab.main()'
            result=subprocess.run([str(python),'-c',code],cwd=failure,env=env,capture_output=True,text=True)
            assert result.returncode==2 and json.loads(result.stdout)['numerical_calls_started']==1
            assert (failure/'checkpoint.json').read_bytes()==selected
            checks.append('failed numerical fit is charged and preserves prior published checkpoint')

        # Mature every eligible production execution, with identical facts in all arms.
        eid=a['execution_id'];before_scores=None
        if arm=='ledger':before_scores=[r['ledger_score'] for r in results[arm] if r['kind']=='backtest']
        prior=[{'origin':job['origin'],'outcome_recorded_at':job['outcome_recorded_at'],
                'execution_id':eid,'actual':job['actual'],'future_timestamps':job['future_timestamps'],
                'series_id':job['series_id'],'unit':job['request']['unit']}]
        prepare(work,jobs[1],prior,arm)
        first_sync=invoke('sync');before_retry=(work/'experiments.jsonl').read_bytes()
        assert len(first_sync['result']['matured_execution_ids'])==2
        assert first_sync['result']['provider_calls']==0
        assert invoke('sync')['result']['matured_execution_ids']==[]
        assert (work/'experiments.jsonl').read_bytes()==before_retry
        assert sum(r['event']=='matured' for r in records(work))==2
        state=invoke('status');assert not state['result']['tested_configurations'] and state['result']['checkpoint'] is None
        viewed=invoke('review')
        if arm=='ledger':
            # Six scored backtests plus both mature production executions.
            assert viewed['result']['record_count']==8
            assert any(c['kind']=='forecast' and c['n']==1 for c in viewed['result']['matched_comparisons'])
            immutable=[r['ledger_score'] for r in records(work) if r['event']=='result' and r['kind']=='backtest']
            assert immutable==before_scores
        checks.append(arm+': new-origin reset, matured outcome ingestion, immutable historical evidence')
    for left,right in [('plain','gnomon'),('gnomon','ledger')]:
        for a,b in zip(results[left],results[right],strict=True):
            assert a['point']==b['point'] and a['metrics']==b['metrics']
        checks.append(left+'='+right+': exact numerical equivalence across all eight fits')
    from .pipeline import gate
    report={'complete':True,'audit_failures':[],
            'arms':{a:{'workflow_complete':11,'tasks':12} for a in ARMS}}
    assert gate(report)['passed']
    report['arms']['plain']['workflow_complete']=10
    assert not gate(report)['passed']
    report['arms']['plain']['workflow_complete']=12;report['audit_failures']=['synthetic integrity failure']
    assert not gate(report)['passed']
    checks.append('pilot promotion requires >=11/12 in every arm and no integrity failure')
    from .test_maturation import run_checks as check_maturation
    check_maturation(root/'maturation-tests')
    checks.append('maturation identity, visibility, idempotence, independent expected metrics and no new fits')
    from .test_orchestration import run_checks
    run_checks(root/'orchestration-tests')
    checks.append('bounded termination correction, live pinned Hermes handoff, exact API budget and phase interventions')
    final_inventory=runtime_inventory()
    dump(root/'final-runtime-inventory.json',final_inventory)
    assert initial_inventory==final_inventory
    checks.append('pinned package parity before and after native Hermes; lazy installations disabled')
    assert all(sha(HERE/n)==h for n,h in frozen.items())
    dump(root/'passed.json',{'passed':True,'checks':checks,'tested_sources':frozen})
    print(json.dumps({'passed':True,'checks':checks}))


if __name__=='__main__':main()
