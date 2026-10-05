"""Prepare a separate 12-case cohort and matched experiment; never dispatch model calls."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys

from benchmarks.workflow.schema import Case
from .common import ROOT, digest, read, write


def case(name, domain, question, inputs, numbers=None, choices=None, episode=None):
    numbers, choices = numbers or {}, choices or {}
    result = {'schema_version':2,'id':'cpu_product_'+name,'kind':'synthetic','domain':domain,
              'question':question, 'available_at_cutoff':inputs,
              'answer_schema':{'numbers':list(numbers),'choices':list(choices)},
              'oracle':{'numbers':numbers,'choices':choices,
                        'tolerances':{key:1e-6 for key in numbers}},
              'tags':['cpu-product-v1','synthetic','not-a-forecast-leaderboard']}
    if episode:
        result['episode'] = episode
        episode[-1]['oracle'] = copy.deepcopy(result['oracle'])
    Case.from_dict(result)
    return result


def cases():
    result = [
        case('forecast','forecasting',
             'Forecast the next three values. The process repeats every four observations. Return f1, f2 and f3.',
             {'history':[13,17,11,19]*8,'horizon':3,'season':4}, {'f1':13,'f2':17,'f3':11}),
        case('observed_statistic','statistics',
             'Calculate population standard deviation of the supplied observed values. This is a description, not a forecast.',
             {'values':[2,4,4,4,5,5,7,9]}, {'population_std':2}),
        case('matched_selection','evaluation',
             'Select the model with the lowest mean absolute error using only origins where BOTH models have predictions and actuals. Report matched origins and winning MAE.',
             {'origins':[1,2,3],'actual':[10,10,100], 'predictions':{'A':[9,9,0],'B':[8,8,None]}},
             {'matched_origins':2,'winning_mae':1},{'model':'A'}),
        case('budget','evaluation',
             'Each candidate/fold forecast costs one call. With the supplied hard call allowance, how many complete folds can compare all candidates? Do not execute model calls.',
             {'candidates':['A','B','C','D','E','F','G','H','I'],'call_allowance':25},
             {'complete_folds':2,'calls_used':18}),
        case('missing_dependency','environment',
             'The requested provider requires an absent package. Network installation is unavailable. Choose report_unavailable or claim_success. Do not substitute another provider.',
             {'requested_provider':'uninstalled_vendor','dependency_installed':False,'network':False},
             choices={'action':'report_unavailable'}),
        case('horizon_alignment','validation',
             'The requested forecast horizon is three timestamps. The supplied provider response contains two points. Choose reject_response or accept_response.',
             {'future_timestamps':['2026-02-01','2026-02-02','2026-02-03'],'point':[3,4]},
             choices={'action':'reject_response'}),
        case('elapsed_time','temporal',
             'Compute elapsed hours between these instants using their explicit UTC offsets.',
             {'start':'2026-04-05T01:30:00+11:00','end':'2026-04-05T03:30:00+10:00'},
             {'hours':3}),
        case('scale_comparison','evaluation',
             'Report the equally weighted mean of per-series MASE. Do not pool raw errors across units.',
             {'series':[{'mae':2,'training_seasonal_scale':4},{'mae':30,'training_seasonal_scale':10}]},
             {'macro_mase':1.75}),
        case('no_completed_evidence','memory',
             'Choose whether to use this episode as completed forecasting evidence at the decision time: eligible or pending.',
             {'decision_time':'2026-02-01T00:00:00Z','episode_last_target':'2026-02-03T00:00:00Z',
              'partial_error':0.1}, choices={'episode_status':'pending'}),
    ]
    result.append(case('delayed_outcome','outcomes',
        'Commit a last-value forecast before outcomes arrive, then score only revealed outcomes. Preserve the original forecast.',
        {'history':[21,24,23]}, {'original_forecast':23,'absolute_error':4}, episode=[
            {'name':'forecast','revealed':{},'answer_schema':{'numbers':['forecast']},'oracle':{'numbers':{'forecast':23}}},
            {'name':'outcome','revealed':{'actual':27,'instruction':'Report your original forecast and its absolute error.'},
             'answer_schema':{'numbers':['original_forecast','absolute_error']},
             'oracle':{'numbers':{'original_forecast':23,'absolute_error':4}}}]))
    result.append(case('revised_outcome','outcomes',
        'Retain the committed forecast and distinguish the first observed outcome from its later revision.',
        {'committed_forecast':31,'actual':33}, {'forecast':31,'initial_error':2,'revised_error':3}, episode=[
            {'name':'first_actual','revealed':{},'answer_schema':{'numbers':['error']},'oracle':{'numbers':{'error':2}}},
            {'name':'revision','revealed':{'revised_actual':28,'instruction':'Report forecast, initial_error and revised_error.'},
             'answer_schema':{'numbers':['forecast','initial_error','revised_error']},
             'oracle':{'numbers':{'forecast':31,'initial_error':2,'revised_error':3}}}]))
    result.append(case('recall_without_reforecast','memory',
        'Review the supplied recorded forecasts. Identify the lower-MAE model; no new forecast is requested or needed. Retain the decision for the follow-up.',
        {'recorded_forecasts':{'A':[5,9],'B':[7,8]},'actuals':[6,10]},
        {'winning_mae':1}, {'model':'A'}, episode=[
            {'name':'compare','revealed':{},'answer_schema':{'numbers':['winning_mae'],'choices':['model']},
             'oracle':{'numbers':{'winning_mae':1},'choices':{'model':'A'}}},
            {'name':'recall','revealed':{'instruction':'Report the model and MAE from the previous comparison; do not invent a new result.'},
             'answer_schema':{'numbers':['winning_mae'],'choices':['model']},
             'oracle':{'numbers':{'winning_mae':1},'choices':{'model':'A'}}}]))
    return result


def prepare(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    cohort=cases()
    payload=''.join(json.dumps(c,sort_keys=True)+'\n' for c in cohort)
    path=output/'cases.jsonl'
    if path.exists() and path.read_text()!=payload:
        raise ValueError('Refusing to overwrite different cohort')
    if not path.exists():
        with path.open('x') as out:
            out.write(payload)
    orders=[]
    rng=random.Random(140)
    for repetition in range(3):
        order=['ordinary','lean','full']
        rng.shuffle(order)
        orders.append({'repetition':repetition,'arm_order':order})
    write(output/'cohort.json',{'cases_sha256':digest(cohort),'tasks':len(cohort),
        'repetitions':3,'scored_episodes':108,'schedule':orders,
        'interpretation':'Synthetic workflow correctness; not independent domain coverage or a live model test',
        'scoring_limits':'Answer oracles do not prove tool use, durable storage, or absence of reruns; inspect traces separately',
        'execution_status':'not_run; model, endpoint, images and spending limit must be supplied'})
    source=ROOT/'benchmarks/workflow/experiment'
    experiment=copy.deepcopy(read(source/'experiment.example.json'))
    providers=copy.deepcopy(read(source/'providers.example.json'))
    experiment['command']=[sys.executable,str(ROOT/'benchmarks/workflow/driver.py')]
    experiment['driver_files']=[str(ROOT/'benchmarks/workflow/driver.py')]
    experiment['common']['prompt_file']='prompt.txt'
    experiment['common']['provider_config_file']='providers.json'
    experiment['arms']['lean']['description']='Same ordinary software plus six default Gnomon execution tools; ledger and temporal tools disabled.'
    experiment['arms']['lean']['tool_contract']='Python plus default MCP session in a separate container; same model providers as ordinary/full.'
    providers['backends']['lean']['options'].pop('execution_options',None)
    write(output/'experiment.example.json',experiment)
    write(output/'providers.example.json',providers)
    prompt=source/'prompt.txt'
    if not prompt.exists():
        prompt=source/'prompt.example.txt'
    target=output/'prompt.txt'
    if not target.exists():
        target.write_text(prompt.read_text())
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(prepare(args.output))


if __name__=='__main__':
    main()
