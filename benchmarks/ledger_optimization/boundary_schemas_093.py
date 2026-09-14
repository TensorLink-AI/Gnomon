"""Common agent tool surface for the prospective, metered three-arm trial."""
from copy import deepcopy

from .hermes_boundary_093 import NATIVE_TOOLS


def obj(properties, required=()):
    return {'type':'object','properties':properties,'required':list(required),'additionalProperties':False}


def schema(name, description, parameters):
    return {'type':'function','function':{'name':name,'description':description,'parameters':parameters}}


def tool_schemas(native_tools):
    string={'type':'string'}
    lab=obj({'operation':{'type':'string','enum':['start','status','review','backtest','commit','sync']},
             'config':{'type':'object','description':'Exact model configuration from TASK.md; used only by backtest/commit.'},
             'execution_id':{'type':'string','description':'Commit an existing execution; preserve the returned ID.'},
             'offset':{'type':'integer','minimum':0,'maximum':1000000},
             'limit':{'type':'integer','minimum':1,'maximum':100},
             'pair':{'type':'array','items':string,'minItems':2,'maxItems':2,
                     'description':'Two exact config_ids from a returned ledger review card.'}},['operation'])
    tools=[schema('lab',
        'All model fitting and forecast selection must use this metered lab. start creates baseline evidence; '
        'status inspects it; review pages prior outcomes (offset/limit/pair); backtest requires config; '
        'commit selects a tested config or execution_id; sync updates local evidence. '
        'Inspect the returned exit_code and JSON stdout. Exploration/selection phase and the shared fit limit are enforced.',lab),
        schema('project_list','List current project files without executing code.',obj({})),
        schema('evidence_read','Read raw project text with pagination and SHA-256; never execute its content.',obj({
            'path':string,'offset':{'type':'integer','minimum':0},
            'max_chars':{'type':'integer','minimum':1,'maximum':16384}},['path'])),
        schema('data_summary','Summarize an explicit history.csv row window and optional visible grouping column; no model fits.',obj({
            'column':string,'start_row':{'type':'integer','minimum':0},
            'end_row':{'type':'integer','minimum':0},'group_by':string},['column'])),
        schema('notes_write','Write decision.json (a JSON object) or notes/*.md (at most 16384 UTF-8 bytes). '
               'Cannot modify source, data, budgets or execution evidence.',obj({'path':string,'text':string},['path','text']))]
    selected={t['function']['name']:deepcopy(t) for t in native_tools if t['function']['name'] in NATIVE_TOOLS}
    if set(selected)!=NATIVE_TOOLS:
        raise ValueError('Pinned Hermes did not expose every required native memory and text-skill tool.')
    for name,t in selected.items():
        f=t['function'];p=f['parameters'];p['additionalProperties']=False
        f['description']+=' Benchmark boundary: native text memory only; no terminal execution or skill shell preprocessing.'
        if name in ('memory','skill_manage') and 'operations' in p['properties']:
            op=p['properties']['operations'];op.update(minItems=1,maxItems=32)
            op['items']['additionalProperties']=False
        if name=='skill_manage':
            p['properties']['operations']['items']['properties']['file_path']={
                'type':'string','pattern':r'^(SKILL\.md|(references|templates)/[a-zA-Z0-9_-]+\.(md|txt))$',
                'description':'Text files only: SKILL.md, references/*.md or *.txt, templates/*.md or *.txt.'}
        if name=='skill_view':
            p['properties']['file_path']={'type':'string','description':'SKILL.md, references/*.md or *.txt, templates/*.md or *.txt only.'}
        # No plugin namespaces, paths, or arbitrary category directory trees.
        props=p['properties']['operations']['items']['properties'] if name=='skill_manage' else p['properties']
        for key in ('name','category'):
            if key in props:
                props[key].update(pattern=r'^[a-z0-9][a-z0-9_-]{0,63}$')
        tools.append(t)
    return tools


def attach(agent, boundary):
    agent.tools=tool_schemas(agent.tools)
    agent.valid_tool_names={t['function']['name'] for t in agent.tools}
    agent.execution_boundary=boundary
