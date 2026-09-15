"""Build a new common workflow-progress capsule from the frozen 096 capsule.

No live source is modified, and this builder does not authorize dispatch.
"""
import ast
import hashlib
import json
from pathlib import Path

BASE_CAPSULE_SHA = '1935bfc673c4cc71b314d18c86ee732254fb68e07aadb1e91e55b377f39eaaed'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build(source, output):
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink() or source in output.resolve().parents:
        raise ValueError('Require a fresh capsule outside the frozen source')
    if digest((source/'capsule.json').read_bytes()) != BASE_CAPSULE_SHA:
        raise ValueError('Expected the exact frozen 096 capsule')
    base = json.loads((source/'capsule.json').read_text())
    package = source/'benchmarks/hermes_ml_checkpoint_v6'
    content = {}
    for path in package.iterdir():
        if path.is_symlink() or path.is_dir() and path.name != '__pycache__':
            raise ValueError('Unexpected source entry')
        if path.is_file():
            content[path.name] = path.read_bytes()
    if {name: digest(raw) for name, raw in content.items()} != base['sources']:
        raise ValueError('Frozen 096 sources changed')
    fragment = Path(__file__).with_name('workflow_progress_097.py').read_text()
    node = next(n for n in ast.parse(fragment).body if isinstance(n, ast.FunctionDef) and n.name == 'workflow_progress')
    transport = content['transport.py'].decode()
    marker = 'class Proxy(ThreadingHTTPServer):'
    if transport.count(marker) != 1:
        raise ValueError('Unexpected transport class boundary')
    transport = transport.replace(marker, ast.get_source_segment(fragment, node)+'\n\n'+marker)
    transport = transport.replace('import hashlib\n', 'import hashlib\nimport csv\n', 1)
    marker = "        current_phase = phase(number, seconds)\n"
    if transport.count(marker) != 1:
        raise ValueError('Unexpected transport phase boundary')
    transport = transport.replace(marker, marker + '''        progress = workflow_progress(self.server.work, number, seconds) if self.server.work is not None else None
        dump(str(prefix) + '-workflow-state.json', {'request_number':number,'seconds_remaining':seconds})
        if progress is not None:
            notice = json.dumps(progress, sort_keys=True, separators=(',', ':'))
            payload['messages'] = [*payload['messages'], {'role':'system','content':notice}]
            dump(str(prefix) + '-workflow-progress.json', {'progress':progress,'notice':notice})
''')
    content['transport.py'] = transport.encode()
    auditor = Path(__file__).with_name('workflow_audit_097.py').read_text()
    audit_node = next(n for n in ast.parse(auditor).body
                      if isinstance(n, ast.FunctionDef) and n.name == 'audit_workflow_progress')
    analyze = content['analyze.py'].decode()
    marker = 'def analyze(root, output=None):'
    if analyze.count(marker) != 1:
        raise ValueError('Unexpected analyzer boundary')
    analyze = analyze.replace(marker, ast.get_source_segment(auditor, audit_node)+'\n\n'+marker)
    marker = "        r['http_errors']=sum(v['status']!=200 for v in receipt)\n"
    if analyze.count(marker) != 1:
        raise ValueError('Unexpected analyzer session boundary')
    analyze = analyze.replace(marker, "        r['workflow_progress_audit']=audit_workflow_progress(p.parent,job)\n"
                             "        checks+=r['workflow_progress_audit']['checks']\n"+marker)
    content['analyze.py'] = analyze.encode()
    content['TASK.md'] += (
        '\nCommon progress reminders at requests 4, 8 and 11 report your current workflow\n'
        'and remaining budget before exploration closes. They use only your visible\n'
        'task and execution records, perform no fits, and choose no model/settings.\n'
        'Complete review, baseline and an ML comparison early; inspect source code or\n'
        'additional summaries only when useful within the remaining exploration budget.\n'
        'The history target column is value; the history header names its covariates.\n').encode()
    content['PROTOCOL.md'] += (
        '\n## Undeployed 097 common workflow-progress amendment\n\n'
        'All three arms receive the same deterministic current-progress reminder at\n'
        'requests 4, 8 and 11, only while more than 90 seconds remain. Original and\n'
        'forwarded messages and each reminder are retained. No new fit, model choice,\n'
        'outcome, information source, or extra call/time budget is introduced.\n'
        'The frozen 096 pilot remains an independent retained experiment. This new\n'
        'capsule requires fresh arm state, a new prospective plan, a worker/transport\n'
        'preflight and a new pilot. No accuracy-dependent continuation is allowed.\n').encode()
    for name, raw in content.items():
        if name.endswith('.py'):
            compile(raw, name, 'exec')
    after = {name: digest(raw) for name, raw in content.items()}
    manifest = {
        'status': 'offline_prototype_not_dispatch_ready', 'candidate': '097_common_workflow_progress',
        'parent_capsule_sha256': BASE_CAPSULE_SHA, 'parent_sources': base['sources'],
        'base_sources': base['base_sources'], 'sources': after,
        'changed_from_096': sorted(n for n in content if after[n] != base['sources'][n]),
        'fragment_sha256': digest(fragment.encode()), 'common_to_all_arms': True,
        'workflow_audit_sha256': digest(auditor.encode()),
        'engy_calls': 0, 'provider_calls': 0, 'final_gate_opened': False,
        'outstanding': ['exact worker and transport integration',
                        'new prospective plan', 'terminal 096 evidence', 'new gated pilot'],
    }
    target = output/'benchmarks/hermes_ml_checkpoint_v6'
    target.mkdir(parents=True)
    for name, raw in content.items():
        (target/name).write_bytes(raw)
    (output/'capsule.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest
