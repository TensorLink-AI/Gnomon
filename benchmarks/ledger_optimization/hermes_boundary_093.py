"""Host-owned Hermes adapter for the prospective 093 trial (not yet dispatched).

Override batch dispatch, including its sequential/concurrent entry points. Never
fall back to the Hermes terminal, inline executor, delegation, or plugin registry.
Native memory callbacks are supplied by the host, with explicit text-only gates.
"""
from __future__ import annotations

import json
import re
import threading
import time

from .execution_boundary_093 import BoundaryRejected, fields


NATIVE_TOOLS = frozenset({'memory', 'skills_list', 'skill_view', 'skill_manage'})


def parse_arguments(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise BoundaryRejected(f'Duplicate argument field: {key}')
            result[key] = value
        return result

    def constant(value):
        raise BoundaryRejected(f'Non-finite JSON value: {value}')

    if not isinstance(raw, str) or len(raw.encode('utf-8')) > 65536:
        raise BoundaryRejected('Tool arguments must be a JSON object of at most 65536 bytes.')
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    fields(value, value if isinstance(value, dict) else ())
    # Overflowing exponent notation is also non-finite, despite valid JSON syntax.
    json.dumps(value, allow_nan=False)
    return value


def text(value):
    if not isinstance(value, str) or len(value.encode('utf-8')) > 32768:
        raise BoundaryRejected('Expected bounded text (at most 32768 UTF-8 bytes).')


def name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', value):
        raise BoundaryRejected('Use a local skill name or category with letters, digits, hyphens or underscores.')


def skill_path(value):
    if value == 'SKILL.md':
        return
    if not isinstance(value, str) or not re.fullmatch(r'(references|templates)/[a-zA-Z0-9_-]+\.(md|txt)', value):
        raise BoundaryRejected('Only SKILL.md or text files in references/ and templates/ are supported.')


def validate_native(tool, args):
    """Validate the entire batch before any native mutation; no raw kwargs escape."""
    if tool == 'memory':
        fields(args, {'target', 'action', 'content', 'new_text', 'old_text', 'operations'}, {'target'})
        if args['target'] not in ('memory', 'user'):
            raise BoundaryRejected('Unknown memory target.')
        if 'operations' in args:
            fields(args, {'target', 'operations'}, {'target', 'operations'})
            ops = args['operations']
        else:
            ops = [{k: v for k, v in args.items() if k != 'target'}]
        if not isinstance(ops, list) or not 1 <= len(ops) <= 32:
            raise BoundaryRejected('Supply between 1 and 32 memory operations.')
        for op in ops:
            fields(op, {'action', 'content', 'new_text', 'old_text'}, {'action'})
            if op['action'] not in ('add', 'replace', 'remove'):
                raise BoundaryRejected('Unsupported memory operation.')
            for key in set(op) - {'action'}:
                text(op[key])
            if op['action'] in ('replace', 'remove') and not op.get('old_text'):
                raise BoundaryRejected('Replacing/removing memory requires old_text.')
            if op['action'] in ('add', 'replace') and not (op.get('content') or op.get('new_text')):
                raise BoundaryRejected('Adding/replacing memory requires content.')
    elif tool == 'skills_list':
        fields(args, {'category'})
        if 'category' in args:
            name(args['category'])
    elif tool == 'skill_view':
        fields(args, {'name', 'file_path'}, {'name'})
        name(args['name'])
        if 'file_path' in args:
            skill_path(args['file_path'])
    elif tool == 'skill_manage':
        fields(args, {'operations'}, {'operations'})
        ops = args['operations']
        if not isinstance(ops, list) or not 1 <= len(ops) <= 32:
            raise BoundaryRejected('Supply between 1 and 32 text-skill operations.')
        options = {'create': {'content', 'category'},
                   'patch': {'content', 'old_string', 'new_string', 'replace_all', 'file_path'},
                   'delete': set(), 'write_file': {'file_path', 'file_content'},
                   'remove_file': {'file_path'}}
        for op in ops:
            fields(op, {'name', 'action', 'content', 'category', 'old_string', 'new_string',
                        'replace_all', 'file_path', 'file_content'}, {'name', 'action'})
            name(op['name'])
            if not isinstance(op['action'], str) or op['action'] not in options:
                raise BoundaryRejected('Unsupported text-skill operation.')
            fields(op, {'name', 'action'} | options[op['action']])
            if 'category' in op:
                name(op['category'])
            if 'file_path' in op:
                skill_path(op['file_path'])
            if op['action'] in ('write_file', 'remove_file') and 'file_path' not in op:
                raise BoundaryRejected('A text file_path is required.')
            if 'replace_all' in op and type(op['replace_all']) is not bool:
                raise BoundaryRejected('replace_all must be boolean.')
            for key in ('content', 'file_content', 'old_string', 'new_string'):
                if key in op:
                    text(op[key])
    else:
        raise BoundaryRejected('Native tool is not admitted.')


def rejection(error, *, started=False):
    return {'status': 'error', 'error': {'code': 'EXECUTION_BOUNDARY_REJECTED', 'message': str(error)},
            'execution_started': started}


class HermesBoundary:
    def __init__(self, lab, *, native, record, deadline=float('inf'), now=time.time):
        if set(native) != NATIVE_TOOLS:
            raise ValueError('Preserve all four native memory/text-skill tools with explicit host callbacks.')
        self.lab, self.native, self.record = lab, dict(native), record
        self.deadline, self.now = deadline, now
        self.lock = threading.RLock()

    def invoke(self, tool, raw, call_id, *, cancelled=False):
        with self.lock:
            # Persist intent before executing anything, including malformed/denied calls.
            self.record({'stage': 'requested', 'tool_call_id': call_id, 'tool': tool, 'raw_arguments': raw})
            started = False
            try:
                if cancelled or self.now() >= self.deadline:
                    raise BoundaryRejected('Session cancelled or wall-clock deadline reached; nothing new executed.')
                args = parse_arguments(raw)
                if tool in NATIVE_TOOLS:
                    validate_native(tool, args)
                    started = True
                    result = {'status': 'ok', 'result': self.native[tool](args), 'scope': 'native_text_memory'}
                else:
                    result = self.lab.dispatch(tool, args)
            except (ValueError, TypeError, OSError, RecursionError) as error:
                result = rejection(error, started=started)
            self.record({'stage': 'returned', 'tool_call_id': call_id, 'tool': tool, 'result': result})
            return json.dumps(result, ensure_ascii=False, allow_nan=False)


def bounded_agent_class(base):
    """Use with the pinned Hermes class; attach boundary before run_conversation."""
    class MeteredAgent(base):
        def _handle_max_iterations(self, messages, api_call_count):
            return 'Request budget reached. The saved checkpoint remains authoritative.'

        def _uniquify_tool_call_ids(self, calls):
            # This is the pinned runtime's first tool-validation hook, before it
            # fills empty arguments, repairs names, or deduplicates parsed JSON.
            raw=[{'id':c.id,'name':c.function.name,'arguments':c.function.arguments} for c in calls]
            self.execution_boundary.record({'stage':'raw_batch','calls':raw})
            ids=[c.id for c in calls]
            if any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids):
                raise BoundaryRejected('Missing or duplicate tool-call IDs; batch not executed.')
            self._boundary_raw_calls={c['id']:c for c in raw}

        @staticmethod
        def _deduplicate_tool_calls(calls):
            return calls  # Every emitted call receives its own result and accounting.

        @staticmethod
        def _cap_delegate_task_calls(calls):
            return calls  # Delegation is unavailable; retain its rejection evidence.

        def _repair_tool_call(self, name):
            return None  # Do not reinterpret an unknown operation as an admitted tool.

        def _execute_tool_calls(self, assistant_message, messages, effective_task_id, api_call_count=0, **kwargs):
            self._executing_tools = True
            try:
                calls = assistant_message.tool_calls
                ids = [call.id for call in calls]
                if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
                    # An invalid envelope cannot be paired safely. Abort before any mutation.
                    self.execution_boundary.record({'stage': 'invalid_batch', 'tool_call_ids': ids})
                    raise BoundaryRejected('Missing or duplicate tool-call IDs; batch not executed.')
                for call in calls:
                    raw=getattr(self,'_boundary_raw_calls',{}).get(call.id)
                    result = self.execution_boundary.invoke(
                        raw['name'] if raw else call.function.name,
                        raw['arguments'] if raw else call.function.arguments, call.id,
                        cancelled=getattr(self, '_interrupt_requested', False))
                    messages.append({'role': 'tool', 'tool_call_id': call.id,
                                     'name': call.function.name, 'content': result})
                    flush = getattr(self, '_flush_messages_to_session_db', None)
                    if flush is not None:
                        try:
                            persisted = flush(messages) is not False
                        except Exception:
                            persisted = False
                        if not persisted:
                            self._incremental_persistence_failed = True
                            return  # No later tools execute after a failed canonical append.
            finally:
                self._executing_tools = False
                self._boundary_raw_calls = {}

        # Defensive interception of alternate entry points; none calls the base executor.
        _execute_tool_calls_sequential = _execute_tool_calls
        _execute_tool_calls_concurrent = _execute_tool_calls

        def _invoke_tool(self, function_name, function_args, effective_task_id, tool_call_id=None, **kwargs):
            return self.execution_boundary.invoke(function_name, json.dumps(function_args, allow_nan=False),
                                                  tool_call_id, cancelled=getattr(self, '_interrupt_requested', False))

    return MeteredAgent


def native_callbacks(agent):
    """Pinned Hermes API only. Host must use fresh per-arm homes with sync disabled.

    Crucially, skill_view(preprocess=False) prevents inline shell interpolation.
    These are explicit imports, never registry resolution of an arbitrary tool.
    """
    from tools.memory_tool import memory_tool
    from tools.skills_tool import skill_view, skills_list
    from tools.skill_manager_tool import skill_manage
    return {'memory': lambda args: memory_tool(**args, store=agent._memory_store),
            'skills_list': lambda args: skills_list(**args),
            'skill_view': lambda args: skill_view(**args, preprocess=False),
            'skill_manage': lambda args: skill_manage(action=None, name=None, operations=args['operations'])}
