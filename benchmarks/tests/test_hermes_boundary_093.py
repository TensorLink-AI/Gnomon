"""Hermes dispatch regressions, independent of paid model responses."""
import json
from types import SimpleNamespace
import unittest

from benchmarks.ledger_optimization.hermes_boundary_093 import (
    BoundaryRejected, HermesBoundary, NATIVE_TOOLS, bounded_agent_class, parse_arguments,
    validate_native, native_callbacks,
)


def call(name, args, ident='a'):
    return SimpleNamespace(id=ident, function=SimpleNamespace(name=name, arguments=args))


class NoNativeExecution:
    def _execute_tool_calls(self, *args, **kwargs):
        raise AssertionError('Base batch dispatcher was reached')

    _execute_tool_calls_sequential = _execute_tool_calls
    _execute_tool_calls_concurrent = _execute_tool_calls
    _invoke_tool = _execute_tool_calls


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.events, self.lab_calls, self.native_calls = [], [], []
        def dispatch(tool, args):
            self.lab_calls.append((tool, args))
            return {'status': 'ok' if tool == 'lab' else 'error'}
        self.lab = SimpleNamespace(dispatch=dispatch)
        native = {tool: lambda args, tool=tool: self.native_calls.append((tool,args)) for tool in NATIVE_TOOLS}
        self.boundary = HermesBoundary(self.lab, native=native, record=self.events.append)
        self.agent = bounded_agent_class(NoNativeExecution)()
        self.agent.execution_boundary = self.boundary

    def test_every_batch_entry_point_preserves_ids_and_denies_terminal(self):
        for method in ('_execute_tool_calls', '_execute_tool_calls_sequential', '_execute_tool_calls_concurrent'):
            with self.subTest(method=method):
                messages=[]
                batch=SimpleNamespace(tool_calls=[call('terminal','{"command":"python local_probe.py"}','bad'),
                                                   call('lab','{"operation":"status"}','good')])
                getattr(self.agent,method)(batch,messages,'task',2)
                self.assertEqual([m['tool_call_id'] for m in messages],['bad','good'])
                self.assertEqual([json.loads(m['content'])['status'] for m in messages],['error','ok'])
                self.assertFalse(self.agent._executing_tools)
        self.assertEqual(len(self.events),12)

    def test_direct_invoke_cannot_reach_native_registry(self):
        self.assertEqual(json.loads(self.agent._invoke_tool('delegate_task',{},'task'))['status'],'error')
        self.assertFalse(self.native_calls)

    def test_json_ambiguity_and_nonfinite_values_never_dispatch(self):
        for raw in ('[]', '{"a":1,"a":2}', '{"config":{"alpha":1,"alpha":2}}',
                    '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '{broken', '{} trailing'):
            with self.subTest(raw=raw):
                result=json.loads(self.boundary.invoke('lab',raw,'a'))
                self.assertEqual(result['status'],'error')
                self.assertFalse(result['execution_started'])
        self.assertFalse(self.lab_calls)

    def test_deadline_interrupt_and_duplicate_ids_prevent_mutations(self):
        self.boundary.deadline=0
        self.assertEqual(json.loads(self.boundary.invoke('memory','{}','a'))['status'],'error')
        self.boundary.deadline=float('inf');self.agent._interrupt_requested=True
        messages=[]
        self.agent._execute_tool_calls(SimpleNamespace(tool_calls=[call('lab','{"operation":"start"}')]),messages,'t')
        self.assertFalse(self.lab_calls)
        self.agent._interrupt_requested=False
        with self.assertRaises(BoundaryRejected):
            self.agent._execute_tool_calls(SimpleNamespace(tool_calls=[call('lab','{}'),call('lab','{}')]),[],'t')
        self.assertFalse(self.lab_calls)

    def test_failed_persistence_stops_later_tools(self):
        self.agent._flush_messages_to_session_db=lambda messages:False
        batch=SimpleNamespace(tool_calls=[call('lab','{"operation":"status"}','one'),call('lab','{"operation":"start"}','two')])
        self.agent._execute_tool_calls(batch,[],'t')
        self.assertEqual(len(self.lab_calls),1)
        self.assertTrue(self.agent._incremental_persistence_failed)

    def test_rejected_native_batch_has_no_partial_mutation(self):
        args={'operations':[{'action':'create','name':'lesson','content':'lesson text'},
                            {'action':'write_file','name':'lesson','file_path':'scripts/probe.py','file_content':'from numerical import predict'}]}
        result=json.loads(self.boundary.invoke('skill_manage',json.dumps(args),'a'))
        self.assertFalse(result['execution_started']);self.assertFalse(self.native_calls)

    def test_memory_and_text_skills_remain_available(self):
        calls={'memory':{'target':'memory','operations':[{'action':'add','content':'A dated observation'}]},
               'skills_list':{}, 'skill_view':{'name':'lesson','file_path':'references/evidence.md'},
               'skill_manage':{'operations':[{'action':'create','name':'lesson','content':'A dated lesson'}]}}
        for tool,args in calls.items():
            self.assertEqual(json.loads(self.boundary.invoke(tool,json.dumps(args),tool))['status'],'ok')
        self.assertEqual(len(self.native_calls),4);self.assertFalse(self.lab_calls)

    def test_native_arguments_cannot_enable_preprocessing_or_escape_paths(self):
        cases=[('skill_view',{'name':'lesson','preprocess':True}),
               ('skill_view',{'name':'../lesson'}), ('skill_view',{'name':'plugin:lesson'}),
               ('skill_view',{'name':'lesson','file_path':'../../lab.py'}),
               ('memory',{'target':'memory','action':'add','content':'x','store':'elsewhere'}),
               ('skill_manage',{'operations':[{'name':'lesson','action':'create','content':'x','category':'../work'}]})]
        for tool,args in cases:
            self.assertEqual(json.loads(self.boundary.invoke(tool,json.dumps(args),'a'))['status'],'error')
        self.assertFalse(self.native_calls)

    def test_failed_intent_receipt_prevents_dispatch(self):
        def fail(event): raise OSError('disk full')
        self.boundary.record=fail
        with self.assertRaises(OSError):self.boundary.invoke('lab','{"operation":"start"}','a')
        self.assertFalse(self.lab_calls)

    def test_runtime_preprocessing_cannot_erase_invalid_arguments(self):
        calls=[call('lab','{"operation":"status"}','one'),
               call('lab','{"operation":"start","operation":"status"}','two')]
        self.agent._uniquify_tool_call_ids(calls)
        self.assertEqual(len(self.agent._deduplicate_tool_calls(calls)),2)
        # Even if the runtime rewrites the parsed arguments for message storage,
        # dispatch validates the raw original captured before preprocessing.
        calls[1].function.arguments='{"operation":"status"}'
        messages=[]
        self.agent._execute_tool_calls(SimpleNamespace(tool_calls=calls),messages,'t')
        self.assertEqual(len(self.lab_calls),1)
        self.assertEqual(json.loads(messages[1]['content'])['status'],'error')
        self.assertIsNone(self.agent._repair_tool_call('terminal'))


if __name__ == '__main__':
    unittest.main()
