import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,MagicMock
from benchmarks.ledger_optimization.ml_service_admission import decision,wait_ready,probe,_network,MAX_PROBES,PAYLOAD

GOOD=json.dumps({'choices':[{'message':{'content':'READY'}}],'usage':{'total_tokens':20}}).encode()
ERROR=json.dumps({'error':{'type':'rate_limit_error','code':429}}).encode()


class AdmissionTest(unittest.TestCase):
    def test_200_error_is_not_healthy(self):
        self.assertEqual(decision(200,ERROR),{'ready':False,'retryable':True,'cause':'service_unavailable'})

    def test_authorization_contract_failure_stops(self):
        self.assertFalse(decision(401,b'no')['retryable'])
        self.assertFalse(decision(200,b'{}')['ready'])
        self.assertTrue(decision(504,b'<html>timeout</html>')['retryable'])

    def test_reasoning_response_is_service_evidence_not_task_completion(self):
        body=json.dumps({'choices':[{'message':{'reasoning_content':'ready'}}]}).encode()
        self.assertTrue(decision(200,body)['ready'])

    def test_wait_then_admit_has_exact_accounting_and_no_agent_work(self):
        sequence=iter([(429,ERROR,1),(200,GOOD,2)]);waits=[]
        with tempfile.TemporaryDirectory() as parent:
            root=Path(parent)/'admission';r=wait_ready(root,lambda:next(sequence),sleep=waits.append)
            self.assertTrue(r['ready']);self.assertEqual(r['probe_count'],2);self.assertEqual(waits,[60])
            self.assertFalse(r['agent_started']);self.assertEqual(r['agent_requests'],0)
            for p in root.glob('*-request.json'):self.assertEqual(json.loads(p.read_text()),PAYLOAD)
            self.assertEqual(r['attempts'][1]['usage'],{'total_tokens':20})
            with self.assertRaises(FileExistsError):wait_ready(root,lambda:next(sequence),sleep=waits.append)

    def test_persistent_outage_is_bounded(self):
        waits=[]
        with tempfile.TemporaryDirectory() as parent:
            r=wait_ready(Path(parent)/'admission',lambda:(429,ERROR,1),sleep=waits.append)
            self.assertFalse(r['ready']);self.assertFalse(r['waiting'])
            self.assertEqual(r['probe_count'],MAX_PROBES);self.assertEqual(len(waits),MAX_PROBES-1)
            self.assertEqual(r['terminal_cause'],'service_admission_exhausted')

    def test_nonretryable_failure_does_not_spend_more_probes(self):
        waits=[]
        with tempfile.TemporaryDirectory() as parent:
            r=wait_ready(Path(parent)/'admission',lambda:(401,b'{}',1),sleep=waits.append)
            self.assertFalse(r['ready']);self.assertEqual(r['probe_count'],1);self.assertEqual(waits,[])

    def test_provider_echoed_credential_is_redacted_before_return(self):
        response=MagicMock();response.__enter__.return_value=response
        response.status=503;response.read.return_value=b'synthetic-secret-key'
        with patch('benchmarks.ledger_optimization.ml_service_admission.urllib.request.urlopen',return_value=response):
            status,raw=_network('synthetic-secret-key')
        self.assertEqual(status,503);self.assertEqual(raw,b'[REDACTED]')
        with self.assertRaises(ValueError):probe('')

    def test_wall_deadline_terminates_slow_worker(self):
        with tempfile.TemporaryDirectory() as parent:
            script=Path(parent)/'slow.py';script.write_text('import time\ntime.sleep(30)\n')
            with patch('benchmarks.ledger_optimization.ml_service_admission.__file__',str(script)),patch('benchmarks.ledger_optimization.ml_service_admission.PROBE_TIMEOUT',0.05):
                status,raw,seconds=probe('synthetic-secret')
            self.assertEqual(status,504);self.assertLess(seconds,2)
            self.assertIn(b'wall_clock_deadline',raw);self.assertNotIn(b'synthetic-secret',raw)

    def test_worker_contract_failure_stops_admission(self):
        with tempfile.TemporaryDirectory() as parent:
            script=Path(parent)/'bad.py';script.write_text('raise SystemExit(1)\n')
            with patch('benchmarks.ledger_optimization.ml_service_admission.__file__',str(script)):
                status,raw,_=probe('synthetic-secret')
            self.assertEqual(status,400);self.assertFalse(decision(status,raw)['retryable'])


if __name__=='__main__':unittest.main()
