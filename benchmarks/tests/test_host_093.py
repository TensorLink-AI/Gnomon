"""Prospective host transport and preflight dispatch-gate checks, no network APIs."""
import http.client
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from benchmarks.hermes_ml_checkpoint_v6 import run, transport


class HostTests(unittest.TestCase):
    def test_metadata_and_request_cap_with_structured_selection_notice(self):
        seen=[]
        class Response:
            status=200
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):return b'{"choices":[{"message":{"content":"synthetic-key"}}]}'
        def upstream(request,**kwargs):
            seen.append(json.loads(request.data));return Response()
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'out';out.mkdir();work=Path(tmp)/'work';work.mkdir()
            with patch.object(transport.urllib.request,'urlopen',upstream),transport.proxy(out,'synthetic-key',work,time.time()+480) as url:
                port=int(url.split(':')[-1].split('/')[0])
                def post(payload):
                    c=http.client.HTTPConnection('127.0.0.1',port)
                    c.request('POST','/v1/chat/completions',json.dumps(payload),{'Content-Type':'application/json'})
                    response=c.getresponse();value=response.read();status=response.status;c.close();return status,value
                self.assertEqual(post({'name':'model'})[0],404);self.assertFalse(seen)
                for i in range(16):
                    status,body=post({'model':'wrong','messages':[{'role':'user','content':'synthetic'}]})
                    self.assertEqual(status,200);self.assertNotIn(b'synthetic-key',body)
                    self.assertEqual(seen[-1]['model'],'deepseek-v4.1-flash')
                    self.assertEqual(seen[-1]['max_tokens'],3072)
                    notice=[m['content'] for m in seen[-1]['messages'] if m['role']=='system']
                    self.assertEqual(bool(notice),i>=12)
                    if notice:
                        self.assertIn('lab tool',notice[0]);self.assertNotIn('python lab.py',notice[0])
                self.assertEqual(post({'messages':[{'role':'user','content':'excess'}]})[0],400)
                self.assertEqual(len(seen),16)
                budget=json.loads((work/'agent-budget.json').read_text())
                self.assertEqual(budget['remaining_requests'],0);self.assertEqual(budget['forwarded_requests'],16)
                self.assertEqual(len(list(out.glob('api-*-forwarded.json'))),16)
                self.assertEqual(len(list(out.glob('blocked-request-*.json'))),1)

    def test_wrong_preflight_source_is_rejected_before_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);preflight=root/'preflight.json'
            preflight.write_text('{"passed":true,"tested_sources":{}}')
            with patch('sys.argv',['run','--pilot','--output',str(root/'run'),'--preflight',str(preflight)]),\
                 patch.object(run,'key',side_effect=AssertionError('credential must not be read')) as key:
                with self.assertRaisesRegex(AssertionError,'Fresh exact-source'):
                    run.main()
                key.assert_not_called()

    def test_expired_deadline_never_forwards_to_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'out';out.mkdir();work=Path(tmp)/'work';work.mkdir()
            with patch.object(transport.urllib.request,'urlopen') as upstream,\
                 transport.proxy(out,'synthetic-key',work,time.time()-1) as url:
                port=int(url.split(':')[-1].split('/')[0])
                connection=http.client.HTTPConnection('127.0.0.1',port)
                connection.request('POST','/v1/chat/completions',json.dumps({'messages':[{'role':'user','content':'late'}]}))
                response=connection.getresponse();self.assertEqual(response.status,408);response.read();connection.close()
                upstream.assert_not_called()
                self.assertEqual(len(list(out.glob('blocked-request-deadline-*.json'))),1)
                self.assertFalse(list(out.glob('api-*-forwarded.json')))


if __name__=='__main__':unittest.main()
