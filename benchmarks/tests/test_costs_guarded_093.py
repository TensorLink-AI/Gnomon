import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.ledger_optimization.costs_guarded_093 import summarize, token_count


def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class CostTests(unittest.TestCase):
    def test_pilot_copy_not_counted_twice_and_inflight_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot=root/'plain/series/round-0';later=root/'plain/series/round-3'
            for folder in (pilot,later):
                dump(folder/'api-01-forwarded.json',{})
                dump(folder/'api-01-response.json',{'usage':{'total_tokens':100}})
                dump(folder/'service-admission/probe-01-receipt.json',{'usage':{'total_tokens':2}})
            dump(later/'api-02-forwarded.json',{})
            dump(root/'plain/series/work/api-01-response.json',{'usage':{'total_tokens':99999}})
            report=summarize(root)
            self.assertEqual(report['deduplicated_total']['forwarded_requests'],3)
            self.assertEqual(report['deduplicated_total']['reported_tokens'],200)
            self.assertEqual(report['deduplicated_total']['requests_without_reported_usage'],1)
            self.assertEqual(report['deduplicated_total']['readiness_reported_tokens'],4)
            self.assertEqual(report['stages']['retained_pilot']['reported_tokens'],100)
            self.assertEqual(report['stages']['continuation']['reported_tokens'],100)
            self.assertIsNone(report['billing_dollars'])

    def test_unknown_usage_and_orphan_responses_never_fabricate_zero_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'ledger/series/round-4'
            dump(folder/'api-01-forwarded.json',{})
            dump(folder/'api-01-response.json',{'usage':{'total_tokens':None}})
            dump(folder/'api-01-receipt.json',{'status':500})
            dump(folder/'api-02-response.json',{'usage':{'total_tokens':999}})
            dump(folder/'service-admission/probe-01-receipt.json',{})
            total=summarize(root)['deduplicated_total']
            self.assertEqual(total['requests_without_reported_usage'],1)
            self.assertEqual(total['readiness_unknown_usage'],1)
            self.assertEqual(total['orphan_responses'],1)
            self.assertEqual(total['api_errors'],1)
            self.assertEqual(total['reported_tokens'],0)
        for value in (None,True,-1,1.1,'20'):
            self.assertIsNone(token_count({'usage':{'total_tokens':value}}))
        self.assertEqual(token_count({'usage':{'total_tokens':0}}),0)


if __name__=='__main__':unittest.main()
