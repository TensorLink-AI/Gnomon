import io
import unittest
from benchmarks.ledger_optimization.source_header_probe import header


class SourceHeaderProbeTest(unittest.TestCase):
    def test_stops_before_observations(self):
        stream=io.BytesIO(b'@frequency hourly\n@data\nsecret-future-observation\n')
        self.assertEqual(header(stream),['@frequency hourly','@data'])
        self.assertEqual(stream.read(),b'secret-future-observation\n')

    def test_unbounded_or_missing_header_rejected(self):
        for value in (b'# x\n'*201,b'@frequency hourly\n',b'x'*8193):
            with self.assertRaises(ValueError):header(io.BytesIO(value))


if __name__=='__main__':unittest.main()
