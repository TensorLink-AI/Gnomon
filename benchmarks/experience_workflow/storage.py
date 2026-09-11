"""Public Gnomon ingestion and a capable, persistent, read-only SQL alternative."""
from datetime import datetime
import math
import sqlite3
import time

from gnomon import ForecastResult, InferenceEngine, TemporalLedger
from gnomon.ids import FixedClock

from .scenario import canonical, shifted

# This query is deliberately public to the control. No historical scores are
# stored: the CTE computes them from the same arrived raw events as Gnomon.
REFERENCE_SQL = """
WITH eligible AS (
 SELECT p.* FROM predictions p
 WHERE p.series_id=:series_id AND p.unit IS :unit AND p.horizon=:horizon
 AND p.origin BETWEEN :start AND :end AND p.recorded_at<=:recorded_as_of
 AND p.revision=json_extract(:providers, '$.' || p.provider)
 AND NOT EXISTS (SELECT 1 FROM json_each(:context_filters) f
                 WHERE json_extract(p.context, '$.' || f.key) IS NOT f.value)
), visible AS (
 SELECT a.*, ROW_NUMBER() OVER (
   PARTITION BY series_id,unit,valid_time ORDER BY source_available_at DESC,sequence DESC) AS rn
 FROM actuals a WHERE a.series_id=:series_id AND a.unit IS :unit
 AND a.source_available_at<=:source_as_of AND a.recorded_at<=:recorded_as_of
), losses AS (
 SELECT p.origin,p.provider,SQRT(AVG((LOG1P(p.point)-LOG1P(a.value))*(LOG1P(p.point)-LOG1P(a.value)))) AS loss
 FROM eligible p JOIN visible a ON a.rn=1 AND a.valid_time=p.valid_time
 GROUP BY p.origin,p.provider HAVING COUNT(*)=:horizon
), matched AS (
 SELECT origin FROM losses GROUP BY origin
 HAVING COUNT(*)=(SELECT COUNT(*) FROM json_each(:providers))
)
SELECT provider,COUNT(*) AS matched_origins,AVG(loss) AS score
FROM losses JOIN matched USING(origin) GROUP BY provider
ORDER BY score,(SELECT id FROM json_each(:providers) WHERE key=provider)
""".strip()


def parameters(query):
    return {k: canonical(v) if isinstance(v, dict) else v for k, v in query.items()
            if k not in ('metric', 'recent_origins')}


def sql_answer(rows, providers):
    scores = {p: next(row['score'] for row in rows if row['provider'] == p) for p in providers} if rows else {}
    return dict(matched_origins=rows[0]['matched_origins'] if rows else 0,
                scores=scores, ranking=sorted(scores, key=scores.get))


class Store:
    def __init__(self, path, arm):
        self.arm = arm
        self.path = path
        self.path.mkdir(parents=True, exist_ok=False)
        self.seen = set()
        self.notebook = {}
        self.saved_queries = {'matched_evidence': REFERENCE_SQL}
        self.receipts = []
        self.ingest_seconds = 0.
        self.query_seconds = 0.
        self.writes = 0
        if arm == 'gnomon':
            self.ledger = TemporalLedger(path / 'ledger.db')
        elif arm == 'sqlite':
            self.db = sqlite3.connect(path / 'events.db')
            self.db.row_factory = sqlite3.Row
            self.db.create_function('LOG1P', 1, math.log1p, deterministic=True)
            self.db.create_function('SQRT', 1, math.sqrt, deterministic=True)
            self.db.executescript('''
              CREATE TABLE predictions(event_id TEXT,provider TEXT,revision TEXT,series_id TEXT,unit TEXT,
                origin TEXT,recorded_at TEXT,horizon INTEGER,step INTEGER,valid_time TEXT,point REAL,context TEXT);
              CREATE INDEX forecast_task ON predictions(series_id,unit,origin,provider,revision);
              CREATE TABLE actuals(event_id TEXT,sequence INTEGER,series_id TEXT,unit TEXT,valid_time TEXT,
                source_available_at TEXT,recorded_at TEXT,value REAL);
              CREATE INDEX actual_visibility ON actuals(series_id,unit,valid_time,source_available_at,recorded_at);
            ''')
        else:
            raise ValueError('Unknown arm')

    def ingest(self, events, now):
        started = time.perf_counter()
        incoming = [e for e in events if e['recorded_at'] <= now and e['event_id'] not in self.seen]
        for event in incoming:
            if self.arm == 'gnomon':
                self.ledger.clock = FixedClock(datetime.fromisoformat(event['recorded_at']))
                if event['kind'] == 'forecast':
                    execution = execute(event)
                    self.ledger.record_execution(execution)
                    self.ledger.record_decision_summary(execution_id=execution.execution_id,
                        rationale='Shared shadow forecast event; context known at origin.', assumptions=[],
                        invalidation_conditions=['An observation or declared context is revised.'],
                        context=[dict(key=k, value=v, valid_from=event['request']['cutoff'],
                            valid_to=shifted(event['request']['cutoff'], 1),
                            source_available_at=event['request']['cutoff'], source_ref=event['event_id'])
                            for k, v in event['context'].items()])
                else:
                    self.ledger.append_actual(**{k: event[k] for k in
                        ('series_id', 'unit', 'valid_time', 'source_available_at', 'value')}, source_ref=event['event_id'])
            else:
                if event['kind'] == 'forecast':
                    req = event['request']
                    self.db.executemany('INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', [
                        (event['event_id'], event['provider'], event['revision'], req['series_id'], req['unit'],
                         req['cutoff'], event['recorded_at'], req['horizon'], i, t, value, canonical(event['context']))
                        for i, (t, value) in enumerate(zip(req['future_timestamps'], event['point']))])
                else:
                    self.db.execute('INSERT INTO actuals VALUES (?,?,?,?,?,?,?,?)', tuple(event[k] for k in
                        ('event_id', 'sequence', 'series_id', 'unit', 'valid_time', 'source_available_at', 'recorded_at', 'value')))
            self.seen.add(event['event_id'])
        if self.arm == 'sqlite':
            self.db.commit()
            self.writes = self.db.total_changes
        else:
            self.writes = self.ledger._committed_row_writes
        self.ingest_seconds += time.perf_counter() - started
        self.receipts.append(dict(now=now, event_ids=[e['event_id'] for e in incoming]))
        return self.receipts[-1]

    def query(self, query):
        started = time.perf_counter()
        try:
            if self.arm == 'gnomon':
                answer = self.ledger.compare_context(**query)
                window = answer['evidence_summary']['lifetime']
                return dict(matched_origins=answer['matched_origins'],
                    scores={r['provider']: r['score'] for r in window['ranking']},
                    ranking=[r['provider'] for r in window['ranking']])
            return sql_answer(self.sql(REFERENCE_SQL, parameters(query)), query['providers'])
        finally:
            self.query_seconds += time.perf_counter() - started

    def sql(self, statement, params):
        if self.arm != 'sqlite':
            raise ValueError('SQL tool belongs to the SQLite control')
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
        self.db.set_authorizer(lambda action, *args: sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY)
        ticks = 0

        def progress():
            nonlocal ticks
            ticks += 1
            return ticks > 2000  # two million VM steps per query

        self.db.set_progress_handler(progress, 1000)
        try:
            rows = self.db.execute(statement, params).fetchmany(501)
            if len(rows) > 500:
                raise ValueError('Result exceeds 500 rows; narrow or aggregate your query')
            result = [dict(row) for row in rows]
            if len(canonical(result).encode()) > 24000:
                raise ValueError('Result exceeds 24000 UTF-8 bytes; narrow or aggregate your query')
            return result
        finally:
            self.db.set_authorizer(None)
            self.db.set_progress_handler(None, 0)

    def close(self):
        if self.arm == 'sqlite':
            self.db.close()
        (self.path / 'state.json').write_text(canonical(dict(notebook=self.notebook,
            saved_queries=self.saved_queries, receipts=self.receipts, writes=self.writes,
            ingest_seconds=self.ingest_seconds, query_seconds=self.query_seconds)) + '\n')


def execute(event):
    """Typed fixed-candidate playback, same code for both arms, zero future truth."""
    engine = InferenceEngine()
    engine.register(event['provider'], lambda req: ForecastResult(point=tuple(event['point']),
        series_id=req.series_id, unit=req.unit, timestamps=req.future_timestamps),
        revision=event['revision'], deterministic=True, lifecycle='stateless')
    return engine.forecast(event['provider'], event['request'])
