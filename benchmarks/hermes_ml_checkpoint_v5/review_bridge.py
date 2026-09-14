"""092 treatment: frozen 091 comparison on unchanged published 1.2.0 storage."""
import importlib.metadata

import history_091
from visible_agent_review import review


class ProductionHistory:
    def __init__(self, db):
        from gnomon.build_info import build_info
        if (importlib.metadata.version('gnomon-forecast') != '1.2.0'
                or build_info()['source_sha256'] != '9723394ccb6d9e11991b312e01bac47c767c69407b6b33d36971cb6e48b6a22e'):
            raise ValueError('092 requires the frozen published Gnomon 1.2.0 runtime')
        self.db = db

    def execution(self, *args, **kwargs):
        return self.db.execution(*args, **kwargs)

    def actuals_as_of(self, *args, **kwargs):
        return self.db.actuals_as_of(*args, **kwargs)

    def compare_history(self, **kwargs):
        return history_091.compare_history(self.db, **kwargs)


def corrected_review(db, records, task, evidence_math, path, **options):
    value = review(ProductionHistory(db), records, task, evidence_math, path, **options)
    value['comparison_implementation'] = 'development-history-091'
    value['storage_runtime'] = 'published-gnomon-1.2.0'
    return value
