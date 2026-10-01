"""Bounded provider subprocess. A timeout never implies that a remote call did not run."""
import multiprocessing as mp
from queue import Empty
import tomllib
from gnomon import GnomonSession
from .storage import ServiceError


def validate_config(path):
    if path:
        with open(path, 'rb') as handle:
            cfg = tomllib.load(handle)
        if any(k in cfg for k in ('ledger_path', 'routers', 'memory', 'allow_outcome_writes')):
            raise ServiceError('INVALID_CONFIG', 'Provider config must not configure a ledger, routers, memory or outcome writes.')


def _forecast(config, provider, request, queue):
    try:
        validate_config(config)
        with GnomonSession.from_config(config) as session:
            queue.put(('ok', session.engine.forecast(provider, request, use_cache=False)))
    except Exception:
        queue.put(('error', None))  # Provider exceptions can contain secrets.


def forecast(config, provider, request, timeout=60):
    context = mp.get_context('spawn')
    queue = context.Queue()
    process = context.Process(target=_forecast, args=(config, provider, request, queue))
    process.start()
    try:
        try:
            state, result = queue.get(timeout=timeout)
        except Empty:
            raise ServiceError('OUTCOME_UNKNOWN', 'Forecast timed out; external execution may have occurred.', 504) from None
        if state != 'ok':
            raise ServiceError('OUTCOME_UNKNOWN', 'Provider execution failed; inspect provider status before retrying.', 502)
        return result
    finally:
        process.join(0.2)
        if process.is_alive():
            process.terminate()
            process.join(2)
        if process.is_alive():
            process.kill()
            process.join()
        queue.close()
