"""Concurrent driver for AnomLLM's official online control runner.

The official ``src/online_api.py`` walks the evaluation set one series at a
time. Every series is an independent request, so with a reasoning model whose
replies take minutes each, the 400-series ``trend`` split needs days of wall
clock. This driver keeps the official protocol byte-for-byte and only changes
the scheduling:

* the request is built by the official ``create_batch_api_configs()[variant]``,
* it is sent by the official ``send_openai_request``,
* the reply is appended to the official ``<variant>.jsonl`` in the official
  ``{'custom_id', 'request', 'response'}`` shape,
* already-present ``custom_id`` values are skipped, so a partial sequential run
  resumes here without redoing work,
* scoring stays the official ``src/result_agg.py``.

Deviations from ``online_api.py``, both deliberate:

* series are dispatched to a thread pool instead of a ``for`` loop;
* the upstream 503 branch sleeps in an unbounded ``while True``, which would
  wedge a worker forever. Here a 503 is retried on the same bounded backoff
  schedule as any other error.

Nothing here changes a prompt, a sampling parameter, or a prediction: the reply
for a given series does not depend on what the other series are doing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AnomLLM's official online control concurrently.",
    )
    parser.add_argument("--anomllm-root", required=True,
                        help="Path to a clone of rose-stl-lab/AnomLLM")
    parser.add_argument("--data", required=True,
                        help="Dataset name, e.g. trend, point, range")
    parser.add_argument("--model", required=True,
                        help="Model id as it appears in credentials.yml")
    parser.add_argument("--variant", default="0shot-text",
                        help="Official prompt variant (default: 0shot-text)")
    parser.add_argument("--workers", type=int, default=12,
                        help="Concurrent in-flight requests (default: 12)")
    parser.add_argument("--num-retries", type=int, default=4,
                        help="Attempts per series before giving up (default: 4)")
    parser.add_argument("--base-url", default=None,
                        help="Optional OpenAI-compatible endpoint override")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY",
                        help="Environment variable for --base-url credentials")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Keep secrets out of the official checkout and command line.  This also
    # lets the exact official request sender target any OpenAI-compatible
    # provider without rewriting credentials.yml.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from benchmarks.common.envfile import load_env_file  # noqa: E402
    from benchmarks.common.manifest import write_manifest  # noqa: E402
    load_env_file(repo_root)

    root = Path(args.anomllm_root).expanduser().resolve()
    # The official code resolves credentials.yml, data/ and results/ relative to
    # the checkout root, and imports its own modules as top-level names.
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))

    from config import create_batch_api_configs  # noqa: E402
    from data.synthetic import SyntheticDataset  # noqa: E402
    import openai_api  # noqa: E402
    from openai_api import send_openai_request  # noqa: E402

    if args.base_url:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise SystemExit(
                f"{args.api_key_env} is required when --base-url is set")
        openai_api.credentials[args.model] = {
            "api_key": api_key, "base_url": args.base_url}

    request_func = create_batch_api_configs()[args.variant]

    eval_dataset = SyntheticDataset(f"data/synthetic/{args.data}/eval/")
    eval_dataset.load()
    train_dataset = SyntheticDataset(f"data/synthetic/{args.data}/train/")
    train_dataset.load()

    results_dir = root / "results" / "synthetic" / args.data / args.model
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl_fn = results_dir / f"{args.variant}.jsonl"

    done: set[str] = set()
    if jsonl_fn.exists():
        # A row whose response is null is a transport failure that got
        # written, not an answer: counting it as done meant a transient
        # provider error became a permanently-scored empty prediction that
        # no resume ever retried. Null rows are dropped from the file (so
        # the official aggregator never scores them twice) and their
        # series re-enter the pending set.
        kept: list[str] = []
        dropped = 0
        with jsonl_fn.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("response") is None:
                    dropped += 1
                    continue
                kept.append(line)
                done.add(record["custom_id"])
        if dropped:
            jsonl_fn.write_text(
                "".join(entry + "\n" for entry in kept), encoding="utf-8"
            )
            print(f"dropped {dropped} null-response row(s) from "
                  f"{jsonl_fn.name}; those series will be retried",
                  flush=True)

    pending = []
    for index in range(1, len(eval_dataset) + 1):
        custom_id = (
            f"{args.data}_{args.model}_{args.variant}_{str(index).zfill(5)}"
        )
        if custom_id not in done:
            pending.append((index, custom_id))

    print(f"{len(eval_dataset)} series, {len(done)} already done, "
          f"{len(pending)} to run on {args.workers} workers", flush=True)
    if not pending:
        write_manifest(
            root / "gnomonbench" / "synthetic" / args.data /
            args.model.replace("/", "--") / args.variant,
            benchmark="anomllm", condition=f"control/{args.variant}",
            target=args.data, model=args.model, base_url=args.base_url,
            workers=args.workers, status="ok", completed=len(done),
        )
        return 0

    write_lock = threading.Lock()
    counter = {"finished": 0, "failed": 0, "null": 0}

    def run_one(item: tuple[int, str]) -> None:
        index, custom_id = item
        request = request_func(eval_dataset.series[index - 1], train_dataset)
        for attempt in range(args.num_retries):
            try:
                response = send_openai_request(request, args.model)
            except Exception as exc:  # noqa: BLE001 - official code does the same
                wait = 2 ** (attempt + 3)
                print(f"[{custom_id}] attempt {attempt + 1} failed: {exc}; "
                      f"retrying in {wait}s", flush=True)
                time.sleep(wait)
                continue
            record = {
                "custom_id": custom_id,
                "request": request,
                "response": response,
            }
            with write_lock:
                with jsonl_fn.open("a") as handle:
                    json.dump(record, handle)
                    handle.write("\n")
                counter["finished"] += 1
                if response is None:
                    counter["null"] += 1
                seen = counter["finished"] + counter["failed"]
                print(f"[{seen}/{len(pending)}] {custom_id} "
                      f"({'null' if response is None else 'ok'})", flush=True)
            return
        with write_lock:
            counter["failed"] += 1
        print(f"[{custom_id}] gave up after {args.num_retries} attempts",
              flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, item) for item in pending]
        for future in as_completed(futures):
            future.result()

    print(f"done: {counter['finished']} written "
          f"({counter['null']} with null content), "
          f"{counter['failed']} abandoned", flush=True)
    total_complete = len(done) + counter["finished"] - counter["null"]
    status = "ok" if total_complete == len(eval_dataset) else "incomplete"
    write_manifest(
        root / "gnomonbench" / "synthetic" / args.data /
        args.model.replace("/", "--") / args.variant,
        benchmark="anomllm", condition=f"control/{args.variant}",
        target=args.data, model=args.model, base_url=args.base_url,
        workers=args.workers, status=status, completed=total_complete,
        expected=len(eval_dataset), failed=counter["failed"],
        null_responses=counter["null"],
    )
    return 0 if status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
