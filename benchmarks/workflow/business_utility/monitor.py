"""Resource guard launching the existing workflow CLI unchanged, one arm at a time."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

GIB = 1024**3


def memory_available(proc=Path("/proc"), cgroup=Path("/sys/fs/cgroup")):
    values = dict(line.split(":", 1) for line in (proc / "meminfo").read_text().splitlines())
    available = int(values["MemAvailable"].split()[0])*1024
    # Container /proc/meminfo can report the physical host's much larger RAM.
    # Treat charged file cache conservatively; do not assume it will be reclaimed.
    boundary = cgroup / "memory.max"
    if boundary.exists():
        maximum = boundary.read_text().strip()
        if maximum != "max":
            available = min(available, max(0, int(maximum)-int((cgroup / "memory.current").read_text())))
    return available


def cpu_ticks(proc=Path("/proc")):
    values = [int(v) for v in (proc / "stat").read_text().splitlines()[0].split()[1:9]]
    return sum(values), values[3]+values[4]


def descendants(root, known, proc=Path("/proc")):
    """Track only the owned PID and descendants, bound to process start times."""
    entries = {}
    for path in proc.iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rpartition(")")[2].split()
            entries[int(path.name)] = (int(fields[1]), int(fields[19]), int(fields[21])*os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            continue
    owned = {pid for pid, start in known.items() if pid in entries and entries[pid][1] == start}
    if not known and root in entries:
        owned.add(root)
    while True:
        following = owned | {pid for pid, (parent, _, _) in entries.items() if parent in owned}
        if following == owned:
            break
        owned = following
    known.update({pid: entries[pid][1] for pid in owned})
    return sum(entries[pid][2] for pid in owned), len(owned)


def resource_stop(available, rss):
    return available < 8*GIB or rss > 3*GIB


def stop_group(process, known):
    # run_workflow owns additional process groups; bind descendants by start time.
    descendants(process.pid, known)
    def signal_owned(sig):
        for pid, born in list(known.items()):
            try:
                fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
                if int(fields[19]) == born:
                    os.kill(pid, sig)
            except (OSError, ValueError, IndexError):
                pass
    signal_owned(signal.SIGTERM)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    descendants(process.pid, known)
    signal_owned(signal.SIGKILL)
    # The leader may exit before its descendants. Kill only this owned group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run(args):
    from gnomon.product_contract import __version__
    if __version__ != "1.2.0":
        raise ValueError("Live business trials require the identified 1.2.0 baseline; this runtime differs")
    experiment = Path(args.experiment).resolve()
    spec = json.loads(experiment.read_text())
    budget = spec["common"]["budget"]
    if budget["jobs"] != 1 or budget["infrastructure_retries"] != 0 or budget["timeout_seconds"] != 180:
        raise ValueError("registered launch requires one worker, no retries and 180 seconds/task")
    # Require the explicit freeze commit, including experiment and corpus bytes.
    root = Path(__file__).resolve().parents[3]
    targets = ["src/gnomon", "benchmarks/workflow", str(experiment.parent), str(Path(args.cases).resolve().parent)]
    state = subprocess.run(["git", "status", "--porcelain", "--", *targets], cwd=root,
                           capture_output=True, text=True, check=True).stdout
    if state:
        raise ValueError("commit release/runtime, harness and prepared inputs before live execution")
    subprocess.run(["git", "ls-files", "--error-unmatch", str(experiment), str(Path(args.cases).resolve())],
                   cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if resource_stop(memory_available(), 0):
        raise RuntimeError("less than 8 GiB available; no trial started")
    lock_path = root / "results/business-utility.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    argv = [sys.executable, "-m", "benchmarks.workflow.run_workflow", "--cases", str(Path(args.cases).resolve()),
            "--experiment", str(experiment), "--arm", args.arm, "--arm-command", shlex.join(spec["command"]),
            "--output-dir", str(output.resolve()), "--timeout", "180", "--jobs", "1", "--infrastructure-retries", "0"]
    cores = sorted(os.sched_getaffinity(0))[:2]
    def child_limits():
        os.nice(10)
        os.sched_setaffinity(0, cores)
    process = subprocess.Popen(argv, start_new_session=True, preexec_fn=child_limits)
    known, last = {}, cpu_ticks()
    reason = None
    try:
        with (output / "resources.jsonl").open("x") as log:
            while process.poll() is None:
                current = cpu_ticks()
                total, idle = current[0]-last[0], current[1]-last[1]
                last = current
                rss, count = descendants(process.pid, known)
                available = memory_available()
                log.write(json.dumps({"time_unix": time.time(), "host_cpu_percent": 100*(1-idle/total) if total else None,
                    "effective_available_bytes": available, "availability_basis": "minimum of host available and cgroup uncharged memory",
                    "owned_process_rss_bytes": rss, "owned_process_count": count,
                    "container_memory": "separately capped by existing backend; not included in host descendant RSS"}) + "\n")
                log.flush()
                if resource_stop(available, rss):
                    reason = "registered_memory_stop"
                    stop_group(process, known)
                    break
                time.sleep(2)
    except BaseException:
        stop_group(process, known)
        raise
    finally:
        if process.poll() is None:
            stop_group(process, known)
        (output / "resource-stop.json").write_text(json.dumps({"reason": reason, "returncode": process.returncode,
            "automatic_restart": False, "container_cleanup": "existing backend cleanup; hard-kill survivors have backend lifetime timers"}, indent=2)+"\n")
    return process.returncode if not reason else 75


if __name__ == "__main__":
    def requested_stop(signum, frame):
        raise KeyboardInterrupt("requested stop")
    signal.signal(signal.SIGTERM, requested_stop)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--arm", choices=["ordinary", "lean", "full"], required=True)
    parser.add_argument("--output-dir", required=True)
    raise SystemExit(run(parser.parse_args()))
