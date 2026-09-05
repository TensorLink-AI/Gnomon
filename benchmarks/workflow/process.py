"""Bounded local POSIX driver execution; not a sandbox or remote cancellation."""

import os
import selectors
import signal
import subprocess
import time


class ProcessLimit(RuntimeError):
    pass


def run_process(argv, *, input, timeout, stdout_limit=2_097_152, stderr_limit=65_536, env=None):
    """Kill this invocation's process group on timeout/output limit or completion.

    Descendants that deliberately create new sessions can escape; operator drivers
    are trusted. Generated code needs a separate isolation boundary. No shell is
    used. The byte ceilings apply during reads, not after buffering all output.
    """
    if os.name != "posix":
        raise OSError("bounded Workflow driver execution requires POSIX")
    if len(input) > 1_048_576:
        raise ProcessLimit("input_limit")
    started = time.monotonic()
    process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True, env=env)
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    limits = {"stdout": stdout_limit, "stderr": stderr_limit}
    position = 0
    try:
        with selectors.DefaultSelector() as selector:
            for pipe, name in ((process.stdout, "stdout"), (process.stderr, "stderr"), (process.stdin, "stdin")):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_WRITE if name == "stdin" else selectors.EVENT_READ, name)
            while selector.get_map():
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise ProcessLimit("timeout")
                for key, _ in selector.select(min(remaining, 0.1)):
                    pipe, name = key.fileobj, key.data
                    if name == "stdin":
                        try:
                            position += os.write(pipe.fileno(), input[position:position + 8192])
                        except BrokenPipeError:
                            position = len(input)
                        if position == len(input):
                            selector.unregister(pipe)
                            pipe.close()
                    else:
                        data = os.read(pipe.fileno(), min(8192, limits[name] + 1 - len(buffers[name])))
                        buffers[name].extend(data)
                        if len(buffers[name]) > limits[name]:
                            raise ProcessLimit("output_limit")
                        if not data:
                            selector.unregister(pipe)
                            pipe.close()
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise ProcessLimit("timeout")
            try:
                returncode = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                raise ProcessLimit("timeout") from None
            return subprocess.CompletedProcess(argv, returncode, bytes(buffers["stdout"]), bytes(buffers["stderr"]))
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for pipe in (process.stdin, process.stdout, process.stderr):
            pipe.close()
        process.wait()
