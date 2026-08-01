"""Route the official MTBench evaluation scripts' LLM calls to OpenRouter.

MTBench's evaluation scripts dispatch on ``--model`` to hardcoded API
functions in ``evaluation/api_call.py`` (OpenAI, Anthropic, Gemini,
DeepSeek), each needing its own key pasted into the source. This module
leaves every official script byte-for-byte unmodified and instead
replaces the ``evaluation.api_call`` send functions in memory before the
script runs, so that any ``--model`` choice is served by one OpenRouter
model through one ``OPENROUTER_API_KEY``.

The patched functions preserve each original's return shape (message
object with ``.content`` vs. plain string), because the official scripts
depend on those shapes.
"""

from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.openrouter import OpenRouterClient  # noqa: E402


def install(mtbench_root: Path, openrouter_model: str,
            temperature: float = 0.7) -> OpenRouterClient:
    """Import ``evaluation.api_call`` from the checkout and replace its
    send functions with OpenRouter-backed equivalents. Returns the shared
    client so callers can read usage totals."""
    root = str(mtbench_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    client = OpenRouterClient(openrouter_model, temperature=temperature,
                              max_tokens=4096)
    api_call = importlib.import_module("evaluation.api_call")

    def _text(content: str, max_tokens=None, **_ignored) -> str:
        return client.completions(
            [{"role": "user", "content": content}], n=1
        )[0]

    # Official callers use `.content` on this one (it returns a message
    # object); every other send function returns a plain string.
    def send_message_object(content, model=None, max_tokens=None,
                            temperature=None, timeout=None):
        return SimpleNamespace(content=_text(content))

    api_call.send_to_openai_chatgpt = send_message_object
    for name in ("send_to_anthropic_claude", "send_to_google_gemini",
                 "send_to_deepseek", "send_to_deepseek_r1",
                 "send_to_openai_o1"):
        if hasattr(api_call, name):
            setattr(api_call, name, lambda content, *a, _f=_text, **k: _f(content))
    return client


def run_official_script(mtbench_root: Path, script_relpath: str,
                        script_args: list[str], openrouter_model: str,
                        temperature: float = 0.7) -> OpenRouterClient:
    """Run one official evaluation script with patched LLM routing.

    Mirrors the official usage exactly: the script executes with its own
    directory as the working directory (their scripts rely on relative
    imports and ``sys.path.append("../..")``).
    """
    import os

    script_path = (mtbench_root / script_relpath).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"No such official script: {script_path}")
    client = install(mtbench_root, openrouter_model, temperature)

    script_dir = str(script_path.parent)
    previous_cwd = os.getcwd()
    previous_argv = sys.argv
    sys.path.insert(0, script_dir)
    try:
        os.chdir(script_dir)
        sys.argv = [script_path.name] + script_args
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        os.chdir(previous_cwd)
        sys.argv = previous_argv
    return client
