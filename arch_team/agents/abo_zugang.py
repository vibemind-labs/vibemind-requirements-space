"""Subscription access for one-off runs: a hard per-run call cap and the switches the runner sets.

All model traffic of the pipeline goes through the OpenAI SDK's chat completions (mining, evaluation,
rewrite). ``install_call_cap`` wraps the sync and async ``create`` once per process; every call is
counted *before* it is sent, so call ``limit + 1`` never leaves the machine. Exceeding the cap raises
``CallCapReached``, a subclass of ``LLMUnavailableError``, so the fail-closed path ends the run with an
error instead of continuing with partial results.

Switches (environment, all off by default so the normal server behaviour is unchanged):
  ARCH_MINING_JSON=1     mine without tool calling, the model answers with a JSON object
  ARCH_EVAL_INPROCESS=1  evaluate requirements in this process instead of via HTTP to the backend
  ARCH_KG_SKIP=1         skip the knowledge-graph phase (no vector store, no extra model calls)
"""
from __future__ import annotations

import functools
import os
import threading
from typing import Any, Callable, Dict, Optional

from backend.core.llm import LLMUnavailableError


def flag(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in ("1", "true", "yes", "on")


class CallCapReached(LLMUnavailableError):
    """The run asked for more model calls than it was cleared for."""


class CallCounter:
    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        self.limit = int(limit)
        self.calls = 0
        self.by_model: Dict[str, int] = {}
        self._lock = threading.Lock()

    def take(self, model: Optional[str]) -> None:
        with self._lock:
            if self.calls >= self.limit:
                raise CallCapReached(f"call_cap: {self.limit} calls used, run stopped before call {self.calls + 1}")
            self.calls += 1
            key = str(model or "?")
            self.by_model[key] = self.by_model.get(key, 0) + 1


_counter: Optional[CallCounter] = None
_installed = False


def current_counter() -> Optional[CallCounter]:
    return _counter


def _wrap_sync(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _counter is not None:
            _counter.take(kwargs.get("model"))
        return original(self, *args, **kwargs)
    return create


def _wrap_async(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    async def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _counter is not None:
            _counter.take(kwargs.get("model"))
        return await original(self, *args, **kwargs)
    return create


def install_call_cap(limit: int) -> CallCounter:
    """Start a fresh counter for this run and make sure every chat completion passes through it."""
    global _counter, _installed
    from openai.resources.chat import completions as _cc  # type: ignore

    if not _installed:
        _cc.Completions.create = _wrap_sync(_cc.Completions.create)
        _cc.AsyncCompletions.create = _wrap_async(_cc.AsyncCompletions.create)
        _installed = True
    _counter = CallCounter(limit)
    return _counter


def estimate_calls(chunks: int, requirements: int, rewrite_attempts: int) -> Dict[str, int]:
    """Upper bound for one run in quick mode with the knowledge graph skipped.

    mining: one call per chunk; validation: one call per requirement (all criteria in one call);
    rewrite of a failed requirement: one rewrite plus, per attempt, one re-evaluation and (except after
    the last attempt) one more rewrite = 2 * attempts calls.
    """
    attempts = max(1, int(rewrite_attempts))
    mining = max(0, int(chunks))
    validation = max(0, int(requirements))
    rewrite_worst = validation * 2 * attempts
    return {"mining": mining, "validation": validation, "rewrite_worst": rewrite_worst,
            "total_worst": mining + validation + rewrite_worst}
