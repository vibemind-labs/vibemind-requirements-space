"""One-off Arch-Factory run over a subscription route, with a call count cleared in advance.

    python -m arch_team.abo_lauf WHITE_PAPER.md --schaetzen
        counts chunks and prints the call estimate; no model call.
    python -m arch_team.abo_lauf WHITE_PAPER.md --freigabe N --out result.json
        runs mining -> validation -> rewrite (quick mode, knowledge graph off) with at most
        min(N, ARCH_MAX_CALLS) model calls. Without --freigabe nothing is sent.

The model endpoint comes from the environment as usual (OPENROUTER_BASE_URL / OPENROUTER_API_KEY /
OPENAI_MODEL); for the subscription route the base URL points at a local OpenAI-compatible shim.
SQLITE_PATH defaults to ./arch_lauf.db; schema and criteria are created there before the first call.
After mining, the worst case for the rest of the run is computed; if it does not fit the cleared
number the run stops before validation. Any model failure ends the run with an error (fail-closed).
Exit codes: 0 completed, 1 stopped with error, 2 refused to start.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_MAX_CALLS = 120
SWITCHES = {"ARCH_MINING_JSON": "1", "ARCH_EVAL_INPROCESS": "1", "ARCH_KG_SKIP": "1", "REWRITE_MAX_ATTEMPTS": "1"}


def _max_calls() -> int:
    try:
        return max(1, int(os.environ.get("ARCH_MAX_CALLS", str(DEFAULT_MAX_CALLS))))
    except ValueError:
        return DEFAULT_MAX_CALLS


def _rewrite_attempts() -> int:
    try:
        return max(1, int(os.environ.get("REWRITE_MAX_ATTEMPTS", "1")))
    except ValueError:
        return 1


def count_chunks(path: Path, chunk_size: int, chunk_overlap: int) -> int:
    from backend.core.ingest import chunk_payloads, extract_texts

    records = extract_texts(path.name, path.read_bytes(), "")
    raw = [r for r in records if str(r.get("text") or "").strip()]
    return len(chunk_payloads(raw, max_tokens=chunk_size, overlap_tokens=chunk_overlap)) if raw else 0


def estimate(path: Path, chunk_size: int, chunk_overlap: int, reqs_per_chunk: int) -> Dict[str, Any]:
    from arch_team.agents.abo_zugang import estimate_calls

    chunks = count_chunks(path, chunk_size, chunk_overlap)
    assumed = chunks * reqs_per_chunk
    return {"chunks": chunks, "assumed_requirements": assumed, "rewrite_attempts": _rewrite_attempts(),
            **estimate_calls(chunks, assumed, _rewrite_attempts()), "cap": _max_calls()}


def _install_mining_guard(cap: int) -> None:
    """Stop right after mining when the worst case for validation + rewrite would pass the cap."""
    from arch_team.agents import chunk_miner
    from arch_team.agents.abo_zugang import CallCapReached, current_counter, estimate_calls

    cls = chunk_miner.ChunkMinerAgent
    if getattr(cls.mine_files_or_texts_collect, "_abo_guard", False):
        return
    original = cls.mine_files_or_texts_collect

    def guarded(self: Any, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        items = original(self, *args, **kwargs)
        counter = current_counter()
        used = counter.calls if counter else 0
        rest = estimate_calls(0, len(items), _rewrite_attempts())["total_worst"]
        if used + rest > cap:
            raise CallCapReached(
                f"estimate: {used} calls for mining + up to {rest} for {len(items)} requirements "
                f"> cleared {cap}; stopped before validation")
        return items

    guarded._abo_guard = True  # type: ignore[attr-defined]
    cls.mine_files_or_texts_collect = guarded


def run(path: Path, freigabe: int, model: str, chunk_size: int, chunk_overlap: int) -> Dict[str, Any]:
    from arch_team.agents.abo_zugang import install_call_cap
    from arch_team.agents.master_agent import run_master_workflow

    from backend.core.db import init_db

    init_db()  # local SQLite (SQLITE_PATH): schema and the criteria the evaluation reads; no model call
    cap = min(int(freigabe), _max_calls())
    counter = install_call_cap(cap)
    _install_mining_guard(cap)
    result = asyncio.run(run_master_workflow(
        [str(path)], model=model, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        use_llm_kg=False, mode="quick"))
    ok = bool(result.get("success"))
    return {
        "requirements": result.get("requirements") or [],
        "validation": result.get("validation_results") or {},
        "rewrite": result.get("rewrite_results"),
        "laeufe": {"aufrufe": counter.calls, "grenze": cap, "je_modell": dict(counter.by_model),
                   "abgebrochen": not ok, "grund": None if ok else str(result.get("error") or "unknown")},
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="arch_team.abo_lauf")
    ap.add_argument("white_paper", type=Path)
    ap.add_argument("--schaetzen", action="store_true", help="only count chunks and print the estimate")
    ap.add_argument("--freigabe", type=int, help="cleared number of model calls for this run")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "haiku"))
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--chunk-overlap", type=int, default=200)
    ap.add_argument("--annahme-je-chunk", type=int, default=6, help="assumed requirements per chunk (estimate only)")
    a = ap.parse_args(argv)

    for k, v in SWITCHES.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("SQLITE_PATH", "arch_lauf.db")  # own file per run directory, not the server's app.db
    if not a.white_paper.is_file():
        print(json.dumps({"error": "white_paper_fehlt"}), file=sys.stderr)
        return 2
    if a.schaetzen:
        print(json.dumps(estimate(a.white_paper, a.chunk_size, a.chunk_overlap, a.annahme_je_chunk), indent=2))
        return 0
    if not a.freigabe or a.freigabe < 1:
        print(json.dumps({"error": "freigabe_fehlt", "hint": "run --schaetzen first, then pass --freigabe N"}),
              file=sys.stderr)
        return 2
    from backend.core import settings
    if getattr(settings, "MOCK_MODE", False):
        print(json.dumps({"error": "mock_mode_an", "hint": "a run with MOCK_MODE would invent results"}),
              file=sys.stderr)
        return 2

    out = run(a.white_paper, a.freigabe, a.model, a.chunk_size, a.chunk_overlap)
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    print(json.dumps(out["laeufe"], ensure_ascii=False))
    return 1 if out["laeufe"]["abgebrochen"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
