# -*- coding: utf-8 -*-
"""Run-level fail-closed checks for the master workflow.

A run must not "succeed" when the model was unreachable: no requirements from a
non-empty document, or any validation/rewrite error, aborts the run with
LLMUnavailableError. Only an explicit MOCK_MODE (tests, local demos) lets it pass.
"""
from __future__ import annotations

from typing import Any, Dict, List

from backend.core.llm import _fail


def check_mining(requirements: List[Dict[str, Any]], file_records: List[Dict[str, Any]]) -> None:
    has_text = any((rec.get("data") or b"").strip() for rec in file_records)
    if has_text and not requirements:
        _fail("mining", "no requirements from a non-empty document")


def check_validation(error_count: int, total: int) -> None:
    if error_count:
        _fail("validation", f"{error_count} of {total} requirements could not be evaluated")


def check_rewrite(error_count: int, total: int) -> None:
    if error_count:
        _fail("rewrite", f"{error_count} of {total} rewrites failed")
