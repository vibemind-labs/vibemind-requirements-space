# -*- coding: utf-8 -*-
"""Fail-closed: an unreachable model aborts with LLMUnavailableError instead of
inventing scores, rewrites or an empty "successful" result. No model is called."""
from __future__ import annotations

import pytest
import requests

from backend.core import llm, settings
from backend.core.llm import LLMUnavailableError
from arch_team.agents import fail_closed
from arch_team.tools import validation_tools

KEYS = ["clarity", "testability"]


@pytest.fixture()
def ohne_mock(monkeypatch):
    monkeypatch.setattr(settings, "MOCK_MODE", False)
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")


@pytest.fixture()
def mit_mock(monkeypatch):
    monkeypatch.setattr(settings, "MOCK_MODE", True)
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")


class _KaputterClient:
    """Stands in for the OpenAI client; every call fails like an unreachable gateway."""

    def __init__(self, *a, **k):
        self.chat = self
        self.completions = self

    def create(self, **_):
        raise ConnectionError("gateway unreachable")


class _LeererClient(_KaputterClient):
    """Answers, but leaves one requirement out of a batch."""

    def create(self, **_):
        class M:
            content = '{"results": [{"id": "R1", "details": [{"criterion": "clarity", "score": 0.9}]}]}'

        class C:
            message = M()

        class R:
            choices = [C()]

        return R()


def _mit_client(monkeypatch, client_cls):
    monkeypatch.setattr(settings, "MOCK_MODE", False)
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-key-not-real")
    monkeypatch.setattr(llm, "_OpenAIClient", client_cls)
    monkeypatch.setattr(llm, "OPENAI_IS_V1_RUNTIME", True)
    monkeypatch.setattr(settings, "get_llm_config", lambda: {"api_key": "x", "base_url": "http://127.0.0.1:9/v1"})


def test_evaluate_without_key_raises(ohne_mock):
    with pytest.raises(LLMUnavailableError, match="evaluate: no_api_key"):
        llm.llm_evaluate("Das System muss in 2 s antworten.", KEYS, {})


def test_evaluate_model_error_raises_instead_of_invented_scores(monkeypatch):
    _mit_client(monkeypatch, _KaputterClient)
    with pytest.raises(LLMUnavailableError, match="ConnectionError"):
        llm.llm_evaluate("Das System muss in 2 s antworten.", KEYS, {})


def test_batch_missing_result_raises(monkeypatch):
    _mit_client(monkeypatch, _LeererClient)
    reqs = [{"id": "R1", "text": "A"}, {"id": "R2", "text": "B"}]
    with pytest.raises(LLMUnavailableError, match="missing_results: 1"):
        llm.llm_evaluate_batch(reqs, KEYS, {})


def test_rewrite_suggest_apply_raise(ohne_mock):
    with pytest.raises(LLMUnavailableError):
        llm.llm_rewrite("Text", {})
    with pytest.raises(LLMUnavailableError):
        llm.llm_suggest("Text", {})
    with pytest.raises(LLMUnavailableError):
        llm.llm_apply_with_suggestions("Text", {}, [{"correction": "x"}], "merge")


def test_explicit_mock_mode_still_marks_mock_scores(mit_mock):
    details = llm.llm_evaluate("Das System muss in 2 s antworten.", KEYS, {})
    assert [d["criterion"] for d in details] == KEYS
    assert all(d["mock"] for d in details)


def test_evaluation_api_unreachable_raises(ohne_mock, monkeypatch):
    def kaputt(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(validation_tools.requests, "post", kaputt)
    with pytest.raises(LLMUnavailableError, match="evaluate_api"):
        validation_tools.evaluate_requirement("Text")
    with pytest.raises(LLMUnavailableError, match="rewrite_api"):
        validation_tools.rewrite_requirement("Text")


def test_run_checks(ohne_mock):
    doc = [{"filename": "wp.md", "data": b"# Ziel\nEtwas bauen."}]
    with pytest.raises(LLMUnavailableError, match="no requirements"):
        fail_closed.check_mining([], doc)
    fail_closed.check_mining([], [{"filename": "leer.md", "data": b"  \n"}])
    fail_closed.check_mining([{"req_id": "R1"}], doc)
    with pytest.raises(LLMUnavailableError, match="2 of 5"):
        fail_closed.check_validation(2, 5)
    fail_closed.check_validation(0, 5)
    with pytest.raises(LLMUnavailableError, match="1 of 3 rewrites"):
        fail_closed.check_rewrite(1, 3)


def test_run_checks_pass_in_mock_mode(mit_mock):
    fail_closed.check_mining([], [{"filename": "wp.md", "data": b"Text"}])
    fail_closed.check_validation(3, 3)


def test_chunk_miner_adapter_error_raises(ohne_mock):
    from arch_team.agents.chunk_miner import ChunkMinerAgent

    class KaputterAdapter:
        def create(self, **_):
            raise ConnectionError("gateway unreachable")

    miner = object.__new__(ChunkMinerAgent)
    miner.adapter = KaputterAdapter()
    miner.temperature = 0.0
    with pytest.raises(LLMUnavailableError, match="chunk_mining"):
        miner._mine_chunk("Das System muss X.", {"sha1": "abc", "chunkIndex": 0}, None)
