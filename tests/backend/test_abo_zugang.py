# -*- coding: utf-8 -*-
"""Subscription access for one-off runs: call cap, JSON mining, in-process evaluation, runner refusals.
No model is called: the OpenAI SDK talks to an in-memory httpx transport, adapters are fakes."""
from __future__ import annotations

import json

import httpx
import pytest

from backend.core import settings
from backend.core.llm import LLMUnavailableError
from arch_team import abo_lauf
from arch_team.agents import abo_zugang, chunk_miner
from arch_team.agents.abo_zugang import CallCapReached, estimate_calls, install_call_cap
from arch_team.tools import validation_tools

ANTWORT = {"id": "x", "object": "chat.completion", "created": 0, "model": "haiku",
           "choices": [{"index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"}}]}


@pytest.fixture()
def ohne_mock(monkeypatch):
    monkeypatch.setattr(settings, "MOCK_MODE", False)


@pytest.fixture()
def ohne_zaehler(monkeypatch):
    monkeypatch.setattr(abo_zugang, "_counter", None)


def _client(seen):
    from openai import OpenAI

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["model"])
        return httpx.Response(200, json=ANTWORT)

    return OpenAI(api_key="test", base_url="http://127.0.0.1:9/v1",
                  http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_grenze_haelt_den_naechsten_aufruf_vor_dem_senden(ohne_zaehler) -> None:
    seen: list = []
    zaehler = install_call_cap(3)
    client = _client(seen)
    for _ in range(3):
        client.chat.completions.create(model="haiku", messages=[{"role": "user", "content": "hi"}])
    with pytest.raises(CallCapReached) as e:
        client.chat.completions.create(model="haiku", messages=[{"role": "user", "content": "hi"}])
    assert isinstance(e.value, LLMUnavailableError)
    assert seen == ["haiku"] * 3 and zaehler.calls == 3 and zaehler.by_model == {"haiku": 3}


def test_neuer_lauf_beginnt_bei_null(ohne_zaehler) -> None:
    seen: list = []
    install_call_cap(1)
    _client(seen).chat.completions.create(model="haiku", messages=[{"role": "user", "content": "a"}])
    zweiter = install_call_cap(2)
    _client(seen).chat.completions.create(model="haiku", messages=[{"role": "user", "content": "b"}])
    assert zweiter.calls == 1 and len(seen) == 2


def test_schaetzung() -> None:
    assert estimate_calls(3, 10, 1) == {"mining": 3, "validation": 10, "rewrite_worst": 20, "total_worst": 33}
    assert estimate_calls(2, 5, 3)["rewrite_worst"] == 30


class _Adapter:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _miner(answer):
    m = object.__new__(chunk_miner.ChunkMinerAgent)
    m.temperature = 0.0
    m.adapter = _Adapter(answer)
    return m


PAYLOAD = {"sha1": "abcdef123", "chunkIndex": 2, "sourceFile": "white_paper.md"}


def test_json_mining_ohne_werkzeuge(monkeypatch, ohne_mock) -> None:
    monkeypatch.setenv("ARCH_MINING_JSON", "1")
    antwort = '```json\n{"requirements": [{"title": "Kunden zahlen kontaktlos", "tag": "functional"},' \
              ' {"title": "Antwort unter 2 s", "tag": "performance"}]}\n```'
    m = _miner(antwort)
    items = m._mine_chunk("Der Laden soll ...", PAYLOAD, model_override="haiku")
    assert [i["title"] for i in items] == ["Kunden zahlen kontaktlos", "Antwort unter 2 s"]
    assert [i["req_id"] for i in items] == ["REQ-abcdef-002", "REQ-abcdef-002-a"]  # existing suffix rule
    call = m.adapter.calls[0]
    assert "tools" not in call and "tool_choice" not in call and call["model"] == "haiku"
    assert '"requirements"' in call["messages"][0]["content"]


def test_json_mining_leer_ist_erlaubt(monkeypatch, ohne_mock) -> None:
    monkeypatch.setenv("ARCH_MINING_JSON", "1")
    assert _miner('{"requirements": []}')._mine_chunk("Nur Einleitung.", PAYLOAD, None) == []


@pytest.mark.parametrize("antwort", ["Hier sind die Anforderungen: ...", '["a"]', '{"foo": 1}',
                                     ConnectionError("shim down")])
def test_json_mining_kaputte_antwort_bricht_ab(monkeypatch, ohne_mock, antwort) -> None:
    monkeypatch.setenv("ARCH_MINING_JSON", "1")
    with pytest.raises(LLMUnavailableError):
        _miner(antwort)._mine_chunk("Text", PAYLOAD, None)


def test_bewertung_im_prozess(monkeypatch, ohne_mock) -> None:
    from backend.services.evaluation_service import EvaluationService

    monkeypatch.setenv("ARCH_EVAL_INPROCESS", "1")
    monkeypatch.setattr(validation_tools.requests, "post",
                        lambda *a, **k: pytest.fail("no HTTP in in-process mode"))
    seen = {}

    def fake(self, text, *, criteria_keys=None, **_):
        seen["args"] = (text, criteria_keys)
        return {"requirementText": text, "evaluation": [{"criterion": "clarity", "score": 0.8}],
                "score": 0.8, "verdict": "pass"}

    monkeypatch.setattr(EvaluationService, "evaluate_single", fake)
    r = validation_tools.evaluate_requirement("Kunden zahlen kontaktlos", ["clarity"])
    assert r == {"score": 0.8, "verdict": "pass", "evaluation": [{"criterion": "clarity", "score": 0.8}]}
    assert seen["args"] == ("Kunden zahlen kontaktlos", ["clarity"])


def test_bewertung_im_prozess_fehler_bricht_ab(monkeypatch, ohne_mock) -> None:
    from backend.services.evaluation_service import EvaluationService

    monkeypatch.setenv("ARCH_EVAL_INPROCESS", "1")

    def kaputt(self, *a, **k):
        raise RuntimeError("criteria table missing")

    monkeypatch.setattr(EvaluationService, "evaluate_single", kaputt)
    with pytest.raises(LLMUnavailableError):
        validation_tools.evaluate_requirement("x")


def test_laeufer_schaetzt_ohne_modell(tmp_path, capsys, monkeypatch, ohne_zaehler) -> None:
    for k in [*abo_lauf.SWITCHES, "ARCH_MAX_CALLS", "SQLITE_PATH"]:
        monkeypatch.delenv(k, raising=False)  # main() sets defaults; monkeypatch removes them again
    wp = tmp_path / "white_paper.md"
    wp.write_text("# Ziel\n\n" + "Der Hofladen bietet Selbstbedienung rund um die Uhr. " * 20, encoding="utf-8")
    assert abo_lauf.main([str(wp), "--schaetzen", "--annahme-je-chunk", "5"]) == 0
    s = json.loads(capsys.readouterr().out)
    assert s["chunks"] >= 1 and s["assumed_requirements"] == 5 * s["chunks"]
    assert s["total_worst"] == s["chunks"] + 3 * s["assumed_requirements"] and s["cap"] == 120
    assert abo_zugang.current_counter() is None


def test_laeufer_verweigert_ohne_freigabe_und_mit_mock(tmp_path, monkeypatch, ohne_zaehler) -> None:
    wp = tmp_path / "white_paper.md"
    wp.write_text("Text", encoding="utf-8")
    for k in [*abo_lauf.SWITCHES, "SQLITE_PATH"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(abo_lauf, "run", lambda *a, **k: pytest.fail("must not run"))
    assert abo_lauf.main([str(wp)]) == 2
    assert abo_lauf.main([str(wp), "--freigabe", "0"]) == 2
    assert abo_lauf.main([str(tmp_path / "fehlt.md"), "--freigabe", "10"]) == 2
    monkeypatch.setattr(settings, "MOCK_MODE", True)
    assert abo_lauf.main([str(wp), "--freigabe", "10"]) == 2


def test_waechter_stoppt_vor_der_validierung(monkeypatch, ohne_zaehler) -> None:
    monkeypatch.setenv("REWRITE_MAX_ATTEMPTS", "1")
    monkeypatch.setattr(chunk_miner.ChunkMinerAgent, "mine_files_or_texts_collect",
                        lambda self, *a, **k: [{"req_id": f"R{i}"} for i in range(10)])
    install_call_cap(25)
    abo_lauf._install_mining_guard(25)
    m = object.__new__(chunk_miner.ChunkMinerAgent)
    with pytest.raises(CallCapReached, match="stopped before validation"):
        m.mine_files_or_texts_collect([])  # 0 used + 10 + 20 = 30 > 25
    abo_lauf._install_mining_guard(40)  # already guarded: the first cap stays
    install_call_cap(40)
    monkeypatch.setattr(chunk_miner.ChunkMinerAgent, "mine_files_or_texts_collect",
                        lambda self, *a, **k: [{"req_id": "R1"}])
    abo_lauf._install_mining_guard(40)
    assert len(m.mine_files_or_texts_collect([])) == 1  # 0 + 1 + 2 = 3 <= 40
