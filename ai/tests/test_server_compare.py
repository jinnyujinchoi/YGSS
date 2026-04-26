"""Regression tests for /server/compare endpoint behavior."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence, Tuple
import importlib.util

from fastapi import FastAPI
from fastapi.testclient import TestClient

AI_ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = AI_ROOT / "src" / "api" / "routes" / "server.py"
SPEC = importlib.util.spec_from_file_location("ygss_server_route", SERVER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Failed to load server module spec: {SERVER_PATH}")
server = importlib.util.module_from_spec(SPEC)
sys.modules["ygss_server_route"] = server
SPEC.loader.exec_module(server)


class DummyModel:
    """Minimal cross-encoder stub used for deterministic endpoint tests."""

    def __init__(self, scores: Sequence[float]) -> None:
        self._scores = list(scores)

    def predict(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        return self._scores[: len(pairs)]


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(server.router, prefix="/server")
    return TestClient(app)


def test_compare_returns_empty_for_empty_candidates() -> None:
    client = _make_client()
    response = client.post("/server/compare", json={"question": "질문", "candidateList": []})
    assert response.status_code == 200
    assert response.json() == {"results": []}


def test_compare_keeps_top3_and_applies_threshold(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_model", lambda: DummyModel([0.12, 0.91, -3.0, 0.44]))
    monkeypatch.setattr(server, "CROSS_ENCODER_THRESHOLD", -0.5)

    client = _make_client()
    payload = {
        "question": "중도인출이 가능한가요?",
        "candidateList": [
            {"termId": 101, "answer": "A"},
            {"termId": 102, "answer": "B"},
            {"termId": 103, "answer": "C"},
            {"termId": 104, "answer": "D"},
        ],
    }

    response = client.post("/server/compare", json=payload)
    assert response.status_code == 200

    result = response.json()["results"]
    assert len(result) == 3
    assert [row["termId"] for row in result] == [102, 104, 101]


def test_compare_contains_expected_answer_in_top3(monkeypatch) -> None:
    monkeypatch.setattr(server, "_get_model", lambda: DummyModel([0.35, -1.2, 0.77]))
    monkeypatch.setattr(server, "CROSS_ENCODER_THRESHOLD", -0.5)

    client = _make_client()
    payload = {
        "question": "퇴직연금 수령 방법 알려줘",
        "candidateList": [
            {"termId": 1, "answer": "오답 후보"},
            {"termId": 2, "answer": "정답과 무관한 문장"},
            {"termId": 3, "answer": "정답 후보"},
        ],
    }

    response = client.post("/server/compare", json=payload)
    assert response.status_code == 200
    result = response.json()["results"]

    # termId=3을 known-good pair로 간주하여 top-3 포함 회귀를 보장한다.
    returned_term_ids = {row["termId"] for row in result}
    assert 3 in returned_term_ids
