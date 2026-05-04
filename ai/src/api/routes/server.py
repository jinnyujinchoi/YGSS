import os
from functools import lru_cache
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

router = APIRouter()

# STEP C defaults:
# - v3 selected ko-reranker: results/cross_encoder_comparison_20260502_113638.json
# - STEP C grid-search recommendation is applied via env override and can be updated without code changes.
DEFAULT_CROSS_ENCODER_MODEL = "Dongjin-kr/ko-reranker"
DEFAULT_CROSS_ENCODER_THRESHOLD = 0.516129
DEFAULT_CROSS_ENCODER_TOP_N = 3

CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", DEFAULT_CROSS_ENCODER_MODEL)
CROSS_ENCODER_THRESHOLD = float(
    os.getenv("CROSS_ENCODER_THRESHOLD", str(DEFAULT_CROSS_ENCODER_THRESHOLD))
)
CROSS_TOP_N = int(os.getenv("CROSS_TOP_N", str(DEFAULT_CROSS_ENCODER_TOP_N)))


@lru_cache(maxsize=1)
def _get_model() -> CrossEncoder:
    """Lazily initialize the cross-encoder once per process."""
    return CrossEncoder(CROSS_ENCODER_MODEL)

class Candidate(BaseModel):
    termId: int
    answer: str

class CompareRequest(BaseModel):
    question: str
    candidateList: List[Candidate]

@router.post("/compare")
def compare(req: CompareRequest):
    question = req.question
    candidates = req.candidateList

    if not candidates:  # 후보가 없으면 바로 반환
        return {"results": []}
    
    # 각 candidate.answer와 question 점수 계산
    scores = _get_model().predict([(question, c.answer) for c in candidates])

    # score와 candidate 객체를 튜플로 묶고 내림차순 정렬
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    filtered = [(c, s) for c, s in ranked if float(s) >= CROSS_ENCODER_THRESHOLD]

    top3 = filtered[:CROSS_TOP_N]
    # 결과 반환 (termId, answer, score 포함)
    result = [{"termId": c.termId, "answer": c.answer, "score": float(s)} for c, s in top3]
    return {"results": result}
