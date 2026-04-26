import os
from functools import lru_cache
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

router = APIRouter()

# Step B calibration artifacts:
# - selected alias `mmarco_multilingual` maps to model_id
#   `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
# - ai/experiments/rag_eval/results/cross_encoder_comparison_20260426_180439.md
# - ai/experiments/rag_eval/results/threshold_calibration_cross-encoder_mmarco-mMiniLMv2-L12-H384-v1.json
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_CROSS_ENCODER_THRESHOLD = -2.4304255588090604
DEFAULT_CROSS_ENCODER_TOP_N = 3

CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", DEFAULT_CROSS_ENCODER_MODEL)
CROSS_ENCODER_THRESHOLD = float(
    os.getenv("CROSS_ENCODER_THRESHOLD", str(DEFAULT_CROSS_ENCODER_THRESHOLD))
)


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

    # STEP B에서는 top-N=3을 유지 (STEP C에서 튜닝)
    top3 = filtered[:DEFAULT_CROSS_ENCODER_TOP_N]
    # 결과 반환 (termId, answer, score 포함)
    result = [{"termId": c.termId, "answer": c.answer, "score": float(s)} for c, s in top3]
    return {"results": result}
