"""Cross-encoder abstraction for reranking candidates."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Dict, List, Sequence, Tuple

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from sentence_transformers import CrossEncoder as STCrossEncoder
except Exception:  # pragma: no cover
    STCrossEncoder = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:  # pragma: no cover
    AutoModelForSequenceClassification = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]


@dataclass
class CandidateAnswer:
    answer_id: int
    term_id: int
    answer: str


@dataclass
class PairScore:
    raw_score: float
    normalized_score: float


@dataclass
class CrossEncoderCandidate:
    name: str
    hf_id: str
    language: str


class BaseCrossEncoderScorer:
    def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        raise NotImplementedError

    def normalize_raw_scores(self, raw_scores: Sequence[float]) -> List[float]:
        return [1.0 / (1.0 + math.exp(-float(score))) for score in raw_scores]

    def score_pairs_detailed(self, pairs: Sequence[Tuple[str, str]]) -> List[PairScore]:
        raw_scores = self.score_pairs(pairs)
        normalized_scores = self.normalize_raw_scores(raw_scores)
        return [
            PairScore(raw_score=float(raw), normalized_score=float(norm))
            for raw, norm in zip(raw_scores, normalized_scores)
        ]


class SentenceTransformerScorer(BaseCrossEncoderScorer):
    def __init__(self, model_name: str) -> None:
        if STCrossEncoder is None:
            raise RuntimeError("sentence-transformers is not available")
        self.model = STCrossEncoder(model_name)

    def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        scores = self.model.predict(list(pairs))
        return [float(score) for score in scores]


class TransformersDirectScorer(BaseCrossEncoderScorer):
    def __init__(self, model_name: str) -> None:
        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise RuntimeError("transformers is not available")
        if torch is None:
            raise RuntimeError("torch is not available")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        left = [pair[0] for pair in pairs]
        right = [pair[1] for pair in pairs]
        encoded = self.tokenizer(
            left,
            right,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits
        if logits.ndim == 2 and logits.shape[1] > 1:
            values = logits[:, 1]
        else:
            values = logits.reshape(-1)
        return [float(item) for item in values.detach().cpu().numpy().tolist()]


class CrossEncoderRegistry:
    def __init__(self, candidates: Sequence[CrossEncoderCandidate]) -> None:
        self.candidates = {item.name: item for item in candidates}
        self._cache: Dict[str, BaseCrossEncoderScorer] = {}

    def get(self, name: str) -> BaseCrossEncoderScorer:
        if name in self._cache:
            return self._cache[name]
        if name not in self.candidates:
            raise RuntimeError(f"Unknown model candidate: {name}")
        scorer, _ = _load_scorer(self.candidates[name].hf_id)
        self._cache[name] = scorer
        return scorer


def _load_scorer(model_name: str) -> Tuple[BaseCrossEncoderScorer, str]:
    errors: List[str] = []
    try:
        return SentenceTransformerScorer(model_name), "sentence-transformers"
    except Exception as exc:
        errors.append(f"sentence-transformers: {exc}")

    try:
        return TransformersDirectScorer(model_name), "transformers-direct"
    except Exception as exc:
        errors.append(f"transformers-direct: {exc}")

    raise RuntimeError(f"Failed to load cross-encoder '{model_name}'. Attempts: {' | '.join(errors)}")


def load_candidates_with_status(
    candidates: Sequence[CrossEncoderCandidate],
) -> Tuple[Dict[str, BaseCrossEncoderScorer], Dict[str, Dict[str, object]]]:
    loaded: Dict[str, BaseCrossEncoderScorer] = {}
    status: Dict[str, Dict[str, object]] = {}

    for candidate in candidates:
        started = time.perf_counter()
        attempts: List[str] = ["sentence-transformers", "transformers-direct"]
        try:
            scorer, loader = _load_scorer(candidate.hf_id)
            loaded[candidate.name] = scorer
            status[candidate.name] = {
                "status": "ok",
                "loader": loader,
                "hf_id": candidate.hf_id,
                "language": candidate.language,
                "attempted_loaders": attempts,
                "load_seconds": round(time.perf_counter() - started, 4),
            }
        except Exception as exc:
            status[candidate.name] = {
                "status": "failed",
                "hf_id": candidate.hf_id,
                "language": candidate.language,
                "attempted_loaders": attempts,
                "error": str(exc),
                "load_seconds": round(time.perf_counter() - started, 4),
            }
    return loaded, status


def rerank_candidates(
    query: str,
    candidates: Sequence[CandidateAnswer],
    scorer: BaseCrossEncoderScorer,
    threshold: float,
    top_n: int,
) -> List[Tuple[CandidateAnswer, float]]:
    if not candidates:
        return []
    pairs = [(query, candidate.answer) for candidate in candidates]
    scores = scorer.score_pairs(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
    filtered = [(candidate, score) for candidate, score in ranked if score >= threshold]
    return filtered[:top_n]


def rerank_candidates_detailed(
    query: str,
    candidates: Sequence[CandidateAnswer],
    scorer: BaseCrossEncoderScorer,
    raw_threshold: float,
    top_n: int,
) -> List[Tuple[CandidateAnswer, PairScore]]:
    if not candidates:
        return []
    pairs = [(query, candidate.answer) for candidate in candidates]
    detailed_scores = scorer.score_pairs_detailed(pairs)
    ranked = sorted(
        zip(candidates, detailed_scores),
        key=lambda item: item[1].raw_score,
        reverse=True,
    )
    filtered = [(candidate, score) for candidate, score in ranked if score.raw_score >= raw_threshold]
    return filtered[:top_n]
