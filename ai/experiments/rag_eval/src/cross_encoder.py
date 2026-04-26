"""Cross-encoder abstraction for reranking candidates."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from sentence_transformers import CrossEncoder as STCrossEncoder
except Exception:  # pragma: no cover
    STCrossEncoder = None  # type: ignore[assignment]


@dataclass
class CandidateAnswer:
    """Candidate answer unit for reranking."""

    answer_id: int
    term_id: int
    answer: str


class BaseCrossEncoderScorer:
    """Cross-encoder scorer base class."""

    def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """Return relevance scores for (query, answer) pairs."""
        raise NotImplementedError


class SentenceTransformerScorer(BaseCrossEncoderScorer):
    """Wrapper for sentence-transformers CrossEncoder models."""

    def __init__(self, model_name: str) -> None:
        if STCrossEncoder is None:
            raise RuntimeError("sentence-transformers is not available")
        self.model = STCrossEncoder(model_name)

    def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        scores = self.model.predict(list(pairs))
        return [float(score) for score in scores]


class LexicalFallbackScorer(BaseCrossEncoderScorer):
    """Offline fallback scorer based on lexical overlap."""

    @staticmethod
    def _keywords(text: str) -> List[str]:
        return re.findall(r"[가-힣A-Za-z0-9]+", text.lower())

    @staticmethod
    def _token_overlap(a: str, b: str) -> float:
        a_tokens = set(LexicalFallbackScorer._keywords(a))
        b_tokens = set(LexicalFallbackScorer._keywords(b))
        if not a_tokens and not b_tokens:
            return 0.0
        union = a_tokens | b_tokens
        if not union:
            return 0.0
        return len(a_tokens & b_tokens) / len(union)

    def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        scores: List[float] = []
        for query, answer in pairs:
            overlap = self._token_overlap(query, answer)
            ratio = SequenceMatcher(None, query, answer).ratio()
            score = ((overlap + ratio) / 2.0) * 10.0
            scores.append(float(score))
        return scores


class CrossEncoderRegistry:
    """Loads and provides named cross-encoder scorers."""

    def __init__(self, model_map: Dict[str, str], scorer_mode: str = "lexical") -> None:
        self.model_map = model_map
        self.scorer_mode = scorer_mode
        self._cache: Dict[str, BaseCrossEncoderScorer] = {}

    def get(self, name: str) -> BaseCrossEncoderScorer:
        """Get scorer by logical model name."""
        if name in self._cache:
            return self._cache[name]

        if self.scorer_mode == "lexical":
            scorer: BaseCrossEncoderScorer = LexicalFallbackScorer()
        else:
            model_name = self.model_map[name]
            scorer = SentenceTransformerScorer(model_name)

        self._cache[name] = scorer
        return scorer


def rerank_candidates(
    query: str,
    candidates: Sequence[CandidateAnswer],
    scorer: BaseCrossEncoderScorer,
    threshold: float,
    top_n: int,
) -> List[Tuple[CandidateAnswer, float]]:
    """Rerank and filter candidates by cross-encoder scores."""
    if not candidates:
        return []

    pairs = [(query, candidate.answer) for candidate in candidates]
    scores = scorer.score_pairs(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
    filtered = [(candidate, score) for candidate, score in ranked if score >= threshold]
    return filtered[:top_n]
