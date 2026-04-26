"""Evaluation metrics for retrieval and reranking."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Set


@dataclass
class QueryEvaluation:
    """Per-query evaluation record."""

    predicted_ids: List[int]
    correct_ids: List[int]


def precision_at_k(predicted_ids: Sequence[int], correct_ids: Sequence[int], k: int) -> float:
    """Compute Precision@K."""
    if k <= 0:
        return 0.0
    top_k = list(predicted_ids[:k])
    if not top_k:
        return 0.0
    correct = set(correct_ids)
    hits = sum(1 for item in top_k if item in correct)
    return hits / k


def recall_at_k(predicted_ids: Sequence[int], correct_ids: Sequence[int], k: int) -> float:
    """Compute Recall@K."""
    gold = set(correct_ids)
    if not gold:
        return 0.0
    top_k = list(predicted_ids[:k])
    hits = sum(1 for item in top_k if item in gold)
    return hits / len(gold)


def reciprocal_rank(predicted_ids: Sequence[int], correct_ids: Sequence[int]) -> float:
    """Compute reciprocal rank for a single query."""
    gold = set(correct_ids)
    for idx, item in enumerate(predicted_ids, start=1):
        if item in gold:
            return 1.0 / idx
    return 0.0


def compute_metric_summary(records: Sequence[QueryEvaluation], cutoffs: Sequence[int]) -> Dict[str, float]:
    """Aggregate metrics across all queries."""
    if not records:
        return {"mrr": 0.0, "no_result_rate": 1.0}

    summary: Dict[str, float] = {}
    for k in cutoffs:
        p_vals = [precision_at_k(row.predicted_ids, row.correct_ids, k) for row in records]
        r_vals = [recall_at_k(row.predicted_ids, row.correct_ids, k) for row in records]
        summary[f"precision_at_{k}"] = mean(p_vals)
        summary[f"recall_at_{k}"] = mean(r_vals)

    rr_vals = [reciprocal_rank(row.predicted_ids, row.correct_ids) for row in records]
    no_result = [1.0 if len(row.predicted_ids) == 0 else 0.0 for row in records]

    summary["mrr"] = mean(rr_vals)
    summary["no_result_rate"] = mean(no_result)
    return summary
