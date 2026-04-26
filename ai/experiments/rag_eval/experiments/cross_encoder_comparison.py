"""Compare validated cross-encoder models and calibrate threshold."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import statistics
import time
from typing import Dict, List, Sequence, Tuple

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.cross_encoder import CandidateAnswer, CrossEncoderRegistry, rerank_candidates_detailed
from src.embedder import OpenAIEmbeddingConfig, create_embedder
from src.metrics import reciprocal_rank
from src.redis_search import DumpVectorSearcher, parse_chat_dummy_sql


@dataclass
class EvalSample:
    """Evaluation row."""

    question: str
    correct_answer_ids: List[int]
    category: str


@dataclass
class ModelComparisonResult:
    """Per-model comparison metric output."""

    alias: str
    model_name: str
    top3_accuracy: float
    mrr: float
    avg_latency_ms: float
    p95_latency_ms: float
    query_count: int


@dataclass
class ThresholdCalibration:
    """Threshold calibration output."""

    model_alias: str
    model_name: str
    best_threshold: float
    best_f1: float
    precision_at_best: float
    recall_at_best: float
    positive_count: int
    negative_count: int


def _load_config(root: Path) -> Dict:
    cfg_path = root / "config.yaml"
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def _load_eval_set(path: Path) -> List[EvalSample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [EvalSample(**item) for item in payload]


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_validation_results(path: Path) -> List[Dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("results", []))


def _build_candidates_for_query(
    question: str,
    query_vector: np.ndarray,
    searcher: DumpVectorSearcher,
    bi_top_k: int,
    bi_threshold: float,
):
    bi_hits = searcher.search(query_vector=query_vector, k=bi_top_k, min_similarity=bi_threshold)
    return [
        CandidateAnswer(answer_id=hit.answer_id, term_id=hit.term_id, answer=hit.answer)
        for hit in bi_hits
    ]


def _evaluate_model(
    alias: str,
    model_name: str,
    eval_set: Sequence[EvalSample],
    query_vectors: Sequence[np.ndarray],
    searcher: DumpVectorSearcher,
    bi_top_k: int,
    bi_threshold: float,
) -> Tuple[ModelComparisonResult, Dict[str, List[float]], Dict[int, List[CandidateAnswer]]]:
    registry = CrossEncoderRegistry(model_map={alias: model_name}, scorer_mode="sentence-transformers")
    scorer = registry.get(alias)

    reciprocal_ranks: List[float] = []
    top3_hits: List[float] = []
    latencies_ms: List[float] = []

    pair_scores = {"positive": [], "negative": []}
    query_candidates: Dict[int, List[CandidateAnswer]] = {}

    for idx, (sample, query_vector) in enumerate(zip(eval_set, query_vectors)):
        candidates = _build_candidates_for_query(
            question=sample.question,
            query_vector=query_vector,
            searcher=searcher,
            bi_top_k=bi_top_k,
            bi_threshold=bi_threshold,
        )
        query_candidates[idx] = candidates
        if not candidates:
            reciprocal_ranks.append(0.0)
            top3_hits.append(0.0)
            latencies_ms.append(0.0)
            continue

        started = time.perf_counter()
        detailed = rerank_candidates_detailed(
            query=sample.question,
            candidates=candidates,
            scorer=scorer,
            raw_threshold=-1e9,
            top_n=len(candidates),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.append(float(elapsed_ms))

        predicted_ids = [candidate.answer_id for candidate, _ in detailed[:3]]
        reciprocal_ranks.append(reciprocal_rank(predicted_ids=predicted_ids, correct_ids=sample.correct_answer_ids))
        top3_hits.append(1.0 if any(item in set(sample.correct_answer_ids) for item in predicted_ids) else 0.0)

        for candidate, score in detailed:
            label = "positive" if candidate.answer_id in set(sample.correct_answer_ids) else "negative"
            pair_scores[label].append(float(score.raw_score))

    result = ModelComparisonResult(
        alias=alias,
        model_name=model_name,
        top3_accuracy=float(statistics.mean(top3_hits)) if top3_hits else 0.0,
        mrr=float(statistics.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        avg_latency_ms=float(statistics.mean(latencies_ms)) if latencies_ms else 0.0,
        p95_latency_ms=float(np.percentile(np.array(latencies_ms, dtype=np.float64), 95)) if latencies_ms else 0.0,
        query_count=len(eval_set),
    )
    return result, pair_scores, query_candidates


def _calibrate_threshold(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    model_alias: str,
    model_name: str,
) -> Tuple[ThresholdCalibration, List[Tuple[float, float, float, float]]]:
    if not positive_scores:
        raise RuntimeError("No positive scores found for threshold calibration")

    all_scores = np.array(list(positive_scores) + list(negative_scores), dtype=np.float64)
    labels = np.array([1] * len(positive_scores) + [0] * len(negative_scores), dtype=np.int32)

    low = float(np.percentile(all_scores, 1))
    high = float(np.percentile(all_scores, 99))
    thresholds = np.linspace(low, high, num=200)

    curve: List[Tuple[float, float, float, float]] = []
    best_f1 = -1.0
    best_threshold = float(thresholds[0])
    best_precision = 0.0
    best_recall = 0.0

    for threshold in thresholds:
        preds = (all_scores >= threshold).astype(np.int32)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        curve.append((float(threshold), float(precision), float(recall), float(f1)))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
            best_precision = float(precision)
            best_recall = float(recall)

    calibration = ThresholdCalibration(
        model_alias=model_alias,
        model_name=model_name,
        best_threshold=best_threshold,
        best_f1=best_f1,
        precision_at_best=best_precision,
        recall_at_best=best_recall,
        positive_count=len(positive_scores),
        negative_count=len(negative_scores),
    )
    return calibration, curve


def _plot_calibration(
    output_path: Path,
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    curve: Sequence[Tuple[float, float, float, float]],
    selected_threshold: float,
) -> None:
    thresholds = [row[0] for row in curve]
    f1_vals = [row[3] for row in curve]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(negative_scores, bins=35, alpha=0.6, label="negative", color="#e76f51")
    plt.hist(positive_scores, bins=35, alpha=0.6, label="positive", color="#2a9d8f")
    plt.axvline(selected_threshold, color="black", linestyle="--", linewidth=1.5, label="best F1 threshold")
    plt.title("Score Distribution")
    plt.xlabel("Raw score")
    plt.ylabel("Count")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(thresholds, f1_vals, color="#264653", linewidth=2.0, label="F1")
    plt.axvline(selected_threshold, color="black", linestyle="--", linewidth=1.5)
    plt.title("Threshold vs F1")
    plt.xlabel("Raw score threshold")
    plt.ylabel("F1")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _write_comparison_md(
    path: Path,
    rows: Sequence[ModelComparisonResult],
    selected_alias: str,
    selected_model_name: str,
) -> None:
    lines: List[str] = []
    lines.append("# Cross-Encoder Comparison")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().isoformat()}")
    lines.append(f"- Selected model alias: {selected_alias}")
    lines.append(f"- Selected model_id: {selected_model_name}")
    lines.append("")
    lines.append("| alias | model | Top-3 accuracy | MRR | avg latency (ms/query) | p95 latency (ms/query) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row.alias} | {row.model_name} | {row.top3_accuracy:.4f} | {row.mrr:.4f} | "
            f"{row.avg_latency_ms:.2f} | {row.p95_latency_ms:.2f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run comparison on validated candidates and calibrate the selected model threshold."""
    load_dotenv()

    root = Path(__file__).resolve().parent.parent
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(root)
    eval_set = _load_eval_set(root / "dataset" / "eval_set.json")
    answer_map = parse_chat_dummy_sql((root / "../../../exec/basic_insert.sql").resolve())

    validation_path = results_dir / "model_candidate_validation.json"
    if not validation_path.exists():
        raise FileNotFoundError(
            "Validation output not found. Run experiments/validate_cross_encoder_candidates.py first."
        )

    validation_rows = _load_validation_results(validation_path)
    eligible = [item for item in validation_rows if item.get("eligible_for_comparison")]
    if not eligible:
        raise RuntimeError("No eligible models found from validation report.")

    openai_cfg = OpenAIEmbeddingConfig(
        api_key_env=cfg["openai"]["api_key_env"],
        base_url_env=cfg["openai"]["base_url_env"],
        model=cfg["openai"]["embedding_model"],
    )
    embedder = create_embedder(
        provider_name=cfg["embedding"]["provider"],
        openai_config=openai_cfg,
        dimension=int(cfg["embedding"]["dimension"]),
    )

    searcher = DumpVectorSearcher.from_sql_records(records=answer_map, embed_text_fn=embedder.embed_texts)
    query_vectors = embedder.embed_texts([sample.question for sample in eval_set])

    bi_top_k = int(cfg["retrieval"]["bi_top_k"])
    bi_threshold = 0.45

    model_results: List[ModelComparisonResult] = []
    raw_scores_by_alias: Dict[str, Dict[str, List[float]]] = {}

    for item in eligible:
        alias = str(item["alias"])
        model_name = str(item["model_name"])
        result, pair_scores, _ = _evaluate_model(
            alias=alias,
            model_name=model_name,
            eval_set=eval_set,
            query_vectors=query_vectors,
            searcher=searcher,
            bi_top_k=bi_top_k,
            bi_threshold=bi_threshold,
        )
        model_results.append(result)
        raw_scores_by_alias[alias] = pair_scores

    by_alias = {row.alias: row for row in model_results}
    baseline_latency = by_alias.get("baseline_en").p95_latency_ms if "baseline_en" in by_alias else None

    selectable = model_results
    if baseline_latency is not None and baseline_latency > 0.0:
        selectable = [
            row
            for row in model_results
            if row.p95_latency_ms <= baseline_latency * 2.0
        ]
        if not selectable:
            selectable = model_results

    selected = sorted(selectable, key=lambda row: (row.top3_accuracy, row.mrr), reverse=True)[0]

    selected_scores = raw_scores_by_alias[selected.alias]
    calibration, curve = _calibrate_threshold(
        positive_scores=selected_scores["positive"],
        negative_scores=selected_scores["negative"],
        model_alias=selected.alias,
        model_name=selected.model_name,
    )

    model_tag = _sanitize(selected.model_name)
    calibration_png = results_dir / f"threshold_calibration_{model_tag}.png"
    calibration_json = results_dir / f"threshold_calibration_{model_tag}.json"
    _plot_calibration(
        output_path=calibration_png,
        positive_scores=selected_scores["positive"],
        negative_scores=selected_scores["negative"],
        curve=curve,
        selected_threshold=calibration.best_threshold,
    )

    calibration_payload = {
        "generated_at": datetime.now().isoformat(),
        "calibration": asdict(calibration),
        "curve": [
            {
                "threshold": row[0],
                "precision": row[1],
                "recall": row[2],
                "f1": row[3],
            }
            for row in curve
        ],
    }
    calibration_json.write_text(json.dumps(calibration_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    timestamp = _timestamp()
    comparison_json = results_dir / f"cross_encoder_comparison_{timestamp}.json"
    comparison_md = results_dir / f"cross_encoder_comparison_{timestamp}.md"

    payload = {
        "generated_at": datetime.now().isoformat(),
        "bi_encoder_context": {
            "bi_top_k": bi_top_k,
            "bi_threshold": bi_threshold,
            "note": "Top-3 comparison uses rank-only reranking (no model-specific raw threshold).",
        },
        "models": [asdict(row) for row in model_results],
        "selected_model": {
            "alias": selected.alias,
            "model_name": selected.model_name,
            "selection_reason": "highest Top-3 accuracy then MRR under p95 <= 2x baseline constraint",
            "recommended_cross_encoder_threshold": calibration.best_threshold,
        },
        "artifacts": {
            "calibration_json": str(calibration_json.relative_to(root)),
            "calibration_png": str(calibration_png.relative_to(root)),
        },
    }
    comparison_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_comparison_md(
        comparison_md,
        model_results,
        selected_alias=selected.alias,
        selected_model_name=selected.model_name,
    )

    print(f"[done] comparison json  : {comparison_json}")
    print(f"[done] comparison md    : {comparison_md}")
    print(f"[done] selected model   : {selected.model_name}")
    print(f"[done] threshold        : {calibration.best_threshold:.6f}")
    print(f"[done] calibration png  : {calibration_png}")


if __name__ == "__main__":
    main()
