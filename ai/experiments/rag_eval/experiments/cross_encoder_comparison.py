"""Compare configured cross-encoder candidates and save metrics artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import statistics
import time
from typing import Dict, List, Sequence, Tuple

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.cross_encoder import (
    CandidateAnswer,
    CrossEncoderCandidate,
    load_candidates_with_status,
    rerank_candidates_detailed,
)
from src.embedder import OpenAIEmbeddingConfig, create_embedder
from src.metrics import reciprocal_rank
from src.redis_search import DumpVectorSearcher, parse_chat_dummy_sql


@dataclass
class ModelComparisonResult:
    alias: str
    model_name: str
    top3_accuracy: float
    mrr: float
    avg_latency_ms: float
    p95_latency_ms: float
    query_count: int


def _load_config(root: Path) -> Dict:
    return yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))


def _load_eval_set(path: Path) -> List[Dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _plot_distribution(path: Path, positive_scores: Sequence[float], negative_scores: Sequence[float], label: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.hist(negative_scores, bins=35, alpha=0.6, label="negative", color="#e76f51")
    plt.hist(positive_scores, bins=35, alpha=0.6, label="positive", color="#2a9d8f")
    plt.title(f"Raw Score Distribution: {label}")
    plt.xlabel("Raw score")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _f1_curve(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> Tuple[List[Dict[str, float]], float]:
    if not positive_scores:
        return [], 0.0
    all_scores = np.array(list(positive_scores) + list(negative_scores), dtype=np.float64)
    labels = np.array([1] * len(positive_scores) + [0] * len(negative_scores), dtype=np.int32)
    thresholds = np.linspace(float(np.percentile(all_scores, 1)), float(np.percentile(all_scores, 99)), num=100)
    rows: List[Dict[str, float]] = []
    best_f1 = 0.0
    for threshold in thresholds:
        preds = (all_scores >= threshold).astype(np.int32)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1})
        best_f1 = max(best_f1, f1)
    return rows, best_f1


def _plot_f1_curve(path: Path, rows: Sequence[Dict[str, float]], label: str) -> None:
    if not rows:
        return
    plt.figure(figsize=(10, 4))
    plt.plot([r["threshold"] for r in rows], [r["f1"] for r in rows], color="#264653")
    plt.title(f"F1 Curve: {label}")
    plt.xlabel("Raw score threshold")
    plt.ylabel("F1")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    load_dotenv()

    root = Path(__file__).resolve().parent.parent
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(root)
    eval_set = _load_eval_set(root / cfg["data"]["eval_set_path"])
    answer_map = parse_chat_dummy_sql((root / cfg["data"]["chat_dummy_sql_path"]).resolve())

    openai_cfg = OpenAIEmbeddingConfig(
        api_key_env=cfg["openai"]["api_key_env"],
        base_url_env=cfg["openai"]["base_url_env"],
        model=cfg["openai"]["embedding_model"],
    )
    embedder = create_embedder(provider_name=cfg["embedding"]["provider"], openai_config=openai_cfg, dimension=1536)

    searcher = DumpVectorSearcher.from_sql_records(records=answer_map, embed_text_fn=embedder.embed_texts)
    query_vectors = embedder.embed_texts([item["question"] for item in eval_set])

    candidates = [CrossEncoderCandidate(**item) for item in cfg["models"]["cross_encoders"]["candidates"]]
    loaded, status = load_candidates_with_status(candidates)
    (results_dir / "model_load_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    bi_top_k = int(cfg["retrieval"]["bi_top_k"])
    bi_threshold = 0.45

    rows: List[ModelComparisonResult] = []
    artifact_map: Dict[str, Dict[str, str]] = {}

    for candidate in candidates:
        scorer = loaded.get(candidate.name)
        if scorer is None:
            continue

        reciprocal_ranks: List[float] = []
        top3_hits: List[float] = []
        latencies_ms: List[float] = []
        positives: List[float] = []
        negatives: List[float] = []

        for sample, query_vector in zip(eval_set, query_vectors):
            hits = searcher.search(query_vector=query_vector, k=bi_top_k, min_similarity=bi_threshold)
            rerank_inputs = [CandidateAnswer(answer_id=h.answer_id, term_id=h.term_id, answer=h.answer) for h in hits]
            if not rerank_inputs:
                reciprocal_ranks.append(0.0)
                top3_hits.append(0.0)
                latencies_ms.append(0.0)
                continue

            started = time.perf_counter()
            detailed = rerank_candidates_detailed(
                query=sample["question"],
                candidates=rerank_inputs,
                scorer=scorer,
                raw_threshold=-1e9,
                top_n=len(rerank_inputs),
            )
            latencies_ms.append((time.perf_counter() - started) * 1000.0)

            predicted_ids = [item.answer_id for item, _ in detailed[:3]]
            reciprocal_ranks.append(reciprocal_rank(predicted_ids=predicted_ids, correct_ids=sample["correct_answer_ids"]))
            top3_hits.append(1.0 if any(x in set(sample["correct_answer_ids"]) for x in predicted_ids) else 0.0)

            gold = set(sample["correct_answer_ids"])
            for item, score in detailed:
                (positives if item.answer_id in gold else negatives).append(float(score.raw_score))

        f1_rows, best_f1 = _f1_curve(positives, negatives)
        ts = _timestamp()
        dist_png = results_dir / f"score_distribution_{candidate.name}_{ts}.png"
        f1_png = results_dir / f"f1_curve_{candidate.name}_{ts}.png"
        _plot_distribution(dist_png, positives, negatives, candidate.name)
        _plot_f1_curve(f1_png, f1_rows, candidate.name)

        artifact_map[candidate.name] = {
            "distribution_png": str(dist_png.relative_to(root)),
            "f1_curve_png": str(f1_png.relative_to(root)),
            "best_f1": f"{best_f1:.6f}",
        }

        rows.append(
            ModelComparisonResult(
                alias=candidate.name,
                model_name=candidate.hf_id,
                top3_accuracy=float(statistics.mean(top3_hits)) if top3_hits else 0.0,
                mrr=float(statistics.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
                avg_latency_ms=float(statistics.mean(latencies_ms)) if latencies_ms else 0.0,
                p95_latency_ms=float(np.percentile(np.array(latencies_ms, dtype=np.float64), 95)) if latencies_ms else 0.0,
                query_count=len(eval_set),
            )
        )

    payload = {
        "generated_at": datetime.now().isoformat(),
        "bi_encoder_context": {"bi_top_k": bi_top_k, "bi_threshold": bi_threshold},
        "models": [asdict(item) for item in rows],
        "model_load_status": status,
        "model_artifacts": artifact_map,
    }

    timestamp = _timestamp()
    out_json = results_dir / f"cross_encoder_comparison_{timestamp}.json"
    out_md = results_dir / f"cross_encoder_comparison_{timestamp}.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Cross-Encoder Comparison", "", f"- Generated at: {datetime.now().isoformat()}", ""]
    lines.append("| alias | model | Top-3 accuracy | MRR | avg latency (ms/query) | p95 latency (ms/query) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row.alias} | {row.model_name} | {row.top3_accuracy:.4f} | {row.mrr:.4f} | {row.avg_latency_ms:.2f} | {row.p95_latency_ms:.2f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] comparison json: {out_json}")
    print(f"[done] comparison md  : {out_md}")


if __name__ == "__main__":
    main()
