"""Threshold sweep runner for RAG retrieval evaluation."""

from __future__ import annotations

import json
from datetime import datetime
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import yaml
from dotenv import load_dotenv

from .cross_encoder import CandidateAnswer, CrossEncoderCandidate, rerank_candidates
from .embedder import OpenAIEmbeddingConfig, create_embedder
from .metrics import QueryEvaluation, compute_metric_summary
from .preflight import run_preflight
from .redis_search import DumpVectorSearcher, RedisVectorSearcher, parse_chat_dummy_sql


def _load_config() -> Dict:
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("config.yaml not found. Run from ai/experiments/rag_eval")
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _save_plot(output_path: Path, threshold_rows: List[Dict[str, float]]) -> None:
    thresholds = [row["threshold"] for row in threshold_rows]
    p5 = [row.get("precision_at_5", 0.0) for row in threshold_rows]
    r5 = [row.get("recall_at_5", 0.0) for row in threshold_rows]
    mrr = [row.get("mrr", 0.0) for row in threshold_rows]
    no_result = [row.get("no_result_rate", 0.0) for row in threshold_rows]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, p5, marker="o", label="Precision@5")
    plt.plot(thresholds, r5, marker="s", label="Recall@5")
    plt.plot(thresholds, mrr, marker="^", label="MRR")
    plt.plot(thresholds, no_result, marker="x", label="No-result rate")
    plt.axvline(0.45, color="red", linestyle="--", linewidth=1.5, label="Current threshold=0.45")
    plt.xlabel("Bi-Encoder Similarity Threshold")
    plt.ylabel("Metric")
    plt.title("RAG Threshold Sweep")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _move_invalid_results(output_dir: Path) -> None:
    invalid_dir = output_dir / "_invalid"
    invalid_dir.mkdir(parents=True, exist_ok=True)
    for file_path in output_dir.glob("baseline_*"):
        if file_path.is_file():
            shutil.move(str(file_path), str(invalid_dir / f"legacy_{file_path.name}"))


def main() -> None:
    load_dotenv()
    cfg = _load_config()
    root = Path.cwd()
    output_dir = _resolve_path(cfg["results"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    preflight = run_preflight(cfg, root)
    if not preflight.passed:
        _move_invalid_results(output_dir)
        print(f"[fail] preflight failed: {preflight.report_path}")
        sys.exit(1)

    eval_set_path = _resolve_path(cfg["data"]["eval_set_path"])
    sql_path = _resolve_path(cfg["data"]["chat_dummy_sql_path"])

    eval_set_payload = json.loads(eval_set_path.read_text(encoding="utf-8"))
    answer_map = parse_chat_dummy_sql(sql_path)

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

    source_mode = cfg["data"]["source_mode"]
    if source_mode == "redis":
        searcher = RedisVectorSearcher(
            answer_map=answer_map,
            scan_count=int(cfg["redis"]["scan_count"]),
            max_keys=int(cfg["redis"]["max_keys"]),
            timeout_seconds=int(cfg["redis"]["timeout_seconds"]),
            key_pattern=str(cfg["redis"].get("key_pattern", "*")),
        )
    else:
        dump_path = _resolve_path(cfg["data"]["embedding_dump_path"])
        if dump_path.exists():
            searcher = DumpVectorSearcher.from_jsonl(path=dump_path, answer_map=answer_map)
        else:
            searcher = DumpVectorSearcher.from_sql_records(records=answer_map, embed_text_fn=embedder.embed_texts)

    candidates = [CrossEncoderCandidate(**item) for item in cfg["models"]["cross_encoders"]["candidates"]]
    production = [item for item in candidates if item.name == "production_baseline"]
    if not production:
        raise RuntimeError("production_baseline candidate not found in config")

    from .cross_encoder import load_candidates_with_status

    loaded, status = load_candidates_with_status(production)
    if "production_baseline" not in loaded:
        raise RuntimeError(f"production_baseline failed to load: {status.get('production_baseline')}")
    scorer = loaded["production_baseline"]

    thresholds = [float(value) for value in cfg["retrieval"]["thresholds"]]
    bi_top_k = int(cfg["retrieval"]["bi_top_k"])
    cross_threshold = float(cfg["retrieval"]["cross_encoder_threshold"])
    cross_top_n = int(cfg["retrieval"]["cross_top_n"])
    cutoffs = [int(value) for value in cfg["retrieval"]["metric_cutoffs"]]

    threshold_rows: List[Dict[str, float]] = []
    per_threshold_examples: Dict[str, List[Dict]] = {}

    query_vectors = embedder.embed_texts([sample["question"] for sample in eval_set_payload])
    for threshold in thresholds:
        records: List[QueryEvaluation] = []
        examples: List[Dict] = []

        for sample, query_vector in zip(eval_set_payload, query_vectors):
            bi_hits = searcher.search(query_vector=query_vector, k=bi_top_k, min_similarity=threshold)
            candidates_for_query = [
                CandidateAnswer(answer_id=hit.answer_id, term_id=hit.term_id, answer=hit.answer)
                for hit in bi_hits
            ]

            reranked = rerank_candidates(
                query=sample["question"],
                candidates=candidates_for_query,
                scorer=scorer,
                threshold=cross_threshold,
                top_n=cross_top_n,
            )

            predicted_ids = [cand.answer_id for cand, _ in reranked]
            records.append(QueryEvaluation(predicted_ids=predicted_ids, correct_ids=sample["correct_answer_ids"]))

            if len(examples) < 5:
                examples.append(
                    {
                        "question": sample["question"],
                        "category": sample["category"],
                        "gold_ids": sample["correct_answer_ids"],
                        "bi_candidates": [hit.answer_id for hit in bi_hits],
                        "final_predicted_ids": predicted_ids,
                    }
                )

        metrics = compute_metric_summary(records=records, cutoffs=cutoffs)
        metrics["threshold"] = threshold
        threshold_rows.append(metrics)
        per_threshold_examples[f"{threshold:.2f}"] = examples

    timestamp = _timestamp()
    raw_json = output_dir / f"raw_threshold_sweep_{timestamp}.json"
    raw_png = output_dir / f"raw_threshold_sweep_{timestamp}.png"

    payload = {
        "generated_at": datetime.now().isoformat(),
        "timestamp": timestamp,
        "git_commit": _get_git_commit(),
        "preflight_passed": True,
        "embedder": f"openai/{cfg['openai']['embedding_model']}",
        "models_loaded": preflight.models_loaded,
        "config_snapshot": {
            "embedding_provider": cfg["embedding"]["provider"],
            "source_mode": source_mode,
            "cross_encoder_threshold": cross_threshold,
            "cross_top_n": cross_top_n,
            "bi_top_k": bi_top_k,
            "metric_cutoffs": cutoffs,
        },
        "threshold_results": threshold_rows,
        "examples": per_threshold_examples,
    }

    raw_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _save_plot(raw_png, threshold_rows)

    baseline_json = output_dir / f"baseline_threshold_sweep_{timestamp}.json"
    baseline_png = output_dir / f"baseline_threshold_sweep_{timestamp}.png"
    baseline_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _save_plot(baseline_png, threshold_rows)

    print(f"[done] baseline json: {baseline_json}")
    print(f"[done] baseline png : {baseline_png}")
    print(f"[done] raw json     : {raw_json}")
    print(f"[done] raw png      : {raw_png}")


if __name__ == "__main__":
    main()
