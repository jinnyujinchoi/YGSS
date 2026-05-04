"""STEP C: 4-parameter grid search for ko_reranker."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import itertools
import json
from pathlib import Path
import time
from typing import Dict, List, Sequence

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from src.cross_encoder import CandidateAnswer, CrossEncoderRegistry, rerank_candidates
from src.embedder import OpenAIEmbeddingConfig, create_embedder
from src.metrics import QueryEvaluation, compute_metric_summary
from src.preflight import run_preflight
from src.redis_search import DumpVectorSearcher, parse_chat_dummy_sql


@dataclass
class EvalSample:
    question: str
    correct_answer_ids: List[int]
    category: str


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_config(root: Path) -> Dict:
    return yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))


def _load_eval_set(path: Path) -> List[EvalSample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [EvalSample(**item) for item in payload]


def _f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else (2 * p * r) / (p + r)


def _evaluate_one_combo(
    combo_id: int,
    combo: tuple[float, int, float, int],
    eval_set: Sequence[EvalSample],
    query_vectors: Sequence[np.ndarray],
    searcher: DumpVectorSearcher,
    scorer,
) -> Dict[str, float]:
    bi_threshold, bi_top_k, cross_threshold, cross_top_n = combo

    records: List[QueryEvaluation] = []
    latency_ms: List[float] = []

    for sample, query_vector in zip(eval_set, query_vectors):
        started = time.perf_counter()
        bi_hits = searcher.search(query_vector=query_vector, k=bi_top_k, min_similarity=bi_threshold)
        candidates = [
            CandidateAnswer(answer_id=hit.answer_id, term_id=hit.term_id, answer=hit.answer)
            for hit in bi_hits
        ]
        reranked = rerank_candidates(
            query=sample.question,
            candidates=candidates,
            scorer=scorer,
            threshold=cross_threshold,
            top_n=cross_top_n,
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        latency_ms.append(float(elapsed))
        records.append(
            QueryEvaluation(
                predicted_ids=[item.answer_id for item, _ in reranked],
                correct_ids=sample.correct_answer_ids,
            )
        )

    summary = compute_metric_summary(records=records, cutoffs=[1, 3, 5])
    precision_at_3 = float(summary.get("precision_at_3", 0.0))
    recall_at_3 = float(summary.get("recall_at_3", 0.0))
    precision_at_5 = float(summary.get("precision_at_5", 0.0))
    recall_at_5 = float(summary.get("recall_at_5", 0.0))

    return {
        "threshold_id": combo_id,
        "bi_threshold": bi_threshold,
        "bi_top_k": bi_top_k,
        "cross_threshold": cross_threshold,
        "cross_top_n": cross_top_n,
        "precision_at_1": float(summary.get("precision_at_1", 0.0)),
        "precision_at_3": precision_at_3,
        "precision_at_5": precision_at_5,
        "recall_at_1": float(summary.get("recall_at_1", 0.0)),
        "recall_at_3": recall_at_3,
        "recall_at_5": recall_at_5,
        "f1_at_3": _f1(precision_at_3, recall_at_3),
        "f1_at_5": _f1(precision_at_5, recall_at_5),
        "mrr": float(summary.get("mrr", 0.0)),
        "no_result_rate": float(summary.get("no_result_rate", 1.0)),
        "avg_latency_ms": float(np.mean(latency_ms)) if latency_ms else 0.0,
        "p95_latency_ms": float(np.percentile(np.array(latency_ms, dtype=np.float64), 95)) if latency_ms else 0.0,
    }


def main() -> None:
    load_dotenv()
    root = Path(__file__).resolve().parent.parent
    cfg = _load_config(root)

    preflight = run_preflight(cfg=cfg, root=root)
    if not preflight.passed:
        raise RuntimeError(f"Preflight failed. Report: {preflight.report_path}")

    eval_set = _load_eval_set((root / cfg["data"]["eval_set_path"]).resolve())
    answer_map = parse_chat_dummy_sql((root / cfg["data"]["chat_dummy_sql_path"]).resolve())

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

    # Cache embeddings once for all 420 combinations.
    query_vectors = embedder.embed_texts([sample.question for sample in eval_set])
    searcher = DumpVectorSearcher.from_sql_records(records=answer_map, embed_text_fn=embedder.embed_texts)

    model_id = str(cfg["models"]["selected"]["hf_id"])
    scorer_mode = str(cfg["models"].get("scorer_mode", "sentence-transformers"))
    registry = CrossEncoderRegistry(model_map={"selected": model_id}, scorer_mode=scorer_mode)
    scorer = registry.get("selected")

    grid = cfg["grid_search"]
    combos = list(
        itertools.product(
            [float(x) for x in grid["bi_threshold"]],
            [int(x) for x in grid["bi_top_k"]],
            [float(x) for x in grid["cross_threshold"]],
            [int(x) for x in grid["cross_top_n"]],
        )
    )

    results_dir = root / cfg["results"]["output_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = _timestamp()
    partial_path = results_dir / f"grid_search_partial_{ts}.csv"
    rows: List[Dict[str, float]] = []

    for idx, combo in enumerate(tqdm(combos, desc="grid-search", total=len(combos)), start=1):
        rows.append(_evaluate_one_combo(idx, combo, eval_set, query_vectors, searcher, scorer))
        if idx % 30 == 0:
            pd.DataFrame(rows).to_csv(partial_path, index=False)

    df = pd.DataFrame(rows)
    csv_path = results_dir / f"grid_search_{ts}.csv"
    json_path = results_dir / f"grid_search_{ts}.json"
    df.to_csv(csv_path, index=False)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "selected_model": cfg["models"]["selected"],
        "combo_count": len(combos),
        "eval_set_count": len(eval_set),
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] partial: {partial_path}")
    print(f"[done] csv    : {csv_path}")
    print(f"[done] json   : {json_path}")


if __name__ == "__main__":
    main()
