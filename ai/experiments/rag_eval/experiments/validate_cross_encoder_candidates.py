"""Validate cross-encoder candidates before comparison experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import statistics
import time
from typing import Dict, List, Sequence, Tuple

from dotenv import load_dotenv
import numpy as np

from src.redis_search import QARecord, parse_chat_dummy_sql

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover
    CrossEncoder = None  # type: ignore[assignment]

try:
    import resource
except Exception:  # pragma: no cover
    resource = None  # type: ignore[assignment]


@dataclass
class ModelCandidate:
    """Candidate model metadata for pre-validation."""

    alias: str
    model_name: str
    expected_license: str
    license_allows_commercial: bool


@dataclass
class ValidationResult:
    """Validation result for one candidate model."""

    alias: str
    model_name: str
    loadable: bool
    input_pair_supported: bool
    raw_score_min: float
    raw_score_max: float
    raw_score_mean: float
    raw_score_p05: float
    raw_score_p95: float
    avg_latency_ms: float
    p95_latency_ms: float
    model_size_mb: float
    memory_delta_mb: float
    license_name: str
    license_allows_commercial: bool
    eligible_for_comparison: bool
    excluded_reasons: List[str]
    error: str


def _rss_mb() -> float:
    if resource is None:
        return 0.0
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KB.
    if usage > 10_000_000:
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def _estimate_model_size_mb(model: CrossEncoder) -> float:
    module = getattr(model, "model", None)
    if module is None:
        return 0.0
    total_bytes = 0
    for param in module.parameters():
        total_bytes += int(param.numel()) * int(param.element_size())
    return total_bytes / (1024.0 * 1024.0)


def _build_validation_pairs(
    eval_set_path: Path,
    sql_path: Path,
    sample_size: int = 20,
) -> List[Tuple[str, str]]:
    payload = json.loads(eval_set_path.read_text(encoding="utf-8"))
    answer_map: Dict[int, QARecord] = parse_chat_dummy_sql(sql_path)

    pairs: List[Tuple[str, str]] = []
    negatives: List[Tuple[str, str]] = []

    all_answer_ids = sorted(answer_map.keys())
    for item in payload:
        question = item["question"]
        gold_ids = item["correct_answer_ids"]

        for answer_id in gold_ids:
            answer = answer_map.get(answer_id)
            if answer is not None:
                pairs.append((question, answer.answer))

        for answer_id in all_answer_ids:
            if answer_id not in gold_ids:
                answer = answer_map.get(answer_id)
                if answer is not None:
                    negatives.append((question, answer.answer))
                break

    half = max(sample_size // 2, 1)
    selected_pairs = pairs[:half] + negatives[: sample_size - half]
    if not selected_pairs:
        raise RuntimeError("Could not build validation sample pairs")
    return selected_pairs


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), p))


def _validate_one(
    candidate: ModelCandidate,
    sample_pairs: Sequence[Tuple[str, str]],
    baseline_p95_ms: float | None,
) -> ValidationResult:
    excluded_reasons: List[str] = []

    if CrossEncoder is None:
        return ValidationResult(
            alias=candidate.alias,
            model_name=candidate.model_name,
            loadable=False,
            input_pair_supported=False,
            raw_score_min=0.0,
            raw_score_max=0.0,
            raw_score_mean=0.0,
            raw_score_p05=0.0,
            raw_score_p95=0.0,
            avg_latency_ms=0.0,
            p95_latency_ms=0.0,
            model_size_mb=0.0,
            memory_delta_mb=0.0,
            license_name=candidate.expected_license,
            license_allows_commercial=candidate.license_allows_commercial,
            eligible_for_comparison=False,
            excluded_reasons=["sentence-transformers not available"],
            error="sentence-transformers import failed",
        )

    before_rss = _rss_mb()
    try:
        model = CrossEncoder(candidate.model_name)
    except Exception as exc:
        return ValidationResult(
            alias=candidate.alias,
            model_name=candidate.model_name,
            loadable=False,
            input_pair_supported=False,
            raw_score_min=0.0,
            raw_score_max=0.0,
            raw_score_mean=0.0,
            raw_score_p05=0.0,
            raw_score_p95=0.0,
            avg_latency_ms=0.0,
            p95_latency_ms=0.0,
            model_size_mb=0.0,
            memory_delta_mb=max(_rss_mb() - before_rss, 0.0),
            license_name=candidate.expected_license,
            license_allows_commercial=candidate.license_allows_commercial,
            eligible_for_comparison=False,
            excluded_reasons=["model load failed"],
            error=str(exc),
        )

    after_rss = _rss_mb()
    memory_delta_mb = max(after_rss - before_rss, 0.0)

    pair_supported = False
    probe_scores: List[float] = []
    try:
        probe = model.predict([sample_pairs[0]])
        probe_scores = [float(value) for value in probe]
        pair_supported = len(probe_scores) == 1
    except Exception as exc:
        excluded_reasons.append("pair input not supported")
        error_text = str(exc)
    else:
        error_text = ""

    latencies_ms: List[float] = []
    raw_scores: List[float] = []
    if pair_supported:
        for pair in sample_pairs:
            started = time.perf_counter()
            values = model.predict([pair])
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(float(elapsed_ms))
            raw_scores.extend(float(v) for v in values)

    avg_latency = statistics.mean(latencies_ms) if latencies_ms else 0.0
    p95_latency = _percentile(latencies_ms, 95.0)

    if baseline_p95_ms is not None and baseline_p95_ms > 0.0 and p95_latency > baseline_p95_ms * 2.0:
        excluded_reasons.append("latency > 2x baseline")

    if not candidate.license_allows_commercial:
        excluded_reasons.append("license not allowed for commercial use")

    if candidate.expected_license.lower() in {"unknown", "unspecified", ""}:
        excluded_reasons.append("license is unclear")

    if not pair_supported:
        excluded_reasons.append("pair input unsupported")

    result = ValidationResult(
        alias=candidate.alias,
        model_name=candidate.model_name,
        loadable=True,
        input_pair_supported=pair_supported,
        raw_score_min=min(raw_scores) if raw_scores else 0.0,
        raw_score_max=max(raw_scores) if raw_scores else 0.0,
        raw_score_mean=statistics.mean(raw_scores) if raw_scores else 0.0,
        raw_score_p05=_percentile(raw_scores, 5.0),
        raw_score_p95=_percentile(raw_scores, 95.0),
        avg_latency_ms=avg_latency,
        p95_latency_ms=p95_latency,
        model_size_mb=_estimate_model_size_mb(model),
        memory_delta_mb=memory_delta_mb,
        license_name=candidate.expected_license,
        license_allows_commercial=candidate.license_allows_commercial,
        eligible_for_comparison=len(excluded_reasons) == 0,
        excluded_reasons=excluded_reasons,
        error=error_text,
    )
    return result


def _write_markdown(path: Path, results: Sequence[ValidationResult], sample_size: int) -> None:
    lines: List[str] = []
    lines.append("# Cross-Encoder Candidate Validation")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().isoformat()}")
    lines.append(f"- Sample pairs: {sample_size} (query-passage format)")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| alias | model | loadable | p95 latency (ms) | raw score range | size (MB) | memory delta (MB) | license | eligible |"
    )
    lines.append("|---|---|---:|---:|---|---:|---:|---|---:|")

    for row in results:
        score_range = f"[{row.raw_score_min:.4f}, {row.raw_score_max:.4f}]"
        lines.append(
            "| "
            + f"{row.alias} | {row.model_name} | {row.loadable} | {row.p95_latency_ms:.2f} | "
            + f"{score_range} | {row.model_size_mb:.2f} | {row.memory_delta_mb:.2f} | "
            + f"{row.license_name} | {row.eligible_for_comparison} |"
        )

    lines.append("")
    lines.append("## Exclusion Reasons")
    lines.append("")
    for row in results:
        reasons = ", ".join(row.excluded_reasons) if row.excluded_reasons else "None"
        err = row.error or "None"
        lines.append(f"- {row.alias}: reasons={reasons}; error={err}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Validate model candidates and save report artifacts."""
    load_dotenv()

    root = Path(__file__).resolve().parent.parent
    eval_set_path = root / "dataset" / "eval_set.json"
    sql_path = (root / "../../../exec/basic_insert.sql").resolve()
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        ModelCandidate(
            alias="baseline_en",
            model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
            expected_license="apache-2.0",
            license_allows_commercial=True,
        ),
        ModelCandidate(
            alias="ko_reranker",
            model_name="Dongjin-kr/ko-reranker",
            expected_license="unknown",
            license_allows_commercial=False,
        ),
        ModelCandidate(
            alias="mmarco_multilingual",
            model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            expected_license="apache-2.0",
            license_allows_commercial=True,
        ),
        ModelCandidate(
            alias="bge_reranker_m3",
            model_name="BAAI/bge-reranker-v2-m3",
            expected_license="mit",
            license_allows_commercial=True,
        ),
    ]

    sample_pairs = _build_validation_pairs(eval_set_path=eval_set_path, sql_path=sql_path, sample_size=20)

    results: List[ValidationResult] = []
    baseline_p95_ms: float | None = None
    for candidate in candidates:
        row = _validate_one(candidate, sample_pairs, baseline_p95_ms=baseline_p95_ms)
        results.append(row)
        if candidate.alias == "baseline_en" and row.loadable and row.p95_latency_ms > 0:
            baseline_p95_ms = row.p95_latency_ms

    validation_json = results_dir / "model_candidate_validation.json"
    validation_md = results_dir / "model_candidate_validation.md"

    payload = {
        "generated_at": datetime.now().isoformat(),
        "sample_pair_count": len(sample_pairs),
        "results": [asdict(item) for item in results],
    }
    validation_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(validation_md, results, sample_size=len(sample_pairs))

    eligible = [item.alias for item in results if item.eligible_for_comparison]
    print(f"[done] validation json: {validation_json}")
    print(f"[done] validation md  : {validation_md}")
    print(f"[done] eligible models: {eligible}")


if __name__ == "__main__":
    main()
