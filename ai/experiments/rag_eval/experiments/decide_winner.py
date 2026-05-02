"""Decide deployment candidate from comparison output."""

from __future__ import annotations

from datetime import datetime
import glob
import json
from pathlib import Path


def _latest(pattern: str) -> Path:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return Path(matches[-1])


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    results_dir = root / "results"

    baseline_path = _latest(str(results_dir / "baseline_threshold_sweep_*.json"))
    comparison_path = _latest(str(results_dir / "cross_encoder_comparison_*.json"))

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))

    baseline_rows = baseline.get("threshold_results", [])
    if not baseline_rows:
        raise RuntimeError("No baseline threshold results found")

    baseline_best_top3 = max(float(item.get("hit_at_3", 0.0)) for item in baseline_rows)
    baseline_latency = None

    models = comparison.get("models", [])
    best_candidate = None
    for model in models:
        top3 = float(model.get("top3_accuracy", 0.0))
        p95 = float(model.get("p95_latency_ms", 0.0))
        pass_quality = top3 >= baseline_best_top3 + 0.10
        pass_latency = True if baseline_latency in (None, 0.0) else p95 <= baseline_latency * 2.0
        if pass_quality and pass_latency:
            if best_candidate is None or top3 > float(best_candidate.get("top3_accuracy", 0.0)):
                best_candidate = model

    if best_candidate:
        decision = "운영 후보 선정"
        summary = f"{best_candidate['alias']} ({best_candidate['model_name']})"
    else:
        decision = "현행 유지"
        summary = "기준(Top-3 >= baseline+10%p, latency p95 <= baseline*2) 통과 모델 없음"

    lines = [
        "# Decision",
        "",
        f"- Generated at: {datetime.now().isoformat()}",
        f"- Baseline file: {baseline_path.name}",
        f"- Comparison file: {comparison_path.name}",
        f"- Baseline best Top-3: {baseline_best_top3:.4f}",
        f"- Decision: {decision}",
        f"- Summary: {summary}",
    ]

    if decision == "현행 유지":
        lines.extend(
            [
                "",
                "## 가능한 원인 가설",
                "- 평가셋 크기(현재 36개)로 모델 간 차이가 충분히 분리되지 않았을 수 있음",
                "- bi-encoder 후보 생성 품질이 병목이라 cross-encoder 이득이 제한될 수 있음",
                "- 한국어 후보 모델의 도메인 적합도/학습 데이터 분포가 서비스 질의와 불일치할 수 있음",
                "",
                "## 다음 실험 제안",
                "- eval_set을 80~100개로 확장 후 동일 프로토콜로 재측정",
                "- bi_threshold/bi_top_k/cross_top_n 좁은 grid로 72조합 탐색",
                "- 카테고리별(정의/비교/맥락) 분할 성능 분석 추가",
            ]
        )

    out_path = results_dir / "decision.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] decision: {out_path}")


if __name__ == "__main__":
    main()
