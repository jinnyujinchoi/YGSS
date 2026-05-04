"""Pick best combination from STEP C grid-search result."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

BASELINE_P95_MS = 164.0
MAX_P95_MS = BASELINE_P95_MS * 1.5
BASELINE_NO_RESULT = 0.0
MAX_NO_RESULT = BASELINE_NO_RESULT + 0.05


def _load_latest_grid_csv(results_dir: Path) -> Path:
    rows = sorted(results_dir.glob("grid_search_*.csv"))
    if not rows:
        raise FileNotFoundError("grid_search_*.csv not found")
    return rows[-1]


def _pick_best(df: pd.DataFrame) -> pd.DataFrame:
    constrained = df[(df["p95_latency_ms"] <= MAX_P95_MS) & (df["no_result_rate"] <= MAX_NO_RESULT)]
    if constrained.empty:
        constrained = df.copy()
    return constrained.sort_values(["f1_at_3", "mrr", "precision_at_3"], ascending=[False, False, False])


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    results_dir = root / cfg["results"]["output_dir"]

    csv_path = _load_latest_grid_csv(results_dir)
    df = pd.read_csv(csv_path)
    ranked = _pick_best(df)

    best = ranked.iloc[0].to_dict()
    top5 = ranked.head(5)

    md_lines = [
        "# STEP C Optimal Combination",
        "",
        f"- Generated at: {datetime.now().isoformat()}",
        f"- Source: {csv_path.name}",
        "",
        "## Best",
        "",
        f"- bi_threshold: {best['bi_threshold']}",
        f"- bi_top_k: {int(best['bi_top_k'])}",
        f"- cross_threshold: {best['cross_threshold']}",
        f"- cross_top_n: {int(best['cross_top_n'])}",
        f"- f1_at_3: {best['f1_at_3']:.6f}",
        f"- precision_at_3: {best['precision_at_3']:.6f}",
        f"- recall_at_3: {best['recall_at_3']:.6f}",
        f"- mrr: {best['mrr']:.6f}",
        f"- p95_latency_ms: {best['p95_latency_ms']:.3f}",
        f"- no_result_rate: {best['no_result_rate']:.6f}",
        "",
        "## Current vs Recommended",
        "",
        "- current: bi_threshold=0.45, bi_top_k=10, cross_threshold=6.0, cross_top_n=3",
        (
            f"- recommended: bi_threshold={best['bi_threshold']}, bi_top_k={int(best['bi_top_k'])}, "
            f"cross_threshold={best['cross_threshold']}, cross_top_n={int(best['cross_top_n'])}"
        ),
        "",
        "## Top 5 Candidates",
        "",
        "| rank | bi_threshold | bi_top_k | cross_threshold | cross_top_n | f1_at_3 | mrr | p95_latency_ms | no_result_rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for rank, (_, row) in enumerate(top5.iterrows(), start=1):
        md_lines.append(
            f"| {rank} | {row['bi_threshold']:.2f} | {int(row['bi_top_k'])} | {row['cross_threshold']:.2f} "
            f"| {int(row['cross_top_n'])} | {row['f1_at_3']:.6f} | {row['mrr']:.6f} | {row['p95_latency_ms']:.2f} | {row['no_result_rate']:.6f} |"
        )

    out_md = results_dir / "optimal_combination.md"
    out_json = results_dir / "optimal_combination.json"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] {out_md}")
    print(f"[done] {out_json}")


if __name__ == "__main__":
    main()
