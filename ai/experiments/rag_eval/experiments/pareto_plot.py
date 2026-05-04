"""Pareto front plot for F1@3 vs p95 latency."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


def _pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    rows = df.sort_values(["p95_latency_ms", "f1_at_3"], ascending=[True, False])
    best_f1 = -1.0
    keep = []
    for _, row in rows.iterrows():
        if row["f1_at_3"] > best_f1:
            keep.append(row)
            best_f1 = float(row["f1_at_3"])
    return pd.DataFrame(keep).sort_values("p95_latency_ms")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    results_dir = root / cfg["results"]["output_dir"]

    grid_csv = sorted(results_dir.glob("grid_search_*.csv"))[-1]
    df = pd.read_csv(grid_csv)

    best_json = results_dir / "optimal_combination.json"
    best = pd.read_json(best_json, typ="series").to_dict() if best_json.exists() else None

    front = _pareto_front(df)

    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x="p95_latency_ms", y="f1_at_3", s=35, alpha=0.45, color="#1f77b4")
    sns.lineplot(data=front, x="p95_latency_ms", y="f1_at_3", marker="o", color="#d62728", linewidth=2)

    current = df[
        (df["bi_threshold"] == 0.45)
        & (df["bi_top_k"] == 10)
        & (df["cross_threshold"] == 6.0)
        & (df["cross_top_n"] == 3)
    ]
    if not current.empty:
        plt.scatter(current["p95_latency_ms"], current["f1_at_3"], color="black", s=90, label="current")

    if best is not None:
        plt.scatter([best["p95_latency_ms"]], [best["f1_at_3"]], color="green", s=90, label="recommended")

    plt.title("STEP C Grid Search: Pareto (Latency p95 vs F1@3)")
    plt.xlabel("p95 latency (ms)")
    plt.ylabel("F1@3")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out_path = results_dir / "grid_search_pareto.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
