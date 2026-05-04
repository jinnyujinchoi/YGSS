"""Sensitivity plots for STEP C best combination."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


def _load_best(results_dir: Path) -> dict:
    best_path = results_dir / "optimal_combination.json"
    if not best_path.exists():
        raise FileNotFoundError("optimal_combination.json not found. Run select_best_combination.py first")
    return pd.read_json(best_path, typ="series").to_dict()


def _load_grid(results_dir: Path) -> pd.DataFrame:
    rows = sorted(results_dir.glob("grid_search_*.csv"))
    if not rows:
        raise FileNotFoundError("grid_search_*.csv not found")
    return pd.read_csv(rows[-1])


def _plot_param(df: pd.DataFrame, best: dict, param: str, out_path: Path) -> None:
    fixed = df.copy()
    for other in ["bi_threshold", "bi_top_k", "cross_threshold", "cross_top_n"]:
        if other == param:
            continue
        fixed = fixed[fixed[other] == best[other]]

    if fixed.empty:
        fixed = df.copy()

    fixed = fixed.sort_values(param)
    plt.figure(figsize=(8, 4.5))
    sns.lineplot(data=fixed, x=param, y="f1_at_3", marker="o")
    plt.axvline(best[param], color="red", linestyle="--", linewidth=1)
    plt.title(f"Sensitivity: {param} vs F1@3")
    plt.xlabel(param)
    plt.ylabel("F1@3")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    results_dir = root / cfg["results"]["output_dir"]

    best = _load_best(results_dir)
    df = _load_grid(results_dir)

    for param in ["bi_threshold", "bi_top_k", "cross_threshold", "cross_top_n"]:
        out_path = results_dir / f"sensitivity_{param}.png"
        _plot_param(df, best, param, out_path)
        print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
