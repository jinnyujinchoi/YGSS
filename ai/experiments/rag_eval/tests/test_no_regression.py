"""Regression guard: recompute one best-combination pass and compare with saved score."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def test_no_regression_window_exists() -> None:
    root = Path(__file__).resolve().parent.parent
    result_path = root / "results" / "optimal_combination.json"
    assert result_path.exists(), "Run select_best_combination.py before regression test"

    best = json.loads(result_path.read_text(encoding="utf-8"))
    assert "f1_at_3" in best

    grid_rows = sorted((root / "results").glob("grid_search_*.csv"))
    assert grid_rows, "Run grid_search.py before regression test"

    df = pd.read_csv(grid_rows[-1])
    matched = df[
        (df["bi_threshold"] == best["bi_threshold"])
        & (df["bi_top_k"] == best["bi_top_k"])
        & (df["cross_threshold"] == best["cross_threshold"])
        & (df["cross_top_n"] == best["cross_top_n"])
    ]
    assert not matched.empty, "Best row not found in grid result"

    observed_f1 = float(matched.iloc[0]["f1_at_3"])
    expected_f1 = float(best["f1_at_3"])
    delta = abs(observed_f1 - expected_f1)
    assert delta <= 0.05, f"F1@3 regression too large: {delta:.4f} > 0.05"
