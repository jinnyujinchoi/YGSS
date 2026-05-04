# STEP C Optimal Combination

- Generated at: 2026-05-04T06:26:02.575980
- Source: grid_search_partial_20260504_054903.csv

## Best

- bi_threshold: 0.3
- bi_top_k: 15
- cross_threshold: 0.41
- cross_top_n: 3
- f1_at_3: 0.274170
- precision_at_3: 0.185185
- recall_at_3: 0.527778
- mrr: 0.527778
- p95_latency_ms: 243.165
- no_result_rate: 0.361111

## Current vs Recommended

- current: bi_threshold=0.45, bi_top_k=10, cross_threshold=6.0, cross_top_n=3
- recommended: bi_threshold=0.3, bi_top_k=15, cross_threshold=0.41, cross_top_n=3

## Top 5 Candidates

| rank | bi_threshold | bi_top_k | cross_threshold | cross_top_n | f1_at_3 | mrr | p95_latency_ms | no_result_rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.30 | 15 | 0.41 | 3 | 0.274170 | 0.527778 | 243.17 | 0.361111 |
| 2 | 0.30 | 15 | 0.41 | 5 | 0.274170 | 0.527778 | 236.14 | 0.361111 |
| 3 | 0.30 | 15 | 0.46 | 3 | 0.274170 | 0.527778 | 233.36 | 0.361111 |
| 4 | 0.30 | 15 | 0.46 | 5 | 0.274170 | 0.527778 | 234.18 | 0.361111 |
| 5 | 0.30 | 20 | 0.41 | 3 | 0.274170 | 0.527778 | 313.82 | 0.361111 |
