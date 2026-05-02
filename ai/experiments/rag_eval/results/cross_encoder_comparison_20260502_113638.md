# Cross-Encoder Comparison

- Generated at: 2026-05-02T11:36:38.996430

| alias | model | Top-3 accuracy | MRR | avg latency (ms/query) | p95 latency (ms/query) |
|---|---|---:|---:|---:|---:|
| production_baseline | cross-encoder/ms-marco-MiniLM-L-12-v2 | 0.4167 | 0.2824 | 41.53 | 41.06 |
| dragonkue_bge_m3_ko | dragonkue/bge-reranker-v2-m3-ko | 0.8333 | 0.6528 | 123.81 | 160.67 |
| ko_reranker | Dongjin-kr/ko-reranker | 0.9167 | 0.7454 | 122.92 | 163.68 |
| bge_m3 | BAAI/bge-reranker-v2-m3 | 0.8333 | 0.6852 | 120.12 | 159.69 |
