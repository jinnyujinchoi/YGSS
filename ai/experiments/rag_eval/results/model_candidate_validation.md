# Cross-Encoder Candidate Validation

- Generated at: 2026-04-26T17:54:30.416828
- Sample pairs: 20 (query-passage format)

## Summary

| alias | model | loadable | p95 latency (ms) | raw score range | size (MB) | memory delta (MB) | license | eligible |
|---|---|---:|---:|---|---:|---:|---|---:|
| baseline_en | cross-encoder/ms-marco-MiniLM-L-12-v2 | True | 387.60 | [5.6400, 9.3910] | 127.26 | 154.64 | apache-2.0 | True |
| ko_reranker | Dongjin-kr/ko-reranker | False | 0.00 | [0.0000, 0.0000] | 0.00 | 576.03 | unknown | False |
| mmarco_multilingual | cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | True | 272.41 | [-5.8444, 10.6795] | 448.77 | 145.47 | apache-2.0 | True |
| bge_reranker_m3 | BAAI/bge-reranker-v2-m3 | False | 0.00 | [0.0000, 0.0000] | 0.00 | 580.23 | mit | False |

## Exclusion Reasons

- baseline_en: reasons=None; error=None
- ko_reranker: reasons=model load failed; error=Can't load the model for 'Dongjin-kr/ko-reranker'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure 'Dongjin-kr/ko-reranker' is the correct path to a directory containing a file named pytorch_model.bin.
- mmarco_multilingual: reasons=None; error=None
- bge_reranker_m3: reasons=model load failed; error=Can't load the model for 'BAAI/bge-reranker-v2-m3'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure 'BAAI/bge-reranker-v2-m3' is the correct path to a directory containing a file named pytorch_model.bin.
