# RAG Evaluation Framework

> ⚠️ 이 평가 프레임워크는 OpenAI `text-embedding-3-small` 임베딩 + 실제 cross-encoder 모델 로드를 전제로 합니다. 키가 없거나 모델 로드가 실패하면 평가가 중단됩니다. 이는 의도된 동작이며, fallback으로 측정값을 생성하지 않습니다.

## 목적
운영 RAG retrieval 파이프라인의 임계값/Top-K 트레이드오프를 정량화하기 위한 독립 실험 프레임워크입니다. 운영 코드(`backend/`, `ai/src/api/routes/server.py`)는 수정하지 않고 `ai/experiments/rag_eval/`에서만 실행합니다.

## 환경 요구사항
- Python: 3.10+ 권장
- OS: macOS/Linux 권장
- HW: CPU 가능, 대형 모델은 GPU 권장
- 필수: `OPENAI_API_KEY`
- 모델 첫 다운로드 용량(대략)
- `cross-encoder/ms-marco-MiniLM-L-12-v2`: ~120MB
- `bongsoo/kpf-cross-encoder-v1`: ~400MB
- `Dongjin-kr/ko-reranker`: ~400MB
- `BAAI/bge-reranker-v2-m3`: ~2GB
- 예상 OpenAI 임베딩 비용: 평가셋 36문항 + 코퍼스 임베딩 기준 소액(토큰량에 비례, 실행 전 과금 정책 확인)

## 설치 및 실행
```bash
cd ai/experiments/rag_eval
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## 사전 로드 검증
```bash
python -m pytest tests/test_model_loading.py -v
```

## Baseline 재측정
```bash
python -m src.runner
```
- 실행 시작 시 preflight 게이트를 강제 실행합니다.
- preflight 실패 시 즉시 종료하며 결과는 측정값으로 인정되지 않습니다.

## Cross-Encoder 비교 재측정
```bash
python -m experiments.cross_encoder_comparison
python -m experiments.decide_winner
```

## 결과 파일 규칙
- preflight 통과 산출물만 `results/` 루트 측정값으로 사용
- `results/_invalid/` 파일은 측정값으로 사용 금지
- 모델 로드 상태: `results/model_load_status.json`
- preflight 리포트: `results/preflight_*.json`
- baseline: `results/baseline_threshold_sweep_*.json`, `*.png`
- 비교: `results/cross_encoder_comparison_*.json`, `*.md`
- 의사결정: `results/decision.md`

## 알려진 로드 이슈
- `sentence-transformers` 로드 실패 시 `transformers-direct` 로더로 2차 시도
- 두 로더 모두 실패하면 `results/model_load_status.json`의 `error`를 확인

