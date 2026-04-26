# RAG Evaluation Framework (STEP A)

## 목적
운영 RAG retrieval 파이프라인의 임계값/Top-K 트레이드오프를 정량화하기 위한 독립 실험 프레임워크입니다.
운영 코드(`backend/`, `ai/src/api/routes/server.py`)는 수정하지 않고 `ai/experiments/rag_eval/`에서만 실행합니다.
현재 baseline은 offline fallback 기반 smoke test 결과이며, 프레임워크 실행 검증용입니다.
즉, 운영 RAG 성능을 대표하지 않습니다.

## 디렉토리 구조
- `config.yaml`: 모델/임계값/데이터 소스 설정
- `dataset/eval_set.json`: 평가 질문셋(36개)
- `src/embedder.py`: OpenAI `text-embedding-3-small` 래퍼 + 로컬 fallback
- `src/redis_search.py`: Redis SCAN 기반 검색(운영 Java `VectorRepository.searchAllPrefixes` 동등 로직) + dump/sql 기반 모드
- `src/cross_encoder.py`: 운영 Cross-Encoder 및 한국어 모델 비교용 추상화
- `src/metrics.py`: Precision@K, Recall@K, MRR, no-result rate
- `src/runner.py`: threshold sweep 실행 및 결과 저장
- `results/`: baseline 및 반복 실행 결과 저장

## 실행 방법
1. 가상환경 생성
```bash
cd ai/experiments/rag_eval
python3 -m venv .venv
source .venv/bin/activate
```

2. 의존성 설치
```bash
pip install -r requirements.txt
```

3. 환경변수 준비
```bash
cp .env.example .env
```
- OpenAI 사용 시 `OPENAI_API_KEY`를 설정
- Redis 모드 사용 시 `RAG_EVAL_REDIS_HOST` 등 설정
- 운영 Redis는 기본 차단이며 `ENABLE_PROD_REDIS=true` + `PROCEED_PROD_REDIS=YES`가 둘 다 있어야만 접근

4. baseline 실행
```bash
python -m src.runner
```

## 평가셋 구성 근거
- 근거 데이터: `exec/basic_insert.sql`의 `chat_dummy` 실제 Q&A
- `eval_set.json`은 해당 실데이터의 질문 의도를 유지한 패러프레이즈 질문으로 구성
- 정답 ID(`correct_answer_ids`)는 `chat_dummy` INSERT 순서 기반 ID 사용
- 카테고리 분포:
  - 정의형: 9
  - 계산형: 4
  - 비교형: 13
  - 맥락형: 10

## 메트릭 정의
- `Precision@K`: 상위 K개 결과 중 정답 비율
- `Recall@K`: 정답 집합 중 상위 K에서 회수된 비율
- `MRR`: 첫 정답의 역순위 평균
- `No-result rate`: 최종 결과가 0건인 질의 비율

## 결과 해석 가이드
중요: 현재 `baseline_threshold_sweep.*`는 offline fallback smoke test 산출물입니다. 운영 RAG 성능 지표로 해석하면 안 됩니다.

- x축: Bi-Encoder 유사도 임계값 (`0.30~0.60`)
- y축: Precision@5, Recall@5, MRR, 실패율
- 빨간 점선: 현재 운영 임계값 `0.45`
- 해석 포인트:
  - 임계값 상승 시 보통 실패율은 증가하고 정밀도는 증가 가능
  - 임계값 하향 시 재현율은 증가 가능하나 노이즈 증가 위험
  - 목표는 서비스 정책에 맞는 균형점 선택

## 결과 파일 규칙
- 커밋 대상 baseline:
  - `results/baseline_threshold_sweep.json`
  - `results/baseline_threshold_sweep.png`
- 반복 실행 산출물:
  - `results/raw_threshold_sweep_{timestamp}.json`
  - `results/raw_threshold_sweep_{timestamp}.png`
  - `.gitignore`에서 `tmp_`/`raw_` 임시 파일 패턴 제외

## STEP B (Cross-Encoder 교체) 실험
- 후보 검증:
  - `python -m experiments.validate_cross_encoder_candidates`
  - 산출물: `results/model_candidate_validation.md`
- 모델 비교 + 임계값 보정:
  - `python -m experiments.cross_encoder_comparison`
  - 산출물:
    - `results/cross_encoder_comparison_{timestamp}.json`
    - `results/cross_encoder_comparison_{timestamp}.md`
    - `results/threshold_calibration_{model}.png`

### score 필드 의미
- `/server/compare` 응답의 `score`는 선택된 Cross-Encoder의 **raw score**입니다.
- 모델마다 raw score 범위가 달라 threshold 해석도 모델별로 달라집니다.
- 따라서 threshold는 모델별로 정답/오답 분포에서 별도로 보정해야 합니다.

## 주의 사항
- Redis 검색은 `SCAN`만 사용, `KEYS` 미사용
- Redis 부하 제어 옵션(`scan_count`, `max_keys`, `timeout_seconds`) 제공
- 기본 모드는 `sql_dump`(로컬 SQL 기반)이며 운영 Redis 접근은 비활성화
