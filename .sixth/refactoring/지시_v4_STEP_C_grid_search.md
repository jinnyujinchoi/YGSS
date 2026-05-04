# 지시서 v4 — STEP C: 4-Parameter Grid Search

> **대상 에이전트:** Codex (VSCode)
> **플래너:** Claude Opus 4.7
> **작성일:** 2026-05-02
> **선행 문서:** `지시.md` (v1), `지시_v2_QA보강.md` (v2), `지시_v3_colab재측정.md` (v3)
> **상태:** v3까지 모두 완료 → STEP C 진입

---

## 0. 이 문서의 위치

v3까지의 흐름:

```
v1 STEP A (rag-eval-framework)      → 평가 프레임워크 ✅
v1 STEP B (ko-cross-encoder)        → 영어 모델 → 한국어 모델 교체 ✅ (v2/v3에서 재측정)
v2 (rag-eval-hardening)             → fallback 제거 + preflight 게이트 ✅
v3 (Colab 재측정)                    → 4모델 비교 + 분기 A 확정 ✅
                                       ↓
v4 STEP C (rag-grid-search)         ← 본 지시서, 4-parameter grid search
```

### v3 측정 결과 (확정값)

Colab T4 GPU 환경, eval_set 36개, OpenAI text-embedding-3-small.

| 모델 | Top-3 정확도 | MRR | p95 latency |
|---|---:|---:|---:|
| production_baseline (영어) | 41.67% | 0.282 | 41ms |
| dragonkue_bge_m3_ko | 83.33% | 0.653 | 161ms |
| **ko_reranker (Dongjin-kr/ko-reranker)** ⭐ | **91.67%** | **0.745** | **164ms** |
| bge_m3 | 83.33% | 0.685 | 160ms |

**선정 모델: `Dongjin-kr/ko-reranker`** (Top-3 91.67%, baseline 대비 +50%p)

이 모델 기준의 최적 cross-encoder threshold는 v3 결과에서 best F1 = 0.516129로 산출됨.

---

## 1. 작업 범위

### 브랜치
**`rag-grid-search`** (from `refactor`)

```
master
└── refactor
      ├── rag-eval-framework      ← v1 STEP A (refactor에 머지됨)
      ├── ko-cross-encoder        ← v1 STEP B (refactor에 머지됨)
      ├── rag-eval-hardening      ← v2+v3 (refactor에 머지됨, 5/2 기준)
      └── rag-grid-search         ← v4 (이번 작업, refactor에서 분기)
```

### 손대는 범위
- `ai/experiments/rag_eval/experiments/grid_search.py` (신설)
- `ai/experiments/rag_eval/experiments/sensitivity_analysis.py` (신설)
- `ai/experiments/rag_eval/results/` — grid search 결과
- `ai/experiments/rag_eval/notebooks/` — Colab 실행 노트북 추가 (선택)
- `ai/src/api/routes/server.py` — 환경변수 기본값만 업데이트
- `ai/experiments/rag_eval/tests/test_no_regression.py` (신설)

### 손대지 않는 범위
- `backend/` 전체 — Java 코드는 **권장 변경 사항만 PR 본문에 명시**, 직접 수정 금지
- 운영 Redis, 운영 DB, 운영 FastAPI
- v2의 preflight 게이트, fallback 제거 구조

---

## 2. Codex 프롬프트

```
[Context]
프로젝트: YGSS
레포 루트: /Users/jinnysmacbookair/DEV/YGSS

선행 작업 (refactor 브랜치에 모두 머지됨):
- v1 STEP A: ai/experiments/rag_eval/ 평가 프레임워크 구축 완료
- v1 STEP B: 한국어 cross-encoder 후보 4종 비교 (v3에서 Colab으로 재실행)
- v2: fallback 제거 + preflight 게이트 도입
- v3: Colab에서 4모델 비교 측정 완료 → ko_reranker 선정

v3 측정 결과 (results/cross_encoder_comparison_20260502_113638.json):
- production_baseline (영어): Top-3 41.67%, MRR 0.282, p95 41ms
- dragonkue_bge_m3_ko: Top-3 83.33%, MRR 0.653, p95 161ms
- ko_reranker (Dongjin-kr/ko-reranker): Top-3 91.67%, MRR 0.745, p95 164ms ← 선정
- bge_m3: Top-3 83.33%, MRR 0.685, p95 160ms

ko_reranker의 best F1 cutoff: 0.516129 (results 디렉토리의 f1_curve_ko_reranker_*.png 참조)

튜닝 대상 4개 파라미터 (현재 운영값):
1. Bi-Encoder 코사인 유사도 임계값
   - 위치: backend/src/main/java/com/ygss/backend/global/redis/VectorRepository.java line ~84
   - 코드: if(sim < 0.45) continue;
   - 현재값: 0.45
2. Bi-Encoder top-K
   - 위치: backend/src/main/java/com/ygss/backend/chatbot/service/ChatBotServiceImpl.java line ~33
   - 코드: vectorRepository.searchAllPrefixes(..., 10)
   - 현재값: 10
3. Cross-Encoder 점수 임계값
   - 위치: ai/src/api/routes/server.py
   - 현재값: 6.0 (영어 모델 기준 매직 넘버, 한국어 모델로 교체 필요)
   - v3 산출 cutoff: 0.516129 (ko_reranker 기준)
4. Cross-Encoder top-N (최종 답변 수)
   - 위치: ai/src/api/routes/server.py
   - 코드: top3 = filtered[:3]
   - 현재값: 3

건드리면 안 되는 부분:
- 백엔드 Java 코드는 직접 수정 금지 — PR 본문에 정확한 변경 라인을 명시한 "권장 변경 제안"으로만 제출
- 운영 Redis, 운영 DB, 운영 FastAPI
- v2의 preflight 게이트, fallback 제거 구조
- ai/experiments/rag_eval/src/ 의 기존 함수 시그니처

[Task]
선정된 ko_reranker 기준으로 4-parameter grid search를 수행하고, 운영 적용을 위한 PR을 생성한다.

단계:

──────────────────────────────────────────
단계 1. config.yaml 정비
──────────────────────────────────────────

1-1. v3 결과를 반영하도록 config.yaml 업데이트
- embedding.provider 를 "openai"로 (v2 하드닝 유지)
- models.cross_encoders.candidates 에서 운영 후보로 ko_reranker만 선정 명시
- 새로운 grid search 섹션 추가:
    grid_search:
      bi_threshold: [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
      bi_top_k: [5, 10, 15, 20]
      cross_threshold: [0.41, 0.46, 0.51, 0.56, 0.61]   # v3 best F1 0.516 ± 20%
      cross_top_n: [1, 3, 5]

총 조합: 7 × 4 × 5 × 3 = 420개

1-2. config.yaml에 selected_model 키 추가
    models:
      selected:
        name: ko_reranker
        hf_id: Dongjin-kr/ko-reranker
        best_f1_cutoff: 0.516129
        source: results/cross_encoder_comparison_20260502_113638.json

──────────────────────────────────────────
단계 2. grid_search.py 작성
──────────────────────────────────────────

2-1. ai/experiments/rag_eval/experiments/grid_search.py 신설

기능:
- preflight 게이트 통과 후 진입 (v2 가드 그대로 사용)
- config.grid_search 의 4개 파라미터 조합을 itertools.product 로 생성
- 각 조합마다:
  * Bi-Encoder retrieval 실행 (bi_threshold, bi_top_k 적용)
  * Cross-Encoder rerank 실행 (cross_threshold, cross_top_n 적용)
  * Precision@K, Recall@K, F1, MRR, no-result rate, latency 측정
- 진행률 출력 (tqdm) — 420 조합 × eval_set 36개라 장시간 소요
- 30 조합마다 중간 저장 (results/grid_search_partial_{ts}.csv)
- 완료 시 results/grid_search_{ts}.csv + .json 저장

2-2. 임베딩 캐싱
- eval_set 의 모든 질문 임베딩을 시작 시점에 1회 계산
- 모든 조합에서 재사용 (OpenAI 호출 비용 절감)
- 마찬가지로 chat_dummy SQL 의 모든 답변 임베딩도 1회 계산 후 캐싱

2-3. 결과 컬럼
    threshold_id, bi_threshold, bi_top_k, cross_threshold, cross_top_n,
    precision_at_1, precision_at_3, precision_at_5,
    recall_at_1, recall_at_3, recall_at_5,
    f1_at_3, f1_at_5,
    mrr, no_result_rate,
    avg_latency_ms, p95_latency_ms

──────────────────────────────────────────
단계 3. 최적 조합 선정 로직
──────────────────────────────────────────

3-1. ai/experiments/rag_eval/experiments/select_best_combination.py 신설

선정 기준:
- Primary: F1@3 최대화
- Secondary: latency p95 가 baseline (v3의 ko_reranker 측정값 164ms) 의 1.5배 이내 (~246ms)
- Tertiary: no_result_rate ≤ baseline + 5%p

선정 결과를 results/optimal_combination.md 에 저장:
- 최적 조합 (4개 파라미터 값)
- 메트릭 (F1, P, R, MRR, latency)
- 현재 운영값 (0.45 / 10 / 6.0 / 3) 대비 개선폭
- 차순위 후보 5개 (sensitivity_analysis 용)

──────────────────────────────────────────
단계 4. 민감도 분석
──────────────────────────────────────────

4-1. ai/experiments/rag_eval/experiments/sensitivity_analysis.py 신설

기능:
- 최적 조합의 4개 파라미터 중 하나만 변화시키고 나머지 3개는 고정
- 각 파라미터의 F1 변화 곡선 그래프 생성
- 출력: results/sensitivity_{param_name}.png × 4개

해석 가이드:
- 곡선이 평평하면 그 파라미터는 robust (운영에서 약간 벗어나도 OK)
- 곡선이 가파르면 sensitive (운영에서 정확히 적용해야 함)

4-2. Pareto front 시각화
- ai/experiments/rag_eval/experiments/pareto_plot.py 신설
- x축: latency p95, y축: F1@3
- 모든 420 조합을 점으로 plot
- Pareto front 위 점들을 선으로 연결
- 최적 조합과 현재 운영값을 다른 색으로 표시
- 출력: results/grid_search_pareto.png

──────────────────────────────────────────
단계 5. 운영 적용을 위한 PR 패키지
──────────────────────────────────────────

5-1. ai/src/api/routes/server.py 환경변수 기본값 업데이트
- CROSS_ENCODER_MODEL 기본값 → "Dongjin-kr/ko-reranker"
- CROSS_ENCODER_THRESHOLD 기본값 → 단계 3 산출 cross_threshold
- CROSS_TOP_N 환경변수 신설, 기본값 → 단계 3 산출 cross_top_n
- 변경 이유 주석 1~2줄 (v3 + STEP C 결과 링크)

5-2. 백엔드 Java 권장 변경 사항 — 직접 수정하지 않고 PR 본문에 정확히 명시

권장 사항 1: VectorRepository.java
  - 현재: if(sim < 0.45) continue;
  - 변경 권장: if(sim < {단계3_bi_threshold}) continue;
  - 더 권장: application.yml 의 chatbot.bi-encoder.threshold 로 외부화
    @Value("${chatbot.bi-encoder.threshold:0.45}")
    private double biThreshold;

권장 사항 2: ChatBotServiceImpl.java
  - 현재: vectorRepository.searchAllPrefixes(..., 10);
  - 변경 권장: vectorRepository.searchAllPrefixes(..., {단계3_bi_top_k});
  - 더 권장: application.yml 의 chatbot.bi-encoder.top-k 로 외부화

권장 사항 3: application.yml 신설 키
    chatbot:
      bi-encoder:
        threshold: {단계3_bi_threshold}
        top-k: {단계3_bi_top_k}
      cross-encoder:
        threshold: {단계3_cross_threshold}
        top-n: {단계3_cross_top_n}

5-3. PR 본문 구조

```
## STEP C — 4-Parameter Grid Search 결과

### Before / After

| 항목 | 현재 운영값 | STEP C 권장값 |
|---|---|---|
| Bi-Encoder threshold | 0.45 | {산출값} |
| Bi-Encoder top-K | 10 | {산출값} |
| Cross-Encoder model | ms-marco-MiniLM-L-12-v2 (영어) | Dongjin-kr/ko-reranker |
| Cross-Encoder threshold | 6.0 (영어 모델 매직 넘버) | {산출값} |
| Cross-Encoder top-N | 3 | {산출값} |

### 측정값 비교 (eval_set 36개 기준)

| 메트릭 | 현재 운영값 적용 시 | STEP C 권장값 적용 시 | Δ |
|---|---|---|---|
| F1@3 | {x} | {y} | +{z}%p |
| Top-3 정확도 | 41.67% (v3 baseline) | {y}% | +{z}%p |
| MRR | 0.282 (v3 baseline) | {y} | +{z} |
| p95 latency | 41ms (영어 모델, 평가 부정확) | {y}ms | — |

### 백엔드 권장 변경 사항

(위 5-2 내용)

### 민감도 분석

(sensitivity_*.png 4개 첨부)

### Pareto front

(grid_search_pareto.png 첨부)
```

──────────────────────────────────────────
단계 6. 회귀 가드
──────────────────────────────────────────

6-1. ai/experiments/rag_eval/tests/test_no_regression.py 신설
- 권장값을 적용한 상태에서 measure 한 번 더 실행
- 결과가 grid_search 에서 도출한 값에서 ±5%p 이내인지 검증
- pytest -v 로 실행

6-2. README 갱신
- STEP C 결과 섹션 추가
- 권장값 표 + 그래프 링크
- "운영 적용 시 다음 모니터링 지표 권장" 섹션:
  * Top-3 정확도 (eval_set 매주 재측정)
  * p95 latency
  * no_result_rate
  * OpenAI 임베딩 호출 실패율

[Constraints]
- 언어: Python 3.10+
- 추가 의존성: pandas, seaborn, tqdm
- grid search 는 420 조합 × eval_set 36개 호출 → Colab T4 GPU 기준 약 30분~1시간 예상
- 진행 로그 + 중간 저장 필수 (세션 끊겨도 재개 가능)
- 백엔드 Java 코드는 절대 직접 수정하지 않음 — PR 본문 권장 사항으로만 제출
- ai/src/api/routes/server.py 는 환경변수 기본값만 업데이트
- 측정 중 OpenAI API 비용은 약 $0.10 미만 (질문 임베딩 캐싱 시)
- v3의 ko_reranker 결과는 그대로 인용 (재측정 불필요)

[Output]
- experiments/grid_search.py
- experiments/select_best_combination.py
- experiments/sensitivity_analysis.py
- experiments/pareto_plot.py
- results/grid_search_{ts}.csv + .json
- results/optimal_combination.md
- results/sensitivity_{4_params}.png
- results/grid_search_pareto.png
- 수정된 ai/src/api/routes/server.py (환경변수 기본값만)
- 신설된 tests/test_no_regression.py
- 갱신된 README.md
- PR 본문: Before/After 표 + 권장 변경 라인 + 민감도/Pareto 그래프

[성공 기준]
- 420 조합 grid search 가 재현 가능 (스크립트 한 줄로 실행)
- 최적 조합이 v3 baseline (Top-3 41.67%) 대비 F1 ≥ +5%p
- latency 회귀 없음 (p95 ≤ ko_reranker baseline 164ms × 1.5 = 246ms)
- 백엔드 PR 본문에 "왜 이 값인가" 가 그래프 + 표로 답변됨
- ai/src/api/routes/server.py 변경이 최소한이고 환경변수로 조정 가능
- 회귀 테스트 통과

[금지 사항]
- 백엔드 Java 코드 직접 수정 금지
- preflight 게이트 우회 금지
- v3에서 이미 측정된 4모델 재측정 금지 (ko_reranker만 grid search 대상)
- 측정값이 기준 미달인데 "권장"으로 결론 내지 말 것
  * 만약 grid search 결과가 ko_reranker 단독 (단계1-2의 selected) 보다 개선이 없으면
    → "현재 운영값 + 모델만 ko_reranker로 교체" 를 권장값으로 결론
```

---

## 3. 사용자(시니어 아키텍트) 측의 후속 의사결정

STEP C 결과에 따라 다음 분기가 발생한다.

### 분기 C-1: 4개 파라미터 모두 의미 있는 개선
- 권장값을 백엔드 PR로 제출
- 백엔드 팀에 application.yml 외부화 제안

### 분기 C-2: cross-encoder 모델 교체만 의미 있고 나머지는 robust
- 백엔드 권장: "현재 운영값 유지 (0.45 / 10), AI만 모델 교체"
- 더 단순한 PR로 종결

### 분기 C-3: ko_reranker 적용 시 latency가 운영 SLA 초과
- 백엔드 권장: "단계적 도입" — 우선 staging 환경에서 부하 테스트
- A/B 테스트 후 점진 적용 제안

---

## 4. v3까지의 결과물과의 관계

- v1, v2, v3 의 산출물은 모두 보존
- STEP C는 ko_reranker 단일 모델 기준으로 grid search 수행
- v3 의 4모델 비교 결과를 그대로 인용 (재측정 불필요)
- 기존 cross_encoder_comparison_*.json 의 best_f1 cutoff 0.516129 를 cross_threshold 그리드의 중심값으로 사용

---

## 5. 다음 채팅에서 사용자가 할 일

1. 본 v4 지시서 검토
2. Codex에 v4 프롬프트 전달 (로컬 또는 Colab)
3. grid search 실행 (Colab T4 권장, 약 30분~1시간)
4. 결과 (optimal_combination.md, grid_search_pareto.png) 검토
5. 분기 C-1/C-2/C-3 판정
6. 백엔드 PR 제출 결정

---

## 6. 백엔드 적용 이후 (이번 리팩토링 범위 외)

STEP C가 완료된 뒤, 다음 작업이 남아있다 (별도 사이클).

- 백엔드 PR 머지 + 운영 적용
- 운영 환경에서 1~2주 지표 모니터링
- 평가 셋을 80~100개로 확장한 재측정
- v1 지시서 "후속 작업 섹션" 항목들:
  * Jedis → JedisPool 전환
  * Redis Vector Search (HNSW) 도입
  * ChatBotServiceImpl 거대 try-catch 분해
  * 하드코딩 fallback 포트폴리오 제거
  * GPT/Gemini 죽은 코드 정리
