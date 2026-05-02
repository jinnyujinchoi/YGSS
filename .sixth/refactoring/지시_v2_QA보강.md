# 지시서 v2 — QA 피드백 반영 (STEP A 재측정 + 가드 추가)

> **대상 에이전트:** Codex (VSCode)
> **플래너:** Claude Opus 4.7
> **작성일:** 2026-04-29
> **선행 문서:** `지시.md` (v1)
> **상태:** STEP A는 완료됐으나 QA 게이트 미통과 → **재측정 필요**

---

## 0. 이 문서가 만들어진 이유

QA 에이전트가 STEP A/B 결과물을 검증하면서 두 개의 치명적 결함을 발견했다.

1. **embedder가 `local-hash` fallback으로 동작 중** — 의미 기반 검색이 아닌 해시 기반 더미. Bi-Encoder가 사실상 작동 안 함.
2. **한국어 모델 2종 로드 실패** (Dongjin-kr/ko-reranker, BAAI/bge-reranker-v2-m3) — 비교 대상이 사실상 영어 모델 2개뿐.

이 상태에서 나온 Top-3 정답률 19.44%, MRR 1.4%p 개선은 측정값이 아니라 노이즈다. 그럼에도 "운영 후보로 선정"이라는 결론이 나왔다.

**근본 원인은 Codex가 아니라 v1 지시서다.** v1은 "fallback이 동작하면 즉시 실패하라"는 가드를 명시하지 않았고, smoke test와 실측정의 분리도 강제하지 않았다. 따라서 v2에서는 다음 두 가지를 동시에 처리한다.

- (a) 이미 만들어진 코드의 fallback 구조 자체를 제거
- (b) "측정 가능한 상태인지" 검증하는 게이트를 도입해 같은 사고가 재발하지 않도록 함

---

## 1. 작업 범위

### 브랜치
**`rag-eval-hardening`** (from `ko-cross-encoder`)

- v1 STEP A(`rag-eval-framework`)와 STEP B(`ko-cross-encoder`)는 이미 작업 완료 상태이며 서로 머지된 상태
- master에는 아직 머지하지 않음 (v2까지 완료 후 일괄 결정 예정)
- v2는 v1 결과물 위에서 보정하는 작업이므로 master가 아닌 `ko-cross-encoder`에서 분기

```
master (보호)
   │
   ├── rag-eval-framework      ← v1 STEP A (완료)
   │       │
   │       └── ko-cross-encoder  ← v1 STEP B (완료)
   │               │
   │               └── rag-eval-hardening  ← v2 (이번 작업)
```

네이밍 근거: "hardening"은 느슨한 시스템에 가드를 추가해 견고하게 만드는 작업의 표준 용어. 이번 작업의 본질(fallback 제거 + preflight 게이트 + 측정 신뢰도 확보)과 일치.

### 손대는 범위
- `ai/experiments/rag_eval/` 전체 (config, embedder, cross_encoder, runner, README, tests)
- 그 외 디렉토리는 **여전히 read-only**

### 손대지 않는 범위
- `backend/` 전체
- `ai/src/api/routes/server.py` (STEP B 작업 영역, v2에서는 안 건드림)
- 운영 Redis, 운영 DB

---

## 2. Codex 프롬프트

```
[Context]
프로젝트: YGSS
레포 루트: /Users/jinnysmacbookair/DEV/YGSS

선행 작업: STEP A (rag-eval-framework) 완료, 그러나 QA 미통과.

확인된 결함 (코드 인용):

1. ai/experiments/rag_eval/config.yaml
   embedding:
     provider: local-hash       ← 본 측정에서 그대로 쓰임
     dimension: 1536
   models:
     scorer_mode: lexical       ← Cross-Encoder도 어휘 일치로 fallback 가능
     korean_cross_encoder: Dongjin-kr/ko-reranker   ← 1종만 등록

2. ai/experiments/rag_eval/src/embedder.py
   create_embedder()가 provider_name == "local-hash"를 정상 경로로 받음.
   "Use local-hash provider for offline runs"라는 주석이 있어 fallback 의도였으나
   config 기본값이 local-hash라 본 측정에 그대로 사용됨.

3. ai/experiments/rag_eval/src/cross_encoder.py
   LexicalFallbackScorer 존재. 모델 로드 실패 시 무음으로 fallback 가능한 구조.

4. 결과 수치: Top-3 정답률 영어/다국어 모델 동일(19.44%), MRR 1.4%p 차이.
   성공 기준(≥10%p) 미달이나 다국어 모델이 "운영 후보로 선정"됨.

QA 판정:
- 측정 가능성: 미통과 (성공 기준 미달 + baseline 무효)
- 재현 가능성: 경고 (한국어 모델 로드 실패 환경 미명시)

[Task]
ai/experiments/rag_eval/ 의 평가 프레임워크를 "측정 가능한 상태"로 재구축하고,
재측정한 결과로 STEP B 결론(모델 교체 vs 현행 유지)을 다시 도출한다.

단계는 5개로 분할되며, 각 단계는 이전 단계가 통과해야 다음으로 넘어간다.

──────────────────────────────────────────
단계 1. Fallback 제거 + 측정 모드 강제
──────────────────────────────────────────

1-1. config.yaml 변경
- embedding.provider 의 허용 값을 "openai" 단일로 제한
- "local-hash"는 config에서 완전히 삭제 (옵션조차 없게)
- models.scorer_mode 키 자체를 삭제 (lexical fallback 금지)
- models.korean_cross_encoder를 단일 키에서 모델 후보 리스트로 변경:
    cross_encoders:
      candidates:
        - name: production_baseline
          hf_id: cross-encoder/ms-marco-MiniLM-L-12-v2
          language: en
        - name: kpf_ko
          hf_id: bongsoo/kpf-cross-encoder-v1
          language: ko
        - name: ko_reranker
          hf_id: Dongjin-kr/ko-reranker
          language: ko
        - name: bge_m3
          hf_id: BAAI/bge-reranker-v2-m3
          language: multi

1-2. embedder.py 변경
- LocalHashEmbeddingProvider 클래스를 삭제
- create_embedder()에서 "local-hash" 분기 삭제
- provider_name이 "openai"가 아니면 RuntimeError 발생
- OpenAI 키 누락 시 즉시 실패하도록 (fallback 금지)

1-3. cross_encoder.py 변경
- LexicalFallbackScorer 클래스를 삭제
- create_scorer() 또는 동등한 팩토리에서 fallback 분기 제거
- sentence-transformers 로드 실패 시 즉시 RuntimeError

1-4. README 상단에 경고 박스 추가
> ⚠️ 이 평가 프레임워크는 OpenAI text-embedding-3-small 임베딩 +
> 실제 cross-encoder 모델 로드를 전제로 한다.
> 키가 없거나 모델 로드가 실패하면 평가가 중단된다.
> 이는 의도된 동작이며, fallback으로 측정값을 생성하지 않는다.

──────────────────────────────────────────
단계 2. 한국어 모델 로드 실패 원인 해결
──────────────────────────────────────────

2-1. requirements.txt 갱신
- sentence-transformers >= 2.7.0 (safetensors 지원 안정 버전)
- transformers >= 4.40.0
- torch >= 2.1
- safetensors (명시적 추가)
- accelerate (일부 reranker 로드에 필요)

2-2. cross_encoder.py에 다중 로드 전략 도입
- 1차 시도: sentence_transformers.CrossEncoder(hf_id)
- 1차 실패 시 2차: transformers.AutoModelForSequenceClassification + AutoTokenizer 직접 로드
- 2차도 실패 시 모델별로 raise (해당 모델만 제외, 다른 모델 실험은 계속)
- 각 모델 로드 결과를 results/model_load_status.json에 기록:
    {
      "production_baseline": {"status": "ok", "loader": "sentence-transformers", "load_seconds": 12.3},
      "kpf_ko": {"status": "ok", "loader": "transformers-direct", "load_seconds": 24.1},
      "ko_reranker": {"status": "failed", "error": "...", "attempted_loaders": [...]},
      ...
    }

2-3. 테스트 추가
- ai/experiments/rag_eval/tests/test_model_loading.py
- 4개 후보 모델이 모두 로드되는지 검증
- 하나라도 실패 시 해당 환경의 정확한 에러 메시지를 출력
- pytest로 실행 가능

2-4. README에 "환경 요구사항" 섹션 신설
- Python 버전, OS, GPU/CPU 여부
- 4개 모델의 첫 다운로드 용량 (대략)
- 알려진 로드 이슈 + 해결 방법 (현재 프로젝트에서 발견한 것)
- 모든 모델 로드 검증 명령: `python -m pytest tests/test_model_loading.py -v`

──────────────────────────────────────────
단계 3. 측정 가능성 게이트 도입
──────────────────────────────────────────

3-1. ai/experiments/rag_eval/src/preflight.py 신설
- 본격 측정 전에 다음을 모두 검증 (하나라도 실패하면 측정 중단):
  a) config.embedding.provider == "openai"
  b) OPENAI_API_KEY 환경변수 존재 + 더미 호출 1회 성공
  c) Redis 연결 가능 + scan으로 최소 1개 키 확인
  d) cross_encoders.candidates의 모든 모델 로드 성공
  e) eval_set.json 항목 ≥ 30개 + correct_answer_ids가 비어있지 않음
- 게이트 결과를 results/preflight_{timestamp}.json 으로 기록

3-2. runner.py 수정
- 첫 줄에서 preflight 실행, 실패 시 즉시 sys.exit(1)
- preflight 통과 후에만 본 측정 진입
- 결과 JSON에 다음 메타데이터 포함:
    "preflight_passed": true,
    "embedder": "openai/text-embedding-3-small",
    "models_loaded": [...],
    "timestamp": "...",
    "git_commit": "..."

3-3. 결과 파일 명명 규칙
- preflight 미통과 상태에서 만들어진 모든 파일은 results/_invalid/ 로 자동 이동
- preflight 통과 시에만 results/ 루트에 baseline_*.json 저장
- README에 "results/_invalid/ 의 파일은 측정값으로 사용 금지" 명시

──────────────────────────────────────────
단계 4. 재측정 실행
──────────────────────────────────────────

4-1. STEP A baseline 재측정
- python -m src.runner
- preflight 통과 확인
- results/baseline_threshold_sweep_{date}.json + .png 생성
- 이전 결과(local-hash 기반)는 results/_invalid/legacy_localhash_*.* 로 이동

4-2. STEP B Cross-Encoder 비교 재측정
- python -m experiments.cross_encoder_comparison
- 4개 모델 모두 로드된 상태에서 평가 셋 전체 통과
- results/cross_encoder_comparison_{date}.json + .md 생성
- 모델별 raw score 분포 + F1 곡선 그래프 저장

4-3. 결과 판정 (자동화)
- experiments/decide_winner.py 작성
- 입력: cross_encoder_comparison_*.json
- 판정 기준 (v1 지시서의 성공 기준 그대로):
    Top-3 정답률 ≥ baseline + 10%p AND latency p95 ≤ baseline × 2
- 통과한 모델이 있으면: 해당 모델을 "운영 후보"로 결정 → results/decision.md
- 통과한 모델이 없으면: "현행 유지 + 원인 분석" → results/decision.md
  * 이 경우 결정문에 다음 포함: 가능한 원인 가설 3개 + 다음 실험 제안

──────────────────────────────────────────
단계 5. PR 작성
──────────────────────────────────────────

5-1. PR 제목
refactor(rag-eval): remove fallback paths and re-measure with real embeddings

5-2. PR 본문에 다음 표 포함
| 항목 | v1 (local-hash) | v2 (openai) | 변화 |
|---|---|---|---|
| Embedder | local-hash | text-embedding-3-small | 실측정 가능 |
| 로드 성공 모델 | 2/4 | 4/4 (또는 N/4) | 한국어 후보 포함 |
| Baseline Top-3 | 19.44% (무효) | X% | 실측정 |
| Best 모델 Top-3 | 19.44% (무효) | Y% | 실측정 |
| 결정 | 다국어 모델 선정 (근거 부족) | (재판정 결과) | 데이터 기반 |

5-3. PR 본문에 다음 명시
- 이전 결과를 _invalid/로 이동한 이유
- 한국어 모델 중 로드 실패가 남아있다면 그 환경 조건
- decide_winner.py 결과를 그대로 인용 (운영 후보 또는 현행 유지)
- 만약 "현행 유지" 결정이 나왔다면, STEP C는 "현재 운영값 주변 좁은 grid"로 축소될 수 있음을 명시

[Constraints]
- v2의 모든 코드 변경은 ai/experiments/rag_eval/ 안에서만 발생
- backend/, ai/src/, ai/Dockerfile 등 다른 영역은 손대지 않음
- preflight 게이트를 우회하는 옵션(--skip-preflight 등)을 만들지 말 것
- "현행 유지" 결정이 나오면 그 결정을 받아들일 것 (억지로 모델 교체 결론 내지 말 것)
- OpenAI API 비용 모니터링 (예상 비용을 실행 전 README에 명시)
- 한국어 모델 다운로드 용량이 클 수 있음 (BAAI/bge-reranker-v2-m3는 ~2GB)
  * .gitignore에 모델 캐시 경로 추가 확인

[Output]
- 수정된 config.yaml, embedder.py, cross_encoder.py, runner.py
- 신설된 preflight.py, decide_winner.py
- tests/test_model_loading.py (4개 모델 모두 로드 검증)
- 갱신된 README.md (경고 박스 + 환경 요구사항 + 결과 해석)
- results/preflight_*.json, baseline_*, cross_encoder_comparison_*, decision.md
- 이전 무효 결과는 results/_invalid/ 로 이동
- PR 본문에 v1 vs v2 비교표 + 재판정 결과

[성공 기준]
- preflight 게이트가 동작 (의도적으로 OPENAI_API_KEY를 비워서 실행하면 즉시 실패해야 함)
- results/ 에 있는 모든 측정값이 실제 임베딩 기반 (local-hash 흔적 0)
- 4개 모델 중 로드 실패가 있다면 그 사실이 status JSON과 README에 명시됨
- decide_winner.py가 "운영 후보 선정" 또는 "현행 유지" 중 하나를 명확히 출력
- "현행 유지" 결정이 나오는 경우도 정상 종료 코드(0)로 처리
- v1의 19.44% 수치는 본 보고서/PR에서 일절 측정값으로 인용되지 않음
```

---

## 3. 사용자(시니어 아키텍트) 측의 후속 의사결정

재측정 결과에 따라 다음 분기가 발생한다.

### 분기 A: 한국어 모델이 baseline 대비 ≥10%p 상승
- 원래 계획대로 STEP B 머지 → STEP C(grid search) 진행
- v1 지시.md의 STEP C 프롬프트를 그대로 사용 가능

### 분기 B: 모든 모델이 baseline 근처 (≤5%p 차이)
- "Cross-Encoder 교체"는 효과가 없다는 결론
- STEP B는 "현행 유지" PR로 종결 (모델 교체 없이 평가 도구만 머지)
- STEP C의 탐색 공간을 좁힘:
  * `cross_threshold` 후보를 baseline 단일 값으로 고정
  * 실질 grid search는 `bi_threshold × bi_top_k × cross_top_n` 3차원
  * 총 조합: 6 × 4 × 3 = 72개로 축소

### 분기 C: 한국어 모델이 baseline 대비 5~10%p 상승
- 통계적 유의미성 검증 필요 (현재 평가 셋 36개로는 부족할 수 있음)
- STEP B를 중단하고 평가 셋을 80~100개로 확장 후 재측정
- 또는 분기 B로 진행하고 후속 실험에서 다시 검토

**v2 결과 보고 후 어느 분기로 갈지는 사용자가 결정한다.**
v2 작업 자체는 분기 판정까지만 책임지고, STEP C 진입 여부는 별도 의사결정이다.

---

## 4. v1 지시서와의 관계

- v1 `지시.md`는 **그대로 보존**한다 (작업 이력)
- v2 결과로 v1의 STEP B 결론이 뒤집히면, v1의 STEP C 프롬프트는 "분기 B"에 맞게 좁혀서 다시 발행한다
- v1의 STEP A 성공 기준(`Precision@5 = X%, Recall@5 = Y%`)은 **v2 측정값으로 채워진다** — v1에 표기된 19.44%는 모두 무효 처리

---

## 5. 다음 채팅에서 사용자가 할 일

1. 이 v2 지시서 검토 → 수정사항 있으면 알려주기
2. Codex에 v2 프롬프트 전달
3. Codex 작업 완료 후 결과(`decision.md`)를 가져와서 분기 A/B/C 판정 요청
4. 분기에 따라 STEP C 지시서를 v1 그대로 갈지, 좁힌 버전으로 갈지 결정
