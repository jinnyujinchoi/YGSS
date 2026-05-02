# 지시서 v3 — Colab 재측정 실행

> **대상 에이전트:** Codex (VSCode)
> **플래너:** Claude Opus 4.7
> **작성일:** 2026-04-29
> **선행 문서:** `지시.md` (v1), `지시_v2_QA보강.md` (v2)
> **상태:** v2 코드 하드닝 완료 → **로컬 환경 제약으로 측정 미실행** → Colab으로 환경 이전

---

## 0. 이 문서의 위치

v2가 끝난 뒤 측정 단계에서 두 가지 환경 제약에 막혔다.

1. **OpenAI quota 부족** → 사용자가 결제로 해결 (트랙 1, Codex 작업 아님)
2. **로컬 디스크 부족 + HuggingFace 모델 로드 실패** → Colab으로 환경 이전 (트랙 2, 본 지시서)

본 문서는 **트랙 2**를 다룬다. v2의 하드닝 코드를 거의 그대로 재사용하되, Colab 환경에서 실행 가능한 노트북을 만들고 결과를 다시 가져와 PR로 묶는다.

부수 목표: **다음 RAG 프로젝트에서도 재사용 가능한 Colab 노트북 템플릿을 만든다.**
이 노트북은 일회용이 아니라 자산이다.

---

## 1. 작업 범위

### 브랜치
**`refactor/rag-measurement-execution`** (from `refactor/rag-eval-hardening`)

```
master (보호)
   │
   └── refactor/rag-eval-framework      ← v1 STEP A (완료)
          │
          └── refactor/ko-cross-encoder  ← v1 STEP B (완료)
                  │
                  └── refactor/rag-eval-hardening  ← v2 (완료, 측정 미실행)
                          │
                          └── refactor/rag-measurement-execution  ← v3 (이번 작업)
```

### 손대는 범위
- `ai/experiments/rag_eval/notebooks/` (신설) — Colab 노트북
- `ai/experiments/rag_eval/results/` — Colab 실행 결과 가져와서 커밋
- `ai/experiments/rag_eval/README.md` — Colab 실행 가이드 추가
- `ai/experiments/rag_eval/src/` — Colab 호환을 위한 최소 수정만 (예: 경로 처리)

### 손대지 않는 범위
- `backend/` 전체
- `ai/src/` (server.py 포함)
- v2의 핵심 가드 로직 (preflight, fallback 제거 구조)는 보존

---

## 2. Codex 프롬프트

```
[Context]
프로젝트: YGSS
레포 루트: /Users/jinnysmacbookair/DEV/YGSS

선행 작업:
- v2 (refactor/rag-eval-hardening) 완료
- ai/experiments/rag_eval/ 의 코드 하드닝(fallback 제거, preflight 게이트, 무효 결과 _invalid 이동) 완료
- 그러나 로컬에서 OpenAI quota + 디스크 부족 + HF 모델 로드 실패로 측정 미실행

확인된 환경 제약:
1. 로컬 macOS: 디스크 1.1GiB 여유 → 2GB급 모델 3개 로드 불가
2. HF 모델 로드 에러: "pytorch_model.bin not found" → safetensors 처리 이슈 (sentence-transformers 또는 transformers 버전 호환 문제 가능성)
3. Redis: 샌드박스에서는 권한 부족, 권한 상승 시 통과
4. OpenAI: 사용자가 별도로 결제 정상화 진행 중 (이 작업과 병행)

핵심 발견:
- runner.py는 source_mode="sql_dump"일 때 운영 Redis 없이도 SQL 덤프에서 직접 임베딩을 생성해 측정 가능
- 즉 Colab에 SQL 덤프 + eval_set + src/ 코드만 가져가면 측정 완전 재현 가능

[Task]
ai/experiments/rag_eval/notebooks/ 에 Colab/Kaggle 환경에서 실행 가능한 평가 노트북을 만들고,
실행 결과(baseline 측정, cross-encoder 비교, decision)를 results/ 에 가져와 커밋한다.

단계:

──────────────────────────────────────────
단계 1. notebooks/ 디렉토리 구조 신설
──────────────────────────────────────────

ai/experiments/rag_eval/notebooks/
  ├── README.md                          # Colab 실행 가이드
  ├── 01_baseline_threshold_sweep.ipynb  # STEP A 재측정
  ├── 02_cross_encoder_comparison.ipynb  # STEP B 4모델 비교
  ├── 03_decision.ipynb                  # 최종 판정 자동화
  └── colab_bootstrap.py                 # 공통 부트스트랩 (각 노트북에서 import)

각 노트북은 다음 공통 구조를 따른다:
  Cell 1: 환경 검증 (GPU 확인, Python 버전)
  Cell 2: 의존성 설치 (!pip install ...)
  Cell 3: 레포 클론 또는 파일 업로드 처리
  Cell 4: colab_bootstrap.py 실행 (경로 셋업, secrets 로드)
  Cell 5: src/ 코드를 그대로 import해서 측정 실행
  Cell 6: 결과 저장 (Drive 마운트 또는 다운로드 셀)

──────────────────────────────────────────
단계 2. colab_bootstrap.py 작성
──────────────────────────────────────────

목적: Colab과 로컬 양쪽에서 동일하게 동작하는 환경 셋업 모듈.

기능:
- google.colab 모듈 존재 여부로 환경 감지
- Colab이면:
  * google.colab.userdata에서 OPENAI_API_KEY 로드 (Colab Secrets 권장)
  * /content/YGSS-rag-eval/ 작업 디렉토리 생성
  * eval_set.json, basic_insert.sql, src/ 디렉토리 업로드 또는 git clone 처리
  * source_mode를 자동으로 "sql_dump"로 강제 (Redis 없으므로)
- 로컬이면:
  * load_dotenv()로 기존 동작
  * 변경 없이 그대로 사용

환경별 분기 함수:
  def setup_environment() -> dict:
      """returns {'env': 'colab'|'local', 'workdir': Path, 'config_overrides': {...}}"""

──────────────────────────────────────────
단계 3. 01_baseline_threshold_sweep.ipynb 작성
──────────────────────────────────────────

내용:
- src.runner.main()을 그대로 호출
- 단, config 오버라이드로 source_mode="sql_dump" 강제
- 이미 v2의 하드닝 가드(preflight, _invalid 이동)가 작동하므로 추가 안전장치 없이 그대로 실행
- 결과 파일을 /content/results/ 에 저장 후 google.colab.files.download() 또는 Drive로 export

기대 출력:
- baseline_threshold_sweep_{ts}.json
- baseline_threshold_sweep_{ts}.png
- preflight_{ts}.json (통과 상태)

──────────────────────────────────────────
단계 4. 02_cross_encoder_comparison.ipynb 작성
──────────────────────────────────────────

내용:
- v2 config의 4개 cross-encoder 후보를 모두 로드
  * cross-encoder/ms-marco-MiniLM-L-12-v2 (baseline)
  * dragonkue/bge-reranker-v2-m3-ko (한국어 SOTA)
  * Dongjin-kr/ko-reranker (한국어 fine-tuned)
  * BAAI/bge-reranker-v2-m3 (다국어 SOTA)
- Colab T4/L4 GPU 환경에서 로드 (CPU도 가능하지만 느림)
- v2의 cross_encoder.py 다중 로더 그대로 사용
- 한국어 모델 로드 실패 원인이 v2 환경에선 "디스크 부족 + safetensors"였다면, Colab에선 둘 다 해결됨
- 만약 Colab에서도 실패하는 모델이 있으면 status JSON에 기록하고 측정 계속

측정:
- 각 모델별 평가 셋 36개 통과
- Top-1, Top-3, Top-5 정답률
- MRR
- 평균 latency (모델별)
- raw score 분포 + sigmoid 정규화 점수 분포 (히스토그램)
- F1 곡선으로 모델별 최적 cutoff 도출

기대 출력:
- cross_encoder_comparison_{ts}.json
- cross_encoder_comparison_{ts}.md (사람이 읽는 리포트)
- model_load_status_{ts}.json
- threshold_calibration_{model}.png (모델별)

──────────────────────────────────────────
단계 5. 03_decision.ipynb 작성
──────────────────────────────────────────

내용:
- 01번, 02번 결과를 입력으로 받음
- v2 지시서의 decide_winner.py 로직을 노트북에 인라인
- 판정 기준 (v1 지시서 그대로):
    Top-3 정답률 ≥ baseline + 10%p AND latency p95 ≤ baseline × 2
- 분기:
  * 통과한 모델 있음 → 운영 후보 결정
  * 통과한 모델 없음 → 현행 유지 + 원인 분석 (가설 3개 + 다음 실험 제안)

기대 출력:
- decision_{ts}.md
  * 결정 (운영 후보 / 현행 유지)
  * 4모델 비교표
  * Before/After 메트릭 표
  * 백엔드 적용용 변경 라인 (운영 후보 결정 시)
  * 후속 실험 제안 (현행 유지 결정 시)

──────────────────────────────────────────
단계 6. notebooks/README.md 작성
──────────────────────────────────────────

다음을 포함:
1. 사전 준비물
   - Colab 또는 Kaggle 계정
   - OpenAI API 키 (quota 정상화 상태)
   - 업로드 파일: src/, dataset/eval_set.json, exec/basic_insert.sql, config.yaml, requirements.txt
2. Colab Secrets 설정 방법
   - 좌측 열쇠 아이콘 → OPENAI_API_KEY 추가 → Notebook access 허용
3. 실행 순서
   - 01 → 02 → 03 순서로 실행
   - 각 노트북 결과(JSON/PNG/MD)를 다운로드
4. 로컬로 결과 가져오는 방법
   - Drive 마운트 또는 직접 다운로드
   - results/ 디렉토리에 배치 후 git add
5. 비용 예상
   - 평가 셋 36개 × 7 임계값 = 임베딩 호출 ~252회
   - text-embedding-3-small 기준 약 $0.0001 × 252 ≈ $0.025 (셋 캐싱 시 더 적음)
   - 모델 추론은 Colab GPU 무료 한도 내 처리 가능
6. Kaggle 대안
   - 인터넷 연결이 켜져 있어야 함 (HF 모델 다운로드용)
   - Notebook secrets에 OpenAI 키 등록
   - 이외 동일

──────────────────────────────────────────
단계 7. src/ 코드 Colab 호환 점검 (최소 수정)
──────────────────────────────────────────

다음만 점검 후 필요시 수정:
- 모든 경로 처리가 Path.cwd() 기준이라 작업 디렉토리만 맞으면 동작 (확인 필요)
- runner.py의 _get_git_commit()이 Colab에서 실패하지 않는지 확인 (이미 try-except 있음, OK)
- redis_search.py가 import 시점에 Redis에 연결하지 않는지 확인 (lazy init이어야 함)
- preflight.py가 source_mode="sql_dump"일 때 redis_scan 체크를 skip하는 로직 확인/추가

수정이 필요한 경우만 최소한으로 수정. 큰 리팩토링 금지.

──────────────────────────────────────────
단계 8. 결과 가져와서 PR 작성
──────────────────────────────────────────

Colab 실행 → 결과 다운로드 → 로컬 results/에 배치 → git add → PR 생성.

PR 제목:
feat(rag-eval): execute v2 measurement on Colab and decide winner

PR 본문:
1. v2에서 미실행했던 측정을 Colab 환경에서 완료
2. 4모델 비교표
3. decision_{ts}.md 인용
4. v1 → v2 → v3 흐름 요약
5. 다음 액션 (분기 A/B/C 중 하나)

[Constraints]
- 노트북은 Colab 무료 티어(T4 GPU 또는 CPU)에서 동작 가능해야 함
- BAAI/bge-reranker-v2-m3 (~2GB)는 Colab 무료 디스크(~70GB) 안에서 충분히 로드 가능
- OpenAI 키는 절대 노트북에 하드코딩하지 말 것 (Colab Secrets 또는 input() 사용)
- v2의 preflight 게이트를 노트북에서 우회하지 말 것 (그대로 통과시켜야 함)
- src/ 코드는 노트북 전용 분기를 추가하지 않는다 (env 감지는 colab_bootstrap.py에서만)
- 노트북은 reproducible해야 함 (random seed 고정, 모델 버전 명시)

[Output]
- ai/experiments/rag_eval/notebooks/ 디렉토리 일체
- ai/experiments/rag_eval/results/ 에 Colab 실행 결과 (baseline_*, cross_encoder_comparison_*, decision_*, model_load_status_*)
- ai/experiments/rag_eval/README.md 갱신 (Colab 섹션 추가)
- src/ 최소 수정 (필요한 경우만)
- PR 본문에 v1→v2→v3 흐름 + 결정문

[성공 기준]
- 노트북을 새 Colab 세션에서 실행하면 처음부터 끝까지 통과 (재현 가능)
- 4모델 중 최소 3개 로드 성공 (한국어 모델 1개 이상 포함 필수)
- preflight 통과 + baseline 측정값 생성
- decision_{ts}.md가 "운영 후보 선정" 또는 "현행 유지" 중 하나를 명확히 출력
- 노트북 자체가 다음 RAG 프로젝트에서 모델만 바꿔서 재사용 가능한 구조

[금지 사항]
- v2의 하드닝 가드를 우회하지 말 것
- 측정 결과가 안 나왔는데 임의로 "선정"하지 말 것 (현행 유지 결정도 정당한 결과)
- OpenAI 키나 다른 secret을 노트북에 직접 입력하지 말 것
- runner.py를 Colab 전용으로 갈아엎지 말 것 (공통 코드 유지)
```

---

## 3. 트랙 1 (사용자 작업) — OpenAI Quota 정상화

Codex 작업과 별개로 사용자가 직접 처리해야 하는 항목.

### 체크리스트
- [ ] OpenAI Platform → Billing → 결제 수단 등록/확인
- [ ] Usage limits 확인 (Hard limit이 quota를 막고 있지 않은지)
- [ ] API Keys → 새 키 생성 (또는 기존 키 유효성 확인)
- [ ] `text-embedding-3-small` 사용 가능 권한 확인 (대부분 기본 포함)
- [ ] 테스트: `curl https://api.openai.com/v1/embeddings -H "Authorization: Bearer $KEY" -d '{"model":"text-embedding-3-small","input":"test"}'`
- [ ] Colab Secrets에 키 등록

### 예상 비용 (실측 기준)
- baseline_threshold_sweep: 36 questions × 1회 임베딩 ≈ $0.0003
- cross_encoder_comparison: 임베딩 재사용 가능 (캐시) ≈ $0
- 전체: 1달러 미만

비용보다 환경 셋업이 더 큰 작업이다.

---

## 4. Colab/Kaggle 선택 가이드

### Colab 권장 (기본)
- 장점: Drive 연동 쉬움, T4 GPU 무료 제공, Secrets 기능
- 단점: 90분 비활성 시 세션 끊김 (전체 측정은 30분 이내 끝나므로 문제 없음)

### Kaggle 대안
- 장점: 12시간 세션, P100/T4 무료
- 단점: HF 모델 다운로드 시 인터넷 토글 필요, Drive 없음

처음엔 Colab으로 가고, 막히면 Kaggle로 옮길 수 있게 두 환경 모두 호환되는 노트북을 만든다.

---

## 5. 자산화 관점 (다음 프로젝트를 위한 재사용)

이번에 만드는 노트북은 일회용이 아니다. 다음 RAG 프로젝트에서 모델 후보와 평가 셋만 바꾸면 그대로 재사용 가능해야 한다.

### 재사용 가능 요소
- `colab_bootstrap.py` — 환경 감지 로직, OpenAI Secrets 패턴
- `01_baseline_threshold_sweep.ipynb` — Bi-Encoder 임계값 스윕 템플릿
- `02_cross_encoder_comparison.ipynb` — N개 모델 비교 템플릿
- `03_decision.ipynb` — 자동 판정 로직

### 다음 프로젝트에서 바꿀 부분만 명확히
README에 "이 노트북을 다른 프로젝트에 적용하려면 다음만 바꾸세요" 섹션 추가:
- config.yaml (모델 후보, 임계값 그리드)
- dataset/eval_set.json (평가 셋)
- 데이터 소스 (sql_dump 또는 Redis)

---

## 6. v3 작업 후 분기

v3 결과로 결정문이 나오면 다음 중 하나로 진행:

- **분기 A** (한국어 모델 ≥10%p 상승): v1 STEP C(grid search) 진행 가능
- **분기 B** (모든 모델 동등 ≤5%p): STEP B "현행 유지"로 종결, STEP C 좁은 grid로 축소
- **분기 C** (5~10%p 애매): 평가 셋 80~100개로 확장 후 재측정

이 분기 결정은 v3가 끝난 뒤 사용자와 별도 논의.

---

## 7. 다음 채팅에서 사용자가 할 일

1. 본 v3 지시서 검토
2. OpenAI billing 정상화 (트랙 1)
3. Codex에 v3 프롬프트 전달 (트랙 2)
4. Colab에서 노트북 실행 → 결과 다운로드 → 로컬 results/에 배치 → 커밋
5. decision.md 결과를 가져와 분기 A/B/C 판정 요청
