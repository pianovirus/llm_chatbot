# 🚀 AI Agent

## 🛠️ 설치 및 실행 방법

### 설정 및 배포 (Python + Next.js)
```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필수 라이브러리 설치 (중요!)
pip install -r requirements.txt

# 서버 실행
python agent_core.py
```ㅁ

### 2. 프론트엔드 설정 (Next.js)

``` bash
# 라이브러리 설치
npm install

# 개발 서버 실행
npm run dev

# 배포
npm run build

npm run start

# change port
npm run start -- -p 80

# [실전 팁] 서버가 꺼지지 않게 하려면? (PM2 사용)
## PM2 설치:
npm install -g pm2

## Next.js 프로젝트 PM2로 실행:
# 프로젝트 루트 폴더에서 실행
pm2 start npm --name "my-rag-chatbot" -- start

## 상태 확인 및 관리:
pm2 status   # 현재 실행 중인 리스트 확인
pm2 logs     # 실시간 로그 확인
pm2 restart my-rag-chatbot  # 재시작

## 3. [주의 사항] 환경 변수(.env) 관리
배포 시 가장 많이 실수하는 부분입니다.
API 주소: 개발 때는 localhost:8000이었지만, 실제 배포 시 파이썬 백엔드 주소가 바뀐다면 .env를 해당 도메인이나 고정 IP로 수정하고 **다시 빌드(npm run build)**해야 합니다.

## 4. [배포 방식] 어디에 배포하시나요?
현재 어떤 환경에 배포하실 계획인가요? 환경에 따라 방법이 살짝 다릅니다.
Vercel (가장 추천): Next.js 만든 회사에서 운영하며, 깃허브 연결만 하면 자동으로 빌드부터 배포까지 끝내줍니다. (무료 플랜 존재)
개인 서버 (Ubuntu/AWS 등): 위에서 설명한 npm run build + PM2 조합으로 직접 운영합니다.
Docker: Dockerfile을 만들어 컨테이너로 배포합니다.
```

### 3. DB 설치 및 접속
```
docker pull postgres

docker run --name all-that-ask-db \
  -e POSTGRES_PASSWORD=1234 \
  -p 5432:5432 \
  -d postgres

  

docker exec -it pg-vector psql -U myuser -d llm_chatbot

\c llm_chatbot;

truncate table rag_vectors;
```


### 4. .env 설정
```
# Database Configuration
DB_HOST=127.0.0.1
DB_PORT=5432
DB_USER=myuser
DB_PASSWORD={your_password_here}
DB_NAME=llm_chatbot
DB_TABLE_NAME=rag_vectors

# LLM Configuration (Ollama - 맥북 리소스 고려 gemma2:9b 추천)
LLM_MODEL=gemma2:9b
OLLAMA_BASE_URL=http://localhost:11434

# Embedding Configuration (BGE-Small 로컬 실행용)
# 🚨 384차원을 사용하는 BGE-Small 모델로 설정
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
EMBEDDING_DIMENSION=384

BASE_DOCS_URL=http://localhost:8000/files/
```

# agent_core.py 이해 가이드

## 1. 현재 상태에 대한 이해

현재 `agent_core.py`는 하나의 큰 파일이지만 실제로는 여러 시스템 역할이 합쳐진 구조입니다.

파일을 나누지 못하는 것은 이상한 일이 아닙니다.

지금 단계는 **리팩토링 단계가 아니라 이해 단계**입니다.

파일 분리의 실제 순서는 다음과 같습니다.

```
이해 → 구조 인식 → 정리 → 파일 분리
```

지금은 **구조 인식 단계**입니다.

---

# 2. 파일 분리가 어려운 이유

## 2.1 전역 객체 공유

여러 함수가 다음 객체들을 공유하고 있습니다.

```python
embed_model
reranker
llm
llm_chain
rag_pipeline
```

이 객체들은 시스템 전체에서 사용되기 때문에  
파일을 분리하면 초기화 순서나 import 문제가 생길 수 있다는 불안이 생깁니다.

---

## 2.2 HTTP 서버가 전체 시스템을 직접 사용

`RAGHandler` 내부에서 다음 객체들을 직접 사용합니다.

- rag_pipeline
- llm_chain
- embed_model

그래서 파일을 나누면 다음과 같은 고민이 생깁니다.

```
어디서 import 해야 하지?
```

이 역시 정상적인 상황입니다.

---

# 3. 지금 필요한 것은 파일 분리가 아니다

현재 필요한 작업은 **파일 분리**가 아니라 **함수 지도 만들기**입니다.

즉 코드 수정 없이 다음 질문에 답하는 것입니다.

```
이 함수는 무슨 역할인가?
```

---

# 4. agent_core.py 함수 역할 지도

## 시스템 초기화

### get_llm()

LLM provider에 따라 모델 객체를 생성합니다.

- Gemini
- OpenAI
- Ollama

---

## 텍스트 유틸리티

### safe_single_line()

문자열을 한 줄로 정리합니다.

---

### normalize_keyword()

한국어 조사 등을 제거하여 검색 키워드를 정리합니다.

예

```
연차신청은 → 연차신청
```

---

### make_compact_query()

검색을 위해

- 특수문자 제거
- 공백 제거

된 문자열을 생성합니다.

---

### generate_phrase_candidates()

검색을 위한 phrase 후보를 생성합니다.

```
original
compact
spaced_phrase
pair_phrase
triple_phrase
```

---

### extract_search_keywords()

사용자 질문에서 핵심 검색 키워드를 추출합니다.

불필요한 단어(stopwords)는 제거합니다.

---

## 문서 처리

### build_document_url()

문서 title을 기반으로 PDF URL을 생성합니다.

---

### limit_results_per_title()

한 문서에서 너무 많은 chunk가 검색되는 것을 제한합니다.

---

## 검색 품질 개선

### rerank_results()

reranker 모델을 사용하여 검색 결과 순서를 재정렬합니다.

---

### rewrite_query_for_retrieval()

사용자 질문을 검색 친화적인 질의로 변환합니다.

예

```
휴가 신청 어떻게 해요?
→ 휴가 신청 절차
```

---

## 핵심 검색 함수

### get_internal_context()

이 시스템의 핵심 retrieval 함수입니다.

다음 작업을 수행합니다.

1. 질문 임베딩 생성
2. 키워드 추출
3. phrase 후보 생성
4. PostgreSQL hybrid search 실행
5. 결과 점수 계산
6. 문서별 chunk 제한
7. rerank 적용

---

## 답변 생성

### build_answer_prompt()

검색된 문서를 기반으로 LLM에 전달할 최종 프롬프트를 생성합니다.

---

# 5. LangGraph RAG 파이프라인

### rewrite_query_node()

사용자 질문을 검색용 질의로 변환합니다.

---

### retrieve_context_node()

DB에서 관련 문서를 검색합니다.

---

### prepare_prompt_node()

검색된 문서를 기반으로

- context 생성
- prompt 생성

을 수행합니다.

---

### create_rag_pipeline()

전체 RAG 파이프라인을 구성합니다.

```
rewrite_query
→ retrieve_context
→ prepare_prompt
```

---

# 6. HTTP 서버

## RAGHandler

API 요청을 처리하는 서버 클래스입니다.

---

### _send_json()

JSON 응답을 전송합니다.

---

### _send_sse()

SSE 스트리밍 이벤트를 전송합니다.

---

### do_GET()

다음 요청을 처리합니다.

```
/files
/search
```

---

# 7. /search 요청 처리 흐름

전체 시스템 흐름

```
사용자 질문
↓
do_GET()
↓
rag_pipeline.stream()
↓
rewrite_query_node()
↓
retrieve_context_node()
↓
prepare_prompt_node()
↓
LLM 응답 생성
↓
SSE 스트리밍 반환
```

---

# 8. RAG 핵심 함수

이 시스템에서 가장 중요한 함수는 다음 세 가지입니다.

```
rewrite_query_for_retrieval
get_internal_context
build_answer_prompt
```

각 단계는 다음 역할을 합니다.

```
질문 정제
→ 문서 검색
→ 근거 기반 답변 생성
```

이 세 단계가 **RAG 시스템의 핵심 구조**입니다.

---

# 9. 이해 단계 체크리스트

다음 질문에 답할 수 있으면 큰 그림을 이해한 상태입니다.

1. 사용자 질문이 어디서 들어오는가
2. 검색 질의는 어디서 변환되는가
3. DB 검색은 어디서 수행되는가
4. 최종 프롬프트는 어디서 만들어지는가

---

# 10. 현재 단계 목표

현재 목표는 파일 분리가 아닙니다.

현재 목표

```
각 함수가 어떤 역할을 하는지 이해한다
```

이 작업만으로도

```
막연한 공포 → 이해 가능한 구조
```

로 바뀝니다.

---

# 11. 중요한 사실

파일을 나눌 수 있어야 코드를 이해한 것이 아닙니다.

실제 순서는 다음과 같습니다.

```
함수 역할 이해
→ 호출 흐름 이해
→ 구조 인식
→ 파일 분리
```

지금은 **구조 인식 단계**입니다.

# RAG 성능 튜닝 기록 (Performance Optimization)

이 문서는 고객 오류 대응용 RAG 챗봇의 **응답 속도 개선 과정**을 정리한 것이다.  
주요 목표는 **답변 품질을 유지하면서 응답 시간을 줄이는 것**이었다.

대상 챗봇은 다음과 같은 고객 문의를 처리한다.

- 설치 실패 오류 해결 방법
- 로그 위치 확인
- 권한 설정 방법
- 설정 오류 및 연동 문제

---

# 1. 초기 성능 문제

초기 측정 결과 응답 시간이 너무 길었다.

    pipeline: ~4.7s
    llm invoke: ~5.2s
    total: ~10s

문제점

- 사용자 경험 저하
- 실시간 고객 대응 챗봇으로 사용하기 어려움

---

# 2. 원인 분석

성능 병목은 크게 두 가지였다.

## 1) Retrieval Pipeline 지연

- 불필요한 Query Rewrite
- 과도한 Context 생성
- Prompt에 많은 문서가 포함됨

## 2) LLM 생성 지연

- Prompt 크기가 너무 큼
- 문서 Context가 과도하게 포함됨
- 답변 형식이 지나치게 복잡함
- Preview 모델 사용

---

# 3. 최적화 과정

다음과 같은 단계로 성능 튜닝을 진행하였다.

---

# 3.1 Query Rewrite 비활성화

Query Rewrite 단계가 latency를 증가시키고 있었다.

    QUERY_REWRITE_ENABLED=false

결과

    pipeline
    4.7s → 약 2.4s

---

# 3.2 Prompt Context 크기 축소

기존 구현에서는 검색 결과 전체가 Prompt에 포함되었다.

    context = "\n".join([f"제목: {r['title']}\n내용: {r['content']}" for r in results])

이로 인해 Prompt 길이가 과도하게 증가하였다.

    prompt length ≈ 2400+

### 개선 방법

Prompt에 포함되는 문서 수와 길이를 제한하였다.

    MAX_RESULTS_IN_PROMPT = 2
    MAX_CONTENT_CHARS_PER_RESULT = 500

결과

    prompt length
    2462 → 약 1100

---

# 3.3 Retrieval 결과 중복 제거

같은 문서 제목이 여러 번 포함되는 것을 방지하였다.

    seen_titles = set()

이를 통해 불필요한 Context 중복을 제거하였다.

---

# 3.4 문서 Content 길이 제한

각 문서 Chunk의 길이를 제한하였다.

    _trim_text(content, MAX_CONTENT_CHARS_PER_RESULT)

효과

- Prompt 크기 감소
- LLM 입력 토큰 감소
- 생성 속도 개선

---

# 3.5 Prompt 지시문 간소화

기존 프롬프트는 장문의 구조화된 답변을 유도하였다.

기존 형식

    ## 📢 조치 안내
    ### 🛠️ 상세 절차
    ### 💡 주의 사항

개선된 지시문

    지식 데이터를 근거로 답변하세요.
    데이터에 없는 내용은 추측하지 마세요.
    간결하게 작성하세요.

효과

- 답변 길이 감소
- 생성 시간 감소

---

# 3.6 LLM 모델 변경

기존 모델

    gemini-3-flash-preview

Preview 모델은 latency가 일정하지 않았다.

변경 모델

    gemini-flash-lite-latest

결과

    LLM latency
    약 5s → 약 1s

---

# 4. 최종 성능 결과

최적화 후 측정 결과

    pipeline: 2.16s
    prompt length: 1108
    llm invoke: 1.06s
    total: 3.22s

성능 개선

    약 10초 → 약 3.2초

약 **3배 이상의 성능 개선**을 달성하였다.

---

# 5. 품질 영향

답변 품질은 크게 저하되지 않았다.

오히려 일부 경우에서는 품질이 개선되었다.

원인

- Context 노이즈 감소
- 핵심 문서 중심 Retrieval
- Prompt 단순화

효과

- 이전에 찾지 못했던 답변 탐색 성공
- Hallucination 감소

---

# 6. 현재 아키텍처

최종 RAG 구조

    사용자 질문
          ↓
    LangGraph Pipeline
          ↓
    Vector Retrieval (PostgreSQL + pgvector)
          ↓
    Context Filtering / Trimming
          ↓
    Gemini Flash Lite
          ↓
    답변 생성 + 문서 링크 제공

---

# 7. 향후 개선 계획

## 데이터 확장

다음 영역의 문서를 추가할 예정

- 설치 실패 오류
- 로그 위치
- 권한 설정
- 설정 오류
- 연동 문제

## 질문 테스트 데이터 구축

실제 고객 질문 기반 테스트 세트를 구축하여 다음 항목을 검증한다.

- 답변 정확도
- 문서 링크 정확성
- 응답 속도
- Hallucination 여부

## 문서 Chunk 구조 개선

단순 길이 기반 분할 대신  
**문제 해결 단위(chunk)** 기반 분할을 적용할 예정이다.

예

    설치 실패 해결 절차
    로그 확인 절차
    권한 설정 방법

---

# 8. 요약

성능 개선 핵심 전략

- Query Rewrite 비활성화
- Prompt Context 축소
- Retrieval 결과 제한
- 문서 Content 길이 제한
- 빠른 LLM 모델 사용

결과

    응답 속도
    10초 → 약 3초

답변 품질을 유지하면서 **RAG 성능을 크게 개선하였다.**