import os
# 🚨 Mac 환경에서 Segmentation Fault 방지
# - sentence-transformers / tokenizers 계열이 Mac에서 병렬 실행 시 간헐적으로 크래시 나는 경우가 있어 비활성화
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================================================
# 0) 기본 라이브러리 import
# =========================================================
import json
import re
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
import warnings
import ssl
import urllib.parse
import unicodedata
import threading
import mimetypes
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote
from typing import TypedDict, List, Dict, Any

# =========================================================
# 1) 환경설정(config.py)
# - .env에서 읽은 값들이 여기로 들어온다고 보면 됨
# - 예: LLM_PROVIDER, MODEL_NAME, DB_CONFIG, TABLE_NAME 등
# =========================================================
from config import *

warnings.filterwarnings("ignore")

# =========================================================
# 2) SSL 설정 (인증서 관련 에러 방지)
# - 일부 환경에서 HTTPS 요청 시 인증서 에러가 나는 걸 완화
# =========================================================
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# =========================================================
# 3) LLM / Embedding / Reranker 라이브러리
# =========================================================
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer, CrossEncoder
from langgraph.graph import StateGraph, START, END

# =========================================================
# 4) 로컬 문서(PDF) 제공을 위한 storage 폴더 준비
# - /files/<filename> 요청이 오면 여기에서 파일을 읽어 반환
# =========================================================
STORAGE_DIR = os.path.join(os.getcwd(), "storage")
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

# =========================================================
# 5) 멀티스레드 안전장치
# - ThreadingHTTPServer가 동시에 요청을 처리할 수 있으므로,
#   모델/리랭커/LLM 호출이 겹칠 때 안전하게 잠그기 위한 lock
# =========================================================
embedding_lock = threading.Lock()
rerank_lock = threading.Lock()
llm_lock = threading.Lock()

# =========================================================
# 6) LLM Factory
# - 환경변수(LLM_PROVIDER)에 따라 Gemini/OpenAI/Ollama 중 하나 선택
# =========================================================
# (역할) 환경설정에 따라 LLM 객체를 생성하는 팩토리 함수
def get_llm():
    """설정된 프로바이더에 따라 LLM 인스턴스를 생성합니다."""
    temp = 0
    if LLM_PROVIDER == "google":
        print(f"✨ [System] Google Gemini 엔진 사용 중... ({MODEL_NAME})")
        return ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            version="v1",
            temperature=temp
        )
    elif LLM_PROVIDER == "openai":
        print(f"✨ [System] OpenAI 엔진 사용 중... ({MODEL_NAME})")
        return ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=temp)
    else:
        print(f"✨ [System] Ollama 엔진 사용 중... ({MODEL_NAME})")
        return Ollama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=temp, timeout=180)

# =========================================================
# 7) 모델 초기화 (프로세스 시작 시 1회 로드)
# - embed_model: 문서/질문 임베딩 생성 (RAG 검색 핵심)
# - reranker: (옵션) 검색 결과 재정렬 모델
# - llm / llm_chain: 답변 생성 모델 + 출력 파서(String)
# =========================================================
print(f"⏳ [System] 임베딩 모델 로드 중... ({EMBED_MODEL_NAME})")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

reranker = None
if RERANK_ENABLED:
    try:
        print(f"⏳ [System] Reranker 모델 로드 중... ({RERANK_MODEL_NAME})")
        reranker = CrossEncoder(RERANK_MODEL_NAME)
    except Exception as e:
        print(f"❌ Reranker 로드 실패: {e}")

llm = get_llm()

# LLM 출력 형식을 "문자열"로 통일하기 위한 체인
# - Gemini/OpenAI/Ollama의 반환 타입 차이를 흡수하기 좋음
llm_chain = llm | StrOutputParser()

print("✅ [System] PostgreSQL RAG 엔진 가동 준비 완료.")
print("=" * 50)
print(f"현재 선택된 공급자: {LLM_PROVIDER}")
print(f"현재 선택된 모델명: {MODEL_NAME}")
print(f"실제 생성된 객체 타입: {type(llm)}")
print("=" * 50)

# =========================================================
# 8) LangGraph Pipeline State 정의
# - 파이프라인에서 전달/누적될 상태 값들의 타입 힌트
# =========================================================
class PipelineState(TypedDict, total=False):
    original_query: str           # 사용자가 입력한 원 질문
    effective_query: str          # 검색용으로 다듬은 질문(옵션)
    search_results: List[Dict[str, Any]]  # DB에서 찾은 문서 chunk들
    context_combined: str         # 검색 결과를 합친 컨텍스트 문자열
    prompt: str                   # LLM에 전달할 최종 프롬프트
    doc_links: Dict[str, str]     # 문서 제목 -> 다운로드 URL

# =========================================================
# 9) 텍스트 유틸: 검색 품질을 조금이라도 높이기 위한 전처리
# =========================================================
# (역할) 텍스트를 한 줄로 정리 (개행/연속 공백 제거)
def safe_single_line(text: str) -> str:
    """텍스트를 한 줄로 정리(연속 공백/개행 제거)."""
    return " ".join(text.strip().split()) if text else ""

# (역할) 한국어 조사/어미를 간단 규칙으로 제거해 검색용 키워드 정규화
def normalize_keyword(token: str) -> str:
    """
    한국어 조사/어미 비슷한 접미를 제거해서 검색 키워드를 정리.
    (완벽한 형태소 분석이 아니라, 가벼운 규칙 기반 정리)
    """
    if not token:
        return ""
    token = unicodedata.normalize("NFC", token.strip())
    suffixes = [
        "에서는","으로는","에게서","까지는","부터는",
        "에서","으로","에게","까지","부터",
        "처럼","하고","이며","이고","하면","라는","라고",
        "니다","어요","아요","를","을","은","는","이","가","도","만","와","과","에","의","로"
    ]
    for s in sorted(suffixes, key=len, reverse=True):
        if len(token) > len(s) + 1 and token.endswith(s):
            token = token[:-len(s)]
            break
    return token.strip()

# (역할) 특수문자/공백 제거한 compact query 생성 (phrase 매칭용)
def make_compact_query(query: str) -> str:
    """특수문자 제거 + 공백 제거 → 붙여쓴 형태의 compact 문자열 생성."""
    if not query:
        return ""
    q = re.sub(r"[\"'`“”‘’()\[\]{}:;,./\\!?@#$%^&*+=|<>~-]+", " ", safe_single_line(query))
    return q.replace(" ", "")

# (역할) 원문/compact/토큰 결합 등 다양한 phrase 후보 생성
def generate_phrase_candidates(query: str) -> Dict[str, str]:
    """
    검색에 활용할 phrase 후보 생성.
    - original: 원문
    - compact: 공백/특수문자 제거 버전
    - spaced_phrase: 핵심 토큰 2~3개를 띄어쓴 버전
    - pair/triple_phrase: 핵심 토큰을 붙여쓴 버전
    """
    q = safe_single_line(query)
    raw_tokens = re.findall(r"[가-힣A-Za-z0-9_-]{2,}", q)
    normalized = []
    seen = set()
    for t in raw_tokens:
        nt = normalize_keyword(t)
        if len(nt) >= 2 and nt not in seen:
            normalized.append(nt)
            seen.add(nt)
    return {
        "original": q,
        "compact": make_compact_query(q),
        "spaced_phrase": " ".join(normalized[:3]) if normalized else q,
        "pair_phrase": "".join(normalized[:2]),
        "triple_phrase": "".join(normalized[:3]),
    }

# (역할) 검색용 핵심 키워드 추출 (stopword 제거)
def extract_search_keywords(query: str, max_keywords: int = 4):
    """
    검색 키워드 후보를 뽑아냄.
    - stopwords 제거
    - 길이가 긴 키워드를 우선
    """
    raw_tokens = re.findall(r"[가-힣A-Za-z0-9_-]{2,}", query)
    stopwords = {"방법","문의","관련","설명","기능","화면","문서","가이드","알려줘","알려","주세요","부탁","어떻게","어디","왜","무엇"}
    cleaned, seen = [], set()
    for t in raw_tokens:
        n = normalize_keyword(t)
        if len(n) >= 2 and n not in stopwords and n not in seen:
            seen.add(n)
            cleaned.append(n)
    return sorted(cleaned, key=lambda x: -len(x))[:max_keywords]

# (역할) 문서 title을 파일 서빙 URL(/files/...)로 변환
def build_document_url(title_val: str):
    """문서 title을 /files/... URL로 변환."""
    name = title_val if title_val.lower().endswith(".pdf") else f"{title_val}.pdf"
    encoded = urllib.parse.quote(unicodedata.normalize("NFC", name))
    return f"{BASE_DOCS_URL.rstrip('/')}/{encoded}"

# (역할) 같은 문서(title)에서 최대 chunk 개수 제한 (컨텍스트 편향 방지)
def limit_results_per_title(results: list, max_per_title: int = 2):
    """
    같은 문서(title)에서 너무 많은 chunk가 뽑히면 컨텍스트가 편향되므로 제한.
    """
    limited, counter = [], defaultdict(int)
    for item in results:
        t = item.get("title", "Unknown")
        if counter[t] < max_per_title:
            limited.append(item)
            counter[t] += 1
    return limited

# (역할) (옵션) reranker 모델로 검색 결과 재정렬
def rerank_results(query: str, results: list, top_n: int = 5):
    """
    (옵션) reranker가 켜져 있으면 검색 결과를 재정렬.
    """
    if not results or not RERANK_ENABLED or reranker is None:
        return results[:top_n]
    try:
        pairs = [(query, f"{r['title']}\n{r['content']}") for r in results]
        with rerank_lock:
            scores = reranker.predict(pairs)
        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results[:top_n]
    except:
        return results[:top_n]

# =========================================================
# 10) Query Rewrite (옵션)
# - 질문을 "검색에 더 잘 걸리는 명사형"으로 바꿔 retrieval 품질을 올리는 용도
# =========================================================
# (역할) LLM으로 질문을 검색 친화형으로 재작성 (옵션)
def rewrite_query_for_retrieval(original_query: str) -> str:
    if not QUERY_REWRITE_ENABLED:
        return original_query
    prompt = f"사용자 질문을 검색용 명사 위주 키워드로 요약하세요. 결과만 한 줄로 출력.\n질문: {original_query}"
    try:
        with llm_lock:
            rewritten = llm.invoke(prompt)
        # LLM 종류에 따라 반환 타입이 달라 content만 뽑아내기
        content = rewritten.content if hasattr(rewritten, 'content') else str(rewritten)
        return safe_single_line(content)
    except:
        return original_query

# =========================================================
# 11) Retrieval (PostgreSQL + pgvector)
# - 벡터 검색 + 키워드 검색 + phrase 매칭을 점수로 합산하는 하이브리드 검색
# =========================================================
# (역할) DB에서 질문과 관련된 문서 chunk들을 검색해 top 결과를 반환
def get_internal_context(query: str):
    # (1) 질문을 임베딩 벡터로 변환
    with embedding_lock:
        query_vec = embed_model.encode(
            "Represent this sentence for searching relevant passages: " + query
        ).tolist()

    # (2) 키워드/phrase 후보 생성 (하이브리드 검색에 사용)
    k = extract_search_keywords(query)     # 현재 SQL에서는 query만 쓰지만, 확장 가능
    p = generate_phrase_candidates(query)

    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        register_vector(conn)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # (3) SQL: vector + keyword + phrase score를 합산
        sql = f"""
            WITH vector_matches AS (
                SELECT id, (embedding <=> %s::vector) AS dist
                FROM {TABLE_NAME}
                ORDER BY dist ASC
                LIMIT 50
            ),
            phrase_matches AS (
                SELECT id,
                       (CASE WHEN %s IS NOT NULL AND title ILIKE '%%'||%s||'%%' THEN 2.0 ELSE 0 END) AS phrase_score
                FROM {TABLE_NAME}
                LIMIT 100
            ),
            keyword_matches AS (
                SELECT id,
                       ts_rank_cd(content_search_vector, websearch_to_tsquery('simple', %s)) AS rank
                FROM {TABLE_NAME}
                LIMIT 100
            )
            SELECT t.*,
                   COALESCE(1.0/(60+v.dist*100),0)
                 + COALESCE(k.rank*10,0)
                 + COALESCE(p.phrase_score,0) AS combined_score
            FROM {TABLE_NAME} t
            LEFT JOIN vector_matches v ON t.id=v.id
            LEFT JOIN keyword_matches k ON t.id=k.id
            LEFT JOIN phrase_matches p ON t.id=p.id
            WHERE v.id IS NOT NULL OR k.id IS NOT NULL OR p.id IS NOT NULL
            ORDER BY combined_score DESC
            LIMIT {RERANK_CANDIDATE_LIMIT}
        """
        cursor.execute(sql, (query_vec, p['original'], p['original'], query))
        rows = cursor.fetchall()

        # (4) DB row -> 응답 dict로 정리
        results = [{
            "content": r["original_content"],
            "title": r["title"] or "Unknown",
            "url": build_document_url(r["title"]),
            "combined_score": float(r["combined_score"])
        } for r in rows]

        # (5) 같은 title 문서 편향 제한 + (옵션) rerank 적용
        results = limit_results_per_title(results, max_per_title=MAX_CHUNKS_PER_TITLE)
        return rerank_results(query, results, top_n=RERANK_TOP_N)

    except Exception as e:
        print(f"DB Error: {e}")
        return []
    finally:
        if conn:
            conn.close()

# =========================================================
# 12) Prompt Builder
# ⚠️ 형님 요청: 프롬프트는 수정하지 않음 (원본 그대로)
# =========================================================
# (역할) 검색된 context를 넣어 최종 LLM 프롬프트 문자열을 생성
def build_answer_prompt(original_query: str, effective_query: str, context: str) -> str:
    return (
        f"지식 데이터를 근거로 답변하세요. 모르면 모른다고 하세요.\n\n"
        f"[질문]: {original_query}\n"
        f"[데이터]:\n{context}\n\n"
        f"응답은 반드시 [## 📢 조치 안내], [### 🛠️ 상세 절차], [### 💡 주의 사항] 순서로 작성하세요."
    )

# =========================================================
# 13) LangGraph Nodes
# - rewrite -> retrieve -> prepare 순서로 state를 누적
# =========================================================
# (역할) 파이프라인 1단계: 질문을 검색용으로 재작성
def rewrite_query_node(state: PipelineState):
    return {"effective_query": rewrite_query_for_retrieval(state["original_query"])}

# (역할) 파이프라인 2단계: DB에서 관련 문서 chunk 검색
def retrieve_context_node(state: PipelineState):
    query = state.get("effective_query") or state["original_query"]
    return {"search_results": get_internal_context(query)}

# (역할) 파이프라인 3단계: context/prompt/doc_links를 조립
def prepare_prompt_node(state: PipelineState):
    results = state.get("search_results", [])

    # LLM에 넣을 context 문자열 구성 (title + content)
    context = "\n".join([f"제목: {r['title']}\n내용: {r['content']}" for r in results])

    # 프론트에 문서 링크도 같이 주기 위한 map
    links = {r["title"]: r["url"] for r in results if r["url"]}

    return {
        "context_combined": context,
        "prompt": build_answer_prompt(state["original_query"], state.get("effective_query", ""), context),
        "doc_links": links
    }

# (역할) LangGraph로 rewrite->retrieve->prepare 실행 흐름을 구성하고 컴파일
def create_rag_pipeline():
    graph = StateGraph(PipelineState)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("prepare_prompt", prepare_prompt_node)

    # 실행 순서
    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve_context")
    graph.add_edge("retrieve_context", "prepare_prompt")
    graph.add_edge("prepare_prompt", END)

    return graph.compile()

rag_pipeline = create_rag_pipeline()

# =========================================================
# 14) HTTP Server
# - /files/<name> : storage/에서 파일 다운로드
# - /search?query= : RAG 파이프라인 실행 + SSE로 스트리밍 응답
# - /feedback : 사용자 피드백을 DB에 임베딩으로 저장(학습/개선용)
# =========================================================
class RAGHandler(BaseHTTPRequestHandler):
    # (역할) 일반 JSON 응답을 반환하는 헬퍼
    def _send_json(self, status_code, payload):
        """일반 JSON 응답."""
        self.send_response(status_code)
        self.send_header("Content-type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    # (역할) SSE 이벤트 한 번을 전송하는 헬퍼
    def _send_sse(self, data):
        """SSE(Server-Sent Events)로 한 이벤트 전송."""
        self.wfile.write(f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8"))
        self.wfile.flush()

    # (역할) GET 요청 라우팅: /files, /search 처리
    def do_GET(self):
        parsed = urlparse(self.path)

        # -----------------------------
        # (A) /files/<filename> 제공
        # -----------------------------
        if parsed.path.startswith("/files/"):
            full_path = os.path.join(STORAGE_DIR, unquote(parsed.path[7:]))
            if os.path.exists(full_path):
                self.send_response(200)
                self.send_header("Content-type", mimetypes.guess_type(full_path)[0] or "application/octet-stream")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(full_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)
            return

        # -----------------------------
        # (B) /search?query=... 실행
        # - SSE로 진행 상태(status) + 답변(chunk) 스트리밍
        # -----------------------------
        if parsed.path == "/search":
            query = parse_qs(parsed.query).get("query", [""])[0].strip()

            self.send_response(200)
            self.send_header("Content-type", "text/event-stream; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                # 1) 상태: 질문 요약 단계
                self._send_sse({"status": "🧠🔗⚙️ 질문 요약 중..."})

                # 파이프라인 state 시작값
                state = {"original_query": query}

                # 2) LangGraph 파이프라인 실행 (rewrite -> retrieve -> prepare)
                for update in rag_pipeline.stream(state, stream_mode="updates"):
                    self._send_sse({"status": "🔍 검색 중..."})
                    for node, out in update.items():
                        state.update(out)
                        if node == "retrieve_context":
                            self._send_sse({"status": f"🔍 검색 완료 ({len(state['search_results'])}건)"})

                # 3) 상태: 답변 생성 시작
                self._send_sse({"status": "🤖✨ 답변 생성 중..."})

                # 4) LLM 답변을 chunk 단위로 스트리밍
                for chunk in llm_chain.stream(state["prompt"]):
                    self._send_sse({"status": "💬✍️ 답변 중..."})
                    if chunk:
                        self._send_sse({"chunk": chunk})

                # 5) 문서 링크가 있으면 답변 뒤에 붙여서 제공
                if state.get("doc_links"):
                    links = "\n".join([f"- [{t}]({u})" for t, u in state["doc_links"].items()])
                    self._send_sse({"chunk": f"\n\n---\n### 🔗 관련 문서\n{links}"})

                # 6) SSE 종료 시그널
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

            except Exception as e:
                self._send_sse({"error": str(e)})

    # (역할) POST 요청 라우팅: /feedback 처리
    def do_POST(self):
        # -----------------------------
        # (C) /feedback 저장
        # - 질문/답변을 하나의 텍스트로 만들고 임베딩해서 DB에 저장
        # -----------------------------
        if self.path == "/feedback":
            try:
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._save_feedback(data["query"], data["answer"])
                self._send_json(200, {"status": "success"})
            except Exception as e:
                self._send_json(500, {"error": str(e)})

    # (역할) 사용자 피드백을 임베딩 후 DB에 저장
    def _save_feedback(self, q, a):
        """feedback을 임베딩해서 TABLE_NAME에 저장."""
        text = f"질문: {q}\n답변: {a}"
        vec = embed_model.encode(text).tolist()

        conn = psycopg2.connect(**DB_CONFIG)
        try:
            register_vector(conn)
            cur = conn.cursor()
            cur.execute(
                f"INSERT INTO {TABLE_NAME} (content_type, data_source, original_content, embedding, title) "
                f"VALUES (%s, %s, %s, %s, %s)",
                ("text", "feedback", text, vec, "검증된 답변"),
            )
            conn.commit()
        finally:
            conn.close()

# =========================================================
# 15) 서버 실행 엔트리포인트
# =========================================================
if __name__ == "__main__":
    print(f"📡 {LLM_PROVIDER.upper()} 기반 RAG 가동: http://localhost:8000")
    ThreadingHTTPServer(("0.0.0.0", 8000), RAGHandler).serve_forever()