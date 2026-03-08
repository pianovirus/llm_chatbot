import os
# 🚨 Mac 환경에서 Segmentation Fault 방지
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any

# 1. 환경 변수 로드
load_dotenv()
warnings.filterwarnings("ignore")

# SSL 설정 (인증서 관련 에러 방지)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- LangChain 모델 라이브러리 ---
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer, CrossEncoder
from langgraph.graph import StateGraph, START, END

# 2. .env 설정값 로드
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
MODEL_NAME = os.getenv("LLM_MODEL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
TABLE_NAME = os.getenv("DB_TABLE_NAME")
BASE_DOCS_URL = os.getenv("BASE_DOCS_URL", "http://localhost:8000/files/")
DB_NAME = os.getenv("DB_NAME")

# Query Rewrite 및 Reranker 설정
QUERY_REWRITE_ENABLED = os.getenv("QUERY_REWRITE_ENABLED", "true").lower() == "true"
QUERY_REWRITE_MAX_CHARS = int(os.getenv("QUERY_REWRITE_MAX_CHARS", "80"))
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "false").lower() == "true"
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
RERANK_CANDIDATE_LIMIT = int(os.getenv("RERANK_CANDIDATE_LIMIT", "20"))
MAX_CHUNKS_PER_TITLE = int(os.getenv("MAX_CHUNKS_PER_TITLE", "2"))

# PostgreSQL 연결 정보
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "dbname": os.getenv("DB_NAME")
}

STORAGE_DIR = os.path.join(os.getcwd(), "storage")
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

embedding_lock = threading.Lock()
rerank_lock = threading.Lock()
llm_lock = threading.Lock()

# --- 🚀 LLM 팩토리 함수: 모델 로직 분리 ---

def get_llm():
    """설정된 프로바이더에 따라 LLM 인스턴스를 생성합니다."""
    temp = 0
    if LLM_PROVIDER == "google":
        print(f"✨ [System] Google Gemini 엔진 사용 중... ({MODEL_NAME})")
        return ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=GOOGLE_API_KEY, version="v1", temperature=temp)
    elif LLM_PROVIDER == "openai":
        print(f"✨ [System] OpenAI 엔진 사용 중... ({MODEL_NAME})")
        return ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=temp)
    else:
        print(f"✨ [System] Ollama 엔진 사용 중... ({MODEL_NAME})")
        return Ollama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=temp, timeout=180)

# 3. 모델 초기화
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
# 모든 모델의 출력을 String으로 통일하기 위한 파서 체인 생성
llm_chain = llm | StrOutputParser()

print("✅ [System] PostgreSQL RAG 엔진 가동 준비 완료.")

print("="*50)
print(f"현재 선택된 공급자: {LLM_PROVIDER}")
print(f"현재 선택된 모델명: {MODEL_NAME}")
print(f"실제 생성된 객체 타입: {type(llm)}")
print("="*50)

# --- LangGraph & 유틸리티 로직 (원래 코드 유지) ---

class PipelineState(TypedDict, total=False):
    original_query: str
    effective_query: str
    search_results: List[Dict[str, Any]]
    context_combined: str
    prompt: str
    doc_links: Dict[str, str]

def safe_single_line(text: str) -> str:
    return " ".join(text.strip().split()) if text else ""

def normalize_keyword(token: str) -> str:
    if not token: return ""
    token = unicodedata.normalize("NFC", token.strip())
    suffixes = ["에서는", "으로는", "에게서", "까지는", "부터는", "에서", "으로", "에게", "까지", "부터", "처럼", "하고", "이며", "이고", "하면", "라는", "라고", "니다", "어요", "아요", "를", "을", "은", "는", "이", "가", "도", "만", "와", "과", "에", "의", "로"]
    for s in sorted(suffixes, key=len, reverse=True):
        if len(token) > len(s) + 1 and token.endswith(s):
            token = token[:-len(s)]
            break
    return token.strip()

def make_compact_query(query: str) -> str:
    if not query: return ""
    q = re.sub(r"[\"'`“”‘’()\[\]{}:;,./\\!?@#$%^&*+=|<>~-]+", " ", safe_single_line(query))
    return q.replace(" ", "")

def generate_phrase_candidates(query: str) -> Dict[str, str]:
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
        "original": q, "compact": make_compact_query(q),
        "spaced_phrase": " ".join(normalized[:3]) if normalized else q,
        "pair_phrase": "".join(normalized[:2]), "triple_phrase": "".join(normalized[:3]),
    }

def extract_search_keywords(query: str, max_keywords: int = 4):
    raw_tokens = re.findall(r"[가-힣A-Za-z0-9_-]{2,}", query)
    stopwords = {"방법", "문의", "관련", "설명", "기능", "화면", "문서", "가이드", "알려줘", "알려", "주세요", "부탁", "어떻게", "어디", "왜", "무엇"}
    cleaned, seen = [], set()
    for t in raw_tokens:
        n = normalize_keyword(t)
        if len(n) >= 2 and n not in stopwords and n not in seen:
            seen.add(n)
            cleaned.append(n)
    return sorted(cleaned, key=lambda x: -len(x))[:max_keywords]

def build_document_url(title_val: str):
    name = title_val if title_val.lower().endswith(".pdf") else f"{title_val}.pdf"
    encoded = urllib.parse.quote(unicodedata.normalize("NFC", name))
    return f"{BASE_DOCS_URL.rstrip('/')}/{encoded}"

def limit_results_per_title(results: list, max_per_title: int = 2):
    limited, counter = [], defaultdict(int)
    for item in results:
        t = item.get("title", "Unknown")
        if counter[t] < max_per_title:
            limited.append(item)
            counter[t] += 1
    return limited

def rerank_results(query: str, results: list, top_n: int = 5):
    if not results or not RERANK_ENABLED or reranker is None: return results[:top_n]
    try:
        pairs = [(query, f"{r['title']}\n{r['content']}") for r in results]
        with rerank_lock: scores = reranker.predict(pairs)
        for r, s in zip(results, scores): r["rerank_score"] = float(s)
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results[:top_n]
    except: return results[:top_n]

def rewrite_query_for_retrieval(original_query: str) -> str:
    if not QUERY_REWRITE_ENABLED: return original_query
    prompt = f"사용자 질문을 검색용 명사 위주 키워드로 요약하세요. 결과만 한 줄로 출력.\n질문: {original_query}"
    try:
        with llm_lock: rewritten = llm.invoke(prompt)
        # LLM 종류에 따라 응답 객체가 다를 수 있으므로 텍스트만 추출
        content = rewritten.content if hasattr(rewritten, 'content') else str(rewritten)
        return safe_single_line(content)
    except: return original_query

def get_internal_context(query: str):
    with embedding_lock:
        query_vec = embed_model.encode("Represent this sentence for searching relevant passages: " + query).tolist()
    
    k = extract_search_keywords(query)
    p = generate_phrase_candidates(query)
    
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        register_vector(conn)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        # 하이브리드 검색 SQL (원본 로직 유지)
        sql = f"""
            WITH vector_matches AS (SELECT id, (embedding <=> %s::vector) AS dist FROM {TABLE_NAME} ORDER BY dist ASC LIMIT 50),
            phrase_matches AS (SELECT id, (CASE WHEN %s IS NOT NULL AND title ILIKE '%%'||%s||'%%' THEN 2.0 ELSE 0 END) AS phrase_score FROM {TABLE_NAME} LIMIT 100),
            keyword_matches AS (SELECT id, ts_rank_cd(content_search_vector, websearch_to_tsquery('simple', %s)) AS rank FROM {TABLE_NAME} LIMIT 100)
            SELECT t.*, COALESCE(1.0/(60+v.dist*100),0) + COALESCE(k.rank*10,0) + COALESCE(p.phrase_score,0) AS combined_score
            FROM {TABLE_NAME} t LEFT JOIN vector_matches v ON t.id=v.id LEFT JOIN keyword_matches k ON t.id=k.id LEFT JOIN phrase_matches p ON t.id=p.id
            WHERE v.id IS NOT NULL OR k.id IS NOT NULL OR p.id IS NOT NULL ORDER BY combined_score DESC LIMIT {RERANK_CANDIDATE_LIMIT}
        """
        cursor.execute(sql, (query_vec, p['original'], p['original'], query))
        rows = cursor.fetchall()
        results = [{"content": r["original_content"], "title": r["title"] or "Unknown", "url": build_document_url(r["title"]), "combined_score": float(r["combined_score"])} for r in rows]
        results = limit_results_per_title(results, max_per_title=MAX_CHUNKS_PER_TITLE)
        return rerank_results(query, results, top_n=RERANK_TOP_N)
    except Exception as e: print(f"DB Error: {e}"); return []
    finally: 
        if conn: conn.close()

def build_answer_prompt(original_query: str, effective_query: str, context: str) -> str:
    return (f"지식 데이터를 근거로 답변하세요. 모르면 모른다고 하세요.\n\n[질문]: {original_query}\n[데이터]:\n{context}\n\n"
            f"응답은 반드시 [## 📢 조치 안내], [### 🛠️ 상세 절차], [### 💡 주의 사항] 순서로 작성하세요.")

# --- LangGraph Nodes ---
def rewrite_query_node(state: PipelineState): return {"effective_query": rewrite_query_for_retrieval(state["original_query"])}
def retrieve_context_node(state: PipelineState): return {"search_results": get_internal_context(state.get("effective_query") or state["original_query"])}
def prepare_prompt_node(state: PipelineState):
    results = state.get("search_results", [])
    context = "\n".join([f"제목: {r['title']}\n내용: {r['content']}" for r in results])
    links = {r["title"]: r["url"] for r in results if r["url"]}
    return {"context_combined": context, "prompt": build_answer_prompt(state["original_query"], state.get("effective_query", ""), context), "doc_links": links}

def create_rag_pipeline():
    graph = StateGraph(PipelineState)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("prepare_prompt", prepare_prompt_node)
    graph.add_edge(START, "rewrite_query"); graph.add_edge("rewrite_query", "retrieve_context"); graph.add_edge("retrieve_context", "prepare_prompt"); graph.add_edge("prepare_prompt", END)
    return graph.compile()

rag_pipeline = create_rag_pipeline()

# --- HTTP Server ---
class RAGHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code, payload):
        self.send_response(status_code); self.send_header("Content-type", "application/json; charset=utf-8"); self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _send_sse(self, data):
        self.wfile.write(f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")); self.wfile.flush()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/files/"):
            full_path = os.path.join(STORAGE_DIR, unquote(parsed.path[7:]))
            if os.path.exists(full_path):
                self.send_response(200); self.send_header("Content-type", mimetypes.guess_type(full_path)[0] or "application/octet-stream"); self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
                with open(full_path, "rb") as f: self.wfile.write(f.read())
            else: self.send_error(404)
            return

        if parsed.path == "/search":
            query = parse_qs(parsed.query).get("query", [""])[0].strip()
            self.send_response(200); self.send_header("Content-type", "text/event-stream; charset=utf-8"); self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
            try:
                self._send_sse({"status": "🧠🔗⚙️ 질문 요약 중..."})
                state = {"original_query": query}
                for update in rag_pipeline.stream(state, stream_mode="updates"):
                    for node, out in update.items():
                        state.update(out)
                        if node == "retrieve_context": self._send_sse({"status": f"🔍 검색 완료 ({len(state['search_results'])}건)"})
                
                self._send_sse({"status": "🤖✨ 답변 생성 중..."})
                # 🚀 스트리밍 시작 (llm_chain 사용으로 모델 독립적 처리)
                for chunk in llm_chain.stream(state["prompt"]):
                    self._send_sse({"status": "💬✍️ 답변 중..."})
                    if chunk: self._send_sse({"chunk": chunk})
                
                if state.get("doc_links"):
                    links = "\n".join([f"- [{t}]({u})" for t, u in state["doc_links"].items()])
                    self._send_sse({"chunk": f"\n\n---\n### 🔗 관련 문서\n{links}"})
                self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
            except Exception as e: self._send_sse({"error": str(e)})

    def do_POST(self):
        if self.path == "/feedback":
            try:
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._save_feedback(data["query"], data["answer"])
                self._send_json(200, {"status": "success"})
            except Exception as e: self._send_json(500, {"error": str(e)})

    def _save_feedback(self, q, a):
        text = f"질문: {q}\n답변: {a}"
        vec = embed_model.encode(text).tolist()
        conn = psycopg2.connect(**DB_CONFIG)
        try:
            register_vector(conn); cur = conn.cursor()
            cur.execute(f"INSERT INTO {TABLE_NAME} (content_type, data_source, original_content, embedding, title) VALUES (%s, %s, %s, %s, %s)", ("text", "feedback", text, vec, "검증된 답변"))
            conn.commit()
        finally: conn.close()

if __name__ == "__main__":
    print(f"📡 {LLM_PROVIDER.upper()} 기반 RAG 가동: http://localhost:8000")
    ThreadingHTTPServer(("0.0.0.0", 8000), RAGHandler).serve_forever()