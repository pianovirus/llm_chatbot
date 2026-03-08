# main.py
import os
# 🚨 Mac 환경에서 Segmentation Fault 방지
# - sentence-transformers / tokenizers 계열이 Mac에서 병렬 실행 시 간헐적으로 크래시 나는 경우가 있어 비활성화
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================================================
# 0) 기본 라이브러리 import
# =========================================================
import json
import warnings
import ssl
import threading
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote
from typing import TypedDict, List, Dict, Any

# =========================================================
# 1) 환경설정(config.py)
# =========================================================
from config import *

# =========================================================
# 1-1) text_utils 분리 모듈 import
# =========================================================
from text_utils import safe_single_line

# =========================================================
# 1-2) retrieval 분리 모듈 import (✅ 이번 작업 핵심)
# =========================================================
from retrieval import get_internal_context

warnings.filterwarnings("ignore")

# =========================================================
# 2) SSL 설정 (인증서 관련 에러 방지)
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
# =========================================================
STORAGE_DIR = os.path.join(os.getcwd(), "storage")
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

# =========================================================
# 5) 멀티스레드 안전장치
# =========================================================
embedding_lock = threading.Lock()
rerank_lock = threading.Lock()
llm_lock = threading.Lock()

# =========================================================
# 6) LLM Factory
# =========================================================
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
# 7) 모델 초기화
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
llm_chain = llm | StrOutputParser()

print("✅ [System] PostgreSQL RAG 엔진 가동 준비 완료.")
print("=" * 50)
print(f"현재 선택된 공급자: {LLM_PROVIDER}")
print(f"현재 선택된 모델명: {MODEL_NAME}")
print(f"실제 생성된 객체 타입: {type(llm)}")
print("=" * 50)

# =========================================================
# 8) LangGraph Pipeline State 정의
# =========================================================
class PipelineState(TypedDict, total=False):
    original_query: str
    effective_query: str
    search_results: List[Dict[str, Any]]
    context_combined: str
    prompt: str
    doc_links: Dict[str, str]

# =========================================================
# 10) Query Rewrite (옵션)
# =========================================================
def rewrite_query_for_retrieval(original_query: str) -> str:
    if not QUERY_REWRITE_ENABLED:
        return original_query
    prompt = f"사용자 질문을 검색용 명사 위주 키워드로 요약하세요. 결과만 한 줄로 출력.\n질문: {original_query}"
    try:
        with llm_lock:
            rewritten = llm.invoke(prompt)
        content = rewritten.content if hasattr(rewritten, 'content') else str(rewritten)
        return safe_single_line(content)
    except:
        return original_query

# =========================================================
# 12) Prompt Builder (⚠️ 프롬프트 수정 금지 유지)
# =========================================================
def build_answer_prompt(original_query: str, effective_query: str, context: str) -> str:
    return (
        f"지식 데이터를 근거로 답변하세요. 모르면 모른다고 하세요.\n\n"
        f"[질문]: {original_query}\n"
        f"[데이터]:\n{context}\n\n"
        f"응답은 반드시 [## 📢 조치 안내], [### 🛠️ 상세 절차], [### 💡 주의 사항] 순서로 작성하세요."
    )

# =========================================================
# 13) LangGraph Nodes
# =========================================================
def rewrite_query_node(state: PipelineState):
    return {"effective_query": rewrite_query_for_retrieval(state["original_query"])}

def retrieve_context_node(state: PipelineState):
    q = state.get("effective_query") or state["original_query"]
    results = get_internal_context(
        q,
        embed_model=embed_model,
        embedding_lock=embedding_lock,
        db_config=DB_CONFIG,
        table_name=TABLE_NAME,
        base_docs_url=BASE_DOCS_URL,
        max_chunks_per_title=MAX_CHUNKS_PER_TITLE,
        rerank_candidate_limit=RERANK_CANDIDATE_LIMIT,
        rerank_top_n=RERANK_TOP_N,
        rerank_enabled=RERANK_ENABLED,
        reranker=reranker,
        rerank_lock=rerank_lock,
    )
    return {"search_results": results}

def prepare_prompt_node(state: PipelineState):
    results = state.get("search_results", [])
    context = "\n".join([f"제목: {r['title']}\n내용: {r['content']}" for r in results])
    links = {r["title"]: r["url"] for r in results if r.get("url")}
    return {
        "context_combined": context,
        "prompt": build_answer_prompt(state["original_query"], state.get("effective_query", ""), context),
        "doc_links": links
    }

def create_rag_pipeline():
    graph = StateGraph(PipelineState)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("prepare_prompt", prepare_prompt_node)

    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve_context")
    graph.add_edge("retrieve_context", "prepare_prompt")
    graph.add_edge("prepare_prompt", END)

    return graph.compile()

rag_pipeline = create_rag_pipeline()

# =========================================================
# 14) HTTP Server
# =========================================================
class RAGHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code, payload):
        self.send_response(status_code)
        self.send_header("Content-type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _send_sse(self, data):
        self.wfile.write(f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def do_GET(self):
        parsed = urlparse(self.path)

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

        if parsed.path == "/search":
            query = parse_qs(parsed.query).get("query", [""])[0].strip()

            self.send_response(200)
            self.send_header("Content-type", "text/event-stream; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                self._send_sse({"status": "🧠🔗⚙️ 질문 요약 중..."})
                state = {"original_query": query}

                for update in rag_pipeline.stream(state, stream_mode="updates"):
                    self._send_sse({"status": "🔍 검색 중..."})
                    for node, out in update.items():
                        state.update(out)
                        if node == "retrieve_context":
                            self._send_sse({"status": f"🔍 검색 완료 ({len(state['search_results'])}건)"})

                self._send_sse({"status": "🤖✨ 답변 생성 중..."})
                for chunk in llm_chain.stream(state["prompt"]):
                    self._send_sse({"status": "💬✍️ 답변 중..."})
                    if chunk:
                        self._send_sse({"chunk": chunk})

                if state.get("doc_links"):
                    links = "\n".join([f"- [{t}]({u})" for t, u in state["doc_links"].items()])
                    self._send_sse({"chunk": f"\n\n---\n### 🔗 관련 문서\n{links}"})

                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

            except Exception as e:
                self._send_sse({"error": str(e)})

    def do_POST(self):
        if self.path == "/feedback":
            try:
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._save_feedback(data["query"], data["answer"])
                self._send_json(200, {"status": "success"})
            except Exception as e:
                self._send_json(500, {"error": str(e)})

    def _save_feedback(self, q, a):
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

if __name__ == "__main__":
    print(f"📡 {LLM_PROVIDER.upper()} 기반 RAG 가동: http://localhost:8000")
    ThreadingHTTPServer(("0.0.0.0", 8000), RAGHandler).serve_forever()