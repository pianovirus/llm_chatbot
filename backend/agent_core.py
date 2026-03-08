# main.py
import os
# 🚨 Mac 환경에서 Segmentation Fault 방지
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
import ssl
import threading
from http.server import ThreadingHTTPServer
from typing import TypedDict, List, Dict, Any

from config import *
from text_utils import safe_single_line
from retrieval import get_internal_context
from server import make_handler  # ✅ RAGHandler 분리

warnings.filterwarnings("ignore")

# SSL 설정
try:
    _create_unverified_https_context = ssl._create_unverified_https_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# ✅ main.py는 "조립자" 역할만: 모델/그래프 초기화 + 서버 실행에 필요한 최소 import만 유지
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer, CrossEncoder
from langgraph.graph import StateGraph, START, END

# storage 폴더 준비
STORAGE_DIR = os.path.join(os.getcwd(), "storage")
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

# locks
embedding_lock = threading.Lock()
rerank_lock = threading.Lock()
llm_lock = threading.Lock()

def get_llm():
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

class PipelineState(TypedDict, total=False):
    original_query: str
    effective_query: str
    search_results: List[Dict[str, Any]]
    context_combined: str
    prompt: str
    doc_links: Dict[str, str]

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

# ⚠️ 프롬프트 수정 금지 유지
def build_answer_prompt(original_query: str, effective_query: str, context: str) -> str:
    return (
        f"지식 데이터를 근거로 답변하세요. 모르면 모른다고 하세요.\n\n"
        f"[질문]: {original_query}\n"
        f"[데이터]:\n{context}\n\n"
        f"응답은 반드시 [## 📢 조치 안내], [### 🛠️ 상세 절차], [### 💡 주의 사항] 순서로 작성하세요."
    )

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

if __name__ == "__main__":
    print(f"📡 {LLM_PROVIDER.upper()} 기반 RAG 가동: http://localhost:8000")

    Handler = make_handler(
        storage_dir=STORAGE_DIR,
        rag_pipeline=rag_pipeline,
        llm_chain=llm_chain,
        embed_model=embed_model,
        db_config=DB_CONFIG,
        table_name=TABLE_NAME,
        llm_provider=LLM_PROVIDER,
    )

    ThreadingHTTPServer(("0.0.0.0", 8000), Handler).serve_forever()