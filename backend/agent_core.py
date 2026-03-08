# main.py
import os
import ssl
import threading
import warnings
from http.server import ThreadingHTTPServer

from config import *
from pipeline import create_rag_pipeline
from retrieval import get_internal_context
from server import make_handler

# 🚨 Mac 환경에서 Segmentation Fault 방지
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 로그 노이즈 줄이기
warnings.filterwarnings("ignore")

# SSL 설정 (인증서 관련 에러 방지)
# - 일부 환경에서 HTTPS 요청 시 인증서 에러가 나는 걸 완화
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# ✅ main.py는 "조립자": 모델 초기화 + 파이프라인 조립 + 서버 실행만 담당
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer, CrossEncoder

# storage 폴더 준비 (server.py의 /files 라우팅에서 사용)
STORAGE_DIR = os.path.join(os.getcwd(), "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

# locks (ThreadingHTTPServer 동시 요청 대비)
embedding_lock = threading.Lock()
rerank_lock = threading.Lock()
llm_lock = threading.Lock()


def get_llm():
    """설정된 프로바이더에 따라 LLM 인스턴스를 생성합니다."""
    temp = 0
    if LLM_PROVIDER == "google":
        print(f"✨ [System] Google Gemini 엔진 사용 중... ({MODEL_NAME})")
        return ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            version="v1",
            temperature=temp,
        )
    elif LLM_PROVIDER == "openai":
        print(f"✨ [System] OpenAI 엔진 사용 중... ({MODEL_NAME})")
        return ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=temp)
    else:
        print(f"✨ [System] Ollama 엔진 사용 중... ({MODEL_NAME})")
        return Ollama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=temp, timeout=180)


# =========================
# 모델 초기화 (1회)
# =========================
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


def build_retrieval_kwargs():
    """retrieval.get_internal_context에 넘길 kwargs를 매번 같은 형태로 구성."""
    return dict(
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


# ✅ pipeline 조립 (LangGraph) — pipeline.py 내부에 프롬프트 포함(수정 금지 유지)
rag_pipeline = create_rag_pipeline(
    llm=llm,
    llm_lock=llm_lock,
    query_rewrite_enabled=QUERY_REWRITE_ENABLED,
    get_internal_context_fn=get_internal_context,
    retrieval_kwargs_builder=build_retrieval_kwargs,
)


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