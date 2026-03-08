# agent_core.py (main entry)
# ============================================================
# 이 파일은 "프로젝트의 부팅/조립(bootstrapper)" 역할을 합니다.
#
# ✅ 하는 일
#  1) 환경설정(config.py) 로드
#  2) SSL/경고 필터 등 런타임 환경 안전장치 설정
#  3) 모델(Embedding / Reranker / LLM) 1회 초기화
#  4) Retrieval(DB 검색) 파라미터 주입 빌더 구성
#  5) LangGraph 기반 RAG 파이프라인(pipeline.py) 조립
#  6) HTTP 서버(server.py) Handler를 생성하고 서버 실행
#
# ❌ 여기서 하지 않는 일
#  - DB 검색 SQL 작성/실행: retrieval.py
#  - 텍스트 전처리: text_utils.py
#  - 파이프라인 노드/프롬프트 조립: pipeline.py
#  - HTTP 라우팅(/files, /search, /feedback): server.py
#
# 목표: agent_core.py는 "어디서 무엇이 로드되고 조립되는지" 한눈에 보이게 유지
# ============================================================

# ------------------------------------------------------------
# 0) 표준 라이브러리: OS/SSL/스레딩/서버 실행
# ------------------------------------------------------------
import os
import ssl
import threading
import warnings
from http.server import ThreadingHTTPServer

# ------------------------------------------------------------
# 1) 프로젝트 모듈 import
# ------------------------------------------------------------
# (config.py) .env 기반 설정값(모델명, DB 설정, 기능 토글 등)을 제공하는 모듈
#  - LLM_PROVIDER, MODEL_NAME, EMBED_MODEL_NAME, DB_CONFIG, TABLE_NAME 등
from config import *

# (retrieval.py) pgvector(PostgreSQL) 기반 하이브리드 검색 함수
#  - 실제 DB 연결/쿼리/스코어링은 retrieval.py로 격리되어 있음
from retrieval import get_internal_context

# (pipeline.py) LangGraph 기반 RAG 파이프라인 생성기
#  - rewrite -> retrieve -> prepare 단계 구성 + 프롬프트 조립(프롬프트 원문 유지)
from pipeline import create_rag_pipeline

# (server.py) HTTP 요청 라우팅(/files, /search, /feedback)을 담당하는 Handler factory
#  - make_handler(...) 로 main에서 의존성을 주입해 Handler 클래스를 생성
from server import make_handler

# ------------------------------------------------------------
# 2) 런타임 안전장치: Mac tokenizers 병렬/경고/SSL 설정
# ------------------------------------------------------------
# 🚨 Mac 환경에서 Segmentation Fault 방지
# - sentence-transformers / tokenizers 계열이 Mac에서 병렬 실행 시 간헐적으로 크래시 나는 경우가 있어 비활성화
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 로그 노이즈 줄이기
# - 지금 단계에서는 안정적인 부팅/모듈화가 목표라 경고를 숨김
warnings.filterwarnings("ignore")

# SSL 설정 (인증서 관련 에러 방지)
# - 일부 환경에서 HTTPS 요청(특히 외부 API 호출) 시 인증서 문제로 실패하는 케이스 완화
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Python/플랫폼에 따라 _create_unverified_context가 없을 수 있음
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# ------------------------------------------------------------
# 3) AI/ML 라이브러리 import (모델 초기화에 필요)
# ------------------------------------------------------------
# LLM Provider별 LangChain 래퍼
# - Ollama: 로컬 모델 서버
# - Google: Gemini
# - OpenAI: ChatGPT 계열
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# LLM 출력 타입을 "문자열"로 통일해주기 위한 파서
from langchain_core.output_parsers import StrOutputParser

# 임베딩/리랭커 모델
# - SentenceTransformer: query/document 임베딩 생성 (RAG 검색 핵심)
# - CrossEncoder: (옵션) 검색 결과 rerank
from sentence_transformers import SentenceTransformer, CrossEncoder

# ------------------------------------------------------------
# 4) 로컬 파일 서빙 디렉토리 준비 (/files 라우팅에서 사용)
# ------------------------------------------------------------
# server.py의 /files/<filename> 요청이 들어오면 이 폴더에서 파일을 읽어 반환
STORAGE_DIR = os.path.join(os.getcwd(), "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

# ------------------------------------------------------------
# 5) 멀티스레드 락 (ThreadingHTTPServer 동시 요청 대비)
# ------------------------------------------------------------
# - embedding_lock: 임베딩 모델 호출 동시성 보호(안정성)
# - rerank_lock: reranker 호출 동시성 보호(안정성)
# - llm_lock: query rewrite LLM 호출 동시성 보호(안정성)
embedding_lock = threading.Lock()
rerank_lock = threading.Lock()
llm_lock = threading.Lock()

# ------------------------------------------------------------
# 6) LLM Factory: 환경설정(LLM_PROVIDER)에 따라 LLM 선택/생성
# ------------------------------------------------------------
def get_llm():
    """설정된 프로바이더(google/openai/ollama)에 따라 LLM 인스턴스를 생성."""
    temp = 0

    if LLM_PROVIDER == "google":
        print(f"✨ [System] Google Gemini 엔진 사용 중... ({MODEL_NAME})")
        return ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            version="v1",
            temperature=temp,
        )

    if LLM_PROVIDER == "openai":
        print(f"✨ [System] OpenAI 엔진 사용 중... ({MODEL_NAME})")
        return ChatOpenAI(
            model=MODEL_NAME,
            api_key=OPENAI_API_KEY,
            temperature=temp,
        )

    # default: ollama
    print(f"✨ [System] Ollama 엔진 사용 중... ({MODEL_NAME})")
    return Ollama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=temp,
        timeout=180,
    )

# ------------------------------------------------------------
# 7) 모델 초기화 (프로세스 시작 시 1회)
# ------------------------------------------------------------
def init_models():
    """
    (역할) 모델을 1회 초기화하고 필요한 객체들을 반환.
    - Embedding / (옵션) Reranker / LLM / LLM Chain
    """
    # (1) Embedding 모델 로드: 문서/질문을 벡터로 바꾸는 역할
    print(f"⏳ [System] 임베딩 모델 로드 중... ({EMBED_MODEL_NAME})")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # (2) Reranker 모델 로드(옵션): 검색 결과를 더 좋은 순서로 재정렬
    reranker = None
    if RERANK_ENABLED:
        try:
            print(f"⏳ [System] Reranker 모델 로드 중... ({RERANK_MODEL_NAME})")
            reranker = CrossEncoder(RERANK_MODEL_NAME)
        except Exception as e:
            # reranker는 옵션이라 실패해도 서버는 동작할 수 있게 함
            print(f"❌ Reranker 로드 실패: {e}")
            reranker = None

    # (3) LLM 로드 + 문자열 파서 체인 구성
    llm = get_llm()
    llm_chain = llm | StrOutputParser()

    # (4) 부팅 상태 출력
    print("✅ [System] PostgreSQL RAG 엔진 가동 준비 완료.")
    print("=" * 50)
    print(f"현재 선택된 공급자: {LLM_PROVIDER}")
    print(f"현재 선택된 모델명: {MODEL_NAME}")
    print(f"실제 생성된 객체 타입: {type(llm)}")
    print("=" * 50)

    return embed_model, reranker, llm, llm_chain

# ------------------------------------------------------------
# 8) agent_core 실행 진입점
# ------------------------------------------------------------
def main():
    """
    (역할) agent_core 실행 진입 함수
    - 테스트/재사용/가독성을 위해 부팅 로직을 main()으로 감쌈
    - 동작은 기존과 동일: 파이프라인 조립 후 HTTP 서버 실행
    """
    # 1) 모델 초기화
    embed_model, reranker, llm, llm_chain = init_models()

    # 2) Retrieval 주입 파라미터 빌더
    def build_retrieval_kwargs():
        return dict(
            # 임베딩 모델/락
            embed_model=embed_model,
            embedding_lock=embedding_lock,

            # DB 연결/테이블/문서 URL
            db_config=DB_CONFIG,
            table_name=TABLE_NAME,
            base_docs_url=BASE_DOCS_URL,

            # 검색 결과 조절(제한/리랭크)
            max_chunks_per_title=MAX_CHUNKS_PER_TITLE,
            rerank_candidate_limit=RERANK_CANDIDATE_LIMIT,
            rerank_top_n=RERANK_TOP_N,

            # rerank 옵션 + 모델/락
            rerank_enabled=RERANK_ENABLED,
            reranker=reranker,
            rerank_lock=rerank_lock,
        )

    # 3) RAG Pipeline 조립 (LangGraph)
    rag_pipeline = create_rag_pipeline(
        llm=llm,
        llm_lock=llm_lock,
        query_rewrite_enabled=QUERY_REWRITE_ENABLED,
        get_internal_context_fn=get_internal_context,
        retrieval_kwargs_builder=build_retrieval_kwargs,
    )

    # 4) 서버 실행
    print(f"📡 {LLM_PROVIDER.upper()} 기반 RAG 가동: http://localhost:8000")

    Handler = make_handler(
        storage_dir=STORAGE_DIR,
        rag_pipeline=rag_pipeline,
        llm_chain=llm_chain,

        # feedback 저장에 사용
        embed_model=embed_model,
        db_config=DB_CONFIG,
        table_name=TABLE_NAME,

        # 현재는 사용하지 않지만 추후 provider별 분기/디버그 등에 활용 가능
        llm_provider=LLM_PROVIDER,
    )

    ThreadingHTTPServer(("0.0.0.0", 8000), Handler).serve_forever()

# ------------------------------------------------------------
# 9) 실제 실행
# ------------------------------------------------------------
if __name__ == "__main__":
    main()