# pipeline.py
# ============================================================
# RAG Pipeline 모듈 (LangGraph)
# ------------------------------------------------------------
# 이 파일은 "질문 -> 검색어 재작성 -> 문맥 검색 -> 프롬프트 조립"까지의
# RAG 전처리 파이프라인을 정의합니다.
#
# 포함 내용:
# 1) PipelineState : 그래프 상태 스키마
# 2) 노드 함수들     : rewrite / retrieve / prepare
# 3) create_rag_pipeline() : 그래프 조립 및 compile
#
# 설계 의도:
# - 실제 서버 실행(agent_core.py)과 파이프라인 로직을 분리
# - 필요한 의존성(LLM, lock, retrieval 함수, 설정값)은 바깥에서 주입
# - 테스트/교체/확장이 쉬운 구조 유지
#
# 주의:
# - 형님 원본 프롬프트는 수정하지 않는 원칙 유지
# ============================================================

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END


class PipelineState(TypedDict, total=False):
    """
    ========================================================
    LangGraph 상태 객체 스키마
    --------------------------------------------------------
    파이프라인이 진행되면서 노드 간에 주고받는 상태(state)의 구조를 정의합니다.

    각 필드 설명:
    - original_query   : 사용자가 처음 입력한 원본 질문
    - effective_query  : 검색 친화적으로 재작성된 질의
    - search_results   : retrieval 단계에서 가져온 검색 결과 목록
    - context_combined : 검색 결과들을 하나의 긴 컨텍스트 문자열로 합친 값
    - prompt           : 최종 LLM 답변 생성용 프롬프트
    - doc_links        : 관련 문서 제목 -> URL 매핑

    total=False 의미:
    - 모든 필드가 처음부터 반드시 존재할 필요는 없음
    - 각 노드가 필요한 값을 점진적으로 추가하는 방식으로 사용 가능

    예시 흐름:
    1) 시작 시:
       {"original_query": "에러 코드 500 해결 방법 알려줘"}

    2) rewrite 후:
       {
         "original_query": "...",
         "effective_query": "500 에러 해결 방법"
       }

    3) retrieve 후:
       {
         "original_query": "...",
         "effective_query": "...",
         "search_results": [...]
       }

    4) prepare 후:
       {
         "original_query": "...",
         "effective_query": "...",
         "search_results": [...],
         "context_combined": "...",
         "prompt": "...",
         "doc_links": {...}
       }
    ========================================================
    """
    original_query: str
    effective_query: str
    search_results: List[Dict[str, Any]]
    context_combined: str
    prompt: str
    doc_links: Dict[str, str]


def create_rag_pipeline(
    *,
    llm,
    llm_lock,
    query_rewrite_enabled: bool,
    get_internal_context_fn,
    retrieval_kwargs_builder,
):
    """
    ========================================================
    RAG 파이프라인 생성 함수
    --------------------------------------------------------
    LangGraph 기반 파이프라인을 조립하고 compile 해서 반환합니다.

    이 함수는 "파이프라인 정의 + 의존성 주입"의 중심입니다.

    주입받는 인자:
    - llm:
      query rewrite에 사용할 LLM 객체
      예: Gemini, OpenAI, Ollama 등

    - llm_lock:
      LLM 동시 호출 안정성을 위한 락
      멀티스레드 환경에서 query rewrite 호출 충돌을 줄이기 위함

    - query_rewrite_enabled:
      검색어 재작성 기능 on/off 플래그
      False면 원본 질문을 그대로 검색에 사용

    - get_internal_context_fn:
      실제 검색 함수
      보통 retrieval.py 의 get_internal_context 함수가 주입됨

    - retrieval_kwargs_builder:
      검색 함수에 필요한 각종 인자(embed_model, DB 설정, reranker 등)를
      한 번에 dict 형태로 만들어 주는 함수

    반환:
    - compile 된 LangGraph runnable 객체

    파이프라인 흐름:
    START
      -> rewrite_query
      -> retrieve_context
      -> prepare_prompt
      -> END
    ========================================================
    """

    def rewrite_query_for_retrieval(original_query: str) -> str:
        """
        ====================================================
        검색용 질의 재작성 함수
        ----------------------------------------------------
        사용자의 자연어 질문을 "검색에 더 적합한 키워드형 질의"로 바꿉니다.

        목적:
        - 사용자가 길게 말한 문장을
          검색 친화적인 명사/핵심 키워드 형태로 축약
        - 내부 문서 검색 적중률 향상

        예:
        입력:
          "잡 모니터에서 failed 상태 로그를 어디서 확인하나요?"
        출력:
          "잡 모니터 failed 상태 로그 확인"

        동작 방식:
        - query_rewrite_enabled=False 이면 원본 그대로 반환
        - True 이면 LLM을 사용해 한 줄 키워드 질의로 변환

        예외 처리:
        - rewrite 중 LLM 호출 실패 시 원본 질문으로 fallback

        왜 llm_lock을 쓰나?
        - 멀티스레드 환경에서 LLM 호출 중 간헐적 충돌 방지
        ====================================================
        """
        if not query_rewrite_enabled:
            return original_query

        prompt = (
            f"사용자 질문을 검색용 명사 위주 키워드로 요약하세요. 결과만 한 줄로 출력.\n"
            f"질문: {original_query}"
        )

        try:
            with llm_lock:
                rewritten = llm.invoke(prompt)

            # LangChain 계열 객체는 보통 .content 속성을 가짐
            # 없으면 그냥 문자열로 변환
            content = rewritten.content if hasattr(rewritten, "content") else str(rewritten)

            # 불필요한 줄바꿈/다중 공백 제거 후 1줄로 정리
            return " ".join(content.strip().split()) if content else original_query

        except Exception:
            return original_query

    def build_answer_prompt(original_query: str, effective_query: str, context: str) -> str:
        """
        ====================================================
        최종 답변 프롬프트 생성 함수
        ----------------------------------------------------
        검색 결과(context)를 기반으로 LLM이 답변할 때 사용할 프롬프트를 만듭니다.

        입력:
        - original_query  : 사용자가 실제로 물은 원본 질문
        - effective_query : 검색에 사용된 재작성 질의
        - context         : 검색 결과를 합쳐 만든 문맥 문자열

        출력:
        - LLM에 넣을 최종 프롬프트 문자열

        주의:
        - 형님 원본 프롬프트는 수정하지 않는 원칙 유지
        - effective_query 인자는 현재 프롬프트 본문에 직접 쓰이지 않지만,
          필요 시 디버깅/확장용으로 유지 가능

        목적:
        - "질문 + 근거 데이터 + 응답 형식 지시"를 하나의 텍스트로 묶기
        ====================================================
        """
        return (
            f"지식 데이터를 근거로 답변하세요. 모르면 모른다고 하세요.\n\n"
            f"[질문]: {original_query}\n"
            f"[데이터]:\n{context}\n\n"
            f"응답은 반드시 [## 📢 조치 안내], [### 🛠️ 상세 절차], [### 💡 주의 사항] 순서로 작성하세요."
        )

    def rewrite_query_node(state: PipelineState):
        """
        ====================================================
        Node 1: rewrite_query
        ----------------------------------------------------
        입력 상태(state)에서 original_query를 읽어
        검색용 effective_query를 생성하는 노드입니다.

        입력 예:
        {
          "original_query": "잡 로그 삭제 정책은 어디서 설정해?"
        }

        출력 예:
        {
          "effective_query": "잡 로그 삭제 정책 설정"
        }

        특징:
        - state 전체를 바꾸는 것이 아니라
          새로 추가/갱신할 필드만 dict로 반환
        - LangGraph가 반환값을 기존 state에 merge 함
        ====================================================
        """
        return {
            "effective_query": rewrite_query_for_retrieval(state["original_query"])
        }

    def retrieve_context_node(state: PipelineState):
        """
        ====================================================
        Node 2: retrieve_context
        ----------------------------------------------------
        실제 내부 문서 검색을 수행하는 노드입니다.

        검색 질의 선택 우선순위:
        1) effective_query가 있으면 그것 사용
        2) 없으면 original_query 사용

        처리 절차:
        1) 검색 질의(q) 결정
        2) retrieval_kwargs_builder() 호출
           - embed_model, DB 설정, reranker, lock 등 검색에 필요한 인자 구성
        3) get_internal_context_fn(...) 호출
           - 실제 검색 수행
        4) 검색 결과 리스트를 state에 저장

        출력:
        {
          "search_results": [...]
        }

        search_results 예시 구조:
        [
          {
            "title": "Job Monitor Guide",
            "content": "...",
            "url": "https://..."
          },
          ...
        ]
        ====================================================
        """
        q = state.get("effective_query") or state["original_query"]

        # 검색 함수에 넘길 런타임 의존성(dict) 생성
        kwargs = retrieval_kwargs_builder()

        # 실제 검색 수행
        results = get_internal_context_fn(q, **kwargs)

        return {"search_results": results}

    def prepare_prompt_node(state: PipelineState):
        """
        ====================================================
        Node 3: prepare_prompt
        ----------------------------------------------------
        검색 결과를 바탕으로 최종 답변 생성용 prompt와
        문서 링크 목록을 준비하는 노드입니다.

        처리 절차:
        1) search_results를 읽음
        2) 각 검색 결과를 '제목 + 내용' 형태로 이어붙여 context 생성
        3) URL이 있는 문서만 doc_links로 수집
        4) build_answer_prompt(...) 호출
        5) context_combined / prompt / doc_links 반환

        왜 필요한가?
        - 검색 결과는 리스트 구조이므로 그대로 LLM에 넣기보다
          사람이 읽는 텍스트처럼 합쳐서 prompt에 삽입해야 하기 때문

        출력 예:
        {
          "context_combined": "제목: ...\n내용: ...",
          "prompt": "지식 데이터를 근거로 답변하세요...",
          "doc_links": {
             "문서A": "https://...",
             "문서B": "https://..."
          }
        }
        ====================================================
        """
        results = state.get("search_results", [])

        # 검색 결과들을 LLM이 읽기 쉬운 하나의 긴 텍스트로 결합
        context = "\n".join(
            [f"제목: {result['title']}\n내용: {result['content']}" for result in results]
        )

        # 링크가 있는 결과만 추려 제목 -> URL 매핑 생성
        links = {
            result["title"]: result["url"]
            for result in results
            if result.get("url")
        }

        return {
            "context_combined": context,
            "prompt": build_answer_prompt(
                state["original_query"],
                state.get("effective_query", ""),
                context,
            ),
            "doc_links": links,
        }

    # ========================================================
    # 그래프 조립
    # --------------------------------------------------------
    # 여기서 LangGraph의 노드와 간선을 등록하여
    # 전체 파이프라인 흐름을 정의합니다.
    #
    # 흐름:
    # START
    #   -> rewrite_query
    #   -> retrieve_context
    #   -> prepare_prompt
    #   -> END
    # ========================================================
    graph = StateGraph(PipelineState)

    # 노드 등록
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("prepare_prompt", prepare_prompt_node)

    # 간선 등록 (실행 순서 정의)
    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve_context")
    graph.add_edge("retrieve_context", "prepare_prompt")
    graph.add_edge("prepare_prompt", END)

    # compile() 후 실제 실행 가능한 runnable 객체가 됨
    return graph.compile()