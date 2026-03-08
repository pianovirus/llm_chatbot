# pipeline.py
# =========================================================
# RAG Pipeline 모듈 (LangGraph)
# - PipelineState / 노드 함수 / create_rag_pipeline() 포함
# - main.py에서 필요한 의존성(LLM, locks, retrieval 함수, config 값)을 "주입"받아 조립
# - ⚠️ 프롬프트는 수정하지 않음 (형님 원본 그대로)
# =========================================================

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END


class PipelineState(TypedDict, total=False):
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
    (역할) LangGraph 파이프라인을 생성/컴파일하여 반환.

    주입받는 것:
    - llm, llm_lock: query rewrite 시 LLM 호출에 사용
    - query_rewrite_enabled: rewrite on/off
    - get_internal_context_fn: retrieval.get_internal_context
    - retrieval_kwargs_builder: main에 있는 환경값/모델/락을 모아 kwargs dict를 만드는 함수
    """

    # (역할) 검색용 query로 재작성 (옵션)
    def rewrite_query_for_retrieval(original_query: str) -> str:
        if not query_rewrite_enabled:
            return original_query
        prompt = f"사용자 질문을 검색용 명사 위주 키워드로 요약하세요. 결과만 한 줄로 출력.\n질문: {original_query}"
        try:
            with llm_lock:
                rewritten = llm.invoke(prompt)
            content = rewritten.content if hasattr(rewritten, "content") else str(rewritten)
            return " ".join(content.strip().split()) if content else original_query
        except:
            return original_query

    # ⚠️ 프롬프트 수정 금지 유지 (형님 원본 그대로)
    def build_answer_prompt(original_query: str, effective_query: str, context: str) -> str:
        return (
            f"지식 데이터를 근거로 답변하세요. 모르면 모른다고 하세요.\n\n"
            f"[질문]: {original_query}\n"
            f"[데이터]:\n{context}\n\n"
            f"응답은 반드시 [## 📢 조치 안내], [### 🛠️ 상세 절차], [### 💡 주의 사항] 순서로 작성하세요."
        )

    # (역할) Node1: rewrite
    def rewrite_query_node(state: PipelineState):
        return {"effective_query": rewrite_query_for_retrieval(state["original_query"])}

    # (역할) Node2: retrieve
    def retrieve_context_node(state: PipelineState):
        q = state.get("effective_query") or state["original_query"]
        kwargs = retrieval_kwargs_builder()
        results = get_internal_context_fn(q, **kwargs)
        return {"search_results": results}

    # (역할) Node3: prepare prompt/links
    def prepare_prompt_node(state: PipelineState):
        results = state.get("search_results", [])
        context = "\n".join([f"제목: {r['title']}\n내용: {r['content']}" for r in results])
        links = {r["title"]: r["url"] for r in results if r.get("url")}
        return {
            "context_combined": context,
            "prompt": build_answer_prompt(state["original_query"], state.get("effective_query", ""), context),
            "doc_links": links,
        }

    # (역할) 그래프 조립
    graph = StateGraph(PipelineState)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("prepare_prompt", prepare_prompt_node)

    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve_context")
    graph.add_edge("retrieve_context", "prepare_prompt")
    graph.add_edge("prepare_prompt", END)

    return graph.compile()