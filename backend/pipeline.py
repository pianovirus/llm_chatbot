# pipeline.py
# =========================================================
# RAG Pipeline 모듈 (LangGraph)
# - PipelineState / 노드 함수 / create_rag_pipeline() 포함
# - main.py에서 필요한 의존성(LLM, locks, retrieval 함수, config 값)을 "주입"받아 조립
# - 성능 개선: prompt에 들어가는 context 양 제한
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

    # -----------------------------
    # 성능 튜닝 파라미터
    # -----------------------------
    MAX_RESULTS_IN_PROMPT = 2
    MAX_CONTENT_CHARS_PER_RESULT = 500

    def _clean_text(text: str) -> str:
        if not text:
            return ""
        return " ".join(str(text).strip().split())

    def _trim_text(text: str, limit: int) -> str:
        text = _clean_text(text)
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "…"

    # (역할) 검색용 query로 재작성 (옵션)
    def rewrite_query_for_retrieval(original_query: str) -> str:
        if not query_rewrite_enabled:
            return original_query

        prompt = (
            f"사용자 질문을 검색용 명사 위주 키워드로 요약하세요. "
            f"결과만 한 줄로 출력.\n질문: {original_query}"
        )
        try:
            with llm_lock:
                rewritten = llm.invoke(prompt)
            content = rewritten.content if hasattr(rewritten, "content") else str(rewritten)
            return " ".join(content.strip().split()) if content else original_query
        except Exception:
            return original_query

    # 출력 형식은 유지하되, 지시문을 약간 간결화
    def build_answer_prompt(original_query: str, effective_query: str, context: str) -> str:
        return (
            f"지식 데이터를 근거로 답변하세요. "
            f"데이터에 없는 내용은 추측하지 말고 모른다고 하세요.\n\n"
            f"[질문]: {original_query}\n"
            f"[데이터]:\n{context}\n\n"
            f"응답은 반드시 [## 📢 조치 안내], [### 🛠️ 상세 절차], [### 💡 주의 사항] 순서로 작성하세요. "
            f"불필요하게 길게 쓰지 말고 간결하게 작성하세요."
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

        # 1) 제목 기준 중복 제거
        deduped = []
        seen_titles = set()
        for r in results:
            title = (r.get("title") or "").strip()
            key = title or id(r)
            if key in seen_titles:
                continue
            seen_titles.add(key)
            deduped.append(r)

        # 2) prompt에는 상위 N개만 사용
        selected = deduped[:MAX_RESULTS_IN_PROMPT]

        # 3) 각 content는 길이 제한
        context_parts = []
        for r in selected:
            title = _clean_text(r.get("title", "제목 없음"))
            content = _trim_text(r.get("content", ""), MAX_CONTENT_CHARS_PER_RESULT)
            context_parts.append(f"제목: {title}\n내용: {content}")

        context = "\n\n".join(context_parts)

        # 링크는 전체 results가 아니라 selected 기준으로만 구성
        links = {
            r["title"]: r["url"]
            for r in selected
            if r.get("title") and r.get("url")
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