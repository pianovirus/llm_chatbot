# retrieval.py
# ============================================================
# Retrieval(검색) 모듈
# ------------------------------------------------------------
# PostgreSQL(pgvector)에 저장된 문서 chunk들을 하이브리드 방식으로 검색
#
# 주요 기능
# 1) 질문(query)을 embedding 벡터로 변환
# 2) vector 검색 + keyword 검색 + phrase 검색 점수 결합
# 3) 같은 문서(title) 결과 편향 제한
# 4) (옵션) reranker 모델로 결과 재정렬
#
# 설계 목적
# - 검색 로직을 다른 모듈(agent_core.py, pipeline.py)와 분리
# - embed_model, reranker, locks 등을 외부에서 주입받아 사용
# ============================================================

import urllib.parse
import unicodedata
from collections import defaultdict
from typing import Any, Dict, List

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from text_utils import extract_search_keywords, generate_phrase_candidates


# ============================================================
# 문서 title -> 실제 파일 접근 URL 생성
# ============================================================
def build_document_url(title_val: str, base_docs_url: str) -> str:
    """
    DB에 저장된 title 값을 실제 파일 접근 URL로 변환.

    예
    title = "job_monitor"
    base_docs_url = "http://localhost:8000/files"

    결과
    http://localhost:8000/files/job_monitor.pdf
    """

    name = title_val if title_val.lower().endswith(".pdf") else f"{title_val}.pdf"

    # 한글 파일명을 고려해 NFC 정규화 후 URL 인코딩
    encoded = urllib.parse.quote(unicodedata.normalize("NFC", name))

    return f"{base_docs_url.rstrip('/')}/{encoded}"


# ============================================================
# 같은 문서(title)에서 최대 chunk 수 제한
# ============================================================
def limit_results_per_title(results: list, max_per_title: int = 2):
    """
    하나의 문서에서 너무 많은 chunk가 상위 결과를 차지하지 않도록 제한.

    이유
    - 하나의 PDF가 여러 chunk로 저장되어 있으면
      검색 상위 결과가 같은 문서로 도배될 수 있음
    """

    limited = []
    counter = defaultdict(int)

    for item in results:
        title = item.get("title", "Unknown")

        if counter[title] < max_per_title:
            limited.append(item)
            counter[title] += 1

    return limited


# ============================================================
# (옵션) reranker로 결과 재정렬
# ============================================================
def rerank_results(
    query: str,
    results: list,
    top_n: int,
    rerank_enabled: bool,
    reranker,
    rerank_lock,
):
    """
    CrossEncoder 기반 reranker로 검색 결과를 재정렬.

    rerank_enabled=False 또는 모델이 없으면
    기존 결과 그대로 top_n 반환
    """

    if not results or not rerank_enabled or reranker is None:
        return results[:top_n]

    try:
        # query와 문서를 pair로 만들어 reranker 입력 생성
        pairs = [(query, f"{r['title']}\n{r['content']}") for r in results]

        with rerank_lock:
            scores = reranker.predict(pairs)

        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)

        # 점수 기준 내림차순 정렬
        results.sort(key=lambda x: x["rerank_score"], reverse=True)

        return results[:top_n]

    except Exception:
        return results[:top_n]


# ============================================================
# 메인 검색 함수
# ============================================================
def get_internal_context(
    query: str,
    *,
    embed_model,
    embedding_lock,
    db_config: Dict[str, Any],
    table_name: str,
    base_docs_url: str,
    max_chunks_per_title: int,
    rerank_candidate_limit: int,
    rerank_top_n: int,
    rerank_enabled: bool,
    reranker,
    rerank_lock,
):
    """
    사용자 질문(query)에 대해 내부 문서 chunk 검색 수행.

    처리 단계
    1) query -> embedding 벡터 변환
    2) 키워드 / phrase 후보 생성
    3) PostgreSQL 하이브리드 검색
    4) 결과 dict 변환
    5) title 편향 제한
    6) (옵션) reranker 적용
    """

    # --------------------------------------------------------
    # 1) 질문을 embedding 벡터로 변환
    # --------------------------------------------------------
    with embedding_lock:
        query_vec = embed_model.encode(
            "Represent this sentence for searching relevant passages: " + query
        ).tolist()

    # --------------------------------------------------------
    # 2) 키워드 / phrase 후보 생성
    # --------------------------------------------------------
    _keywords = extract_search_keywords(query)
    phrase_info = generate_phrase_candidates(query)

    conn = None

    try:
        conn = psycopg2.connect(**db_config)
        register_vector(conn)

        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # ----------------------------------------------------
        # 3) 하이브리드 검색 SQL
        # ----------------------------------------------------
        sql = f"""
            WITH vector_matches AS (
                SELECT id, (embedding <=> %s::vector) AS dist
                FROM {table_name}
                ORDER BY dist ASC
                LIMIT 50
            ),
            phrase_matches AS (
                SELECT id,
                       (CASE
                            WHEN %s IS NOT NULL
                             AND title ILIKE '%%'||%s||'%%'
                            THEN 2.0
                            ELSE 0
                        END) AS phrase_score
                FROM {table_name}
                LIMIT 100
            ),
            keyword_matches AS (
                SELECT id,
                       ts_rank_cd(content_search_vector, websearch_to_tsquery('simple', %s)) AS rank
                FROM {table_name}
                LIMIT 100
            )
            SELECT t.*,
                   COALESCE(1.0/(60+v.dist*100),0)
                 + COALESCE(k.rank*10,0)
                 + COALESCE(p.phrase_score,0) AS combined_score
            FROM {table_name} t
            LEFT JOIN vector_matches v ON t.id=v.id
            LEFT JOIN keyword_matches k ON t.id=k.id
            LEFT JOIN phrase_matches p ON t.id=p.id
            WHERE v.id IS NOT NULL OR k.id IS NOT NULL OR p.id IS NOT NULL
            ORDER BY combined_score DESC
            LIMIT {rerank_candidate_limit}
        """

        cursor.execute(
            sql,
            (
                query_vec,
                phrase_info["original"],
                phrase_info["original"],
                query,
            ),
        )

        rows = cursor.fetchall()

        # ----------------------------------------------------
        # 4) DB row -> 결과 dict 변환
        # ----------------------------------------------------
        results = [
            {
                "content": r["original_content"],
                "title": r["title"] or "Unknown",
                "url": build_document_url(r["title"] or "Unknown", base_docs_url),
                "combined_score": float(r["combined_score"]),
            }
            for r in rows
        ]

        # ----------------------------------------------------
        # 5) 같은 문서 편향 제한
        # ----------------------------------------------------
        results = limit_results_per_title(
            results,
            max_per_title=max_chunks_per_title,
        )

        # ----------------------------------------------------
        # 6) (옵션) reranker 적용
        # ----------------------------------------------------
        return rerank_results(
            query,
            results,
            top_n=rerank_top_n,
            rerank_enabled=rerank_enabled,
            reranker=reranker,
            rerank_lock=rerank_lock,
        )

    except Exception as e:
        print(f"DB Error: {e}")
        return []

    finally:
        if conn:
            conn.close()