# retrieval.py
# =========================================================
# Retrieval(검색) 모듈
# - PostgreSQL(pgvector)에서 문서 chunk를 하이브리드 방식으로 검색
# - 같은 title 편향 제한, (옵션) reranker 재정렬 포함
# - main.py에서 embed_model, reranker, locks를 "주입"받아 사용 (안전한 모듈화)
# =========================================================

import urllib.parse
import unicodedata
from collections import defaultdict
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

# text_utils에서 전처리 함수 사용
from text_utils import extract_search_keywords, generate_phrase_candidates


# (역할) 문서 title을 파일 서빙 URL(/files/...)로 변환
def build_document_url(title_val: str, base_docs_url: str) -> str:
    name = title_val if title_val.lower().endswith(".pdf") else f"{title_val}.pdf"
    encoded = urllib.parse.quote(unicodedata.normalize("NFC", name))
    return f"{base_docs_url.rstrip('/')}/{encoded}"


# (역할) 같은 문서(title)에서 최대 chunk 개수 제한 (컨텍스트 편향 방지)
def limit_results_per_title(results: list, max_per_title: int = 2):
    limited, counter = [], defaultdict(int)
    for item in results:
        t = item.get("title", "Unknown")
        if counter[t] < max_per_title:
            limited.append(item)
            counter[t] += 1
    return limited


# (역할) (옵션) reranker 모델로 검색 결과 재정렬
def rerank_results(
    query: str,
    results: list,
    top_n: int,
    rerank_enabled: bool,
    reranker,
    rerank_lock,
):
    if not results or not rerank_enabled or reranker is None:
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


# (역할) DB에서 질문과 관련된 문서 chunk들을 검색해 top 결과를 반환
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
    # (1) 질문을 임베딩 벡터로 변환
    with embedding_lock:
        query_vec = embed_model.encode(
            "Represent this sentence for searching relevant passages: " + query
        ).tolist()

    # (2) 키워드/phrase 후보 생성 (하이브리드 검색에 사용)
    _k = extract_search_keywords(query)  # 현재 SQL에서는 query만 쓰지만 확장 가능
    p = generate_phrase_candidates(query)

    conn = None
    try:
        conn = psycopg2.connect(**db_config)
        register_vector(conn)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # (3) SQL: vector + keyword + phrase score를 합산
        sql = f"""
            WITH vector_matches AS (
                SELECT id, (embedding <=> %s::vector) AS dist
                FROM {table_name}
                ORDER BY dist ASC
                LIMIT 50
            ),
            phrase_matches AS (
                SELECT id,
                       (CASE WHEN %s IS NOT NULL AND title ILIKE '%%'||%s||'%%' THEN 2.0 ELSE 0 END) AS phrase_score
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
        cursor.execute(sql, (query_vec, p["original"], p["original"], query))
        rows = cursor.fetchall()

        # (4) DB row -> 응답 dict로 정리
        results = [
            {
                "content": r["original_content"],
                "title": r["title"] or "Unknown",
                "url": build_document_url(r["title"], base_docs_url),
                "combined_score": float(r["combined_score"]),
            }
            for r in rows
        ]

        # (5) 같은 title 문서 편향 제한 + (옵션) rerank 적용
        results = limit_results_per_title(results, max_per_title=max_chunks_per_title)
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