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

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer, CrossEncoder
from langgraph.graph import StateGraph, START, END

# 2. .env 설정값 로드
MODEL_NAME = os.getenv("LLM_MODEL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
TABLE_NAME = os.getenv("DB_TABLE_NAME")
BASE_DOCS_URL = os.getenv("BASE_DOCS_URL", "http://localhost:8000/files/")
DB_NAME = os.getenv("DB_NAME")

# Query Rewrite 설정
QUERY_REWRITE_ENABLED = os.getenv("QUERY_REWRITE_ENABLED", "true").lower() == "true"
QUERY_REWRITE_MAX_CHARS = int(os.getenv("QUERY_REWRITE_MAX_CHARS", "80"))

# Reranker 설정
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "false").lower() == "true"
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
RERANK_CANDIDATE_LIMIT = int(os.getenv("RERANK_CANDIDATE_LIMIT", "20"))

# 같은 문서(title)에서 최종 후보로 허용할 최대 chunk 수
MAX_CHUNKS_PER_TITLE = int(os.getenv("MAX_CHUNKS_PER_TITLE", "2"))

# PostgreSQL 연결 정보
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "dbname": os.getenv("DB_NAME")
}

# ✅ 문서 서빙은 원래 방식 유지
STORAGE_DIR = os.path.join(os.getcwd(), "storage")
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

print(f"📁 STORAGE_DIR: {STORAGE_DIR}")
print(f"📁 storage exists: {os.path.exists(STORAGE_DIR)}")

embedding_lock = threading.Lock()
rerank_lock = threading.Lock()
llm_lock = threading.Lock()

# 3. 모델 로드
print(f"⏳ [System] 임베딩 모델 로드 중... ({EMBED_MODEL_NAME})")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

reranker = None
if RERANK_ENABLED:
    try:
        print(f"⏳ [System] Reranker 모델 로드 중... ({RERANK_MODEL_NAME})")
        reranker = CrossEncoder(RERANK_MODEL_NAME)
        print("✅ [System] Reranker 모델 로드 완료.")
    except Exception as e:
        print(f"❌ [System] Reranker 로드 실패: {e}")
        reranker = None

llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0, timeout=180)
print("✅ [System] PostgreSQL RAG 엔진 가동 준비 완료.")


# --- LangGraph State ---

class PipelineState(TypedDict, total=False):
    original_query: str
    effective_query: str
    search_results: List[Dict[str, Any]]
    context_combined: str
    prompt: str
    doc_links: Dict[str, str]


# --- 유틸리티 함수 ---

def safe_single_line(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.strip().split())


def normalize_keyword(token: str) -> str:
    """
    한국어 조사/어미를 아주 단순하게 제거해서 검색 키워드 품질 개선
    """
    if not token:
        return ""

    token = unicodedata.normalize("NFC", token.strip())

    suffixes = [
        "에서는", "으로는", "에게서", "까지는", "부터는",
        "에서", "으로", "에게", "까지", "부터", "처럼",
        "하고", "이며", "이고", "하면",
        "라는", "라고", "니다", "어요", "아요",
        "를", "을", "은", "는", "이", "가", "도", "만",
        "와", "과", "에", "의", "로"
    ]

    for suffix in sorted(suffixes, key=len, reverse=True):
        if len(token) > len(suffix) + 1 and token.endswith(suffix):
            token = token[:-len(suffix)]
            break

    return token.strip()


def make_compact_query(query: str) -> str:
    """
    공백 제거 + 구두점 단순 정리
    예: '가상 프린터 포트 번호' -> '가상프린터포트번호'
    """
    if not query:
        return ""
    q = safe_single_line(query)
    q = re.sub(r"[\"'`“”‘’()\[\]{}:;,./\\!?@#$%^&*+=|<>~-]+", " ", q)
    q = safe_single_line(q)
    return q.replace(" ", "")


def generate_phrase_candidates(query: str) -> Dict[str, str]:
    """
    phrase / compact / pair phrase 후보 생성
    """
    q = safe_single_line(query)
    compact = make_compact_query(q)

    raw_tokens = re.findall(r"[가-힣A-Za-z0-9_-]{2,}", q)
    normalized_tokens = []
    seen = set()
    for token in raw_tokens:
        t = normalize_keyword(token)
        if len(t) >= 2 and t not in seen:
            normalized_tokens.append(t)
            seen.add(t)

    # 토큰 2~3개를 붙인 pair/triple phrase
    pair_phrase = ""
    if len(normalized_tokens) >= 2:
        pair_phrase = "".join(normalized_tokens[:2])

    triple_phrase = ""
    if len(normalized_tokens) >= 3:
        triple_phrase = "".join(normalized_tokens[:3])

    spaced_phrase = " ".join(normalized_tokens[:3]) if normalized_tokens else q

    phrases = {
        "original": q,
        "compact": compact,
        "spaced_phrase": spaced_phrase,
        "pair_phrase": pair_phrase,
        "triple_phrase": triple_phrase,
    }

    print("🧱 [Phrase Candidates]")
    for k, v in phrases.items():
        print(f"   - {k}: {v}")

    return phrases


def extract_search_keywords(query: str, max_keywords: int = 4):
    """
    질문에서 의미 있는 키워드 최대 4개 추출
    """
    if not query:
        return []

    raw_tokens = re.findall(r"[가-힣A-Za-z0-9_-]{2,}", query)

    stopwords = {
        "방법", "문의", "관련", "설명", "기능", "화면", "문서", "가이드",
        "알려줘", "알려", "주세요", "부탁", "어떻게", "어디", "왜", "무엇",
        "뭐", "정도", "가능", "확인", "찾기", "찾나요", "있나요", "되나요",
        "해주세요", "보여줘", "설정해", "설정법", "필요한", "필요", "알려"
    }

    cleaned = []
    seen = set()

    for token in raw_tokens:
        normalized = normalize_keyword(token)
        if len(normalized) < 2:
            continue
        if normalized in stopwords:
            continue
        if normalized not in seen:
            seen.add(normalized)
            cleaned.append(normalized)

    # compact phrase도 추가 시도
    compact = make_compact_query(query)
    if len(compact) >= 4 and compact not in seen:
        cleaned.insert(0, compact)

    cleaned = sorted(cleaned, key=lambda x: (-len(x), cleaned.index(x)))
    return cleaned[:max_keywords]


def build_document_url(title_val: str):
    """
    ✅ 원래 방식 유지: title 기반으로만 링크 생성
    """
    file_url_name = title_val if title_val.lower().endswith(".pdf") else f"{title_val}.pdf"
    normalized = unicodedata.normalize("NFC", file_url_name)
    encoded = urllib.parse.quote(normalized)
    base_url = BASE_DOCS_URL.rstrip("/") + "/"
    return f"{base_url}{encoded}"


def limit_results_per_title(results: list, max_per_title: int = 2):
    """
    같은 문서(title)가 너무 많이 상위 결과를 차지하지 않도록 제한
    """
    limited = []
    counter = defaultdict(int)

    for item in results:
        title = item.get("title", "Unknown Document")
        if counter[title] < max_per_title:
            limited.append(item)
            counter[title] += 1

    print(f"📚 [Dedup] title당 최대 {max_per_title}개 유지 -> {len(results)}개 중 {len(limited)}개 사용")
    return limited


def rerank_results(query: str, results: list, top_n: int = 5):
    """
    1차 검색 결과를 CrossEncoder reranker로 재정렬
    """
    if not results:
        return results

    if not RERANK_ENABLED or reranker is None:
        return results[:top_n]

    try:
        pairs = []
        for r in results:
            text = f"{r.get('title', '')}\n{r.get('content', '')}"
            pairs.append((query, text))

        with rerank_lock:
            scores = reranker.predict(pairs)

        reranked = []
        for r, score in zip(results, scores):
            item = dict(r)
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        print(f"🔁 [Reranker] top_n={top_n}")
        for i, item in enumerate(reranked[:top_n], 1):
            print(
                f"   {i}. {item.get('title', 'Unknown')} | "
                f"rerank_score={item.get('rerank_score', 0):.4f} | "
                f"combined_score={item.get('combined_score', 0):.4f}"
            )

        return reranked[:top_n]

    except Exception as e:
        print(f"❌ [Reranker Error] {e}")
        return results[:top_n]


def rewrite_query_for_retrieval(original_query: str) -> str:
    """
    검색용 질의 전처리:
    - 짧고 명사 중심
    - 메뉴/절차 표현은 줄이고 핵심 대상/속성 위주
    - 실패 시 원문 반환
    """
    if not QUERY_REWRITE_ENABLED:
        return original_query

    original_query = safe_single_line(original_query)
    if not original_query:
        return original_query

    prompt = f"""
당신은 RAG 검색용 질의 전처리기입니다.
사용자 질문을 문서 검색에 유리한 짧은 질의로 바꾸세요.

규칙:
1. 답변하지 말고 검색용 질의만 출력하세요.
2. 군더더기 표현(예: 알려줘, 어떻게, 설정할때, 필요한)은 제거하세요.
3. 핵심 명사와 속성 중심으로 정리하세요.
4. 가능하면 복합명사는 유지하세요. 예: 가상프린터, 포트번호
5. 한국어로 출력하세요.
6. 최대 {QUERY_REWRITE_MAX_CHARS}자 이내로 출력하세요.
7. 따옴표, 번호, 설명문 없이 결과만 한 줄로 출력하세요.

사용자 질문:
{original_query}
""".strip()

    try:
        with llm_lock:
            rewritten = llm.invoke(prompt)

        rewritten = safe_single_line(rewritten)
        if not rewritten:
            return original_query

        if len(rewritten) > QUERY_REWRITE_MAX_CHARS:
            return original_query

        print(f"🪄 [Query Rewrite] '{original_query}' -> '{rewritten}'")
        return rewritten

    except Exception as e:
        print(f"❌ [Query Rewrite Error] {e}")
        return original_query


def get_internal_context(query: str):
    """
    하이브리드 검색:
    - Vector similarity
    - PostgreSQL FTS
    - phrase / compact phrase boost
    - 제목/본문 직접 매칭 가중치
    - 문서 중복 제한
    - Reranker 재정렬
    """
    with embedding_lock:
        instruction = "Represent this sentence for searching relevant passages: "
        query_vec = embed_model.encode(instruction + query).tolist()

    keywords = extract_search_keywords(query, max_keywords=4)
    k1 = keywords[0] if len(keywords) > 0 else None
    k2 = keywords[1] if len(keywords) > 1 else None
    k3 = keywords[2] if len(keywords) > 2 else None
    k4 = keywords[3] if len(keywords) > 3 else None

    phrases = generate_phrase_candidates(query)
    phrase_original = phrases["original"] or None
    phrase_compact = phrases["compact"] or None
    phrase_spaced = phrases["spaced_phrase"] or None
    phrase_pair = phrases["pair_phrase"] or None
    phrase_triple = phrases["triple_phrase"] or None

    print(f"\n🧩 [Keyword Extraction] query='{query}' -> keywords={keywords}")

    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        register_vector(conn)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        sql = f"""
            WITH vector_matches AS (
                SELECT id, (embedding <=> %s::vector) AS dist
                FROM {TABLE_NAME}
                ORDER BY dist ASC
                LIMIT 50
            ),
            phrase_matches AS (
                SELECT
                    id,
                    (
                        CASE
                            WHEN %s IS NOT NULL
                             AND content_search_vector @@ phraseto_tsquery('simple', %s)
                            THEN 3.0 ELSE 0
                        END
                    ) +
                    (
                        CASE
                            WHEN %s IS NOT NULL AND title ILIKE '%%' || %s || '%%'
                            THEN 2.0 ELSE 0
                        END
                    ) +
                    (
                        CASE
                            WHEN %s IS NOT NULL AND original_content ILIKE '%%' || %s || '%%'
                            THEN 1.5 ELSE 0
                        END
                    ) +
                    (
                        CASE
                            WHEN %s IS NOT NULL AND title ILIKE '%%' || %s || '%%'
                            THEN 1.4 ELSE 0
                        END
                    ) +
                    (
                        CASE
                            WHEN %s IS NOT NULL AND original_content ILIKE '%%' || %s || '%%'
                            THEN 1.0 ELSE 0
                        END
                    ) +
                    (
                        CASE
                            WHEN %s IS NOT NULL AND title ILIKE '%%' || %s || '%%'
                            THEN 1.2 ELSE 0
                        END
                    ) +
                    (
                        CASE
                            WHEN %s IS NOT NULL AND original_content ILIKE '%%' || %s || '%%'
                            THEN 0.8 ELSE 0
                        END
                    ) AS phrase_score
                FROM {TABLE_NAME}
                WHERE
                    (%s IS NOT NULL AND content_search_vector @@ phraseto_tsquery('simple', %s))
                    OR (%s IS NOT NULL AND title ILIKE '%%' || %s || '%%')
                    OR (%s IS NOT NULL AND original_content ILIKE '%%' || %s || '%%')
                    OR (%s IS NOT NULL AND title ILIKE '%%' || %s || '%%')
                    OR (%s IS NOT NULL AND original_content ILIKE '%%' || %s || '%%')
                    OR (%s IS NOT NULL AND title ILIKE '%%' || %s || '%%')
                    OR (%s IS NOT NULL AND original_content ILIKE '%%' || %s || '%%')
                LIMIT 100
            ),
            keyword_matches AS (
                SELECT
                    id,
                    ts_rank_cd(content_search_vector, websearch_to_tsquery('simple', %s)) AS rank
                FROM {TABLE_NAME}
                WHERE
                    content_search_vector @@ websearch_to_tsquery('simple', %s)
                    OR (%s IS NOT NULL AND original_content ILIKE '%%' || %s || '%%')
                    OR (%s IS NOT NULL AND original_content ILIKE '%%' || %s || '%%')
                    OR (%s IS NOT NULL AND original_content ILIKE '%%' || %s || '%%')
                    OR (%s IS NOT NULL AND original_content ILIKE '%%' || %s || '%%')
                    OR (%s IS NOT NULL AND title ILIKE '%%' || %s || '%%')
                    OR (%s IS NOT NULL AND title ILIKE '%%' || %s || '%%')
                    OR (%s IS NOT NULL AND title ILIKE '%%' || %s || '%%')
                    OR (%s IS NOT NULL AND title ILIKE '%%' || %s || '%%')
                LIMIT 100
            )
            SELECT
                t.original_content,
                t.title,
                t.source_url,
                t.data_source,

                COALESCE(1.0 / (60 + v.dist * 100), 0) +
                COALESCE(k.rank * 10.0, 0) +
                COALESCE(p.phrase_score, 0) +

                (
                    CASE
                        WHEN %s IS NOT NULL AND t.title ILIKE '%%' || %s || '%%' THEN 1.2
                        WHEN %s IS NOT NULL AND t.title ILIKE '%%' || %s || '%%' THEN 0.9
                        WHEN %s IS NOT NULL AND t.title ILIKE '%%' || %s || '%%' THEN 0.7
                        WHEN %s IS NOT NULL AND t.title ILIKE '%%' || %s || '%%' THEN 0.5
                        ELSE 0
                    END
                ) +

                (
                    CASE
                        WHEN %s IS NOT NULL AND t.original_content ILIKE '%%' || %s || '%%' THEN 0.8
                        WHEN %s IS NOT NULL AND t.original_content ILIKE '%%' || %s || '%%' THEN 0.55
                        WHEN %s IS NOT NULL AND t.original_content ILIKE '%%' || %s || '%%' THEN 0.4
                        WHEN %s IS NOT NULL AND t.original_content ILIKE '%%' || %s || '%%' THEN 0.25
                        ELSE 0
                    END
                ) AS combined_score

            FROM {TABLE_NAME} t
            LEFT JOIN vector_matches v ON t.id = v.id
            LEFT JOIN keyword_matches k ON t.id = k.id
            LEFT JOIN phrase_matches p ON t.id = p.id
            WHERE v.id IS NOT NULL OR k.id IS NOT NULL OR p.id IS NOT NULL
            ORDER BY combined_score DESC
            LIMIT {RERANK_CANDIDATE_LIMIT};
        """

        params = (
            query_vec,

            # phrase_matches SELECT
            phrase_spaced, phrase_spaced,
            phrase_original, phrase_original,
            phrase_original, phrase_original,
            phrase_compact, phrase_compact,
            phrase_compact, phrase_compact,
            phrase_pair, phrase_pair,
            phrase_pair, phrase_pair,

            # phrase_matches WHERE
            phrase_spaced, phrase_spaced,
            phrase_original, phrase_original,
            phrase_original, phrase_original,
            phrase_compact, phrase_compact,
            phrase_compact, phrase_compact,
            phrase_pair, phrase_pair,
            phrase_pair, phrase_pair,

            # keyword_matches
            query,
            query,
            k1, k1,
            k2, k2,
            k3, k3,
            k4, k4,
            k1, k1,
            k2, k2,
            k3, k3,
            k4, k4,

            # final title boost
            k1, k1,
            k2, k2,
            k3, k3,
            k4, k4,

            # final content boost
            k1, k1,
            k2, k2,
            k3, k3,
            k4, k4,
        )

        cursor.execute(sql, params)
        rows = cursor.fetchall()

        results = []
        print(f"🔍 [Hybrid Search Debug] 질문: '{query}'")
        for row in rows:
            score = float(row["combined_score"])
            print(f"   - [{row['data_source']}] {row['title']} | 통합 점수: {score:.4f}")

            if score > 0.001:
                title_val = row["title"] if row["title"] else "Unknown Document"
                url = build_document_url(title_val)

                results.append({
                    "content": row["original_content"],
                    "title": title_val,
                    "url": url,
                    "combined_score": score,
                    "data_source": row.get("data_source"),
                })

        results = limit_results_per_title(results, max_per_title=MAX_CHUNKS_PER_TITLE)
        results = rerank_results(query, results, top_n=RERANK_TOP_N)

        print("📄 [Final Results Preview]")
        for i, item in enumerate(results, 1):
            preview = safe_single_line(item.get("content", ""))[:180]
            print(f"   {i}. {item.get('title')} | {preview}")

        return results

    except Exception as err:
        print(f"❌ [DB Error] {err}")
        return []
    finally:
        if conn:
            conn.close()


def build_doc_links(search_results: List[Dict[str, Any]]) -> Dict[str, str]:
    unique_links = {}
    for r in search_results:
        if r["title"] not in unique_links and r["url"]:
            unique_links[r["title"]] = r["url"]
    return unique_links


def build_answer_prompt(original_query: str, effective_query: str, context_combined: str) -> str:
    return (
        f"당신은 기술지원 전문가입니다. 반드시 제공된 [지식 데이터]를 근거로 답변하세요.\n"
        f"직접적인 절차가 없더라도 데이터 내의 메뉴명, 버튼 이름 등을 활용해 논리적으로 추론하여 안내할 수 있습니다.\n"
        f"단, 데이터에 전혀 근거가 없는 내용은 지어내지 마세요.\n"
        f"모르면 모른다고 말하세요.\n\n"
        f"### [원래 사용자 질문]\n{original_query}\n\n"
        f"### [검색용 전처리 질문]\n{effective_query}\n\n"
        f"### [지식 데이터]\n{context_combined}\n\n"
        f"### [응답 규칙]\n"
        f"1. 반드시 다음 순서를 지키세요: [## 📢 조치 안내] -> [### 🛠️ 상세 절차] -> [### 💡 주의 사항]\n"
        f"2. 추론한 내용일 경우 반드시 '매뉴얼 기반 추론 절차입니다'라고 명시하세요.\n"
        f"3. 아주 상세하고 친절하게 답변하세요.\n"
        f"4. 관련 문서 제목이나 메뉴명을 답변 중 자연스럽게 인용해도 됩니다.\n"
        f"5. 검색용 전처리 질문은 내부 검색 보조용이므로, 최종 답변은 반드시 원래 사용자 질문에 자연스럽게 답하세요.\n"
        f"6. 문서 안에 실제 값(예: 번호, 포트번호, 주소, 버전, 경로)이 직접 명시되어 있으면 절차 설명보다 그 값을 우선적으로 답하세요.\n"
    )


# --- LangGraph Nodes ---

def rewrite_query_node(state: PipelineState) -> Dict[str, Any]:
    original_query = state["original_query"]
    effective_query = rewrite_query_for_retrieval(original_query)
    return {
        "effective_query": effective_query
    }


def retrieve_context_node(state: PipelineState) -> Dict[str, Any]:
    effective_query = state.get("effective_query") or state["original_query"]
    search_results = get_internal_context(effective_query)
    return {
        "search_results": search_results
    }


def prepare_prompt_node(state: PipelineState) -> Dict[str, Any]:
    original_query = state["original_query"]
    effective_query = state.get("effective_query") or original_query
    search_results = state.get("search_results", [])

    context_combined = "\n".join(
        [
            f"### [데이터 {i+1}]\n제목: {r['title']}\n내용: {r['content']}"
            for i, r in enumerate(search_results)
        ]
    )

    prompt = build_answer_prompt(original_query, effective_query, context_combined)
    doc_links = build_doc_links(search_results)

    return {
        "context_combined": context_combined,
        "prompt": prompt,
        "doc_links": doc_links
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


# --- HTTP 핸들러 ---

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

    def _send_done(self):
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _send_heartbeat(self):
        """
        프론트 연결 유지용 heartbeat
        """
        try:
            self.wfile.write(b": keep-alive\n\n")
            self.wfile.flush()
        except Exception as e:
            print(f"⚠️ heartbeat send failed: {e}")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def serve_static_file(self, file_path):
        try:
            decoded_filename = unquote(file_path)
            full_path = os.path.join(STORAGE_DIR, decoded_filename)

            print("========== FILE DEBUG ==========")
            print(f"Requested raw  : {file_path}")
            print(f"Requested name : {decoded_filename}")
            print(f"Full path      : {full_path}")
            print(f"Exists?        : {os.path.exists(full_path)}")
            print(f"Is file?       : {os.path.isfile(full_path)}")

            if os.path.exists(full_path) and os.path.isfile(full_path):
                self.send_response(200)
                mime_type, _ = mimetypes.guess_type(full_path)
                self.send_header("Content-type", mime_type or "application/octet-stream")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(full_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                print(f"❌ File not found: {full_path}")
                self.send_error(404, "File Not Found")
        except Exception as e:
            print(f"🚨 serve_static_file error: {e}")
            self.send_error(500, str(e))

    def do_POST(self):
        if self.path == "/feedback":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(content_length)

                if not raw_body:
                    self._send_json(400, {"status": "error", "message": "요청 본문이 비어 있습니다."})
                    return

                data = json.loads(raw_body)
                query = data.get("query")
                answer = data.get("answer")

                if not query or not answer:
                    self._send_json(400, {"status": "error", "message": "query 와 answer 는 필수입니다."})
                    return

                self._save_feedback_to_db(query, answer)
                self._send_json(200, {"status": "success", "message": "성공적으로 학습되었습니다."})

            except Exception as e:
                print(f"🚨 피드백 저장 오류: {e}")
                self._send_json(500, {"status": "error", "message": str(e)})

    def do_GET(self):
        parsed_path = urlparse(self.path)

        # ✅ 원래 방식 유지
        if parsed_path.path.startswith("/files/"):
            file_path = parsed_path.path[len("/files/"):]
            self.serve_static_file(file_path)
            return

        if parsed_path.path == "/search":
            params = parse_qs(parsed_path.query)
            original_query = params.get("query", [""])[0].strip()

            if not original_query:
                self._send_json(400, {"status": "error", "message": "query 파라미터가 필요합니다."})
                return

            self.send_response(200)
            self.send_header("Content-type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                self._send_sse({"status": "🧠 질문 분석 중..."})
                self._send_heartbeat()

                current_state: PipelineState = {
                    "original_query": original_query
                }

                for update in rag_pipeline.stream(current_state, stream_mode="updates"):
                    for node_name, node_output in update.items():
                        current_state.update(node_output)

                        if node_name == "rewrite_query":
                            effective_query = current_state.get("effective_query", original_query)
                            if QUERY_REWRITE_ENABLED:
                                if effective_query != original_query:
                                    self._send_sse({"status": f"🪄 질문 분석 완료: {effective_query}"})
                                else:
                                    self._send_sse({"status": "🪄 질문 분석 완료"})
                            else:
                                self._send_sse({"status": "🪄 질문 분석 비활성화"})
                            self._send_heartbeat()

                        elif node_name == "retrieve_context":
                            result_count = len(current_state.get("search_results", []))
                            self._send_sse({"status": f"🔍 문서 검색 완료 ({result_count}건)"})
                            self._send_heartbeat()

                        elif node_name == "prepare_prompt":
                            self._send_sse({"status": "📝 답변 프롬프트 준비 완료"})
                            self._send_heartbeat()

                search_results = current_state.get("search_results", [])
                prompt = current_state.get("prompt", "")
                doc_links = current_state.get("doc_links", {})

                if search_results and prompt:
                    self._send_sse({"status": "🤖 응답 생성 시작..."})
                    self._send_heartbeat()

                    last_ping = time.time()
                    for chunk in llm.stream(prompt):
                        now = time.time()
                        if now - last_ping > 5:
                            self._send_heartbeat()
                            last_ping = now

                        self._send_sse({"status": "💬 답변 중..."})
                        if chunk:
                            self._send_sse({"chunk": chunk})

                    if doc_links:
                        valid_links_text = "\n".join(
                            [f"- [{title}]({url})" for title, url in doc_links.items()]
                        )
                        self._send_sse({"chunk": "\n\n---\n### 🔗 관련 문서\n" + valid_links_text})
                else:
                    self._send_sse({"chunk": "관련 정보를 찾지 못했습니다."})

                self._send_done()

            except Exception as e:
                print(f"🚨 서버 오류: {e}")
                self._send_sse({"error": str(e)})
                self._send_done()

    def _save_feedback_to_db(self, query, answer):
        combined_text = f"질문: {query}\n전문가 답변: {answer}"

        with embedding_lock:
            vector = embed_model.encode(combined_text).tolist()

        conn = None
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            register_vector(conn)
            cursor = conn.cursor()

            sql = f"""
                INSERT INTO {TABLE_NAME}
                (content_type, data_source, original_content, embedding, title, content_search_vector)
                VALUES (%s, %s, %s, %s, %s, to_tsvector('simple', %s))
            """
            cursor.execute(
                sql,
                ("text", "feedback", combined_text, vector, "검증된 답변", combined_text)
            )
            conn.commit()

        finally:
            if conn:
                conn.close()


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 8000), RAGHandler)
    print("📡 RAG Service 가동: http://localhost:8000")
    server.serve_forever()