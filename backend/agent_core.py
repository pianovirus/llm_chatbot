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

# 2. .env 설정값 로드
MODEL_NAME = os.getenv("LLM_MODEL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
TABLE_NAME = os.getenv("DB_TABLE_NAME")
BASE_DOCS_URL = os.getenv("BASE_DOCS_URL", "http://localhost:8000/files/")
DB_NAME = os.getenv("DB_NAME")

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


# --- 유틸리티 함수 ---

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


def extract_search_keywords(query: str, max_keywords: int = 3):
    """
    질문에서 의미 있는 키워드 최대 3개 추출
    """
    if not query:
        return []

    raw_tokens = re.findall(r"[가-힣A-Za-z0-9_-]{2,}", query)

    stopwords = {
        "방법", "문의", "관련", "설명", "기능", "화면", "문서", "가이드",
        "알려줘", "알려", "주세요", "부탁", "어떻게", "어디", "왜", "무엇",
        "뭐", "정도", "가능", "확인", "찾기", "찾나요", "있나요", "되나요",
        "해주세요", "보여줘", "설정해", "설정법"
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

    # 긴 키워드 우선
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

    # reranker 비활성화 또는 로드 실패 시 그대로 상위 N개만 사용
    if not RERANK_ENABLED or reranker is None:
        return results[:top_n]

    try:
        pairs = []
        for r in results:
            # 제목 + 본문을 함께 넣어 relevance 판별
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


def get_internal_context(query: str):
    """
    하이브리드 검색:
    - Vector similarity
    - PostgreSQL FTS
    - 제목/본문 직접 매칭 가중치
    - 문서 중복 제한
    - Reranker 재정렬
    """
    with embedding_lock:
        instruction = "Represent this sentence for searching relevant passages: "
        query_vec = embed_model.encode(instruction + query).tolist()

    keywords = extract_search_keywords(query, max_keywords=3)
    k1 = keywords[0] if len(keywords) > 0 else None
    k2 = keywords[1] if len(keywords) > 1 else None
    k3 = keywords[2] if len(keywords) > 2 else None

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

                (
                    CASE
                        WHEN %s IS NOT NULL AND t.title ILIKE '%%' || %s || '%%' THEN 0.8
                        WHEN %s IS NOT NULL AND t.title ILIKE '%%' || %s || '%%' THEN 0.6
                        WHEN %s IS NOT NULL AND t.title ILIKE '%%' || %s || '%%' THEN 0.4
                        ELSE 0
                    END
                ) +

                (
                    CASE
                        WHEN %s IS NOT NULL AND t.original_content ILIKE '%%' || %s || '%%' THEN 0.5
                        WHEN %s IS NOT NULL AND t.original_content ILIKE '%%' || %s || '%%' THEN 0.35
                        WHEN %s IS NOT NULL AND t.original_content ILIKE '%%' || %s || '%%' THEN 0.2
                        ELSE 0
                    END
                ) AS combined_score

            FROM {TABLE_NAME} t
            LEFT JOIN vector_matches v ON t.id = v.id
            LEFT JOIN keyword_matches k ON t.id = k.id
            WHERE v.id IS NOT NULL OR k.id IS NOT NULL
            ORDER BY combined_score DESC
            LIMIT {RERANK_CANDIDATE_LIMIT};
        """

        params = (
            query_vec,
            query,
            query,

            k1, k1,
            k2, k2,
            k3, k3,

            k1, k1,
            k2, k2,
            k3, k3,

            k1, k1,
            k2, k2,
            k3, k3,

            k1, k1,
            k2, k2,
            k3, k3
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

        # 같은 title 중복 제한
        results = limit_results_per_title(results, max_per_title=MAX_CHUNKS_PER_TITLE)

        # reranker 적용
        results = rerank_results(query, results, top_n=RERANK_TOP_N)

        return results

    except Exception as err:
        print(f"❌ [DB Error] {err}")
        return []
    finally:
        if conn:
            conn.close()


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
            query = params.get("query", [""])[0].strip()

            if not query:
                self._send_json(400, {"status": "error", "message": "query 파라미터가 필요합니다."})
                return

            self.send_response(200)
            self.send_header("Content-type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                self._send_sse({"status": "🔍 지식 데이터 검색 중..."})
                self._send_heartbeat()

                search_results = get_internal_context(query)

                self._send_sse({"status": "📚 검색 결과 정리 중..."})
                self._send_heartbeat()

                if search_results:
                    context_combined = "\n".join(
                        [f"### [데이터 {i+1}]\n제목: {r['title']}\n내용: {r['content']}" for i, r in enumerate(search_results)]
                    )

                    self._send_sse({"status": "📝 답변 프롬프트 준비 중..."})
                    self._send_heartbeat()

                    prompt = (
                        f"당신은 기술지원 전문가입니다. 반드시 제공된 [지식 데이터]를 근거로 답변하세요.\n"
                        f"직접적인 절차가 없더라도 데이터 내의 메뉴명, 버튼 이름 등을 활용해 논리적으로 추론하여 안내할 수 있습니다.\n"
                        f"단, 데이터에 전혀 근거가 없는 내용은 지어내지 마세요.\n"
                        f"모르면 모른다고 말하세요.\n\n"
                        f"### [지식 데이터]\n{context_combined}\n\n"
                        f"### [사용자 질문]\n{query}\n\n"
                        f"### [응답 규칙]\n"
                        f"1. 반드시 다음 순서를 지키세요: [## 📢 조치 안내] -> [### 🛠️ 상세 절차] -> [### 💡 주의 사항]\n"
                        f"2. 추론한 내용일 경우 반드시 '매뉴얼 기반 추론 절차입니다'라고 명시하세요.\n"
                        f"3. 아주 상세하고 친절하게 답변하세요.\n"
                        f"4. 관련 문서 제목이나 메뉴명을 답변 중 자연스럽게 인용해도 됩니다."
                    )

                    self._send_sse({"status": "🤖 LLM 응답 생성 시작..."})
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

                    unique_links = {}
                    for r in search_results:
                        if r["title"] not in unique_links and r["url"]:
                            unique_links[r["title"]] = r["url"]

                    valid_links_text = "\n".join(
                        [f"- [{title}]({url})" for title, url in unique_links.items()]
                    )

                    if valid_links_text:
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