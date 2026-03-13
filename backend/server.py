# server.py
# =========================================================
# HTTP Server Handler 모듈
# - RAGHandler만 분리
# - main.py에서 필요한 의존성(파이프라인/모델/DB/디렉토리)을 주입해서 사용
# - /files     : 파일 서빙
# - /search    : 기존 웹용 SSE 검색/답변 스트리밍
# - /feedback  : 사용자 피드백 저장
# - /kakao     : 카카오톡 챗봇용 POST 엔드포인트
# =========================================================

import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

import psycopg2
from pgvector.psycopg2 import register_vector


def make_handler(
    *,
    storage_dir: str,
    rag_pipeline,
    llm_chain,
    embed_model,
    db_config,
    table_name: str,
    llm_provider: str,
):
    """
    (역할) 의존성을 주입받아 RAGHandler 클래스를 만들어 반환하는 factory
    - ThreadingHTTPServer(("",8000), HandlerClass) 형태로 넣기 위해 클래스가 필요함
    - 이 방식으로 server.py <-> main.py 순환 import를 피함
    """

    class RAGHandler(BaseHTTPRequestHandler):
        # -------------------------------------------------
        # 공통 응답 헬퍼
        # -------------------------------------------------
        def _send_json(self, status_code, payload):
            self.send_response(status_code)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

        def _send_sse(self, data):
            self.wfile.write(
                f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")
            )
            self.wfile.flush()

        def _read_json_body(self):
            content_length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
            return json.loads(raw.decode("utf-8"))

        def _run_rag_once(self, query: str):
            import time

            t0 = time.time()

            state = {"original_query": query}

            for update in rag_pipeline.stream(state, stream_mode="updates"):
                for _, out in update.items():
                    state.update(out)

            t1 = time.time()
            print(f"[TIME] pipeline: {t1 - t0:.2f}s")

            prompt = state.get("prompt", "")
            if not prompt:
                raise ValueError("RAG pipeline did not produce a prompt.")

            print(f"[INFO] prompt length: {len(prompt)}")

            answer = llm_chain.invoke(prompt)
            t2 = time.time()

            print(f"[TIME] llm invoke: {t2 - t1:.2f}s")
            print(f"[TIME] total: {t2 - t0:.2f}s")

            if answer is None:
                answer = ""

            answer = str(answer).strip()

            if state.get("doc_links"):
                links = "\n".join(
                    [f"- {title}: {url}" for title, url in state["doc_links"].items()]
                )
                answer += f"\n\n관련 문서:\n{links}"

            return answer, state

        def _truncate_for_kakao(self, text: str, limit: int = 1000):
            """
            카카오 simpleText 길이 방어용.
            너무 길면 잘라서 반환.
            """
            text = (text or "").strip()
            if len(text) <= limit:
                return text
            return text[: limit - 4].rstrip() + " ..."

        def _make_kakao_response(self, text: str):
            """
            카카오톡 챗봇 응답 포맷
            """
            safe_text = self._truncate_for_kakao(text, limit=1000)
            return {
                "version": "2.0",
                "template": {
                    "outputs": [
                        {
                            "simpleText": {
                                "text": safe_text
                            }
                        }
                    ]
                }
            }

        # -------------------------------------------------
        # CORS / Preflight
        # -------------------------------------------------
        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        # -------------------------------------------------
        # GET 요청 라우팅
        # -------------------------------------------------
        def do_GET(self):
            parsed = urlparse(self.path)

            # -----------------------------
            # (A) /files/<filename> 제공
            # -----------------------------
            if parsed.path.startswith("/files/"):
                full_path = os.path.join(storage_dir, unquote(parsed.path[7:]))

                if os.path.exists(full_path) and os.path.isfile(full_path):
                    self.send_response(200)
                    self.send_header(
                        "Content-type",
                        mimetypes.guess_type(full_path)[0] or "application/octet-stream",
                    )
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()

                    with open(full_path, "rb") as f:
                        self.wfile.write(f.read())
                else:
                    self.send_error(404)
                return

            # -----------------------------
            # (B) /search?query=... 실행
            # - SSE로 진행 상태(status) + 답변(chunk) 스트리밍
            # -----------------------------
            if parsed.path == "/search":
                query = parse_qs(parsed.query).get("query", [""])[0].strip()

                self.send_response(200)
                self.send_header("Content-type", "text/event-stream; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                try:
                    self._send_sse({"status": "🧠🔗⚙️ 질문 요약 중..."})

                    state = {"original_query": query}

                    # rewrite -> retrieve -> prepare
                    for update in rag_pipeline.stream(state, stream_mode="updates"):
                        self._send_sse({"status": "🔍 검색 중..."})
                        for node, out in update.items():
                            state.update(out)
                            if node == "retrieve_context":
                                self._send_sse(
                                    {"status": f"🔍 검색 완료 ({len(state.get('search_results', []))}건)"}
                                )

                    self._send_sse({"status": "🤖✨ 답변 생성 중..."})

                    # 웹용은 기존처럼 스트리밍 유지
                    for chunk in llm_chain.stream(state["prompt"]):
                        self._send_sse({"status": "💬✍️ 답변 중..."})
                        if chunk:
                            self._send_sse({"chunk": chunk})

                    if state.get("doc_links"):
                        links = "\n".join(
                            [f"- [{t}]({u})" for t, u in state["doc_links"].items()]
                        )
                        self._send_sse({"chunk": f"\n\n---\n### 🔗 관련 문서\n{links}"})

                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()

                except Exception as e:
                    self._send_sse({"error": str(e)})
                return

            # -----------------------------
            # (C) health check
            # -----------------------------
            if parsed.path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "provider": llm_provider,
                    },
                )
                return

            self._send_json(404, {"error": "Not Found"})

        # -------------------------------------------------
        # POST 요청 라우팅
        # -------------------------------------------------
        def do_POST(self):
            # -----------------------------
            # (D) /feedback 저장
            # - 질문/답변을 하나의 텍스트로 만들고 임베딩해서 DB에 저장
            # -----------------------------
            if self.path == "/feedback":
                try:
                    data = self._read_json_body()
                    self._save_feedback(data["query"], data["answer"])
                    self._send_json(200, {"status": "success"})
                except Exception as e:
                    self._send_json(500, {"error": str(e)})
                return

            # -----------------------------
            # (E) /kakao
            # - 카카오톡 챗봇용 POST 엔드포인트
            # - userRequest.utterance 를 받아 기존 RAG -> LLM 실행
            # - 카카오 응답 JSON으로 반환
            # -----------------------------
            if self.path == "/kakao":
                try:
                    data = self._read_json_body()

                    user_request = data.get("userRequest", {}) or {}
                    query = (user_request.get("utterance") or "").strip()

                    if not query:
                        self._send_json(
                            200,
                            self._make_kakao_response("질문이 비어 있습니다. 질문을 입력해 주세요.")
                        )
                        return

                    answer, _state = self._run_rag_once(query)

                    if not answer:
                        answer = "답변을 생성하지 못했습니다. 다시 질문해 주세요."

                    self._send_json(200, self._make_kakao_response(answer))

                except Exception as e:
                    # 카카오는 가능하면 200 + 안내문으로 돌려주는 편이 운영상 유리
                    self._send_json(
                        200,
                        self._make_kakao_response(
                            f"처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.\n오류: {str(e)}"
                        ),
                    )
                return

            # -----------------------------
            # (F) 일반 JSON 테스트용 엔드포인트 (선택)
            # - Postman/curl로 기존 RAG 재사용 테스트 가능
            # - body: {"query": "..."}
            # -----------------------------
            if self.path == "/ask":
                try:
                    data = self._read_json_body()
                    query = (data.get("query") or "").strip()

                    if not query:
                        self._send_json(400, {"error": "query is required"})
                        return

                    answer, state = self._run_rag_once(query)
                    self._send_json(
                        200,
                        {
                            "query": query,
                            "answer": answer,
                            "doc_links": state.get("doc_links", {}),
                        },
                    )
                except Exception as e:
                    self._send_json(500, {"error": str(e)})
                return

            self._send_json(404, {"error": "Not Found"})

        # -------------------------------------------------
        # 사용자 피드백 저장
        # -------------------------------------------------
        def _save_feedback(self, q, a):
            text = f"질문: {q}\n답변: {a}"
            vec = embed_model.encode(text).tolist()

            conn = psycopg2.connect(**db_config)
            try:
                register_vector(conn)
                cur = conn.cursor()
                cur.execute(
                    f"INSERT INTO {table_name} (content_type, data_source, original_content, embedding, title) "
                    f"VALUES (%s, %s, %s, %s, %s)",
                    ("text", "feedback", text, vec, "검증된 답변"),
                )
                conn.commit()
            finally:
                conn.close()

        # 기본 로그를 조금 보기 좋게
        def log_message(self, format, *args):
            print(f"[HTTP] {self.address_string()} - {format % args}")

    return RAGHandler