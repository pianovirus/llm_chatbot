# server.py
# =========================================================
# HTTP Server Handler 모듈
# - RAGHandler만 분리
# - main.py에서 필요한 의존성(파이프라인/모델/DB/디렉토리)을 주입해서 사용
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
        # (역할) 일반 JSON 응답을 반환하는 헬퍼
        def _send_json(self, status_code, payload):
            self.send_response(status_code)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

        # (역할) SSE 이벤트 한 번을 전송하는 헬퍼
        def _send_sse(self, data):
            self.wfile.write(f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

        # (역할) GET 요청 라우팅: /files, /search 처리
        def do_GET(self):
            parsed = urlparse(self.path)

            # -----------------------------
            # (A) /files/<filename> 제공
            # -----------------------------
            if parsed.path.startswith("/files/"):
                full_path = os.path.join(storage_dir, unquote(parsed.path[7:]))
                if os.path.exists(full_path):
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
                    # 1) 상태: 질문 요약 단계
                    self._send_sse({"status": "🧠🔗⚙️ 질문 요약 중..."})

                    # 파이프라인 state 시작값
                    state = {"original_query": query}

                    # 2) LangGraph 파이프라인 실행 (rewrite -> retrieve -> prepare)
                    for update in rag_pipeline.stream(state, stream_mode="updates"):
                        self._send_sse({"status": "🔍 검색 중..."})
                        for node, out in update.items():
                            state.update(out)
                            if node == "retrieve_context":
                                self._send_sse({"status": f"🔍 검색 완료 ({len(state['search_results'])}건)"})

                    # 3) 상태: 답변 생성 시작
                    self._send_sse({"status": "🤖✨ 답변 생성 중..."})

                    # 4) LLM 답변을 chunk 단위로 스트리밍
                    for chunk in llm_chain.stream(state["prompt"]):
                        self._send_sse({"status": "💬✍️ 답변 중..."})
                        if chunk:
                            self._send_sse({"chunk": chunk})

                    # 5) 문서 링크가 있으면 답변 뒤에 붙여서 제공
                    if state.get("doc_links"):
                        links = "\n".join([f"- [{t}]({u})" for t, u in state["doc_links"].items()])
                        self._send_sse({"chunk": f"\n\n---\n### 🔗 관련 문서\n{links}"})

                    # 6) SSE 종료 시그널
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()

                except Exception as e:
                    self._send_sse({"error": str(e)})

        # (역할) POST 요청 라우팅: /feedback 처리
        def do_POST(self):
            # -----------------------------
            # (C) /feedback 저장
            # - 질문/답변을 하나의 텍스트로 만들고 임베딩해서 DB에 저장
            # -----------------------------
            if self.path == "/feedback":
                try:
                    data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                    self._save_feedback(data["query"], data["answer"])
                    self._send_json(200, {"status": "success"})
                except Exception as e:
                    self._send_json(500, {"error": str(e)})

        # (역할) 사용자 피드백을 임베딩 후 DB에 저장
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

    return RAGHandler