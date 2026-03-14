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
            # -----------------------------
            if parsed.path == "/search":
                query = parse_qs(parsed.query).get("query", [""])[0].strip()

                self.send_response(200)
                self.send_header("Content-type", "text/event-stream; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                try:
                    # [단계 1] 시작 알림 (Gemini 요약 시작)
                    self._send_sse({"status": "🧠 질문의 의도를 분석하고 있습니다..."})

                    state = {"original_query": query}

                    # [단계 2] 파이프라인 스트리밍
                    # stream_mode="updates"는 노드가 '종료'될 때 이벤트를 줍니다.
                    for update in rag_pipeline.stream(state, stream_mode="updates"):
                        for node, out in update.items():
                            
                            if node == "rewrite_query":
                                # 분석 노드가 끝남 -> 즉시 검색 노드가 시작될 것임을 알림
                                eq = out.get("effective_query", query)
                                self._send_sse({"status": f"🔍 분석 완료! 키워드 [{eq}](으)로 지식 베이스를 탐색 중입니다..."})
                                
                            elif node == "retrieve_context":
                                # 검색 노드가 끝남 -> 다음인 정리(prepare) 단계 예고
                                results = out.get("search_results", [])
                                count = len(results)
                                self._send_sse({"status": f"✅ 정보 {count}건을 찾았습니다. 답변 구성을 위해 내용을 정리합니다..."})
                                
                            elif node == "prepare_prompt":
                                # 프롬프트 준비 끝 -> 실제 LLM 답변 생성 예고
                                self._send_sse({"status": "✍️ 정리가 완료되었습니다. 답변 작성을 시작합니다..."})

                            # 상태 업데이트는 항상 로그 전송 후에 수행
                            state.update(out)

                    # [단계 3] 최종 LLM 답변 생성
                    # stream 루프가 끝나고 실제 첫 글자가 나오기 전까지의 공백을 메워줍니다.
                    self._send_sse({"status": "🤖 답변을 생성 중입니다..."})

                    for chunk in llm_chain.stream(state["prompt"]):
                        if chunk:
                            self._send_sse({"chunk": chunk})

                    # [단계 4] 관련 문서 링크 추가
                    if state.get("doc_links"):
                        links = "\n".join([f"- [{t}]({u})" for t, u in state["doc_links"].items()])
                        self._send_sse({"chunk": f"\n\n---\n### 🔗 관련 문서\n{links}"})

                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()

                except Exception as e:
                    self._send_sse({"error": str(e)})

        def do_POST(self):
            if self.path == "/feedback":
                try:
                    data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                    self._save_feedback(data["query"], data["answer"])
                    self._send_json(200, {"status": "success"})
                except Exception as e:
                    self._send_json(500, {"error": str(e)})

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