# server.py
# ============================================================
# HTTP Server Handler 모듈
# ------------------------------------------------------------
# 이 파일은 "HTTP 요청을 실제로 받아서 처리하는 라우터" 역할을 합니다.
#
# 핵심 역할:
# 1) /files/<filename>  -> 로컬 저장 파일 서빙
# 2) /search?query=...  -> RAG 검색 + LLM 답변 스트리밍(SSE)
# 3) /feedback          -> 사용자가 검증한 Q/A를 DB에 저장
#
# 설계 의도:
# - 서버 실행 자체(agent_core.py)와 요청 처리 로직(server.py)을 분리
# - 필요한 의존성(파이프라인, 모델, DB 정보, 파일 디렉토리)을
#   바깥(agent_core.py)에서 주입받아 사용
#
# 즉, 이 파일은 단독으로 부팅하는 주체가 아니라
# "요청이 들어왔을 때 어떻게 처리할지"를 담당하는 모듈입니다.
# ============================================================

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
    ========================================================
    Handler Factory 함수
    --------------------------------------------------------
    BaseHTTPRequestHandler를 상속한 RAGHandler 클래스를
    "동적으로 만들어 반환"하는 팩토리 함수입니다.

    왜 클래스 바깥에서 바로 정의하지 않고 factory 형태로 만드나?
    - BaseHTTPRequestHandler는 생성자에 원하는 인자를 자유롭게 넣기 어렵습니다.
    - 그래서 이 함수가 외부 의존성들을 클로저로 캡처한 뒤,
      내부 클래스(RAGHandler)가 그 값들을 그대로 사용하도록 만듭니다.

    주입받는 의존성:
    - storage_dir : /files 요청 시 실제 파일을 읽어올 디렉토리
    - rag_pipeline: LangGraph 기반 RAG 파이프라인
    - llm_chain   : 최종 답변 생성용 LLM 스트리밍 체인
    - embed_model : feedback 저장 시 임베딩 생성 모델
    - db_config   : PostgreSQL 연결 정보
    - table_name  : 벡터/문서 저장 테이블 이름
    - llm_provider: 현재 사용 중인 LLM 공급자 이름
                    (현재 직접 사용하지 않지만 향후 분기/로그용으로 유지 가능)

    반환:
    - BaseHTTPRequestHandler를 상속한 RAGHandler 클래스
    ========================================================
    """

    class RAGHandler(BaseHTTPRequestHandler):
        """
        ====================================================
        실제 HTTP 요청을 처리하는 핸들러 클래스
        ----------------------------------------------------
        이 클래스의 인스턴스는 HTTP 요청 1건마다 생성되어
        GET / POST 요청을 처리합니다.

        주요 메서드:
        - do_GET()  : GET 요청 처리
        - do_POST() : POST 요청 처리
        - _send_json() : 일반 JSON 응답 헬퍼
        - _send_sse()  : SSE(Server-Sent Events) 전송 헬퍼
        - _save_feedback() : 피드백 DB 저장 헬퍼
        ====================================================
        """

        # ----------------------------------------------------
        # HTTP/1.1 사용
        # ----------------------------------------------------
        # keep-alive로 인해 연결이 오래 붙잡히는 문제를 피하기 위해
        # close_connection=True 와 함께 사용합니다.
        # SSE나 브라우저/프론트엔드와의 연결에서 간헐적인 고착을 줄이는 목적입니다.
        protocol_version = "HTTP/1.1"
        close_connection = True

        def _send_json(self, status_code, payload):
            """
            =================================================
            일반 JSON 응답 전송 헬퍼
            -------------------------------------------------
            REST 스타일의 일반 응답을 보낼 때 사용합니다.

            사용 예:
            - /feedback 성공 응답
            - /feedback 에러 응답

            입력:
            - status_code: HTTP 상태 코드 (예: 200, 500)
            - payload    : JSON으로 직렬화할 파이썬 dict

            헤더 설정:
            - Content-type: application/json
            - CORS 허용
            - Connection: close 로 연결 고착 방지
            =================================================
            """
            self.send_response(status_code)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "close")
            self.end_headers()

            self.wfile.write(
                json.dumps(payload, ensure_ascii=False).encode("utf-8")
            )

        def _send_sse(self, data):
            """
            =================================================
            SSE(Server-Sent Events) 이벤트 1건 전송 헬퍼
            -------------------------------------------------
            /search 응답은 일반 JSON이 아니라 SSE 스트림입니다.
            즉, 응답을 한 번에 끝내는 것이 아니라
            "조금씩 중간 진행상황과 답변 조각(chunk)"을 흘려보냅니다.

            입력:
            - data: dict 형태 데이터
                    예) {"status": "..."}
                        {"chunk": "..."}
                        {"error": "..."}

            전송 형식:
            - SSE는 "data: ...\\n\\n" 형식으로 내려가야 함
            - flush()를 호출해야 브라우저/클라이언트가 즉시 받음
            =================================================
            """
            self.wfile.write(
                f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")
            )
            self.wfile.flush()

        def do_GET(self):
            """
            =================================================
            GET 요청 라우팅 메서드
            -------------------------------------------------
            처리 대상:
            1) /files/<filename>
               - storage_dir 안의 파일을 그대로 내려줌
            2) /search?query=...
               - RAG 파이프라인 실행
               - 단계별 상태 메시지 SSE 전송
               - 최종 답변 chunk 스트리밍 전송

            동작 흐름:
            - 먼저 URL 파싱
            - 파일 요청인지 검색 요청인지 분기
            - 해당하는 로직 수행
            =================================================
            """
            parsed = urlparse(self.path)

            # =================================================
            # (A) /files/<filename>
            # -------------------------------------------------
            # 목적:
            # - 프론트엔드나 답변 링크에서 파일 접근 시 사용
            # - storage_dir 내부 파일을 바이너리 그대로 반환
            #
            # 예:
            #   GET /files/manual.pdf
            #
            # 주의:
            # - URL 인코딩된 파일명(%20 등)을 unquote로 복원
            # - mimetypes로 Content-Type 추정
            # =================================================
            if parsed.path.startswith("/files/"):
                full_path = os.path.join(storage_dir, unquote(parsed.path[7:]))

                if os.path.exists(full_path):
                    self.send_response(200)
                    self.send_header(
                        "Content-type",
                        mimetypes.guess_type(full_path)[0] or "application/octet-stream",
                    )
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Connection", "close")
                    self.end_headers()

                    with open(full_path, "rb") as f:
                        self.wfile.write(f.read())
                else:
                    self.send_error(404)

                return

            # =================================================
            # (B) /search?query=...
            # -------------------------------------------------
            # 목적:
            # - 사용자의 질문을 받아 RAG 검색 + 답변 생성을 수행
            #
            # 처리 단계:
            # 1) query 추출
            # 2) SSE 응답 시작
            # 3) rag_pipeline.stream(...) 으로 중간 단계 진행
            # 4) 최종 prompt를 기반으로 llm_chain.stream(...) 실행
            # 5) 관련 문서 링크 추가
            # 6) [DONE] 전송
            #
            # 왜 SSE를 쓰나?
            # - 사용자가 기다리는 동안 "지금 무엇을 하는지" 보여줄 수 있음
            # - 답변도 chunk 단위로 바로바로 표시 가능
            # =================================================
            if parsed.path == "/search":
                query = parse_qs(parsed.query).get("query", [""])[0].strip()

                # SSE 응답 시작
                self.send_response(200)
                self.send_header("Content-type", "text/event-stream; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Connection", "close")
                self.end_headers()

                try:
                    # --------------------------------------------
                    # [단계 1] 사용자의 질문 의도 분석 시작 안내
                    # --------------------------------------------
                    self._send_sse({"status": "🧠 질문의 의도를 분석하고 있습니다..."})

                    # 파이프라인 실행용 초기 상태
                    # original_query는 항상 최초 사용자 입력을 보존
                    state = {"original_query": query}

                    # --------------------------------------------
                    # [단계 2] LangGraph 파이프라인 단계별 실행
                    # --------------------------------------------
                    # stream_mode="updates"는 각 노드가 끝날 때
                    # 해당 노드의 출력값을 이벤트처럼 넘겨줍니다.
                    for update in rag_pipeline.stream(state, stream_mode="updates"):
                        for node, out in update.items():

                            # ------------------------------------
                            # rewrite_query 노드 완료
                            # ------------------------------------
                            # 사용자의 질문을 더 검색 친화적으로 바꾸는 단계
                            # 예: 질문을 핵심 키워드 중심으로 재작성
                            if node == "rewrite_query":
                                eq = out.get("effective_query", query)
                                self._send_sse({
                                    "status": f"🔍 분석 완료! 키워드 [{eq}](으)로 지식 베이스를 탐색 중입니다..."
                                })

                            # ------------------------------------
                            # retrieve_context 노드 완료
                            # ------------------------------------
                            # 실제 검색 결과를 가져온 후,
                            # 몇 건 찾았는지 사용자에게 안내
                            elif node == "retrieve_context":
                                results = out.get("search_results", [])
                                count = len(results)
                                self._send_sse({
                                    "status": f"✅ 정보 {count}건을 찾았습니다. 답변 구성을 위해 내용을 정리합니다..."
                                })

                            # ------------------------------------
                            # prepare_prompt 노드 완료
                            # ------------------------------------
                            # 검색 결과와 질문을 합쳐 LLM 입력용 prompt를
                            # 최종 조립한 시점
                            elif node == "prepare_prompt":
                                self._send_sse({
                                    "status": "✍️ 정리가 완료되었습니다. 답변 작성을 시작합니다..."
                                })

                            # ------------------------------------
                            # state 갱신
                            # ------------------------------------
                            # 각 노드의 출력(out)을 전체 state에 누적
                            # 이후 노드/최종 답변 생성 단계에서 활용
                            state.update(out)

                    # --------------------------------------------
                    # [단계 3] 최종 LLM 답변 생성 시작
                    # --------------------------------------------
                    # pipeline 종료 후 실제 답변 생성 직전 안내 메시지
                    self._send_sse({"status": "🤖 답변을 생성 중입니다..."})

                    # llm_chain.stream(...)은 최종 prompt를 받아
                    # 답변을 chunk 단위로 흘려줍니다.
                    for chunk in llm_chain.stream(state["prompt"]):
                        if chunk:
                            self._send_sse({"chunk": chunk})

                    # --------------------------------------------
                    # [단계 4] 관련 문서 링크 부착
                    # --------------------------------------------
                    # 파이프라인 state에 doc_links가 있으면
                    # 답변 하단에 Markdown 링크 목록으로 추가
                    #
                    # 형식 예:
                    # - [문서명](url)
                    if state.get("doc_links"):
                        links = "\n".join(
                            [f"- [{title}]({url})" for title, url in state["doc_links"].items()]
                        )
                        self._send_sse({"chunk": f"\n\n---\n### 🔗 관련 문서\n{links}"})

                    # --------------------------------------------
                    # [단계 5] 스트림 종료 신호
                    # --------------------------------------------
                    # 프론트엔드가 "답변 종료"를 인식할 수 있도록
                    # 특수 토큰 [DONE] 전송
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()

                except Exception as e:
                    # 검색/생성 중 에러가 발생해도
                    # SSE 형식으로 에러를 내려 프론트엔드가 처리 가능하게 함
                    self._send_sse({"error": str(e)})

        def do_POST(self):
            """
            =================================================
            POST 요청 라우팅 메서드
            -------------------------------------------------
            현재는 /feedback 엔드포인트만 처리합니다.

            목적:
            - 사용자가 검증한 질문/답변 쌍을 벡터 DB에 저장
            - 나중에 검색/학습/검증 데이터로 활용 가능

            입력 JSON 예:
            {
              "query": "...",
              "answer": "..."
            }

            처리 절차:
            1) 요청 body 읽기
            2) JSON 파싱
            3) _save_feedback() 호출
            4) 성공/실패 JSON 응답 반환
            =================================================
            """
            if self.path == "/feedback":
                try:
                    content_length = int(self.headers.get("Content-Length", 0))
                    raw_body = self.rfile.read(content_length)
                    data = json.loads(raw_body)

                    self._save_feedback(data["query"], data["answer"])
                    self._send_json(200, {"status": "success"})

                except Exception as e:
                    self._send_json(500, {"error": str(e)})

        def _save_feedback(self, q, a):
            """
            =================================================
            피드백 저장 헬퍼
            -------------------------------------------------
            사용자가 "좋은 답변"이라고 판단한 질문/답변 쌍을
            벡터 DB 테이블에 저장합니다.

            입력:
            - q: 질문 문자열
            - a: 답변 문자열

            처리 절차:
            1) 질문/답변을 하나의 텍스트로 결합
            2) embed_model로 임베딩 생성
            3) PostgreSQL 연결
            4) table_name에 INSERT
            5) commit 후 연결 종료

            저장 목적:
            - 향후 검색 품질 개선
            - 검증된 답변 축적
            - 피드백 기반 지식 베이스 확장

            현재 저장 컬럼:
            - content_type     : "text"
            - data_source      : "feedback"
            - original_content : 질문+답변 텍스트
            - embedding        : 벡터
            - title            : "검증된 답변"
            =================================================
            """
            text = f"질문: {q}\n답변: {a}"

            # 질문+답변 전체를 하나의 문서처럼 임베딩
            vec = embed_model.encode(text).tolist()

            conn = psycopg2.connect(**db_config)

            try:
                register_vector(conn)
                cur = conn.cursor()

                cur.execute(
                    f"INSERT INTO {table_name} "
                    f"(content_type, data_source, original_content, embedding, title) "
                    f"VALUES (%s, %s, %s, %s, %s)",
                    ("text", "feedback", text, vec, "검증된 답변"),
                )

                conn.commit()

            finally:
                conn.close()

    return RAGHandler