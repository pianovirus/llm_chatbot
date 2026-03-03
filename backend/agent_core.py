import os
import json
import numpy as np
import mysql.connector
import warnings
import ssl
import urllib.parse
import unicodedata
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

# 1. 환경 변수 로드
load_dotenv()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError: pass
else: ssl._create_default_https_context = _create_unverified_https_context

from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer

# 경고 무시
warnings.filterwarnings("ignore")

# 2. 설정값
MODEL_NAME = os.getenv("LLM_MODEL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD", ""), 
    "database": os.getenv("DB_NAME")
}
TABLE_NAME = os.getenv("DB_TABLE_NAME")
BASE_DOCS_URL = os.getenv("BASE_DOCS_URL", "http://localhost:3000/sources/")

# 💡 [핵심] 세션별 대화 기록을 저장할 메모리 저장소
chat_histories = {}

# 3. 모델 로드
print(f"⏳ [System] 모델 로드 중... ({MODEL_NAME})")
embed_model = SentenceTransformer('BAAI/bge-m3')
llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0)
print(f"✅ [System] 로드 완료.")

def classify_intent(query: str, history_text: str = "") -> str:
    """의도 분석: 맥락을 포함하여 CHAT 또는 INFO 판별"""
    try:
        prompt = (
            f"이전 대화 맥락:\n{history_text}\n\n"
            f"사용자의 마지막 질문: '{query}'\n"
            f"이 질문의 의도가 정보 검색(INFO)인지 일반 대화(CHAT)인지 판단하세요. "
            f"오직 'CHAT' 또는 'INFO'라고만 한 단어로 답변하세요."
        )
        response = llm.invoke(prompt).strip().upper()
        return "INFO" if "INFO" in response else "CHAT"
    except:
        return "CHAT"

def get_internal_context(query: str):
    """DB 지식 검색 및 동적 URL 생성 (NFC 및 URL 인코딩 적용)"""
    query_vec = embed_model.encode(query)
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    try:
        sql = f"SELECT original_content, title, source_url, VECTOR_TO_STRING(embedding) as vector_str FROM {TABLE_NAME}"
        cursor.execute(sql)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            if not row['vector_str']: continue
            db_vec = np.array(json.loads(row['vector_str']))
            distance = np.linalg.norm(query_vec - db_vec)
            
            if distance < 2.8:
                file_name = row['source_url']
                normalized_file_name = unicodedata.normalize('NFC', file_name)
                encoded_file_name = urllib.parse.quote(normalized_file_name)
                full_url = f"{BASE_DOCS_URL}{encoded_file_name}" if file_name else ""
                
                results.append({
                    "distance": distance,
                    "content": row['original_content'],
                    "title": row['title'],
                    "url": full_url
                })
        results.sort(key=lambda x: x['distance'])
        return results[:5] 
    finally:
        cursor.close(); conn.close()

# --- HTTP 핸들러 ---

class RAGHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def do_GET(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path != "/search": return
        
        params = parse_qs(parsed_path.query)
        query = params.get('query', [''])[0]
        # 💡 프론트엔드에서 보낸 sessionId 수신 (없으면 기본값)
        session_id = params.get('sessionId', ['default_session'])[0]
        
        if not query: return

        self.send_response(200)
        self.send_header('Content-type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Connection', 'close')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        print(f"\n📡 [요청 수신 | Session: {session_id}]: {query}")
        
        try:
            # 1. 💡 해당 세션의 이전 대화 기록 불러오기
            if session_id not in chat_histories:
                chat_histories[session_id] = []
            
            history = chat_histories[session_id]
            # 최근 4개의 메시지만 맥락으로 사용 (토큰 절약 및 성능 최적화)
            history_text = "\n".join([f"{h['role']}: {h['content']}" for h in history[-4:]])

            self._send_sse({"status": "🧠 맥락 분석 및 의도 파악 중..."})
            intent = classify_intent(query, history_text)
            
            full_assistant_reply = "" # 대화 기록 저장을 위해 답변을 모아둠

            if intent == "CHAT":
                self._send_sse({"status": "💬 지원 도우미 연결"})
                prompt = f"당신은 기술 엔지니어입니다. 이전 대화를 참고하여 답변하세요.\n\n[이전 대화]\n{history_text}\n\n사용자: {query}"
                for chunk in llm.stream(prompt):
                    if chunk: 
                        self._send_sse({"chunk": chunk})
                        full_assistant_reply += chunk
                
            else:
                self._send_sse({"status": "🔍 지식 데이터 분석 중..."})
                search_results = get_internal_context(query)
                
                if search_results:
                    self._send_sse({"status": "📋 가이드 생성 및 출처 확인 중..."})
                    context_text = "\n".join([f"- {r['content']}" for r in search_results])
                    
                    source_links = []
                    seen_urls = set()
                    for r in search_results:
                        if r['url'] and r['url'] not in seen_urls:
                            source_links.append(f"- [{r['title']}]({r['url']})")
                            seen_urls.add(r['url'])
                    
                    source_section = "\n\n---\n### 🔗 참고 문서\n" + "\n".join(source_links) if source_links else ""
                    
                    prompt = (
                        f"당신은 숙련된 시니어 기술 지원 엔지니어입니다. 아래 맥락과 지식 데이터를 바탕으로 답변하세요.\n\n"
                        f"### [이전 대화 맥락]\n{history_text}\n\n"
                        f"### [지식 데이터]\n{context_text}\n\n"
                        f"### [응답 규칙]\n"
                        f"1. **구조**: [## 📢 조치 안내] -> [### 🛠️ 해결 절차] -> [### 💡 추가 제언] 순서로 작성하세요.\n"
                        f"2. **줄바꿈 최소화**: 문단 사이에는 빈 줄을 '단 하나'만 사용하세요.\n"
                        f"3. **완결성**: 정중하고 간결하게 마무리하세요.\n\n"
                        f"### [사용자 질문]\n{query}"
                    )
                    
                    for chunk in llm.stream(prompt):
                        if chunk: 
                            self._send_sse({"chunk": chunk})
                            full_assistant_reply += chunk
                    
                    if source_section:
                        self._send_sse({"chunk": source_section})
                        full_assistant_reply += source_section
                        
                else:
                    self._send_sse({"status": "🌐 일반 지식 가이드 생성 중..."})
                    prompt = f"맥락: {history_text}\n질문: {query}\n기술 엔지니어로서 핵심 위주로 답변하세요."
                    for chunk in llm.stream(prompt):
                        if chunk: 
                            self._send_sse({"chunk": chunk})
                            full_assistant_reply += chunk

            # 2. 💡 이번 대화 내용을 메모리에 저장
            chat_histories[session_id].append({"role": "user", "content": query})
            chat_histories[session_id].append({"role": "assistant", "content": full_assistant_reply})

            self._send_done()
            print(f"✅ 처리 완료 (Intent: {intent})")

        except Exception as e:
            print(f"❌ 에러: {e}")
            self._send_sse({"error": str(e)})
            try: self._send_done()
            except: pass

    def _send_sse(self, data):
        payload = f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        self.wfile.write(payload.encode('utf-8'))
        self.wfile.flush()

    def _send_done(self):
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

if __name__ == "__main__":
    server = HTTPServer(('localhost', 8000), RAGHandler)
    print(f"📡 Intelligent Agent Core 가동 (Memory Session 기반): http://localhost:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 서버를 종료합니다.")
        server.server_close()