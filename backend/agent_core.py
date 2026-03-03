import os

# 🚨 Mac 환경에서 Segmentation Fault 방지
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import numpy as np
import mysql.connector
import warnings
import ssl
import urllib.parse
import unicodedata
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

from langchain_core.runnables import RunnableParallel, RunnableLambda

# 1. 환경 변수 로드
load_dotenv()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError: pass
else: ssl._create_default_https_context = _create_unverified_https_context

from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
embedding_lock = threading.Lock()

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

chat_histories = {}

# 3. 모델 로드 (추론을 위해 Timeout을 넉넉히 설정)
print(f"⏳ [System] 모델 로드 중... ({MODEL_NAME})")
embed_model = SentenceTransformer('BAAI/bge-m3')
llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0, timeout=180)
print(f"✅ [System] 로드 완료.")

def get_internal_context(query: str):
    """DB 지식 검색 (추론을 위해 재료를 충분히 확보)"""
    with embedding_lock:
        query_vec = embed_model.encode(query)
    
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        sql = f"SELECT original_content, title, source_url, VECTOR_TO_STRING(embedding) as vector_str FROM {TABLE_NAME}"
        cursor.execute(sql)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            if not row['vector_str']: continue
            db_vec = np.array(json.loads(row['vector_str']))
            distance = np.linalg.norm(query_vec - db_vec)
            
            # 지식 단서를 충분히 가져오기 위해 2.6까지 허용
            if distance < 2.6:
                file_name = row['source_url']
                normalized = unicodedata.normalize('NFC', file_name)
                encoded = urllib.parse.quote(normalized)
                results.append({
                    "distance": distance, "content": row['original_content'], "title": row['title'], "url": f"{BASE_DOCS_URL}{encoded}"
                })
        results.sort(key=lambda x: x['distance'])
        return results[:3] 
    except: return []
    finally:
        if conn and conn.is_connected(): conn.close()

# --- HTTP 핸들러 ---

class RAGHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def do_GET(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path != "/search": return
        params = parse_qs(parsed_path.query)
        query = params.get('query', [''])[0]
        session_id = params.get('sessionId', ['default_session'])[0]
        if not query: return

        self.send_response(200)
        self.send_header('Content-type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'close')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        try:
            if session_id not in chat_histories: chat_histories[session_id] = []
            history_text = "\n".join([f"{h['role']}: {h['content'][:100]}..." for h in chat_histories[session_id][-2:]])

            self._send_sse({"status": "🔍 매뉴얼 상세 분석 중..."})
            search_results = get_internal_context(query)
            
            full_reply = ""
            
            if search_results:
                # 데이터를 명확히 구분하여 전달
                context_combined = "\n".join([f"[매뉴얼 데이터 {i+1}]:\n{r['content']}\n" for i, r in enumerate(search_results)])
                
                # 🚨 [강력한 지시] 불성실한 답변 방지 및 직접 서술 강제
                prompt = (
                    f"당신은 기술 지원 센터의 시니어 엔지니어입니다. 제공된 [지식 데이터]를 분석하여 사용자의 질문에 성실히 답변하세요.\n\n"
                    f"### [지식 데이터]\n{context_combined}\n\n"
                    f"### [사용자 질문]\n{query}\n\n"
                    f"### [응답 규칙 - 필독]\n"
                    f"1. **직접 서술**: '참조 문서를 보세요' 혹은 '데이터에 정보가 있습니다' 같은 말은 절대 하지 마세요. 데이터에 있는 구체적인 설치 방법, 설정값, 단계를 직접 본문에 풀어서 쓰세요.\n"
                    f"2. **친절한 가이드**: 사용자가 문서를 직접 찾아볼 필요가 없도록, 당신이 읽어주는 것처럼 상세하게 단계별(1., 2., 3.)로 설명하세요.\n"
                    f"3. **추론 및 응용**: 질문에 대한 완벽한 문장이 없더라도, 데이터의 기술적 단서를 조합하여 '게이트웨이 설치를 위해서는 ~한 절차가 필요할 것으로 판단됩니다'와 같이 능동적으로 추론하세요.\n"
                    f"4. **무관한 질문 차단**: 질문이 데이터와 0% 확률로 무관하면(예: 날씨) 데이터를 무시하고 일반 답변만 하세요.\n"
                    f"5. **구조**: 반드시 [## 📢 조치 안내] -> [### 🛠️ 상세 설치 절차] -> [### 💡 핵심 주의사항] 순서로 작성하세요."
                )
                
                for chunk in llm.stream(prompt):
                    if chunk: 
                        self._send_sse({"chunk": chunk}); full_reply += chunk
                
                # 답변에 실질적인 정보(설치, 방법, 연결 등)가 포함된 경우에만 링크 노출
                is_related = any(kw in full_reply for kw in ["설치", "방법", "절차", "연결", "설정"])
                if is_related:
                    source_links = list(set([f"- [{r['title']}]({r['url']})" for r in search_results]))
                    source_section = "\n\n---\n### 🔗 참고 문서\n" + "\n".join(source_links)
                    self._send_sse({"chunk": source_section}); full_reply += source_section
            else:
                for chunk in llm.stream(f"친절한 기술 상담원으로서 답변하세요: {query}"):
                    if chunk: 
                        self._send_sse({"chunk": chunk}); full_reply += chunk

            chat_histories[session_id].append({"role": "user", "content": query})
            chat_histories[session_id].append({"role": "assistant", "content": full_reply})
            if len(chat_histories[session_id]) > 4: chat_histories[session_id].pop(0)

            self._send_done()

        except Exception as e:
            self._send_sse({"error": str(e)}); self._send_done()

    def _send_sse(self, data):
        try:
            self.wfile.write(f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode('utf-8'))
            self.wfile.flush()
        except: pass

    def _send_done(self):
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except: pass

if __name__ == "__main__":
    server = ThreadingHTTPServer(('localhost', 8000), RAGHandler)
    print(f"📡 High-Fidelity Reasoning Agent 가동: http://localhost:8000")
    server.serve_forever()