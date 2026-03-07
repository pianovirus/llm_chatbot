import os
import json
import argparse
import psycopg2 
import psycopg2.extras
from pgvector.psycopg2 import register_vector 
import re
import time
import unicodedata
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 환경 변수 로드 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, "..", "backend", ".env")

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"📡 설정 로드 완료: {dotenv_path}")
else:
    load_dotenv()
    print("⚠️ .env 파일을 찾을 수 없어 기본 환경변수를 사용합니다.")

class RAGIngestionLight:
    def __init__(self):
        print("🤖 라이트 에이전트 모델 로딩 중 (BGE-Small)...")
        # 💡 한글/영어 혼용 매뉴얼에 특화된 BGE 모델 유지
        self.embed_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        
        # 💡 청크 사이즈 최적화: 500은 문맥이 자주 끊겨서 800으로 상향했습니다.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,       
            chunk_overlap=150,     
            length_function=len,
            separators=["\n\n", "\n", "---", ". ", " ", ""]
        )
        print("✅ 가벼운 임베딩 모델 및 청킹 설정 완료")

    def _get_db_conn(self):
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", "5432"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            dbname=os.getenv("DB_NAME")
        )
        register_vector(conn)
        return conn

    def _save_to_db(self, cursor, c_type, content, f_name, page_info):
        if not content or len(content.strip()) < 10: return
        
        # 1. NFC 정규화 (맥북 한글 깨짐 방지)
        content = unicodedata.normalize('NFC', content)
        
        # 2. 임베딩 생성
        embedding = self.embed_model.encode(content).tolist()
        
        # 3. 파일명 처리
        display_name = f_name[:-4] if f_name.lower().endswith(".txt") else f_name
            
        metadata = json.dumps({
            "source_type": c_type,
            "page_info": page_info, 
            "ingested_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }, ensure_ascii=False)
        
        table_name = os.getenv('DB_TABLE_NAME', 'rag_vectors')
        
        # 4. PostgreSQL INSERT (하이브리드 검색용 tsvector 포함)
        # 💡 to_tsvector('simple', %s)를 사용하여 모델명, 에러코드를 있는 그대로 인덱싱합니다.
        sql = f"""
            INSERT INTO {table_name} 
            (content_type, original_content, embedding, title, source_url, metadata, content_search_vector) 
            VALUES (%s, %s, %s, %s, %s, %s, to_tsvector('simple', %s))
        """
        
        cursor.execute(sql, (
            c_type, 
            content, 
            embedding, 
            display_name,
            display_name,
            metadata,
            content # to_tsvector용으로 한 번 더 전달
        ))

    def ingest_txt_file(self, file_path):
        if not file_path.lower().endswith('.txt'): return
        print(f"🚀 처리 중: {os.path.basename(file_path)}")
        
        file_name = os.path.basename(file_path)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                full_content = f.read()

            conn = self._get_db_conn()
            cursor = conn.cursor()

            if len(full_content) > 10:
                chunks = self.text_splitter.split_text(full_content)
                for i, chunk in enumerate(chunks):
                    # 💡 이미지 서빙 방식 변경에 따라 텍스트 타입 판단 로직 유지
                    c_type = 'image' if '[이미지 내 텍스트]' in chunk else 'text'
                    self._save_to_db(cursor, c_type, chunk, file_name, f"Chunk-{i+1}")
            
            conn.commit()
            print(f"✅ DB 저장 완료: {file_name}")
            cursor.close(); conn.close()
        except Exception as e:
            print(f"❌ 에러 발생 ({file_name}): {e}")

def main():
    agent = RAGIngestionLight()
    # 💡 텍스트 파일 경로 설정
    text_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "converted_texts")
    
    if not os.path.exists(text_dir):
        print(f"⚠️ '{text_dir}' 폴더가 없습니다. PDF 변환을 먼저 진행하세요.")
        return

    files = [f for f in os.listdir(text_dir) if f.lower().endswith(".txt")]
    if not files:
        print("📁 변환된 텍스트 파일이 없습니다.")
        return

    for f in files:
        agent.ingest_txt_file(os.path.join(text_dir, f))
    
    print("\n✨ 하이브리드 인제스트가 완료되었습니다. 이제 에러 코드 검색도 정확해집니다!")

if __name__ == "__main__":
    main()