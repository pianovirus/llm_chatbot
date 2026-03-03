import os
import io
import json
import argparse
import mysql.connector
import fitz  # PyMuPDF
import re
import time
from PIL import Image
import easyocr
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 환경 변수 로드 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, "..", "backend", ".env")

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"📡 설정 로드 완료: {dotenv_path}")
else:
    load_dotenv()
    print("⚠️ 지정된 경로에 .env가 없어 기본 경로에서 로드를 시도합니다.")

class RAGIngestionAgent:
    def __init__(self):
        print("🤖 에이전트 모델 로딩 중 (Offline 체크 포함)...")
        self.embed_model = SentenceTransformer('BAAI/bge-m3')
        
        model_id = "Salesforce/blip-image-captioning-base"
        try:
            self.caption_processor = BlipProcessor.from_pretrained(model_id, use_fast=True, local_files_only=True)
            self.caption_model = BlipForConditionalGeneration.from_pretrained(model_id, local_files_only=True)
            print("✅ BLIP 모델을 로컬 캐시에서 로드했습니다.")
        except Exception:
            print("🌐 로컬 캐시가 없어 온라인 연결을 시도합니다...")
            self.caption_processor = BlipProcessor.from_pretrained(model_id, use_fast=True)
            self.caption_model = BlipForConditionalGeneration.from_pretrained(model_id)

        self.reader = easyocr.Reader(['ko', 'en'])
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,       
            chunk_overlap=100,     
            length_function=len,
            separators=["\n\n", "\n", ". ", "!", "?", " ", ""]
        )
        print("✅ 모델 및 청킹 설정 완료")

    def _get_db_conn(self):
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT", "3306")
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASSWORD", "")
        database = os.getenv("DB_NAME")

        return mysql.connector.connect(
            host=host, port=port, user=user, password=password,
            database=database, auth_plugin='mysql_native_password'
        )

    def _clean_text(self, text):
        if not text: return ""
        text = "".join(ch for ch in text if ch.isprintable())
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^가-힣a-zA-Z0-9\s.,?!\'\"()\-\[\]]', '', text)
        return text.strip() if len(text.strip()) >= 10 else ""

    def _analyze_image(self, image_bytes):
        try:
            raw_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            inputs = self.caption_processor(raw_image, return_tensors="pt")
            out = self.caption_model.generate(**inputs)
            caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
            ocr_text = " ".join(self.reader.readtext(image_bytes, detail=0))
            if not re.search(r'[가-힣a-zA-Z]{2,}', ocr_text):
                return f"[이미지 요약]: {caption}"
            return f"[이미지 요약]: {caption} [OCR 추출]: {ocr_text}"
        except: return ""

    def _save_to_db(self, cursor, c_type, content, f_name, page):
        """💡 수정한 부분: source_url에 전체 경로 대신 f_name(파일명)만 저장"""
        if not content or len(content) < 15: return
        
        vector = self.embed_model.encode(content).tolist()
        vector_str = "[" + ",".join(map(str, vector)) + "]"
        
        metadata = json.dumps({
            "page": page, 
            "ingested_at": time.strftime('%Y-%m-%d %H:%M:%S')
        }, ensure_ascii=False)
        
        table_name = os.getenv('DB_TABLE_NAME')
        
        # 💡 source_url 컬럼에 f_name(파일명)을 그대로 넣습니다.
        sql = f"""
            INSERT INTO {table_name} 
            (content_type, original_content, embedding, title, source_url, metadata) 
            VALUES (%s, %s, STRING_TO_VECTOR(%s), %s, %s, %s)
        """
        cursor.execute(sql, (c_type, content, vector_str, f_name, f_name, metadata))

    def ingest_file(self, file_path):
        if not file_path.lower().endswith('.pdf'): return
        print(f"🚀 분석 시작: {file_path}")
        doc = fitz.open(file_path)
        file_name = os.path.basename(file_path)
        
        try:
            conn = self._get_db_conn()
            cursor = conn.cursor()
            all_text_fragments = []
            
            for page_num, page in enumerate(doc):
                curr_pg = page_num + 1
                blocks = page.get_text("blocks")
                page_content = " ".join([b[4] for b in blocks if b[4]])
                
                cleaned = self._clean_text(page_content)
                if cleaned: all_text_fragments.append(cleaned)
                
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    img_data = doc.extract_image(xref)["image"]
                    description = self._analyze_image(img_data)
                    if description: 
                        # 💡 함수 호출 인자 수정 (source_url 삭제)
                        self._save_to_db(cursor, 'image', description, file_name, f"Page {curr_pg}")
                
                print(f"   ㄴ {curr_pg}페이지 스캔 완료")
            
            full_text = " ".join(all_text_fragments)
            if len(full_text) > 50:
                chunks = self.text_splitter.split_text(full_text)
                for i, chunk in enumerate(chunks):
                    # 💡 함수 호출 인자 수정 (source_url 삭제)
                    self._save_to_db(cursor, 'text', chunk, file_name, f"Chunk-{i+1}")
            
            conn.commit()
            print(f"✅ 데이터 저장 성공: {file_name}")
            cursor.close(); conn.close()
        except Exception as e:
            print(f"❌ 에러 발생 ({file_name}): {e}")
        finally:
            doc.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    args = parser.parse_args()
    
    agent = RAGIngestionAgent()
    source_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sources")
    
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
        return

    if args.file:
        target_path = os.path.join(source_dir, args.file)
        if os.path.exists(target_path):
            agent.ingest_file(target_path)
    else:
        files = [f for f in os.listdir(source_dir) if f.lower().endswith(".pdf")]
        for f in files:
            agent.ingest_file(os.path.join(source_dir, f))

if __name__ == "__main__":
    main()