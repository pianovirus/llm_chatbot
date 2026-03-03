import os
import mysql.connector
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# 1. 환경 변수 로드
load_dotenv()

try:
    # 2. 모델 로드 및 벡터 생성 (1024차원)
    print("--- 1. 벡터 생성 중 ---")
    model = SentenceTransformer('BAAI/bge-m3')
    content = "사내 보안 규정에 대한 테스트 문장입니다."
    vector = model.encode(content).tolist()
    
    # 리스트를 MySQL이 인식 가능한 "[0.1,0.2...]" 문자열로 변환
    vector_str = "[" + ",".join(map(str, vector)) + "]"
    
    # 3. 메타데이터 준비
    metadata = {"file_name": "test.pdf", "page": 1}

    # 4. DB 연결
    print("--- 2. MySQL 접속 중 ---")
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database="llm_chatbot"  # 직접 명시하거나 .env 사용
    )
    cursor = conn.cursor()

    # 5. 인서트 쿼리 (MySQL 9.x 최적화 문법)
    print("--- 3. 데이터 인서트 시도 ---")
    sql = """
    INSERT INTO rag_vectors (content_type, original_content, embedding, metadata) 
    VALUES (%s, %s, STRING_TO_VECTOR(%s), %s)
    """
    
    # 바인딩 데이터를 튜플로 전달
    cursor.execute(sql, (
        'text', 
        content, 
        vector_str, 
        json.dumps(metadata)
    ))
    
    conn.commit()
    print(f"✅ [성공] ID {cursor.lastrowid}번으로 데이터가 저장되었습니다!")

except mysql.connector.Error as err:
    print(f"❌ [DB 에러] {err}")
except Exception as e:
    print(f"❌ [기타 에러] {e}")
finally:
    if 'conn' in locals() and conn.is_connected():
        cursor.close()
        conn.close()
        print("--- 테스트 종료 ---")