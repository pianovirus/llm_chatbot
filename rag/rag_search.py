# rag_search.py
import os
import json
import numpy as np
import mysql.connector
import warnings
import argparse
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# 1. SSL 관련 지저분한 경고 메시지 무시
warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')
load_dotenv()

class RAGSearcher:
    def __init__(self):
        print("🔍 검색 엔진 로딩 중 (BGE-M3)...")
        # 임베딩 모델 로드
        self.embed_model = SentenceTransformer('BAAI/bge-m3')
        print("✅ 검색 준비 완료!")

    def _get_db_conn(self):
        return mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME")
        )

    def search(self, user_query, top_k=3):
        # 질문을 벡터로 변환
        query_vec = self.embed_model.encode(user_query)
        
        conn = self._get_db_conn()
        cursor = conn.cursor(dictionary=True)

        try:
            # MySQL 9.x에서 벡터를 문자열로 가져오는 함수 사용
            sql = f"SELECT id, content_type, original_content, metadata, VECTOR_TO_STRING(embedding) as vector_str FROM {os.getenv('DB_TABLE_NAME')}"
            cursor.execute(sql)
            rows = cursor.fetchall()

            if not rows:
                return []

            results = []
            for row in rows:
                if not row['vector_str']:
                    continue
                
                # DB의 벡터 문자열을 numpy 배열로 변환
                db_vec = np.array(json.loads(row['vector_str']))
                
                # 유클리드 거리(L2) 계산
                distance = np.linalg.norm(query_vec - db_vec)
                
                row['distance'] = float(distance)
                results.append(row)

            # 거리가 짧은 순(유사한 순)으로 정렬
            results.sort(key=lambda x: x['distance'])
            return results[:top_k]

        except Exception as e:
            print(f"❌ 검색 도중 오류 발생: {e}")
            return []
        finally:
            cursor.close()
            conn.close()

# 이 파일이 직접 실행될 때만 아래 로직이 작동합니다.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Vector Search Tool")
    parser.add_argument("--query", type=str, required=True, help="검색할 질문")
    parser.add_argument("--top", type=int, default=5, help="출력할 결과 개수")
    args = parser.parse_args()

    # 검색기 인스턴스 생성 및 실행
    searcher = RAGSearcher()
    results = searcher.search(args.query, top_k=args.top)

    print(f"\n#️⃣ '{args.query}' 검색 결과:\n" + "="*50)
    
    if not results:
        print("데이터가 없습니다. DB에 데이터가 있는지 확인하세요.")
    else:
        for i, res in enumerate(results, 1):
            print(f"[{i}] 거리(유사도): {res['distance']:.4f}")
            # 전체 내용을 출력하도록 수정했습니다.
            print(f"📝 내용:\n{res['original_content']}")
            print("-" * 50)