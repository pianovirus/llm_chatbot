# rag_ingest_light.py
import os
import json
import time
import unicodedata

import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 환경 변수(.env) 로드
# ------------------------------------------------------------
# 이 스크립트는 DB 접속 정보(DB_HOST, DB_USER 등)를 환경 변수에서 읽습니다.
# 우선순위는 다음과 같습니다.
#
# 1) 현재 파일 기준 ../backend/.env 가 있으면 그 파일을 로드
# 2) 없으면 시스템 환경 변수 또는 현재 작업 디렉토리 기준 .env 사용
#
# 목적:
# - 로컬 개발 환경에서 backend/.env 를 재사용하기 위함
# - 배포/운영 환경에서는 OS 환경변수만으로도 동작 가능하게 하기 위함
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, "..", "backend", ".env")

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"📡 설정 로드 완료: {dotenv_path}")
else:
    load_dotenv()
    print("⚠️ .env 파일을 찾을 수 없어 기본 환경변수를 사용합니다.")


class RAGIngestionLight:
    """
    ============================================================
    RAGIngestionLight
    ------------------------------------------------------------
    텍스트 파일(.txt)을 읽어서 다음 과정을 수행하는 경량 인제스트 클래스입니다.

    1) 텍스트 파일 읽기
    2) 긴 본문을 청크로 분할
    3) 각 청크에 대해 임베딩 생성
    4) PostgreSQL + pgvector 테이블에 저장
    5) 하이브리드 검색용 tsvector도 함께 저장

    사용 목적:
    - 변환된 텍스트 파일들을 RAG 검색 대상 DB에 적재
    - 문서 검색 정확도 향상
    - 키워드 검색 + 벡터 검색을 동시에 가능하게 함
    ============================================================
    """

    def __init__(self):
        """
        ========================================================
        생성자
        --------------------------------------------------------
        인제스트에 필요한 핵심 객체를 1회 초기화합니다.

        초기화 대상:
        - SentenceTransformer 임베딩 모델
        - 텍스트 청킹기(RecursiveCharacterTextSplitter)

        왜 여기서 초기화하나?
        - 파일마다 모델을 다시 로드하면 매우 느리기 때문
        - 한 번만 로드하고 모든 파일 처리에 재사용하기 위함
        ========================================================
        """
        print("🤖 라이트 에이전트 모델 로딩 중 (BGE-Small)...")

        # 한글/영어 혼용 문서에도 비교적 안정적으로 동작하는 임베딩 모델
        self.embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

        # 긴 문서를 일정 크기의 청크로 나누기 위한 설정
        # - chunk_size: 한 청크의 최대 길이
        # - chunk_overlap: 이전 청크와 겹치는 길이
        # 겹침을 주는 이유:
        # - 문맥이 청크 경계에서 끊기는 문제를 완화하기 위해서
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", "---", ". ", " ", ""],
        )

        print("✅ 가벼운 임베딩 모델 및 청킹 설정 완료")

    def _get_db_conn(self):
        """
        ========================================================
        DB 연결 생성 함수
        --------------------------------------------------------
        환경 변수에서 PostgreSQL 접속 정보를 읽어서 연결을 생성합니다.

        추가 작업:
        - pgvector 사용을 위해 register_vector(conn) 호출

        반환:
        - psycopg2 connection 객체

        사용 이유:
        - DB 연결 생성 로직을 한 곳에 모아두면 유지보수가 쉬움
        - host/port/user/password/dbname 변경 시 이 함수만 보면 됨
        ========================================================
        """
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", "5432"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            dbname=os.getenv("DB_NAME"),
        )
        register_vector(conn)
        return conn

    def _save_to_db(self, cursor, c_type, content, f_name, page_info):
        """
        ========================================================
        단일 청크 DB 저장 함수
        --------------------------------------------------------
        하나의 텍스트 청크를 DB에 저장합니다.

        입력값:
        - cursor: psycopg2 cursor
        - c_type: content 유형 ("text" 또는 "image")
        - content: 실제 청크 본문
        - f_name: 원본 파일명
        - page_info: 청크 위치 정보 (예: Chunk-1)

        처리 절차:
        1) 너무 짧은 텍스트는 저장 생략
        2) macOS 한글 조합 문제 방지를 위해 NFC 정규화
        3) 임베딩 생성
        4) 메타데이터 JSON 생성
        5) PostgreSQL 테이블에 INSERT
        6) 하이브리드 검색용 content_search_vector도 함께 저장

        주의:
        - 이 함수는 commit을 하지 않습니다.
        - commit은 상위 함수(ingest_txt_file)에서 파일 단위로 수행합니다.
        ========================================================
        """
        # 너무 짧은 텍스트는 검색 가치가 낮고 노이즈가 되므로 저장하지 않음
        if not content or len(content.strip()) < 10:
            return

        # macOS 환경에서 한글 정규화 이슈를 줄이기 위한 NFC 정규화
        content = unicodedata.normalize("NFC", content)

        # 임베딩 생성
        embedding = self.embed_model.encode(content).tolist()

        # 파일명이 .txt 로 끝나면 표시용 title/source_url 에서는 확장자 제거
        display_name = f_name[:-4] if f_name.lower().endswith(".txt") else f_name

        # 검색/추적용 메타데이터 구성
        metadata = json.dumps(
            {
                "source_type": c_type,
                "page_info": page_info,
                "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            ensure_ascii=False,
        )

        table_name = os.getenv("DB_TABLE_NAME", "rag_vectors")

        # PostgreSQL INSERT
        # content_search_vector 는 tsvector 컬럼으로,
        # 벡터 검색 외에 키워드 검색 성능 향상을 위해 함께 저장
        sql = f"""
            INSERT INTO {table_name}
            (content_type, original_content, embedding, title, source_url, metadata, content_search_vector)
            VALUES (%s, %s, %s, %s, %s, %s, to_tsvector('simple', %s))
        """

        cursor.execute(
            sql,
            (
                c_type,
                content,
                embedding,
                display_name,
                display_name,
                metadata,
                content,
            ),
        )

    def ingest_txt_file(self, file_path):
        """
        ========================================================
        단일 txt 파일 인제스트 함수
        --------------------------------------------------------
        하나의 텍스트 파일을 읽고, 청크로 나눈 뒤 DB에 저장합니다.

        입력값:
        - file_path: 처리할 .txt 파일의 절대/상대 경로

        처리 절차:
        1) txt 파일인지 확인
        2) 파일 내용 읽기
        3) DB 연결 생성
        4) 텍스트를 청크로 분할
        5) 각 청크를 _save_to_db()로 저장
        6) 파일 단위 commit 수행
        7) 예외 발생 시 에러 로그 출력

        content type 판정:
        - '[이미지 내 텍스트]' 문자열이 포함되면 image
        - 아니면 text
        ========================================================
        """
        if not file_path.lower().endswith(".txt"):
            return

        print(f"🚀 처리 중: {os.path.basename(file_path)}")
        file_name = os.path.basename(file_path)

        conn = None
        cursor = None

        try:
            # UTF-8 텍스트 파일 읽기
            with open(file_path, "r", encoding="utf-8") as f:
                full_content = f.read()

            conn = self._get_db_conn()
            cursor = conn.cursor()

            # 내용이 충분히 있을 때만 청크 분할 및 저장 수행
            if len(full_content) > 10:
                chunks = self.text_splitter.split_text(full_content)

                for i, chunk in enumerate(chunks):
                    # OCR/이미지 추출 텍스트 여부를 간단히 식별
                    c_type = "image" if "[이미지 내 텍스트]" in chunk else "text"

                    self._save_to_db(
                        cursor=cursor,
                        c_type=c_type,
                        content=chunk,
                        f_name=file_name,
                        page_info=f"Chunk-{i + 1}",
                    )

            conn.commit()
            print(f"✅ DB 저장 완료: {file_name}")

        except Exception as e:
            print(f"❌ 에러 발생 ({file_name}): {e}")

            # 에러가 발생했을 때 트랜잭션이 살아 있으면 롤백
            if conn is not None:
                conn.rollback()

        finally:
            # 자원 정리
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    def ingest_all_txt_files(self, text_dir):
        """
        ========================================================
        폴더 단위 일괄 인제스트 함수
        --------------------------------------------------------
        지정된 폴더 안의 모든 .txt 파일을 찾아 순차적으로 DB에 저장합니다.

        입력값:
        - text_dir: txt 파일들이 들어 있는 폴더 경로

        처리 절차:
        1) 폴더 존재 여부 확인
        2) .txt 파일 목록 수집
        3) 파일이 없으면 안내 메시지 출력 후 종료
        4) 각 파일에 대해 ingest_txt_file() 호출

        사용 이유:
        - main() 에서 폴더 처리 로직을 깔끔하게 분리하기 위함
        - 단일 파일 처리 로직과 전체 파일 반복 로직을 분리하여 가독성 향상
        ========================================================
        """
        if not os.path.exists(text_dir):
            print(f"⚠️ '{text_dir}' 폴더가 없습니다. PDF 변환을 먼저 진행하세요.")
            return

        files = [f for f in os.listdir(text_dir) if f.lower().endswith(".txt")]

        if not files:
            print("📁 변환된 텍스트 파일이 없습니다.")
            return

        for file_name in files:
            file_path = os.path.join(text_dir, file_name)
            self.ingest_txt_file(file_path)

        print("\n✨ 하이브리드 인제스트가 완료되었습니다. 이제 에러 코드 검색도 정확해집니다!")


def main():
    """
    ============================================================
    프로그램 진입점
    ------------------------------------------------------------
    전체 인제스트 프로세스를 시작합니다.

    처리 절차:
    1) RAGIngestionLight 객체 생성
    2) converted_texts 폴더 경로 계산
    3) 해당 폴더 안의 모든 txt 파일 일괄 처리

    이 함수는 '프로그램 시작 흐름'만 담당하고,
    실제 파일 처리/DB 저장 로직은 클래스 내부 메서드에 위임합니다.
    ============================================================
    """
    agent = RAGIngestionLight()

    # 현재 파일 기준 converted_texts 폴더를 입력 소스로 사용
    text_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "converted_texts")

    agent.ingest_all_txt_files(text_dir)


if __name__ == "__main__":
    main()