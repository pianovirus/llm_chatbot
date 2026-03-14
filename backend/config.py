import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER")
MODEL_NAME = os.getenv("LLM_MODEL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
TABLE_NAME = os.getenv("DB_TABLE_NAME")
BASE_DOCS_URL = os.getenv("BASE_DOCS_URL")
DB_NAME = os.getenv("DB_NAME")

# Query Rewrite 및 Reranker 설정
QUERY_REWRITE_ENABLED = os.getenv("QUERY_REWRITE_ENABLED")
QUERY_REWRITE_MAX_CHARS = int(os.getenv("QUERY_REWRITE_MAX_CHARS"))
RERANK_ENABLED = os.getenv("RERANK_ENABLED")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
RERANK_CANDIDATE_LIMIT = int(os.getenv("RERANK_CANDIDATE_LIMIT"))
MAX_CHUNKS_PER_TITLE = int(os.getenv("MAX_CHUNKS_PER_TITLE"))

# PostgreSQL 연결 정보
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "dbname": os.getenv("DB_NAME")
}