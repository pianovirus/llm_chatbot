# text_utils.py
# =========================================================
# 텍스트 전처리 유틸 모듈
# - RAG 검색 품질을 조금이라도 높이기 위한 "문장 정리/키워드 정규화/phrase 후보 생성" 담당
# - main(app)에서 import해서 사용
# =========================================================

import re
import unicodedata
from typing import Dict

# (역할) 텍스트를 한 줄로 정리 (개행/연속 공백 제거)
def safe_single_line(text: str) -> str:
    """텍스트를 한 줄로 정리(연속 공백/개행 제거)."""
    return " ".join(text.strip().split()) if text else ""

# (역할) 한국어 조사/어미를 간단 규칙으로 제거해 검색용 키워드 정규화
def normalize_keyword(token: str) -> str:
    """
    한국어 조사/어미 비슷한 접미를 제거해서 검색 키워드를 정리.
    (완벽한 형태소 분석이 아니라, 가벼운 규칙 기반 정리)
    """
    if not token:
        return ""
    token = unicodedata.normalize("NFC", token.strip())
    suffixes = [
        "에서는", "으로는", "에게서", "까지는", "부터는",
        "에서", "으로", "에게", "까지", "부터",
        "처럼", "하고", "이며", "이고", "하면", "라는", "라고",
        "니다", "어요", "아요", "를", "을", "은", "는", "이", "가", "도", "만", "와", "과", "에", "의", "로"
    ]
    for s in sorted(suffixes, key=len, reverse=True):
        if len(token) > len(s) + 1 and token.endswith(s):
            token = token[:-len(s)]
            break
    return token.strip()

# (역할) 특수문자/공백 제거한 compact query 생성 (phrase 매칭용)
def make_compact_query(query: str) -> str:
    """특수문자 제거 + 공백 제거 → 붙여쓴 형태의 compact 문자열 생성."""
    if not query:
        return ""
    q = re.sub(r"[\"'`“”‘’()\[\]{}:;,./\\!?@#$%^&*+=|<>~-]+", " ", safe_single_line(query))
    return q.replace(" ", "")

# (역할) 원문/compact/토큰 결합 등 다양한 phrase 후보 생성
def generate_phrase_candidates(query: str) -> Dict[str, str]:
    """
    검색에 활용할 phrase 후보 생성.
    - original: 원문
    - compact: 공백/특수문자 제거 버전
    - spaced_phrase: 핵심 토큰 2~3개를 띄어쓴 버전
    - pair/triple_phrase: 핵심 토큰을 붙여쓴 버전
    """
    q = safe_single_line(query)
    raw_tokens = re.findall(r"[가-힣A-Za-z0-9_-]{2,}", q)
    normalized = []
    seen = set()
    for t in raw_tokens:
        nt = normalize_keyword(t)
        if len(nt) >= 2 and nt not in seen:
            normalized.append(nt)
            seen.add(nt)
    return {
        "original": q,
        "compact": make_compact_query(q),
        "spaced_phrase": " ".join(normalized[:3]) if normalized else q,
        "pair_phrase": "".join(normalized[:2]),
        "triple_phrase": "".join(normalized[:3]),
    }

# (역할) 검색용 핵심 키워드 추출 (stopword 제거)
def extract_search_keywords(query: str, max_keywords: int = 4):
    """
    검색 키워드 후보를 뽑아냄.
    - stopwords 제거
    - 길이가 긴 키워드를 우선
    """
    raw_tokens = re.findall(r"[가-힣A-Za-z0-9_-]{2,}", query)
    stopwords = {
        "방법", "문의", "관련", "설명", "기능", "화면", "문서", "가이드",
        "알려줘", "알려", "주세요", "부탁", "어떻게", "어디", "왜", "무엇"
    }
    cleaned, seen = [], set()
    for t in raw_tokens:
        n = normalize_keyword(t)
        if len(n) >= 2 and n not in stopwords and n not in seen:
            seen.add(n)
            cleaned.append(n)
    return sorted(cleaned, key=lambda x: -len(x))[:max_keywords]