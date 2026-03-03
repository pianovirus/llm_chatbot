CREATE DATABASE IF NOT EXISTS llm_chatbot;
USE llm_chatbot;

CREATE TABLE rag_vectors (
    id INT AUTO_INCREMENT PRIMARY KEY,
    
    -- [데이터 성격 구분] 텍스트인지 이미지(OCR)인지 구분
    content_type ENUM('text', 'image') NOT NULL,
    
    -- [핵심 데이터] 원문 또는 이미지 요약 내용
    original_content TEXT NOT NULL, 
    
    -- [벡터 데이터] BGE-M3 (1024차원) 임베딩 저장
    embedding VECTOR(1024) NOT NULL, 
    
    -- [에이전트의 핵심: 출처 정보] 
    -- 문서 제목 (예: "2026년 POS 장애 대응 매뉴얼.pdf")
    title VARCHAR(255),
    -- 실제 참조 가능한 URL 또는 서버 내 파일 경로
    source_url TEXT,
    
    -- [확장용] 페이지 번호, 작성자, 태그 등 기타 유연한 정보 저장
    metadata JSON, 
    
    -- [관리용]
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 검색 최적화를 위한 인덱스 (옵션)
    INDEX idx_content_type (content_type)
);