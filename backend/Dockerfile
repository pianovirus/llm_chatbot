# 1. 파이썬 환경 설정
FROM python:3.10-slim

# 2. 필수 라이브러리 설치 (Mac 전용 설정 등이 있으므로 리눅스 호환용)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 전체 코드 복사
COPY . .

# 5. Hugging Face용 환경 변수 및 포트 설정
# 코드 내에서 8000 포트를 사용 중이므로, 이를 7860으로 강제하거나 
# agent_core.py의 포트 번호를 수정해야 합니다.
ENV PORT=7860

# 6. 실행 (agent_core.py 실행)
# 팁: 아래 CMD 실행 전, agent_core.py 마지막 줄의 8000을 7860으로 수정해 주세요!
CMD ["python", "agent_core.py"]