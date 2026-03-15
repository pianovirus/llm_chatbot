# 1. 파이썬 환경
FROM python:3.10-slim

# 2. 필수 리눅스 패키지 (psycopg2 등을 위해 필요)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. 경로 주의! backend 폴더 안에 있는 파일을 가져와야 함
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 5. 환경 변수
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# 6. 실행 (이제 /app 안에 agent_core.py가 들어있음)
CMD ["python", "agent_core.py"]