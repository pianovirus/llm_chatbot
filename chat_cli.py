import requests
import sys
import json
import threading
import time

# 전역 변수
stop_indicator = False

def display_thinking():
    """AI 응답 전까지 애니메이션 표시"""
    chars = [".  ", ".. ", "...", "   "]
    idx = 0
    while not stop_indicator:
        sys.stdout.write(f"\r🤔 AI가 생각하는 중{chars[idx % 4]}")
        sys.stdout.flush()
        idx += 1
        time.sleep(0.4)
    # 애니메이션 종료 시 해당 라인 깨끗이 지우기
    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()

def ask_question(query):
    global stop_indicator
    indicator_thread = None
    
    try:
        stop_indicator = False
        indicator_thread = threading.Thread(target=display_thinking, daemon=True)
        indicator_thread.start()

        # stream=True로 요청
        with requests.get("http://localhost:8000/search", params={"query": query}, stream=True, timeout=300) as response:
            if response.status_code == 200:
                first_chunk_received = False
                
                # 라인 단위로 읽기
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    decoded_line = line.decode('utf-8')
                    if not decoded_line.startswith("data: "):
                        continue
                    
                    content = decoded_line[6:].strip()
                    
                    # [핵심] 종료 신호 수신 시 루프 즉시 탈출
                    if content == "[DONE]":
                        stop_indicator = True # 애니메이션 정지
                        break
                    
                    try:
                        data = json.loads(content)
                        
                        # 진행 상태 출력
                        if "status" in data:
                            sys.stdout.write(f"\n[시스템] {data['status']}\n")
                            sys.stdout.flush()
                        
                        # 실제 답변 조각 출력
                        if "chunk" in data:
                            if not first_chunk_received:
                                stop_indicator = True # 첫 글자 오면 애니메이션 끄기
                                time.sleep(0.1) # 애니메이션 스레드가 줄 지울 시간 확보
                                first_chunk_received = True
                                print(f"🤖 AI 답변:\n" + "━"*40)
                            
                            print(data['chunk'], end="", flush=True)
                            
                    except json.JSONDecodeError:
                        continue
                
                # 답변 출력이 끝나면 줄바꿈
                print(f"\n" + "━"*40)
            else:
                stop_indicator = True
                print(f"\n❌ 서버 에러: {response.status_code}")
                
    except Exception as e:
        stop_indicator = True
        print(f"\n❌ 연결 중 오류 발생: {e}")
    finally:
        # 어떤 경우에도 애니메이션 스레드를 멈추고 자원 정리
        stop_indicator = True
        if indicator_thread and indicator_thread.is_alive():
            indicator_thread.join(timeout=0.2)

def main():
    # 터미널 한글 출력 설정
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("="*60)
    print("   MIRROR CODE Agentic RAG System")
    print("   (종료: exit, quit, q)")
    print("="*60)

    while True:
        try:
            # 1. 사용자 입력 대기 (이게 안 나오면 위 루프가 안 끝난 것)
            query = input("\n>> ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit', 'q', '종료']:
                print("👋 프로그램을 종료합니다.")
                break

            # 2. 질문 실행
            ask_question(query)
            
            # 3. 명시적으로 표준 출력 비우기
            sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n👋 프로그램을 종료합니다.")
            break

if __name__ == "__main__":
    main()