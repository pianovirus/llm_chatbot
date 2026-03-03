"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";

export default function ChatPage() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<{ role: string; content: string; status?: string }[]>([]);
  const [currentStatus, setCurrentStatus] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentStatus, isLoading]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userQuery = input;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userQuery }]);
    setIsLoading(true);
    setCurrentStatus("서버 연결 중...");

    try {
      const response = await fetch(`http://localhost:8000/search?query=${encodeURIComponent(userQuery)}`);
      if (!response.body) return;

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantMessage = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const jsonStr = line.replace("data: ", "").trim();
            if (jsonStr === "[DONE]") {
              setIsLoading(false);
              setCurrentStatus("");
              break;
            }

            try {
              const data = JSON.parse(jsonStr);
              if (data.status) setCurrentStatus(data.status);
              if (data.chunk) {
                assistantMessage += data.chunk;
                setMessages((prev) => {
                  const lastMsg = prev[prev.length - 1];
                  if (lastMsg?.role === "assistant") {
                    return [...prev.slice(0, -1), { ...lastMsg, content: assistantMessage }];
                  }
                  return [...prev, { role: "assistant", content: assistantMessage }];
                });
              }
            } catch (e) {
              console.error("JSON 파싱 에러", e);
            }
          }
        }
      }
    } catch (error) {
      console.error("통신 에러:", error);
      setCurrentStatus("오류가 발생했습니다.");
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * 💡 줄바꿈 과잉 현상 해결 및 자소 분리(NFC) 대응
   */
  const formatContent = (content: string) => {
    if (!content) return "";
    return content
      .replace(/\\n/g, "\n")         // JSON 인코딩된 줄바꿈 복구
      .replace(/\n{3,}/g, "\n\n")    // 과도한 개행 축소
      .normalize("NFC")              // 💡 한글 자소 분리 현상(Mac NFD)을 표준(NFC)으로 결합
      .trim();
  };

  return (
    <div className="flex flex-col h-screen max-w-2xl mx-auto p-4 bg-gray-50 text-black font-sans">
      <header className="py-4 border-b">
        <h1 className="text-xl font-bold text-blue-600">올댓애스크</h1>
      </header>

      {/* 메시지 리스트 */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`max-w-[95%] p-4 rounded-2xl shadow-sm ${
              msg.role === "user" 
                ? "bg-blue-600 text-white rounded-tr-none" 
                : "bg-white border border-gray-200 text-gray-800 rounded-tl-none"
            }`}>
              {msg.role === "user" ? (
                <div className="whitespace-pre-wrap text-sm">{msg.content}</div>
              ) : (
                <div className="prose prose-sm max-w-none break-words whitespace-pre-wrap leading-tight text-inherit">
                  <ReactMarkdown 
                    components={{
                      h2: ({node, ...props}) => <h2 className="text-lg font-bold border-b pb-1 mb-3 text-blue-800" {...props} />,
                      h3: ({node, ...props}) => <h3 className="text-md font-bold mt-3 mb-1 text-gray-700" {...props} />,
                      p: ({node, ...props}) => <p className="mb-2 last:mb-0 leading-relaxed" {...props} />,
                      ul: ({node, ...props}) => <ul className="list-disc pl-4 mb-2" {...props} />,
                      li: ({node, ...props}) => <li className="mb-0.5" {...props} />,
                      hr: () => <hr className="my-4 border-gray-100" />,
                      // 🔗 링크 스타일 및 속성 유지
                      a: ({node, ...props}) => (
                        <a 
                          className="text-blue-500 hover:text-blue-700 underline underline-offset-4 font-bold transition-colors break-all" 
                          target="_blank" 
                          rel="noopener noreferrer" 
                          {...props} 
                        />
                      ),
                    }}
                  >
                    {formatContent(msg.content)}
                  </ReactMarkdown>
                  
                  {!isLoading && i === messages.length - 1 && (
                    <div className="mt-4 pt-3 border-t border-dashed border-gray-200">
                      <p className="text-xs text-blue-500 font-semibold mb-1">💡 추가 도움이 필요하신가요?</p>
                      <p className="text-[11px] text-gray-400">
                        위 조치로 해결되지 않으면 언제든 물어봐 주세요.
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {/* 인디케이터 위치 유지 */}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-blue-50 text-blue-600 text-sm px-4 py-2 rounded-full border border-blue-100 animate-pulse flex items-center gap-2">
              <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce" />
              {currentStatus || "AI가 답변을 생성 중입니다..."}
            </div>
          </div>
        )}
        
        <div ref={scrollRef} />
      </div>

      {/* 입력창 */}
      <form onSubmit={handleSubmit} className="p-4 border-t bg-white flex gap-2">
        <input
          className="flex-1 border border-gray-300 rounded-full px-5 py-3 focus:outline-none focus:ring-2 focus:ring-blue-400 text-black placeholder-gray-400"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="문제를 입력하세요..."
          disabled={isLoading}
        />
        <button 
          className="bg-blue-600 text-white px-6 py-2 rounded-full font-bold disabled:bg-gray-300 transition-colors"
          disabled={isLoading}
        >
          {isLoading ? "..." : "전송"}
        </button>
      </form>
    </div>
  );
}