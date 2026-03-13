"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";

export default function ChatPage() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<{ role: string; content: string; status?: string; isFeedbackSent?: boolean }[]>([]);
  const [currentStatus, setCurrentStatus] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentStatus, isLoading]);

  // 🚨 피드백 전송 함수
  const handleFeedback = async (index: number, query: string, answer: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, answer }),
      });

      if (response.ok) {
        // 성공 시 해당 메시지의 버튼 상태 업데이트
        setMessages((prev) =>
          prev.map((msg, i) => (i === index ? { ...msg, isFeedbackSent: true } : msg))
        );
        
      }
    } catch (error) {
      console.error("피드백 전송 실패:", error);
      alert("피드백 전송 중 오류가 발생했습니다.");
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userQuery = input;
    setInput("");
    
    // 메시지 추가 (이전 답변들까지 포함)
    setMessages((prev) => [...prev, { role: "user", content: userQuery }]);
    setIsLoading(true);
    setCurrentStatus("서버 연결 중...");

    try {
      const response = await fetch(`${API_BASE_URL}/search?query=${encodeURIComponent(userQuery)}`);
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

  const formatContent = (content: string) => {
    if (!content) return "";
    return content
      .replace(/\\n/g, "\n")
      .replace(/\n{3,}/g, "\n\n")
      .normalize("NFC")
      .trim();
  };

  return (
    <div className="flex flex-col h-screen max-w-2xl mx-auto p-4 bg-gray-50 text-black font-sans">
      <header className="py-4 border-b flex justify-between items-center">
        <h1 className="text-xl font-bold text-blue-600">올댓애스크</h1>
        <span className="text-[10px] bg-blue-100 text-blue-600 px-2 py-0.5 rounded-full font-semibold">AI Assistant</span>
      </header>

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
                  
                  {/* 🚨 피드백 UI 추가 영역 */}
                  {!isLoading && i === messages.length - 1 && (
                    <div className="mt-4 pt-3 border-t border-dashed border-gray-200">
                      <div className="flex items-center justify-between gap-2">
                        <div>
                          <p className="text-xs text-blue-500 font-semibold mb-1">💡 해결에 도움이 되었나요?</p>
                          <p className="text-[11px] text-gray-400">도움이 되었다면 버튼을 눌러 AI를 학습시켜주세요.</p>
                        </div>
                        <button
                          onClick={() => handleFeedback(i, messages[i-1]?.content || "", msg.content)}
                          disabled={msg.isFeedbackSent}
                          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold transition-all shadow-sm ${
                            msg.isFeedbackSent 
                              ? "bg-green-100 text-green-600 cursor-default" 
                              : "bg-blue-50 text-blue-600 hover:bg-blue-600 hover:text-white active:scale-95"
                          }`}
                        >
                          {msg.isFeedbackSent ? "학습완료 ✅" : "도움됨 👍"}
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

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