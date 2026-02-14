interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  isStreaming?: boolean;
}

export default function ChatMessage({
  role,
  content,
  isStreaming,
}: ChatMessageProps) {
  if (role === "user") {
    return (
      <div className="message-appear flex justify-end">
        <div className="max-w-[85%] md:max-w-[70%] rounded-2xl rounded-br-sm px-4 py-2.5 bg-zinc-800 text-zinc-100 text-[15px] leading-relaxed">
          {content}
        </div>
      </div>
    );
  }

  return (
    <div className="message-appear flex justify-start gap-3">
      <div className="flex-shrink-0 w-6 h-6 rounded-md bg-gradient-to-br from-cyan-400 to-blue-500 flex items-center justify-center mt-0.5">
        <span className="text-[10px] font-bold text-white">T</span>
      </div>
      <div className="max-w-[85%] md:max-w-[70%]">
        <div
          className={`text-[15px] leading-relaxed text-zinc-300 whitespace-pre-wrap ${
            isStreaming && !content ? "streaming-cursor" : ""
          }`}
        >
          {content}
          {isStreaming && content && <span className="streaming-cursor" />}
        </div>
      </div>
    </div>
  );
}
