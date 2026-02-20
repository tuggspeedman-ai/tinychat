import ReactMarkdown from "react-markdown";

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

  // Thinking indicator before first token arrives
  if (isStreaming && !content) {
    return (
      <div className="message-appear flex justify-start gap-3">
        <div className="flex-shrink-0 w-6 h-6 rounded-md bg-gradient-to-br from-cyan-400 to-blue-500 flex items-center justify-center mt-0.5">
          <span className="text-[10px] font-bold text-white">T</span>
        </div>
        <div className="flex items-center gap-1 py-2">
          <span className="thinking-dot w-1.5 h-1.5 rounded-full bg-zinc-500" />
          <span className="thinking-dot w-1.5 h-1.5 rounded-full bg-zinc-500 [animation-delay:0.15s]" />
          <span className="thinking-dot w-1.5 h-1.5 rounded-full bg-zinc-500 [animation-delay:0.3s]" />
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
        <div className="text-[15px] leading-relaxed text-zinc-300 markdown-content">
          <ReactMarkdown
            components={{
              code({ className, children, ...props }) {
                const isBlock = className?.startsWith("language-");
                if (isBlock) {
                  return (
                    <div className="my-3 rounded-lg bg-zinc-900 border border-zinc-800 overflow-hidden">
                      <div className="flex items-center px-4 py-1.5 bg-zinc-800/60 border-b border-zinc-800">
                        <span className="text-[11px] text-zinc-500">
                          {className?.replace("language-", "") || "code"}
                        </span>
                      </div>
                      <pre className="px-4 py-3 overflow-x-auto">
                        <code className="text-[13px] leading-relaxed text-zinc-300 font-mono">
                          {children}
                        </code>
                      </pre>
                    </div>
                  );
                }
                return (
                  <code
                    className="px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-200 text-[13px] font-mono"
                    {...props}
                  >
                    {children}
                  </code>
                );
              },
              pre({ children }) {
                return <>{children}</>;
              },
              p({ children }) {
                return <p className="mb-2 last:mb-0">{children}</p>;
              },
              strong({ children }) {
                return <strong className="font-semibold text-zinc-100">{children}</strong>;
              },
              em({ children }) {
                return <em className="italic text-zinc-200">{children}</em>;
              },
              ul({ children }) {
                return <ul className="mb-2 ml-4 list-disc space-y-1 marker:text-zinc-600">{children}</ul>;
              },
              ol({ children }) {
                return <ol className="mb-2 ml-4 list-decimal space-y-1 marker:text-zinc-500">{children}</ol>;
              },
              li({ children }) {
                return <li className="pl-1">{children}</li>;
              },
              a({ href, children }) {
                return (
                  <a href={href} target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:text-cyan-300 underline underline-offset-2">
                    {children}
                  </a>
                );
              },
              h1({ children }) {
                return <h1 className="text-lg font-semibold text-zinc-100 mb-2 mt-3">{children}</h1>;
              },
              h2({ children }) {
                return <h2 className="text-base font-semibold text-zinc-100 mb-2 mt-3">{children}</h2>;
              },
              h3({ children }) {
                return <h3 className="text-sm font-semibold text-zinc-100 mb-1 mt-2">{children}</h3>;
              },
              blockquote({ children }) {
                return <blockquote className="border-l-2 border-zinc-700 pl-3 my-2 text-zinc-400 italic">{children}</blockquote>;
              },
            }}
          >
            {content}
          </ReactMarkdown>
          {isStreaming && content && <span className="streaming-cursor" />}
        </div>
      </div>
    </div>
  );
}
