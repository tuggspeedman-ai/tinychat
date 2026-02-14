export default function Header({ onClear }: { onClear: () => void }) {
  return (
    <header className="flex items-center justify-between px-4 py-3 border-b border-zinc-800/50">
      <div className="flex items-center gap-3">
        <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-cyan-400 to-blue-500 flex items-center justify-center">
          <span className="text-xs font-bold text-white">T</span>
        </div>
        <div>
          <h1 className="text-sm font-semibold text-zinc-100">TinyChat</h1>
          <p className="text-[11px] text-zinc-500">561M params &middot; trained from scratch</p>
        </div>
      </div>
      <button
        onClick={onClear}
        className="flex items-center gap-1.5 text-xs text-zinc-500 hover:text-zinc-300 transition-colors px-2.5 py-1.5 rounded-lg border border-zinc-800 hover:border-zinc-700 hover:bg-zinc-800/50"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 5v14M5 12h14" />
        </svg>
        New chat
      </button>
    </header>
  );
}
