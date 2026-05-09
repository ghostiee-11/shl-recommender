// The input row. Auto-grows as the user types (up to a cap), submits on
// Enter, newline on Shift+Enter. Disabled state mirrors the loading
// state of the chat, the agent's contract is one inflight call.

import { useEffect, useRef } from "react";

interface Props {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  disabled?: boolean;
  placeholder?: string;
}

export function Composer({
  value,
  onChange,
  onSubmit,
  disabled,
  placeholder = "Describe the role you're hiring for…",
}: Props) {
  const ref = useRef<HTMLTextAreaElement | null>(null);

  // Auto-grow.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }, [value]);

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!disabled && value.trim()) onSubmit();
    }
  }

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (!disabled && value.trim()) onSubmit();
      }}
      className="border-t border-rule px-5 py-4 bg-paper"
    >
      <div className="flex items-end gap-3">
        <textarea
          ref={ref}
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          className="flex-1 resize-none bg-transparent outline-none placeholder:text-ink-faint text-[15px] leading-relaxed disabled:opacity-50 max-h-[220px]"
          aria-label="Your message"
        />
        <button
          type="submit"
          disabled={disabled || !value.trim()}
          className="shrink-0 inline-flex items-center gap-2 px-3.5 py-2 rounded-[3px] bg-pine text-paper text-[13px] font-medium tracking-wide disabled:opacity-30 disabled:cursor-not-allowed hover:bg-pine-2 transition-colors"
        >
          Send
          <kbd
            aria-hidden
            className="font-mono text-[10px] opacity-70 border border-paper/30 rounded px-1 py-[1px]"
          >
            ⏎
          </kbd>
        </button>
      </div>
      <p className="eyebrow mt-3 text-ink-faint">
        Stateless · 8-turn cap · refusals are deterministic
      </p>
    </form>
  );
}
