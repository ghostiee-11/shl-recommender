// The conversation surface. Sticky header inside, scrolling message
// list, composer pinned to the bottom. Auto-scrolls on new messages
// so a long reply doesn't trap the user above the latest reply.

import { useEffect, useRef } from "react";
import type { Message } from "../lib/types";
import { Composer } from "./Composer";
import { MessageBubble } from "./MessageBubble";

interface Props {
  messages: Message[];
  pending: boolean;
  onSend: (text: string) => void;
  onReset: () => void;
  composerValue: string;
  setComposerValue: (v: string) => void;
}

export function ChatPanel({
  messages,
  pending,
  onSend,
  onReset,
  composerValue,
  setComposerValue,
}: Props) {
  const scrollRef = useRef<HTMLDivElement | null>(null);

  // Pin to bottom on each new turn (or on pending state change so the
  // typing dots stay visible).
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages.length, pending]);

  return (
    <section
      className="paper flex flex-col min-h-160 lg:min-h-180 lift-in lift-in-3"
      aria-label="Conversation"
    >
      <header className="px-5 py-3 border-b border-rule flex items-center justify-between">
        <span className="eyebrow">Conversation</span>
        <button
          type="button"
          onClick={onReset}
          className="eyebrow link-underline hover:text-ink"
          disabled={messages.length === 0 && !pending}
        >
          Reset
        </button>
      </header>

      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-5 py-6 space-y-6 scroll-smooth"
      >
        {messages.map((m, i) => (
          <MessageBubble key={i} role={m.role} content={m.content} />
        ))}
        {pending && (
          <MessageBubble role="assistant" content="" pending />
        )}
      </div>

      <Composer
        value={composerValue}
        onChange={setComposerValue}
        onSubmit={() => composerValue.trim() && onSend(composerValue.trim())}
        disabled={pending}
      />
    </section>
  );
}
