// One conversational turn. Two visual modes:
//
// * user: a tight pill on the right, ink-on-cream, no decoration.
// * assistant: full-width prose on the left, framed by an italic
//   Fraunces "-" mark, like a quoted aside in an editorial.
//
// State-hint stripping happens here so a message ever rendered in
// the UI never leaks the embedded HTML comment.

import { stripStateHint } from "../lib/text";

interface Props {
  role: "user" | "assistant";
  content: string;
  /** While true, render typing dots in place of the assistant's content. */
  pending?: boolean;
}

export function MessageBubble({ role, content, pending }: Props) {
  const text = stripStateHint(content);

  if (role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[78%] rounded-[3px] bg-bubble-user text-bubble-user-fg px-4 py-2.5 text-[14.5px] leading-relaxed">
          {text}
        </div>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-[auto_1fr] gap-4 group">
      <div
        aria-hidden
        className="display display-italic text-[28px] leading-none -mt-1 select-none"
      >
       ,
      </div>
      <div className="text-ink leading-relaxed text-[15px] whitespace-pre-wrap">
        {pending ? (
          <span className="inline-flex items-center gap-1 text-ink-mute">
            <span className="dot" />
            <span className="dot" />
            <span className="dot" />
          </span>
        ) : (
          text
        )}
      </div>
    </div>
  );
}
