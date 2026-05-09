// Top-level layout + state.
//
// Layout: header → hero → asymmetric two-column work area → footer.
// State: a single useReducer would be overkill, three useStates and
// one async effect keep the data flow legible.

import { useCallback, useState } from "react";
import { ApiError, postChat } from "./lib/api";
import type { Message, Recommendation } from "./lib/types";
import { ChatPanel } from "./components/ChatPanel";
import { Header } from "./components/Header";
import { Sidebar } from "./components/Sidebar";
import { SuggestedPrompts } from "./components/SuggestedPrompts";

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [endOfConversation, setEndOfConversation] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [composer, setComposer] = useState("");

  const sendMessage = useCallback(
    async (text: string) => {
      const userMsg: Message = { role: "user", content: text };
      const nextMessages = [...messages, userMsg];
      setMessages(nextMessages);
      setComposer("");
      setPending(true);
      setError(null);

      try {
        const resp = await postChat(nextMessages);
        setMessages([
          ...nextMessages,
          { role: "assistant", content: resp.reply },
        ]);
        if (resp.recommendations.length > 0) {
          setRecommendations(resp.recommendations);
        }
        setEndOfConversation(resp.end_of_conversation);
      } catch (err) {
        const msg =
          err instanceof ApiError
            ? `${err.message}${err.status ? ` (status ${err.status})` : ""}`
            : "Couldn't reach the agent. Is the API running on :8000?";
        setError(msg);
      } finally {
        setPending(false);
      }
    },
    [messages],
  );

  const reset = useCallback(() => {
    setMessages([]);
    setRecommendations([]);
    setEndOfConversation(false);
    setError(null);
    setComposer("");
  }, []);

  const handlePromptPick = useCallback(
    (text: string) => {
      // From a clean state we send immediately so the demo is one-click.
      // Mid-conversation we just drop the text into the composer so
      // the user can edit before sending.
      if (messages.length === 0) {
        void sendMessage(text);
      } else {
        setComposer(text);
      }
    },
    [messages.length, sendMessage],
  );

  const isFresh = messages.length === 0;

  return (
    <div className="min-h-svh flex flex-col">
      <Header />

      <main className="flex-1 mx-auto w-full max-w-[1280px] px-6 lg:px-10 pt-12 lg:pt-16 pb-16">
        <Hero />

        {isFresh && (
          <div className="mt-12 lift-in lift-in-2">
            <SuggestedPrompts onPick={handlePromptPick} />
          </div>
        )}

        <div className="mt-12 grid grid-cols-1 lg:grid-cols-[1.4fr_1fr] gap-8 lg:gap-10 items-stretch">
          <ChatPanel
            messages={messages}
            pending={pending}
            onSend={sendMessage}
            onReset={reset}
            composerValue={composer}
            setComposerValue={setComposer}
          />
          <Sidebar
            recommendations={recommendations}
            endOfConversation={endOfConversation}
          />
        </div>

        {error && (
          <div
            role="alert"
            className="mt-6 max-w-2xl mx-auto paper border-warn/40 bg-warn-bg/40 px-4 py-3 text-[13px] text-warn"
          >
            {error}
          </div>
        )}
      </main>

      <Footer />
    </div>
  );
}

function Hero() {
  return (
    <section className="grid grid-cols-1 lg:grid-cols-[1.5fr_1fr] gap-8 items-end lift-in lift-in-1">
      <div>
        <p className="eyebrow mb-5">A conversational catalog</p>
        <h1 className="display text-[58px] sm:text-[72px] lg:text-[92px] tracking-[-0.024em]">
          Find the right{" "}
          <span className="display-italic">SHL</span> assessment
          <br className="hidden sm:block" />
          {" "}in conversation,
          <br className="hidden sm:block" />
          {" "}not <span className="display-italic">keywords.</span>
        </h1>
      </div>
      <div className="text-ink-mute text-[15px] leading-relaxed max-w-[36ch] lg:pb-3">
        <p>
          Most catalogs make you guess the right vocabulary first.
          This one asks the same questions a senior practitioner
          would, and only commits to a shortlist when it has enough
          to defend the choice.
        </p>
        <p className="mt-3 mono-tag">
          Built on the live SHL catalog · 377 individual test solutions
        </p>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="border-t border-rule mt-12">
      <div className="mx-auto max-w-[1280px] px-6 lg:px-10 py-6 flex flex-wrap items-center justify-between gap-4 text-[12px] text-ink-mute">
        <span className="mono-tag">
          SHL Labs · AI Intern take-home · {new Date().getFullYear()}
        </span>
        <span className="mono-tag">
          Stateless · 8-turn cap · grounded recommendations
        </span>
      </div>
    </footer>
  );
}
