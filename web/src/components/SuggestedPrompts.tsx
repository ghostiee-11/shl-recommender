// Curated starter scenarios. Each card is sized to what we want
// recruiters to actually try first, and each demonstrates a
// different one of the four agent behaviors (clarify, recommend,
// refine, compare). The fifth (refusal) is handled implicitly by
// any off-topic input and doesn't deserve a "click to refuse" CTA.

interface Prompt {
  eyebrow: string;
  text: string;
  description: string;
}

const PROMPTS: Prompt[] = [
  {
    eyebrow: "Recommend",
    text: "Hiring 500 entry-level contact-centre agents in the Philippines. Bilingual English / Tagalog. Prefer remote-administered, under 30 minutes.",
    description: "A complete brief, the agent should commit to a shortlist quickly.",
  },
  {
    eyebrow: "Clarify",
    text: "I need an assessment.",
    description: "Vague intent. The agent should ask a focused question, not guess.",
  },
  {
    eyebrow: "Compare",
    text: "What's the difference between OPQ32r and the Verify Numerical Reasoning tests?",
    description: "Two named instruments. The agent should explain, grounded in the catalog.",
  },
  {
    eyebrow: "Refine",
    text: "Hiring a senior Java developer who'll work closely with non-technical stakeholders. Mid-to-senior level, Spring Boot, AWS.",
    description: "A solid first turn. Then try: \"Actually, drop the personality tests.\"",
  },
];

interface Props {
  onPick: (text: string) => void;
}

export function SuggestedPrompts({ onPick }: Props) {
  return (
    <section className="lift-in lift-in-3" aria-label="Try a scenario">
      <p className="eyebrow mb-3">Try a scenario</p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {PROMPTS.map((p) => (
          <button
            key={p.eyebrow}
            type="button"
            onClick={() => onPick(p.text)}
            className="text-left paper p-4 hover:border-rule-strong hover:bg-paper-2/40 transition-colors group"
          >
            <span className="mono-tag text-pine">{p.eyebrow}</span>
            <p className="mt-2 text-[14px] text-ink leading-snug">
              <span className="ink-highlight group-hover:bg-fern transition-colors">
                {p.text}
              </span>
            </p>
            <p className="mt-2 text-[12px] text-ink-mute leading-snug">
              {p.description}
            </p>
          </button>
        ))}
      </div>
    </section>
  );
}
