// The right rail. Three states:
//
// 1. No recommendations yet → editorial empty state explaining what
//    will appear here. This is the place to teach the user, since the
//    chat is on the left.
// 2. Recommendations exist → cards stack, newest set replaces older.
// 3. End of conversation → a small "Done" affordance up top.
//
// The card animation is staggered through inline animationDelay so
// Tailwind doesn't have to ship 10 utility classes for every index.

import type { Recommendation } from "../lib/types";
import { RecommendationCard } from "./RecommendationCard";

interface Props {
  recommendations: Recommendation[];
  endOfConversation: boolean;
}

export function Sidebar({ recommendations, endOfConversation }: Props) {
  return (
    <aside
      className="lg:sticky lg:top-6 lift-in lift-in-4"
      aria-label="Shortlisted assessments"
    >
      <div className="flex items-baseline justify-between mb-4">
        <span className="eyebrow">Shortlist</span>
        {endOfConversation && (
          <span className="mono-tag text-pine flex items-center gap-1.5">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-pine" />
            Conversation closed
          </span>
        )}
      </div>

      {recommendations.length === 0 ? (
        <EmptyShortlist />
      ) : (
        <div className="space-y-3">
          {recommendations.map((r, i) => (
            <RecommendationCard key={r.url} rec={r} index={i} />
          ))}
          <p className="eyebrow text-ink-faint pt-2">
            {recommendations.length} of up to 10 · grounded to catalog
          </p>
        </div>
      )}
    </aside>
  );
}

function EmptyShortlist() {
  return (
    <div className="paper p-6">
      <p className="display text-[28px] leading-[1.05] mb-4">
        No shortlist <span className="display-italic">yet.</span>
      </p>
      <p className="text-ink-mute text-[14px] leading-relaxed mb-5">
        Tell the agent who you're hiring for. Once it has enough
        context, role, level, the kind of thing you want to measure,
        a grounded list of SHL assessments will appear here.
      </p>
      <ul className="space-y-2 text-[13px] text-ink-soft">
        {[
          "Names and links come from the live SHL catalog.",
          "URLs are verified against the catalog before display.",
          "If the agent isn't ready to commit, this rail stays empty.",
        ].map((line) => (
          <li key={line} className="flex gap-2">
            <span aria-hidden className="text-pine mt-[5px]">·</span>
            <span>{line}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
