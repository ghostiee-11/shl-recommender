// One assessment in the shortlist. Editorial layout: tiny mono test-type
// label up top, serif name, a thin horizontal rule, then a "Open in
// catalog" link that mirrors the magazine's "continue on page X"
// pagination cue.

import { TEST_TYPE_LABEL } from "../lib/types";
import type { Recommendation } from "../lib/types";

interface Props {
  rec: Recommendation;
  index: number;
}

export function RecommendationCard({ rec, index }: Props) {
  return (
    <article
      className="paper p-5 hover:border-rule-strong transition-colors"
      style={{ animation: `lift-in 700ms var(--ease-editorial) both`, animationDelay: `${index * 60 + 100}ms` }}
    >
      <div className="flex items-baseline justify-between gap-3 mb-2">
        <span className="mono-tag flex items-center gap-2">
          <span className="inline-flex items-center justify-center w-4 h-4 border border-pine text-pine text-[10px] font-mono leading-none">
            {rec.test_type}
          </span>
          {TEST_TYPE_LABEL[rec.test_type]}
        </span>
        <span className="mono-tag text-ink-faint">
          {String(index + 1).padStart(2, "0")}
        </span>
      </div>

      <h3
        className="display text-[22px] leading-[1.05] mb-3"
        style={{ fontWeight: 400 }}
      >
        {rec.name}
      </h3>

      <div className="border-t border-rule pt-3 flex items-center justify-between">
        <a
          href={rec.url}
          target="_blank"
          rel="noreferrer"
          className="text-[13px] text-pine link-underline"
          style={{
            backgroundImage: `linear-gradient(to right, var(--color-pine) 0%, var(--color-pine) 100%)`,
          }}
        >
          Open in SHL catalog
        </a>
        <span aria-hidden className="text-ink-faint text-[13px]">
          ↗
        </span>
      </div>
    </article>
  );
}
