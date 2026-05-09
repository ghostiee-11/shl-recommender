// Editorial masthead. The brand mark is set in italic Fraunces so it
// reads as "publication name" rather than "logo". The right-rail meta
// (status dot + nav links) keeps the bar honest about being software.

import { useEffect, useState } from "react";
import { getHealth } from "../lib/api";

type HealthStatus = "checking" | "ok" | "down";

export function Header() {
  const [status, setStatus] = useState<HealthStatus>("checking");

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then(() => !cancelled && setStatus("ok"))
      .catch(() => !cancelled && setStatus("down"));
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <header className="border-b border-rule lift-in lift-in-1">
      <div className="mx-auto max-w-[1280px] px-6 lg:px-10 py-5 flex items-center justify-between">
        <a href="/" className="group flex items-baseline gap-3 -ml-1">
          <span
            aria-hidden
            className="display display-italic text-[28px] leading-none translate-y-[1px]"
          >
            shl
          </span>
          <span className="eyebrow">Assessment Concierge</span>
        </a>

        <nav className="flex items-center gap-6 text-[13px]">
          <StatusPill status={status} />
          <a
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
            className="link-underline text-ink-soft hover:text-ink"
          >
            Source
          </a>
          <a
            href="https://www.shl.com/solutions/products/product-catalog/"
            target="_blank"
            rel="noreferrer"
            className="link-underline text-ink-soft hover:text-ink"
          >
            Catalog
          </a>
        </nav>
      </div>
    </header>
  );
}

function StatusPill({ status }: { status: HealthStatus }) {
  const dotColor =
    status === "ok"
      ? "bg-pine"
      : status === "down"
        ? "bg-warn"
        : "bg-ink-faint animate-pulse";
  const label =
    status === "ok"
      ? "Live"
      : status === "down"
        ? "Offline"
        : "Connecting";
  return (
    <span
      className="inline-flex items-center gap-2 mono-tag"
      title={`API ${label.toLowerCase()}`}
    >
      <span
        aria-hidden
        className={`inline-block w-1.5 h-1.5 rounded-full ${dotColor}`}
      />
      {label}
    </span>
  );
}
