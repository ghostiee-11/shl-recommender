// The agent embeds a hidden HTML comment carrying its state in every
// assistant reply (stateless reconstruction). It's invisible in
// rendered markdown but appears as a literal in raw text, so before
// we render the reply, we strip it.
//
// Mirror of the regex in src/shl_recommender/agent/state.py.
const STATE_HINT_RE = /<!--state:\{.*?\}-->/gs;

export function stripStateHint(text: string): string {
  return text.replace(STATE_HINT_RE, "").trim();
}
