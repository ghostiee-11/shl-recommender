"""One-time pre-compute of catalog embeddings.

Reads the catalog snapshot, builds the same enriched per-item search
documents the runtime uses, and embeds them with OpenAI's
``text-embedding-3-small``. Writes an L2-normalized ``(n_items, 1536)``
float32 matrix to ``data/catalog_embeddings.npy``.

Why offline:
* The catalog is pinned and stable, so on-startup re-embedding is
  pure waste.
* Shipping the matrix lets the runtime image drop torch + sentence-
  transformers + transformers (~430 MB), keeping us under Render's
  512 MB free-tier ceiling.
* One-time API cost is roughly $0.0004 (377 items × ~50 tokens).

Usage:
    OPENAI_API_KEY=... PYTHONPATH=src python scripts/embed_catalog.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shl_recommender.catalog.loader import build_search_doc, load_catalog  # noqa: E402
from shl_recommender.llm.openai_embeddings import OpenAIEmbedder  # noqa: E402

OUT = ROOT / "data" / "catalog_embeddings.npy"


def _load_env_file() -> None:
    """Tiny .env loader so this script is self-contained."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


async def main() -> None:
    _load_env_file()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")

    print("Loading catalog...")
    catalog = load_catalog()
    docs = [build_search_doc(it) for it in catalog.items]
    print(f"  {len(docs)} items")

    print("Calling OpenAI embeddings (text-embedding-3-small)...")
    t0 = time.perf_counter()
    embedder = OpenAIEmbedder(api_key)
    matrix = await embedder.encode(docs, batch_size=96)
    print(f"  shape={matrix.shape} dtype={matrix.dtype} took={time.perf_counter() - t0:.2f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, matrix)
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    asyncio.run(main())
