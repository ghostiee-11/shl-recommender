"""Render docs/approach.md to a 2-page PDF for SHL submission.

Pipeline:
  1. markdown -> HTML via the ``markdown`` package
  2. Wrap in a small editorial CSS sheet
  3. Headless Chrome (Playwright) -> A4 PDF

Why Playwright instead of WeasyPrint: WeasyPrint needs cairo + pango +
gobject system libraries that don't ship with macOS by default and are
a pain to install. Playwright's headless Chrome handles the print path
out of the box.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import markdown
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "approach.md"
OUT = ROOT / "docs" / "approach.pdf"


CSS_TEXT = """
@page {
    size: A4;
    margin: 14mm 16mm;
}

* { box-sizing: border-box; }

html, body {
    font-family: "Helvetica Neue", "Inter", "Segoe UI", system-ui, sans-serif;
    color: #1a1a18;
    font-size: 9.4pt;
    line-height: 1.32;
    margin: 0;
}

h1 {
    font-size: 16pt;
    font-weight: 600;
    margin: 0 0 6pt 0;
    letter-spacing: -0.01em;
}
h2 {
    font-size: 11pt;
    font-weight: 600;
    margin: 8pt 0 3pt 0;
    color: #1f3d2e;
    letter-spacing: -0.005em;
}
h3 { font-size: 10pt; font-weight: 600; margin: 6pt 0 2pt 0; }

p { margin: 0 0 4pt 0; }
ul, ol { margin: 0 0 4pt 0; padding-left: 16pt; }
li { margin: 0 0 1.5pt 0; }
strong { color: #1a1a18; }

code, pre {
    font-family: "SF Mono", "Menlo", "Consolas", monospace;
    font-size: 8.4pt;
    color: #1f3d2e;
}
pre {
    background: #f4f0e6;
    padding: 6pt 8pt;
    border-radius: 2pt;
    line-height: 1.25;
    margin: 4pt 0;
    overflow: hidden;
    page-break-inside: avoid;
    white-space: pre-wrap;
}
code { background: #f4f0e6; padding: 0 2pt; border-radius: 2pt; }
pre code { background: none; padding: 0; }

a { color: #1f3d2e; text-decoration: underline; word-break: break-all; }

blockquote {
    border-left: 2pt solid #1f3d2e;
    margin: 0 0 4pt 0;
    padding: 0 0 0 8pt;
    color: #3a3936;
}

hr { border: none; border-top: 0.5pt solid #d6cdb8; margin: 6pt 0; }
"""


async def _render_pdf(html: str, out: Path) -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        try:
            page = await browser.new_page()
            await page.set_content(html, wait_until="networkidle")
            await page.pdf(
                path=str(out),
                format="A4",
                margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
                print_background=True,
                prefer_css_page_size=True,
            )
        finally:
            await browser.close()


def main() -> None:
    md_text = SRC.read_text(encoding="utf-8")
    body = markdown.markdown(
        md_text,
        extensions=["fenced_code", "tables", "sane_lists"],
        output_format="html5",
    )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{CSS_TEXT}</style></head><body>{body}</body></html>"
    )
    asyncio.run(_render_pdf(html, OUT))
    size_kb = OUT.stat().st_size / 1024
    print(f"Wrote {OUT} ({size_kb:.1f} KiB)")


if __name__ == "__main__":
    main()
