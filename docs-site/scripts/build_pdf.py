#!/usr/bin/env python3
"""Render the built Speedtronic documentation into a single print-ready PDF.

Docusaurus has no first-party PDF export, so this script drives a headless
browser instead:

1. Read the sidebar to obtain the canonical page order.
2. Visit each page, wait for Mermaid diagrams to finish rendering, then capture
   the live DOM (so client-rendered SVG diagrams are preserved).
3. Namespace every ``id``/fragment href per page so anchors stay unique once
   the pages are concatenated into one document.
4. Emit a cover page, a linked table of contents, and the merged body, then
   print to PDF with running headers and page numbers.

Usage (from ``docs-site/``)::

    python scripts/build_pdf.py [--output path.pdf]
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import urljoin, urlparse

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
DEFAULT_OUTPUT = ROOT / "static" / "pdf" / "speedtronic-documentation.pdf"
BASE_URL = "http://127.0.0.1:8765"
TITLE = "Speedtronic Documentation"
VERSION = "2.0.0"

# Wait for this long for Mermaid to finish before capturing a page.
MERMAID_TIMEOUT_MS = 20_000


def slugify(route: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", route.lower()).strip("-") or "docs"


def page_url(route: str) -> str:
    """Map a docs route to its built file.

    ``trailingSlash`` is false, so the overview lives at ``/docs.html``. Using
    ``/docs`` would hit the real ``build/docs/`` directory and return a
    directory listing instead of the page.
    """
    return f"{BASE_URL}/docs.html" if not route else f"{BASE_URL}/docs/{route}.html"


ORDER_JS = """
() => {
  const out = [];
  const nav = document.querySelector("nav[aria-label='Docs sidebar']");
  if (!nav) return out;
  const seen = new Set();
  const walk = (listEl, category) => {
    for (const li of listEl.children) {
      if (!li.classList.contains('menu__list-item')) continue;
      const cat = li.classList.contains('theme-doc-sidebar-item-category')
        ? li
        : li.querySelector(':scope > .theme-doc-sidebar-item-category');
      let group = category;
      if (cat) {
        const label = cat.querySelector('.categoryLinkLabel, span');
        group = label ? label.textContent.trim() : category;
      }
      const link = li.querySelector(':scope > .theme-doc-sidebar-item-link > a.menu__link')
                || li.querySelector(':scope > a.menu__link');
      if (link) {
        const href = link.getAttribute('href') || '';
        if (href.startsWith('/docs')) {
          const route = href.replace('/docs', '').replace(/^\\//, '').replace(/\\/$/, '');
          if (!seen.has(route)) {
            seen.add(route);
            out.push({category: group, title: link.innerText.trim(), route});
          }
        }
      }
      const nested = li.querySelector(':scope > ul.menu__list');
      if (nested) walk(nested, group);
    }
  };
  const root = nav.querySelector('ul.menu__list');
  if (root) walk(root, '');
  return out;
}
"""


def read_doc_order(page) -> list[tuple[str, str, str]]:
    """Return ``(category, title, route)`` triples in sidebar order.

    Category headers are also ``<a class="menu__link">`` elements, so they are
    excluded from the page list and instead used to group the table of
    contents.
    """
    page.goto(page_url(""), wait_until="networkidle")
    page.wait_for_selector(
        "nav[aria-label='Docs sidebar'] a.menu__link", state="attached", timeout=30_000
    )
    entries = page.evaluate(ORDER_JS)
    return [(e["category"], e["title"], e["route"]) for e in entries]


def wait_for_mermaid(page) -> int:
    """Block until every Mermaid container holds a rendered SVG."""
    page.wait_for_timeout(600)
    deadline = time.time() + MERMAID_TIMEOUT_MS / 1000
    while time.time() < deadline:
        pending = page.evaluate(
            """() => Array.from(document.querySelectorAll(
                 '.docusaurus-mermaid-container, pre.mermaid'))
               .filter(el => !el.querySelector('svg')).length"""
        )
        if pending == 0:
            break
        page.wait_for_timeout(250)
    return page.locator(".docusaurus-mermaid-container svg").count()


def capture_page(page, route: str) -> tuple[str, str, int]:
    """Return ``(route, article_html, diagram_count)`` for one documentation page."""
    url = page_url(route)
    page.goto(url, wait_until="networkidle")
    page.wait_for_selector("article .theme-doc-markdown", timeout=30_000)
    diagrams = wait_for_mermaid(page)
    article = page.locator("article").first
    body = article.inner_html()
    title = page.evaluate(
        "() => { const h = document.querySelector('article h1');"
        " return h ? h.textContent.trim() : document.title; }"
    )
    return route, body, diagrams


SVG_BLOCK = re.compile(r"<svg\b.*?</svg>", re.S)


def namespace_ids(markup: str, prefix: str) -> str:
    """Prefix element ids and same-document fragment hrefs to avoid collisions.

    ``<svg>`` blocks are left untouched on purpose. Mermaid scopes all of its
    styling through an id selector that also appears inside the SVG's embedded
    ``<style>`` text; renaming only the attribute would break every rule and
    leave the diagram filled black.
    """

    def fix_segment(segment: str) -> str:
        segment = re.sub(r'id="([^"]+)"', lambda m: f'id="{prefix}-{m.group(1)}"', segment)
        return re.sub(r'href="#([^"]+)"', lambda m: f'href="#{prefix}-{m.group(1)}"', segment)

    out: list[str] = []
    cursor = 0
    for match in SVG_BLOCK.finditer(markup):
        out.append(fix_segment(markup[cursor : match.start()]))
        out.append(match.group(0))
        cursor = match.end()
    out.append(fix_segment(markup[cursor:]))
    return "".join(out)


def strip_nav_chrome(markup: str) -> str:
    """Remove breadcrumbs, pagination and edit links that do not print well."""
    for pattern in (
        r'<nav class="theme-doc-breadcrumbs.*?</nav>',
        r'<nav class="docusaurus-mt-lg pagination-nav.*?</nav>',
        r'<div class="tocCollapsible.*?</div>',
    ):
        markup = re.sub(pattern, "", markup, flags=re.S)
    return markup


PRINT_CSS = """
@page { size: Letter; margin: 18mm 16mm 16mm; }
* { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
html, body { background: #fff !important; color: #111 !important; }
/* Neutralise Docusaurus dark-mode custom properties regardless of the
   captured data-theme attribute. */
:root, [data-theme='dark'], [data-theme='light'] {
  --ifm-color-emphasis-0: #fff; --ifm-color-emphasis-100: #1b1b1d;
  --ifm-background-color: #fff; --ifm-font-color-base: #111;
  --ifm-heading-color: #111; --ifm-toc-border-color: #d0d0d0;
}
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       font-size: 10.5pt; line-height: 1.5; margin: 0; }

.cover { height: 235mm; display: flex; flex-direction: column;
         justify-content: center; text-align: center; page-break-after: always; }
.cover h1 { font-size: 30pt; margin: 0 0 6mm; letter-spacing: -0.5pt; }
.cover .sub { font-size: 14pt; color: #444; margin: 0 0 14mm; }
.cover .meta { font-size: 10pt; color: #666; line-height: 1.9; }
.cover .meta code { font-size: 9.5pt; }

.toc { page-break-after: always; }
.toc h2 { font-size: 17pt; margin: 0 0 6mm; }
.toc ol { list-style: none; padding-left: 0; margin: 0; }
.toc li { margin: 0 0 1.6mm; font-size: 10pt; }
.toc a { color: #111; text-decoration: none; display: flex;
         justify-content: space-between; gap: 6mm; }
.toc a .t { flex: 0 1 auto; }
.toc a .n { flex: 0 0 auto; color: #666; font-variant-numeric: tabular-nums; }
.toc .grp { font-weight: 700; margin-top: 4.5mm; font-size: 10.5pt; }
/* Invisible 1pt marker used to resolve section start pages between passes. */
.pgmark { color: #fff; font-size: 1pt; line-height: 0; }

section.doc { page-break-before: always; }
section.doc > h1 { font-size: 20pt; margin: 0 0 5mm; padding-bottom: 2mm;
                   border-bottom: 1.5pt solid #d0d0d0; }
h2 { font-size: 14pt; margin: 7mm 0 2.5mm; page-break-after: avoid; }
h3 { font-size: 12pt; margin: 5mm 0 2mm; page-break-after: avoid; }
h4, h5, h6 { font-size: 10.5pt; margin: 4mm 0 1.5mm; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }

pre { background: #f6f8fa !important; border: 1pt solid #d8dee4; border-radius: 3pt;
      padding: 2.5mm 3mm; font-size: 8.2pt; line-height: 1.42; overflow: visible;
      white-space: pre-wrap; word-break: break-word; page-break-inside: avoid; }
code { font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace; }
:not(pre) > code { background: #f0f2f4; padding: 0.3mm 1.1mm; border-radius: 2pt;
                   font-size: 8.8pt; }

table { border-collapse: collapse; width: 100%; margin: 3mm 0; font-size: 9pt;
        page-break-inside: avoid; }
th, td { border: 0.6pt solid #c8cdd3; padding: 1.6mm 2mm; text-align: left;
         vertical-align: top; }
th { background: #eef1f4; font-weight: 700; }

blockquote { margin: 3mm 0; padding: 1.5mm 3mm; border-left: 2.5pt solid #b9c0c8;
             background: #f7f9fb; page-break-inside: avoid; }

.docusaurus-mermaid-container { text-align: center; margin: 4mm 0;
                                page-break-inside: avoid; }
/* Constrain BOTH axes so very wide or very tall graphs keep their aspect
   ratio instead of being squashed into an unreadable strip. */
.docusaurus-mermaid-container svg { width: auto !important; height: auto !important;
                                    max-width: 100%; max-height: 185mm; }

a { color: #0b5cad; text-decoration: none; }
.admonition { border-left: 3pt solid #6b7a8a; background: #f5f7f9;
              padding: 2mm 3mm; margin: 3mm 0; page-break-inside: avoid; }
.alert { --ifm-alert-foreground: #111; }
hr { border: 0; border-top: 0.6pt solid #d0d0d0; margin: 5mm 0; }
img, svg { max-width: 100%; }
.pagination-nav, .theme-doc-breadcrumbs, .hash-link { display: none !important; }
"""


def build_html(
    pages: list[tuple[str, str, str, str]],
    css_href: str,
    generated: str,
    page_numbers: dict[str, int] | None = None,
) -> str:
    """Assemble the merged print document.

    ``pages`` holds ``(category, title, route, markup)``. When
    ``page_numbers`` is omitted the table of contents renders placeholder
    numbers of the same width, so a second pass cannot shift pagination and
    invalidate the page numbers it is about to embed.
    """
    counts = page_numbers or {}
    width = len(str(max(counts.values(), default=999))) if counts else 3

    toc: list[str] = []
    current: str | None = None
    for category, title, route, _ in pages:
        if category and category != current:
            current = category
            toc.append(f'<li class="grp">{html.escape(category)}</li>')
        number = counts.get(route, 0)
        toc.append(
            f'<li><a href="#page-{slugify(route)}">'
            f'<span class="t">{html.escape(title)}</span>'
            f'<span class="n">{number if number else "&nbsp;" * width}</span></a></li>'
        )

    body: list[str] = []
    for _, _, route, markup in pages:
        marker = f'<span class="pgmark">PGMARK{slugify(route)}</span>'
        body.append(
            f'<section class="doc" id="page-{slugify(route)}">{marker}{markup}</section>'
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{TITLE} {VERSION}</title>
<link rel="stylesheet" href="{css_href}">
<style>{PRINT_CSS}</style>
</head><body>
<div class="cover">
  <h1>{TITLE}</h1>
  <p class="sub">GPU-agnostic PyTorch training, documented from source</p>
  <div class="meta">
    Version <code>{VERSION}</code><br>
    Generated {generated}<br>
    <code>github.com/bench-labs-org/Speedtronic</code><br>
    <code>speedtronic-docs.pages.dev</code>
  </div>
</div>
<div class="toc">
  <h2>Contents</h2>
  <ol>{''.join(toc)}</ol>
</div>
{''.join(body)}
</body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if not (BUILD / "index.html").exists():
        print(f"error: {BUILD} not found. Run `npm run build` first.", file=sys.stderr)
        return 1

    global BASE_URL
    BASE_URL = f"http://127.0.0.1:{args.port}"

    import http.server
    import socketserver
    import threading

    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(BUILD))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            run(args)
        finally:
            server.shutdown()
    return 0


def run(args) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        # The site defaults to dark mode; Mermaid bakes its theme into the SVG,
        # so capture under a light colour scheme or diagrams print as black
        # blocks on white paper.
        context = browser.new_context(
            viewport={"width": 1280, "height": 1800}, color_scheme="light"
        )
        page = context.new_page()
        order = read_doc_order(page)
        if not order:
            raise SystemExit("no documentation pages found in the sidebar")
        print(f"discovered {len(order)} pages from the sidebar")

        pages: list[tuple[str, str, str, str]] = []
        diagrams = 0
        for index, (category, title, route) in enumerate(order, 1):
            _, markup, count = capture_page(page, route)
            diagrams += count
            markup = strip_nav_chrome(namespace_ids(markup, f"page-{slugify(route)}"))
            pages.append((category, title or route or "Overview", route, markup))
            print(f"  [{index:2d}/{len(order)}] {route or '/'}  ({count} diagrams)")

        css_files = sorted((BUILD / "assets" / "css").glob("*.css"))
        css_href = f"assets/css/{css_files[0].name}" if css_files else ""
        generated = date.today().isoformat()
        temp = BUILD / "__print.html"

        def render(target: Path, numbers: dict[str, int] | None) -> None:
            temp.write_text(
                build_html(pages, css_href, generated, numbers), encoding="utf-8"
            )
            page.goto(f"{BASE_URL}/__print.html", wait_until="networkidle")
            page.emulate_media(media="print")
            page.wait_for_timeout(1200)
            page.pdf(
                path=str(target),
                format="Letter",
                print_background=True,
                display_header_footer=True,
                header_template=(
                    '<div style="font-size:7pt;width:100%;padding:0 16mm;color:#888;'
                    "font-family:sans-serif;display:flex;justify-content:space-between;\">"
                    f"<span>{TITLE}</span><span>Speedtronic {VERSION}</span></div>"
                ),
                footer_template=(
                    '<div style="font-size:7pt;width:100%;padding:0 16mm;color:#888;'
                    'font-family:sans-serif;text-align:center;">'
                    '<span class="pageNumber"></span> / <span class="totalPages"></span></div>'
                ),
                margin={"top": "16mm", "bottom": "14mm", "left": "0", "right": "0"},
            )

        try:
            # Pass 1: placeholder TOC numbers, then read each section's real
            # start page from the invisible markers.
            probe = BUILD / "__probe.pdf"
            render(probe, None)
            import pymupdf

            numbers: dict[str, int] = {}
            with pymupdf.open(probe) as doc:
                for _cat, _title, route, _markup in pages:
                    marker = f"PGMARK{slugify(route)}"
                    for index in range(doc.page_count):
                        if doc[index].search_for(marker):
                            numbers[route] = index + 1
                            break
            probe.unlink(missing_ok=True)
            print(f"resolved start pages for {len(numbers)}/{len(pages)} sections")

            # Pass 2: placeholders and real numbers share a width, so the table
            # of contents cannot reflow and invalidate the numbers.
            render(args.output, numbers)
        finally:
            temp.unlink(missing_ok=True)
        browser.close()

    size_kb = args.output.stat().st_size / 1024
    print(f"\nwrote {args.output} ({size_kb:.0f} KB, {len(pages)} pages, {diagrams} diagrams)")


if __name__ == "__main__":
    from functools import partial

    raise SystemExit(main())
