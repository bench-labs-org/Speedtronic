#!/usr/bin/env python3
"""Post-build gate for the production Cloudflare Pages bundle.

Docusaurus's own ``onBrokenLinks`` check only understands routes, so it cannot
validate links that point at files in ``static/`` (notably the documentation
PDF). This script is stricter than the built-in check: it walks every internal
``href``/``src`` in the emitted HTML and requires the target to exist on disk,
covering both routes and static assets.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"

# Assets that are expected to be copied verbatim from static/.
REQUIRED_FILES = [
    Path("pdf/speedtronic-documentation.pdf"),
]

HREF = re.compile(r'(?:href|src)="([^"]+)"')
SKIP_PREFIXES = ("http://", "https://", "mailto:", "tel:", "data:", "#", "//")


def route_exists(target: str) -> bool:
    """Return True when a root-relative path resolves to a built file."""
    path = target.split("#", 1)[0].split("?", 1)[0].strip()
    if not path or path == "/":
        candidate = BUILD / "index.html"
    else:
        rel = path.lstrip("/")
        base = BUILD / rel
        # /docs/x  -> docs/x.html, docs/x/index.html, or docs/x
        options = [
            base,
            base.with_suffix(".html"),
            base / "index.html",
        ]
        candidate = next((o for o in options if o.is_file()), None)
        if candidate is None:
            return False
    return candidate.is_file()


def main() -> int:
    if not BUILD.is_dir():
        print("build directory is missing", file=sys.stderr)
        return 1

    problems: list[str] = []

    missing_assets = [f for f in REQUIRED_FILES if not (BUILD / f).is_file()]
    problems += [f"required static asset missing: {f}" for f in missing_assets]

    checked = 0
    for page in sorted(BUILD.rglob("*.html")):
        text = page.read_text(encoding="utf-8", errors="ignore")
        if "localhost:3000" in text or "127.0.0.1" in text:
            problems.append(f"local URL in {page.relative_to(BUILD)}")
        for target in HREF.findall(text):
            if target.startswith(SKIP_PREFIXES):
                continue
            if not target.startswith("/"):
                continue
            checked += 1
            if not route_exists(target):
                problems.append(
                    f"broken link on {page.relative_to(BUILD)}: {target}"
                )

    if problems:
        for problem in sorted(set(problems)):
            print(f"  {problem}", file=sys.stderr)
        print(f"\nproduction validation failed ({len(set(problems))} problems)", file=sys.stderr)
        return 1

    size = sum((BUILD / f).stat().st_size for f in REQUIRED_FILES) / 1024
    print(f"production URL check passed: {BUILD}")
    print(f"internal links checked: {checked} (routes and static assets)")
    print(f"static assets present: {len(REQUIRED_FILES)} ({size:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
