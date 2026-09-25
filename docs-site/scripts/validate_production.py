#!/usr/bin/env python3
"""Reject local-only URLs in a production Cloudflare Pages build."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "build"


def main() -> None:
    if not ROOT.exists():
        raise SystemExit("build directory is missing")
    offenders = []
    for path in ROOT.rglob("*"):
        if path.suffix not in {".html", ".xml", ".json"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "localhost:3000" in text:
            offenders.append(str(path.relative_to(ROOT)))
    if offenders:
        raise SystemExit("local URLs found in production build: " + ", ".join(offenders))
    print(f"production URL check passed: {ROOT}")


if __name__ == "__main__":
    main()
