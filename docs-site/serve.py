#!/usr/bin/env python3
"""Serve the compiled Docusaurus site without requiring Node.js."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "build"


class CleanURLHandler(SimpleHTTPRequestHandler):
    """Resolve Docusaurus clean URLs to .html files or directory indexes."""

    def translate_path(self, path: str) -> str:
        relative = path.split("?", 1)[0].split("#", 1)[0].lstrip("/")
        requested = (ROOT / relative).resolve()
        if ROOT not in requested.parents and requested != ROOT:
            return str(ROOT / "__invalid_path__")
        if requested.is_dir():
            index = requested / "index.html"
            if index.exists():
                return str(index)
        html = requested.with_suffix(".html")
        if html.exists():
            return str(html)
        return str(requested)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve docs-site/build locally")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3000)
    args = parser.parse_args()
    if not ROOT.exists():
        raise SystemExit("build directory is missing; run: npm run build")
    handler = partial(CleanURLHandler, directory=str(ROOT))
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        print(f"Serving {ROOT} at http://{args.host}:{args.port}/")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
