#!/usr/bin/env python3
"""Fail the docs build when local source/test/config/example coverage drifts."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

SITE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SITE_ROOT.parent
SOURCE_ROOT = REPO_ROOT / "src" / "speedtronic"
INVENTORY = SITE_ROOT / "docs" / "reference" / "generated-source-inventory.md"
SIDEBARS = SITE_ROOT / "sidebars.js"


def relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def module_name(path: Path) -> str:
    parts = list(path.relative_to(SOURCE_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(["speedtronic", *parts])


def expected_anchor(path: Path, symbol: str, member: str | None = None) -> str:
    base = module_name(path).replace(".", "-")
    result = f"api-{base}-{symbol}"
    return f"{result}-{member}" if member else result


def required_ids() -> set[str]:
    ids: set[str] = set()
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ids.add(expected_anchor(path, node.name))
            elif isinstance(node, ast.ClassDef):
                ids.add(expected_anchor(path, node.name))
                for member in node.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        ids.add(expected_anchor(path, node.name, member.name))
    return ids


def main() -> None:
    errors: list[str] = []
    if not INVENTORY.exists():
        print(f"error: generated inventory is missing: {relative(INVENTORY)}", file=sys.stderr)
        raise SystemExit(1)
    text = INVENTORY.read_text(encoding="utf-8")
    sidebars = SIDEBARS.read_text(encoding="utf-8")

    source_files = [
        path
        for path in sorted(SOURCE_ROOT.rglob("*.py"))
        if "__pycache__" not in path.parts
    ]
    for path in source_files:
        if relative(path) not in text:
            errors.append(f"module path missing from inventory: {relative(path)}")

    for path in sorted((REPO_ROOT / "tests").rglob("*.py")):
        if relative(path) not in text:
            errors.append(f"test path missing from inventory: {relative(path)}")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                test_id = f"{path.stem}::{node.name}"
                if test_id not in text:
                    errors.append(f"test missing from inventory: {test_id}")

    for directory in ("configs", "examples"):
        for path in sorted((REPO_ROOT / directory).glob("*")):
            if path.is_file() and relative(path) not in text:
                errors.append(f"shipped file missing from inventory: {relative(path)}")

    for symbol_id in sorted(required_ids()):
        if f"#{symbol_id}" not in text:
            errors.append(f"symbol anchor missing from inventory: {symbol_id}")

    if "'reference/generated-source-inventory'" not in sidebars:
        errors.append("generated source inventory is not listed in sidebars.js")

    doc_files = sorted((SITE_ROOT / "docs").rglob("*.md"))
    doc_files += sorted((SITE_ROOT / "docs").rglob("*.mdx"))
    for path in doc_files:
        route = path.relative_to(SITE_ROOT / "docs").with_suffix("").as_posix()
        if route == "reference/generated-source-inventory":
            continue
        if f"'{route}'" not in sidebars and f'"{route}"' not in sidebars:
            errors.append(f"document is not represented in sidebar: {route}")

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)

    module_count = len(source_files)
    symbol_count = len(required_ids())
    test_count = sum(
        1
        for path in (REPO_ROOT / "tests").rglob("*.py")
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    )
    print(f"coverage ok: {module_count} modules, {symbol_count} declarations, {test_count} tests")


if __name__ == "__main__":
    main()
