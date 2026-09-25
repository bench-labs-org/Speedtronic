#!/usr/bin/env python3
"""Generate an exhaustive AST-backed inventory of Speedtronic's local sources."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable

SITE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SITE_ROOT.parent
SOURCE_ROOT = REPO_ROOT / "src" / "speedtronic"
OUTPUT = SITE_ROOT / "docs" / "reference" / "generated-source-inventory.md"


def relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def module_name(path: Path) -> str:
    relative_path = path.relative_to(SOURCE_ROOT).with_suffix("")
    parts = list(relative_path.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(["speedtronic", *parts])


def anchor(path: Path, symbol: str, member: str | None = None) -> str:
    base = module_name(path).replace(".", "-")
    suffix = f"{base}-{symbol}"
    return f"api-{suffix}-{member}" if member else f"api-{suffix}"


def first_sentence(docstring: str | None) -> str:
    if not docstring:
        return "No module docstring."
    return re.sub(r"\s+", " ", docstring.strip().split(". ", 1)[0] + ".").strip()


def render_expr(node: ast.AST | None, limit: int = 96) -> str:
    if node is None:
        return ""
    try:
        value = ast.unparse(node).replace("|", "\\|").replace("\n", " ")
    except Exception:
        value = type(node).__name__
    return value if len(value) <= limit else value[: limit - 1] + "…"


def function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = ast.unparse(node.args)
    returns = f" -> {render_expr(node.returns, 160)}" if node.returns else ""
    return f"{prefix} {node.name}({args}){returns}"


def class_signature(node: ast.ClassDef) -> str:
    bases = ", ".join(ast.unparse(base) for base in node.bases)
    return f"class {node.name}({bases})" if bases else f"class {node.name}"


def assignment_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.extend(item.id for item in target.elts if isinstance(item, ast.Name))
    return names


def write_heading(lines: list[str], text: str, symbol_id: str, level: int = 4) -> None:
    lines.append(f"{'#' * level} {text} {{#{symbol_id}}}")
    lines.append("")


def iter_python_files(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def append_symbol_table(lines: list[str], path: Path, rows: list[tuple[str, str, int]]) -> None:
    if not rows:
        return
    lines.append("| Kind | Symbol | Line |")
    lines.append("|---|---:|---:|")
    for kind, symbol, line in rows:
        escaped = symbol.replace("|", "\\|")
        symbol_id = anchor(path, re.sub(r"[^A-Za-z0-9_.-]", "-", symbol).strip("-"))
        lines.append(f"| {kind} | `{escaped}` | {line} |")
    lines.append("")


def generate_module(lines: list[str], path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    name = module_name(path)
    lines.extend([f"### `{relative(path)}`", ""])
    lines.append(f"**Import path:** `{name}`<br />")
    lines.append(f"**Purpose:** {first_sentence(ast.get_docstring(tree))}<br />")
    lines.append(f"**Lines:** {len(source.splitlines())}")
    lines.append("")

    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = "." * (node.level or 0) + (node.module or "")
            imports.extend(f"{module}.{alias.name}" for alias in node.names)
    if imports:
        lines.append(f"**Imported modules:** {', '.join(f'`{item}`' for item in imports)}")
        lines.append("")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbol_id = anchor(path, node.name)
            write_heading(lines, f"Function `{node.name}`", symbol_id)
            lines.extend(["```python", function_signature(node), "```", ""])
            if ast.get_docstring(node):
                lines.extend([ast.get_docstring(node).strip(), ""])
        elif isinstance(node, ast.ClassDef):
            symbol_id = anchor(path, node.name)
            write_heading(lines, f"Class `{node.name}`", symbol_id)
            lines.extend(["```python", class_signature(node), "```", ""])
            if ast.get_docstring(node):
                lines.extend([ast.get_docstring(node).strip(), ""])

            rows: list[tuple[str, str, int]] = []
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    member_id = anchor(path, node.name, member.name)
                    write_heading(lines, f"Method `{member.name}`", member_id, level=5)
                    lines.extend(["```python", function_signature(member), "```", ""])
                    member_doc = ast.get_docstring(member)
                    if member_doc:
                        lines.extend([member_doc.strip(), ""])
                elif isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
                    rows.append(("field", member.target.id, member.lineno))
                elif isinstance(member, ast.Assign):
                    for member_name in assignment_names(member):
                        rows.append(("class attribute", member_name, member.lineno))
            if rows:
                lines.append("**Class attributes and fields**")
                lines.append("")
                append_symbol_table(lines, path, rows)

    module_rows: list[tuple[str, str, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            for symbol in assignment_names(node):
                module_rows.append(("constant/alias", symbol, node.lineno))
    if module_rows:
        lines.append("**Module constants and aliases**")
        lines.append("")
        append_symbol_table(lines, path, module_rows)
    lines.append("")


def generate_test_inventory(lines: list[str]) -> None:
    lines.extend(["## Test modules", "", "Every test function defined by the repository is listed here.", ""])
    for path in iter_python_files(REPO_ROOT / "tests"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
        ]
        lines.append(f"### `{relative(path)}`")
        lines.append("")
        if functions:
            lines.extend(f"- `{path.stem}::{name}`" for name in functions)
        else:
            lines.append("_No top-level test functions found._")
        lines.append("")


def generate_asset_inventory(lines: list[str]) -> None:
    lines.extend(["## Shipped configurations and examples", ""])
    for directory in ("configs", "examples"):
        for path in sorted((REPO_ROOT / directory).glob("*")):
            if path.is_file():
                lines.append(f"- `{relative(path)}`")
    lines.append("")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        "id: generated-source-inventory",
        "title: Generated Source Inventory",
        "sidebar_label: Full AST Inventory",
        "description: Exhaustive generated inventory of every local Python module, class, function, method, test, configuration, and example.",
        "---",
        "",
        "# Generated source inventory",
        "",
        ":::info",
        "This page is generated from the local repository by `scripts/generate_source_inventory.py` during every documentation build. It inventories every implementation module and every top-level class/function/method without importing PyTorch or contacting the network.",
        ":::",
        "",
        "## Implementation modules",
        "",
    ]
    for path in iter_python_files(SOURCE_ROOT):
        generate_module(lines, path)
    generate_test_inventory(lines)
    generate_asset_inventory(lines)
    OUTPUT.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {relative(OUTPUT)}")


if __name__ == "__main__":
    main()
