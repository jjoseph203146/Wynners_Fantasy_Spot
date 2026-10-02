#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path

TARGET = Path("/home/mwynn/nfl_data_engine/app.py")


def src_segment(text: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(text, node)
    if segment is None:
        return ast.dump(node, include_attributes=False)
    return segment


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    tree = ast.parse(text)

    print("=" * 100)
    print("READ-ONLY APP.PY LATE-SWAP STRUCTURE DIAGNOSTIC")
    print("=" * 100)
    print(f"Target: {TARGET}")
    print()

    slate_defs = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "slate_pool"
    ]

    print(f"slate_pool() definitions: {len(slate_defs)}")
    for i, node in enumerate(slate_defs, 1):
        print(
            f"  [{i}] lines {node.lineno}-{node.end_lineno}"
        )
    print()

    print("Assignments involving selected_pool:")
    selected_hits = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            else:
                targets = [node.target]

            target_names = []
            for target in targets:
                if isinstance(target, ast.Name):
                    target_names.append(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    target_names.extend(
                        elt.id
                        for elt in target.elts
                        if isinstance(elt, ast.Name)
                    )

            if "selected_pool" in target_names:
                selected_hits.append(node)

    if not selected_hits:
        print("  NONE")
    else:
        for i, node in enumerate(
            sorted(selected_hits, key=lambda n: n.lineno),
            1,
        ):
            print(
                f"\n  [{i}] lines {node.lineno}-{node.end_lineno}"
            )
            print(src_segment(text, node))
    print()

    print("Calls to slate_pool(...):")
    slate_calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "slate_pool"
    ]

    if not slate_calls:
        print("  NONE")
    else:
        for i, node in enumerate(
            sorted(slate_calls, key=lambda n: n.lineno),
            1,
        ):
            print(
                f"\n  [{i}] lines {node.lineno}-{node.end_lineno}"
            )
            print(src_segment(text, node))
    print()

    print("Calls to render_late_swap_stage1(...):")
    late_calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "render_late_swap_stage1"
    ]

    if not late_calls:
        print("  NONE")
    else:
        for i, node in enumerate(
            sorted(late_calls, key=lambda n: n.lineno),
            1,
        ):
            print(
                f"\n  [{i}] lines {node.lineno}-{node.end_lineno}"
            )
            print(src_segment(text, node))
    print()

    print("Nearby source around every selected_pool assignment:")
    lines = text.splitlines()
    for i, node in enumerate(
        sorted(selected_hits, key=lambda n: n.lineno),
        1,
    ):
        start = max(1, node.lineno - 8)
        end = min(len(lines), node.end_lineno + 8)

        print()
        print("-" * 100)
        print(
            f"selected_pool context [{i}] lines {start}-{end}"
        )
        print("-" * 100)

        for line_no in range(start, end + 1):
            print(
                f"{line_no:>6}: {lines[line_no - 1]}"
            )

    print()
    print("=" * 100)
    print("DIAGNOSTIC COMPLETE — NO FILES MODIFIED")
    print("=" * 100)


if __name__ == "__main__":
    main()
