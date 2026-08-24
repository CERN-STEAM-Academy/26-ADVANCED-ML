#!/usr/bin/env python3
"""Build the student notebooks from the solutions notebooks.

Student notebooks are **build artefacts**. Never hand-edit one: the next build will
silently discard your edit, and the two versions will drift apart in ways nobody notices
until a room full of people is looking at the wrong cell.

The convention
--------------
In a *code* cell::

    # TODO(hint): compute the group-normalised advantage
    # BEGIN SOLUTION
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    # END SOLUTION

becomes::

    # TODO: compute the group-normalised advantage
    raise NotImplementedError

The hint comment is whatever contiguous run of comment lines immediately precedes the
``BEGIN SOLUTION`` marker and starts with ``TODO(hint):``. It is preserved verbatim apart
from dropping the ``(hint)``. The replacement is indented to match the marker, so a hole
inside a function body stays inside the function body.

In a *markdown* cell::

    <!-- TODO(hint): which leg of the deadly triad did each config break? -->
    <!-- BEGIN SOLUTION -->
    CONFIG_A removes the target network, so ...
    <!-- END SOLUTION -->

becomes a blockquote inviting an answer. Everything else in the markdown - headers,
LaTeX, narrative - is preserved byte for byte, in both versions. That is deliberate: the
student and solutions notebooks must read as the same document, so that the solutions can
be published after the session without anyone having to re-find their place.

Guarantees
----------
* **Idempotent.** Running the tool on its own output changes nothing, because the output
  contains no markers.
* **Loud on unbalanced markers.** A ``BEGIN`` without an ``END``, an ``END`` without a
  ``BEGIN``, or a nested ``BEGIN`` is a hard error naming the cell and the line. A
  silently truncated exercise is worse than a failed build.
* **Outputs stripped** from student notebooks, **kept** in solutions notebooks, so the
  author can see expected results without rerunning.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from typing import Any

CODE_BEGIN = re.compile(r"^(\s*)#\s*BEGIN SOLUTION\s*$")
CODE_END = re.compile(r"^\s*#\s*END SOLUTION\s*$")
CODE_HINT = re.compile(r"^(\s*)#\s*TODO\(hint\):\s*(.*)$")
CODE_COMMENT = re.compile(r"^\s*#")

MD_BEGIN = re.compile(r"^\s*<!--\s*BEGIN SOLUTION\s*-->\s*$")
MD_END = re.compile(r"^\s*<!--\s*END SOLUTION\s*-->\s*$")
MD_HINT = re.compile(r"^\s*<!--\s*TODO\(hint\):\s*(.*?)\s*-->\s*$")

DEFAULT_HINT = "implement this"
PLACEHOLDER = "raise NotImplementedError"


class MarkerError(RuntimeError):
    """Raised on unbalanced or malformed solution markers."""


def _lines(source: Any) -> list[str]:
    """Notebook sources are either a string or a list of lines. Normalise to lines."""
    if isinstance(source, str):
        return source.splitlines(keepends=True)
    return list(source)


def strip_code(lines: list[str], where: str) -> tuple[list[str], int]:
    """Replace every marked block in a code cell. Returns (new lines, blocks removed)."""
    out: list[str] = []
    index = 0
    removed = 0

    while index < len(lines):
        line = lines[index]
        begin = CODE_BEGIN.match(line.rstrip("\n"))
        if begin is None:
            if CODE_END.match(line.rstrip("\n")):
                raise MarkerError(f"{where}: 'END SOLUTION' at line {index + 1} with no matching BEGIN")
            out.append(line)
            index += 1
            continue

        indent = begin.group(1)
        begin_line = index + 1

        # Rewrite the trailing run of hint comments we have already emitted.
        hint_found = False
        for back in range(len(out) - 1, -1, -1):
            stripped = out[back].rstrip("\n")
            if not CODE_COMMENT.match(stripped):
                break
            hint = CODE_HINT.match(stripped)
            if hint is not None:
                out[back] = f"{hint.group(1)}# TODO: {hint.group(2)}\n"
                hint_found = True
                break
        if not hint_found:
            out.append(f"{indent}# TODO: {DEFAULT_HINT}\n")
            print(f"  warning: {where} has a solution block with no TODO(hint) comment", file=sys.stderr)

        # Consume up to the matching END.
        index += 1
        closed = False
        while index < len(lines):
            inner = lines[index].rstrip("\n")
            if CODE_BEGIN.match(inner):
                raise MarkerError(f"{where}: nested 'BEGIN SOLUTION' at line {index + 1}")
            index += 1
            if CODE_END.match(inner):
                closed = True
                break
        if not closed:
            raise MarkerError(f"{where}: 'BEGIN SOLUTION' at line {begin_line} was never closed")

        out.append(f"{indent}{PLACEHOLDER}\n")
        removed += 1

    return out, removed


def strip_markdown(lines: list[str], where: str) -> tuple[list[str], int]:
    """Replace every marked block in a markdown cell with an answer placeholder."""
    out: list[str] = []
    index = 0
    removed = 0

    while index < len(lines):
        line = lines[index]
        if MD_END.match(line.rstrip("\n")):
            raise MarkerError(f"{where}: markdown 'END SOLUTION' at line {index + 1} with no matching BEGIN")
        if not MD_BEGIN.match(line.rstrip("\n")):
            out.append(line)
            index += 1
            continue

        begin_line = index + 1

        # A hint comment immediately above becomes the prompt.
        prompt = None
        for back in range(len(out) - 1, -1, -1):
            stripped = out[back].strip()
            if not stripped:
                continue
            hint = MD_HINT.match(stripped)
            if hint is not None:
                prompt = hint.group(1)
                out.pop(back)
            break

        index += 1
        closed = False
        while index < len(lines):
            inner = lines[index].rstrip("\n")
            if MD_BEGIN.match(inner):
                raise MarkerError(f"{where}: nested markdown 'BEGIN SOLUTION' at line {index + 1}")
            index += 1
            if MD_END.match(inner):
                closed = True
                break
        if not closed:
            raise MarkerError(f"{where}: markdown 'BEGIN SOLUTION' at line {begin_line} was never closed")

        heading = f"> **Your answer**{' - ' + prompt if prompt else ''}\n"
        out.extend([heading, ">\n", "> _(replace this line)_\n"])
        removed += 1

    return out, removed


def strip_notebook(notebook: dict, name: str = "notebook") -> tuple[dict, dict[str, int]]:
    """Return a student copy of a notebook plus a small report."""
    result = copy.deepcopy(notebook)
    stats = {"code_blocks": 0, "markdown_blocks": 0, "cells": len(result.get("cells", []))}

    for i, cell in enumerate(result.get("cells", [])):
        where = f"{name} cell {i} ({cell.get('cell_type')})"
        lines = _lines(cell.get("source", ""))

        if cell.get("cell_type") == "code":
            new_lines, removed = strip_code(lines, where)
            stats["code_blocks"] += removed
            cell["source"] = new_lines
            # Student notebooks ship clean: no outputs, no execution counts, no stale
            # "this ran in 40 minutes" metadata.
            cell["outputs"] = []
            cell["execution_count"] = None
            cell.get("metadata", {}).pop("execution", None)
        elif cell.get("cell_type") == "markdown":
            new_lines, removed = strip_markdown(lines, where)
            stats["markdown_blocks"] += removed
            cell["source"] = new_lines

    return result, stats


def strip_python_source(text: str, name: str = "module") -> tuple[str, int]:
    """Same transformation, applied to a plain ``.py`` file."""
    lines, removed = strip_code(text.splitlines(keepends=True), name)
    return "".join(lines), removed


def count_markers(notebook: dict) -> int:
    """How many solution blocks does this notebook still contain?"""
    total = 0
    for cell in notebook.get("cells", []):
        for line in _lines(cell.get("source", "")):
            stripped = line.rstrip("\n")
            if CODE_BEGIN.match(stripped) or MD_BEGIN.match(stripped):
                total += 1
    return total


def build(source_path: str, dest_path: str, verbose: bool = True) -> dict[str, int]:
    with open(source_path) as handle:
        notebook = json.load(handle)

    student, stats = strip_notebook(notebook, name=os.path.basename(source_path))

    # Idempotency check: the output must contain no markers, so re-running is a no-op.
    remaining = count_markers(student)
    if remaining:
        raise MarkerError(f"{dest_path}: {remaining} solution markers survived stripping")

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    with open(dest_path, "w") as handle:
        json.dump(student, handle, indent=1, ensure_ascii=False)
        handle.write("\n")

    if verbose:
        print(
            f"  {source_path} -> {dest_path}: {stats['cells']} cells, "
            f"{stats['code_blocks']} code holes, {stats['markdown_blocks']} prose holes"
        )
    return stats


def default_pairs(notebooks_dir: str) -> list[tuple[str, str]]:
    """Every ``*_solutions.ipynb`` in the directory, paired with its student output."""
    pairs = []
    for entry in sorted(os.listdir(notebooks_dir)):
        if entry.endswith("_solutions.ipynb"):
            source = os.path.join(notebooks_dir, entry)
            dest = os.path.join(notebooks_dir, entry.replace("_solutions.ipynb", "_student.ipynb"))
            pairs.append((source, dest))
    return pairs


def main(argv: list[str] | None = None) -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", help="a single solutions notebook (default: all of them)")
    parser.add_argument("dest", nargs="?", help="output path for that notebook")
    parser.add_argument(
        "--notebooks-dir",
        default=os.path.join(here, "notebooks"),
        help="directory scanned for *_solutions.ipynb when no source is given",
    )
    parser.add_argument(
        "--strip-package",
        metavar="DEST_DIR",
        help=(
            "also write a hole-y copy of the rlpractice package to DEST_DIR. Use this if "
            "you want to hand students a distribution in which the reference reward "
            "implementations are not sitting in rewards.py."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    verbose = not args.quiet

    if args.source and not args.dest:
        parser.error("give both a source and a destination, or neither")

    pairs = [(args.source, args.dest)] if args.source else default_pairs(args.notebooks_dir)
    if not pairs:
        print(f"no *_solutions.ipynb found in {args.notebooks_dir}", file=sys.stderr)
        return 1

    if verbose:
        print("building student notebooks:")
    try:
        for source, dest in pairs:
            build(source, dest, verbose=verbose)

        if args.strip_package:
            package_dir = os.path.join(here, "rlpractice")
            out_dir = os.path.join(args.strip_package, "rlpractice")
            os.makedirs(out_dir, exist_ok=True)
            if verbose:
                print("stripping package:")
            for entry in sorted(os.listdir(package_dir)):
                if not entry.endswith(".py"):
                    continue
                with open(os.path.join(package_dir, entry)) as handle:
                    text = handle.read()
                stripped, removed = strip_python_source(text, name=entry)
                with open(os.path.join(out_dir, entry), "w") as handle:
                    handle.write(stripped)
                if verbose and removed:
                    print(f"  {entry}: {removed} holes")
    except MarkerError as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
