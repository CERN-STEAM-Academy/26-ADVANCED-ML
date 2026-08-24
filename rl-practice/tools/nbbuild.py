#!/usr/bin/env python3
"""Build ``.ipynb`` files from plain Python cell lists.

Authoring notebooks as raw JSON is miserable and error-prone: source is a list of lines
with trailing newlines, outputs and execution counts have to be kept consistent, and a
single misplaced comma produces a file Jupyter refuses to open. So notebook *content* is
written as ordinary Python here - ``md("...")`` and ``code("...")`` - and this module
turns it into a valid notebook.

The solutions notebooks are then executed once with ``nbconvert --execute`` to fill in
their outputs, because the spec wants outputs kept in the solutions and stripped from the
student versions.
"""

from __future__ import annotations

import json
import os
from typing import Any

KERNELSPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}

LANGUAGE_INFO = {
    "codemirror_mode": {"name": "ipython", "version": 3},
    "file_extension": ".py",
    "mimetype": "text/x-python",
    "name": "python",
    "nbconvert_exporter": "python",
    "pygments_lexer": "ipython3",
    "version": "3.11.9",
}


def _source_lines(text: str) -> list[str]:
    """Notebook ``source`` is a list of lines, each keeping its trailing newline."""
    text = text.strip("\n")
    if not text:
        return []
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def md(text: str) -> dict[str, Any]:
    """A markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": _source_lines(text)}


def code(text: str) -> dict[str, Any]:
    """A code cell, with no outputs; outputs are filled in by executing the notebook."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source_lines(text),
    }


def notebook(cells: list[dict[str, Any]], title: str = "") -> dict[str, Any]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": KERNELSPEC,
            "language_info": LANGUAGE_INFO,
            **({"title": title} if title else {}),
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(cells: list[dict[str, Any]], path: str, title: str = "") -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        json.dump(notebook(cells, title), handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    n_code = sum(1 for cell in cells if cell["cell_type"] == "code")
    print(f"wrote {path}: {len(cells)} cells ({n_code} code, {len(cells) - n_code} markdown)")
    return path


def validate(path: str) -> None:
    """Parse with nbformat and check the solution markers are balanced."""
    import nbformat

    notebook_obj = nbformat.read(path, as_version=4)
    nbformat.validate(notebook_obj)

    opens = closes = 0
    for i, cell in enumerate(notebook_obj.cells):
        source = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        for marker, delta in (("BEGIN SOLUTION", 1), ("END SOLUTION", -1)):
            count = source.count(marker)
            if delta > 0:
                opens += count
            else:
                closes += count
        depth = 0
        for line in source.splitlines():
            if "BEGIN SOLUTION" in line:
                depth += 1
            elif "END SOLUTION" in line:
                depth -= 1
            if depth < 0:
                raise ValueError(f"{path} cell {i}: END SOLUTION before BEGIN SOLUTION")
        if depth != 0:
            raise ValueError(f"{path} cell {i}: unbalanced solution markers (depth {depth})")

    if opens != closes:
        raise ValueError(f"{path}: {opens} BEGIN markers but {closes} END markers")
    print(f"validated {path}: {len(notebook_obj.cells)} cells, {opens} solution blocks")
