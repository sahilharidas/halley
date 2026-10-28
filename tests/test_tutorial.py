# pyright: basic
"""The tutorial's code blocks run, and print what the pages say they print.

Each page under ``docs/tutorial`` is a sequence of fenced ``python`` blocks; a ``text`` block that
follows one is what it prints, and a ``python`` block without one prints nothing. The pages form
two chains, the first run and reading it, then a model of your own, its data and its scenarios.
A chain's pages run in order, each page's blocks in a module seeded with the previous page's
names, and a chain starts from an empty namespace, so a page can only use what the pages it says
it continues from define. Everything runs from one temporary working directory. A page's blocks
are also written to a real file, ``<page>.py``, that the module is registered under: the engine
reads a model's source for its messages and for ``explain``, and each block is compiled at its
line in that file so that the lines agree. A block's output is compared with its page's after
the parts that vary by machine are normalised: where a message points (a line of the page's own
file becomes its line within its block, which is what the page shows as ``term.py``; a file of
the package keeps its path within the package, and its line is not compared, since the shipped
models change) and the trailing digits of a float, which another C library's ``exp`` and ``pow``
may round the other way.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import halley

TUTORIAL = Path(__file__).resolve().parent.parent / "docs" / "tutorial"
CHAINS = (("first-run", "reading-a-run"), ("your-own-model", "your-own-data", "scenarios"))
PAGES = tuple(page for chain in CHAINS for page in chain)
PACKAGE = Path(halley.__file__).resolve().parent
FENCE = re.compile(r"^```([^\n]*)\n(.*?)^```$", re.MULTILINE | re.DOTALL)
LANGUAGES = {"python", "text", "bash"}
IN_PACKAGE = ".../halley/"


def blocks(page: str) -> list[tuple[str, str | None]]:
    """The page's python blocks, each with the text block that follows it, if one does. Every
    fence is one of the languages the pages use, and a text block follows a python block."""
    fences = [
        (m.group(1), m.group(2))
        for m in FENCE.finditer((TUTORIAL / f"{page}.md").read_text(encoding="utf-8"))
    ]
    out: list[tuple[str, str | None]] = []
    for i, (lang, body) in enumerate(fences):
        assert lang in LANGUAGES, (page, lang)
        if lang == "text":
            previous = fences[i - 1][0] if i else None
            assert previous == "python", (page, i)
        if lang != "python":
            continue
        expected = fences[i + 1][1] if i + 1 < len(fences) and fences[i + 1][0] == "text" else None
        out.append((body, expected))
    return out


def normalised(text: str, page_file: Path | None = None, starts: tuple[int, ...] = ()) -> str:
    """The output with where a message points and the trailing digits of floats normalised."""

    def location(m: re.Match[str]) -> str:
        file, line = m.group(1), int(m.group(2))
        if page_file is not None and Path(file) == page_file:
            start = max(s for s in starts if s <= line)
            return f"(term.py:{line - start + 1})"
        if file.startswith(IN_PACKAGE):
            return f"({file}:LINE)"
        path = Path(file)
        if path.is_absolute() and path.resolve().is_relative_to(PACKAGE):
            return f"({IN_PACKAGE}{path.resolve().relative_to(PACKAGE).as_posix()}:LINE)"
        return m.group(0)

    text = re.sub(r"\(([^()\n]+):(\d+)\)", location, text)
    text = re.sub(r"-?\d+\.\d{7,}(?:e[-+]?\d+)?", lambda m: f"{float(m.group()):.6g}", text)
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def test_the_pages_are_in_the_toctree() -> None:
    index = (TUTORIAL / "index.md").read_text(encoding="utf-8")
    for page in (*PAGES, "where-next"):
        assert f"\n{page}\n" in index, page


def test_every_code_block_runs_and_prints_what_the_page_says(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    checked = 0
    for chain in CHAINS:
        names: dict[str, Any] = {}
        for page in chain:
            slug = page.replace("-", "_")
            path = tmp_path / f"{slug}.py"
            page_blocks = blocks(page)
            path.write_text("\n".join(code for code, _expected in page_blocks), encoding="utf-8")
            starts: list[int] = []
            line = 1
            for code, _expected in page_blocks:
                starts.append(line)
                line += code.count("\n") + 1
            module = types.ModuleType(slug)
            module.__dict__.update(names)
            module.__name__ = slug
            module.__file__ = str(path)
            monkeypatch.setitem(sys.modules, slug, module)
            for number, ((code, expected), start) in enumerate(
                zip(page_blocks, starts, strict=True), start=1
            ):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    exec(compile("\n" * (start - 1) + code, str(path), "exec"), module.__dict__)
                if expected is None:
                    assert out.getvalue() == "", (page, number, out.getvalue())
                else:
                    got = normalised(out.getvalue(), path, tuple(starts))
                    assert got == normalised(expected), (page, number)
                    checked += 1
            names = {k: v for k, v in module.__dict__.items() if not k.startswith("__")}
    assert checked == 21
