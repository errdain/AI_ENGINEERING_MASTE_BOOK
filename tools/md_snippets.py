"""Run the Python code blocks inside concept markdown files and keep their printed output honest.

Convention used by the learner-first concept pages:

    ```python
    print(1 + 1)
    ```

    ```text
    2
    ```

A ```python block that is immediately followed by a ```text block is a *runnable example*: the text block must equal what the code prints.
Each runnable block is executed in a fresh interpreter (so every example on a page is self-contained).

    python tools/md_snippets.py --fill  FILE.md [FILE.md ...]     # rewrite the output blocks from real runs
    python tools/md_snippets.py --check FILE.md [FILE.md ...]     # exit 1 if any output block is stale or any block fails
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

PAIR = re.compile(r"```python\n(?P<code>.*?)\n```\n\n```text\n(?P<out>.*?)\n?```", re.S)


NUM = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def same_output(expected: str, actual: str, rtol: float = 0.03, atol: float = 2e-3) -> bool:
    """Equal text, with numbers allowed to differ slightly (other CPUs/BLAS/library versions round differently)."""
    if expected == actual:
        return True
    if NUM.sub("#", expected) != NUM.sub("#", actual):
        return False
    a, b = [float(x) for x in NUM.findall(expected)], [float(x) for x in NUM.findall(actual)]
    return len(a) == len(b) and all(abs(x - y) <= atol + rtol * max(abs(x), abs(y)) for x, y in zip(a, b))


def run(code: str, cwd: pathlib.Path) -> str:
    p = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120, cwd=cwd)
    if p.returncode != 0:
        raise RuntimeError(f"code failed:\n{code}\n---\n{p.stderr}")
    return p.stdout.rstrip("\n")


def process(path: pathlib.Path, fill: bool) -> list[str]:
    text = path.read_text(); problems: list[str] = []; out: list[str] = []; last = 0
    for m in PAIR.finditer(text):
        actual = run(m.group("code"), path.parent)
        if not same_output(m.group("out").rstrip("\n"), actual):
            problems.append(f"{path}: output differs for block starting {m.group('code').splitlines()[0]!r}\n--- page ---\n{m.group('out')}\n--- actual ---\n{actual}")
        out.append(text[last:m.start()]); out.append(f"```python\n{m.group('code')}\n```\n\n```text\n{actual}\n```"); last = m.end()
    out.append(text[last:])
    if fill and problems: path.write_text("".join(out))
    return problems


def main(argv: list[str]) -> int:
    mode, files = argv[0], [pathlib.Path(a) for a in argv[1:]]
    bad: list[str] = []
    for f in files: bad += process(f, fill=(mode == "--fill"))
    if mode == "--check" and bad: print("\n".join(bad)); return 1
    print(f"{len(files)} file(s) processed; {len(bad)} block(s) {'updated' if mode == '--fill' else 'stale'}"); return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
