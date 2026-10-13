"""Every runnable example (python block + text output block) in the learner-first concept pages must run and print exactly what the page shows."""
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import md_snippets  # noqa: E402

PAGES = sorted(p for p in ROOT.glob("tier-*/*.md") if md_snippets.PAIR.search(p.read_text()))


@pytest.mark.parametrize("page", PAGES, ids=[p.name for p in PAGES])
def test_examples_run_and_outputs_match(page):
    problems = md_snippets.process(page, fill=False)
    assert not problems, "\n".join(problems)


def test_at_least_one_learner_first_page_exists():
    assert PAGES, "no page with runnable examples found"
