"""Style rules for the published repository, enforced rather than trusted to review.

The project is meant to be read by competition reviewers and, once public, by anyone who finds it.
These rules keep the prose plain and the diffs clean across editors, terminals and locales.

Forbidden characters are written as escape sequences so that this module stays pure ASCII and does
not match its own patterns.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    "cookiecutter-data-science",
    "prize-winner-template",
    ".venv",
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "data",
    "models",
}
EXTENSIONS = {".py", ".md", ".yaml", ".toml", ".txt"}

BANNED = {
    "em dash": re.compile("\u2014"),
    "en dash": re.compile("\u2013"),
    "curly apostrophe": re.compile("[\u2018\u2019]"),
    "curly quotes": re.compile("[\u201c\u201d]"),
    "unicode arrow": re.compile("[\u2190-\u21ff\u27f0-\u27ff\u2b00-\u2b0f]"),
    "emoji": re.compile("[\U0001f000-\U0001faff\u2600-\u27bf]"),
    "decorative banner": re.compile(r"^\s*#\s*[=\-#]{8,}\s*$", re.M),
}


def source_files():
    for path in ROOT.rglob("*"):
        if (
            path.is_file()
            and path.suffix in EXTENSIONS
            and not any(part in SKIP_DIRS for part in path.parts)
        ):
            yield path


@pytest.mark.parametrize("label", sorted(BANNED))
def test_no_banned_characters(label):
    pattern = BANNED[label]
    hits = []
    for path in source_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for match in pattern.finditer(text):
            line = text[: match.start()].count("\n") + 1
            hits.append(f"{path.relative_to(ROOT)}:{line}")
    assert not hits, f"{label} found in {len(hits)} place(s):\n" + "\n".join(hits[:20])


def test_source_files_are_discovered():
    """Guard against the scan matching nothing and passing vacuously."""
    files = list(source_files())
    assert len(files) > 20, f"only {len(files)} files scanned, the filter is too aggressive"
    assert any(p.suffix == ".py" for p in files)
    assert any(p.suffix == ".md" for p in files)
