"""Every module imports, and every documented entry point exists.

Six modules once shipped unimportable: they referenced names that had been dropped from config.py
during a restructure, so the whole training path was dead code that no test touched. Nothing caught
it because the test suite only ever imported the four modules it exercised directly.

These tests are deliberately shallow. They import and they check that advertised callables exist;
they run no model and need no data. That is enough to catch a broken import or a Makefile target
pointing at a module that was renamed.
"""

import importlib
import pkgutil
import re
from pathlib import Path

import pytest

import waxal_asr

ROOT = Path(__file__).resolve().parents[1]


def _all_modules() -> list[str]:
    """Every importable module in the package, found by walking it rather than by a hand list."""
    found = []
    for info in pkgutil.walk_packages(waxal_asr.__path__, prefix="waxal_asr."):
        found.append(info.name)
    return sorted(found)


@pytest.mark.parametrize("name", _all_modules())
def test_module_imports(name):
    # Heavy third-party imports live inside functions, so importing a module must stay cheap and
    # must not require torch, transformers or any downloaded weight.
    importlib.import_module(name)


class TestEntryPoints:
    """The commands the documentation tells a reader to run must resolve to real callables."""

    @pytest.mark.parametrize(
        "module", ["waxal_asr.lid", "waxal_asr.analysis", "waxal_asr.modeling.predict",
                   "waxal_asr.modeling.train"]
    )
    def test_module_has_a_main(self, module):
        assert callable(getattr(importlib.import_module(module), "main", None)), (
            f"{module} is invoked with python -m in the Makefile or the docs, so it needs main()"
        )

    def test_console_scripts_resolve(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        block = text.split("[project.scripts]", 1)[1].split("[", 1)[0]
        entries = re.findall(r"^\s*[\w-]+\s*=\s*\"([\w.]+):(\w+)\"", block, re.M)
        assert entries, "no console scripts found to check"
        for module, attr in entries:
            assert callable(getattr(importlib.import_module(module), attr, None)), (
                f"console script points at {module}:{attr}, which does not resolve"
            )

    def test_makefile_module_targets_exist(self):
        # `python -m waxal_asr.<x>` in the Makefile must name a module that exists. Three targets
        # once named modules that had been renamed, so make lid, make data and make train all failed.
        text = (ROOT / "Makefile").read_text(encoding="utf-8")
        for module in set(re.findall(r"-m \$\(PROJECT\)\.([\w.]+)", text)):
            importlib.import_module(f"waxal_asr.{module}")
