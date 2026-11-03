"""A check whose extra is missing skips with the reason, it does not fail (``pytest -rs`` is the checklist).

LeRobot raises ``ImportError`` from a package ``__init__`` when an optional dependency is absent
(``lerobot.datasets`` without ``datasets``); ``find_spec`` on a submodule runs that ``__init__``.
"""

from __future__ import annotations

import pytest
from checks import needs_module


def test_a_package_whose_init_refuses_skips_the_check(tmp_path, monkeypatch):
    pkg = tmp_path / "an_extra"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("raise ImportError(\"'datasets' is required but not installed\")\n")
    (pkg / "dataset.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(pytest.skip.Exception, match="needs an_extra.dataset"):
        needs_module("an_extra.dataset", "the [lerobot] extra")
