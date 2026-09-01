"""The dashboard's server dependencies are declared, and absent ones name the extra.

`[tool.hatch.build.targets.wheel]` packages `strands_robots` entire, so
`strands_robots/dashboard/` ships in every install of `strands-robots` whether or
not the installer asked for a web dashboard. That makes the dashboard's four
server dependencies a packaging contract rather than a developer convenience:
they were once declared only in `[tool.hatch.envs.default].dependencies`, a hatch
development environment that reaches no installed copy, so a PyPI install shipped
the modules and none of their imports and `python -m strands_robots dashboard`
answered `ModuleNotFoundError: No module named 'fastapi'`.

Two failure modes are pinned here, because they are independent and the suite
went green over both:

1. The extra is declared and reachable. `pip` exits 0 on an unknown extra while
   installing none of it, so a written `strands-robots[dashboard]` that
   `[project.optional-dependencies]` does not declare reports success and then
   fails later, at a point that no longer looks related to the install.
2. The refusal names the extra. A bare `ModuleNotFoundError` on `fastapi` sends
   the reader looking for the fault in their own virtualenv; naming
   `strands-robots[dashboard]` points at the thing that actually supplies it.

The drift guard in `test_the_dev_environment_does_not_re_pin_them` is the one
that matters over time: the duplicate pins are gone *because* the hatch
environment sets `features = ["all"]` and `all` now folds the extra in. If a
later edit re-adds literal pins there, the dev environment keeps passing while an
installed copy silently regresses to the original defect -- which is exactly how
this shipped the first time.
"""

from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# The modules `strands_robots/dashboard/__init__.py` gates on. `python-multipart`
# is deliberately absent: it backs FastAPI's form parsing rather than being
# imported by this package, so it has no module name worth naming in a refusal.
# `jwt` IS here: `auth` imports it to sign the operator session token, so an
# install without it fails on the one module it backs.
GATED_MODULES = ("fastapi", "uvicorn", "webauthn", "jwt")

# Distribution names the extra must supply. `python-multipart` IS required here --
# it ships with the extra even though nothing imports it directly.
EXPECTED_REQUIREMENTS = ("fastapi", "uvicorn", "webauthn", "python-multipart", "PyJWT")


def _optional_dependencies() -> dict[str, list[str]]:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]["optional-dependencies"]


def _dev_environment_dependencies() -> list[str]:
    with PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    return data["tool"]["hatch"]["envs"]["default"].get("dependencies", [])


def test_the_dashboard_extra_is_declared() -> None:
    """`pip install 'strands-robots[dashboard]'` resolves to something."""
    extras = _optional_dependencies()
    assert "dashboard" in extras, (
        "the dashboard extra is not declared, so every written "
        "'strands-robots[dashboard]' install hint is a no-op that pip reports as "
        f"success. Declared extras: {sorted(extras)}"
    )


@pytest.mark.parametrize("distribution", EXPECTED_REQUIREMENTS)
def test_the_extra_supplies_every_server_dependency(distribution: str) -> None:
    """Each of the four reaches an installed copy through the extra."""
    declared = _optional_dependencies()["dashboard"]
    names = [requirement.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip() for requirement in declared]
    assert distribution in names, (
        f"{distribution!r} is not in the dashboard extra, so an install that asks "
        f"for the extra still cannot serve the dashboard. Declared: {declared}"
    )


def test_the_extra_is_folded_into_all() -> None:
    """`pip install 'strands-robots[all]'` serves the dashboard.

    `docs/dashboard/index.md` documents `[all]` as an install command, and the
    hatch development environment reaches the four dependencies only through this
    fold -- so it is load-bearing for both the docs and CI.
    """
    assert "strands-robots[dashboard]" in _optional_dependencies()["all"], (
        "the dashboard extra is not folded into `all`, so `pip install "
        "'strands-robots[all]'` -- which docs/dashboard/index.md tells operators "
        "to run -- installs a dashboard it cannot start"
    )


@pytest.mark.parametrize("distribution", EXPECTED_REQUIREMENTS)
def test_the_dev_environment_does_not_re_pin_them(distribution: str) -> None:
    """The dev environment resolves them through the extra, not a second copy.

    A literal pin here would keep `hatch run test` green while an installed copy
    regressed to the original defect, so the duplicate is the drift risk rather
    than the redundancy.
    """
    pinned = [
        requirement
        for requirement in _dev_environment_dependencies()
        if requirement.split(">")[0].split("<")[0].split("=")[0].strip() == distribution
    ]
    assert not pinned, (
        f"{distribution!r} is pinned in [tool.hatch.envs.default].dependencies as "
        f"{pinned}. That environment sets features = ['all'], which already folds "
        "in the dashboard extra, so this copy can drift from the extra while the "
        "dev environment stays green and installed copies regress"
    )


def test_an_absent_dependency_refuses_by_naming_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing the package without the extra names the extra, not just the module.

    Python executes a package's ``__init__`` before any submodule, so the gate
    there is what covers ``server``, ``auth`` and ``record_api`` at once. This
    drives the real import path with the modules made unimportable rather than
    asserting on the source, so deleting the gate fails this test.
    """
    import strands_robots.utils as utils

    for name in GATED_MODULES:
        # `require_optionals` memoises successful imports, and a None entry in
        # sys.modules is what makes importlib raise ImportError for an installed
        # module -- together they simulate the extra being absent.
        utils._lazy_modules.pop(name, None)
        monkeypatch.setitem(sys.modules, name, None)

    monkeypatch.delitem(sys.modules, "strands_robots.dashboard", raising=False)

    with pytest.raises(ImportError) as excinfo:
        importlib.import_module("strands_robots.dashboard")

    message = str(excinfo.value)
    assert "strands-robots[dashboard]" in message, (
        "the refusal does not name the extra that supplies the missing module, so "
        f"it reads as a broken environment. Got: {message!r}"
    )
    for name in GATED_MODULES:
        assert name in message, (
            f"{name!r} is missing from the refusal, so a caller fixing one "
            f"dependency at a time needs another round trip. Got: {message!r}"
        )
