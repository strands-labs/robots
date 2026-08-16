"""The declared mujoco floor must be a release the backend can actually drive.

Every scene this backend builds or edits goes through MuJoCo's ``MjSpec``
procedural API (:mod:`strands_robots.simulation.mujoco.spec_builder`), and that
API arrives in pieces across the 3.x line. Measured against the published
wheels, on a ``Robot("so101", mode="sim")`` smoke that adds an object, steps and
renders:

===============  ===========================================================
mujoco           result
===============  ===========================================================
3.2.0            ``TypeError: Unregistered type : mjOption_``
3.2.7            ``MjVisual`` has no attribute ``global_``
3.3.0 - 3.3.2    module ``mujoco`` has no attribute ``mjtLightType``
3.3.3            ``MjSpec`` has no attribute ``delete``
3.3.4 - 3.4.0    smoke passes, but ``add_robot(urdf_path=...)`` is refused with
                 ``Could not find decoder for resource '<path>.urdf'``
3.5.0 and later  works
===============  ===========================================================

The packaging pins nevertheless declared ``mujoco>=3.2.0``, eight releases below
the first usable one. That is not a cosmetic bound: a resolve that lands on any
of those releases - which pip and uv are free to do, and will do whenever
another requirement caps mujoco - installs a package whose every simulation call
fails with a raw ``AttributeError`` naming a private MuJoCo binding
(``'mujoco._specs.MjSpec' object has no attribute 'delete'``). Nothing in that
message says the install is too old, so it reads as a defect in this package.

Two statements are pinned here, both reading the floor from
:data:`strands_robots.simulation.mujoco.backend._MUJOCO_API_FLOOR` rather than
restating it, so the pins, the refusal and the docs cannot drift apart:

1. every declared ``mujoco`` specifier floors at the API floor and caps the
   major, and
2. an installed build below the floor is refused at the import funnel, naming
   the version, the floor and the remedy - instead of being discovered later as
   an ``AttributeError``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

from strands_robots.simulation.mujoco import backend

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

_FLOOR = Version(".".join(str(part) for part in backend._MUJOCO_API_FLOOR))

# Releases whose MjSpec API cannot carry this backend, and the reason each was
# measured to fail. `mujoco-warp` is a different distribution and is not read.
_BELOW_THE_FLOOR = ["3.2.0", "3.2.7", "3.3.0", "3.3.3", "3.3.4", "3.4.0"]
# Releases that work, plus the shapes a version string arrives in.
_AT_OR_ABOVE_THE_FLOOR = ["3.5.0", "3.6.0", "3.10.0", "3.11.0"]


def _declared_mujoco_specifiers() -> dict[str, str]:
    """Map each extra declaring ``mujoco`` to the specifier it declares."""
    pyproject = tomllib.loads(_PYPROJECT.read_text())
    declared: dict[str, str] = {}
    groups: list[tuple[str, list[str]]] = [("project.dependencies", pyproject["project"].get("dependencies", []))]
    groups += list(pyproject["project"].get("optional-dependencies", {}).items())
    for group, requirements in groups:
        for raw in requirements:
            requirement = Requirement(raw)
            if requirement.name == "mujoco":
                declared[group] = str(requirement.specifier)
    return declared


class TestTheDeclaredPinsStateTheApiFloor:
    """Every ``mujoco`` specifier must admit only releases the backend drives."""

    def test_some_extra_declares_mujoco(self) -> None:
        # Guards the two tests below against passing vacuously if the specifiers
        # move somewhere this parser does not read.
        assert _declared_mujoco_specifiers(), "no extra declares mujoco"

    def test_no_declared_specifier_admits_a_release_below_the_floor(self) -> None:
        offenders = {
            group: (specifier, admitted)
            for group, specifier in _declared_mujoco_specifiers().items()
            if (admitted := [v for v in _BELOW_THE_FLOOR if Requirement(f"mujoco{specifier}").specifier.contains(v)])
        }
        assert not offenders, (
            f"these specifiers admit mujoco releases the backend cannot drive (floor {_FLOOR}): {offenders}"
        )

    def test_every_declared_specifier_caps_the_major(self) -> None:
        # Dependency-bound convention: a >=1.0 project is capped at the next
        # major. An uncapped mujoco floor lets a 4.x resolve in untested.
        uncapped = {
            group: specifier
            for group, specifier in _declared_mujoco_specifiers().items()
            if Requirement(f"mujoco{specifier}").specifier.contains("4.0.0")
        }
        assert not uncapped, f"these mujoco specifiers admit an untested 4.x: {uncapped}"


class TestAnUnderfloorBuildIsRefusedNotDiscovered:
    """The import funnel must name the version, the floor and the remedy."""

    @pytest.mark.parametrize("version", _BELOW_THE_FLOOR)
    def test_a_release_below_the_floor_is_refused(self, version: str) -> None:
        message = backend._mujoco_api_floor_error(version)
        assert message is not None, f"mujoco {version} cannot drive this backend but was accepted"
        assert version in message
        assert str(_FLOOR) in message
        # The remedy has to be runnable, not just a complaint.
        assert "uv pip install" in message

    @pytest.mark.parametrize("version", _AT_OR_ABOVE_THE_FLOOR)
    def test_a_release_at_or_above_the_floor_is_accepted(self, version: str) -> None:
        assert backend._mujoco_api_floor_error(version) is None

    def test_a_version_that_cannot_be_read_is_not_refused(self) -> None:
        # An unreadable version is not evidence of an old build; refusing on it
        # would reject a source build over its version string alone.
        assert backend._mujoco_api_floor_error("") is None
        assert backend._mujoco_api_floor_error("unknown") is None

    def test_a_development_suffix_is_read_as_its_release(self) -> None:
        assert backend._mujoco_api_floor_error("3.4.0.dev123") is not None
        assert backend._mujoco_api_floor_error("3.5.0.dev123") is None

    def test_the_funnel_refuses_an_underfloor_build_and_does_not_cache_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub = SimpleNamespace(__version__="3.4.0")
        monkeypatch.setattr(backend, "_mujoco", None)
        monkeypatch.setattr("strands_robots.utils.require_optional", lambda *a, **k: stub)

        for _ in range(2):
            # Twice: a refused build must be re-reported rather than served from
            # the cache, so the second caller is not told the module is fine.
            with pytest.raises(ImportError, match="3.4.0"):
                backend._ensure_mujoco()
        assert backend._mujoco is None

    def test_the_funnel_accepts_a_build_at_the_floor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stub = SimpleNamespace(__version__=str(_FLOOR))
        monkeypatch.setattr(backend, "_mujoco", None)
        monkeypatch.setattr("strands_robots.utils.require_optional", lambda *a, **k: stub)

        assert backend._ensure_mujoco() is stub

    def test_the_installed_build_satisfies_the_floor(self) -> None:
        # Non-vacuous: the suite itself runs on a build the floor admits, so the
        # refusal above cannot be hiding behind an install nobody has.
        mj = pytest.importorskip("mujoco")
        assert backend._mujoco_api_floor_error(mj.__version__) is None


class TestTheCapabilitiesTheFloorExistsFor:
    """The floor is derived from named APIs; a release dropping one must fail."""

    def test_the_installed_build_exposes_the_spec_api_the_backend_calls(self) -> None:
        mj = pytest.importorskip("mujoco")
        spec = mj.MjSpec()
        missing = [
            name for name in ("compiler", "attach", "delete", "body", "mesh", "recompile") if not hasattr(spec, name)
        ]
        assert not missing, f"MjSpec is missing {missing}; the floor needs re-deriving"
        assert hasattr(mj, "mjtLightType")
        assert hasattr(spec.visual, "global_")
