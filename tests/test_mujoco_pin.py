"""mujoco, and the mujoco-warp that steps it, are capped below their next major release until that engine is ported."""

from pathlib import Path

import pytest
import tomllib
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

EXTRAS = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]["optional-dependencies"]


def _mujoco() -> Requirement:
    return next(r for r in map(Requirement, EXTRAS["mujoco"]) if r.name == "mujoco")


@pytest.mark.parametrize(
    ("version", "admitted"),
    [("3.3.4", True), ("3.11.0", True), ("3.3.3", False), ("4.0.0", False), ("4.1.0", False)],
)
def test_the_mujoco_extra_caps_below_4(version: str, admitted: bool) -> None:
    # mujoco 4.0.0 (PyPI 2026-11-06) is a breaking major; the extra pins >=3.3.4,<4 until it is ported.
    assert _mujoco().specifier.contains(Version(version)) is admitted


def test_every_mujoco_requirement_is_capped() -> None:
    found = [r for group in EXTRAS.values() for r in map(Requirement, group) if r.name == "mujoco"]
    assert found, "no mujoco requirement declared in any extra"
    for req in found:
        assert req.specifier.contains(Version("3.11.0"))
        assert not req.specifier.contains(Version("4.0.0"))


@pytest.mark.parametrize(
    ("version", "admitted"),
    [("3.11.0", True), ("3.17.0", True), ("3.10.0", False), ("4.0.0", False), ("4.0.1", False)],
)
def test_every_mujoco_warp_requirement_is_capped_below_4(version: str, admitted: bool) -> None:
    found = [
        r for group in EXTRAS.values() for r in map(Requirement, group) if canonicalize_name(r.name) == "mujoco-warp"
    ]
    assert found, "no mujoco-warp requirement declared in any extra"
    for req in found:
        assert req.specifier.contains(Version(version)) is admitted


def test_an_extra_with_mujoco_warp_also_requires_mujoco() -> None:
    for extra, group in EXTRAS.items():
        names = {canonicalize_name(Requirement(r).name) for r in group}
        if "mujoco-warp" in names:
            assert "mujoco" in names, f"[{extra}] installs mujoco-warp but does not require mujoco"
