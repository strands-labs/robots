"""mujoco is capped below its next major release until that engine is ported."""

from pathlib import Path

import pytest
import tomllib
from packaging.requirements import Requirement
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
