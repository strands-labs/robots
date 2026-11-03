"""The zenoh range the extras declare is the one LeRobot main resolves, so installing both never fights."""

from pathlib import Path

import pytest
import tomllib
from packaging.requirements import Requirement
from packaging.version import Version

EXTRAS = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]["optional-dependencies"]


def _zenoh(extra: str) -> Requirement:
    return next(r for r in map(Requirement, EXTRAS[extra]) if r.name == "eclipse-zenoh")


@pytest.mark.parametrize("extra", ["lerobot", "mesh"])
@pytest.mark.parametrize(
    ("version", "admitted"),
    [("1.9.0", True), ("1.10.0", True), ("1.10.2", True), ("1.8.0", False), ("1.11.0", False)],
)
def test_the_zenoh_range_matches_lerobot_main(extra: str, version: str, admitted: bool) -> None:
    # LeRobot main's `remote` extra: eclipse-zenoh>=1.9.0,<1.11.0 (it resolves 1.10.2 today).
    assert _zenoh(extra).specifier.contains(Version(version)) is admitted


def test_both_extras_pin_the_same_zenoh() -> None:
    assert _zenoh("lerobot").specifier == _zenoh("mesh").specifier
