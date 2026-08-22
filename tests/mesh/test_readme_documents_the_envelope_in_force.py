"""README's teleop envelope rows describe the bound the module enforces.

The mesh input envelope is documented in README's configuration table, which
that file calls the single source of truth for operators. Both of its rows
carried the retired radian defaults - ``12.566`` "(4pi)" and ``25.133``
"(8pi)" - after the bounds themselves were converted to frame units, and the
value row still named the unit as ``(radians)``. An operator sizing a fleet
from the table would therefore compute against a bound two orders of
magnitude away from the one in force, and pick the wrong direction to tune:
under a frame-unit default a smaller unit is *narrowed* to, not widened from.

Nothing graded the pair, so the drift was invisible to every check on the
pull request that introduced it (#2598). These tests read the numbers out of
the table and compare them to the constants themselves, so a later retune
cannot leave the documentation behind.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from strands_robots.mesh import security

README = pathlib.Path(__file__).resolve().parents[2] / "README.md"

#: Env var -> the module constant its documented default must agree with.
DOCUMENTED_DEFAULTS = {
    "STRANDS_MESH_INPUT_VALUE_ABS": security.DEFAULT_INPUT_VALUE_ABS,
    "STRANDS_MESH_INPUT_SLEW_ABS": security.DEFAULT_INPUT_SLEW_ABS,
}


def _row(var: str) -> str:
    """Return the single README table row documenting ``var``."""
    rows = [
        line
        for line in README.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("|") and f"`{var}`" in line
    ]
    assert len(rows) == 1, f"expected exactly one README table row for {var}, found {len(rows)}"
    return rows[0]


def _documented_default(var: str) -> float:
    """Return the numeric default the README row states for ``var``."""
    cells = [cell.strip() for cell in _row(var).strip().strip("|").split("|")]
    default_cell = cells[-1]
    quoted = re.findall(r"`([^`]+)`", default_cell)
    assert quoted, f"README states no value for {var}'s default: {default_cell!r}"
    return float(quoted[0].replace(",", "").replace("_", ""))


@pytest.mark.parametrize(("var", "constant"), sorted(DOCUMENTED_DEFAULTS.items()))
def test_readme_documents_the_default_in_force(var: str, constant: float) -> None:
    documented = _documented_default(var)
    # The table rounds deliberately - it wrote `25.133` for 8pi - so compare at
    # the precision the table offers rather than demanding the full repr.
    assert documented == pytest.approx(constant, rel=1e-3), (
        f"README documents {var} as {documented} but the module enforces {constant}"
    )


@pytest.mark.parametrize("var", sorted(DOCUMENTED_DEFAULTS))
def test_readme_names_the_unit_the_frames_carry(var: str) -> None:
    """The row states the unit, because the number alone does not imply it.

    ``720`` is a plausible bound in degrees and an implausible one in radians,
    so a row that gives the magnitude without the unit leaves the operator to
    guess which of the two the validator compares against.
    """
    row = _row(var).lower()
    assert "frame unit" in row, f"README's {var} row does not name the unit the frames carry: {row!r}"
