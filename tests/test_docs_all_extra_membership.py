"""The docs describe what ``[all]`` installs, derived from ``pyproject.toml``.

``[all]`` is a convenience bundle, not a union: it names most extras and leaves
the GPU-only backends, the separate-install toolchains and several service
clients opt-in. Three pages tell a reader what it covers, and the membership
they describe is a fact about ``[project.optional-dependencies]`` rather than
prose - so it is derivable, and it drifted.

``docs/getting-started/installation.md`` enumerated ``[all]`` as five extras
(``groot-service`` + ``lerobot`` + ``sim-mujoco`` + ``mesh`` + ``mesh-iot``)
while the bundle had grown to nineteen. The enumeration was a strict subset, so
a reader deciding whether ``[all]`` covered the policy they wanted was told it
did not for fourteen extras it does install - and the code block five lines
below the table called the same bundle "everything". ``docs/architecture.md``
called it a "union", which it is not.

The cells now state the count and name the extras ``[all]`` leaves out, because
that is the actionable half: a reader wants to know what they must still add.
Every rule here derives its expectation from ``pyproject.toml``, and each
failure message prints the text the page should carry, so a new extra makes
these fail with the correction in hand rather than leaving the pages to rot
again.

Deliberately out of scope: *why* a given extra is left out of ``[all]``. The
excluded set has no single rule - ``[sim-isaac]`` and ``[sim-gs]`` need a GPU
and say so in the README, while ``[cosmos3-service]`` is two pure-Python
packages and ``[microduck]`` is one - so documenting the reason per extra is a
maintainer's call, not a derivation. These rules grade *which* extras are
excluded, never why.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _ROOT / "pyproject.toml"
_INSTALL_PAGE = _ROOT / "docs" / "getting-started" / "installation.md"
_ARCHITECTURE_PAGE = _ROOT / "docs" / "architecture.md"
_INDEX_PAGE = _ROOT / "docs" / "index.md"

# The bundle is a developer convenience, so its tooling extra is not a
# capability a reader installs it for; the pages describe capability extras.
_TOOLING_EXTRAS = frozenset({"dev"})

_ALL_ROW = "| `[all]` |"

# Words that describe the bundle as complete. ``[all]`` is not, so a page using
# one of these tells a reader they need no further extra.
_COMPLETENESS_CLAIMS = ("union", "everything", "every policy", "every extra")
_NEGATED_CLAIM = re.compile(
    r"not(?:\*\*)?\s+(?:a\s+)?(?:" + "|".join(re.escape(c) for c in _COMPLETENESS_CLAIMS) + r")"
)
_EXTRA_IN_TEXT = re.compile(r"`\[([a-z0-9][a-z0-9-]*)\]`")


def _extras() -> dict[str, list[str]]:
    """Every entry of ``[project.optional-dependencies]``."""
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def _closure(extras: dict[str, list[str]], name: str) -> set[str]:
    """Extras reachable from ``name`` through ``strands-robots[...]`` specs.

    An extra can name siblings (``[mesh-iot]`` pulls ``[mesh]``), so the set a
    reader gets is the transitive closure rather than the literal list.
    """
    reached: set[str] = set()
    pending = [name]
    while pending:
        for spec in extras.get(pending.pop(), []):
            found = re.search(r"strands-robots\[([^]]+)\]", str(spec))
            if found is None:
                continue
            for part in found.group(1).split(","):
                extra = part.strip()
                if extra and extra not in reached:
                    reached.add(extra)
                    pending.append(extra)
    return reached


def _membership() -> tuple[set[str], set[str], int]:
    """``(installed, left_opt_in, declared_total)`` for ``[all]``."""
    extras = _extras()
    installed = _closure(extras, "all")
    declared = set(extras) - {"all"}
    left_out = declared - installed - _TOOLING_EXTRAS
    return installed, left_out, len(declared)


def _row(page: Path) -> str:
    """The ``[all]`` row of ``page``'s extras table."""
    text = page.read_text(encoding="utf-8")
    assert _ALL_ROW in text, f"{page.name} no longer carries an '{_ALL_ROW}' table row"
    for line in text.splitlines():
        if line.startswith(_ALL_ROW):
            return line
    raise AssertionError(f"{page.name}: '{_ALL_ROW}' found in the page but not at the start of a line")


def _extras_named(text: str) -> set[str]:
    """Extra names written as ``` `[name]` ``` in ``text``."""
    return set(_EXTRA_IN_TEXT.findall(text))


class TestThePagesAgreeWithPyproject:
    """The count and the excluded set both come from ``pyproject.toml``."""

    @pytest.mark.parametrize("page", [_INSTALL_PAGE, _ARCHITECTURE_PAGE], ids=["installation", "architecture"])
    def test_the_row_states_the_derived_counts(self, page: Path) -> None:
        installed, _, declared_total = _membership()
        row = _row(page)
        wanted = f"{len(installed)} of the {declared_total} extras"
        assert wanted in row, (
            f"{page.name}: the `[all]` row must state {wanted!r}, so a reader is not told the bundle is "
            f"narrower (or wider) than it is. The row reads:\n  {row}"
        )

    def test_the_install_row_names_every_extra_left_opt_in(self) -> None:
        _, left_out, _ = _membership()
        row = _row(_INSTALL_PAGE)
        named = _extras_named(row)
        missing = sorted(left_out - named)
        assert not missing, (
            f"installation.md: the `[all]` row must name every extra the bundle leaves opt-in, because that "
            f"is what a reader still has to install. Unnamed: {missing}. Name them as `[extra]`."
        )

    def test_the_install_row_claims_nothing_it_installs_is_opt_in(self) -> None:
        installed, _, _ = _membership()
        row = _row(_INSTALL_PAGE)
        wrongly_named = sorted(_extras_named(row) & installed)
        assert not wrongly_named, (
            f"installation.md: the `[all]` row lists {wrongly_named} as opt-in, but `all` installs them. "
            f"A reader would add an extra they already have."
        )

    def test_no_page_calls_the_bundle_a_union_or_everything(self) -> None:
        _, left_out, _ = _membership()
        assert left_out, "nothing is left opt-in, so 'union' would be accurate and this rule is vacuous"
        for page in (_INSTALL_PAGE, _ARCHITECTURE_PAGE, _INDEX_PAGE):
            text = page.read_text(encoding="utf-8")
            for line in text.splitlines():
                if "strands-robots[all]" not in line and not line.startswith(_ALL_ROW):
                    continue
                # A corrected page says "not a union", so a bare substring test would
                # report the very wording that fixes this. Drop negated forms first and
                # grade what is left, which is the affirmative claim.
                lowered = _NEGATED_CLAIM.sub("", line.lower())
                for claim in _COMPLETENESS_CLAIMS:
                    assert claim not in lowered, (
                        f"{page.name}: {claim!r} describes `[all]` as complete, and {len(left_out)} extras "
                        f"stay opt-in ({sorted(left_out)}). The line reads:\n  {line}"
                    )


class TestTheDerivationIsNotVacuous:
    """The membership split is real, so the rules above have something to grade."""

    def test_the_bundle_installs_most_but_not_all_extras(self) -> None:
        installed, left_out, declared_total = _membership()
        assert len(installed) > 1, f"only {len(installed)} extras reached from `all`; the closure walk is blind"
        assert left_out, "no extra is left opt-in, so `all` really is a union and these rules grade nothing"
        assert len(installed) + len(left_out) + len(_TOOLING_EXTRAS) == declared_total, (
            f"the split does not account for every extra: {len(installed)} installed + {len(left_out)} opt-in "
            f"+ {len(_TOOLING_EXTRAS)} tooling != {declared_total} declared"
        )

    def test_the_closure_follows_an_extra_named_two_levels_down(self) -> None:
        extras = _extras()
        direct = {
            part.strip()
            for spec in extras["all"]
            for match in [re.search(r"strands-robots\[([^]]+)\]", str(spec))]
            if match
            for part in match.group(1).split(",")
        }
        indirect = _closure(extras, "all") - direct
        assert indirect, (
            "every extra `all` installs is named directly by it, so a walk that did not recurse would "
            "still produce the right count and the counts these rules assert would not grade the walk"
        )
        assert indirect <= _closure(extras, "all"), "the closure must contain what it reached indirectly"


class TestTheRulesReportAConstructedDrift:
    """The pages are correct today, so the rules are graded on built exemplars."""

    def test_a_stale_count_is_reported(self) -> None:
        installed, _, declared_total = _membership()
        stale = f"| `[all]` | {len(installed) - 1} of the {declared_total} extras - not a union | x |"
        assert f"{len(installed)} of the {declared_total} extras" not in stale

    def test_an_unnamed_opt_in_extra_is_reported(self) -> None:
        _, left_out, _ = _membership()
        row = "| `[all]` | 19 of the 31 extras - not a union | x |"
        assert sorted(left_out - _extras_named(row)) == sorted(left_out)

    def test_a_union_claim_is_reported(self) -> None:
        stale = "| `[all]` | union | CI / exploration |".lower()
        assert any(c in _NEGATED_CLAIM.sub("", stale) for c in _COMPLETENESS_CLAIMS)

    def test_a_negated_claim_is_not_reported(self) -> None:
        corrected = "| `[all]` | 19 of the 31 extras - **not** a union | x |".lower()
        assert not any(c in _NEGATED_CLAIM.sub("", corrected) for c in _COMPLETENESS_CLAIMS), (
            "the rule must accept a page that says the bundle is NOT complete, or the wording that "
            "fixes this defect would itself be reported"
        )
