"""The documented ABC contract names every public member of ``Policy``.

``docs/policies/custom-policies.md`` carries an "ABC contract" table which is
the surface a subclass author reads before writing a provider. A member absent
from it is not merely undocumented - it is invisible on the page whose job is
to enumerate the contract, so an author cannot know to override it. That had
already happened to eight of the fifteen public members, ``preflight`` among
them, which is the member deciding whether the simulation renders the whole
scene before a rollout.

The expectation is DERIVED from the class rather than pinned as a literal list,
so a member added to ``Policy`` joins the requirement with no edit here. It is a
biconditional: every public member appears in the table, and every name in the
table is a public member (a row for something the class no longer has is
equally misleading).
"""

from __future__ import annotations

import inspect
import pathlib
import re

from strands_robots.policies.base import Policy

_DOC = pathlib.Path(inspect.getfile(Policy)).parents[2] / "docs" / "policies" / "custom-policies.md"
_HEADING = "## ABC contract"


def _public_members() -> set[str]:
    """Public members declared on ``Policy`` itself (not inherited from object)."""
    return {name for name in vars(Policy) if not name.startswith("_")}


def _documented_members() -> set[str]:
    """First identifier of each row in the ABC-contract table."""
    text = _DOC.read_text(encoding="utf-8")
    start = text.index(_HEADING)
    end = text.index("\n## ", start + len(_HEADING))
    names = set()
    for cell in re.findall(r"^\|\s*`([^`]+)`", text[start:end], re.M):
        match = re.search(r"\b([a-z_][a-z0-9_]*)\b", cell.replace("async ", ""))
        if match:
            names.add(match.group(1))
    return names


class TestTheTableAndTheClassAgree:
    """One table, one class, and no member visible in only one of them."""

    def test_the_scan_finds_both_populations(self):
        """Non-vacuity control: an empty side would make either assertion below
        pass for the wrong reason. Floors only - the counts themselves are not
        the claim, so this holds whatever the table currently lists."""
        assert len(_public_members()) >= 15, "the class scan found no members"
        assert _documented_members(), "the table scan found no rows"

    def test_every_public_member_is_documented(self):
        """A member the page omits cannot be overridden by an author who only
        reads the page."""
        missing = sorted(_public_members() - _documented_members())
        assert not missing, f"public Policy members absent from the ABC-contract table: {missing}"

    def test_every_documented_row_is_a_public_member(self):
        """The other direction: a row for a member the class does not have sends
        an author to write something nothing calls."""
        stale = sorted(_documented_members() - _public_members())
        assert not stale, f"ABC-contract table rows naming no public Policy member: {stale}"

    def test_the_abstract_column_matches_the_class(self):
        """``yes`` in the table must mean ``abstractmethod`` on the class, so the
        three members an implementation MUST supply stay the three that raise if
        it does not."""
        text = _DOC.read_text(encoding="utf-8")
        start = text.index(_HEADING)
        end = text.index("\n## ", start + len(_HEADING))
        documented_abstract = set()
        # The name cell may carry a kind suffix after the backticks, e.g.
        # "`provider_name` (property)", so consume it before the column break.
        for row in re.findall(r"^\|\s*`([^`]+)`[^|]*\|([^|]*)\|", text[start:end], re.M):
            cell, abstract = row
            match = re.search(r"\b([a-z_][a-z0-9_]*)\b", cell.replace("async ", ""))
            if match and abstract.strip() == "yes":
                documented_abstract.add(match.group(1))
        actual_abstract = set()
        for name, obj in vars(Policy).items():
            if name.startswith("_"):
                continue
            target = getattr(obj, "fget", None) or getattr(obj, "__func__", None) or obj
            if getattr(target, "__isabstractmethod__", False):
                actual_abstract.add(name)
        assert documented_abstract == actual_abstract, (
            f"table says abstract={sorted(documented_abstract)}, class says {sorted(actual_abstract)}"
        )
