"""The last-push rule names the "Update branch" button, not just ``git push``.

``require_last_push_approval`` keys on who the head commit is attributed to, and
every spelling this repository's guidance named was a ``git push``. The **"Update
branch" button** on the pull request page is the same act -- one click, no local
checkout, no token the operator handles -- and it makes the clicker the last
pusher exactly as a push does.

That silence has a measured cost. #2907's head ``1c66c8f3`` is a base refresh the
button produced: its git *author* is the maintainer, its **committer is
``web-flow``**, all twelve workflow runs on it report
``triggering_actor: cagataycali``, and the branch author is ``logesh4v``. So the
sole approval stopped counting. It happened twice, four days apart, the second
time after the mechanism had been written up in a comment on that same pull
request -- because the remedy the checks print at the moment the finding fires
described "pushing a fix" with "the token it is pushed with", which is neither
what the operator did nor how they did it.

Two tiers, because the rule lives in two kinds of place:

* The text the checks **emit** on a finding, driven through the real renderers.
  This is what an operator reads when it has already happened.
* The population of that guidance, derived from the tree rather than listed, so a
  fourth tool that copies it joins the requirement without an edit here. The
  population is keyed on *stating the consequence*, not on the constant's name:
  ``check_checkout_is_pr_head.py`` also defines ``WHAT_CLEARS_THIS`` and its
  remedy is about fetching the right commit, so it is exempt and is asserted to
  be -- a rule keyed on the name would have demanded the button of a check that
  never mentions approvals.

See scripts/check_last_push_approval.py, scripts/check_pr_head_is_current.py,
issue #3190, and the "PR Workflow" section of AGENTS.md.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"

# The one-click spelling. Required verbatim: an operator searching the guidance
# for what they just pressed is searching for the button's own label.
BUTTON = "Update branch"

# The phrase by which a remedy declares it is talking about this rule. Derived
# population, not a list of filenames.
STATES_THE_CONSEQUENCE = "consumes the approval"


def _load(stem: str) -> Any:
    """Import a ``scripts/`` check by path; they are standalone stdlib modules."""
    spec = importlib.util.spec_from_file_location(stem, _SCRIPTS / f"{stem}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _flat(text: str) -> str:
    """Collapse whitespace, so a phrase wrapped across lines still reads as one.

    Load-bearing rather than tidiness: every one of these texts is a tuple of
    hand-wrapped lines, and ``Update branch`` split over a line break is the
    same silence this file exists to close.
    """
    return " ".join(text.split())


def _remedy_modules() -> dict[str, Any]:
    """Every ``scripts/`` check whose ``WHAT_CLEARS_THIS`` states this rule."""
    found: dict[str, Any] = {}
    for path in sorted(_SCRIPTS.glob("check_*.py")):
        module = _load(path.stem)
        remedy = getattr(module, "WHAT_CLEARS_THIS", None)
        if remedy and STATES_THE_CONSEQUENCE in _flat("\n".join(remedy)):
            found[path.stem] = module
    return found


class TestTheEmittedRemedyNamesTheButton:
    """What a check prints when the finding fires, through the real renderer."""

    def test_the_last_push_finding_report_names_it(self) -> None:
        mod = _load("check_last_push_approval")
        verdict = mod.classify("cagataycali", [mod.Review("cagataycali", "APPROVED")])
        assert verdict.outcome == mod.PUSHER_ONLY_APPROVAL, "fixture must pose the finding"
        report = mod.render(verdict, "strands-labs/robots", 2907, "1c66c8f3")
        assert BUTTON in _flat(report), (
            "the remedy printed on a pusher-only-approval finding does not name the "
            "one-click spelling that produced #2907's head"
        )

    def test_the_last_push_sweep_names_it_too(self) -> None:
        """Both reports, since the sweep is what a scheduled scan reads."""
        mod = _load("check_last_push_approval")
        verdict = mod.classify("cagataycali", [mod.Review("cagataycali", "APPROVED")])
        swept = mod.render_sweep([mod.SweepRow(2907, "1c66c8f3", verdict)], [], "strands-labs/robots")
        assert BUTTON in _flat(swept)

    def test_the_stale_head_finding_names_it(self) -> None:
        """The check that prescribes a refresh is the one that must name it.

        Its remedy already quotes ``Head branch is out of date`` -- the button's
        own label -- while telling the reader not to *push*. Quoting the label
        and naming only the other spelling is the gap at its sharpest.
        """
        mod = _load("check_pr_head_is_current")
        verdict = mod.classify("aaaaaaa", "bbbbbbb")
        assert verdict.outcome == mod.STALE_HEAD_RECORD, "fixture must pose the finding"
        row = mod.Row(2907, verdict, "logesh4v/robots", "feat/x", "MERGEABLE", "BLOCKED", "REVIEW_REQUIRED")
        assert BUTTON in _flat(mod.render_one("strands-labs/robots", row))
        assert BUTTON in _flat(mod.render_sweep("strands-labs/robots", [row], []))


class TestEveryPlaceThatStatesTheRuleNamesTheButton:
    """Derived, so a fourth copy of the guidance joins without an edit here."""

    def test_the_population_is_not_empty(self) -> None:
        """A derived population that resolved to nothing would pass vacuously."""
        found = _remedy_modules()
        assert len(found) >= 2, f"expected the guidance in at least two checks, found {sorted(found)}"

    @pytest.mark.parametrize("stem", sorted(_remedy_modules()))
    def test_each_states_both_spellings(self, stem: str) -> None:
        remedy = _flat("\n".join(_remedy_modules()[stem].WHAT_CLEARS_THIS))
        assert BUTTON in remedy, f"{stem} states the rule but names only a git push"
        assert "git push" in remedy or "push" in remedy, f"{stem} should still name the push spelling"

    def test_a_remedy_that_does_not_state_the_rule_is_exempt(self) -> None:
        """The rule is keyed on the consequence, not on the constant's name.

        ``check_checkout_is_pr_head.py`` defines ``WHAT_CLEARS_THIS`` too and its
        remedy is about fetching the branch by name; requiring the button of it
        would be requiring an approval rule of a check that has no opinion on
        approvals. This holds on either side of the fix, which is the point.
        """
        module = _load("check_checkout_is_pr_head")
        remedy = _flat("\n".join(module.WHAT_CLEARS_THIS))
        assert STATES_THE_CONSEQUENCE not in remedy
        assert "check_checkout_is_pr_head" not in _remedy_modules()


class TestTheWrittenRuleCoversTheButtonAndItsMetadata:
    """AGENTS.md is where the rule is reasoned about rather than reported."""

    @staticmethod
    def _agents() -> str:
        return _flat((_ROOT / "AGENTS.md").read_text(encoding="utf-8"))

    def test_the_rule_names_the_button(self) -> None:
        assert BUTTON in self._agents(), (
            'AGENTS.md documents require_last_push_approval without naming the "Update '
            'branch" button, so the rule as written does not reach the spelling that '
            "spent #2907's approval twice"
        )

    def test_the_commit_metadata_guidance_covers_the_web_flow_shape(self) -> None:
        """The table exists to stop this inference; it stopped two of three shapes.

        A committer of ``web-flow`` is the most reassuring of the three and the
        least likely to prompt a check, because it reads as GitHub having merged
        rather than a person having pushed.
        """
        agents = self._agents()
        assert "web-flow" in agents, "the commit-metadata guidance does not name the button's own shape"
        assert "triggering_actor" in agents
