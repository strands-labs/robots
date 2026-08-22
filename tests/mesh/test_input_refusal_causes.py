"""A refused teleop frame is counted under the guard that refused it.

:meth:`~strands_robots.mesh.input.InputReceiver._on_input` refuses a frame for
six distinct reasons, and four of them shared one ``rejected`` counter while the
other two (the apply-rate ceiling and the per-joint slew bound) each kept their
own. Two consequences followed from the sharing, and only the second is visible
from the counters:

* a report reading ``rejected`` cannot say which guard refused the stream, so
  the reason has to be recovered from the follower's log, and
* the log itself is rate-limited per counter, so a stream refused for one reason
  spends the shared budget and the FIRST refusal of a different reason -- the
  only place a refusal names the value it refused and the bound it exceeded --
  is never written.

The second is what makes the first unrecoverable rather than merely awkward.
These tests pin the per-cause accounting, that the budget is spent per cause,
and that the two causes which already had their own counter keep it.
"""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap
import threading
import time

import pytest

from strands_robots.mesh import input as mesh_input
from strands_robots.mesh.input import InputReceiver
from tests.mesh.test_input_stream_lifecycle import _make_receiver, _RecvMesh

_LOGGER = "strands_robots.mesh.input"

#: The causes this path refuses for, and the per-cause log budget, stated here as
#: well as in the module so a rename cannot quietly narrow what this file grades.
_CAUSES = ("lockout", "freshness", "invalid")
_BUDGET = 5


def _locked_out_receiver():
    mesh = _RecvMesh()
    mesh._estop_lockout = threading.Event()
    mesh._estop_lockout.set()
    return _make_receiver(mesh)


def _fresh(action, seq=0):
    return {"action": action, "seq": seq, "t": time.time()}


def _stale(seq=0):
    return {"action": {"j0": 0.1}, "seq": seq, "t": time.time() - 9999.0}


def _over_the_value_bound(seq=0):
    # |1500| is past DEFAULT_INPUT_VALUE_ABS (720 frame units), so
    # validate_input_frame refuses it and names both numbers.
    return _fresh({"j0": 1500.0}, seq)


#: One frame per cause, and how that cause is reported.
_CASES = (
    pytest.param("lockout", _locked_out_receiver, lambda: _fresh({"j0": 0.1}), id="lockout"),
    pytest.param("freshness", _make_receiver, _stale, id="freshness"),
    pytest.param("invalid", _make_receiver, _over_the_value_bound, id="invalid"),
)


class TestARefusalNamesTheGuardThatRefusedIt:
    @pytest.mark.parametrize(("cause", "make", "frame"), _CASES)
    def test_only_that_cause_and_the_total_advance(self, cause, make, frame, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "0")  # isolate from the rate gate
        recv, applied = make()
        recv._on_input(recv.topic, frame())
        stats = recv.stats
        assert applied == [], "a refused frame must never reach the robot"
        assert stats["rejected"] == 1
        key = f"rejected_{cause}"
        assert key in stats, (
            f"a frame refused by the {cause} guard is counted only in the shared total, so "
            f"nothing reports which guard refused it: {stats}"
        )
        assert stats[key] == 1, f"{cause} did not report itself: {stats}"
        others = [f"rejected_{other}" for other in _CAUSES if other != cause]
        assert all(stats[key] == 0 for key in others), f"a {cause} refusal was counted under another cause too: {stats}"

    def test_the_total_is_the_sum_of_the_causes(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "0")
        recv, _ = _make_receiver()
        for seq in range(3):
            recv._on_input(recv.topic, _stale(seq))
        for seq in range(2):
            recv._on_input(recv.topic, _over_the_value_bound(10 + seq))
        stats = recv.stats
        missing = [f"rejected_{c}" for c in _CAUSES if f"rejected_{c}" not in stats]
        assert missing == [], f"the total is reported with no breakdown behind it: {missing}"
        breakdown = {c: stats[f"rejected_{c}"] for c in _CAUSES}
        assert breakdown == {"lockout": 0, "freshness": 3, "invalid": 2}
        assert stats["rejected"] == sum(breakdown.values()) == 5


class TestTheLogBudgetIsSpentPerCause:
    def test_a_refusal_is_still_stated_after_another_cause_spent_the_budget(self, monkeypatch, caplog):
        """The regression: an unrelated cause must not silence the next one.

        A refusal states the offending value and the bound it exceeded in its log
        line and nowhere else, so a consumer diagnosing "the leader streams
        degrees into a radian envelope" reads that line. Sharing one budget with
        the replay-freshness guard meant a follower that had already refused a
        few stale frames -- an ordinary clock-skew start -- refused the
        out-of-range frame in complete silence.
        """
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "0")
        caplog.set_level(logging.WARNING, logger=_LOGGER)
        recv, applied = _make_receiver()
        for seq in range(_BUDGET):
            recv._on_input(recv.topic, _stale(seq))
        assert recv.stats["rejected"] == _BUDGET, "premise: the budget is spent"
        caplog.clear()

        recv._on_input(recv.topic, _over_the_value_bound(99))

        assert applied == []
        stated = [r.getMessage() for r in caplog.records if "out of range" in r.getMessage()]
        assert stated, (
            "the value refusal was counted and never stated: rejected="
            f"{recv.stats['rejected']} with no line naming the value or the bound, so the "
            "budget spent by the replay-freshness guard silenced a different guard's "
            "only report"
        )
        assert "1500" in stated[0] and "720" in stated[0], (
            f"the refusal must name the value and the bound: {stated[0]!r}"
        )
        # and the counters now say which guard refused it, without reading the log
        assert recv.stats["rejected_invalid"] == 1
        assert recv.stats["rejected_freshness"] == _BUDGET

    def test_one_cause_still_goes_quiet_once_its_own_budget_is_spent(self, monkeypatch, caplog):
        """Control: the budget still exists, it is just no longer shared.

        Passes on both trees - one cause refusing ``_BUDGET + 4`` times still goes
        quiet after ``_BUDGET`` lines. It fails for the tempting over-fix of
        logging every refusal, which would flood a 50 Hz loop.
        """
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "0")
        caplog.set_level(logging.WARNING, logger=_LOGGER)
        recv, _ = _make_receiver()
        over = _BUDGET + 4
        for seq in range(over):
            recv._on_input(recv.topic, _stale(seq))
        assert recv.stats["rejected"] == over
        assert len(caplog.records) == _BUDGET


class TestTheAccountingHasOneOwner:
    """Structural: the fix cannot be undone one refusal site at a time."""

    def test_every_rejected_increment_goes_through_the_helper(self):
        src = textwrap.dedent(inspect.getsource(InputReceiver))
        cls = ast.parse(src).body[0]
        outside: list[str] = []
        for fn in cls.body:
            if not isinstance(fn, ast.FunctionDef) or fn.name in {"__init__", "_refuse"}:
                continue
            for node in ast.walk(fn):
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target]
                    if isinstance(node, ast.AugAssign)
                    else []
                )
                for target in targets:
                    if ast.unparse(target) == "self._rejected":
                        outside.append(f"{fn.name}:{node.lineno}")
        assert outside == [], (
            "a refusal counted the total directly instead of through _refuse, so it "
            f"reports no cause and shares another cause's log budget: {outside}"
        )

    def test_the_budget_is_measured_against_the_cause_not_the_total(self):
        refuse = textwrap.dedent(inspect.getsource(InputReceiver._refuse))
        fn = ast.parse(refuse).body[0]
        gates = [
            ast.unparse(node.test)
            for node in ast.walk(fn)
            if isinstance(node, ast.If) and any("logger.warning" in ast.unparse(child) for child in ast.walk(node))
        ]
        assert gates, "the helper no longer gates its log line"
        for gate in gates:
            assert "self._rejected" not in gate, f"the log budget is measured against the shared total again: {gate!r}"

    def test_every_refusal_site_names_a_declared_cause(self):
        src = textwrap.dedent(inspect.getsource(InputReceiver._on_input))
        fn = ast.parse(src).body[0]
        causes = [
            node.args[0].value
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "self._refuse"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ]
        assert len(causes) >= 4, f"the scan found no refusal sites to grade: {causes}"
        unknown = sorted(set(causes) - set(mesh_input._REJECTION_CAUSES))
        assert unknown == [], f"refusal sites name causes no report enumerates: {unknown}"
        assert set(causes) == set(mesh_input._REJECTION_CAUSES), (
            "a declared cause is never used, so nothing reports it: "
            f"{sorted(set(mesh_input._REJECTION_CAUSES) - set(causes))}"
        )


class TestTheGradedVocabularyMatchesTheModule:
    def test_the_causes_and_the_budget_are_the_ones_the_module_declares(self):
        """This file states both independently, so a rename cannot narrow the sweep."""
        assert tuple(mesh_input._REJECTION_CAUSES) == _CAUSES
        assert mesh_input._REFUSAL_LOG_BUDGET == _BUDGET


class TestTheReportedShapeIsDocumented:
    def test_the_docstring_names_every_reported_cause(self):
        doc = inspect.getdoc(InputReceiver.stats) or ""
        missing = [f"rejected_{c}" for c in _CAUSES if f"rejected_{c}" not in doc]
        assert missing == [], f"stats reports keys its docstring does not name: {missing}"

    def test_the_docstring_does_not_claim_a_cause_this_path_has_no_check_for(self):
        """It named an ACL check the receive path does not make, and omitted the
        frame-validation refusal that a consumer's own diagnosis is built on."""
        doc = (inspect.getdoc(InputReceiver.stats) or "").lower()
        checks = [line for line in inspect.getsource(InputReceiver._on_input).splitlines() if "acl" in line.lower()]
        assert checks == [], f"an ACL check appeared on the receive path: {checks}"
        assert "acl" not in doc, "the docstring names a cause this path never checks"


class TestTheCausesWithTheirOwnCounterKeepIt:
    """Controls: the two guards that already reported themselves are untouched."""

    def test_a_rate_capped_frame_is_not_counted_as_rejected(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_INPUT_MAX_HZ", "10")
        recv, applied = _make_receiver()
        recv._on_input(recv.topic, _fresh({"j0": 0.1}, 0))
        recv._on_input(recv.topic, _fresh({"j0": 0.2}, 1))
        stats = recv.stats
        assert len(applied) == 1
        assert stats["rate_dropped"] == 1
        assert stats["rejected"] == 0
        assert all(stats.get(f"rejected_{c}", 0) == 0 for c in _CAUSES)

    def test_the_breakdown_does_not_absorb_the_separately_counted_causes(self):
        recv, _ = _make_receiver()
        stats = recv.stats
        for key in ("rate_dropped", "slew_rejected"):
            assert key in stats
            assert f"rejected_{key}" not in stats, (
                f"{key} was folded into the rejected breakdown, changing what rejected totals"
            )

    def test_a_single_lockout_refusal_still_totals_one(self):
        """The pre-existing contract: ``rejected`` is still the total."""
        recv, applied = _locked_out_receiver()
        recv._on_input(recv.topic, _fresh({"j0": 0.1}))
        assert applied == []
        assert recv._rejected == 1
