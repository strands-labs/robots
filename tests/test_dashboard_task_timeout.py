"""Task-command waits, sized per action (see task_timeout.py).

The defect: the route sends ``action: "start"``, which mesh core answers with an
immediate ack, but the wait was ``max(timeout, duration + 10)`` - sized for the
blocking ``execute`` variant. A 1-hour run therefore meant the dashboard would sit
on the ack for 3610s, leaving the Run button spinning in "starting" with nothing on screen if
the peer never answered: wedged serial, dead child, lost response.
"""

import math

from strands_robots.dashboard.task_timeout import (
    DEFAULT_ACK_CAP_S,
    task_ack_budget,
    timeout_verdict,
)


def test_start_ack_wait_does_not_scale_with_a_long_run() -> None:
    # THE BUG: an hour-long run used to mean an hour-long wait for the ack.
    timeout_s, kind = task_ack_budget("start", None, 3600.0)
    assert kind == "ack"
    assert timeout_s == DEFAULT_ACK_CAP_S
    assert timeout_s < 3610.0


def test_execute_still_waits_for_the_whole_rollout() -> None:
    # "execute" genuinely blocks until the rollout ends; cutting it short would
    # report a timeout on a task running perfectly.
    timeout_s, kind = task_ack_budget("execute", None, 3600.0)
    assert kind == "rollout"
    assert timeout_s == 3610.0


def test_short_runs_keep_their_full_duration() -> None:
    # For a 20s run both budgets agree, so the longer one costs nothing.
    assert task_ack_budget("start", None, 20.0) == (30.0, "ack")


def test_an_explicit_caller_timeout_is_always_honoured() -> None:
    # It is the caller's business how long they wait - this only sets a floor.
    assert task_ack_budget("start", 600.0, 30.0)[0] == 600.0
    assert task_ack_budget("execute", 7200.0, 30.0)[0] == 7200.0
    assert task_ack_budget("start", 5.0, 30.0)[0] == 40.0, "a tiny timeout does not shrink the floor"


def test_unknown_action_is_treated_as_an_ack_wait() -> None:
    # A future verb must not inherit the unbounded wait by accident.
    for action in ("", "START", "queue", "nonsense", None):
        timeout_s, kind = task_ack_budget(action, None, 3600.0)  # type: ignore[arg-type]
        assert kind == "ack"
        assert timeout_s == DEFAULT_ACK_CAP_S


def test_junk_numbers_cannot_produce_a_nonsense_wait() -> None:
    for bad in (None, "", "abc", float("nan"), -5.0, [1]):
        timeout_s, _ = task_ack_budget("start", bad, bad)  # type: ignore[arg-type]
        assert timeout_s >= 0 and math.isfinite(timeout_s)
    assert task_ack_budget("start", None, 0.0)[0] == 10.0


def test_a_nan_in_either_position_is_rejected_on_its_own() -> None:
    # The existing junk-number cell passes the same value in both positions, so a
    # guard that dropped one of the two would still read green there. These pin
    # each position while the other carries a good number.
    nan = float("nan")

    timeout_s, kind = task_ack_budget("start", nan, 30.0)
    assert math.isfinite(timeout_s) and timeout_s == 40.0, "a NaN ask falls back to the duration floor"
    assert kind == "ack"

    # A NaN duration has to be caught here rather than left to ``max``: unguarded
    # it reaches ``min(max(nan + 10, 0.0), cap)`` as NaN, every comparison against
    # it is False, and the budget comes back as 0.0 -- an instant timeout on a
    # robot that was just told to move, which is the reading this module exists to
    # refuse. Guarded, the duration reads as 0.0 and the ack floor applies.
    timeout_s, kind = task_ack_budget("start", None, nan)
    assert math.isfinite(timeout_s) and timeout_s == 10.0, "a NaN duration must not zero the wait"
    assert kind == "ack"


def test_ack_cap_is_injectable() -> None:
    assert task_ack_budget("start", None, 3600.0, ack_cap_s=45.0)[0] == 45.0


def test_a_timeout_says_the_robot_may_still_move() -> None:
    # THE SAFETY POINT: the command was delivered. Rendering this as "nothing
    # happened" is how an operator walks up to an arm that is about to start.
    v = timeout_verdict("ack", 120.0, "so101-arm-2")
    assert v["motion_possible"] is True
    assert v["timeout_kind"] == "ack"
    assert "no acknowledgement from so101-arm-2 within 120s" in v["error"]  # type: ignore[operator]
    assert "may be loading a policy and about to move" in v["error"]  # type: ignore[operator]
    assert "Check its log" in v["error"]  # type: ignore[operator]


def test_a_rollout_timeout_says_the_task_may_be_fine() -> None:
    v = timeout_verdict("rollout", 3610.0)
    assert v["motion_possible"] is True
    assert v["timeout_kind"] == "rollout"
    assert "the rollout was still running when the wait ended" in v["error"]  # type: ignore[operator]
    assert "may be executing normally" in v["error"]  # type: ignore[operator]
    assert " from " not in v["error"], "no target given - do not invent one"  # type: ignore[operator]
