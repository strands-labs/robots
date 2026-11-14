"""One operator hook: ``Robot(operator=)`` is asked with a ``Move`` for one command and a ``RunPlan`` for one run."""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots import Robot
from strands_robots.robot.run import Move, RunPlan, from_agent
from strands_robots.robot.safety import RequestRefused


def _real(**kw):
    robot = Robot("so101", driver="memory", twin=False, **kw)
    return robot, robot.body


@pytest.mark.parametrize("old", ["confirm", "approve"])
@pytest.mark.parametrize("hardware", [True, False])
def test_the_old_hooks_are_a_type_error_that_names_operator(old, hardware):
    where = {"driver": "memory"} if hardware else {}
    with pytest.raises(TypeError, match=rf"takes no {old}=: use operator="):
        Robot("so101", **where, **{old: lambda r: True})


def test_use_mesh_takes_operator_not_approve():
    pytest.importorskip("strands")
    from strands_robots.mesh.tool import use_mesh

    with pytest.raises(TypeError, match="takes no approve=: use operator="):
        use_mesh(object(), approve=lambda plan: True)


def test_no_operator_refuses_an_agents_move_and_an_agents_run_and_writes_nothing():
    robot, bus = _real()
    with pytest.raises(RequestRefused, match=r"pass operator=<callable\(Move \| RunPlan\) -> bool> to Robot"):
        robot.confirm_act({"shoulder_pan": 0.2})
    with from_agent(), pytest.raises(RequestRefused, match=r"this run came from an agent: pass operator="):
        robot.run(lambda obs: {"shoulder_pan": 0.2}, steps=3)
    assert bus.writes == []
    robot.close()


def test_one_callable_is_asked_with_a_move_for_a_command_and_a_runplan_for_a_run():
    asked: list[Move | RunPlan] = []
    robot, bus = _real(operator=lambda r: asked.append(r) or True)
    robot.confirm_act({"shoulder_pan": 0.2}, seconds=0.05)
    move = asked[-1]
    assert isinstance(move, Move) and move.robot == robot.id and move.source == "agent" and move.seconds == 0.05
    assert list(move.targets) == list(robot.action_keys) and move.targets["shoulder_pan"] == pytest.approx(0.2)
    assert np.allclose(move.vector, list(move.targets.values())) and not move.vector.flags.writeable
    assert str(move).startswith(f"move {robot.id} from agent over 0.05 s: shoulder_pan=0.200") and "\n" not in str(move)
    n = len(asked)
    with from_agent():
        res = robot.run(lambda obs: {"shoulder_pan": 0.1}, steps=3)
    assert res.steps == 3 and len(asked) == n + 1 and isinstance(asked[-1], RunPlan)  # once per run, not per tick
    robot.close()


def test_a_yes_for_moves_only_does_not_approve_a_run_and_a_yes_for_runs_only_does_not_move():
    robot, bus = _real(operator=lambda r: isinstance(r, Move))
    robot.confirm_act({"shoulder_pan": 0.2})
    with from_agent(), pytest.raises(RequestRefused, match="the operator refused the run"):
        robot.run(lambda obs: {"shoulder_pan": 0.3}, steps=3)
    assert len(bus.writes) == 1
    robot.close()
    robot, bus = _real(operator=lambda r: isinstance(r, RunPlan))
    with pytest.raises(RequestRefused, match="the operator gate refused move"):
        robot.confirm_act({"shoulder_pan": 0.2})
    assert bus.writes == []
    robot.close()


def test_the_estop_never_asks_the_operator_and_resume_still_needs_approved_by():
    def never(r):
        raise AssertionError(f"the operator was asked for {r}")

    robot, bus = _real(operator=never)
    robot.estop()
    with pytest.raises(RequestRefused, match="resume needs approved_by="):
        robot.resume()
    robot.resume(approved_by="ana")
    robot.close()
