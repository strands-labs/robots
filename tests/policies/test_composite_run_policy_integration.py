"""End-to-end integration: ``CompositePolicy`` driven through ``run_policy``.

The unit tests in ``test_composite.py`` verify the merge / ownership / RTC
logic of :class:`~strands_robots.policies.composite.CompositePolicy` in
isolation by calling ``get_actions`` directly. They never drive the composite
through the real control loop, so the contract the *whole point* of the policy
relies on -- the merged per-tick action dict actually reaching ``send_action``
as a command for BOTH disjoint joint groups on a real robot -- was unguarded.

These tests close that gap against a real MuJoCo robot:

* Compose two policies over disjoint actuator groups, run a short rollout, and
  assert BOTH groups received a non-zero actuator command. The assertion is on
  the commanded control signal (``mj_data.ctrl``), not the resulting ``qpos``:
  a position servo drives an *un*-commanded joint back toward zero, so joint
  motion alone would not distinguish "commanded" from "released". A regression
  that drops one group from the merge leaves that group's ``ctrl`` at zero.
* Pin the ``run_policy`` <-> ``CompositePolicy`` key contract: ``run_policy``
  names a policy by ``robot_action_keys`` (actuators, not joints -- a robot's
  actuators are not always its joints), and ``CompositePolicy`` must forward
  those keys to *both* children so a group-by-name filter matches what the
  children actually emit.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

import mujoco

from strands_robots import create_policy
from strands_robots.policies.composite import CompositePolicy
from strands_robots.simulation.mujoco.simulation import Simulation


@pytest.fixture
def sim():
    s = Simulation(tool_name="composite_integration", mesh=False)
    s.create_world()
    s.add_robot(name="so100", data_config="so100")
    yield s
    s.cleanup()


def _actuator_ctrl_by_key(sim: Simulation) -> dict[str, float]:
    """Map each action key to its actuator's commanded control value.

    ``robot_action_keys`` are the bare motor names; the compiled model names
    actuators with the robot namespace (``so100/Rotation``), so match on the
    trailing segment.
    """
    model, data = sim.mj_model, sim.mj_data
    by_trailing = {}
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            by_trailing[name.split("/")[-1]] = i
    return {k: float(data.ctrl[by_trailing[k]]) for k in sim.robot_action_keys("so100") if k in by_trailing}


def test_composite_run_policy_commands_both_disjoint_groups(sim):
    """A CompositePolicy split across two disjoint actuator groups must issue a
    command to BOTH groups when driven through ``run_policy`` -- the merged
    partial action dict has to flow to ``send_action`` for each child's group.
    """
    action_keys = list(sim.robot_action_keys("so100"))
    assert len(action_keys) >= 4, action_keys
    mid = len(action_keys) // 2
    lower_group, upper_group = action_keys[:mid], action_keys[mid:]

    comp = CompositePolicy(
        lower=create_policy("mock"),
        upper=create_policy("mock"),
        lower_joints=lower_group,
        upper_joints=upper_group,
    )

    run = sim.run_policy(
        robot_name="so100",
        policy_object=comp,
        n_steps=40,
        control_frequency=50,
        action_horizon=8,
        fast_mode=True,
        seed=3,
    )
    assert run["status"] == "success", run

    ctrl = _actuator_ctrl_by_key(sim)
    lower_cmd = sum(abs(ctrl[k]) for k in lower_group)
    upper_cmd = sum(abs(ctrl[k]) for k in upper_group)
    # Both children's commands reached send_action through the merge. A merge
    # that dropped one group leaves that group's actuators at ctrl == 0.
    assert lower_cmd > 1e-3, f"lower group was never commanded (merge dropped it): {lower_cmd}"
    assert upper_cmd > 1e-3, f"upper group was never commanded (merge dropped it): {upper_cmd}"


def test_run_policy_names_composite_children_by_actuator_keys(sim):
    """``run_policy`` names the policy by ``robot_action_keys`` (actuators),
    and ``CompositePolicy`` forwards those keys to both children so their
    emitted action dicts key on the same names the group filter uses. A child
    left unnamed emits placeholder keys that never resolve -> a failed rollout.
    """
    lower = create_policy("mock")
    upper = create_policy("mock")
    comp = CompositePolicy(lower=lower, upper=upper)

    run = sim.run_policy(
        robot_name="so100",
        policy_object=comp,
        n_steps=4,
        control_frequency=50,
        fast_mode=True,
        seed=1,
    )
    assert run["status"] == "success", run

    action_keys = list(sim.robot_action_keys("so100"))
    assert lower.robot_state_keys == action_keys
    assert upper.robot_state_keys == action_keys
