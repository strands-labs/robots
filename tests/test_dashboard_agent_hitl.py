"""The motion gate pauses the agent (interrupt) instead of ending its turn.

A human yes deposits a one-shot grant the fleet tool consumes; a no cancels
the tool with a sentence the model can relay. Stopping is never gated.
"""

from __future__ import annotations

import pytest

from strands_robots.dashboard import agent_hitl
from strands_robots.dashboard.agent_hitl import (
    INTERRUPT_NAME,
    MOTION_ACTIONS,
    MotionInterruptHook,
    cancel_sentence,
    consume_grant,
    deposit_grant,
    motion_intent,
    response_approves,
)
from strands_robots.dashboard.agent_motion import (
    MOTION_ENV,
    TASK_CONFIRM_ENV,
    task_post_allowed,
)

REAL_PEER = {"presence": {"robot_type": "so101_follower", "connected": True}}
SIM_PEER = {"presence": {"robot_type": "sim"}}

TASK_INPUT = {"action": "task", "target": "arm-1", "instruction": "wave", "duration": 5}


@pytest.fixture(autouse=True)
def _no_env_grant(monkeypatch):
    monkeypatch.delenv(MOTION_ENV, raising=False)
    # Grants are process-global; leave each test a clean slate.
    with agent_hitl._grants_lock:
        agent_hitl._grants.clear()


# --- motion_intent: which calls pause ---------------------------------------


def test_task_on_real_peer_yields_structured_reason():
    reason = motion_intent("fleet", TASK_INPUT, {"arm-1": REAL_PEER})
    assert reason["tool"] == "fleet"
    assert reason["action"] == "task"
    assert reason["target"] == "arm-1"
    assert reason["instruction"] == "wave"
    assert reason["duration"] == 5.0
    assert reason["why_physical"]


def test_sim_peer_is_never_gated():
    assert motion_intent("fleet", TASK_INPUT, {"arm-1": SIM_PEER}) is None


def test_unknown_target_is_treated_as_metal():
    assert motion_intent("fleet", TASK_INPUT, {}) is not None


def test_env_grant_is_the_always_allow_fast_lane():
    env = {MOTION_ENV: "1"}
    assert motion_intent("fleet", TASK_INPUT, {"arm-1": REAL_PEER}, env=env) is None


def test_stop_and_status_are_never_gated():
    for action in ("stop", "stop_all", "status", "peers", "emergency_stop"):
        assert motion_intent("fleet", {"action": action, "target": "arm-1"}, {"arm-1": REAL_PEER}) is None
    for tool, actions in MOTION_ACTIONS.items():
        assert "stop" not in actions and "emergency_stop" not in actions, tool


def test_ungated_tool_passes():
    assert motion_intent("calculator", {"action": "task"}, {"arm-1": REAL_PEER}) is None


def test_robot_mesh_is_never_gated_here_its_sdk_interrupt_owns_that():
    """Double-gating would ask the operator twice for one command (see MOTION_ACTIONS)."""
    assert (
        motion_intent("robot_mesh", {"action": "send", "target": "arm-1", "message": "go"}, {"arm-1": REAL_PEER})
        is None
    )
    assert motion_intent("robot_mesh", {"action": "broadcast", "message": "go"}, {}) is None
    assert motion_intent("robot_mesh", {"action": "peers"}, {}) is None


# --- response interpretation -------------------------------------------------


def test_only_explicit_yes_approves():
    assert response_approves(True)
    assert response_approves("yes")
    assert response_approves({"approve": True})
    assert not response_approves(False)
    assert not response_approves("no")
    assert not response_approves(None)
    assert not response_approves("")
    assert not response_approves({"approve": False})
    assert not response_approves(42)


# --- one-shot grants ----------------------------------------------------------


def test_grant_is_consumed_exactly_once():
    deposit_grant("fleet", TASK_INPUT)
    assert consume_grant("fleet", TASK_INPUT) is True
    assert consume_grant("fleet", TASK_INPUT) is False


def test_grant_does_not_leak_to_a_different_call():
    deposit_grant("fleet", TASK_INPUT)
    other = dict(TASK_INPUT, target="arm-2")
    assert consume_grant("fleet", other) is False
    assert consume_grant("fleet", TASK_INPUT) is True


# --- the hook against the real SDK event --------------------------------------


def _event(tool_input, agent):
    from strands.hooks import BeforeToolCallEvent

    return BeforeToolCallEvent(
        agent=agent,
        selected_tool=None,
        tool_use={"name": "fleet", "toolUseId": "t1", "input": tool_input},
        invocation_state={},
    )


class _FakeState:
    def __init__(self):
        self.interrupts = {}


class _FakeAgent:
    def __init__(self):
        self._interrupt_state = _FakeState()


def test_hook_raises_interrupt_for_physical_motion():
    from strands.interrupt import InterruptException

    hook = MotionInterruptHook(lambda: {"arm-1": REAL_PEER})
    event = _event(TASK_INPUT, _FakeAgent())
    with pytest.raises(InterruptException) as exc:
        hook._gate(event)
    interrupt = exc.value.interrupt
    assert interrupt.name == INTERRUPT_NAME
    assert interrupt.reason["target"] == "arm-1"
    assert event.cancel_tool is False


def test_hook_resume_yes_deposits_grant():
    agent = _FakeAgent()
    hook = MotionInterruptHook(lambda: {"arm-1": REAL_PEER})
    event = _event(TASK_INPUT, agent)

    from strands.interrupt import InterruptException

    with pytest.raises(InterruptException) as exc:
        hook._gate(event)
    # Human answers yes; the SDK stores the response and re-runs the hook.
    agent._interrupt_state.interrupts[exc.value.interrupt.id].response = {"approve": True}
    hook._gate(event)
    assert event.cancel_tool is False
    assert consume_grant("fleet", TASK_INPUT) is True


def test_hook_resume_no_cancels_with_a_relayable_sentence():
    agent = _FakeAgent()
    hook = MotionInterruptHook(lambda: {"arm-1": REAL_PEER})
    event = _event(TASK_INPUT, agent)

    from strands.interrupt import InterruptException

    with pytest.raises(InterruptException) as exc:
        hook._gate(event)
    agent._interrupt_state.interrupts[exc.value.interrupt.id].response = "no"
    hook._gate(event)
    assert isinstance(event.cancel_tool, str)
    assert "NOT sent" in event.cancel_tool
    assert consume_grant("fleet", TASK_INPUT) is False


def test_hook_lets_sims_through_without_interrupt():
    hook = MotionInterruptHook(lambda: {"arm-1": SIM_PEER})
    event = _event(TASK_INPUT, _FakeAgent())
    hook._gate(event)  # no raise
    assert event.cancel_tool is False


def test_hook_treats_unreadable_snapshot_as_metal():
    from strands.interrupt import InterruptException

    def boom():
        raise RuntimeError("snapshot unavailable")

    hook = MotionInterruptHook(boom)
    event = _event(TASK_INPUT, _FakeAgent())
    with pytest.raises(InterruptException):
        hook._gate(event)


def test_cancel_sentence_names_the_target():
    s = cancel_sentence({"target": "arm-1", "instruction": "wave"})
    assert "arm-1" in s and "wave" in s


# --- the gated target comes from a trusted source, not from the model --------
#
# The model authors ``tool_input`` and the ``peers`` action that lists a sim's
# name is deliberately ungated, so a sim peer's name is always within its reach.
# Each row is a gated tool whose target the model does NOT get to name, paired
# with the trusted source that names it instead.

SERIAL_CALL = {"action": "feetech_position", "port": "/dev/ttyACM0", "motor_id": 1, "position": 2000}
POSE_CALL = {"action": "move_motor", "port": "/dev/ttyACM0", "motor_name": "shoulder", "position": 500}
PROXY_CALL = {"action": "task", "instruction": "wave"}
PROXY_ACTIONS = {"arm1_proxy": frozenset({"task"})}
PROXY_BINDING = {"arm1_proxy": "arm-1"}

FLEET = {"arm-1": REAL_PEER, "sim-1": SIM_PEER}

#: (tool, call without a target, the target its trusted source names, extra_actions, bound_targets)
TRUSTED_TARGETS = [
    ("serial_tool", SERIAL_CALL, "/dev/ttyACM0", None, None),
    ("pose_tool", POSE_CALL, "/dev/ttyACM0", None, None),
    ("arm1_proxy", PROXY_CALL, "arm-1", PROXY_ACTIONS, PROXY_BINDING),
]


@pytest.mark.parametrize(("tool", "call", "trusted", "extra", "bound"), TRUSTED_TARGETS)
def test_the_trusted_source_names_the_target(tool, call, trusted, extra, bound):
    """Control: with no target written, the gate resolves it and pauses."""
    reason = motion_intent(tool, dict(call), FLEET, {}, extra_actions=extra, bound_targets=bound)
    assert reason is not None, f"{tool}: a real target was not gated"
    assert reason["target"] == trusted


@pytest.mark.parametrize(("tool", "call", "trusted", "extra", "bound"), TRUSTED_TARGETS)
def test_a_model_written_target_cannot_stand_the_gate_down(tool, call, trusted, extra, bound):
    """Naming a sim in the input does not move the target the gate resolved.

    Standing the gate down here would put a real Feetech command on the port, or
    a task on the bound peer, with no human asked: these tools raise no
    interrupt of their own, so this layer is their only human gate.
    """
    reason = motion_intent(tool, {**call, "target": "sim-1"}, FLEET, {}, extra_actions=extra, bound_targets=bound)
    assert reason is not None, f"{tool}: writing target='sim-1' skipped the gate entirely"
    assert reason["target"] == trusted, f"{tool}: the model renamed the target the operator is shown"


def test_a_tool_that_declares_target_is_still_read_from_it():
    """Over-reach control: ``fleet`` declares ``target``, so it is trusted there.

    The gate and the tool must resolve the same peer from the same field, so a
    genuinely simulated fleet target stays ungated.
    """
    assert motion_intent("fleet", {**TASK_INPUT, "target": "sim-1"}, FLEET) is None
    assert motion_intent("fleet", TASK_INPUT, FLEET) is not None


def test_an_unresolvable_target_is_gated():
    """A direct-serial call with no port resolves to nothing, which is metal."""
    call = {k: v for k, v in SERIAL_CALL.items() if k != "port"}
    reason = motion_intent("serial_tool", call, FLEET, {})
    assert reason is not None
    assert reason["target"] == "(unnamed peer)"


# --- the task-POST confirmation is a posture flag, so it is checked ----------

CONFIRM_ON = {TASK_CONFIRM_ENV: "1"}

#: Spellings an operator or script reaches for to say "no", every one of them truthy.
TRUTHY_SPELLINGS_OF_OFF = ["false", "no", "off", "0", "maybe", 1, [0], {"a": 1}]


@pytest.mark.parametrize("confirmed", TRUTHY_SPELLINGS_OF_OFF)
def test_a_non_boolean_confirmation_does_not_confirm(confirmed):
    """A truthy non-boolean must not satisfy a confirmation the operator required."""
    verdict = task_post_allowed(peer=REAL_PEER, confirmed=confirmed, target="arm-1", env=CONFIRM_ON)
    assert verdict["allowed"] is False, f"{confirmed!r} started real motion"
    assert verdict["confirmed"] is False
    assert "must be a boolean" in verdict["reason"]


@pytest.mark.parametrize(("confirmed", "allowed"), [(True, True), (False, False)])
def test_the_accepted_domain_is_unchanged(confirmed, allowed):
    """Control: a real boolean decides the verdict exactly as before."""
    verdict = task_post_allowed(peer=REAL_PEER, confirmed=confirmed, target="arm-1", env=CONFIRM_ON)
    assert verdict["allowed"] is allowed


def test_a_dashboard_that_asked_for_no_confirmation_reads_no_flag():
    """Control: the guard sits on the branch that reads the flag, not above it.

    With the requirement off the field is never consulted, so a request is not
    refused for the shape of a value that decides nothing.
    """
    verdict = task_post_allowed(peer=REAL_PEER, confirmed="false", target="arm-1", env={})
    assert verdict["allowed"] is True
    assert verdict["gated"] is False
