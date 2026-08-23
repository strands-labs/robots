"""Pin: the robot_mesh approval prompt states what it verified about the target.

The tool gates by ACTION (``_resolve_interrupt_actions``), so before this the
operator prompt asserted ``"Physical effect on peer '<target>'"`` for every gated
single-target call - including one aimed at a peer whose presence reports
``robot_type: "sim"``. The claim was byte-identical for a real arm, a sim twin and
a peer that is not on the fleet snapshot at all, so the one sentence an operator
reads to decide carried a fact the gate had never established.

Two halves are pinned here:

* :func:`~strands_robots.mesh.session.peer_is_physical` classifies a peers-snapshot
  entry, FAIL-CLOSED: physical unless the peer can be SHOWN to be a sim. It reads
  the flat dict :meth:`~strands_robots.mesh.session.PeerInfo.to_dict` returns, and
  its two directions differ on purpose - ``hw`` (a metal marker) is read
  permissively and first, while ``robot_type`` / ``world`` (sim markers) are read
  strictly, so an unreadable marker falls through to metal.
* the prompt reports that verdict instead of asserting physicality, in both the
  single-target and the fleet-wide scope.

What is deliberately NOT changed: WHICH actions are gated. A gated action aimed at
a classified sim still raises the interrupt, because ``robot_type`` and ``world``
arrive over the wire and ``Mesh._on_presence`` authenticates neither - a peer can
claim to be a sim. An unauthenticated self-report is fit to inform an operator and
unfit to replace one, so skipping the approval on it is a safety-posture decision
for a maintainer rather than something this change takes. The controls in
:class:`TestTheGatedActionSetIsUnchanged` fail if it is taken by accident.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import strands_robots.tools.robot_mesh as rmt
from strands_robots.mesh.session import clear_peers, update_peer

_SIM_ID = "so101_sim-a1b2"
_REAL_ID = "so101_real-c3d4"
_ABSENT_ID = "ghost-9999"


@pytest.fixture(autouse=True)
def _isolate():
    """Each test starts from an empty peer registry and un-cached gate config."""
    clear_peers()
    rmt._reset_rate_limits()
    rmt._reset_interrupt_actions_cache()
    yield
    clear_peers()
    rmt._reset_rate_limits()
    rmt._reset_interrupt_actions_cache()


def _announce(peer_id: str, **presence: Any) -> None:
    """Put a peer on the snapshot exactly as ``Mesh._on_presence`` would.

    ``update_peer(caps=data)`` stores the whole presence payload, and
    ``PeerInfo.to_dict`` spreads it at the top level - so the shape the classifier
    sees here is the shape the wire produces, not a hand-built nesting.
    """
    payload: dict[str, Any] = {
        "robot_id": peer_id,
        "hostname": "thor",
        "timestamp": time.time(),
        **presence,
    }
    update_peer(
        peer_id=peer_id,
        peer_type=str(payload.get("robot_type", "robot")),
        hostname="thor",
        caps=payload,
    )


def _sim() -> None:
    _announce(_SIM_ID, robot_type="sim", world=True, sim_robots=["so101"])


def _real() -> None:
    _announce(_REAL_ID, robot_type="robot", hw="so101_follower", connected=True)


def _prompt(action: str, target: str, **kw: Any) -> dict[str, Any]:
    """Drive the real gate and return the interrupt's ``reason`` dict."""
    ctx = MagicMock(name="ToolContext")
    ctx.interrupt.return_value = "y"
    stub = MagicMock()
    stub.tell.return_value = {"status": "ok"}
    stub.emergency_stop.return_value = {"status": "ok"}
    stub.broadcast.return_value = {"status": "ok"}
    fn = getattr(rmt.robot_mesh, "__wrapped__", None) or rmt.robot_mesh
    with patch.object(rmt, "_resolve_mesh", return_value=stub):
        fn(action=action, target=target, tool_context=ctx, **kw)
    if not ctx.interrupt.called:
        pytest.fail(f"the gate did not ask the operator for {action!r} -> {target!r}")
    return dict(ctx.interrupt.call_args.kwargs["reason"])


# --- the classifier -----------------------------------------------------


class TestThePeerClassifierFailsClosed:
    """A peer is metal unless its presence SHOWS it is a sim."""

    def test_an_absent_peer_is_physical(self):
        from strands_robots.mesh.session import peer_is_physical

        physical, why = peer_is_physical(None)
        assert physical is True
        assert "not on the fleet snapshot" in why

    @pytest.mark.parametrize("declared", ["sim", "simulation", "mujoco", "SIM", "  Sim  "])
    def test_a_declared_sim_type_is_a_sim(self, declared):
        from strands_robots.mesh.session import peer_is_physical

        physical, why = peer_is_physical({"peer_id": "p", "robot_type": declared})
        assert physical is False
        assert declared.strip().lower() in why

    @pytest.mark.parametrize("declared", ["robot", "agent", "gateway", "", "simulator", "simulated"])
    def test_every_other_type_is_physical(self, declared):
        """Only the three exact tokens name a sim; anything else fails closed.

        ``simulator`` / ``simulated`` are the near-misses worth naming: a token
        this function does not know must read as metal, never as a sim.
        """
        from strands_robots.mesh.session import peer_is_physical

        assert peer_is_physical({"peer_id": "p", "robot_type": declared})[0] is True

    def test_reported_hardware_overrides_a_sim_claim(self):
        """``hw`` is checked first, so a peer naming real hardware is metal.

        A presence carrying both markers is exactly the case where fail-closed
        has to win: whatever else it says, it says it has an arm.
        """
        from strands_robots.mesh.session import peer_is_physical

        physical, why = peer_is_physical({"peer_id": "p", "robot_type": "sim", "world": True, "hw": "so101_follower"})
        assert physical is True
        assert "so101_follower" in why

    @pytest.mark.parametrize("hw", ["", "   ", None, 0, 1, [], ["so101"]])
    def test_an_unreadable_hardware_marker_does_not_claim_metal_by_itself(self, hw):
        """A non-string / blank ``hw`` is not a hardware report, so the later
        rungs decide. Paired with the sim markers below, this pins that the
        permissive read of ``hw`` is bounded to a non-empty string."""
        from strands_robots.mesh.session import peer_is_physical

        assert peer_is_physical({"peer_id": "p", "robot_type": "sim", "hw": hw})[0] is False

    def test_a_simulation_world_is_a_sim(self):
        """``world``/``sim_robots`` is the sim marker the presence publisher sets
        for a peer registered into a MuJoCo world, whose own ``robot_type`` need
        not be one of the three tokens."""
        from strands_robots.mesh.session import peer_is_physical

        physical, why = peer_is_physical({"peer_id": "p", "robot_type": "robot", "world": True})
        assert physical is False
        assert "simulation world" in why
        assert peer_is_physical({"peer_id": "p", "robot_type": "robot", "sim_robots": ["so101"]})[0] is False

    @pytest.mark.parametrize("world", [1, "true", "yes", [1], {"a": 1}])
    def test_a_truthy_but_unreadable_world_marker_is_not_a_sim(self, world):
        """The sim markers are read STRICTLY: ``world is True``, not merely
        truthy. A value this function cannot read falls through to metal, which
        is the fail-closed direction for a POSITIVE sim marker."""
        from strands_robots.mesh.session import peer_is_physical

        assert peer_is_physical({"peer_id": "p", "robot_type": "robot", "world": world})[0] is True

    @pytest.mark.parametrize("sim_robots", [[], (), "so101", 3, None])
    def test_an_empty_or_unreadable_robot_list_is_not_a_sim(self, sim_robots):
        from strands_robots.mesh.session import peer_is_physical

        assert peer_is_physical({"peer_id": "p", "robot_type": "robot", "sim_robots": sim_robots})[0] is True

    def test_the_verdict_reads_after_the_peer_name(self):
        """Every reason is phrased to follow the peer's name, because that is how
        the prompt composes it."""
        from strands_robots.mesh.session import peer_is_physical

        for peer in (
            None,
            {"peer_id": "p"},
            {"peer_id": "p", "robot_type": "sim"},
            {"peer_id": "p", "hw": "arm"},
            {"peer_id": "p", "world": True},
        ):
            why = peer_is_physical(peer)[1]
            assert why.startswith("it "), why
            assert not why.endswith("."), why


# --- the prompt ---------------------------------------------------------


class TestTheApprovalPromptStatesWhatItVerified:
    """The one sentence an operator reads must not assert an unchecked fact."""

    def test_a_sim_target_is_not_announced_as_a_physical_effect(self):
        _sim()
        reason = _prompt("tell", _SIM_ID, instruction="pick up the cube")
        warning = str(reason["warning"])
        assert "Physical effect" not in warning, warning
        assert "not known to be physical" in warning
        assert "reports itself as sim" in warning
        assert reason["physical"] is False

    def test_a_real_target_is_announced_as_a_physical_effect_and_names_the_hardware(self):
        _real()
        reason = _prompt("tell", _REAL_ID, instruction="pick up the cube")
        warning = str(reason["warning"])
        assert f"Physical effect on peer '{_REAL_ID}'" in warning
        assert "so101_follower" in warning
        assert reason["physical"] is True

    def test_an_absent_target_is_announced_as_a_physical_effect(self):
        """Fail-closed at the prompt: a peer this process has not discovered is
        announced as physical, and the prompt says why."""
        reason = _prompt("tell", _ABSENT_ID, instruction="pick up the cube")
        warning = str(reason["warning"])
        assert f"Physical effect on peer '{_ABSENT_ID}'" in warning
        assert "not on the fleet snapshot" in warning
        assert reason["physical"] is True

    def test_a_sim_and_a_real_target_do_not_get_the_same_sentence(self):
        """The defect in one assertion: the two prompts were byte-identical apart
        from the peer id."""
        _sim()
        _real()
        sim_warning = str(_prompt("tell", _SIM_ID, instruction="go")["warning"])
        rmt._reset_rate_limits()
        real_warning = str(_prompt("tell", _REAL_ID, instruction="go")["warning"])
        assert sim_warning.replace(_SIM_ID, "X") != real_warning.replace(_REAL_ID, "X")

    def test_the_prompt_and_the_structured_verdict_agree(self):
        """``physical`` is the same fact the sentence states, so a host UI reading
        the structured reason cannot disagree with the operator's sentence."""
        _sim()
        _real()
        for target in (_SIM_ID, _REAL_ID, _ABSENT_ID):
            rmt._reset_rate_limits()
            reason = _prompt("tell", target, instruction="go")
            warning = str(reason["warning"])
            assert ("Physical effect" in warning) is bool(reason["physical"]), (target, warning)
            assert str(reason["verified"]) in warning

    def test_the_fleet_wide_scope_is_the_two_broadcast_actions(self):
        """The scope branch and its wording read one named set, so a third
        fleet-wide action cannot reach the prompt with single-target phrasing."""
        assert rmt._FLEET_WIDE_ACTIONS == frozenset({"emergency_stop", "broadcast"})

    def test_a_fleet_wide_action_over_sims_only_does_not_claim_a_physical_effect(self):
        _sim()
        _announce("so101_sim2-e5f6", robot_type="sim", world=True)
        reason = _prompt("emergency_stop", "")
        warning = str(reason["warning"])
        assert "Physical effect" not in warning, warning
        assert "all 2 peers" in warning
        assert reason["physical"] is False

    def test_a_fleet_wide_action_reaching_one_real_peer_claims_a_physical_effect(self):
        _sim()
        _real()
        reason = _prompt("emergency_stop", "")
        warning = str(reason["warning"])
        assert "Fleet-wide physical effect" in warning
        assert "1 of 2 peers" in warning
        assert reason["physical"] is True

    def test_a_fleet_wide_action_with_an_empty_snapshot_claims_a_physical_effect(self):
        """Fail-closed: no peer discovered yet is not evidence of a sim fleet."""
        reason = _prompt("emergency_stop", "")
        warning = str(reason["warning"])
        assert "Fleet-wide physical effect" in warning
        assert "no peer is on the fleet snapshot" in warning
        assert reason["physical"] is True


# --- controls ----------------------------------------------------------


class TestTheGatedActionSetIsUnchanged:
    """Which actions stop and ask a human is NOT what this change touches.

    Each of these passes both before and after: they fail if the classifier is
    ever wired to SKIP an approval, which would let an unauthenticated,
    peer-authored ``robot_type`` disarm an operator gate.
    """

    @pytest.mark.parametrize(
        "action,kw",
        [
            ("tell", {"instruction": "go"}),
            ("stop", {}),
            ("send", {"command": '{"action": "status"}'}),
        ],
    )
    def test_a_gated_action_aimed_at_a_sim_still_asks_the_operator(self, action, kw):
        _sim()
        ctx = MagicMock(name="ToolContext")
        ctx.interrupt.return_value = "y"
        stub = MagicMock()
        fn = getattr(rmt.robot_mesh, "__wrapped__", None) or rmt.robot_mesh
        extra: dict[str, Any] = dict(kw)
        with patch.object(rmt, "_resolve_mesh", return_value=stub):
            fn(action=action, target=_SIM_ID, tool_context=ctx, **extra)
        ctx.interrupt.assert_called_once()

    def test_a_declined_sim_target_is_still_refused(self):
        """The sim classification is a description, not an authorisation: a
        declined approval still stops the action."""
        _sim()
        ctx = MagicMock(name="ToolContext")
        ctx.interrupt.return_value = "n"
        stub = MagicMock()
        fn = getattr(rmt.robot_mesh, "__wrapped__", None) or rmt.robot_mesh
        with patch.object(rmt, "_resolve_mesh", return_value=stub):
            r = fn(action="tell", target=_SIM_ID, instruction="go", tool_context=ctx)
        assert r["status"] == "error"
        assert "declined" in r["content"][0]["text"].lower()
        stub.tell.assert_not_called()

    def test_the_default_gated_set_is_unchanged(self):
        assert rmt._resolve_interrupt_actions() == frozenset(
            {"emergency_stop", "broadcast", "tell", "send", "stop", "rpc"}
        )

    def test_a_fleet_wide_action_still_approves_against_every_peer(self):
        """The approval target is unchanged: fleet-wide actions are still
        presented as reaching all peers, whatever the classification says."""
        _sim()
        reason = _prompt("emergency_stop", "")
        assert reason["target"] == "*ALL_PEERS*"

    def test_the_gate_does_not_start_a_transport_to_classify_a_peer(self):
        """The prompt reads the in-process registry only. A gate that acquired a
        transport to decide what to tell the operator would make the approval
        depend on the network being up."""
        _sim()
        ctx = MagicMock(name="ToolContext")
        ctx.interrupt.return_value = "n"
        fn = getattr(rmt.robot_mesh, "__wrapped__", None) or rmt.robot_mesh
        with patch.object(rmt, "_resolve_mesh", return_value=MagicMock()), patch.object(rmt, "_gateway_mesh") as gw:
            fn(action="tell", target=_SIM_ID, instruction="go", tool_context=ctx)
        gw.assert_not_called()
