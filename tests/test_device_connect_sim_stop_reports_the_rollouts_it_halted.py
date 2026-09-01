"""A remote stop must report which rollouts it halted, not assert that it did.

``SimulationDeviceDriver.stop`` is one of two remote ways into a running
simulation. It lowered ``policy_running`` itself and answered a fixed
``"All policies stopped"``, so four different facts arrived as one sentence:

=========================================  ==================================
situation                                  what the operator was told
=========================================  ==================================
a rollout was in flight and was halted     ``All policies stopped``
nothing was running                        ``All policies stopped``
the world was already torn down            ``All policies stopped``
the world mutated under the stop loop      ``RuntimeError`` past the RPC
=========================================  ==================================

None of them named a robot, so the answer could not be cross-checked against
``list_policies_running``, and the last one was not an answer at all -- an RPC
that raises past the dispatcher reaches the caller as a transport-level failure
carrying none of what did or did not stop.

The other remote way in already answers properly. ``{"action": "stop"}`` on the
mesh (:meth:`strands_robots.mesh.Mesh._dispatch`) calls ``stop_policy`` per
robot, grades each answer through
:func:`strands_robots.mesh.core._reports_failure_to_stop`, names the halted
robots under ``stopped`` and the refusals under ``not_stopped``, and returns
``{"ok": True, "stopped": []}`` for a peer with nothing to halt -- explicitly
NOT a refusal, because "did not stop" on a peer with nothing to stop is the
false alarm that trains an operator to ignore the warning. Every comment
justifying that shape is in the branch; the Device Connect verb next to it
produced no answer to grade.

Why the module did not catch it: the shared stand-in for the wrapped simulation
is a ``MagicMock``, which answers ``hasattr("stop_policy")`` by fabrication,
absorbs the stop, and returns an envelope carrying no verdict. Its own comment
already records the same lesson one attribute earlier -- the robot record is a
real :class:`~strands_robots.simulation.models.SimRobot` because "a mock absorbs
that call while leaving ``policy_running`` exactly as the test set it". The one
cell over the verb asserted a flag and a status, which a fixed success text
satisfies in every row of the table above.

Graded here:

* the answer names the rollouts it halted, and only those;
* nothing to halt answers affirmatively-empty, and the durable stop still ran;
* a world mutating under the loop is answered, not raised;
* a refusal is named, and it is graded through the rule's single owner;
* an answer carrying no verdict is read as neither halt nor idle;
* the two remote surfaces report the same halted set for the same simulation;
* the stand-ins match what the real ``stop_policy`` and the real driver do.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from typing import Any

import pytest

from strands_robots.mesh import Mesh
from strands_robots.mesh.core import _reports_failure_to_stop
from strands_robots.simulation.models import SimRobot
from tests._sim_stop_policy_stand_in import stop_policy_stand_in

pytest.importorskip("device_connect_edge")

# The driver is imported per test, not here. A sibling file replaces the
# device_connect_edge submodules with MagicMocks at import time; importing the
# driver at module scope binds it to whichever of the two won collection, and
# leaves that binding cached for the other file. The autouse fixture below
# restores the real submodules and purges the cache, the same way
# ``tests.test_estop_halts_every_simulation_motion_source`` does over this same
# driver. An autouse fixture is bound to the module that declares it, so
# importing the helper does not bring the sibling's fixture along.
from tests.test_device_connect_hardening import (  # noqa: E402 - after the extra check
    _force_real_device_connect_edge,
)


@pytest.fixture(autouse=True)
def _real_device_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind the driver to the real extra and leave the allowlists permissive."""
    _force_real_device_connect_edge()
    for var in ("DEVICE_CONNECT_RPC_ALLOW", "DEVICE_CONNECT_ESTOP_ALLOW", "DEVICE_CONNECT_ALLOW_INSECURE"):
        monkeypatch.delenv(var, raising=False)


def _sim_driver() -> Any:
    """The driver module, imported after the fixture has bound the real extra."""
    import strands_robots.device_connect.sim_driver as sim_driver

    return sim_driver


class _World:
    def __init__(self, names: list[str]) -> None:
        self.robots = {n: SimRobot(name=n, urdf_path="") for n in names}
        self.sim_time = 0.0
        self.step_count = 0


class _SimPeer:
    """A simulation exposing the two attributes both stop routers look for.

    Deliberately not a ``Mock``: ``hasattr`` is what the routers test, and a
    mock answers every ``hasattr`` truthfully-by-fabrication. ``stop_policy``
    mirrors the real one, including the ``was_running`` verdict in its ``json``
    block -- grounded against production by
    :class:`TestTheStandInMatchesTheRealSimulation`.

    Args:
        names: Robots in the world.
        active: Robots with a rollout in flight.
        refuse: Robots whose ``stop_policy`` answers ``status="error"``.
        silent: Robots whose ``stop_policy`` answers with no verdict at all,
            standing in for a backend that does not report the fact.
        world: ``False`` for a simulation whose world is already torn down.
    """

    def __init__(
        self,
        names: list[str],
        active: tuple[str, ...] = (),
        refuse: tuple[str, ...] = (),
        silent: tuple[str, ...] = (),
        world: bool = True,
    ) -> None:
        self._world = _World(names) if world else None
        self._active = list(active)
        self._refuse = set(refuse)
        self._silent = set(silent)
        self.stop_policy_calls: list[str] = []
        if self._world is not None:
            for name in self._active:
                self._world.robots[name].policy_running = True

    def _active_policy_robots(self) -> list[str]:
        return list(self._active)

    def stop_policy(self, robot_name: str = "") -> dict[str, Any]:
        self.stop_policy_calls.append(robot_name)
        if robot_name in self._refuse:
            return {"status": "error", "content": [{"text": f"Unknown robot '{robot_name}'."}]}
        answer = stop_policy_stand_in(self._world)(robot_name)
        if robot_name in self._silent:
            return {"status": answer["status"], "content": [b for b in answer["content"] if "json" not in b]}
        return answer


def _rpc_stop(peer: _SimPeer) -> dict[str, Any]:
    """Answer the Device Connect ``stop`` RPC gives for *peer*."""
    return asyncio.run(_sim_driver().SimulationDeviceDriver(peer).stop())


def _mesh_stop(peer: _SimPeer) -> dict[str, Any]:
    """Answer the mesh ``{"action": "stop"}`` fanout gives for *peer*."""
    return Mesh(peer, peer_id="sim-1", peer_type="simulation")._dispatch({"action": "stop"})


def _verdict(envelope: dict[str, Any]) -> dict[str, Any]:
    """The ``json`` verdict block of a stop envelope."""
    for block in envelope.get("content", []):
        if isinstance(block.get("json"), dict):
            return dict(block["json"])
    raise AssertionError(f"no json verdict block in {envelope}")


def _text(envelope: dict[str, Any]) -> str:
    return " ".join(block["text"] for block in envelope.get("content", []) if "text" in block)


class TestTheAnswerNamesTheRolloutsItHalted:
    """The halted set is reported, so it can be checked against the simulation."""

    def test_a_halted_rollout_is_named(self) -> None:
        answer = _rpc_stop(_SimPeer(["arm", "arm2"], active=("arm",)))

        assert answer["status"] == "success", answer
        assert _verdict(answer)["stopped"] == ["arm"], answer
        assert "arm" in _text(answer), answer

    def test_an_idle_robot_is_not_named_as_halted(self) -> None:
        """The distinction a fixed success text could not make."""
        answer = _rpc_stop(_SimPeer(["arm", "arm2"], active=("arm",)))

        assert "arm2" not in _verdict(answer)["stopped"], answer

    def test_every_rollout_in_flight_is_named(self) -> None:
        answer = _rpc_stop(_SimPeer(["arm", "arm2"], active=("arm", "arm2")))

        assert _verdict(answer)["stopped"] == ["arm", "arm2"], answer

    def test_the_stop_goes_through_the_verb_that_owns_the_question(self) -> None:
        """Not a private flag write beside it -- ``stop_policy`` unions the
        rollout registry into its answer, which the flag alone cannot (#2833)."""
        peer = _SimPeer(["arm", "arm2"], active=("arm",))

        _rpc_stop(peer)

        assert peer.stop_policy_calls == ["arm", "arm2"], peer.stop_policy_calls


class TestNothingToHaltAnswersEmptyRatherThanRefusing:
    """Over-reach control: an empty answer is affirmative, never an error.

    The mesh branch states the reason in full -- a peer with nothing to halt
    reported as "did not stop" put an operator's own gateway in
    ``peers_not_stopped`` and fired the CRITICAL warning on every stop.
    """

    @pytest.mark.parametrize(
        "peer_factory",
        [
            pytest.param(lambda: _SimPeer(["arm", "arm2"]), id="idle-simulation"),
            pytest.param(lambda: _SimPeer([], world=False), id="world-torn-down"),
            pytest.param(lambda: _SimPeer([]), id="empty-world"),
        ],
    )
    def test_nothing_to_halt_is_a_successful_empty_answer(self, peer_factory: Any) -> None:
        answer = _rpc_stop(peer_factory())

        assert answer["status"] == "success", answer
        assert _verdict(answer) == {"stopped": [], "not_stopped": [], "unreported": []}, answer

    def test_an_idle_simulation_is_not_told_that_policies_stopped(self) -> None:
        answer = _rpc_stop(_SimPeer(["arm"]))

        assert "No rollout was in flight" in _text(answer), answer

    def test_the_durable_stop_still_runs_on_an_idle_robot(self) -> None:
        """Reporting honestly must not make the stop conditional: the stop
        counter is what keeps a worker that has not reached its first frame
        from raising the flag back over this stop."""
        peer = _SimPeer(["arm"])

        _rpc_stop(peer)

        assert peer._world is not None
        assert peer._world.robots["arm"].policy_stops == 1


class TestAStopAnswersRatherThanRaises:
    """A scene teardown racing the loop is answered, like both siblings do."""

    @staticmethod
    def _peer_whose_world_mutates() -> _SimPeer:
        class _Mutating(dict):
            def __iter__(self) -> Any:
                raise RuntimeError("dictionary changed size during iteration")

        peer = _SimPeer(["arm"], active=("arm",))
        assert peer._world is not None
        peer._world.robots = _Mutating()
        return peer

    def test_the_race_is_reported_as_an_error_envelope(self) -> None:
        answer = _rpc_stop(self._peer_whose_world_mutates())

        assert answer["status"] == "error", answer
        assert "changed size during iteration" in _text(answer), answer

    def test_the_race_is_flagged_by_the_shared_failure_rule(self) -> None:
        """So the accounting that reads stop answers counts it as not stopped."""
        assert _reports_failure_to_stop(_rpc_stop(self._peer_whose_world_mutates())) is True

    def test_the_race_does_not_claim_a_halt(self) -> None:
        assert _verdict(_rpc_stop(self._peer_whose_world_mutates()))["stopped"] == []


class TestARefusedRolloutIsNamed:
    """A stop_policy refusal reaches the envelope rather than the payload."""

    def test_a_refusal_makes_the_answer_negative_and_names_the_robot(self) -> None:
        answer = _rpc_stop(_SimPeer(["arm", "arm2"], active=("arm", "arm2"), refuse=("arm2",)))

        assert answer["status"] == "error", answer
        assert _verdict(answer)["not_stopped"] == ["arm2"], answer

    def test_what_did_halt_is_still_reported_beside_the_refusal(self) -> None:
        answer = _rpc_stop(_SimPeer(["arm", "arm2"], active=("arm", "arm2"), refuse=("arm2",)))

        assert _verdict(answer)["stopped"] == ["arm"], answer
        assert _verdict(answer)["results"]["arm2"]["status"] == "error", answer

    def test_the_refusal_rule_is_read_from_its_single_owner(self) -> None:
        """A second copy of the rule is how the mesh branch came to report a
        refusal as a halt; this surface reads the owner instead.

        The rule is the COMPARISON, not the word: building an error envelope
        spells ``"error"`` legitimately. Graded with the same pattern
        ``tests.mesh.test_fleet_stop_reports_a_refused_rollout`` uses over the
        owner's own module, so a third copy fails here rather than drifting.
        """
        source = inspect.getsource(_sim_driver().SimulationDeviceDriver.stop)
        rule = re.compile(r'\.get\(\s*["\']status["\']\s*\)\s*==\s*["\']error["\']')

        assert "_reports_failure_to_stop(" in source, source
        assert rule.findall(source) == [], "the status=='error' rule is spelled outside its owner"


class TestAnAnswerWithNoVerdictIsReadAsNeither:
    """Silence is not evidence of a halt, and not evidence of an idle robot."""

    def test_a_silent_answer_is_not_counted_as_a_halt(self) -> None:
        answer = _rpc_stop(_SimPeer(["arm"], active=("arm",), silent=("arm",)))

        assert _verdict(answer)["stopped"] == [], answer
        assert _verdict(answer)["unreported"] == ["arm"], answer

    def test_a_silent_answer_does_not_license_the_idle_sentence(self) -> None:
        answer = _rpc_stop(_SimPeer(["arm"], active=("arm",), silent=("arm",)))

        assert "No rollout was in flight" not in _text(answer), answer

    @pytest.mark.parametrize(
        ("envelope", "expected"),
        [
            pytest.param({"content": [{"json": {"was_running": True}}]}, True, id="reported-running"),
            pytest.param({"content": [{"json": {"was_running": False}}]}, False, id="reported-idle"),
            pytest.param({"content": [{"text": "Stopped on 'arm'"}]}, None, id="prose-only"),
            pytest.param({"content": [{"json": {"robot": "arm"}}]}, None, id="json-without-the-key"),
            pytest.param({}, None, id="no-content"),
        ],
    )
    def test_the_verdict_reader_is_tri_state(self, envelope: dict[str, Any], expected: bool | None) -> None:
        assert _sim_driver()._reported_a_rollout_in_flight(envelope) is expected


class TestBothRemoteSurfacesReportTheSameHaltedSet:
    """One simulation, two remote operators, one set of facts."""

    @pytest.mark.parametrize(
        ("peer_factory", "halted"),
        [
            pytest.param(lambda: _SimPeer(["arm", "arm2"], active=("arm",)), ["arm"], id="one-rollout"),
            pytest.param(lambda: _SimPeer(["arm", "arm2"]), [], id="none-running"),
            pytest.param(
                lambda: _SimPeer(["arm", "arm2"], active=("arm", "arm2")),
                ["arm", "arm2"],
                id="two-rollouts",
            ),
        ],
    )
    def test_the_two_surfaces_agree(self, peer_factory: Any, halted: list[str]) -> None:
        rpc_answer = _rpc_stop(peer_factory())
        mesh_answer = _mesh_stop(peer_factory())

        assert _verdict(rpc_answer)["stopped"] == halted, rpc_answer
        assert mesh_answer["stopped"] == halted, mesh_answer

    def test_neither_surface_flags_a_stop_that_worked(self) -> None:
        assert _reports_failure_to_stop(_rpc_stop(_SimPeer(["arm"], active=("arm",)))) is False
        assert _reports_failure_to_stop(_mesh_stop(_SimPeer(["arm"], active=("arm",)))) is False


class TestTheStandInMatchesTheRealSimulation:
    """Premise of every cell above: the fake answers what production answers."""

    @staticmethod
    def _real_sim() -> Any:
        pytest.importorskip("mujoco")
        from strands_robots.simulation import create_simulation

        sim: Any = create_simulation("mujoco")
        sim.create_world()
        assert sim.add_robot("so100")["status"] == "success"
        return sim

    def test_the_real_stop_policy_reports_its_verdict_as_data(self) -> None:
        sim = self._real_sim()
        try:
            sim._world.robots["so100"].policy_running = True
            halted = sim.stop_policy("so100")
            idle = sim.stop_policy("so100")
            refusal = sim.stop_policy("no-such-robot")
        finally:
            sim.cleanup()

        in_flight = _sim_driver()._reported_a_rollout_in_flight
        assert in_flight(halted) is True, halted
        assert in_flight(idle) is False, idle
        assert _reports_failure_to_stop(refusal) is True, refusal

    @pytest.mark.parametrize("in_flight", [True, False], ids=["rollout-in-flight", "idle"])
    def test_the_stand_in_answers_what_the_real_verb_answers(self, in_flight: bool) -> None:
        """Key for key, on the same situation - the premise, not a restatement."""
        sim = self._real_sim()
        try:
            sim._world.robots["so100"].policy_running = in_flight
            real = sim.stop_policy("so100")
            sim._world.robots["so100"].policy_running = in_flight
            stood_in = stop_policy_stand_in(sim._world)("so100")
        finally:
            sim.cleanup()

        assert stood_in == real, (stood_in, real)

    def test_the_stand_in_refuses_an_unresolvable_robot_like_the_real_verb(self) -> None:
        """On the dimensions the consumer reads, which is not the prose.

        The real refusal names the resolvable robots and points at
        ``list_robots``; the driver reads ``status`` through the shared failure
        rule and reads no verdict. Copying that sentence into a stand-in would
        be a claim about production that nothing consumes and that drifts the
        first time the hint is reworded.
        """
        sim = self._real_sim()
        try:
            real = sim.stop_policy("no-such-robot")
            stood_in = stop_policy_stand_in(sim._world)("no-such-robot")
        finally:
            sim.cleanup()

        in_flight = _sim_driver()._reported_a_rollout_in_flight
        assert stood_in["status"] == real["status"] == "error", (stood_in, real)
        assert _reports_failure_to_stop(stood_in) is _reports_failure_to_stop(real) is True
        assert in_flight(stood_in) is in_flight(real) is None, (stood_in, real)

    def test_the_real_stop_policy_still_leads_with_its_sentence(self) -> None:
        """The text block stays first: existing readers index it positionally."""
        sim = self._real_sim()
        try:
            answer = sim.stop_policy("so100")
        finally:
            sim.cleanup()

        assert answer["content"][0]["text"] == "Was not running on 'so100'", answer

    def test_the_driver_names_a_real_background_rollout_it_halted(self) -> None:
        """End to end: a real rollout, the real driver, the real stop verb."""
        sim = self._real_sim()
        try:
            assert sim.start_policy(robot_name="so100", policy_provider="mock", duration=30.0)["status"] == "success"
            answer = asyncio.run(_sim_driver().SimulationDeviceDriver(sim).stop())
        finally:
            sim.cleanup()

        assert answer["status"] == "success", answer
        assert _verdict(answer)["stopped"] == ["so100"], answer
        assert "so100" in _text(answer), answer
