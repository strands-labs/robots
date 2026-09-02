"""SimulationDeviceDriver - Device Connect DeviceDriver adapter wrapping a strands-robots Simulation.

Exposes the Simulation's physics stepping, policy execution, and world
state as structured RPCs and events via Device Connect's DeviceDriver interface.
"""

import logging
from collections.abc import Mapping
from typing import Any

from device_connect_edge.drivers import (
    DeviceDriver,
    emit,
    get_rpc_source_device,
    on,
    periodic,
    rpc,
)
from device_connect_edge.types import DeviceIdentity, DeviceStatus

from strands_robots.device_connect._authz import authz_error, is_authorized_caller
from strands_robots.mesh.core import _reports_failure_to_stop
from strands_robots.mesh.security import is_safe_policy_provider
from strands_robots.teleop_mixin import _stop_reported_stopped

logger = logging.getLogger(__name__)


class SimulationDeviceDriver(DeviceDriver):
    """Device Connect device driver wrapping a strands-robots Simulation instance."""

    device_type = "strands_sim"

    def __init__(self, sim):
        super().__init__()
        self._sim = sim

    @property
    def identity(self) -> DeviceIdentity:
        """Static Device Connect identity for the wrapped simulation.

        Returns a :class:`~device_connect_edge.types.DeviceIdentity` reporting
        ``device_type="strands_sim"``, the ``strands-robots`` manufacturer, and
        the simulation's ``tool_name_str`` as the model (falling back to
        ``"simulation"``).
        """
        return DeviceIdentity(
            device_type="strands_sim",
            manufacturer="strands-robots",
            model=getattr(self._sim, "tool_name_str", "simulation"),
            description="Strands Robots MuJoCo simulation",
        )

    @property
    def status(self) -> DeviceStatus:
        """Live availability of the wrapped simulation.

        Returns a :class:`~device_connect_edge.types.DeviceStatus` that is
        ``"busy"`` (``busy_score`` 1.0) when any robot in the world is running a
        policy and ``"idle"`` (``busy_score`` 0.0) otherwise.
        """
        world = getattr(self._sim, "_world", None)
        is_busy = False
        if world:
            for robot in world.robots.values():
                if getattr(robot, "policy_running", False):
                    is_busy = True
                    break
        return DeviceStatus(
            availability="busy" if is_busy else "idle",
            busy_score=1.0 if is_busy else 0.0,
        )

    async def connect(self) -> None:
        """No-op - the Simulation manages its own MuJoCo state."""
        pass

    async def disconnect(self) -> None:
        """No-op - the Simulation manages its own cleanup."""
        pass

    # ── RPCs ──────────────────────────────────────────────────

    @rpc()
    async def execute(
        self,
        instruction: str,
        policy_provider: str = "mock",
        duration: float = 30.0,
        robot_name: str = "",
    ) -> dict[str, Any]:
        """Execute a policy on a simulated robot.

        Args:
            instruction: Natural language task instruction
            policy_provider: Policy backend (mock, lerobot_local, ...)
            duration: Maximum task duration in seconds
            robot_name: Target robot name (empty = first robot)
        """
        # Security hardening: authorize the calling device before mutating
        # simulation state.
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "execute")

        # Determine robot name
        name = robot_name
        if not name:
            world = getattr(self._sim, "_world", None)
            if world and world.robots:
                name = next(iter(world.robots))
            else:
                return {"status": "error", "reason": "no robots in simulation"}

        # Security hardening: restrict policy_provider to the vetted allowlist
        # so a caller cannot steer inference to an arbitrary network endpoint.
        if not is_safe_policy_provider(policy_provider):
            return {"status": "error", "reason": f"policy_provider not allowed: {policy_provider!r}"}

        print(f"[policy] Executing '{policy_provider}' on {name}: {instruction}", flush=True)
        return self._sim.start_policy(
            robot_name=name,
            policy_provider=policy_provider,
            instruction=instruction,
            duration=duration,
        )

    @rpc()
    async def stop(self) -> dict[str, Any]:
        """Stop every rollout in this simulation and report which ones halted.

        Routes each robot through the simulation's own ``stop_policy``, the verb
        that owns the question, and reads the verdict it returns. This used to
        lower ``policy_running`` itself and answer a fixed "All policies
        stopped", which made one sentence out of four different facts: a halted
        rollout, an idle simulation, a world already torn down, and -- because
        the loop had no guard -- a scene teardown racing it, which escaped as a
        ``RuntimeError`` past the RPC instead of an envelope. None of the four
        named a robot, so an operator could not tell what had been halted or
        cross-check it against ``list_policies_running``.

        The mesh fanout (:meth:`~strands_robots.mesh.Mesh._dispatch`, action
        ``stop``) is the other remote way into this same simulation and it
        already answers this way: per-robot ``stop_policy``, each answer graded
        through :func:`~strands_robots.mesh.core._reports_failure_to_stop`, the
        halted robots named. The rule has one owner and this is now its third
        reader rather than a surface that produced no answer to grade.

        A simulation with nothing to halt answers affirmatively-empty rather
        than erroring, for the reason the mesh branch states in full: nothing to
        stop makes "did not stop" wrong rather than conservative, and a false
        negative on the safety path is what trains an operator to ignore the
        warning.

        Returns:
            ``status="success"`` when no rollout refused, naming the halted ones
            in the text and listing them under ``stopped`` in the ``json``
            block. ``status="error"`` when a rollout refused or the world
            mutated under the loop, with the refusals under ``not_stopped`` and
            whatever did halt still under ``stopped``.
        """
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "stop")
        print("[policy] Stop command received - stopping every rollout", flush=True)
        halted: list[str] = []
        refused: dict[str, dict[str, Any]] = {}
        # Robots whose stop answered neither way. Tracked rather than folded
        # into "was not running", so the affirmative sentence below is only
        # spoken when something actually reported the fact.
        unreported: list[str] = []
        try:
            for name in self._stop_targets():
                answer = self._stop_one_rollout(name)
                if _reports_failure_to_stop(answer):
                    refused[name] = answer
                elif (in_flight := _reported_a_rollout_in_flight(answer)) is None:
                    unreported.append(name)
                elif in_flight:
                    halted.append(name)
        # Recovery path: catch broadly. A scene teardown racing this handler can
        # leave ``world.robots`` mid-mutation, and an RPC must answer rather
        # than raise past the dispatcher -- the same reason ``onEmergencyStop``
        # below and the mesh stop branch both guard their own stop loops.
        except Exception as exc:  # noqa: BLE001 - a stop must answer, not raise
            logger.error(
                "[safety] stop: the simulation changed under the stop loop after halting %s; "
                "rollouts may still be executing: %s",
                halted or "nothing",
                exc,
                exc_info=True,
            )
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"The simulation changed under the stop loop: {exc}. "
                            f"Halted {halted} before it did; anything else may still be executing."
                        )
                    },
                    {"json": {"stopped": halted, "not_stopped": [], "unreported": unreported, "error": str(exc)}},
                ],
            }
        if refused:
            names = sorted(refused)
            logger.error(
                "[safety] stop: stop_policy refused for %d rollout(s): %s; those policies may still be executing",
                len(names),
                names,
                exc_info=False,
            )
            return {
                "status": "error",
                "content": [
                    {"text": f"stop_policy refused for: {', '.join(names)}. Those policies may still be executing."},
                    {"json": {"stopped": halted, "not_stopped": names, "unreported": unreported, "results": refused}},
                ],
            }
        if halted:
            text = f"Halted {len(halted)} rollout(s): {', '.join(halted)}"
        elif unreported:
            text = f"Nothing reported a rollout in flight; {len(unreported)} stop(s) answered without a verdict"
        else:
            text = "No rollout was in flight"
        return {
            "status": "success",
            "content": [
                {"text": text},
                {"json": {"stopped": halted, "not_stopped": [], "unreported": unreported}},
            ],
        }

    def _stop_targets(self) -> list[str]:
        """Robots this stop asks about - every robot in the world, or none.

        The population is deliberately the whole world rather than the rollout
        registry: ``stop_policy`` is idempotent, and a blocking ``run_policy``
        raises ``policy_running`` without registering a Future, so a registry-
        only population would leave that rollout untouched. Which of these was
        actually in flight is the answer ``stop_policy`` gives, not a guess made
        here.
        """
        world = getattr(self._sim, "_world", None)
        if world is None:
            return []
        return list(world.robots)

    def _stop_one_rollout(self, robot_name: str) -> dict[str, Any]:
        """Stop one robot's rollout through the verb that owns the question.

        Falls back to the durable flag write for a backend that exposes no
        ``stop_policy``: :meth:`~strands_robots.simulation.base.SimEngine.start_policy`
        is a synchronous passthrough there, so no worker outlives the call and
        the flag is the whole answer. The fallback returns the same envelope
        shape so the caller above grades one thing.
        """
        stop_policy = getattr(self._sim, "stop_policy", None)
        if callable(stop_policy):
            return dict(stop_policy(robot_name=robot_name))
        robot = self._sim._world.robots[robot_name]
        return {
            "status": "success",
            "content": [{"json": {"robot": robot_name, "was_running": bool(robot.request_policy_stop())}}],
        }

    @rpc()
    async def getStatus(self) -> dict[str, Any]:
        """Get simulation state and running policies."""
        if hasattr(self._sim, "get_state"):
            return self._sim.get_state()
        return {"status": "idle"}

    @rpc()
    async def getFeatures(self) -> dict[str, Any]:
        """Get simulation features (joints, actuators, cameras)."""
        return self._sim.get_features()

    @rpc()
    async def step(self, n_steps: int = 1) -> dict[str, Any]:
        """Step simulation physics forward.

        Args:
            n_steps: Number of physics steps to take
        """
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "step")
        return self._sim.step(n_steps)

    @rpc()
    async def reset(self) -> dict[str, Any]:
        """Reset simulation to initial state."""
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "reset")
        return self._sim.reset()

    # ── Events ────────────────────────────────────────────────

    @emit()
    async def policyStarted(self, robot_name: str, instruction: str, policy_provider: str):
        """Emitted when a policy begins execution.

        Args:
            robot_name: The simulated robot running the policy
            instruction: The task instruction
            policy_provider: The policy backend used
        """
        pass

    @emit()
    async def policyComplete(self, robot_name: str, instruction: str, steps: int):
        """Emitted when a policy finishes.

        Args:
            robot_name: The simulated robot
            instruction: The task instruction
            steps: Total steps executed
        """
        pass

    @emit()
    async def emergencyStop(self, reason: str = ""):
        """Emitted when this device triggers an emergency stop.

        Args:
            reason: Why the emergency stop was triggered
        """
        pass

    @on(event_name="emergencyStop")
    async def onEmergencyStop(self, device_id: str, event_name: str, payload: dict[str, Any]) -> None:
        """React to emergencyStop from an authorized safety controller.

        Halts EVERY motion source the simulation has, not only its policies. A
        ``Simulation`` mixes in :class:`~strands_robots.teleop_mixin.TeleopMixin`,
        so a leader arm can be driving it from a thread the ``policy_running``
        flag says nothing about; stopping policies alone left that loop polling
        ``get_action()`` and applying the result through ``send_action`` after an
        operator's emergency stop, with the handler reporting the halt.

        Both stops are attempted even if one fails, and a source that did not
        stop is logged at CRITICAL naming it -- the accounting
        ``reachy_mini_driver.onEmergencyStop`` gives its two stop actions, and
        that ``robot_driver.onEmergencyStop`` gives the stop verdict it reads.
        A stop that arrives here rather than over the mesh is the same operator
        request, so it gets the same accounting.

        Security hardening: only act on emergency-stop events whose source is
        in the emergency-stop allowlist, so a spoofed event from an arbitrary
        device cannot interrupt operations.
        """
        if not is_authorized_caller(device_id, scope="estop"):
            logger.warning("Ignoring emergencyStop from unauthorized source %s", device_id)
            return
        logger.warning("Emergency stop received from %s - halting every motion source", device_id)
        # Attempt EVERY motion source even if one fails, and surface a failure
        # loudly rather than behind a false ack -- the same shape
        # ``reachy_mini_driver.onEmergencyStop`` uses for its two stop actions.
        failures: list[str] = []
        # Policies first: this is a flag write that cannot block, so the cheap
        # kill lands even if the bounded teleop join below outlasts its budget.
        try:
            world = getattr(self._sim, "_world", None)
            if world:
                for robot in world.robots.values():
                    robot.request_policy_stop()
        # Recovery path: catch broadly. A scene teardown racing this handler can
        # leave ``robots`` mid-mutation, and a safety handler must still attempt
        # the teleop stop below rather than crash out.
        except Exception as exc:  # noqa: BLE001 - attempt-every-source e-stop recovery
            failures.append(f"policies: {exc}")
        # Teleoperation is the simulation's OTHER motion source. A ``Simulation``
        # mixes in ``TeleopMixin``, so a leader arm can be driving it, and the
        # policy flag says nothing about that loop: it polls ``get_action()`` and
        # applies the result through ``send_action`` on its own thread. The
        # simulation's own ``cleanup`` stops it for exactly this reason, under
        # this same guard, so an emergency stop cannot leave it out.
        if getattr(self._sim, "_teleop_running", False) or getattr(self._sim, "_teleops", None):
            try:
                envelope = self._sim.stop_teleoperate()
                # Read ``stopped`` rather than ``status``: ``_teleop_stats``
                # derives the status from the session counters, so a session
                # whose frames errored reports "error" after a clean join.
                # ``_stop_reported_stopped`` owns that distinction and is the
                # right reader HERE, where the envelope really is a
                # ``stop_teleoperate`` one -- ``robot_driver`` deliberately does
                # not reuse it because it grades a ``stop_task`` envelope.
                if not _stop_reported_stopped(envelope):
                    reason = " ".join(block["text"] for block in envelope.get("content", []) if "text" in block)
                    failures.append(f"teleoperation: {reason}")
            except Exception as exc:  # noqa: BLE001 - attempt-every-source e-stop recovery
                failures.append(f"stop_teleoperate: {exc}")
        if failures:
            logger.critical(
                "Emergency stop from %s did NOT fully complete: %s",
                device_id,
                "; ".join(failures),
            )

    # ── Periodic state publishing ─────────────────────────────

    def _joint_positions(self, robot_name: str, robot: Any) -> dict[str, float]:
        """Per-joint positions for ``robot_name``, in radians.

        Read through the simulation's own ``get_observation`` surface rather
        than by indexing its state arrays here. That surface already answers
        this exact question, and it owns three facts a second reader has to
        reproduce and can silently get wrong:

        * a joint's position lives at its qpos ADDRESS, not at its joint id.
          The two coincide only while every joint is single-DoF: one floating
          base ahead of a chain shifts every later joint by six, so a humanoid
          reports its pelvis height and base quaternion under leg-joint names.
        * a free joint has no scalar position at all (its qpos is
          ``[xyz, quat]``), so it is excluded from the per-joint state rather
          than reported as a degenerate number.
        * the read is serialised against a concurrent physics step, so a
          published value is a whole number rather than a torn one.

        Images are skipped: this runs on the 10Hz publish loop, which wants
        joint state only.

        Args:
            robot_name: Registered name of the robot to read.
            robot: The world's record for that robot, read for its
                ``joint_names`` so only its own joints are published.

        Returns:
            ``{joint name: position}`` for every joint the observation reports
            a scalar for. Empty when the simulation exposes no observation
            surface.
        """
        get_observation = getattr(self._sim, "get_observation", None)
        if not callable(get_observation):
            return {}
        obs = get_observation(robot_name, skip_images=True)
        return {
            name: float(obs[name])
            for name in getattr(robot, "joint_names", [])
            if isinstance(obs.get(name), (int, float)) and not isinstance(obs.get(name), bool)
        }

    @periodic(interval=0.1, wait_for_completion=True)
    async def _publishState(self):
        """Publish simulation state at 10Hz."""
        world = getattr(self._sim, "_world", None)
        if not world:
            return
        running = {
            name: {"steps": r.policy_steps, "instruction": r.policy_instruction}
            for name, r in world.robots.items()
            if r.policy_running
        }
        if running:
            await self.stateUpdate(
                sim_time=world.sim_time,
                step_count=world.step_count,
                running_policies=running,
            )
            # Publish per-robot joint observations.
            robots = world.robots if isinstance(world.robots, dict) else {}
            for name, robot in robots.items():
                try:
                    joints = self._joint_positions(name, robot)
                    await self.observationUpdate(
                        robot_name=name,
                        sim_time=world.sim_time,
                        step_count=world.step_count,
                        joints=joints,
                    )
                except Exception as e:
                    logger.debug("observationUpdate skipped for %s: %s", name, e)

    @emit()
    async def stateUpdate(
        self, sim_time: float = 0.0, step_count: int = 0, running_policies: dict[str, Any] | None = None
    ) -> None:
        """Periodic simulation state update.

        Args:
            sim_time: Current simulation time
            step_count: Total physics steps
            running_policies: Dict of running policy info per robot
        """
        pass

    @emit()
    async def observationUpdate(
        self, robot_name: str = "", sim_time: float = 0.0, step_count: int = 0, joints: dict[str, float] | None = None
    ) -> None:
        """Periodic per-robot observation with joint positions.

        Args:
            robot_name: Name of the robot
            sim_time: Current simulation time
            step_count: Total physics steps
            joints: Dict of joint name -> position (radians)
        """
        pass


def _reported_a_rollout_in_flight(answer: Mapping[str, Any]) -> bool | None:
    """Whether a ``stop_policy`` answer says a rollout really was in flight.

    Reads the ``was_running`` key
    :meth:`~strands_robots.simulation.mujoco.simulation.MuJoCoSimEngine.stop_policy`
    puts in its ``json`` block. Tri-state on purpose, in the same conservative
    direction as :func:`~strands_robots.mesh.core._reports_failure_to_stop`: an
    envelope that reports the fact neither way is not read as either one.
    Counting silence as a halt names a robot the answer never mentioned;
    counting it as idle lets the caller state "no rollout was in flight" on no
    evidence. Both are the affirmative lie this verb exists to stop telling.

    Args:
        answer: One envelope returned by a stop verb.

    Returns:
        ``True`` or ``False`` as the answer reports it, or ``None`` when the
        answer carries no verdict at all.
    """
    for block in answer.get("content", []):
        payload = block.get("json")
        if isinstance(payload, dict) and "was_running" in payload:
            return bool(payload["was_running"])
    return None
