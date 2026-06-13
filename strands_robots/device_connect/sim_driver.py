"""SimulationDeviceDriver — Device Connect DeviceDriver adapter wrapping a strands-robots Simulation.

Exposes the Simulation's physics stepping, policy execution, and world
state as structured RPCs and events via Device Connect's DeviceDriver interface.
"""

import logging

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
from strands_robots.mesh.security import is_safe_policy_provider

logger = logging.getLogger(__name__)


class SimulationDeviceDriver(DeviceDriver):
    """Device Connect device driver wrapping a strands-robots Simulation instance."""

    device_type = "strands_sim"

    def __init__(self, sim):
        super().__init__()
        self._sim = sim

    @property
    def identity(self) -> DeviceIdentity:
        return DeviceIdentity(
            device_type="strands_sim",
            manufacturer="strands-robots",
            model=getattr(self._sim, "tool_name_str", "simulation"),
            description="Strands Robots MuJoCo simulation",
        )

    @property
    def status(self) -> DeviceStatus:
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
        """No-op — the Simulation manages its own MuJoCo state."""
        pass

    async def disconnect(self) -> None:
        """No-op — the Simulation manages its own cleanup."""
        pass

    # ── RPCs ──────────────────────────────────────────────────

    @rpc()
    async def execute(
        self,
        instruction: str,
        policy_provider: str = "mock",
        duration: float = 30.0,
        robot_name: str = "",
    ) -> dict:
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

        print(f"▶ Executing policy '{policy_provider}' on {name}: {instruction}", flush=True)
        return self._sim.start_policy(
            robot_name=name,
            policy_provider=policy_provider,
            instruction=instruction,
            duration=duration,
        )

    @rpc()
    async def stop(self) -> dict:
        """Stop all running policies."""
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "stop")
        print("⏹ Stop command received — stopping all policies", flush=True)
        world = getattr(self._sim, "_world", None)
        if world:
            for robot in world.robots.values():
                robot.policy_running = False
        return {"status": "success", "content": [{"text": "All policies stopped"}]}

    @rpc()
    async def getStatus(self) -> dict:
        """Get simulation state and running policies."""
        if hasattr(self._sim, "get_state"):
            return self._sim.get_state()
        return {"status": "idle"}

    @rpc()
    async def getFeatures(self) -> dict:
        """Get simulation features (joints, actuators, cameras)."""
        return self._sim.get_features()

    @rpc()
    async def step(self, n_steps: int = 1) -> dict:
        """Step simulation physics forward.

        Args:
            n_steps: Number of physics steps to take
        """
        caller = get_rpc_source_device()
        if not is_authorized_caller(caller, scope="rpc"):
            return authz_error(caller, "step")
        return self._sim.step(n_steps)

    @rpc()
    async def reset(self) -> dict:
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
    async def onEmergencyStop(self, device_id: str, event_name: str, payload: dict):
        """React to emergencyStop from an authorized safety controller.

        Security hardening: only act on emergency-stop events whose source is
        in the emergency-stop allowlist, so a spoofed event from an arbitrary
        device cannot interrupt operations.
        """
        if not is_authorized_caller(device_id, scope="estop"):
            logger.warning("Ignoring emergencyStop from unauthorized source %s", device_id)
            return
        print(f"🛑 Emergency stop received from {device_id} — stopping all policies", flush=True)
        world = getattr(self._sim, "_world", None)
        if world:
            for robot in world.robots.values():
                robot.policy_running = False

    # ── Periodic state publishing ─────────────────────────────

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
            # Publish per-robot joint observations from MuJoCo state
            data = getattr(world, "_data", None)
            robots = world.robots if isinstance(world.robots, dict) else {}
            for name, robot in robots.items():
                try:
                    joint_names = getattr(robot, "joint_names", [])
                    joint_ids = getattr(robot, "joint_ids", [])
                    joints = {}
                    if data is not None and joint_names and joint_ids:
                        for jname, jid in zip(joint_names, joint_ids):
                            joints[jname] = float(data.qpos[jid])
                    await self.observationUpdate(
                        robot_name=name,
                        sim_time=world.sim_time,
                        step_count=world.step_count,
                        joints=joints,
                    )
                except Exception as e:
                    logger.debug("observationUpdate skipped for %s: %s", name, e)

    @emit()
    async def stateUpdate(self, sim_time: float = 0.0, step_count: int = 0, running_policies: dict | None = None):
        """Periodic simulation state update.

        Args:
            sim_time: Current simulation time
            step_count: Total physics steps
            running_policies: Dict of running policy info per robot
        """
        pass

    @emit()
    async def observationUpdate(
        self, robot_name: str = "", sim_time: float = 0.0, step_count: int = 0, joints: dict | None = None
    ):
        """Periodic per-robot observation with joint positions.

        Args:
            robot_name: Name of the robot
            sim_time: Current simulation time
            step_count: Total physics steps
            joints: Dict of joint name -> position (radians)
        """
        pass
