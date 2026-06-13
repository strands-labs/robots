"""Integration tests for Device Connect DeviceDriver adapters.

Requires Docker infrastructure running:
  - Zenoh router (:7447)
  - etcd (:2379)
  - device-registry (:8000)

Start with:
  cd device-connect/packages/device-connect-server
  docker compose -f infra/docker-compose-dev.yml up -d

Run with:
  MESSAGING_BACKEND=zenoh ZENOH_CONNECT=tcp/localhost:7447 \
    DEVICE_CONNECT_ALLOW_INSECURE=true python3 -m pytest tests/test_device_connect_integration.py -v
"""

import asyncio
import os
from unittest.mock import MagicMock

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("DEVICE_CONNECT_ALLOW_INSECURE"),
        reason="Requires Docker infrastructure (set DEVICE_CONNECT_ALLOW_INSECURE=true)",
    ),
]


def _make_mock_robot(tool_name="itest-robot"):
    """Create a mock Robot for integration testing."""
    from dataclasses import dataclass
    from enum import Enum

    class TaskStatus(Enum):
        IDLE = "idle"
        RUNNING = "running"

    @dataclass
    class TaskState:
        status: TaskStatus = TaskStatus.IDLE
        instruction: str = ""
        step_count: int = 0

    robot = MagicMock()
    robot.tool_name_str = tool_name
    robot._task_state = TaskState()
    robot.start_task.return_value = {"status": "success", "content": [{"text": "Task started"}]}
    robot.stop_task.return_value = {"status": "success", "content": [{"text": "Task stopped"}]}
    robot.get_task_status.return_value = {"status": "success", "content": [{"text": "Idle"}]}
    robot.get_features.return_value = {"status": "success", "content": [{"json": {}}]}
    robot.robot = MagicMock()
    robot.robot.get_observation.return_value = {"joint1": 0.5}
    return robot


def _make_mock_sim(tool_name="itest-sim"):
    """Create a mock Simulation for integration testing."""
    sim = MagicMock()
    sim.tool_name_str = tool_name

    robot_data = MagicMock()
    robot_data.policy_running = False
    robot_data.policy_steps = 0
    robot_data.policy_instruction = ""

    world = MagicMock()
    world.robots = {"arm1": robot_data}
    world.sim_time = 0.0
    world.step_count = 0
    sim._world = world

    sim.start_policy.return_value = {"status": "success", "content": [{"text": "Started"}]}
    sim.get_state.return_value = {"status": "success", "content": [{"text": "State"}]}
    sim.get_features.return_value = {"status": "success", "content": [{"json": {}}]}
    sim.step.return_value = {"status": "success", "content": [{"text": "Stepped"}]}
    sim.reset.return_value = {"status": "success", "content": [{"text": "Reset"}]}
    return sim


@pytest.fixture(autouse=True)
def device_connect_env():
    """Set environment for Device Connect messaging.

    Supports both Zenoh and NATS backends.  The backend is chosen by the
    MESSAGING_BACKEND env-var (default ``nats`` to match the standard
    docker-compose-itest.yml setup).
    """
    backend = os.getenv("MESSAGING_BACKEND", "nats")
    os.environ.setdefault("MESSAGING_BACKEND", backend)

    if backend == "zenoh":
        url = os.getenv("ZENOH_CONNECT", "tcp/localhost:7447")
        os.environ.setdefault("ZENOH_CONNECT", url)
        os.environ.setdefault("MESSAGING_URLS", url)
    else:
        url = os.getenv("NATS_URL", "nats://localhost:4222")
        os.environ.setdefault("NATS_URL", url)
        os.environ.setdefault("MESSAGING_URLS", url)

    os.environ.setdefault("DEVICE_CONNECT_ALLOW_INSECURE", "true")
    yield


class TestRobotDriverRegistration:
    """Test that RobotDeviceDriver registers and is discoverable."""

    async def test_robot_driver_registers(self):
        """Create RobotDeviceDriver + DeviceRuntime, verify device is discoverable."""
        from device_connect_edge import DeviceRuntime

        from strands_robots.device_connect.robot_driver import RobotDeviceDriver

        robot = _make_mock_robot()
        driver = RobotDeviceDriver(robot)
        runtime = DeviceRuntime(
            driver=driver,
            device_id="itest-robot-001",
            allow_insecure=True,
        )

        task = asyncio.create_task(runtime.run())
        try:
            # Wait for registration
            await asyncio.sleep(3)

            from device_connect_agent_tools.connection import connect, disconnect, get_connection

            await asyncio.to_thread(connect)
            try:
                conn = get_connection()
                devices = await asyncio.to_thread(conn.list_devices, device_type="strands_robot")
                device_ids = [d["device_id"] for d in devices]
                assert "itest-robot-001" in device_ids, f"Expected itest-robot-001 in {device_ids}"
            finally:
                await asyncio.to_thread(disconnect)
        finally:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass  # teardown: listener task was cancelled, ignore its outcome

    async def test_robot_execute_rpc(self):
        """Discover robot and invoke execute RPC."""
        from device_connect_edge import DeviceRuntime

        from strands_robots.device_connect.robot_driver import RobotDeviceDriver

        robot = _make_mock_robot()
        driver = RobotDeviceDriver(robot)
        runtime = DeviceRuntime(
            driver=driver,
            device_id="itest-robot-exec",
            allow_insecure=True,
        )

        task = asyncio.create_task(runtime.run())
        try:
            await asyncio.sleep(3)

            from device_connect_agent_tools.connection import connect, disconnect, get_connection

            await asyncio.to_thread(connect)
            try:
                conn = get_connection()
                result = await asyncio.to_thread(
                    conn.invoke,
                    "itest-robot-exec",
                    "execute",
                    {"instruction": "test move", "policy_provider": "mock", "duration": 5.0},
                )
                assert "result" in result, f"Expected result in {result}"
                robot.start_task.assert_called_once()
            finally:
                await asyncio.to_thread(disconnect)
        finally:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass  # teardown: listener task was cancelled, ignore its outcome

    async def test_robot_stop_rpc(self):
        """Invoke stop RPC on a registered robot."""
        from device_connect_edge import DeviceRuntime

        from strands_robots.device_connect.robot_driver import RobotDeviceDriver

        robot = _make_mock_robot()
        driver = RobotDeviceDriver(robot)
        runtime = DeviceRuntime(
            driver=driver,
            device_id="itest-robot-stop",
            allow_insecure=True,
        )

        task = asyncio.create_task(runtime.run())
        try:
            await asyncio.sleep(3)

            from device_connect_agent_tools.connection import connect, disconnect, get_connection

            await asyncio.to_thread(connect)
            try:
                conn = get_connection()
                result = await asyncio.to_thread(conn.invoke, "itest-robot-stop", "stop")
                assert "result" in result
                robot.stop_task.assert_called_once()
            finally:
                await asyncio.to_thread(disconnect)
        finally:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass  # teardown: listener task was cancelled, ignore its outcome


class TestSimDriverRegistration:
    """Test that SimulationDeviceDriver registers and is discoverable."""

    async def test_sim_driver_registers(self):
        """Create SimulationDeviceDriver + DeviceRuntime, verify device is discoverable."""
        from device_connect_edge import DeviceRuntime

        from strands_robots.device_connect.sim_driver import SimulationDeviceDriver

        sim = _make_mock_sim()
        driver = SimulationDeviceDriver(sim)
        runtime = DeviceRuntime(
            driver=driver,
            device_id="itest-sim-001",
            allow_insecure=True,
        )

        task = asyncio.create_task(runtime.run())
        try:
            await asyncio.sleep(3)

            from device_connect_agent_tools.connection import connect, disconnect, get_connection

            await asyncio.to_thread(connect)
            try:
                conn = get_connection()
                devices = await asyncio.to_thread(conn.list_devices, device_type="strands_sim")
                device_ids = [d["device_id"] for d in devices]
                assert "itest-sim-001" in device_ids
            finally:
                await asyncio.to_thread(disconnect)
        finally:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass  # teardown: listener task was cancelled, ignore its outcome

    async def test_sim_step_rpc(self):
        """Invoke step RPC on a registered simulation."""
        from device_connect_edge import DeviceRuntime

        from strands_robots.device_connect.sim_driver import SimulationDeviceDriver

        sim = _make_mock_sim()
        driver = SimulationDeviceDriver(sim)
        runtime = DeviceRuntime(
            driver=driver,
            device_id="itest-sim-step",
            allow_insecure=True,
        )

        task = asyncio.create_task(runtime.run())
        try:
            await asyncio.sleep(3)

            from device_connect_agent_tools.connection import connect, disconnect, get_connection

            await asyncio.to_thread(connect)
            try:
                conn = get_connection()
                result = await asyncio.to_thread(conn.invoke, "itest-sim-step", "step", {"n_steps": 10})
                assert "result" in result
                sim.step.assert_called_once_with(10)
            finally:
                await asyncio.to_thread(disconnect)
        finally:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass  # teardown: listener task was cancelled, ignore its outcome


class TestMultipleDevices:
    """Test multiple devices registered simultaneously."""

    async def test_multiple_devices_discoverable(self):
        """Register 3 devices and verify all are discoverable."""
        from device_connect_edge import DeviceRuntime

        from strands_robots.device_connect.robot_driver import RobotDeviceDriver
        from strands_robots.device_connect.sim_driver import SimulationDeviceDriver

        robot1 = _make_mock_robot("robot-a")
        robot2 = _make_mock_robot("robot-b")
        sim1 = _make_mock_sim("sim-c")

        runtimes = []
        tasks = []
        for device_id, driver_cls, instance in [
            ("itest-multi-a", RobotDeviceDriver, robot1),
            ("itest-multi-b", RobotDeviceDriver, robot2),
            ("itest-multi-c", SimulationDeviceDriver, sim1),
        ]:
            driver = driver_cls(instance)
            runtime = DeviceRuntime(
                driver=driver,
                device_id=device_id,
                allow_insecure=True,
            )
            runtimes.append(runtime)
            tasks.append(asyncio.create_task(runtime.run()))

        try:
            await asyncio.sleep(5)

            from device_connect_agent_tools.connection import connect, disconnect, get_connection

            await asyncio.to_thread(connect)
            try:
                conn = get_connection()
                devices = await asyncio.to_thread(conn.list_devices)
                device_ids = {d["device_id"] for d in devices}
                assert "itest-multi-a" in device_ids
                assert "itest-multi-b" in device_ids
                assert "itest-multi-c" in device_ids
            finally:
                await asyncio.to_thread(disconnect)
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


class TestInitDeviceConnectE2E:
    """End-to-end test of init_device_connect()."""

    async def test_init_device_connect_e2e(self):
        """init_device_connect() -> device registers -> discoverable -> invocable."""
        from strands_robots.device_connect import init_device_connect

        robot = _make_mock_robot("e2e-robot")
        runtime = await init_device_connect(robot, peer_id="itest-e2e-robot")

        try:
            # Wait for registration
            await asyncio.sleep(3)

            from device_connect_agent_tools.connection import connect, disconnect, get_connection

            await asyncio.to_thread(connect)
            try:
                conn = get_connection()

                # Discoverable
                devices = await asyncio.to_thread(conn.list_devices, device_type="strands_robot")
                device_ids = [d["device_id"] for d in devices]
                assert "itest-e2e-robot" in device_ids

                # Invocable
                result = await asyncio.to_thread(conn.invoke, "itest-e2e-robot", "getStatus")
                assert "result" in result
            finally:
                await asyncio.to_thread(disconnect)
        finally:
            await runtime.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
