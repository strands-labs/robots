"""Integration test: drive a MuJoCo rollout through a remote PolicyServer.

A :class:`PolicyServer` wrapping a :class:`MockPolicy` runs in one thread; the
simulation resolves ``policy_provider="ws://..."`` to a :class:`RemotePolicy`
and streams observations to it, executing the returned chunks. This exercises
the full client/server path end to end (the same path a real edge robot +
remote GPU would take), just with a cheap policy so it stays in the fast suite.
"""

import pytest

from strands_robots.inference import PolicyServer
from strands_robots.policies.mock import MockPolicy

pytest.importorskip("mujoco", reason="sim rollout requires mujoco - pip install 'strands-robots[sim-mujoco]'")

import strands_robots as sr  # noqa: E402


def test_remote_policy_drives_mujoco_rollout():
    server = PolicyServer(policy=MockPolicy(), host="127.0.0.1", port=0).start()
    sim = sr.Robot("so101", mode="sim")
    try:
        result = sim.run_policy(
            policy_provider=f"ws://127.0.0.1:{server.port}",
            instruction="wave the arm",
            n_steps=40,
            control_frequency=30.0,
        )
        assert result["status"] == "success", result
        payload = next(item["json"] for item in result["content"] if "json" in item)
        assert payload["policy"] == "RemotePolicy"
        assert payload["n_steps"] == 40
        assert payload["action_errors"] == 0
        assert payload["stopped_early"] is False
    finally:
        sim.cleanup()
        server.stop()
