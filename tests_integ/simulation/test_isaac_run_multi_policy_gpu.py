"""GPU integration tests for ``run_multi_policy`` on the Isaac backend.

Real-Isaac (SimulationApp/Kit runtime) coverage for the synchronized
multi-robot control loop (#2158) and its merged-frame recording parity
(#2159), closing the test acceptance of the #2122 parity work (#2160): the
unit suites (``tests/simulation/isaac/test_run_multi_policy_no_recording.py``
and ``test_run_multi_policy_recording.py``) drive the REAL loop against stub
worlds/articulations; these tests drive the same entry point against a real
Kit runtime with two Franka USD articulations in one stage - real physics,
real ``run_on_main`` marshalling, real ``DatasetRecorder`` round-trips.

Covered here:

* a synchronized 2-robot rollout (no recording) completes with per-robot
  step counts and actually moves both robots' joints (real physics);
* a merged recording rollout writes ONE frame per timestep with both robots'
  namespaced ``<robot>__<joint>`` columns non-zero in every frame - the
  real-runtime mirror of the unit-level parity pin (itself mirroring
  ``tests/simulation/mujoco/test_recording_paths.py::test_b4_synchronized_multi_robot_recording``);
* the rollout driven off the Kit main thread (the way an agent tool drives
  it - the #1896 deadlock shape) completes within a bounded time while the
  owning thread runs ``run_pump_forever``;
* ``reset_between=True`` returns the structured #1895 refusal on the real
  backend too, without advancing physics.

Requirements match ``test_isaac_gpu.py``: NVIDIA GPU + CUDA, Isaac Sim 6.0+
installed out-of-band, ``pip install 'strands-robots[sim-isaac,lerobot]'``,
and ``STRANDS_GPU_TEST=1``. Kit startup dominates the wall time, so ONE
module-scoped simulation (one ``SimulationApp`` boot, one stage,
``num_envs=1``, two articulations) is shared across all tests and destroyed
at module teardown. Run with::

    STRANDS_GPU_TEST=1 hatch run test-integ \\
        tests_integ/simulation/test_isaac_run_multi_policy_gpu.py -m gpu -v
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")
pytest.importorskip("lerobot")

_GPU_ENABLED = os.environ.get("STRANDS_GPU_TEST", "0") == "1"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _GPU_ENABLED,
        reason="Requires an NVIDIA GPU + Isaac Sim 6.0. Set STRANDS_GPU_TEST=1 to enable.",
    ),
]

_ROBOTS = ("left", "right")


def _skip_if_isaac_unavailable() -> None:
    from strands_robots.simulation.isaac import IsaacSimulation

    available, reason = IsaacSimulation.is_available()
    if not available:
        pytest.skip(f"Isaac Sim not available: {reason}")


def _assets_root_path() -> str:
    """Resolve the Isaac Sim bundled-assets root (modern then legacy path)."""
    try:
        from isaacsim.storage.native import get_assets_root_path  # type: ignore[import-not-found]
    except ImportError:
        from omni.isaac.nucleus import get_assets_root_path  # type: ignore[import-not-found]

    assets_root = get_assets_root_path()
    assert assets_root, "get_assets_root_path() returned empty"
    return assets_root


# The bundled Franka USD moved between asset releases; probe the 6.0 layout
# first, then the legacy 4.x one (same resolution examples/libero/run.py
# ships, see strands-labs/robots-sim#110).
_FRANKA_USD_SUBPATHS = (
    "Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",  # Isaac Sim 6.0+
    "Isaac/Robots/Franka/franka.usd",  # Isaac Sim 4.x and earlier
)


def _franka_usd_path(assets_root: str) -> str:
    """Pick the Franka USD candidate that exists under ``assets_root``.

    HEAD-probes HTTP candidates and stat-probes filesystem ones; when no
    probe is conclusive (e.g. an ``omniverse://`` Nucleus root), falls back
    to the first (6.0) candidate and lets ``add_robot`` report the miss.
    """
    import urllib.request

    for sub in _FRANKA_USD_SUBPATHS:
        candidate = f"{assets_root}/{sub}"
        if candidate.startswith(("http://", "https://")):
            request = urllib.request.Request(candidate, method="HEAD")  # noqa: S310 - https asset probe
            try:
                with urllib.request.urlopen(request, timeout=30):  # noqa: S310 - https asset probe
                    return candidate
            except OSError:
                continue
        elif os.path.exists(candidate):
            return candidate
    return f"{assets_root}/{_FRANKA_USD_SUBPATHS[0]}"


def _payload(result: dict[str, Any]) -> dict[str, Any]:
    """Return the ``json`` content block of a success envelope."""
    for block in result.get("content", []):
        if "json" in block:
            return block["json"]
    raise AssertionError(f"no json content block in {result}")


def _joint_state(sim: Any, robot_name: str) -> dict[str, float]:
    """Flat float joint observation for one robot (no camera readback)."""
    obs = sim.get_observation(robot_name, skip_images=True)
    state = {k: v for k, v in obs.items() if isinstance(v, float)}
    assert state, f"{robot_name}: observation carries no joint state: {sorted(obs)}"
    return state


@pytest.fixture(scope="module")
def sim_two_frankas():
    """ONE stage (num_envs=1) holding TWO Franka articulations, shared module-wide.

    Kit startup dominates (the suites' documented reason to share the app),
    so all tests below drive this one simulation; teardown destroys the Kit
    app after the module. Proprio-only (headless, no cameras): MockPolicy
    declares ``requires_images=False`` and the recording test pins the
    merged proprio columns, matching the unit-level parity suite.
    """
    from strands_robots.simulation.isaac import IsaacConfig, IsaacSimulation

    _skip_if_isaac_unavailable()

    sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=True))
    try:
        r = sim.create_world()
        assert r["status"] == "success", f"create_world: {r}"

        # Resolved AFTER create_world(): on the pip Isaac Sim wheels the
        # asset-root helper lives in a Kit extension that is only importable
        # once SimulationApp has booted.
        usd_path = _franka_usd_path(_assets_root_path())
        for name, position in (("left", [0.0, 0.0, 0.0]), ("right", [0.0, 1.2, 0.0])):
            r = sim.add_robot(name, usd_path=usd_path, position=position)
            assert r["status"] == "success", f"add_robot {name}: {r}"

        sim.reset()
        sim.step(2)
        yield sim
    finally:
        sim.destroy()


def test_synchronized_two_robot_rollout_moves_real_physics(sim_two_frankas):
    """A recorder-free 2-robot rollout completes with per-robot step counts,
    and BOTH robots' joint state actually moved - real physics tracking the
    MockPolicy sinusoid targets, not a stub articulation echoing zeros.
    """
    from strands_robots.policies.mock import MockPolicy

    sim = sim_two_frankas
    n_steps = 20
    before = {name: _joint_state(sim, name) for name in _ROBOTS}

    r = sim.run_multi_policy(
        policies={name: MockPolicy() for name in _ROBOTS},
        instructions="wave",
        n_steps=n_steps,
        control_frequency=30.0,
    )
    assert r["status"] == "success", r
    payload = _payload(r)
    assert payload["steps"] == n_steps
    assert payload["per_robot_steps"] == {name: n_steps for name in _ROBOTS}

    # Real physics: each robot's articulation moved measurably toward the
    # sinusoid targets over the rollout. A stubbed/unstepped articulation
    # (or a loop that never applied the second robot's targets) leaves the
    # post-rollout read identical to the pre-rollout one.
    for name in _ROBOTS:
        after = _joint_state(sim, name)
        deltas = [abs(after[k] - before[name][k]) for k in before[name] if k in after]
        assert deltas, f"{name}: post-rollout observation lost its joint keys"
        assert max(deltas) > 1e-3, f"{name}: no joint moved during the rollout (max delta {max(deltas):.2e})"


def test_merged_recording_round_trip_records_one_frame_per_timestep(sim_two_frankas, tmp_path):
    """start_recording -> synchronized 2-robot rollout -> save_episode -> stop
    -> reopen from disk: ONE merged frame per timestep, both robots'
    namespaced ``<robot>__<joint>`` columns present, and both robots' action
    halves non-zero in EVERY frame (the B4 mirror on the real runtime -
    interleaved single-robot frames would leave the other half zero).
    """
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from strands_robots.dataset_recorder import read_dataset_episode_indices
    from strands_robots.policies.mock import MockPolicy

    sim = sim_two_frankas
    root = str(tmp_path / "dataset")
    n_steps = 8

    r = sim.start_recording(repo_id="local/isaac_multi_gpu", root=root, fps=10, overwrite=True)
    assert r["status"] == "success", f"start_recording: {r}"

    r = sim.run_multi_policy(
        policies={name: MockPolicy() for name in _ROBOTS},
        instructions="handover",
        n_steps=n_steps,
        control_frequency=10.0,
    )
    assert r["status"] == "success", r
    assert "(recorded)" in r["content"][0]["text"]

    r = sim.save_episode()
    assert r["status"] == "success", f"save_episode: {r}"
    r = sim.stop_recording()
    assert r["status"] == "success", f"stop_recording: {r}"

    r = sim.verify_dataset_episodes(1)
    assert r["status"] == "success", f"verify_dataset_episodes: {r}"

    # Round-trip: reopen the on-disk dataset and assert its truth.
    info = read_dataset_episode_indices(root)
    assert info["total_episodes"] == 1, info
    assert info["total_frames"] == n_steps, info

    ds = LeRobotDataset(repo_id="local/isaac_multi_gpu", root=root)
    assert len(ds) == n_steps, "one merged frame per timestep - no interleaving, no doubling"

    # Merged schema: every robot's action keys, namespaced ``robot__key``.
    af = ds.features["action"]
    action_names = af["names"] if isinstance(af, dict) else getattr(af, "names", None)
    assert action_names is not None
    for name in _ROBOTS:
        expected = [f"{name}__{key}" for key in sim.robot_action_keys(name)]
        assert expected, f"{name}: robot_action_keys returned nothing"
        assert [n for n in action_names if n.startswith(f"{name}__")] == expected

    # State columns are merged and namespaced the same way.
    sf = ds.features["observation.state"]
    state_names = sf["names"] if isinstance(sf, dict) else getattr(sf, "names", None)
    assert state_names is not None
    for name in _ROBOTS:
        joints = sim.robot_joint_names(name)
        assert joints, f"{name}: robot_joint_names returned nothing"
        assert [n for n in state_names if n.startswith(f"{name}__")] == [f"{name}__{j}" for j in joints]

    # Both robots' action halves are non-zero in EVERY frame.
    left_idx = [i for i, n in enumerate(action_names) if n.startswith("left__")]
    right_idx = [i for i, n in enumerate(action_names) if n.startswith("right__")]
    for i in range(len(ds)):
        action = np.asarray(ds[i]["action"])
        left_sum = float(np.abs(action[left_idx]).sum())
        right_sum = float(np.abs(action[right_idx]).sum())
        assert left_sum > 1e-6 and right_sum > 1e-6, (
            f"frame {i}: a robot's action half is all-zero (left={left_sum:.2e}, "
            f"right={right_sum:.2e}) - the frames are interleaved, not merged"
        )


def test_rollout_off_the_kit_main_thread_completes(sim_two_frankas):
    """The rollout driven off the Kit main thread - the way an agent tool
    drives it (#1896's deadlock shape) - completes within a bounded time
    while the owning thread runs ``run_pump_forever``.

    A watchdog releases the pump after a hard timeout so a regression to the
    pre-#1896 shape (the observe/apply hops blocking forever on a pump the
    worker cannot run) FAILS the test instead of wedging the whole CI run.
    """
    from strands_robots.policies.mock import MockPolicy

    sim = sim_two_frankas
    n_steps = 6
    stop = threading.Event()
    box: dict[str, Any] = {}

    def _worker() -> None:
        try:
            # run_multi_policy refuses a worker-thread call while no pump is
            # running (the structured #1896 refusal); wait for the main
            # thread to engage run_pump_forever before entering the loop.
            deadline = time.time() + 30.0
            while not sim._pump_running and time.time() < deadline:
                time.sleep(0.01)
            box["result"] = sim.run_multi_policy(
                policies={name: MockPolicy() for name in _ROBOTS},
                instructions="wave",
                n_steps=n_steps,
                control_frequency=30.0,
            )
        finally:
            stop.set()

    watchdog = threading.Timer(120.0, stop.set)
    watchdog.start()
    # daemon=True: on the deadlock regression this test exists to catch, the
    # worker is parked forever inside the marshal hop; a non-daemon thread
    # would then wedge interpreter shutdown after the watchdog fails the test.
    worker = threading.Thread(target=_worker, name="agent-worker", daemon=True)
    worker.start()
    try:
        sim.run_pump_forever(stop_event=stop)
        worker.join(timeout=10.0)
        assert not worker.is_alive(), (
            "run_multi_policy did not return within the watchdog window when "
            "driven off the Kit main thread - the #1896 deadlock shape"
        )
        result = box.get("result")
        assert result is not None and result["status"] == "success", f"worker result: {result}"
        assert _payload(result)["per_robot_steps"] == {name: n_steps for name in _ROBOTS}
    finally:
        watchdog.cancel()
        stop.set()


def test_reset_between_returns_the_1895_refusal(sim_two_frankas):
    """``reset_between=True`` is refused with the structured #1895 error on
    the real backend too - and refuses BEFORE advancing physics, so the
    articulation handles a mid-run ``reset()`` would tear down stay live.
    """
    from strands_robots.policies.mock import MockPolicy

    sim = sim_two_frankas
    steps_before = _payload(sim.get_state())["step_count"]

    r = sim.run_multi_policy(
        policies={name: MockPolicy() for name in _ROBOTS},
        n_steps=4,
        control_frequency=30.0,
        reset_between=True,
    )
    assert r["status"] == "error", r
    msg = r["content"][0]["text"]
    assert "#1895" in msg
    assert "reset_between" in msg

    # Refused up front: no physics advanced, and the robots remain observable.
    assert _payload(sim.get_state())["step_count"] == steps_before
    for name in _ROBOTS:
        assert _joint_state(sim, name)
