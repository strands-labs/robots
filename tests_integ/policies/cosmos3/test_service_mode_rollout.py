"""Spawned-server MuJoCo rollout integration test for Cosmos 3 service mode.

The sibling file (tests_integ/policies/cosmos3/test_service_mode_live.py)
consumes a **pre-running** RoboLab policy server, so a non-GPU box can point
at a GPU host. This file is the other half of that split - the cosmos3
analogue of tests_integ/groot/test_groot_integration.py: a module-scoped
fixture **spawns** the Cosmos Framework RoboLab policy server itself, so a
single GPU box runs the whole thing with one command, and a short MuJoCo
Panda rollout is driven closed-loop by real server actions.

Enable with (needs a CUDA GPU, the cosmos_framework package, and the model
weights):

    COSMOS3_SPAWN_SERVER=1 \
    hatch run test-integ tests_integ/policies/cosmos3/test_service_mode_rollout.py -v

Environment knobs:

* ``COSMOS3_MODEL`` - checkpoint the spawned server loads
  (default ``nvidia/Cosmos3-Nano-Policy-DROID``).
* ``COSMOS3_SERVER_PORT`` - WebSocket port (default 8000). The port must be
  **free**: this file spawns its own server, so the fixture refuses an
  occupied port before spawning (a pre-running server - e.g. the one
  tests_integ/policies/cosmos3/test_service_mode_live.py consumes, which
  reads the same variable and default - would otherwise answer the readiness
  probe while the spawned child dies of EADDRINUSE unobserved).
* ``COSMOS3_SERVER_TIMEOUT`` - readiness deadline in seconds (default 600;
  a cold checkpoint download + model load can take minutes).

The launch command mirrors the quickstart documented in
:mod:`strands_robots.policies.cosmos3` (``--checkpoint-path`` + ``--port``
are the flags this repo documents for
``cosmos_framework.scripts.action_policy_server_robolab``).
"""

from __future__ import annotations

import importlib.util
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

SPAWN = os.environ.get("COSMOS3_SPAWN_SERVER", "").lower() in ("1", "true", "yes")
MODEL = os.environ.get("COSMOS3_MODEL", "nvidia/Cosmos3-Nano-Policy-DROID")
HOST = "127.0.0.1"
PORT = int(os.environ.get("COSMOS3_SERVER_PORT", "8000"))
# Model load can take minutes (and a cold run downloads the checkpoint), so
# the readiness deadline is generous and env-overridable.
READY_TIMEOUT_S = float(os.environ.get("COSMOS3_SERVER_TIMEOUT", "600"))

# Rollout shape: enough chunks to prove the connection survives repeated
# inference (>= 2), small enough to stay cheap on a single GPU.
N_CHUNKS = 3
SUBSTEPS_PER_ACTION = 10

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not SPAWN,
        reason="Spawns a GPU-holding Cosmos 3 RoboLab policy server. Set COSMOS3_SPAWN_SERVER=1 to enable.",
    ),
    # The default per-test integ timeout (300 s) is shorter than the server
    # readiness deadline; budget for spawn + rollout together.
    pytest.mark.timeout(READY_TIMEOUT_S + 300),
]

# Skip cleanly if optional deps are missing.
pytest.importorskip("websockets")
pytest.importorskip("msgpack")
if importlib.util.find_spec("cosmos_framework") is None:
    pytest.skip(
        "cosmos_framework is not installed (the RoboLab policy server is an "
        "external package, not a dependency of this repo).",
        allow_module_level=True,
    )

from strands_robots.policies import create_policy  # noqa: E402
from strands_robots.policies.cosmos3 import Cosmos3WebsocketClient  # noqa: E402

_PANDA_ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
_PANDA_GRIPPER_JOINT = "finger_joint1"


def _log_tail(log_path: Path, max_bytes: int = 4000) -> str:
    """Last ``max_bytes`` of the spawned server's captured output."""
    try:
        data = log_path.read_bytes()
    except OSError:
        return "<no server log captured>"
    return data[-max_bytes:].decode("utf-8", errors="replace")


def _wait_until_serving(proc: subprocess.Popen, log_path: Path) -> None:
    """Block until the spawned server answers the client handshake, or fail.

    Readiness is the real thing the tests need - a completed WebSocket
    connect + msgpack metadata handshake through the shipped client - not a
    bare TCP accept. The deadline is measured on ``time.monotonic()`` (a wait
    budget must survive a wall-clock step). A spawn that exits before ready
    fails with the server's captured output tail, so a missing checkpoint or
    a bad flag is diagnosable from the pytest output rather than reading as a
    bare timeout.
    """
    import websockets.exceptions as _wse

    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(
                f"Cosmos 3 policy server exited (returncode={proc.returncode}) "
                f"before serving. Server output tail:\n{_log_tail(log_path)}"
            )
        try:
            # ConnectionError (what the client wraps a refused connect into)
            # is an OSError subclass; InvalidHandshake/InvalidStatus cover a
            # server whose HTTP layer is up before the WebSocket route is.
            Cosmos3WebsocketClient(host=HOST, port=PORT).get_server_metadata()
            return
        except (OSError, _wse.WebSocketException):
            time.sleep(2.0)
    pytest.fail(
        f"Cosmos 3 policy server not ready within {READY_TIMEOUT_S:.0f}s "
        f"(COSMOS3_SERVER_TIMEOUT). Server output tail:\n{_log_tail(log_path)}"
    )


def _fail_unless_port_is_free() -> None:
    """Refuse to spawn onto a ``HOST:PORT`` something else already serves.

    ``_wait_until_serving`` probes the port, not the child: a foreign server
    already bound there answers the readiness probe, the fixture yields, and
    the child this fixture spawned dies of EADDRINUSE unobserved - after
    which the teardown's leak assert passes *because* the child crashed. The
    collision is reachable, not hypothetical: the sibling
    tests_integ/policies/cosmos3/test_service_mode_live.py points at a
    pre-running server through the same ``COSMOS3_SERVER_PORT`` with the
    same default, on the same pinned ``127.0.0.1``. Binding first turns the
    collision into a named refusal. ``SO_REUSEADDR`` matches the posture of
    the server the child will run, so a TIME_WAIT remnant from a previous
    run does not refuse a port the child could in fact bind.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((HOST, PORT))
    except OSError as exc:
        pytest.fail(
            f"{HOST}:{PORT} is already in use ({exc}), so a server this fixture "
            f"did not spawn would answer the readiness probe while the spawned "
            f"child dies of EADDRINUSE unobserved. Stop the other server (the "
            f"pre-running one test_service_mode_live.py consumes shares this "
            f"port's variable and default) or set COSMOS3_SERVER_PORT to a "
            f"free port."
        )
    finally:
        probe.close()


def _terminate(proc: subprocess.Popen) -> None:
    """SIGTERM the server's process group; SIGKILL if it does not exit."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            return
        proc.wait(timeout=15)


@pytest.fixture(scope="module")
def cosmos3_server(tmp_path_factory):
    """Spawn the RoboLab policy server for the whole module.

    Launched argv-style (no shell) with the flags this repo documents for
    ``action_policy_server_robolab`` (``--checkpoint-path`` + ``--port``);
    output goes to a file rather than a pipe so a chatty model load cannot
    fill the pipe buffer and stall the server, and so the failure paths can
    report the log tail while the process is still alive. Teardown runs even
    on test failure and must leave no orphan GPU-holding process - the
    finalizer itself asserts the process is gone.
    """
    # Before spawning: the readiness probe below checks the port, not the
    # child, so a port already held by a foreign server must be a named
    # refusal here rather than a false ready.
    _fail_unless_port_is_free()
    log_path = tmp_path_factory.mktemp("cosmos3-server") / "server.log"
    cmd = [
        sys.executable,
        "-m",
        "cosmos_framework.scripts.action_policy_server_robolab",
        "--checkpoint-path",
        MODEL,
        "--port",
        str(PORT),
    ]
    with open(log_path, "wb") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_until_serving(proc, log_path)
            yield {"host": HOST, "port": PORT, "process": proc}
        finally:
            _terminate(proc)
            # Leaking a GPU-holding server is silent damage: the assert lives
            # in the finalizer so a leak fails the run rather than outliving it.
            assert proc.poll() is not None, "spawned Cosmos 3 server outlived fixture teardown"


@pytest.fixture(scope="module")
def policy(cosmos3_server):
    """One policy against the spawned server, with the built-in Panda mapping.

    ``robot="panda"`` applies the DROID-layout -> Panda actuator mapping
    (joint_0..joint_6 -> joint1..joint7, gripper -> finger_joint1), so the
    per-step dicts are keyed by real MuJoCo Panda actuator-target names and
    the mapping documented in strands_robots/policies/cosmos3/policy.py is
    exercised end to end.
    """
    p = create_policy(
        "cosmos3",
        embodiment="droid",
        host=cosmos3_server["host"],
        port=cosmos3_server["port"],
        robot="panda",
    )
    p.set_robot_state_keys([*_PANDA_ARM_JOINTS, _PANDA_GRIPPER_JOINT])
    return p


def _mujoco_observation(mujoco, model, data, rng) -> dict:
    """Flat DROID observation from the live MuJoCo state.

    The three RoboArena camera views are synthesized (random frames - the
    server conditions on them but this test's assertions are about the action
    path, and synthesizing keeps the test off the GL stack); the 7 joint
    positions and the gripper are read from the sim, which is what closes the
    loop: each chunk is conditioned on the state the previous chunk produced.
    """
    img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    obs: dict[str, object] = {
        "observation/wrist_image_left": img,
        "observation/exterior_image_1_left": img,
        "observation/exterior_image_2_left": img,
    }
    for name in _PANDA_ARM_JOINTS:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        obs[name] = float(data.qpos[model.jnt_qposadr[jnt_id]])
    # DROID's gripper observation is normalized [0, 1]; finger_joint1's qpos
    # is metres. Normalize by the joint's own range (read from the model).
    grip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _PANDA_GRIPPER_JOINT)
    lo, hi = (float(v) for v in model.jnt_range[grip_id])
    obs[_PANDA_GRIPPER_JOINT] = (float(data.qpos[model.jnt_qposadr[grip_id]]) - lo) / (hi - lo)
    return obs


def test_spawned_server_drives_short_mujoco_rollout(policy):
    """Closed loop on real server actions: N_CHUNKS chunks drive a Panda arm.

    Acceptance (issue #2513): every action value across the whole rollout is
    a finite float and within the corresponding actuator's ctrlrange when one
    is defined (read from the MjModel); the episode completes (full planned
    chunks x steps, final qpos finite); and the policy is asked at least
    twice, proving the WebSocket connection survives repeated inference.
    """
    mujoco = pytest.importorskip("mujoco")
    panda_mj_description = pytest.importorskip(
        "robot_descriptions.panda_mj_description", reason="robot_descriptions not installed"
    )
    from strands_robots.simulation.mujoco.scene_ops import joint_drive_map

    model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)  # the asset's "home" keyframe
    mujoco.mj_forward(model, data)

    # A joint pose goes only where ctrl IS a joint pose (AGENTS.md actuator
    # rule): classify per actuator via the shared helper rather than assuming
    # the whole robot. The Panda MJCF ships position servos for the 7 arm
    # joints; its gripper is tendon-driven (one ctrl in tendon units coupling
    # two finger joints), so the gripper target is left uncommanded and named.
    servos, _other_drives = joint_drive_map(model, mujoco)
    servo_by_joint_name = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id): act_id for jnt_id, act_id in servos.items()
    }
    missing_servos = [n for n in _PANDA_ARM_JOINTS if n not in servo_by_joint_name]
    assert not missing_servos, f"Panda arm joints without a position servo: {missing_servos}"

    rng = np.random.default_rng(42)
    raw_chunks: list[np.ndarray] = []
    uncommanded: set[str] = set()
    inferences = 0

    for _chunk_idx in range(N_CHUNKS):
        obs = _mujoco_observation(mujoco, model, data, rng)
        steps = policy.get_actions_sync(obs, "pick up the red cube")
        inferences += 1
        assert isinstance(steps, list) and steps

        # The raw [T, D] chunk for the whole-rollout finiteness assertion.
        assert policy.last_rollout is not None
        raw_chunks.append(np.asarray(policy.last_rollout["action"], dtype=np.float64))

        for step in steps:
            for name, value in step.items():
                assert isinstance(value, float), (name, value)
                act_id = servo_by_joint_name.get(name)
                if act_id is None:
                    uncommanded.add(name)
                    continue
                # In-range against the actuator's own ctrlrange (from the
                # model, not hard-coded) before the write, so an out-of-range
                # action never reaches the sim.
                if int(model.actuator_ctrllimited[act_id]):
                    lo, hi = (float(v) for v in model.actuator_ctrlrange[act_id])
                    assert lo <= value <= hi, (name, value, (lo, hi))
                data.ctrl[act_id] = value
            for _ in range(SUBSTEPS_PER_ACTION):
                mujoco.mj_step(model, data)

    # Every action value across the whole rollout is a finite float.
    stacked = np.vstack(raw_chunks)
    assert stacked.ndim == 2 and stacked.shape[0] >= N_CHUNKS
    assert np.isfinite(stacked).all(), (stacked.min(), stacked.max())

    # The episode completed: full planned chunks, and the physics stayed sane.
    assert inferences == N_CHUNKS and inferences >= 2
    assert np.isfinite(data.qpos).all()

    # Only the tendon-driven gripper target may go uncommanded; every arm
    # joint action was written to its position servo.
    assert uncommanded <= {_PANDA_GRIPPER_JOINT}, uncommanded
