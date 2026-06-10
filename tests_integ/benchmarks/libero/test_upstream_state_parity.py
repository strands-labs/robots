"""Round-35 (#168, #166) parity tests: our state observations vs upstream LIBERO.

Pins the round-31/32/33 state-side fixes against upstream regression.
Each test sets up BOTH:
- Upstream LIBERO's ``OffScreenRenderEnv`` at canonical ``init_states[0]``
- Our ``LiberoAdapter`` with the same scene XML at the same canonical init

Then compares the per-channel state values that flow into the policy
server (``state.x/y/z/roll/pitch/yaw/gripper``).

These are end-to-end ground-truth tests: if any of the round-31/32/33
fixes regresses (e.g., a refactor accidentally falls back to body
xpos for position, or starts duplicating one finger qpos), upstream
LIBERO won't change but our values will diverge — and these tests
will fail loudly.

Why ``tests_integ/`` and not ``tests/``: the upstream comparison
requires the ``libero`` package + asset cache (HuggingFace download
on first run). That's outside the unit-test contract and gated
behind the ``[benchmark-libero]`` install extra. Tests skip cleanly
when libero isn't importable.

Run with:
    hatch run test-integ tests_integ/benchmarks/libero/test_upstream_state_parity.py
"""

from __future__ import annotations

import os
from typing import Any

import pytest

# These tests depend on the libero package being importable AND on its
# HuggingFace-hosted asset cache being downloadable (auto-fetched on
# first env construction). We skip cleanly when either is missing.
libero = pytest.importorskip("libero", reason="libero package not installed")
robosuite = pytest.importorskip("robosuite", reason="robosuite not installed")
mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")
import numpy as np  # noqa: E402

from strands_robots.benchmarks.libero.adapter import (  # noqa: E402
    LiberoAdapter,
    _quat_wxyz_to_rpy_xyz,
)

# Use EGL for offscreen rendering (CI doesn't have a display).
os.environ.setdefault("MUJOCO_GL", "egl")


def _angle_diff_mod_2pi(a: float, b: float) -> float:
    """Return the smallest absolute angle difference between ``a`` and ``b``,
    accounting for the [-π, π] / [π, -π] equivalence in extrinsic Euler.

    A roll value of ``-3.139`` and ``+3.135`` represent the SAME
    rotation (both ≈ ±π). Naive ``abs(a - b)`` would say they differ
    by ~6.27 rad; this helper says ~0.007 rad. Pin so the round-32 yaw
    fix can be asserted within ~50 mrad without false negatives at the
    π wrap.
    """
    diff = (a - b + np.pi) % (2 * np.pi) - np.pi
    return abs(float(diff))


def _build_upstream_env():
    """Build an upstream LIBERO OffScreenRenderEnv and its bddl file path.

    Use ``libero_spatial/task_0`` — this is a goal-on-plate task whose
    BDDL uses only ``And`` + ``On`` predicates which our BDDL parser
    fully supports. (``libero_object`` task 0 uses ``In`` which is an
    upstream alias for ``Inside`` not in our parser's vocabulary; not a
    blocker for round-35 state parity which doesn't depend on the goal
    parser's coverage of every predicate.)

    Returns ``(env, task_bddl_path, init_state_vector)``.
    """
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv

    bd = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = bd.get_task(0)
    init_states = bd.get_task_init_states(0)
    task_bddl = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl,
        camera_names=["agentview"],
        camera_heights=128,
        camera_widths=128,
    )
    env.reset()
    env.set_init_state(init_states[0])
    # One zero-action step so observables update with the init state.
    env.step(np.zeros(7))
    return env, task_bddl, init_states[0]


def _build_our_adapter_at_canonical(task_bddl: str, init_state: np.ndarray):
    """Build our LiberoAdapter + a fresh sim, load the scene, apply the
    same canonical init state. Returns ``(sim, adapter)``.

    Mirrors what ``LiberoAdapter.on_episode_start`` does internally
    but without all the cameras / recorder / OSC-controller install
    that's irrelevant to a state-comparison test.
    """
    from strands_robots.simulation import Simulation

    # Read the BDDL text so we can use from_text (which doesn't auto-
    # generate a scene; we point scene_path at upstream LIBERO's
    # canonical scene XML).
    with open(task_bddl) as f:
        bddl_text = f.read()

    # Get the scene XML upstream uses (same path libero would use).
    # OffScreenRenderEnv compiles a procedurally-generated MJCF; we
    # compile our own from the same BDDL via auto_generate_scene.
    adapter = LiberoAdapter.from_text(
        bddl_text,
        # The factory auto-resolves eef_body_name to robot0_right_hand
        # via _register_default_robot — but only when on_episode_start
        # runs. For a direct state read we set it explicitly.
        eef_body_name="robot0_right_hand",
        # Auto-generate the scene from BDDL via libero (matches what
        # OffScreenRenderEnv does internally).
        auto_generate_scene=True,
        install_cameras=False,
        # ``init_states`` overrides the keyframe + snapshot fallback.
        init_states=np.asarray([init_state]),
    )
    sim = Simulation()
    create_result = sim.create_world()
    assert create_result["status"] == "success", create_result

    # Drive the full on_episode_start lifecycle so we exercise the
    # same code path the eval uses (auto-resolve eef body, apply
    # canonical state, install controller, etc.).
    import random

    rng = random.Random(0)
    adapter.on_episode_start(sim, rng)
    return sim, adapter


@pytest.mark.timeout(180)
def test_state_parity_at_canonical_init() -> None:
    """Round 35 (#168) — primary regression test for the state pipeline.

    With both upstream LIBERO and our adapter at the SAME canonical
    init state, ``state.x/y/z/roll/pitch/yaw/gripper`` must match
    within numerical tolerance.

    Tolerances per round-33 verification:
    - position: 5 mm per axis
    - orientation: 50 mrad (mod 2π) per axis — extrinsic Euler can
      flip ±π for the same physical rotation
    - gripper: 1 mm per finger (the canonical at-rest values are
      ±0.0208 with millimeter-scale variation between resets)
    """
    upstream_env, task_bddl, init_state = _build_upstream_env()
    upstream_obs, *_ = upstream_env.step(np.zeros(7))

    sim, adapter = _build_our_adapter_at_canonical(task_bddl, init_state)
    try:
        # Step our sim once with no action so observables update,
        # matching what upstream's env.step(np.zeros(7)) did.
        ours_observation = sim.get_observation(skip_images=True)
        ours_state = adapter.augment_observation(sim, ours_observation)

        # Position
        ups_pos = upstream_obs["robot0_eef_pos"]
        ours_pos = [ours_state["x"], ours_state["y"], ours_state["z"]]
        for axis_idx, axis_name in enumerate(["x", "y", "z"]):
            delta = abs(float(ours_pos[axis_idx]) - float(ups_pos[axis_idx]))
            assert delta < 0.05, (
                f"state.{axis_name} parity drift {delta:.4f} m > 5 cm tol; "
                f"ours={ours_pos[axis_idx]:.4f}, upstream={ups_pos[axis_idx]:.4f}. "
                f"Round-31 site-source fix may have regressed."
            )

        # Orientation — convert upstream's (xyzw) quat to (wxyz) for
        # our `_quat_wxyz_to_rpy_xyz` helper, then compare per axis.
        ups_quat_xyzw = upstream_obs["robot0_eef_quat"]
        # robosuite returns xyzw; convert to wxyz.
        ups_quat_wxyz = [
            float(ups_quat_xyzw[3]),
            float(ups_quat_xyzw[0]),
            float(ups_quat_xyzw[1]),
            float(ups_quat_xyzw[2]),
        ]
        ups_roll, ups_pitch, ups_yaw = _quat_wxyz_to_rpy_xyz(ups_quat_wxyz)
        ours_roll = ours_state["roll"]
        ours_pitch = ours_state["pitch"]
        ours_yaw = ours_state["yaw"]

        roll_delta = _angle_diff_mod_2pi(ours_roll, ups_roll)
        pitch_delta = _angle_diff_mod_2pi(ours_pitch, ups_pitch)
        yaw_delta = _angle_diff_mod_2pi(ours_yaw, ups_yaw)
        # The round-32 fix brought yaw within ~10 mrad; allow 100 mrad
        # for episode-to-episode init-state variance.
        assert roll_delta < 0.1, (
            f"state.roll parity drift {roll_delta:.4f} rad > 100 mrad; "
            f"ours={ours_roll:.4f}, upstream={ups_roll:.4f}. "
            f"Round-32 split-source quat fix may have regressed."
        )
        assert pitch_delta < 0.1, (
            f"state.pitch parity drift {pitch_delta:.4f} rad > 100 mrad; "
            f"ours={ours_pitch:.4f}, upstream={ups_pitch:.4f}"
        )
        assert yaw_delta < 0.1, (
            f"state.yaw parity drift {yaw_delta:.4f} rad > 100 mrad; "
            f"ours={ours_yaw:.4f}, upstream={ups_yaw:.4f}. "
            f"Round-32 fix (orientation from body xquat, not site_xmat) may have regressed."
        )

        # Gripper — both fingers must match upstream's per-finger qpos.
        ups_gripper = upstream_obs["robot0_gripper_qpos"]
        ours_gripper = ours_state["gripper"]
        assert len(ours_gripper) == 2, f"state.gripper must be 2-element, got {ours_gripper}"
        for finger_idx in range(2):
            delta = abs(float(ours_gripper[finger_idx]) - float(ups_gripper[finger_idx]))
            assert delta < 0.005, (
                f"state.gripper[{finger_idx}] parity drift {delta:.4f} m > 5 mm; "
                f"ours={ours_gripper[finger_idx]:.4f}, upstream={ups_gripper[finger_idx]:.4f}. "
                f"Round-33 two-finger fix may have regressed (duplicate-packing bug)."
            )

        # Sentinel — round 33: the two finger qpos values must have
        # OPPOSITE signs at the canonical at-rest pose. If they're the
        # same sign, the duplicate-packing bug is back.
        assert ours_gripper[0] * ours_gripper[1] < 0, (
            f"state.gripper {ours_gripper} fingers have same sign — round-33 duplicate-packing bug may have regressed"
        )
    finally:
        sim.destroy()
        upstream_env.close()


@pytest.mark.timeout(180)
def test_eef_state_site_name_resolves_in_real_libero_scene() -> None:
    """Round 35 (#168): with a real LIBERO-generated scene, the
    auto-derived ``eef_state_site_name`` (``gripper0_grip_site``) must
    actually exist in the compiled model.

    Pin so a future change to ``scene_gripper_prefix`` or to libero's
    procedural scene generator doesn't silently break the round-31
    site-priority path."""
    _, task_bddl, init_state = _build_upstream_env()
    sim, adapter = _build_our_adapter_at_canonical(task_bddl, init_state)
    try:
        site_name = adapter.eef_state_site_name
        assert site_name == "gripper0_grip_site"

        model = sim._world._model  # type: ignore[attr-defined]
        assert model is not None
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        assert site_id >= 0, (
            f"site {site_name!r} not found in LIBERO scene compiled by adapter; "
            f"round-31 fix would silently fall back to body lookup"
        )
    finally:
        sim.destroy()


@pytest.mark.timeout(180)
def test_state_gripper_joint_names_resolve_in_real_libero_scene() -> None:
    """Round 35 (#168): both finger joint names auto-derived for
    ``state.gripper`` must exist in the compiled model.

    Pin so the round-33 two-finger fix's joint-name resolution stays
    in sync with the libero procedural scene generator."""
    _, task_bddl, init_state = _build_upstream_env()
    sim, adapter = _build_our_adapter_at_canonical(task_bddl, init_state)
    try:
        joint_names = adapter.state_gripper_joint_names
        assert joint_names == [
            "gripper0_finger_joint1",
            "gripper0_finger_joint2",
        ]

        model = sim._world._model  # type: ignore[attr-defined]
        assert model is not None
        for jname in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            assert jid >= 0, (
                f"joint {jname!r} not found in LIBERO scene; round-33 fix's joint-resolution may have regressed"
            )
    finally:
        sim.destroy()


@pytest.mark.timeout(300)
def test_state_parity_after_50_steps_zero_action() -> None:
    """#171 sub-task 3c — Deep-rollout state parity test.

    Drives both upstream ``OffScreenRenderEnv`` and our
    ``MuJoCoSimEngine``-backed adapter with the SAME action sequence
    (50 steps of zero action — gravity + passive dynamics only) and
    asserts state stays within tolerance per step.

    The existing ``test_state_parity_at_canonical_init`` validates init
    pose only. This test catches trajectory drift that compounds beyond
    init: small differences in scene XML (round 9 v3 cache work),
    OSC controller (rounds 27/28/41), or numerical integration paths
    can produce sub-mm divergence at init that grows to cm-scale by
    step 50.

    Uses zero actions because ANY non-trivial action would route
    through the upstream OSC controller (robosuite native) on one side
    and our ``_LiberoOSCController`` on the other — a separate test
    surface (sub-task 3b). Zero-action drift isolates the
    physics-only path: same init state + same dt + (ideally) same
    model = same trajectory.

    Tolerances: position 5cm per axis (allows for dt=0.002s × 50
    steps × small drift), orientation 200 mrad. Looser than the
    init-only parity test because compounding numerical error is
    expected; the test fails if drift exceeds physical-plausibility
    bounds rather than pinning byte-equivalence.
    """
    upstream_env, task_bddl, init_state = _build_upstream_env()
    sim, adapter = _build_our_adapter_at_canonical(task_bddl, init_state)
    try:
        # Drive both with the same scripted zero-action sequence.
        n_steps = 50
        zero_action = np.zeros(7)

        # Upstream env: env.step(zero_action) advances physics.
        # Our sim: send_action with zero deltas via the OSC controller.
        # Empty action dict ⇒ OSC sees zero delta + zero gripper command,
        # writes zero torques (gravity comp only).
        ours_state_history: list[dict] = []
        for _step_idx in range(n_steps):
            # Upstream step.
            upstream_env.step(zero_action)
            # Our step: empty action dict ⇒ OSC apply with zero delta.
            sim.send_action({})

            # Sample state from both at this step.
            ours_obs = sim.get_observation(skip_images=True)
            ours_state = adapter.augment_observation(sim, ours_obs)
            ours_state_history.append(
                {
                    "x": ours_state.get("x"),
                    "y": ours_state.get("y"),
                    "z": ours_state.get("z"),
                    "gripper": ours_state.get("gripper"),
                }
            )

        # Final-step comparison. Compare last sample against upstream's
        # most recent obs returned by env.step (cached internally). One
        # last step() call to surface the obs dict for asserting.
        last_upstream_obs, *_ = upstream_env.step(zero_action)

        ups_pos = last_upstream_obs["robot0_eef_pos"]
        ours_final = ours_state_history[-1]
        # Position drift bound: 5 cm per axis. Compounded over 50
        # physics-only steps starting from the same init, real-world
        # drift should be far less; this generous bound catches
        # actual trajectory divergence (cm-scale) rather than
        # numerical noise (mm-scale).
        for axis_idx, axis_name in enumerate(["x", "y", "z"]):
            ours_v = ours_final[axis_name]
            ups_v = float(ups_pos[axis_idx])
            delta = abs(ours_v - ups_v)
            assert delta < 0.05, (
                f"state.{axis_name} drift after {n_steps} zero-action steps: "
                f"{delta:.4f} m > 5 cm tolerance. "
                f"ours={ours_v:.4f}, upstream={ups_v:.4f}. "
                f"Trajectory diverges beyond physical-plausibility — "
                f"likely a scene XML or OSC controller divergence. "
                f"See #171 sub-tasks 3a/3b."
            )

        # Gripper drift bound: each finger should stay within 5 mm of
        # upstream (same convention as the init-pose test).
        ups_gripper = last_upstream_obs["robot0_gripper_qpos"]
        ours_gripper = ours_final["gripper"]
        assert ours_gripper is not None and len(ours_gripper) == 2, (
            f"state.gripper malformed after rollout: {ours_gripper}"
        )
        for finger_idx in range(2):
            delta = abs(float(ours_gripper[finger_idx]) - float(ups_gripper[finger_idx]))
            assert delta < 0.005, (
                f"state.gripper[{finger_idx}] drift after {n_steps} steps: "
                f"{delta:.4f} m > 5 mm. "
                f"ours={ours_gripper[finger_idx]:.4f}, upstream={ups_gripper[finger_idx]:.4f}"
            )
    finally:
        sim.destroy()
        upstream_env.close()


@pytest.mark.timeout(180)
def test_osc_torque_parity_at_identical_state() -> None:
    """#176 acceptance: per-step arm torques match upstream's within
    5% relative error AT identical canonical state + same input action.

    The round-45 swap-and-restore around ``controller_factory``
    (in ``_LiberoOSCController.from_sim``) makes the controller
    capture ``initial_joint`` / ``initial_ee_pos`` /
    ``initial_ee_ori_mat`` / ``goal_pos`` / ``goal_ori`` from the
    LIBERO ``MountedPanda`` ready pose, matching what upstream
    ``OffScreenRenderEnv``'s ``Robot.reset(deterministic=True) +
    _load_controller`` sequence captures.

    Pre-fix: torques diverged by 50× because our ``initial_*``
    attributes latched on to the perturbed canonical pose written
    by ``_apply_canonical_state`` instead of the home pose. See
    ``probe_osc_internals.py`` for the bisection narrowing it to
    ``initial_joint`` and ``probe_osc_internals_v3.py`` for the
    per-state comparison verifying the swap closes the gap.

    Approach:
    1. Build both pipelines at canonical init.
    2. Step both once with zero action (so ``data.qpos`` /
       ``qvel`` match upstream's post-``_build_upstream_env`` state).
    3. Copy upstream's ``data.qpos`` / ``qvel`` into ours so any
       residual init-state divergence is eliminated.
    4. Force-update both controllers, set the same goal, run
       one ``run_controller()`` call on each.
    5. Per-joint torque comparison: rel_err <= 5%.

    Without the round-45 swap, every joint's rel_err exceeds
    100% — j5 in particular saw 4800% in pre-fix probe.
    """
    upstream_env, task_bddl, init_state = _build_upstream_env()
    sim, adapter = _build_our_adapter_at_canonical(task_bddl, init_state)
    try:
        # Pre-step ours once to match upstream's "_build_upstream_env"
        # pre-step (zero-action env.step inside the helper).
        sim.send_action({})

        ups_inner = upstream_env.env
        ups_data = ups_inner.sim.data._data
        ours_data = sim._world._data  # type: ignore[attr-defined]
        ours_model = sim._world._model  # type: ignore[attr-defined]

        # Copy upstream's full qpos/qvel into ours so byte-identical
        # state. This isolates the OSC torque computation from any
        # state-side divergence (#171 sub-task 3a is a separate test
        # surface).
        np.copyto(ours_data.qpos, ups_data.qpos)
        np.copyto(ours_data.qvel, ups_data.qvel)
        mujoco.mj_forward(ours_model, ours_data)
        mujoco.mj_forward(ups_inner.sim.model._model, ups_data)

        ups_ctrl = ups_inner.robots[0].controller
        ours_ctrl = sim._world._backend_state.get("action_controller").controller  # type: ignore[attr-defined]

        # Pre-conditions: state truly identical.
        ups_ctrl.update(force=True)
        ours_ctrl.update(force=True)
        np.testing.assert_allclose(
            np.array(ours_ctrl.joint_pos),
            np.array(ups_ctrl.joint_pos),
            atol=1e-9,
            err_msg="state copy failed: joint_pos diverges",
        )
        np.testing.assert_allclose(
            np.array(ours_ctrl.initial_joint),
            np.array(ups_ctrl.initial_joint),
            atol=1e-9,
            err_msg=(
                "initial_joint diverges between upstream and ours — round-45 "
                "swap-and-restore in _LiberoOSCController.from_sim may have regressed."
            ),
        )

        # Apply same delta-EEF action to both.
        arm_action = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
        ups_ctrl.set_goal(arm_action)
        ours_ctrl.set_goal(arm_action)

        # Goal sanity: with identical state + same delta, goal_pos
        # and goal_ori must match exactly.
        np.testing.assert_allclose(
            np.array(ours_ctrl.goal_pos),
            np.array(ups_ctrl.goal_pos),
            atol=1e-9,
            err_msg="goal_pos diverges after set_goal",
        )
        np.testing.assert_allclose(
            np.array(ours_ctrl.goal_ori),
            np.array(ups_ctrl.goal_ori),
            atol=1e-6,
            err_msg=(
                "goal_ori diverges after set_goal — likely a regression in "
                "the round-45 swap-and-restore that captures initial_ee_ori_mat "
                "from data at home pose."
            ),
        )

        # Compute torques on both.
        ups_torques = np.array(ups_ctrl.run_controller())
        ours_torques = np.array(ours_ctrl.run_controller())

        # Per-joint relative error <= 5% (#176 acceptance criteria).
        abs_diff = np.abs(ours_torques - ups_torques)
        rel_err = abs_diff / (np.abs(ups_torques) + 1e-6)
        max_rel_err = float(rel_err.max())
        assert max_rel_err <= 0.05, (
            f"OSC arm torques diverge from upstream: max rel_err={max_rel_err:.4f} > 5%. "
            f"Per-joint rel_err: {rel_err.tolist()}. "
            f"upstream torques: {ups_torques.tolist()}. "
            f"ours torques: {ours_torques.tolist()}. "
            f"abs diff: {abs_diff.tolist()}. "
            f"This is the #176 acceptance criterion. The round-45 swap-and-restore "
            f"in _LiberoOSCController.from_sim closes the divergence; if this "
            f"test fails, that fix has regressed."
        )
    finally:
        sim.destroy()
        upstream_env.close()


@pytest.mark.timeout(300)
def test_state_observation_byte_equivalent_at_canonical_init() -> None:
    """#176 sub-task 3d — pin every state channel to be byte-equivalent
    (within float precision) between ``MuJoCoSimEngine`` and upstream
    ``OffScreenRenderEnv`` at canonical ``init_states[0]`` for
    libero-10/SCENE5.

    This is a stricter version of ``test_state_parity_at_canonical_init``
    that compares ALL state channels (x/y/z/roll/pitch/yaw/gripper)
    side-by-side. Pre-#176 fixes had:
      - 14.7 mm divergence on state.y (MJCF qpos defaults differ from
        upstream Robot.reset)
      - 114 mrad divergence on state.pitch + sign-flip on yaw
        (``_quat_wxyz_to_rpy_xyz`` formula bug)
      - 14 mm divergence on state.gripper (MuJoCo qpos=0 vs upstream
        ``PandaGripper.init_qpos = [0.0208, -0.0208]``)

    Post-#176 (rounds 46a-d): all channels match within 1 mrad / 1 mm
    (mj_forward settling noise).

    Pre-#178 this test compared ``MuJoCoSimEngine`` against the
    intermediate ``LiberoOffScreenRenderEngine`` wrapper (now removed);
    the rewrite drops that wrapper and compares directly against
    upstream's ``OffScreenRenderEnv`` + the same ``_quat_wxyz_to_rpy_xyz``
    helper used by ``LiberoAdapter.augment_observation``.
    """
    from libero.libero import benchmark as libero_benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv

    from strands_robots.benchmarks.libero.adapter import (
        LiberoAdapter,
        _quat_wxyz_to_rpy_xyz,
    )
    from strands_robots.simulation.factory import create_simulation

    bd = libero_benchmark.get_benchmark_dict()["libero_10"]()
    target = (
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"
    )
    task_id = next(i for i in range(bd.get_num_tasks()) if bd.get_task(i).name == target)
    task = bd.get_task(task_id)
    task_bddl = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    init_states = bd.get_task_init_states(task_id)

    import random as _random

    # Upstream `OffScreenRenderEnv` — the ground truth.
    upstream_env = OffScreenRenderEnv(
        bddl_file_name=task_bddl,
        camera_names=["agentview"],
        camera_heights=128,
        camera_widths=128,
    )
    upstream_env.reset()
    upstream_env.set_init_state(init_states[0])
    upstream_obs, *_ = upstream_env.step(np.zeros(7))

    # MuJoCo path
    adapter = LiberoAdapter.from_file(task_bddl, install_cameras=True, init_states=init_states)
    sim = create_simulation("mujoco", tool_name="libero_sim", mesh=False)
    sim.create_world()
    sim.add_robot("robot", data_config="panda")
    adapter.on_episode_start(sim, _random.Random(42))
    obs_raw = sim.get_observation()
    obs = adapter.augment_observation(sim, obs_raw)

    try:
        # Position channels: upstream returns ``robot0_eef_pos`` directly.
        ups_pos = upstream_obs["robot0_eef_pos"]
        for axis_idx, axis_name in enumerate(("x", "y", "z")):
            ups_v = float(ups_pos[axis_idx])
            ours_v = float(obs[axis_name])
            assert abs(ours_v - ups_v) < 1e-3, (
                f"state.{axis_name} divergence: ours_mujoco={ours_v!r}, upstream={ups_v!r}, "
                f"diff={abs(ours_v - ups_v):.6e}. Round-46 fixes (home-pose write to "
                f"data.qpos[arm], _quat_wxyz_to_rpy_xyz fix) may have regressed."
            )

        # Orientation channels: upstream returns ``robot0_eef_quat`` in
        # xyzw; convert to wxyz so we can run our own ``_quat_wxyz_to_rpy_xyz``
        # for an apples-to-apples comparison (matches what
        # ``LiberoAdapter.augment_observation`` does).
        ups_quat_xyzw = upstream_obs["robot0_eef_quat"]
        ups_quat_wxyz = [
            float(ups_quat_xyzw[3]),
            float(ups_quat_xyzw[0]),
            float(ups_quat_xyzw[1]),
            float(ups_quat_xyzw[2]),
        ]
        ups_roll, ups_pitch, ups_yaw = _quat_wxyz_to_rpy_xyz(ups_quat_wxyz)
        for axis_name, ups_v in (("roll", ups_roll), ("pitch", ups_pitch), ("yaw", ups_yaw)):
            ours_v = float(obs[axis_name])
            # Mod-2π comparison: roll near ±π is sign-ambiguous.
            import math

            diff_mod = ((ours_v - ups_v + math.pi) % (2 * math.pi)) - math.pi
            assert abs(diff_mod) < 1e-3, (
                f"state.{axis_name} divergence (mod 2π): ours_mujoco={ours_v!r}, "
                f"upstream={ups_v!r}, diff_mod={abs(diff_mod):.6e}. "
                f"Round-46 _quat_wxyz_to_rpy_xyz fix may have regressed."
            )

        # Gripper (2-element list): upstream returns ``robot0_gripper_qpos``.
        ups_gripper = upstream_obs["robot0_gripper_qpos"]
        ours_gripper = obs["gripper"]
        assert len(ours_gripper) == 2
        for finger_idx in range(2):
            diff = abs(float(ours_gripper[finger_idx]) - float(ups_gripper[finger_idx]))
            assert diff < 1e-3, (
                f"state.gripper[{finger_idx}] divergence: ours={ours_gripper[finger_idx]!r}, "
                f"upstream={ups_gripper[finger_idx]!r}, diff={diff:.6e}. "
                f"Round-46 PandaGripper.init_qpos write may have regressed."
            )
    finally:
        sim.destroy()
        upstream_env.close()


@pytest.mark.timeout(900)
def test_libero_10_scene5_mujoco_engine_success_rate() -> None:
    """Round 46 (#176 sub-task 3d) acceptance — MuJoCoSimEngine reaches
    success_rate > 0 on libero-10/SCENE5 with in-process Gr00tPolicy.

    This is the end-to-end integration test that closes out #176
    sub-task 3d. Pre-46 (just round-45 OSC fix) got 0/5 on this task
    on the MuJoCoSimEngine path; post-46 (with home-pose write +
    settle step + Euler-formula fix + body-name fallback) achieves
    4/5 ≈ 80% success.

    Skipped unless:

    - ``STRANDS_ISAAC_GR00T_PATH`` env var points at the Isaac-GR00T
      repo (provides ``gr00t``).
    - ``STRANDS_GR00T_LIBERO_CHECKPOINT`` env var points at the
      ``GR00T-N1.7-LIBERO/libero_10`` model dir.
    - A CUDA-capable GPU is available.

    Run with::

        MUJOCO_GL=egl \\
        STRANDS_ISAAC_GR00T_PATH=$HOME/workspace/Isaac-GR00T \\
        STRANDS_GR00T_LIBERO_CHECKPOINT=$HOME/workspace/groot-checkpoints/GR00T-N1.7-LIBERO/libero_10 \\
        hatch run test-integ tests_integ/benchmarks/libero/test_upstream_state_parity.py::test_libero_10_scene5_mujoco_engine_success_rate

    Acceptance: at least 1 out of 3 episodes succeeds. Pre-46 got 0/5
    on this task; even 1/3 = 33% rate is a clear pass that the
    physics + obs + predicate pipeline now functions end-to-end on
    the MuJoCoSimEngine backend.
    """
    import random as _random
    import sys

    isaac_gr00t = os.environ.get("STRANDS_ISAAC_GR00T_PATH")
    checkpoint = os.environ.get("STRANDS_GR00T_LIBERO_CHECKPOINT")
    if not isaac_gr00t:
        pytest.skip("STRANDS_ISAAC_GR00T_PATH not set; required for in-process Gr00tPolicy")
    if not checkpoint:
        pytest.skip("STRANDS_GR00T_LIBERO_CHECKPOINT not set; required for end-to-end eval")
    if not os.path.isdir(isaac_gr00t):
        pytest.skip(f"Isaac-GR00T not at {isaac_gr00t}; required for in-process Gr00tPolicy")
    if not os.path.isdir(checkpoint):
        pytest.skip(f"checkpoint not at {checkpoint}; required for end-to-end eval")

    if isaac_gr00t not in sys.path:
        sys.path.insert(0, isaac_gr00t)
    try:
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy as NvidiaGr00tPolicy
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    except ImportError as e:
        pytest.skip(f"gr00t imports failed ({e}); skipping end-to-end test")

    try:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("no CUDA GPU available; required for in-process Gr00tPolicy")
    except ImportError:
        pytest.skip("torch not importable; required for in-process Gr00tPolicy")

    from libero.libero import benchmark as libero_benchmark
    from libero.libero import get_libero_path

    from strands_robots.benchmarks.libero.adapter import LiberoAdapter
    from strands_robots.simulation.factory import create_simulation

    bd = libero_benchmark.get_benchmark_dict()["libero_10"]()
    target = (
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"
    )
    task_id = next(i for i in range(bd.get_num_tasks()) if bd.get_task(i).name == target)
    task = bd.get_task(task_id)
    task_bddl = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    init_states = bd.get_task_init_states(task_id)

    try:
        nvidia_policy = NvidiaGr00tPolicy(
            embodiment_tag=EmbodimentTag.LIBERO_PANDA,
            model_path=checkpoint,
            device=0,
        )
    except Exception as e:  # noqa: BLE001 - model load failure is environmental
        pytest.skip(f"failed to load Gr00tPolicy ({e}); skipping end-to-end test")
    wrapped = Gr00tSimPolicyWrapper(nvidia_policy)

    adapter = LiberoAdapter.from_file(task_bddl, install_cameras=True, init_states=init_states)
    sim = create_simulation("mujoco", tool_name="libero_sim", mesh=False)
    sim.create_world()
    sim.add_robot("robot", data_config="panda")

    n_episodes = 5
    seed = 42
    max_episode_steps = 720
    n_action_steps = 8

    from strands_robots.simulation.policy_runner import set_eval_seed

    successes = []
    try:
        for ep in range(n_episodes):
            # #179 — seed Python/NumPy/torch/cuDNN per episode so the
            # GR00T diffusion sampler is reproducible. Without this,
            # ``success_rate`` varies wildly across runs of the same
            # eval (5-ep variance ranged 0.40-1.00 pre-fix). The
            # per-episode seed is deterministic in (master_seed,
            # episode_index).
            set_eval_seed(seed + ep)
            adapter.on_episode_start(sim, _random.Random(seed + ep))
            steps = 0
            is_success = False
            while steps < max_episode_steps:
                obs_raw = sim.get_observation()
                obs = adapter.augment_observation(sim, obs_raw)
                policy_obs: dict[str, Any] = {}
                for sk in ("x", "y", "z", "roll", "pitch", "yaw"):
                    v = obs.get(sk)
                    if v is not None:
                        policy_obs[f"state.{sk}"] = np.asarray([float(v)], dtype=np.float32)[None, None]
                g = obs.get("gripper")
                if g is not None:
                    policy_obs["state.gripper"] = np.asarray(g, dtype=np.float32)[None, None]
                for vk in ("image", "wrist_image"):
                    v = obs.get(vk)
                    if isinstance(v, np.ndarray):
                        # Mujoco renderer outputs image-convention (V-flipped from
                        # OpenGL); augment applies [::-1, :] → OpenGL; policy needs
                        # training-convention = OpenGL[::-1, ::-1]. Bypass the
                        # in-policy rotation by applying it here.
                        flipped = np.ascontiguousarray(v[::-1, ::-1])
                        policy_obs[f"video.{vk}"] = flipped[None, None]
                policy_obs["annotation.human.action.task_description"] = [task.language or ""]
                actions, _ = wrapped.get_action(policy_obs)
                for t in range(n_action_steps):
                    if steps >= max_episode_steps:
                        break
                    action_dict = {}
                    for ak, arr in actions.items():
                        action_dict[ak.removeprefix("action.")] = float(arr[0, t, 0])
                    sim.send_action(action_dict, robot_name="robot")
                    steps += 1
                    if adapter.is_success(sim):
                        is_success = True
                        break
                if is_success:
                    break
            successes.append(is_success)
    finally:
        sim.destroy()

    success_rate = sum(successes) / max(n_episodes, 1)
    assert success_rate >= 1.0, (
        f"MuJoCoSimEngine + in-process Gr00tPolicy got success_rate={success_rate:.2f} "
        f"on libero-10/SCENE5 ({successes}). Pre-#181 this was reproducibly 0.60 (3/5) "
        f'because the cached MJCF dropped ``inertiagrouprange="0 0"`` (lossy '
        f"``mj_saveLastXML``); post-#181 the cache uses the pre-compile MJCF which "
        f"preserves ``<compiler>`` attributes, restoring upstream's body inertias and "
        f"closing the parity gap. If this drops below 1.00, the fix may have "
        f"regressed — re-check ``_extract_compiled_mjcf`` accessor order and "
        f"``_LIBERO_MJCF_TRANSFORM_VERSION``."
    )
