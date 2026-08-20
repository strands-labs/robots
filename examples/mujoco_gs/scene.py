# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default scene factory for the MuJoCo-GS example.

Mirrors the canonical MuJoCo-GS-Web scenario: a single small arm at the
origin, a small red cube within reach, a front / topdown / oblique camera
trio, and a ground plane. Pulled out into its own module so the agent and
the Gradio app can share the same setup without duplication.

Robot resolution is environment-dependent. The canonical demo robot is the
SO-101, but its MuJoCo asset does not resolve in every install (the
``robot_descriptions`` → git fallback can fail). So instead of trusting a
single config, :func:`build_default_scene` tries a prioritised list and
verifies that ``add_robot`` actually succeeded — falling back to the SO-100
(the SO-101's direct predecessor, identical 6-DoF kinematics) and then a
Franka Panda. This guarantees the scene always has a real arm, so the agent
never has to "repair" an empty world (which previously caused it to loop
through 20+ tool calls).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strands_robots.simulation import Simulation

logger = logging.getLogger(__name__)


# Default cube placement: ~30 cm in front of the arm, in the workspace.
DEFAULT_CUBE_POSITION = [0.20, 0.20, 0.025]
DEFAULT_CUBE_HALF_SIZE = [0.025, 0.025, 0.025]  # 5 cm cube
DEFAULT_CUBE_COLOR = [0.85, 0.10, 0.10, 1.0]  # canonical "red block"

# Prioritised robot configs. SO-101 is the canonical MuJoCo-GS-Web robot; the
# rest are fallbacks that resolve more reliably. All are small/medium arms
# with a gripper, so the demo prompts ("wave", "move the arm") still make
# sense whichever one loads.
ROBOT_CONFIG_CANDIDATES = ["so101", "so100", "so_arm100", "panda"]

# Human-friendly names for the prompt / status panel.
_ROBOT_PRETTY = {
    "so101": "SO-101 6-DoF arm",
    "so100": "SO-100 6-DoF arm (SO-101 predecessor, identical kinematics)",
    "so_arm100": "SO-ARM100 6-DoF arm",
    "panda": "Franka Emika Panda 7-DoF arm",
}


def _status_of(result) -> str:
    """Pull the ``status`` field out of an AgentTool result dict."""
    if isinstance(result, dict):
        return str(result.get("status", "unknown"))
    return "unknown"


def _add_robot_with_fallback(
    sim: Simulation,
    name: str,
    candidates: list[str],
) -> str:
    """Add a robot named ``name``, trying ``candidates`` until one resolves.

    Args:
        sim: the simulation.
        name: robot name to register in the scene.
        candidates: ordered list of ``data_config`` values to try.

    Returns:
        The ``data_config`` that successfully loaded.

    Raises:
        RuntimeError: if none of the candidates resolve — surfaced loudly so
            the failure is never silent (the old code logged a fake success).
    """
    errors = []
    for cfg in candidates:
        result = sim.add_robot(name=name, data_config=cfg, position=[0.0, 0.0, 0.0])
        status = _status_of(result)
        if status == "success":
            if cfg != candidates[0]:
                logger.warning(
                    "Robot config %r did not resolve; fell back to %r for robot %r.",
                    candidates[0],
                    cfg,
                    name,
                )
            logger.info("Loaded robot %r using data_config=%r.", name, cfg)
            return cfg
        errors.append(f"{cfg}: {status}")
        logger.info("Robot config %r unavailable (%s); trying next candidate.", cfg, status)

    raise RuntimeError(
        "Could not load any arm for the MuJoCo-GS scene. Tried "
        f"{candidates}. Results: {errors}. Install a MuJoCo Menagerie model "
        "or pass a config that resolves on this machine."
    )


def build_default_scene(
    sim: Simulation,
    robot_candidates: list[str] | None = None,
) -> dict:
    """Populate ``sim`` with an arm + red cube + cameras setup.

    Idempotent-ish: assumes ``sim`` is freshly created or has just been
    ``destroy()``ed and re-created. Calling on an already-populated sim will
    raise from MuJoCo's name table.

    Args:
        sim: a ``strands_robots.simulation.Simulation`` instance, *not yet*
            populated. ``create_world()`` will be called for you.
        robot_candidates: optional override of the prioritised robot config
            list. Defaults to :data:`ROBOT_CONFIG_CANDIDATES`.

    Returns:
        A dict summarising what was added — crucially including the
        ``robot_config`` that actually loaded, so the agent's system prompt
        and the UI reflect reality.

    Raises:
        RuntimeError: if no arm config resolves (see
            :func:`_add_robot_with_fallback`).
    """
    candidates = robot_candidates or ROBOT_CONFIG_CANDIDATES

    sim.create_world(timestep=0.002, gravity=[0.0, 0.0, -9.81], ground_plane=True)

    # The important fix: verify the robot actually loaded, with fallback.
    robot_config = _add_robot_with_fallback(sim, name="arm", candidates=candidates)

    sim.add_object(
        name="cube",
        shape="box",
        position=DEFAULT_CUBE_POSITION,
        size=DEFAULT_CUBE_HALF_SIZE,
        color=DEFAULT_CUBE_COLOR,
        mass=0.05,
    )

    # Three cameras framing the whole arm (base at origin, reaches up to
    # ~0.5 m when it waves) plus the cube at [0.2, 0.2]. All look at the
    # workspace centre ~[0.05, 0.05, 0.18] and sit far enough back that the
    # raised arm stays inside the frame with margin.
    sim.add_camera(
        name="front",
        position=[0.05, -0.95, 0.45],
        target=[0.05, 0.05, 0.18],
        fov=58.0,
        width=640,
        height=480,
    )
    sim.add_camera(
        name="topdown",
        position=[0.05, 0.05, 1.25],
        target=[0.05, 0.05, 0.0],
        fov=62.0,
        width=640,
        height=480,
    )
    sim.add_camera(
        name="oblique",
        position=[0.75, -0.7, 0.55],
        target=[0.05, 0.05, 0.18],
        fov=55.0,
        width=640,
        height=480,
    )

    # Stand the arm up on the benchtop in its "home" ready-pose (its zero pose
    # sprawls flat across the surface). We set both qpos and the actuator
    # targets (ctrl) so the pose holds when the sim is stepped.
    _erect_arm(sim, robot_name="arm")

    sim.step(20)  # settle into the home pose (actuators hold it)

    summary = {
        "robot_name": "arm",
        "robot_config": robot_config,
        "robots": [f"arm ({robot_config})"],
        "objects": [f"cube (red box at {DEFAULT_CUBE_POSITION})"],
        "cameras": ["front", "topdown", "oblique", "default"],
    }
    logger.info("MuJoCo-GS scene ready: %s", summary)
    return summary


def _erect_arm(sim, robot_name: str = "arm") -> bool:
    """Stand the arm up in its model's ``home`` keyframe (its zero pose sprawls
    flat). Sets the arm joints' ``qpos`` *and* the matching actuator targets
    (``ctrl``) so the pose holds when stepped. Touches only the robot's own
    hinge/slide joints — never free joints (e.g. the cube). No-op (returns
    ``False``) if the model has no home-like keyframe.
    """
    try:
        import mujoco

        m, d = sim.mj_model, sim.mj_data
        key_id = -1
        for k in range(m.nkey):
            kn = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, k) or "").lower()
            if "home" in kn:
                key_id = k
                break
        if key_id < 0:
            logger.info("No 'home' keyframe found; leaving arm in its zero pose.")
            return False

        kq = m.key_qpos[key_id]
        ns = f"{robot_name}/"
        # Joint types are compared BY VALUE: ``m.jnt_type[j]`` is a numpy integer and
        # ``x in (enum, ...)`` puts the enum on the left of ``==``, where a pybind11
        # enum does not match a numpy integer on every mujoco build.
        hinge_slide = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        # 1) qpos for the arm's own actuated joints.
        for j in range(m.njnt):
            jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if jn.startswith(ns) and int(m.jnt_type[j]) in hinge_slide:
                adr = m.jnt_qposadr[j]
                d.qpos[adr] = kq[adr]
        # 2) actuator targets so position actuators hold the pose.
        for a in range(m.nu):
            jid = int(m.actuator_trnid[a, 0])
            if jid >= 0 and int(m.jnt_type[jid]) in hinge_slide:
                d.ctrl[a] = kq[m.jnt_qposadr[jid]]
        mujoco.mj_forward(m, d)
        logger.info("Arm set to 'home' pose (erected on benchtop).")
        return True
    except Exception:  # pragma: no cover — non-fatal
        logger.warning("Could not set arm home pose.", exc_info=True)
        return False


def make_scene_description(robot_config: str = "so101", robot_name: str = "arm") -> str:
    """Build the scene-description block injected into the agent system prompt.

    Kept in sync with the *actual* robot that loaded so the agent is never
    told the scene contains an SO-101 when it really contains an SO-100.
    """
    pretty = _ROBOT_PRETTY.get(robot_config, f"{robot_config} arm")
    return f"""\
Scene contents (ALREADY BUILT — do not recreate):
  - One {pretty} named `{robot_name}`, mounted at the world origin.
  - One small red cube named `cube`, ~5 cm wide, on the ground at
    [0.20, 0.20, 0.025] (about 30 cm in front of the arm).
  - Three cameras: `front` (hero shot), `topdown` (overhead), `oblique`.
  - A photoreal panorama background composited behind everything.

The cube is dynamic; the panorama is at infinity (no parallax) unless you
swap to a `gsplat` background. Drive the scene with the `Simulation` tool's
`set_joint_positions`, `run_policy`, `move_object`, `apply_force`, and `step`
actions, then call `hybrid_render`.
"""


# Backwards-compatible default description (used if the prompt is built before
# a scene exists). Prefer :func:`make_scene_description` with the real config.
SCENE_DESCRIPTION = make_scene_description()
