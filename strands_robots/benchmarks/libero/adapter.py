"""``LiberoAdapter`` - :class:`BenchmarkProtocol` driven by a LIBERO BDDL file.

LIBERO is a suite of ~130 tabletop manipulation tasks built around a Franka
Panda. Each task ships as a BDDL problem file + an MJCF scene. The adapter
compiles the BDDL ``:goal`` into a sparse success predicate via
:mod:`strands_robots.benchmarks.libero.bddl_parser` and drives the scene
through the standard :class:`BenchmarkProtocol` lifecycle:

1. :meth:`on_episode_start` - optional ``sim.load_scene(scene_path)``, then
   the base ``BenchmarkProtocol`` compatibility check (Panda-only), then
   per-episode jitter of ``(:init ...)`` object positions.
2. :meth:`on_step` - sparse: ``StepInfo(reward=0.0, done=False)``. LIBERO
   does not define a dense reward.
3. :meth:`is_success` - walks the compiled ``:goal`` predicate tree against
   the current sim state.

**Panda-only by design.** LIBERO's scene MJCFs ``<include>`` Panda geometry
and BDDL predicates reference Panda gripper body names
(``robot0_gripper_*``). Retargeting to a different robot would require
rewriting every BDDL predicate against different body names and is out of
scope for this adapter. Subclass :class:`LiberoAdapter` and override
:attr:`supported_robots` + :attr:`default_robot` if you know what you're
doing.

The adapter does NOT require the ``libero`` Python package to be installed -
only a BDDL string / file and (optionally) an MJCF scene path. The
:func:`strands_robots.benchmarks.libero.suite.load_libero_suite` helper is
the one that pulls in the upstream package to discover task files.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from strands_robots.benchmarks.libero.bddl_parser import (
    BDDLParseError,
    BDDLProblem,
    Node,
    compile_goal,
    parse_bddl,
    parse_bddl_file,
)
from strands_robots.simulation.benchmark import BenchmarkProtocol, StepInfo
from strands_robots.simulation.models import SimCamera, SimRobot
from strands_robots.utils import get_base_dir, require_optional

if TYPE_CHECKING:
    import random

    from strands_robots.simulation.base import SimEngine

logger = logging.getLogger(__name__)


class LiberoAdapter(BenchmarkProtocol):
    """Panda-only :class:`BenchmarkProtocol` driven by a parsed LIBERO BDDL task.

    Construct with a BDDL file path (``from_file``) or raw BDDL text
    (``from_text``) - direct ``__init__`` is for advanced use when you
    already have a :class:`BDDLProblem`.

    Example::

        from strands_robots.benchmarks.libero import LiberoAdapter

        adapter = LiberoAdapter.from_file(
            "libero/tasks/libero_spatial/pick_up_the_red_cube.bddl",
            scene_path="libero/assets/scenes/libero_spatial_scene.xml",
        )
        sim.register_benchmark("pick-red-cube", adapter)
        sim.evaluate_benchmark("pick-red-cube", policy_provider="mock",
                               n_episodes=10, seed=42)

    Attributes:
        max_steps: Default 720 (NVIDIA upstream
            ``MultiStepConfig.max_episode_steps`` for LIBERO eval —
            see ``Isaac-GR00T/gr00t/eval/rollout_policy.py``). Override
            per-task by passing ``max_steps=`` to the constructor or
            mutating the attribute after construction. Pre-#168 we used the LIBERO repository's lifelong-learning
            convention of 300; that was too short for libero_10's
            longer-horizon manipulation tasks (e.g. multi-step pick-
            and-place chains) and was contributing to ``success_rate=0``
            on tasks that needed more time. 720 matches the canonical
            GR00T-N1.7-LIBERO eval setup.
        problem: The parsed :class:`BDDLProblem`. Stored for introspection
            (agents may read ``problem.language`` as the instruction).
    """

    max_steps: int = 720
    supported_robots_list: list[str] = ["panda"]
    default_robot_name: str = "panda"

    #: Cameras the ``libero_panda`` ``Gr00tDataConfig`` expects to find on the
    #: sim. Names match the bare keys of its ``video_keys`` (``video.image``
    #: -> ``image``, ``video.wrist_image`` -> ``wrist_image``) so the policy's
    #: ``_build_service_observation`` picks them up directly without an
    #: explicit ``observation_mapping``.
    #:
    #: Poses are world-fixed approximations of LIBERO's RoboSuite-conventional
    #: views (third-person "agentview" + wrist view). When the scene MJCF
    #: declares the canonical RoboSuite cameras (``agentview`` for third-person,
    #: ``robot0_eye_in_hand`` body-mounted to ``robot0_right_hand`` for the
    #: wrist view), :attr:`_scene_camera_aliases` renames them at MJCF-load
    #: time so the model's compiled cameras are exactly ``image`` /
    #: ``wrist_image`` - the static fallbacks below never get installed and
    #: the policy sees the real, gripper-tracked wrist camera. The static
    #: fallback only fires for scenes that *don't* declare the RoboSuite
    #: cameras (e.g. bare-Panda + custom MJCF without the agentview /
    #: eye_in_hand setup). Override either entry by passing
    #: ``cameras={"wrist_image": {"position": [...], ...}}`` to the constructor.
    LIBERO_CAMERAS: dict[str, dict[str, Any]] = {
        "image": {
            "position": [1.0, 0.0, 1.5],
            "target": [0.0, 0.0, 0.85],
            "fov": 60.0,
            "width": 256,
            "height": 256,
        },
        "wrist_image": {
            "position": [0.0, 0.0, 1.4],
            "target": [0.0, 0.0, 0.85],
            "fov": 60.0,
            "width": 256,
            "height": 256,
        },
    }

    def __init__(
        self,
        problem: BDDLProblem,
        *,
        scene_path: str | None = None,
        max_steps: int | None = None,
        init_jitter: float = 0.0,
        install_cameras: bool = True,
        cameras: dict[str, dict[str, Any]] | None = None,
        eef_body_name: str | None = None,
        eef_state_site_name: str | None = None,
        gripper_joint_name: str | None = None,
        state_gripper_joint_names: list[str] | None = None,
        inject_eef_state: bool = True,
        auto_generate_scene: bool = True,
        scene_cache_dir: str | None = None,
        scene_camera_aliases: dict[str, str] | None = None,
        apply_scene_keyframe: bool = True,
        scene_keyframe_index: int = 0,
        scene_robot_prefix: str = "robot0_",
        scene_gripper_prefix: str = "gripper0_",
        init_states: np.ndarray | None = None,
        bddl_source: str | None = None,
        bddl_path: str | None = None,
    ):
        """Construct from a pre-parsed :class:`BDDLProblem`.

        Args:
            problem: Parsed BDDL problem with a non-``None`` ``goal``.
            scene_path: Optional MJCF to ``sim.load_scene()`` on each
                episode start. ``None`` triggers ``auto_generate_scene``
                if enabled (see below).
            max_steps: Override the class-level 300.
            init_jitter: Per-episode ±jitter (metres) applied to xy of every
                object referenced by ``(:init (on A B))`` clauses. Default
                ``0.0`` matches LIBERO's deterministic-reset convention -
                the upstream training data is generated with fixed init
                states per ``(task, seed)`` and the GR00T-LIBERO checkpoint
                expects to see those exact poses (#166). Pass a positive
                value (e.g. ``0.02``) to layer per-episode randomization on
                top of the canonical state - useful for evaluating
                *generalization*, but expect lower nominal success rates
                because the policy is operating slightly out-of-distribution.
            install_cameras: When ``True`` (default), install the cameras
                in :attr:`LIBERO_CAMERAS` (or ``cameras`` override) on
                episode start. Set to ``False`` if your scene MJCF already
                declares the cameras the policy needs - the adapter will
                skip the install step entirely. Generated scenes have their
                cameras renamed to ``image`` / ``wrist_image`` (see
                ``scene_camera_aliases``), so the install step naturally
                no-ops on auto-generated scenes.
            cameras: Override / extend :attr:`LIBERO_CAMERAS`. Keyed by
                camera name, each value is forwarded as ``**kwargs`` to
                :meth:`Simulation.add_camera`. Passing an empty dict
                disables camera installation regardless of
                ``install_cameras``.
            eef_body_name: MuJoCo body name whose pose is read for the
                LIBERO ``state.x/y/z/roll/pitch/yaw`` keys *as a fallback*
                when ``eef_state_site_name`` is unset or doesn't resolve.
                ``None`` (default) triggers auto-resolution from the scene
                at episode start: when :meth:`_register_default_robot`
                discovers a scene-supplied Panda under
                ``scene_robot_prefix``, the adapter searches for the
                canonical RoboSuite EEF body (``<prefix>right_hand`` ->
                ``<prefix>hand`` -> bare ``hand``) and overrides
                ``_eef_body_name`` accordingly. Pass an explicit string
                to disable auto-resolution (useful for non-RoboSuite
                scenes); the legacy bare-Panda default is ``"hand"``.
            eef_state_site_name: MuJoCo *site* name whose pose is read
                for the LIBERO ``state.x/y/z/roll/pitch/yaw`` keys.
                ``None`` (default) auto-resolves to
                ``"<scene_gripper_prefix>grip_site"`` (i.e.
                ``"gripper0_grip_site"`` for LIBERO scenes), which
                matches what RoboSuite's ``OperationalSpaceController``
                reads for ``robot0_eef_pos`` / ``robot0_eef_quat`` —
                the site is at the gripper tip, ~9.7 cm BELOW the
                wrist body and rotated 180° around X to point fingers
                forward. Reading from the *body* (the #168 default)
                feeds GR00T state observations from the wrong point in
                the kinematic chain, which manifested as ``success_rate
                = 0`` across rounds 23-30 of #168 because the policy
                saw out-of-distribution state and emitted near-zero
                deltas. The site is preferred; the body is used as a
                fallback when the site doesn't exist (non-RoboSuite
                scenes).
            gripper_joint_name: Joint name whose ``qpos`` is read for the
                LIBERO ``state.gripper`` key. ``None`` (default) triggers
                auto-resolution from the scene at episode start using
                the RoboSuite gripper-namespace convention: search for
                ``<scene_gripper_prefix>finger_joint1`` (e.g.
                ``gripper0_finger_joint1``) -> ``<scene_robot_prefix>finger_joint1``
                -> bare ``finger_joint1``. Pass an explicit string to
                disable auto-resolution; the legacy bare-Panda default
                is ``"finger_joint1"``. The Menagerie Panda's two-finger
                MJCF equality constraint mirrors the value to the second
                finger, so reading just one is sufficient.
            inject_eef_state: When ``True`` (default), the adapter's
                :meth:`augment_observation` injects ``x`` / ``y`` / ``z``
                / ``roll`` / ``pitch`` / ``yaw`` / ``gripper`` keys
                into the per-step observation so the ``libero_panda``
                ``Gr00tDataConfig`` finds them. Set to ``False`` when the
                sim already exposes those keys (e.g. via a custom
                ``observation_mapping`` on the policy or a backend that
                returns Cartesian state natively).
            auto_generate_scene: When ``True`` (default) AND ``scene_path``
                is ``None``, :meth:`on_episode_start` calls
                :meth:`_generate_scene_from_bddl` to build the scene MJCF
                via the upstream ``libero`` package's procedural
                generator. The generated XML is cached on disk so
                subsequent episodes / processes reuse it without
                re-running ``libero``. Set to ``False`` to keep the
                pre-#164 behaviour of running against a bare Panda when
                no ``scene_path`` is provided.
            scene_cache_dir: Filesystem location for the generated-scene
                cache. Defaults to ``$STRANDS_BASE_DIR/scene_cache/libero/``
                (typically ``~/.strands_robots/scene_cache/libero/``).
                Cache key is SHA256 of the BDDL source so two adapters
                built from the same BDDL share a cached XML.
            scene_camera_aliases: Mapping from MJCF camera name (as
                emitted by LIBERO / RoboSuite) to the policy-side
                observation key expected by the ``libero_panda``
                data_config. Default
                ``{"agentview": "image", "robot0_eye_in_hand": "wrist_image",
                "robot0_eye_in_hand_image": "wrist_image"}`` renames the
                two canonical RoboSuite cameras so
                ``Gr00tPolicy._build_service_observation`` finds them by
                bare-key lookup. Both ``robot0_eye_in_hand`` and the
                ``_image``-suffixed variant are mapped because RoboSuite's
                emitted MJCFs use the bare name on the ``<camera>`` element
                while older convention adds the ``_image`` suffix - this way
                the rename works regardless of upstream version. Pass an
                empty dict to disable renaming (the static fallbacks in
                :attr:`LIBERO_CAMERAS` will then fire because no scene
                camera matches the policy-side ``image`` / ``wrist_image``
                names; the wrist channel becomes a static top-down view
                which puts GR00T-LIBERO out-of-distribution every step).
                When this map is non-empty, its sorted contents are
                hashed into the scene-cache key so a regenerated cache
                automatically picks up alias changes (e.g. a user adding
                a new alias) instead of serving a stale rewrite.
            apply_scene_keyframe: When ``True`` (default) AND a scene was
                loaded, :meth:`on_episode_start` restores qpos/qvel to the
                scene's canonical home state AFTER ``super().on_episode_start``
                and any camera install. Two branches:

                * **Preferred** — when ``model.nkey > 0`` (MJCF declares a
                  ``<keyframe>``, which LIBERO-authored hand-written scenes
                  do): calls ``mujoco.mj_resetDataKeyframe(model, data,
                  scene_keyframe_index)``.
                * **Fallback** — when ``model.nkey == 0`` (MJCFs from the
                  procedural :meth:`_generate_scene_from_bddl` path don't
                  carry a keyframe): snapshot-and-restore. The first
                  episode after a scene compile captures
                  ``data.qpos.copy()`` / ``data.qvel.copy()``; every
                  subsequent episode does ``np.copyto(data.qpos,
                  snapshot.qpos)`` + ``mj_forward`` so derived state
                  reflects the canonical pose. This is the actual fix for
                  #166's ``success_rate=0.00`` symptom on the codepath
                  ``examples/libero_mujoco.py`` exercises.

                The two branches produce equivalent observable state, so
                tests that pin one work for the other; see
                ``TestApplyCanonicalState``. Set to ``False`` to disable
                both branches (useful for diagnostic comparisons against
                the pre-fix behaviour).
            scene_keyframe_index: Which ``<keyframe>`` to apply when
                the keyframe branch is taken (``model.nkey > 0``). Defaults
                to ``0`` (first keyframe), which is the LIBERO convention.
                Pass a different index to select a non-default home pose.
                Ignored when the snapshot fallback fires.
            scene_robot_prefix: Body / joint / actuator name prefix that
                identifies the scene-supplied Panda when the adapter
                pre-registers it in ``world.robots`` (#166
                fix). Default ``"robot0_"`` matches RoboSuite / LIBERO's
                canonical naming for the upstream MJCFs (both
                hand-authored and procedurally-generated). Set to ``""``
                or change to a different prefix when working with a
                custom scene that names its Panda differently. The
                pre-register step no-ops silently when no body matches
                the prefix - super() then falls back to its standard
                ``add_robot`` path.
            scene_gripper_prefix: Body / joint name prefix that
                identifies the scene-supplied gripper. Default
                ``"gripper0_"`` matches RoboSuite's gripper namespace
                (separate from ``scene_robot_prefix`` because RoboSuite
                attaches grippers via its own naming scheme). Used by
                the gripper-joint auto-resolver in
                :meth:`_register_default_robot` when
                ``gripper_joint_name=None`` (default). Ignored when an
                explicit ``gripper_joint_name`` is supplied.
            init_states: Optional ``ndarray[(N, 1+nq+nv)]`` of LIBERO's
                canonical training-distribution init states for this
                task. ``None`` (default) falls through to the keyframe
                / snapshot-restore branches in
                :meth:`_apply_canonical_state`. When supplied,
                :meth:`_apply_init_state` picks one row per episode
                (RNG-seeded so a given seed re-runs the same init
                across re-evaluations) and writes
                ``data.time / data.qpos / data.qvel`` directly. The
                width of each row MUST equal ``1 + model.nq +
                model.nv`` - mismatches indicate the procedurally-
                generated MJCF diverges from upstream LIBERO's scene
                MJCF (e.g. missing ``(:objects ...)`` declarations) and
                are raised loudly rather than silently sliced. The
                array is cached as a member; populate via
                :func:`load_libero_suite` (which lazy-imports
                ``libero.libero.benchmark`` and calls
                ``ts.get_task_init_states(task_id)``) or pass
                explicitly. Without this kwarg the robot starts at
                ``qpos=0`` (the joint-default "stretched flat" pose)
                instead of the canonical "ready" pose GR00T-LIBERO
                expects, which alone drives ``success_rate=0`` (#168 bug I).
            bddl_source: Original BDDL text - stored on the adapter so
                the scene generator can pass it back to ``libero`` (which
                only accepts a *file* path). Set automatically by
                :meth:`from_text`. Tests may set it explicitly when
                building from a pre-parsed :class:`BDDLProblem` and they
                want auto-generation to work.
            bddl_path: Original BDDL file path - same purpose as
                ``bddl_source`` but lets the scene generator skip the
                temp-file step. Set automatically by :meth:`from_file`.

        Raises:
            ValueError: If ``problem.goal`` is ``None``.
        """
        if problem.goal is None:
            raise ValueError(f"LiberoAdapter: BDDL problem {problem.name!r} has no (:goal ...) block")
        self.problem = problem
        self.scene_path = scene_path
        self._init_jitter = float(init_jitter)
        if self._init_jitter < 0:
            raise ValueError(f"init_jitter must be >= 0, got {init_jitter}")
        if max_steps is not None:
            self.max_steps = int(max_steps)
        self._install_cameras = bool(install_cameras)
        # Snapshot the camera config at construction time so subsequent
        # mutations to LIBERO_CAMERAS don't leak across instances.
        self._cameras: dict[str, dict[str, Any]] = (
            {k: dict(v) for k, v in cameras.items()}
            if cameras is not None
            else {k: dict(v) for k, v in self.LIBERO_CAMERAS.items()}
        )
        self._eef_body_name: str = str(eef_body_name) if eef_body_name is not None else "hand"
        self._gripper_joint_name: str = str(gripper_joint_name) if gripper_joint_name is not None else "finger_joint1"
        # EEF state site (#168). The state observations fed to
        # GR00T (state.x/y/z/roll/pitch/yaw) must come from the gripper
        # *tip* — the same site that RoboSuite's
        # ``OperationalSpaceController`` reads for its
        # ``robot0_eef_pos`` / ``robot0_eef_quat`` observables. Reading
        # from the wrist *body* (the #168 default) is ~9.7 cm above
        # the tip and rotated 180° around X, which fed GR00T
        # out-of-distribution state across rounds 23-30 of #168.
        # Auto-default to ``"<scene_gripper_prefix>grip_site"`` (i.e.
        # ``"gripper0_grip_site"`` for LIBERO scenes); user override
        # disables auto-derivation. Empty-string sentinel is treated as
        # "no site available", forcing the body fallback (used by tests
        # that need to exercise the legacy body path).
        self._user_eef_state_site_name: str | None = (
            str(eef_state_site_name) if eef_state_site_name is not None else None
        )
        # Track whether the user explicitly supplied either name so the
        # auto-resolver in :meth:`_register_default_robot` only overrides
        # when the constructor default (``None``) was used. Explicit
        # values - including the legacy bare-Panda strings ``"hand"`` and
        # ``"finger_joint1"`` - are treated as "user knows best, do not
        # touch", which preserves backwards-compat for custom scene users.
        self._user_eef_body_name: str | None = str(eef_body_name) if eef_body_name is not None else None
        self._user_gripper_joint_name: str | None = str(gripper_joint_name) if gripper_joint_name is not None else None
        # State-side gripper joint names (#168). LIBERO trains
        # ``state.gripper`` on a 2-element vector ``[finger1.qpos, finger2.qpos]``
        # — the two fingers have OPPOSITE-sign qpos by physical
        # convention (they move apart). Pre-#168 we read ONE
        # finger from ``obs[gripper_joint_name]`` and packed it as
        # ``[v, v]`` (both positive), which is structurally
        # out-of-distribution for GR00T-LIBERO and produced the
        # near-zero deltas observed across earlier iterations. The current implementation
        # reads both finger qpos directly from ``data.qpos[jnt_qposadr]``.
        # Default auto-derives ``["<gripper_prefix>finger_joint1",
        # "<gripper_prefix>finger_joint2"]`` (i.e.
        # ``["gripper0_finger_joint1", "gripper0_finger_joint2"]`` for
        # LIBERO scenes). User override (a list of joint names) is used
        # as-is for non-RoboSuite gripper layouts.
        self._user_state_gripper_joint_names: list[str] | None = (
            [str(n) for n in state_gripper_joint_names] if state_gripper_joint_names is not None else None
        )
        self._inject_eef_state = bool(inject_eef_state)
        self._auto_generate_scene = bool(auto_generate_scene)
        self._scene_cache_dir = scene_cache_dir
        # Default camera-name alias map matches RoboSuite/LIBERO's two
        # canonical camera names to the bare keys (``image`` /
        # ``wrist_image``) that ``libero_panda``'s Gr00tDataConfig
        # expects. Both ``robot0_eye_in_hand`` (the bare name RoboSuite
        # emits in its compiled MJCFs) and the older ``_image``-suffixed
        # variant are mapped so the rename works regardless of which
        # upstream version produced the scene XML. Passing an empty dict
        # disables renaming.
        self._scene_camera_aliases: dict[str, str] = (
            dict(scene_camera_aliases)
            if scene_camera_aliases is not None
            else {
                "agentview": "image",
                "robot0_eye_in_hand": "wrist_image",
                "robot0_eye_in_hand_image": "wrist_image",
            }
        )
        self._apply_canonical_state_enabled = bool(apply_scene_keyframe)
        self._scene_keyframe_index = int(scene_keyframe_index)
        self._scene_robot_prefix = str(scene_robot_prefix)
        self._scene_gripper_prefix = str(scene_gripper_prefix)
        # LIBERO's canonical training-distribution init states (#168,
        # bug I). When non-None this takes precedence over the keyframe and
        # snapshot-restore branches in _apply_canonical_state. Stored as a
        # 2D ``(N, 1+nq+nv)`` array; per-episode selection is RNG-seeded so
        # a given seed re-runs the same init across re-evaluations.
        if init_states is not None:
            init_states_array = np.asarray(init_states, dtype=np.float64)
            if init_states_array.ndim == 1:
                # Single state - promote to 2D for uniform indexing.
                init_states_array = init_states_array[np.newaxis, :]
            if init_states_array.ndim != 2:
                raise ValueError(f"init_states must be 1D or 2D ndarray, got ndim={init_states_array.ndim}")
            self._init_states: np.ndarray | None = init_states_array
        else:
            self._init_states = None
        # Episode counter for deterministic init-state selection on
        # episode 0. Matches v0.1.1 ``env_libero.py``'s pattern of
        # ``env.set_init_state(init_states[0])`` for the first
        # episode and per-episode RNG-sampled init states for
        # episodes 1+. Pinning episode 0 to idx 0 makes
        # :meth:`prewarm`'s init-state apply (which always uses
        # idx 0) match the policy's actual starting state for
        # episode 0 - the recorder's t=0.00 frame and the policy's
        # first observation are then visually identical (#168 bug D-residual).
        self._episode_count: int = 0
        # Snapshot-and-restore fallback for procedurally-generated MJCFs that
        # don't ship a <keyframe> (the case the post-#168 verification
        # exposed). Captured on the first episode after super() +
        # _install_libero_cameras have run; replayed on every subsequent
        # episode so qpos/qvel land on the same canonical state every time.
        self._canonical_qpos: np.ndarray | None = None
        self._canonical_qvel: np.ndarray | None = None
        self._bddl_source = bddl_source
        self._bddl_path = bddl_path
        self._success_fn: Callable[[SimEngine], bool] = compile_goal(problem.goal)

        # #168 — diagnostic logging gate for the STATE side
        # of the policy interface. Set ``STRANDS_LIBERO_STATE_LOG=1`` to
        # emit one structured INFO log line per ``augment_observation``
        # call for the first ``STRANDS_LIBERO_STATE_LOG_MAX`` (default
        # 50) calls per episode. Pairs with ``STRANDS_LIBERO_ACTION_LOG``
        # (#168) so a single eval run captures both sides of the
        # policy interface for offline bisection.
        #
        # #168 verification showed ``arm_qpos`` advances and
        # ``eef_pos`` tracks deltas, but the policy is commanding tiny
        # deltas (~0.01 in [-1, +1] normalized space). The remaining
        # `success_rate=0` is on the state-input side — GR00T sees an
        # observation that says "EEF is already where it needs to be"
        # and emits near-zero deltas. STATE_LOG dumps exactly what we
        # feed GR00T so we can compare against LIBERO's
        # ``OffScreenRenderEnv`` ground truth at the same canonical
        # init pose.
        self._state_log_enabled = os.environ.get("STRANDS_LIBERO_STATE_LOG", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        try:
            self._state_log_max = int(os.environ.get("STRANDS_LIBERO_STATE_LOG_MAX", "50"))
        except ValueError:
            logger.warning(
                "STRANDS_LIBERO_STATE_LOG_MAX=%r is not an integer; defaulting to 50",
                os.environ.get("STRANDS_LIBERO_STATE_LOG_MAX"),
            )
            self._state_log_max = 50
        self._state_log_step: int = 0

    # Construction helpers

    @classmethod
    def from_file(
        cls,
        bddl_path: str | Path,
        *,
        scene_path: str | None = None,
        max_steps: int | None = None,
        init_jitter: float = 0.0,
        install_cameras: bool = True,
        cameras: dict[str, dict[str, Any]] | None = None,
        eef_body_name: str | None = None,
        eef_state_site_name: str | None = None,
        gripper_joint_name: str | None = None,
        state_gripper_joint_names: list[str] | None = None,
        inject_eef_state: bool = True,
        auto_generate_scene: bool = True,
        scene_cache_dir: str | None = None,
        scene_camera_aliases: dict[str, str] | None = None,
        apply_scene_keyframe: bool = True,
        scene_keyframe_index: int = 0,
        scene_robot_prefix: str = "robot0_",
        scene_gripper_prefix: str = "gripper0_",
        init_states: np.ndarray | None = None,
    ) -> LiberoAdapter:
        """Parse a ``.bddl`` file from disk and build an adapter.

        Raises :class:`FileNotFoundError` / :class:`BDDLParseError` on bad
        input - callers that want structured error dicts should catch and
        convert.
        """
        problem = parse_bddl_file(bddl_path)
        return cls(
            problem,
            scene_path=scene_path,
            max_steps=max_steps,
            init_jitter=init_jitter,
            install_cameras=install_cameras,
            cameras=cameras,
            eef_body_name=eef_body_name,
            eef_state_site_name=eef_state_site_name,
            gripper_joint_name=gripper_joint_name,
            state_gripper_joint_names=state_gripper_joint_names,
            inject_eef_state=inject_eef_state,
            auto_generate_scene=auto_generate_scene,
            scene_cache_dir=scene_cache_dir,
            scene_camera_aliases=scene_camera_aliases,
            apply_scene_keyframe=apply_scene_keyframe,
            scene_keyframe_index=scene_keyframe_index,
            scene_robot_prefix=scene_robot_prefix,
            scene_gripper_prefix=scene_gripper_prefix,
            init_states=init_states,
            bddl_path=str(bddl_path),
        )

    @classmethod
    def from_text(
        cls,
        bddl_text: str,
        *,
        scene_path: str | None = None,
        max_steps: int | None = None,
        init_jitter: float = 0.0,
        install_cameras: bool = True,
        cameras: dict[str, dict[str, Any]] | None = None,
        eef_body_name: str | None = None,
        eef_state_site_name: str | None = None,
        gripper_joint_name: str | None = None,
        state_gripper_joint_names: list[str] | None = None,
        inject_eef_state: bool = True,
        auto_generate_scene: bool = True,
        scene_cache_dir: str | None = None,
        scene_camera_aliases: dict[str, str] | None = None,
        apply_scene_keyframe: bool = True,
        scene_keyframe_index: int = 0,
        scene_robot_prefix: str = "robot0_",
        scene_gripper_prefix: str = "gripper0_",
        init_states: np.ndarray | None = None,
    ) -> LiberoAdapter:
        """Parse a BDDL string directly - useful in tests."""
        problem = parse_bddl(bddl_text)
        return cls(
            problem,
            scene_path=scene_path,
            max_steps=max_steps,
            init_jitter=init_jitter,
            install_cameras=install_cameras,
            cameras=cameras,
            eef_body_name=eef_body_name,
            eef_state_site_name=eef_state_site_name,
            gripper_joint_name=gripper_joint_name,
            state_gripper_joint_names=state_gripper_joint_names,
            inject_eef_state=inject_eef_state,
            auto_generate_scene=auto_generate_scene,
            scene_cache_dir=scene_cache_dir,
            scene_camera_aliases=scene_camera_aliases,
            apply_scene_keyframe=apply_scene_keyframe,
            scene_keyframe_index=scene_keyframe_index,
            scene_robot_prefix=scene_robot_prefix,
            scene_gripper_prefix=scene_gripper_prefix,
            init_states=init_states,
            bddl_source=bddl_text,
        )

    # BenchmarkProtocol interface

    @property
    def supported_robots(self) -> list[str]:
        return list(self.supported_robots_list)

    @property
    def default_robot(self) -> str:
        return self.default_robot_name

    @property
    def instruction(self) -> str:
        """Language instruction from the BDDL ``:language`` clause, or ``""``."""
        return self.problem.language or ""

    @property
    def eef_state_site_name(self) -> str:
        """Resolved MuJoCo site name to read for ``state.x/y/z/roll/pitch/yaw``.

        Returns the user-supplied ``eef_state_site_name`` constructor
        argument when set, otherwise auto-derives from
        ``scene_gripper_prefix + "grip_site"`` (i.e. defaults to
        ``"gripper0_grip_site"`` for LIBERO scenes).

        This is the site RoboSuite's ``OperationalSpaceController`` reads
        for its ``robot0_eef_pos`` / ``robot0_eef_quat`` observables (via
        ``data.site_xpos[eef_site_id]``); reading from the same site
        in :meth:`augment_observation` keeps the state observations
        we feed GR00T at the gripper TIP, matching the kinematic
        location LIBERO-trained checkpoints expect. #168.
        """
        if self._user_eef_state_site_name is not None:
            return self._user_eef_state_site_name
        return f"{self._scene_gripper_prefix}grip_site"

    @property
    def state_gripper_joint_names(self) -> list[str]:
        """Resolved 2-element list of finger joint names for ``state.gripper``.

        Returns the user-supplied ``state_gripper_joint_names``
        constructor argument when set, otherwise auto-derives from
        ``scene_gripper_prefix`` as
        ``["<prefix>finger_joint1", "<prefix>finger_joint2"]``
        (i.e. ``["gripper0_finger_joint1", "gripper0_finger_joint2"]``
        for LIBERO scenes).

        #168: LIBERO trains ``state.gripper`` on a 2-element
        vector ``[finger1.qpos, finger2.qpos]`` whose elements have
        OPPOSITE signs by physical convention (the two fingers move
        apart). Pre-#168 we read ONE finger and packed it as
        ``[v, v]`` (both positive), which fed GR00T structurally
        out-of-distribution state. The property returns the names in
        the order they appear in the trained state vector.
        """
        if self._user_state_gripper_joint_names is not None:
            return list(self._user_state_gripper_joint_names)
        return [f"{self._scene_gripper_prefix}finger_joint1", f"{self._scene_gripper_prefix}finger_joint2"]

    def on_episode_start(self, sim: SimEngine, rng: random.Random) -> None:
        """Auto-generate scene (if needed), load it, capture-or-restore
        canonical state, validate Panda, install cameras, then apply jitter.

        Order matters:

        1. **Scene resolution.** When ``scene_path`` is ``None`` and
           ``auto_generate_scene`` is true, build the scene MJCF from the
           BDDL via the upstream ``libero`` package's procedural generator
           and cache it on disk. Subsequent episodes / processes reuse the
           cached XML without re-running ``libero``.
        2. ``load_scene`` (if a path is now set) - so the base
           compatibility check sees the scene's Panda rather than reporting
           "sim is empty → load default_robot".
        3. **Canonical-state apply.** ``mj_makeData`` (in
           :meth:`Simulation.load_scene`) and ``mj_resetData`` (in
           :meth:`Simulation.reset`) both initialise qpos from the
           joint-default ``qpos0`` and **silently ignore MJCF
           ``<keyframe>`` blocks**. RoboSuite-emitted LIBERO scenes encode
           the canonical home pose in a ``<keyframe>`` (rare); the
           procedurally-generated MJCFs from #165 don't ship one
           (``model.nkey == 0``), so :meth:`_apply_canonical_state` falls
           back to snapshot-and-restore - capture qpos/qvel on the first
           episode, replay on subsequent ones. **Applied IMMEDIATELY after
           ``load_scene``** (before super / install_cameras) so the
           snapshot captures the post-load canonical state without any
           recompile-induced qpos drift; super() and install_cameras
           below then operate on top of canonical state.
        4. ``super().on_episode_start`` - base compat check + auto-load
           ``default_robot`` if the sim is empty. (Note: this *can*
           recompile the spec via ``add_robot``; MuJoCo's ``spec.recompile``
           preserves qpos for existing joints, so the canonical state we
           just restored survives.)
        5. ``_install_libero_cameras`` - inject the cameras the
           ``libero_panda`` ``Gr00tDataConfig`` expects (``image`` /
           ``wrist_image``). Detects scene-supplied cameras via the
           compiled model so the static-pose fallbacks only fire when the
           scene genuinely didn't provide them - this matters for not
           recompiling the spec on top of our just-restored canonical
           state (#166 review finding).
        6. ``_install_render_options`` - populate
           ``sim._world._backend_state["viz_option"]`` with an
           ``mjvOption`` matching upstream LIBERO's
           ``OffScreenRenderEnv`` viewer config (#168 bug E).
           The render path in ``simulation/mujoco/rendering.py`` reads
           that option and threads it to ``Renderer.update_scene(..., scene_option=...)``,
           hiding collision geoms / site markers / joint /
           actuator / COM widgets. Skipped on bare-Panda fallback
           (no scene loaded - default render options are appropriate
           for the user's own MJCF).
        7. ``_apply_init_jitter`` - per-episode RNG-seeded ±jitter to
           init-subject bodies, layered on top of canonical state.
        """
        if self.scene_path is None and self._auto_generate_scene:
            try:
                generated = self._generate_scene_from_bddl()
            except Exception as e:  # noqa: BLE001 - never abort eval on a setup-time error
                logger.warning(
                    "LiberoAdapter: scene auto-generation failed (%s); falling back to bare Panda. "
                    "Install the [benchmark-libero] extra (pip install 'strands-robots[benchmark-libero]') "
                    "or pass scene_path= explicitly to silence this warning.",
                    e,
                )
                generated = None
            if generated is not None:
                self.scene_path = generated

        scene_was_loaded = False
        # Detect "prewarm-fresh ep0": the example script called
        # prewarm() before start_cameras_recording, which loaded the
        # scene + applied init_states[0]. Re-running load_scene here
        # would reset MjData to qpos0 and open a race window where
        # the recorder thread captures gradient or qpos0 frames
        # before _apply_canonical_state restores init_states[0]
        # (#168 bug D-residual).
        #
        # On ep0 with prewarm-fresh state, skip both load_scene and
        # _apply_canonical_state - prewarm has already done both.
        # Bump _episode_count manually so ep1+ follows the normal
        # per-episode reload + RNG-sample lifecycle.
        #
        # Defensive sanity-check (#168): even when the
        # ``libero_prewarm_path`` flag is set and matches
        # ``self.scene_path``, verify that the current model size
        # still matches what prewarm worked on. If the model has
        # been mutated since prewarm ran (e.g. an unexpected
        # ``sim.add_robot`` call between prewarm and
        # evaluate_benchmark, which would weld in a redundant Panda
        # and change ``model.nq``), the flag is stale and the
        # fast-path would skip both load_scene AND the canonical
        # state restore - leaving the recorder to capture qpos0 of
        # a 2-Panda model. Fail loud at WARNING and fall through to
        # the normal lifecycle so on_episode_start can recover.
        backend_state = getattr(getattr(sim, "_world", None), "_backend_state", None)
        prewarm_path = backend_state.get("libero_prewarm_path") if isinstance(backend_state, dict) else None
        is_prewarm_fresh_ep0 = self._episode_count == 0 and self.scene_path and prewarm_path == self.scene_path

        # Sanity-check: if the flag is set but the model size doesn't
        # match init_states[0], the model has been mutated since
        # prewarm ran. Don't take the fast-path; let on_episode_start
        # do the full reload + canonical-state apply. WARNING-level
        # so users can detect their bad call ordering.
        if is_prewarm_fresh_ep0 and self._init_states is not None and self._init_states.shape[0] > 0:
            world = getattr(sim, "_world", None)
            model = getattr(world, "_model", None) if world is not None else None
            if model is not None:
                expected_width = 1 + int(getattr(model, "nq", 0)) + int(getattr(model, "nv", 0))
                actual_width = int(self._init_states[0].shape[0])
                if actual_width != expected_width:
                    logger.warning(
                        "LiberoAdapter.on_episode_start: prewarm-fresh flag is set but "
                        "model size mismatches init_states[0] (1+nq+nv=%d, init_states[0].shape=%d). "
                        "This usually means sim.add_robot or another model-mutating call ran "
                        "between prewarm() and evaluate_benchmark, recompiling the spec and "
                        "invalidating prewarm's setup. Falling through to normal lifecycle.",
                        expected_width,
                        actual_width,
                    )
                    is_prewarm_fresh_ep0 = False
                    if isinstance(backend_state, dict):
                        backend_state.pop("libero_prewarm_path", None)

        if is_prewarm_fresh_ep0:
            # Fast-path: prewarm already loaded the scene. Trust that
            # state; skip load_scene + _register_default_robot.
            #
            # IMPORTANT (#168 user-flagged fix): we
            # intentionally do NOT skip _apply_canonical_state here.
            # ``PolicyRunner._evaluate_with_spec`` calls ``sim.reset()``
            # between prewarm and on_episode_start, which resets
            # ``data.qpos`` to qpos0 - destroying prewarm's
            # init-state apply. The fast-path used to skip
            # _apply_canonical_state on the assumption that prewarm
            # left qpos populated; that's wrong because of the
            # reset(). The recorder thread captured ~6.5 s of qpos0
            # frames during ep1 before ep2's slow path finally
            # restored canonical state.
            #
            # Fix: keep the load_scene skip (avoids redundant spec
            # recompile) and the _register_default_robot skip
            # (idempotent; prewarm did it). But ALWAYS run
            # _apply_canonical_state - it re-applies the init state
            # after PolicyRunner's reset(). This costs nothing extra
            # vs the slow path's _apply_canonical_state call; the
            # only saved work in the fast-path is load_scene's
            # spec recompile.
            #
            # Don't bump _episode_count here either - let
            # _apply_canonical_state's _apply_init_state_branch do
            # it via the existing increment-after-apply logic. That
            # way ep0 still gets idx 0 (deterministic, matching
            # prewarm) and the counter advances naturally.
            logger.debug(
                "LiberoAdapter.on_episode_start: prewarm-fresh ep0 detected (path=%r); "
                "skipping load_scene + _register_default_robot "
                "(canonical-state apply still runs to restore qpos after PolicyRunner.sim.reset)",
                self.scene_path,
            )
            scene_was_loaded = True
            # Clear the flag so a subsequent fresh prewarm() (e.g. user
            # re-evaluates with a different scene) is detected fresh.
            if isinstance(backend_state, dict):
                backend_state.pop("libero_prewarm_path", None)
        elif self.scene_path:
            load_scene = getattr(sim, "load_scene", None)
            if load_scene is None:
                logger.warning(
                    "LiberoAdapter: sim has no load_scene(); skipping scene_path=%r",
                    self.scene_path,
                )
            else:
                result = load_scene(self.scene_path)
                if isinstance(result, dict) and result.get("status") == "error":
                    msg = (result.get("content") or [{}])[0].get("text", "")
                    raise RuntimeError(f"LiberoAdapter: load_scene({self.scene_path!r}) failed: {msg}")
                scene_was_loaded = True
        # Pre-register the default robot in world.robots BEFORE super()
        # runs. Otherwise super().on_episode_start (the base
        # BenchmarkProtocol) would see an empty list_robots() (because
        # Simulation.load_scene resets world.robots = {}) and call its
        # own sim.add_robot — which recompiles the spec, jumping
        # model.nq from N1 → N2 (#166 second-round verification: probe
        # showed 44 → 53 with a LIBERO scene). That recompile would
        # invalidate any qpos snapshot we then capture, since ep1's
        # snapshot would be at N2 but ep2's load_scene resets back to
        # N1 and add_robot fires again. By pre-registering here, super()
        # skips its add_robot and goes straight to the compatibility
        # check; subsequent episodes also pre-register so the snapshot
        # shape is stable across episodes.
        #
        # In the prewarm-fresh-ep0 path, _register_default_robot was
        # already called by prewarm; but it's idempotent (early-returns
        # if "robot" is already in world.robots), so calling again is
        # cheap and ensures the wrapper is registered even if a
        # downstream consumer somehow cleared it.
        if scene_was_loaded and not is_prewarm_fresh_ep0:
            self._register_default_robot(sim)
        # Apply canonical state RIGHT AFTER load_scene + pre-register so
        # the snapshot captures the post-load + post-add_robot state -
        # before super() and install_cameras get a chance to do anything
        # else (#166 review: snapshot taken at the wrong lifecycle point
        # was the prior round's failure mode).
        #
        # Always runs - including on the prewarm-fresh-ep0 fast-path
        # (#168 user-flagged fix). The fast-path used to skip
        # this on the assumption that prewarm had already applied
        # init_states[0]; but PolicyRunner._evaluate_with_spec calls
        # sim.reset() between prewarm and on_episode_start, which
        # wipes prewarm's qpos work. _apply_canonical_state restores
        # the init state after that reset. _apply_init_state_branch
        # itself handles the deterministic-vs-RNG selection via
        # self._episode_count, so calling it on ep0 still picks
        # init_states[0] (matching prewarm's choice).
        if scene_was_loaded and self._apply_canonical_state_enabled:
            self._apply_canonical_state(sim, rng)
        super().on_episode_start(sim, rng)
        if self._install_cameras:
            self._install_libero_cameras(sim)
        if scene_was_loaded:
            # Install render-time visualization options matching upstream
            # LIBERO's ``OffScreenRenderEnv`` viewer config (#168
            # bug E correction). Hides collision geoms, site / joint /
            # actuator / COM markers without modifying the cached MJCF.
            # See :meth:`_install_render_options` for the rationale.
            self._install_render_options(sim)
            # Install OSC_POSE controller so GR00T's task-space delta-EEF
            # actions ({x, y, z, roll, pitch, yaw, gripper}) get
            # converted into the LIBERO scene's torque-mode joint
            # actuators. Without this, _apply_sim_action silently
            # drops every action key (no name match) and the policy
            # effectively sends 0 torque (#168 bug). Best-
            # effort - if controller setup fails (missing robosuite,
            # missing site, etc.), log + continue; the eval will run
            # but actions will be no-ops, which is the same behaviour
            # as before this fix.
            self._install_action_controller(sim)
        if self._init_jitter > 0:
            self._apply_init_jitter(sim, rng)

        # #176 (sub-task 3d) — settle physics by one zero-action
        # control step so the first observation handed to the policy is
        # at "init_state + 1 OSC step", matching upstream LIBERO's
        # ``OffScreenRenderEnv.reset`` which does
        # ``set_init_state(...) + env.step(np.zeros(7))`` (see NVIDIA's
        # ``libero_env.py:257``). Without this settle, the MuJoCoSimEngine
        # path's first observation is at the raw ``init_state[i]`` while
        # upstream's first observation is one control step ahead — the
        # same training data was generated against the
        # post-step state, so feeding the raw init_state to a policy
        # trained on post-step inputs is OOD by a few mm / mrad.
        #
        # The downstream effect: the policy emits actions that are
        # ~2x off from what the offscreen path emits on identical
        # tasks, which (combined with the OSC torque divergence)
        # is enough to keep success_rate at 0 on libero-10.
        # After this fix, the first observation matches upstream within
        # numerical noise (qpos diff < 1e-9 verified empirically).
        #
        # Best-effort: gated on having a working OSC controller (which
        # owns_stepping). If the controller wasn't installed (warning
        # already logged in ``_install_action_controller``), skip the
        # settle to avoid raising — the eval will still run, just at
        # the un-settled init pose.
        if scene_was_loaded:
            world = getattr(sim, "_world", None)
            if world is not None:
                backend_state = getattr(world, "_backend_state", None)
                if isinstance(backend_state, dict) and "action_controller" in backend_state:
                    send_action = getattr(sim, "send_action", None)
                    if callable(send_action):
                        try:
                            send_action({})
                        except Exception as e:  # noqa: BLE001 - never abort eval on settle failure
                            logger.warning(
                                "LiberoAdapter.on_episode_start: settle send_action({}) raised %s; "
                                "first observation will be at raw init_state instead of init_state+1step",
                                e,
                            )

        # #168 — reset per-episode state-log counter so each
        # episode emits its own first N STATE_LOG lines (matches the
        # #168 behaviour for ACTION_LOG via the controller's
        # reset() call inside _install_action_controller above).
        self._state_log_step = 0

    def prewarm(self, sim: SimEngine) -> None:
        """Idempotent setup that should run BEFORE ``sim.start_cameras_recording``.

        Why this exists (#168 / bug D'): the recorder thread
        spawned by :meth:`Simulation.start_cameras_recording` captures
        its first frame *immediately*, before
        :meth:`evaluate_benchmark` (and therefore
        :meth:`on_episode_start`) runs. Without ``viz_option`` already
        in ``world._backend_state``, that first frame renders with
        MuJoCo's default visualization options - collision capsules,
        site markers (``gripper0_ft_frame`` red dot,
        ``gripper0_grip_site_cylinder`` green line), joint axes, and
        actuator widgets all visible. Subsequent frames are clean
        because :meth:`_install_render_options` runs as part of
        ``on_episode_start``.

        Earlier verification (#168) revealed a second, more severe
        race: the renderer returns a skybox-only gradient for any
        ``render(camera=...)`` call when ``data.xpos`` / ``data.xmat``
        haven't been populated yet. ``mujoco.MjData`` allocates these
        arrays at construction (e.g. inside ``Simulation.load_scene``)
        but doesn't compute them - they stay at zero / identity until
        ``mj_forward(model, data)`` runs. Until then,
        ``Renderer.update_scene`` finds the body transforms unset and
        falls back to the skybox-only readback (the gradient artifact).

        Concrete recorder timeline pre-fix:

        ::

            T0:  sim.load_scene(...)        # MjData allocated, NOT forwarded
            T0+: spec.prewarm(sim)          # registers robot + cameras + viz_option
            T1:  sim.start_cameras_recording  # recorder thread launches capture loop
            T1+: evaluate_benchmark starts
                 on_episode_start runs:
                   - load_scene (fresh MjData)
                   - _apply_canonical_state  # ← finally calls mj_forward here

            Meanwhile the recorder thread:
              capture iter 1: render(image)   # before main-thread mj_forward → gradient
              capture iter 1: render(wrist)   # may land after mj_forward → real
                                              #   (depending on race timing)

        The first capture frame for whichever camera renders before
        ``mj_forward`` returns gradient. The 2-pass warmup (#168)
        loop in ``_loop`` didn't help because warmup-of-any-depth
        on any thread can't conjure populated body transforms - only
        ``mj_forward`` does that.

        #168 fix: prewarm calls ``mj_forward`` itself. After
        prewarm, ``data.xpos`` / ``data.xmat`` are populated and the
        recorder's first render returns real geometry regardless of
        thread or camera order.

        This method exposes the *idempotent subset* of
        :meth:`on_episode_start` setup that should run before recording
        starts. Each call is no-op-on-prior-state safe:

        * :meth:`_register_default_robot` - early-returns if
          ``"robot"`` is already in ``world.robots``.
        * :meth:`_install_libero_cameras` - skips cameras already
          present in the model or registry.
        * :meth:`_install_render_options` - overwrites
          ``world._backend_state["viz_option"]`` with a freshly-built
          ``MjvOption``; semantically equivalent on every call.
        * ``mujoco.mj_forward(model, data)`` - populates derived
          state from ``qpos`` / ``qvel``. Safe to call multiple times;
          re-runs the same forward dynamics computation each call.

        Calling :meth:`prewarm` does NOT replace
        :meth:`on_episode_start` - the latter still runs the full
        per-episode lifecycle (canonical-state apply, init jitter,
        super-class compatibility check, etc.). Prewarm is just an
        early-rendering hint.

        Recommended call site (e.g. ``examples/libero_mujoco.py``)::

            sim.load_scene(spec.scene_path)        # scene-supplied Panda is in the loaded MJCF
            spec.prewarm(sim)                      # registers wrapper + viz_option + init_state[0] + mj_forward
            sim.start_cameras_recording(...)       # recorder's first frame is the canonical ready pose
            result = sim.evaluate_benchmark(...)

        IMPORTANT: do NOT call ``sim.add_robot("robot", data_config="panda")``
        between ``load_scene`` and ``prewarm`` for LIBERO scenes. The
        scene MJCF already contains the Panda; ``add_robot`` would weld
        a redundant Panda into the spec via spec recompile, bumping
        ``model.nq`` past what ``init_states[0]`` was sized for, and
        prewarm's init-state apply would silently no-op (logged at
        WARNING). This is bug-D-residual #168; the
        ``_register_default_robot`` step inside prewarm wraps the
        scene-supplied Panda automatically without needing a separate
        ``add_robot`` call.

        Assumes ``sim`` already has the scene loaded (via
        ``sim.load_scene``). Adapters built from a BDDL without a
        scene will see ``self.scene_path is None``; in that case
        prewarm is a no-op since there's nothing to install
        render-options against.

        Best-effort: any individual step's failure is caught and
        logged at WARNING - never aborts the whole prewarm because a
        failure here would just degrade rendering, not crash eval
        (the per-episode :meth:`on_episode_start` will retry).
        """
        if not self.scene_path:
            logger.debug(
                "LiberoAdapter.prewarm: scene_path is None; skipping (scene auto-generation defers to on_episode_start)"
            )
            return

        # Each step is independently idempotent - if the scene's Panda
        # is already wrapped, _register_default_robot early-returns;
        # if cameras are already in the model, _install_libero_cameras
        # skips them; viz_option overwrite is harmless.
        try:
            self._register_default_robot(sim)
        except Exception as e:  # noqa: BLE001 - never abort prewarm on a single-step failure
            logger.warning("LiberoAdapter.prewarm: _register_default_robot raised: %s", e)
        if self._install_cameras:
            try:
                self._install_libero_cameras(sim)
            except Exception as e:  # noqa: BLE001
                logger.warning("LiberoAdapter.prewarm: _install_libero_cameras raised: %s", e)
        try:
            self._install_render_options(sim)
        except Exception as e:  # noqa: BLE001
            logger.warning("LiberoAdapter.prewarm: _install_render_options raised: %s", e)

        # Install the OSC_POSE action controller so GR00T's task-space
        # delta-EEF actions translate to joint torques (#168
        # bug). Same idempotency / best-effort pattern as the other
        # prewarm steps.
        try:
            self._install_action_controller(sim)
        except Exception as e:  # noqa: BLE001
            logger.warning("LiberoAdapter.prewarm: _install_action_controller raised: %s", e)

        # Apply ``init_states[0]`` so the recorder's first frame
        # captures the canonical "ready" pose the policy will see at
        # episode 0 (#168 bug D-residual). Without this,
        # ``data.qpos`` stays at the joint-default zeros that
        # ``load_scene`` left behind, and the t=0.00 recorded frame
        # shows the Panda stretched flat with mugs at MJCF defaults
        # rather than LIBERO's canonical training-distribution start.
        # This pairs with the episode-0-pinned-to-idx-0 logic in
        # :meth:`_apply_init_state_branch` so prewarm + ep0
        # observation are visually identical.
        try:
            self._apply_init_state_for_prewarm(sim)
        except Exception as e:  # noqa: BLE001
            logger.warning("LiberoAdapter.prewarm: init-state apply failed: %s", e)

        # Forward the MjData so xpos/xmat are populated before the
        # recorder thread's first render call (#168 bug D
        # fix). Without this, every render between prewarm() and
        # on_episode_start's _apply_canonical_state returns the
        # skybox-only gradient because Renderer.update_scene finds
        # body transforms unset. #168 also moved this into
        # Simulation.load_scene as the engine-level fix; the prewarm
        # call here is now defense-in-depth, AND ensures ``mj_forward``
        # runs after the init-state apply above so derived state
        # (xpos/xmat) reflects the ready pose, not load_scene's qpos0.
        try:
            self._forward_mj_data(sim)
        except Exception as e:  # noqa: BLE001
            logger.warning("LiberoAdapter.prewarm: mj_forward failed: %s", e)

        # Force one main-thread render to prime any process-wide
        # GL state that the recorder thread inherits (#168
        # bug D defensive). The recorder thread spawned by
        # ``start_cameras_recording`` has its own
        # ``threading.local`` Renderer instance, but some driver-
        # level state (compiled shaders, texture caches, GLContext
        # bind state) is process-shared. The reviewer's variant-B
        # verification (#168) showed that without a main-thread
        # render after prewarm, the recorder thread's first ~15
        # render calls return GL clear-colour gradient even when
        # ``mj_forward`` has populated ``data.xpos / xmat``. A single
        # main-thread render on the same camera primes that shared
        # state so the recorder thread's first call lands warm.
        #
        # Rounds 11-13 attempted thread-side warmup loops, which
        # #168 verification showed don't help (GL context is
        # thread-bound and the per-thread Renderer is cold). The
        # main-thread approach here is different: it primes the
        # process-shared driver state, not the per-thread Renderer.
        # If the driver state assumption is wrong (no shared state),
        # this is a harmless ~33ms render that was never used.
        #
        # Best-effort: render failures (no GL context, missing
        # camera, etc.) are logged at DEBUG and don't abort prewarm.
        try:
            self._warmup_render(sim)
        except Exception as e:  # noqa: BLE001
            logger.debug("LiberoAdapter.prewarm: warmup render failed: %s", e)

    def _warmup_render(self, sim: SimEngine) -> None:
        """Force one synchronous render on the main thread to prime GL state.

        Picks the first registered camera (typically ``image`` for
        LIBERO scenes) and calls ``sim.render(camera_name=cam, ...)``
        once. Discards the result; only the GL state-priming side-effect
        matters.

        Best-effort: any failure (sim has no render(), no cameras
        installed, GL context unavailable) is logged at DEBUG and
        returns silently. The recorder thread's render path has its
        own error handling for persistent failures via
        ``state["errors"][cam]``.

        Camera selection: tries ``self._cameras`` keys in order
        (``image`` then ``wrist_image`` for default LIBERO config),
        falls back to ``"default"`` if the dict is empty. Only
        renders ONE camera - we just need to prime shared state, not
        warm every camera's per-thread renderer (the per-thread
        renderer is the recorder's responsibility).
        """
        render = getattr(sim, "render", None)
        if render is None:
            logger.debug("LiberoAdapter.prewarm: sim has no render(); skipping warmup")
            return
        # Pick the first declared camera; default fallback if none.
        cam_name = next(iter(self._cameras), "default") if self._cameras else "default"
        try:
            render(camera_name=cam_name, width=64, height=64)
        except Exception as e:  # noqa: BLE001 - warmup failures non-fatal
            logger.debug("LiberoAdapter.prewarm: warmup render(%r) failed: %s", cam_name, e)
            return
        logger.debug("LiberoAdapter.prewarm: warmup render(%r) primed GL state", cam_name)

    def _apply_init_state_for_prewarm(self, sim: SimEngine) -> None:
        """Write ``init_states[0]`` to ``world._data`` (best-effort).

        Mirrors :meth:`_apply_init_state_branch` but with
        ``strict=False`` semantics:

        * Width mismatch → DEBUG-log + skip (don't raise). Prewarm
          must not crash the eval pipeline; ``on_episode_start``
          will retry via ``_apply_canonical_state`` which has
          ``strict=True`` and will surface the same diagnostic.
        * No init_states → silent skip (the bare-Panda case
          where ``LiberoAdapter`` was constructed without
          ``init_states=``).
        * Missing mujoco / world / model / data → DEBUG-log + skip.

        Always uses ``init_states[0]``, matching the
        episode-0-deterministic contract in
        :meth:`_apply_init_state_branch`.

        Does NOT increment ``self._episode_count`` - prewarm runs
        BEFORE episode 0; episode 0's call into
        ``_apply_init_state_branch`` (via on_episode_start ->
        _apply_canonical_state) is what bumps the counter.
        """
        if self._init_states is None or self._init_states.shape[0] == 0:
            logger.debug("LiberoAdapter.prewarm: no init_states; skipping init-state apply")
            return

        # Probe whether mujoco is importable so we skip cleanly on
        # non-MuJoCo backends. The actual import is done by
        # _forward_mj_data right after this; here we don't call any
        # mujoco function directly. Using a try-import (vs
        # importlib.util.find_spec) so test fixtures can patch
        # sys.modules["mujoco"] with a stub - find_spec doesn't honour
        # sys.modules patches and would always see the real install.
        try:
            import mujoco  # noqa: F401 - probe-only, real use is in _forward_mj_data
        except ImportError:
            logger.debug("LiberoAdapter.prewarm: mujoco not importable; skipping init-state apply")
            return

        world = getattr(sim, "_world", None)
        if world is None:
            return
        model = getattr(world, "_model", None)
        data = getattr(world, "_data", None)
        if model is None or data is None:
            return

        nq = int(getattr(model, "nq", 0))
        nv = int(getattr(model, "nv", 0))
        if nq == 0 or nv == 0:
            return
        state = self._init_states[0]
        expected_width = 1 + nq + nv
        if state.shape[0] != expected_width:
            # Best-effort: log + skip. Don't raise like the
            # canonical-state branch does - prewarm is a hint, not a
            # hard contract. on_episode_start's _apply_init_state_branch
            # will surface the same width mismatch with strict=True.
            #
            # Logged at WARNING (not DEBUG) because this is almost
            # always the symptom of a bad call ordering: the example
            # script called ``sim.add_robot`` (or another spec-recompiling
            # operation) between ``sim.load_scene`` and ``spec.prewarm``,
            # welding a redundant robot into the spec. That recompile
            # bumps ``nq`` past what the LIBERO ``init_states[0]`` was
            # sized for, and prewarm silently no-ops here. Visible at
            # WARNING level, users can spot the mistake without enabling
            # debug logging (#168 verification).
            logger.warning(
                "LiberoAdapter.prewarm: init_state[0] width %d != 1+nq+nv=%d; skipping init-state apply. "
                "This usually means sim.add_robot (or another spec-recompiling call) ran between "
                "sim.load_scene and spec.prewarm. Recommended call order: load_scene -> prewarm -> "
                "start_cameras_recording -> evaluate_benchmark, with NO sim.add_robot between them.",
                state.shape[0],
                expected_width,
            )
            return

        lock = getattr(sim, "_lock", None)

        def _apply() -> None:
            data.time = float(state[0])
            np.copyto(data.qpos, state[1 : 1 + nq])
            np.copyto(data.qvel, state[1 + nq :])
            # mj_forward is called by _forward_mj_data right after
            # this returns; intentionally not duplicated here.

        if lock is not None:
            with lock:
                _apply()
        else:
            _apply()

        # Mark in backend_state that prewarm has applied init_state[0]
        # for this scene_path. on_episode_start checks this flag on
        # episode 0 and skips its own load_scene + _apply_canonical_state
        # to avoid re-resetting qpos to qpos0 in the race window before
        # the recorder thread captures its first frame (#168
        # bug D-residual). The flag is one-shot for ep0; on_episode_start
        # consumes it (sets ``_episode_count`` to 1 directly) so ep1+
        # follows the normal per-episode reload + RNG-sample lifecycle.
        backend_state = getattr(world, "_backend_state", None)
        if isinstance(backend_state, dict):
            backend_state["libero_prewarm_path"] = self.scene_path

        logger.debug("LiberoAdapter.prewarm: applied init_state[0] (qpos[:%d] + qvel[:%d])", nq, nv)

    def _forward_mj_data(self, sim: SimEngine) -> None:
        """Run ``mujoco.mj_forward(model, data)`` if the sim has both available.

        Best-effort: missing mujoco / world / model / data → debug-log
        + skip without raising. The only failure mode that can't
        degrade silently is mj_forward itself raising on inconsistent
        state, which is genuinely a sim-level bug worth surfacing -
        let it propagate to prewarm's catch-all.
        """
        try:
            import mujoco as _mj
        except ImportError:
            logger.debug("LiberoAdapter._forward_mj_data: mujoco not importable; skipping")
            return

        world = getattr(sim, "_world", None)
        if world is None:
            logger.debug("LiberoAdapter._forward_mj_data: sim has no _world; skipping")
            return
        model = getattr(world, "_model", None)
        data = getattr(world, "_data", None)
        if model is None or data is None:
            logger.debug("LiberoAdapter._forward_mj_data: world missing model/data; skipping")
            return

        # mj_forward populates xpos/xquat/xmat plus other derived state.
        # The lock matches the contract of other state-mutating helpers
        # in this adapter (e.g. _apply_init_state_branch).
        lock = getattr(sim, "_lock", None)
        if lock is not None:
            with lock:
                _mj.mj_forward(model, data)
        else:
            _mj.mj_forward(model, data)

    def on_step(
        self,
        sim: SimEngine,
        obs: dict[str, Any],
        action: dict[str, Any],
    ) -> StepInfo:
        """Sparse step: zero reward, never ``done``. Success is detected by
        :meth:`is_success` at the outer eval loop."""
        return StepInfo(reward=0.0, done=False)

    def augment_observation(
        self,
        sim: SimEngine,
        obs: dict[str, Any],
    ) -> dict[str, Any]:
        """Inject ``x`` / ``y`` / ``z`` / ``roll`` / ``pitch`` / ``yaw`` / ``gripper``
        for the ``libero_panda`` ``Gr00tDataConfig`` schema.

        The ``libero_panda`` data_config declares
        ``state_keys = ["state.x", "state.y", "state.z", "state.roll",
        "state.pitch", "state.yaw", "state.gripper"]``. The policy's
        ``_build_service_observation`` strips the ``state.`` prefix and
        looks up bare keys (``x``, ``y``, …) directly in the robot
        observation. ``Simulation.get_observation()`` only returns
        joint-space readings, so without this hook the server rejects
        every request with ``Server error: State key 'state.x' must be
        in observation``.

        **#168 fix (pose source).** Read EEF pose from the
        gripper *site* (``self.eef_state_site_name`` →
        ``"gripper0_grip_site"`` for LIBERO scenes) instead of the
        wrist *body* (``self._eef_body_name`` → ``"robot0_right_hand"``).
        These differ by ~9.7 cm in z and 180° rotation around X — the
        site is at the gripper TIP, the body is at the wrist. RoboSuite's
        ``OperationalSpaceController`` reads from the site for its own
        ``robot0_eef_pos`` / ``robot0_eef_quat`` observables, so reading
        from the same site here produces state observations matching
        what LIBERO checkpoints were trained against. Reading from the
        body fed GR00T out-of-distribution state, which manifested as
        ``success_rate = 0`` across rounds 23-30.

        Implementation:

        1. Try the SITE path first via direct
           ``mujoco.mj_name2id(model, mjtObj.mjOBJ_SITE, eef_state_site_name)``,
           reading ``data.site_xpos`` + ``data.site_xmat``.
        2. Fall back to the BODY path via
           ``sim.get_body_state(self._eef_body_name)`` for non-RoboSuite
           scenes that don't ship the canonical site (e.g. bare
           Menagerie Panda). The fallback path matches the pre-#168
           behaviour exactly.
        3. Convert the site/body's rotation matrix or quaternion to
           extrinsic XYZ Euler ``(roll, pitch, yaw)`` to match the
           LIBERO/RoboSuite ``mat2euler(..., axes='sxyz')`` convention
           the dataset and policy were trained on.
        4. Read gripper opening from ``obs[self._gripper_joint_name]``
           (already populated by ``Simulation.get_observation``).

        Best-effort: if any source is missing (sim doesn't expose
        ``get_body_state``, site absent and body absent, gripper joint
        absent), the corresponding key is omitted with a debug log. The
        original observation is returned with the resolved keys merged
        in - we never delete or overwrite an obs key the sim already
        provided (so a backend that natively returns Cartesian state
        wins).

        Disable this entirely with ``inject_eef_state=False`` on the
        constructor.
        """
        if not self._inject_eef_state:
            return obs

        merged = dict(obs)

        # End-effector pose. #168: try site lookup first
        # (matches RoboSuite's eef_pos / eef_quat semantics), fall
        # back to body lookup if the site doesn't exist.
        position, quat = self._read_eef_pose(sim)
        if position is not None:
            # Don't overwrite if a backend already supplied these
            # (e.g. via a custom mapping).
            merged.setdefault("x", float(position[0]))
            merged.setdefault("y", float(position[1]))
            merged.setdefault("z", float(position[2]))
        if quat is not None:
            roll, pitch, yaw = _quat_wxyz_to_rpy_xyz(quat)
            merged.setdefault("roll", roll)
            merged.setdefault("pitch", pitch)
            merged.setdefault("yaw", yaw)

        # Gripper — #168. LIBERO trains ``state.gripper`` on
        # ``robot0_gripper_qpos`` from LIBERO/RoboSuite, which is a
        # 2-element array
        # ``[gripper0_finger_joint1.qpos, gripper0_finger_joint2.qpos]``.
        # The two fingers have OPPOSITE-sign qpos by physical
        # convention (they move apart); typical at-rest values are
        # ``[+0.0208, -0.0208]``. Pre-#168 we read only one finger
        # via ``obs[self._gripper_joint_name]`` and packed it as
        # ``[v, v]`` (both positive), which fed GR00T structurally
        # out-of-distribution state — manifest as near-zero policy
        # deltas across rounds 23-32 of #168.
        #
        # Read both finger qpos directly from ``data.qpos[jnt_qposadr]``
        # using the canonical RoboSuite joint names. Falls back to the
        # legacy single-joint duplicate-packing for non-RoboSuite
        # scenes that don't ship two named finger joints.
        gripper_qpos = self._read_gripper_qpos(sim)
        if gripper_qpos is not None:
            merged.setdefault("gripper", gripper_qpos)
        else:
            # Legacy fallback — read one joint from obs and duplicate
            # (preserves pre-#168 behaviour for non-RoboSuite
            # scenes; the duplicate-packing is wrong for LIBERO but
            # there's no better default for unknown gripper layouts).
            gripper_value = obs.get(self._gripper_joint_name)
            if gripper_value is None:
                # Some backends namespace joint keys; try the suffix match.
                for key, val in obs.items():
                    if isinstance(key, str) and key.endswith("/" + self._gripper_joint_name):
                        gripper_value = val
                        break
            if isinstance(gripper_value, (int, float)) and not isinstance(gripper_value, bool):
                merged.setdefault("gripper", [float(gripper_value), float(gripper_value)])
            else:
                logger.debug(
                    "LiberoAdapter: gripper joints %s not found via direct mujoco lookup, "
                    "and obs key %r missing; omitting state.gripper",
                    self.state_gripper_joint_names,
                    self._gripper_joint_name,
                )

        # #168 — flip rendered images vertically to match
        # upstream LIBERO's ``OffScreenRenderEnv`` pixel convention.
        #
        # WHY: our ``sim.render()`` returns top-row-zero (image
        # convention), but upstream LIBERO's ``OffScreenRenderEnv``
        # returns bottom-row-zero (OpenGL framebuffer convention). The
        # GR00T-N1.7-LIBERO checkpoint was trained against upstream's
        # convention with an additional ``[::-1, ::-1]`` rotation
        # applied at training time by ``LiberoEnv._process_observation``
        # (mirrored in the inference path via the policy's
        # ``image_rotation_180`` flag).
        #
        # #168 ``tests_integ/.../diff_libero_obs.py`` measured the
        # delta: ``mean |Δ| = 56/255`` for raw vs raw, but
        # ``mean |Δ| = 5.40/255`` after applying ``[::-1, :]`` to ours
        # — a 10× reduction confirming the vertical-flip-only
        # mismatch (a horizontal mirror would have shown the opposite
        # asymmetry).
        #
        # WHAT: flip our renders vertically so they're in upstream
        # OffScreenRenderEnv convention. The policy's existing
        # ``image_rotation_180`` flag (which applies ``[::-1, ::-1]``)
        # then converts upstream-convention to training-image
        # convention as designed.
        #
        # Applied to BOTH ``image`` and ``wrist_image`` because both
        # come from our mujoco renderer with the same convention. We
        # use ``np.ascontiguousarray`` so downstream serialization
        # (msgpack / numpy.tobytes()) works — reversed views are not
        # contiguous.
        #
        # Best-effort: if the image isn't a numpy ndarray (e.g. a
        # backend supplied a PIL Image) or doesn't have at least 2
        # dimensions, skip silently and let the original value pass
        # through. This preserves backend flexibility.
        for cam_key in ("image", "wrist_image"):
            cam_value = merged.get(cam_key)
            if isinstance(cam_value, np.ndarray) and cam_value.ndim >= 2:
                merged[cam_key] = np.ascontiguousarray(cam_value[::-1, :])

        # #168 — emit one structured STATE_LOG line per
        # ``augment_observation`` call when STRANDS_LIBERO_STATE_LOG=1.
        # Captures the EXACT state values fed to GR00T's policy server,
        # for offline comparison against LIBERO's
        # ``OffScreenRenderEnv.observation_spec()`` ground truth at the
        # same canonical init pose. #168 ACTION_LOG showed the
        # OSC tracks tiny deltas correctly (95% of arm_ctrl is
        # gravity comp); the policy is *commanding* tiny deltas, which
        # points at state-input mismatch (units, frame, or magnitudes).
        if self._state_log_enabled and self._state_log_step < self._state_log_max:
            logger.info(
                "STATE_LOG step=%d x=%s y=%s z=%s roll=%s pitch=%s yaw=%s gripper=%s obs_keys=%s",
                self._state_log_step,
                _fmt_state_value(merged.get("x")),
                _fmt_state_value(merged.get("y")),
                _fmt_state_value(merged.get("z")),
                _fmt_state_value(merged.get("roll")),
                _fmt_state_value(merged.get("pitch")),
                _fmt_state_value(merged.get("yaw")),
                _fmt_state_value(merged.get("gripper")),
                sorted(merged.keys()),
            )
            self._state_log_step += 1

        return merged

    def _read_eef_pose(self, sim: SimEngine) -> tuple[list[float] | None, list[float] | None]:
        """Read EEF position + (wxyz) quaternion for ``augment_observation``.

        **#168 — split sources matching RoboSuite exactly.**
        RoboSuite's ``robosuite/robots/single_arm.py`` reads from TWO
        DIFFERENT POINTS in the kinematic chain::

            def eef_pos(obs_cache):
                return np.array(self.sim.data.site_xpos[self.eef_site_id])
                                                       # ↑ site (gripper tip)
            def eef_quat(obs_cache):
                return T.convert_quat(
                    self.sim.data.get_body_xquat(self.robot_model.eef_name),
                    to="xyzw",
                )                                      # ↑ body (wrist)

        The site sits at the gripper tip; the body sits at the wrist
        (~9.7 cm above the site). Their rotation matrices have a 90°
        offset around Z relative to each other — RoboSuite picks the
        body's orientation deliberately because that's what the
        downstream observable + dataset expects.

        #168 (initial diagnosis): we read both pos AND quat from the
        wrist body — pos was 71.8 mm off in z and orientation 180°
        off in roll. #168 (first attempt): moved BOTH to the site,
        which fixed position (within 5 mm) but introduced a 90° yaw
        offset because site_xmat ≠ body xquat. #168 (final fix):
        splits the reads the way RoboSuite does.

        Returns ``(pos, quat_wxyz)``, either or both of which may be
        ``None`` on failure (logged at DEBUG; caller selectively injects
        only the keys it has).
        """
        # 1. Direct-mujoco read: position from site, orientation from
        # body. This is the LIBERO/RoboSuite path. Both lookups can
        # independently succeed or fail; we mix the successful results
        # with the body-fallback for whichever was missing.
        world = getattr(sim, "_world", None)
        model = getattr(world, "_model", None) if world is not None else None
        data = getattr(world, "_data", None) if world is not None else None

        site_pos: list[float] | None = None
        body_quat: list[float] | None = None

        if model is not None and data is not None:
            try:
                import mujoco as _mj
            except ImportError as e:
                logger.debug("LiberoAdapter: mujoco import failed in _read_eef_pose: %s", e)
                _mj = None  # type: ignore[assignment]

            if _mj is not None:
                # 1a. SITE → position. Matches RoboSuite's
                # ``eef_pos`` observable.
                site_name = self.eef_state_site_name
                if site_name:
                    try:
                        site_id = int(_mj.mj_name2id(model, _mj.mjtObj.mjOBJ_SITE, site_name))
                    except (AttributeError, TypeError, ValueError) as e:
                        logger.debug("LiberoAdapter: mujoco site lookup failed for %r: %s", site_name, e)
                        site_id = -1
                    if site_id >= 0:
                        try:
                            pos_arr = np.asarray(data.site_xpos[site_id], dtype=np.float64)
                            site_pos = [float(c) for c in pos_arr]
                        except (AttributeError, IndexError, ValueError) as e:
                            logger.debug(
                                "LiberoAdapter: failed to read site %r position (site_id=%d): %s",
                                site_name,
                                site_id,
                                e,
                            )

                # 1b. BODY → orientation. Matches RoboSuite's
                # ``eef_quat`` observable: ``data.xquat[eef_body_id]``
                # (mujoco's xquat is wxyz, so no convert_quat needed
                # — our downstream ``_quat_wxyz_to_rpy_xyz`` expects
                # wxyz).
                body_name = self._eef_body_name
                if body_name:
                    try:
                        body_id = int(_mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, body_name))
                    except (AttributeError, TypeError, ValueError) as e:
                        logger.debug("LiberoAdapter: mujoco body lookup failed for %r: %s", body_name, e)
                        body_id = -1
                    if body_id >= 0:
                        try:
                            quat_arr = np.asarray(data.xquat[body_id], dtype=np.float64)
                            body_quat = [float(c) for c in quat_arr]
                        except (AttributeError, IndexError, ValueError) as e:
                            logger.debug(
                                "LiberoAdapter: failed to read body %r xquat (body_id=%d): %s",
                                body_name,
                                body_id,
                                e,
                            )

        # 2. If both were read directly, return the split-source pair
        # (this is the happy path on LIBERO scenes).
        if site_pos is not None and body_quat is not None:
            return (site_pos, body_quat)

        # 3. Body-state fallback for whichever direct read failed
        # (e.g. non-RoboSuite scenes that don't ship the canonical
        # site, or the ``hand``-named body for bare Menagerie Panda).
        # ``sim.get_body_state`` is namespace-aware (handles
        # ``panda_arm/hand`` for multi-robot scenes) and returns
        # ``(pos, quat_wxyz)`` — same shape we promise here.
        get_body_state = getattr(sim, "get_body_state", None)
        fallback_pos: list[float] | None = None
        fallback_quat: list[float] | None = None
        if get_body_state is not None:
            try:
                state_result = get_body_state(body_name=self._eef_body_name)
            except Exception as e:  # noqa: BLE001 - never abort eval on a state lookup
                logger.debug("LiberoAdapter: get_body_state(%r) raised: %s", self._eef_body_name, e)
                state_result = None
            fallback_pos, fallback_quat = _extract_pose(state_result)
        else:
            logger.debug("LiberoAdapter: sim has no get_body_state(); skipping EEF state injection")

        # 4. Mix direct and fallback reads — each axis populated by
        # whichever source succeeded first. Site/body get top
        # priority (canonical RoboSuite split); fallback fills gaps.
        merged_pos = site_pos if site_pos is not None else fallback_pos
        merged_quat = body_quat if body_quat is not None else fallback_quat
        return (merged_pos, merged_quat)

    def _read_gripper_qpos(self, sim: SimEngine) -> list[float] | None:
        """Read both finger qpos for the LIBERO ``state.gripper`` 2-vector.

        #168. Returns
        ``[finger1.qpos, finger2.qpos]`` read directly from
        ``data.qpos[jnt_qposadr]`` for the joint names returned by
        :attr:`state_gripper_joint_names` (default
        ``["gripper0_finger_joint1", "gripper0_finger_joint2"]``).

        LIBERO's training data records ``state.gripper`` as
        ``robot0_gripper_qpos = [qpos[7], qpos[8]]`` for the two
        finger joints; their values have OPPOSITE signs by physical
        convention (the Panda gripper's two-finger MJCF puts each
        finger on its own joint with mirrored ranges, e.g.
        ``[0, +0.04]`` and ``[-0.04, 0]``). Pre-#168 the adapter
        read ONE finger's value and packed it as ``[v, v]`` (both
        positive), which is structurally OOD from training.

        Returns ``None`` (and the caller falls back to the legacy
        single-joint duplicate path) if any of:
        - ``sim._world._model`` / ``data`` are unavailable
        - mujoco isn't importable
        - any of the configured joint names doesn't resolve in the
          compiled model

        ``None`` rather than partial data so the caller's fallback
        logic stays simple — there's no half-good case worth
        propagating.
        """
        world = getattr(sim, "_world", None)
        model = getattr(world, "_model", None) if world is not None else None
        data = getattr(world, "_data", None) if world is not None else None
        if model is None or data is None:
            return None
        try:
            import mujoco as _mj
        except ImportError as e:
            logger.debug("LiberoAdapter: mujoco import failed in _read_gripper_qpos: %s", e)
            return None

        joint_names = self.state_gripper_joint_names
        if not joint_names:
            return None

        finger_qposes: list[float] = []
        for jname in joint_names:
            try:
                jid = int(_mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, jname))
            except (AttributeError, TypeError, ValueError) as e:
                logger.debug("LiberoAdapter: mujoco joint lookup failed for %r: %s", jname, e)
                return None
            if jid < 0:
                logger.debug(
                    "LiberoAdapter: state gripper joint %r not in compiled model (jid=%d)",
                    jname,
                    jid,
                )
                return None
            try:
                qposadr = int(model.jnt_qposadr[jid])
                finger_qposes.append(float(data.qpos[qposadr]))
            except (AttributeError, IndexError, ValueError) as e:
                logger.debug(
                    "LiberoAdapter: failed to read qpos for joint %r (jid=%d): %s",
                    jname,
                    jid,
                    e,
                )
                return None
        return finger_qposes

    def is_success(self, sim: SimEngine) -> bool:
        """Check whether the LIBERO task goal is satisfied.

        Walks the BDDL predicate tree compiled at construction time
        (:func:`compile_goal`) against the current sim state. The
        evaluator was hardened in #170 / #173 / #175 to match
        upstream LIBERO's ``check_ontop`` / ``check_contact``
        semantics byte-for-byte at the moment of contact (#171
        sub-task 3e contact check, #175 round 46d body-name
        ``_main`` suffix fallback).
        """
        return bool(self._success_fn(sim))

    # Internals

    def _generate_scene_from_bddl(self) -> str | None:
        """Build the LIBERO scene MJCF from the BDDL via the upstream ``libero`` package.

        Returns the absolute path to a cached MJCF file, or ``None`` when
        the BDDL source isn't recoverable (the adapter was constructed
        from a pre-parsed :class:`BDDLProblem` without ``bddl_source`` /
        ``bddl_path``). Raises on any other failure path so callers in
        :meth:`on_episode_start` can decide whether to abort or fall back
        to bare-Panda.

        Procedure:

        1. Locate (or write) a ``.bddl`` file on disk - ``libero`` only
           accepts a path. Existing ``bddl_path`` is reused as-is.
        2. Compute SHA256 of the BDDL bytes; cache key is
           ``<scene_cache_dir>/<sha>.xml``. Cache hit → return path
           without touching ``libero`` at all (no GPU / robosuite import).
        3. Cache miss → ``require_optional("libero")`` lazy-imports the
           upstream package, then ``libero.libero.envs.env_wrapper.ControlEnv(
           bddl_file_name=..., has_offscreen_renderer=False, has_renderer=False,
           use_camera_obs=False)`` constructs a robosuite env without
           opening a GL context. Robosuite's ``env.sim.model.get_xml()``
           returns the compiled MJCF as a string.
        4. Apply :attr:`_scene_camera_aliases` via a targeted XML
           rename so the policy-side cameras (``image`` / ``wrist_image``)
           resolve to the LIBERO-canonical viewpoints.
        5. Write the renamed XML to the cache and return the path.

        The LIBERO env is closed after extraction; no robosuite state
        survives this method.
        """
        bddl_path = self._resolve_bddl_path_for_libero()
        if bddl_path is None:
            logger.debug(
                "LiberoAdapter: no BDDL source available for scene generation - "
                "constructed from a pre-parsed BDDLProblem without bddl_source / bddl_path"
            )
            return None

        bddl_bytes = bddl_path.read_bytes()
        cache_key = self._scene_cache_key(bddl_bytes)
        cache_dir = Path(self._scene_cache_dir).expanduser() if self._scene_cache_dir else _default_scene_cache_dir()
        cache_path = cache_dir / f"{cache_key}.xml"
        if cache_path.exists():
            logger.debug("LiberoAdapter: scene cache hit %s", cache_path)
            return str(cache_path)

        # Cache miss - lazy-import libero, build the scene.
        env_wrapper = require_optional(
            "libero.libero.envs.env_wrapper",
            pip_install="libero",
            extra="benchmark-libero",
            purpose="LIBERO scene generation from BDDL",
        )
        ControlEnv = env_wrapper.ControlEnv  # type: ignore[attr-defined]

        # ``has_offscreen_renderer=False`` + ``has_renderer=False`` skip
        # the GL-context bring-up that ``OffScreenRenderEnv`` would
        # otherwise require - we only need the *compiled* model, not
        # rendered frames. ``use_camera_obs=False`` further disables
        # camera observation collection during reset, which would also
        # touch the renderer.
        env = ControlEnv(
            bddl_file_name=str(bddl_path),
            has_offscreen_renderer=False,
            has_renderer=False,
            use_camera_obs=False,
        )
        try:
            xml = _extract_compiled_mjcf(env)
        finally:
            try:
                env.close()
            except Exception as e:  # noqa: BLE001 - close errors are non-fatal
                logger.debug("LiberoAdapter: env.close() raised after extraction: %s", e)

        if self._scene_camera_aliases:
            xml = _rename_mjcf_cameras(xml, self._scene_camera_aliases)

        # #168 used to apply ``_apply_libero_visual_fixes(xml)`` here -
        # rgba alpha=0 on collision geoms + a custom ``<visual>`` block
        # with a stacked ``<headlight>``. #168 verification showed that
        # was the wrong direction:
        #
        # * Upstream LIBERO's ``OffScreenRenderEnv`` emits exactly
        #   ``<visual><map znear="0.001"/></visual>`` (verified
        #   empirically) and gets all its lighting from two ``<light>``
        #   blocks already in the worldbody. #168 stacked a headlight
        #   on top, doubling the illumination - mean RGB jumped to
        #   (136, 117, 89) but the contrast that makes the white mug
        #   pop in upstream was washed out. The user-flagged "agentview
        #   shows arm but no objects" was that contrast loss.
        # * Upstream hides collision capsules at the *renderer* level via
        #   ``mjvOption.geomgroup[0] = 0``, not via MJCF rgba edits.
        #   Same approach for site / joint / actuator markers
        #   (the red dot + green line the reviewer caught are
        #   ``gripper0_ft_frame`` / ``gripper0_grip_site_cylinder``
        #   sites in ``site_group=1``).
        #
        # #168 reverts both transforms here (the cached XML now
        # exactly matches upstream's ``OffScreenRenderEnv.sim.model.get_xml()``)
        # and instead lets ``_install_render_options`` populate
        # ``world._backend_state["viz_option"]`` at episode start; the
        # render path in ``simulation/mujoco/rendering.py`` reads that
        # option and threads it to ``Renderer.update_scene(..., scene_option=...)``,
        # which is what RoboSuite's ``OffScreenRenderEnv`` does. This
        # matches upstream output to within Δ=2.4 RGB units of the
        # reference render.

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(xml)
        logger.info(
            "LiberoAdapter: generated scene MJCF for %s -> %s",
            self.problem.name,
            cache_path,
        )
        return str(cache_path)

    def _resolve_bddl_path_for_libero(self) -> Path | None:
        """Return a ``Path`` to a ``.bddl`` file libero can open, or ``None``.

        - If the adapter was constructed via :meth:`from_file`,
          ``self._bddl_path`` already points at a real file - reuse it.
        - If constructed via :meth:`from_text`, write the source text to
          a stable temp file (keyed by SHA256 of the text) so libero has
          a real path. The temp file lives under
          ``<scene_cache_dir>/.bddl/`` so it's cleaned up alongside the
          scene cache.
        - If neither is set, return ``None``.
        """
        if self._bddl_path is not None:
            p = Path(self._bddl_path).expanduser()
            if p.is_file():
                return p
            logger.debug(
                "LiberoAdapter: bddl_path=%s not on disk; falling back to bddl_source",
                p,
            )
        if self._bddl_source is None:
            return None

        cache_dir = Path(self._scene_cache_dir).expanduser() if self._scene_cache_dir else _default_scene_cache_dir()
        bddl_dir = cache_dir / ".bddl"
        bddl_dir.mkdir(parents=True, exist_ok=True)
        sha = hashlib.sha256(self._bddl_source.encode("utf-8")).hexdigest()
        tmp = bddl_dir / f"{sha}.bddl"
        if not tmp.exists():
            tmp.write_text(self._bddl_source)
        return tmp

    def _scene_cache_key(self, bddl_bytes: bytes) -> str:
        """Compute the scene-cache filename stem for ``bddl_bytes``.

        The key is ``sha256(bddl_bytes || aliases || transform_version)``.
        Including the alias map makes the cache invalidate automatically
        when a user changes :attr:`_scene_camera_aliases` (or the
        adapter's default map evolves, e.g. the #168-r5 fix that adds
        ``"robot0_eye_in_hand": "wrist_image"``). Without this,
        upgrading users would serve the stale on-disk rewrite that
        leaves ``robot0_eye_in_hand`` un-renamed - the GR00T policy
        would keep seeing the static top-down fallback at the
        ``wrist_image`` slot and the wrist channel would be
        out-of-distribution every step.

        ``_LIBERO_MJCF_TRANSFORM_VERSION`` is bumped whenever the
        post-process logic in :meth:`_generate_scene_from_bddl`
        changes (history at the constant's definition - #168 initially added
        collision-geom hiding + headlight boost, then reverted both
        and moved visualisation hiding to render-time mjvOption).
        Bumping invalidates stale on-disk caches generated by prior
        versions so users picking up the upgrade automatically
        regenerate the post-processed XML instead of serving the
        un-fixed cached one. Hashed alongside the alias map for the
        same reason.

        ``json.dumps(..., sort_keys=True)`` makes the hash deterministic
        across Python invocations (dict iteration order is insertion
        order in CPython 3.7+, but tests construct adapters in
        unpredictable orders and we want stable hashing).

        Returns the hex digest (no extension); callers append ``.xml`` /
        ``.bddl`` as appropriate.
        """
        alias_repr = json.dumps(self._scene_camera_aliases, sort_keys=True).encode("utf-8")
        return hashlib.sha256(
            bddl_bytes + b"|aliases:" + alias_repr + b"|tform:" + _LIBERO_MJCF_TRANSFORM_VERSION.encode("utf-8")
        ).hexdigest()

    def _register_default_robot(self, sim: SimEngine) -> None:
        """Wrap the scene-supplied Panda in ``world.robots`` WITHOUT recompiling.

        Goal: make ``sim.list_robots()`` return non-empty BEFORE
        ``super().on_episode_start`` runs, so the base
        :class:`BenchmarkProtocol` skips its unconditional
        ``sim.add_robot(name="robot", ...)`` call. Otherwise that
        unconditional call injects a *second* Panda into the spec
        (scene-supplied + injected = two kinematic chains, ``nq``
        jumps ``44 → 53`` on LIBERO SCENE5) and leaves the redundant
        Panda's plastic shells right in front of the ``image`` camera
        — that's #166's smoking gun: every "real-render" frame
        is yellow-saturated by the second Panda's links 5/6.

        The fix has to register a wrapper for the **existing** Panda
        without recompiling. Two-step detection:

        1. Walk the compiled MuJoCo model's body names looking for the
           robosuite/LIBERO ``robot0_`` prefix (the standard naming
           convention for the hand-authored and procedurally-generated
           LIBERO scenes alike).
        2. If found, build a :class:`SimRobot` whose ``namespace`` matches
           the discovered prefix and whose ``joint_names`` /
           ``actuator_ids`` come from filtering the model's joint /
           actuator pools by the same prefix. Register it directly in
           ``world.robots`` under the canonical key ``"robot"`` so
           super() finds it.

        Best-effort:

        * Sim without a compiled MuJoCo model → debug-log + skip.
          super() will fall through to its own add_robot path; that
          path is the bug we're trying to avoid, but on a non-MuJoCo
          backend it's the only correct behaviour anyway.
        * No body matches the ``robot0_`` prefix → debug-log + skip.
          The scene didn't supply a Panda; super() should add one.
        * Robot already registered under ``"robot"`` (defensive) →
          no-op.
        """
        world = getattr(sim, "_world", None)
        if world is None or not hasattr(world, "robots"):
            return
        if "robot" in world.robots:
            return  # super() will see it and skip its own add

        try:
            import mujoco as _mj
        except ImportError:
            logger.debug("LiberoAdapter: mujoco not importable; skipping pre-register")
            return

        model = getattr(world, "_model", None)
        if model is None:
            logger.debug("LiberoAdapter: no compiled model; skipping pre-register")
            return

        try:
            wrapper = _build_scene_robot_wrapper(
                _mj,
                model,
                prefix=self._scene_robot_prefix,
                gripper_prefix=self._scene_gripper_prefix,
            )
        except Exception as e:  # noqa: BLE001 - never abort eval on a discovery failure
            logger.warning(
                "LiberoAdapter: scene-Panda discovery failed: %s; super() will fall back to its add_robot path",
                e,
            )
            return
        if wrapper is None:
            logger.debug(
                "LiberoAdapter: no body with prefix %r found in scene; super() will add a Panda",
                self._scene_robot_prefix,
            )
            return

        # Register under the key super() would have used so its
        # list_robots() check finds it.
        world.robots["robot"] = wrapper
        logger.debug(
            "LiberoAdapter: registered scene-supplied Panda (arm prefix=%r, gripper prefix=%r) as 'robot' "
            "(joints=%d, actuators=%d)",
            self._scene_robot_prefix,
            self._scene_gripper_prefix,
            len(wrapper.joint_names),
            len(wrapper.actuator_ids),
        )

        # The bare-Panda defaults for ``_eef_body_name`` ("hand") and
        # ``_gripper_joint_name`` ("finger_joint1") don't exist in
        # RoboSuite-emitted scenes - those use ``robot0_right_hand`` for
        # the EEF body and ``gripper0_finger_joint1`` for the gripper
        # joint. Without this auto-resolution, ``augment_observation``
        # silently drops every ``state.x/y/z/roll/pitch/yaw`` and
        # ``state.gripper`` key (because ``get_body_state("hand")``
        # returns body_id=-1 and the gripper-joint suffix-match fails),
        # the GR00T server then rejects every observation with
        # ``State key 'state.x' must be in observation`` and the eval
        # crashes before producing any frame. The user-explicit-override
        # check via ``_user_eef_body_name`` / ``_user_gripper_joint_name``
        # ensures that callers passing a custom value still get their
        # value respected; only the constructor default (``None``)
        # triggers auto-resolution.
        self._resolve_scene_eef_and_gripper(_mj, model)

    def _resolve_scene_eef_and_gripper(self, mj: Any, model: Any) -> None:
        """Auto-resolve EEF body name and gripper joint name from the scene.

        Searches the compiled MuJoCo model for the canonical RoboSuite /
        LIBERO names that the upstream GR00T-LIBERO checkpoint was trained
        against:

        * EEF body: ``<scene_robot_prefix>right_hand`` (RoboSuite default)
          -> ``<scene_robot_prefix>hand`` -> bare ``hand`` /
          ``right_hand``. First match wins.
        * Gripper joint: ``<scene_gripper_prefix>finger_joint1``
          (RoboSuite default; the gripper has its OWN namespace separate
          from the robot's because RoboSuite attaches grippers via a
          dedicated naming scheme) -> ``<scene_robot_prefix>finger_joint1``
          -> bare ``finger_joint1``. First match wins.

        Only fires when the constructor default (``None``) was used.
        Explicit user-supplied values - tracked via
        ``_user_eef_body_name`` / ``_user_gripper_joint_name`` - are
        preserved verbatim (they may legitimately point at a custom
        scene whose body / joint names don't match the conventions
        above).

        Best-effort: any failure (model missing the ``nbody``/``njnt``
        attributes, ``mj_name2id`` raising) is caught and logged at
        DEBUG, leaving the legacy bare-Panda defaults
        (``"hand"`` / ``"finger_joint1"``) in place. That preserves the
        pre-#166 behaviour for non-MuJoCo backends.
        """
        prefix = self._scene_robot_prefix
        gprefix = self._scene_gripper_prefix

        if self._user_eef_body_name is None:
            eef_candidates: list[str] = []
            # Prefix-namespaced first (the case for RoboSuite/LIBERO)
            for suffix in ("right_hand", "hand", "eef"):
                if prefix:
                    eef_candidates.append(f"{prefix}{suffix}")
            # Then bare names as fallback (covers Menagerie's bare Panda)
            eef_candidates.extend(["right_hand", "hand", "eef"])
            resolved = self._first_named(mj, model, names=eef_candidates, obj=mj.mjtObj.mjOBJ_BODY)
            if resolved is not None and resolved != self._eef_body_name:
                logger.debug(
                    "LiberoAdapter: auto-resolved eef_body_name to %r (was %r); scene has prefix %r",
                    resolved,
                    self._eef_body_name,
                    prefix,
                )
                self._eef_body_name = resolved
            elif resolved is None:
                logger.debug(
                    "LiberoAdapter: no scene EEF body found among %r; keeping default %r",
                    eef_candidates,
                    self._eef_body_name,
                )

        if self._user_gripper_joint_name is None:
            grip_candidates: list[str] = []
            # Gripper namespace first (RoboSuite ``gripper0_finger_joint1``)
            if gprefix:
                grip_candidates.append(f"{gprefix}finger_joint1")
            # Robot namespace next (some custom scenes share namespaces)
            if prefix:
                grip_candidates.append(f"{prefix}finger_joint1")
            # Bare fallback (Menagerie Panda)
            grip_candidates.append("finger_joint1")
            resolved = self._first_named(mj, model, names=grip_candidates, obj=mj.mjtObj.mjOBJ_JOINT)
            if resolved is not None and resolved != self._gripper_joint_name:
                logger.debug(
                    "LiberoAdapter: auto-resolved gripper_joint_name to %r (was %r); gripper prefix %r",
                    resolved,
                    self._gripper_joint_name,
                    gprefix,
                )
                self._gripper_joint_name = resolved
            elif resolved is None:
                logger.debug(
                    "LiberoAdapter: no scene gripper joint found among %r; keeping default %r",
                    grip_candidates,
                    self._gripper_joint_name,
                )

    @staticmethod
    def _first_named(mj: Any, model: Any, *, names: list[str], obj: int) -> str | None:
        """Return the first name in ``names`` that resolves to a valid id.

        Walks ``names`` in order and returns the first one for which
        ``mj.mj_name2id(model, obj, name)`` returns a non-negative id.
        Returns ``None`` when no candidate resolves or when ``mj`` lacks
        ``mj_name2id`` (defensive against test stubs).
        """
        mj_name2id = getattr(mj, "mj_name2id", None)
        if mj_name2id is None:
            return None
        try:
            for name in names:
                if mj_name2id(model, obj, name) >= 0:
                    return name
        except Exception as e:  # noqa: BLE001 - never fatal during name resolution
            logger.debug("LiberoAdapter: mj_name2id lookup raised: %s", e)
            return None
        return None

    def _install_render_options(self, sim: SimEngine) -> None:
        """Install LIBERO-canonical render-time visualization options on ``sim``.

        Stores an ``mujoco.MjvOption`` in ``sim._world._backend_state["viz_option"]``
        configured to match upstream LIBERO's
        ``OffScreenRenderEnv`` viewer setup. The render path in
        :mod:`strands_robots.simulation.mujoco.rendering` reads this
        option from ``_backend_state`` and threads it through to
        :meth:`mujoco.Renderer.update_scene` as ``scene_option=``, so
        every rendered frame (per-step observation, ``sim.render()``,
        and the ``start_cameras_recording`` MP4 path which all funnel
        through ``render()``) hides:

        * Collision geoms (``geomgroup[0] = 0``). RoboSuite emits
          collision capsules with explicit coloured ``rgba`` (green for
          arm links, blue for gripper components, yellow for table). All
          110 of them in the LIBERO_10 cache. MuJoCo's default
          ``geomgroup = [1, 1, 1, 1, 1, 1]`` shows them on top of the
          actual visual mesh - the green/blue patches the user originally
          flagged.
        * Site markers (``sitegroup[*] = 0`` for all 6 groups). Includes
          ``gripper0_ft_frame`` (red dot - force-torque frame),
          ``gripper0_grip_site`` (semi-transparent red sphere),
          ``gripper0_grip_site_cylinder`` (green line - grasp-axis
          cylinder). Default ``sitegroup`` shows them; reviewer caught
          them as "red dot + green line in the middle of agentview".
        * Joint visualisations (``mjVIS_JOINT = 0``). Hides the
          axis-arrow widgets MuJoCo draws on each ``<joint>`` definition.
        * Actuator visualisations (``mjVIS_ACTUATOR = 0``). Hides the
          actuator-pose markers.
        * COM markers (``mjVIS_COM = 0``). Hides the per-body
          centre-of-mass widgets.

        These match exactly what ``OffScreenRenderEnv`` configures
        internally - verified empirically against ``upstream-agentview.png``,
        which renders within Δ=2.4 RGB units of upstream when the same
        flags are applied to our cache.

        #168 used to do equivalent work via MJCF post-process
        (``_apply_libero_visual_fixes`` set rgba alpha=0 on collision
        geoms and stacked a ``<headlight>`` block in ``<visual>``).
        That worked for the agentview mean-RGB metric but the
        ``<headlight>`` *doubled* the lighting (upstream already has
        two ``<light>`` blocks in the worldbody) and washed out the
        contrast that makes the white mug pop. #168 reverts the
        MJCF rewrites entirely (cache now matches
        ``OffScreenRenderEnv.sim.model.get_xml()`` verbatim except for
        the camera-name aliases) and instead lets the render-time
        options handle visibility - the architecturally correct place.

        Best-effort: ``mujoco`` not importable -> debug-log + skip.
        ``world`` missing or ``_backend_state`` not a dict -> skip.
        Default :class:`MuJoCoSimulation` always exposes
        ``_world._backend_state``, so the skip cases are defensive
        against test stubs / non-MuJoCo backends.
        """
        try:
            import mujoco as _mj
        except ImportError:
            logger.debug("LiberoAdapter: mujoco not importable; skipping render-options install")
            return

        world = getattr(sim, "_world", None)
        if world is None:
            logger.debug("LiberoAdapter: sim has no _world; skipping render-options install")
            return
        backend_state = getattr(world, "_backend_state", None)
        if not isinstance(backend_state, dict):
            logger.debug("LiberoAdapter: world._backend_state missing or not a dict; skipping")
            return

        # Building the option must not crash the eval - test stubs and
        # partial-mock mujoco modules may lack ``MjvOption`` /
        # ``mjv_defaultOption`` / ``mjtVisFlag.mjVIS_*``. Catch any of
        # those and degrade to default-render-options gracefully (slightly
        # worse visual output but eval completes). The render path
        # tolerates ``viz_option=None`` natively.
        try:
            opt = _mj.MjvOption()
            _mj.mjv_defaultOption(opt)
            # Hide collision geoms (group=0). MuJoCo's default after
            # mjv_defaultOption is geomgroup=[1, 1, 1, 0, 0, 0] - groups 0, 1, 2
            # visible. We turn off group 0.
            opt.geomgroup[0] = 0
            # Hide all site markers. Default after mjv_defaultOption is
            # sitegroup=[1, 1, 1, 0, 0, 0]; turn off all 6.
            for sg in range(6):
                opt.sitegroup[sg] = 0
            opt.flags[_mj.mjtVisFlag.mjVIS_JOINT] = 0
            opt.flags[_mj.mjtVisFlag.mjVIS_ACTUATOR] = 0
            opt.flags[_mj.mjtVisFlag.mjVIS_COM] = 0
        except (AttributeError, TypeError) as e:
            # Partial-mock mujoco module (test stub) - skip silently.
            logger.debug("LiberoAdapter: building MjvOption failed (%s); skipping", e)
            return

        backend_state["viz_option"] = opt
        logger.debug("LiberoAdapter: installed render options on world._backend_state['viz_option']")

    def _install_action_controller(self, sim: SimEngine) -> None:
        """Install OSC_POSE controller for GR00T task-space -> joint torques.

        GR00T-LIBERO outputs 7-dim Cartesian delta-EEF actions
        (``{x, y, z, roll, pitch, yaw, gripper}``); LIBERO scenes use
        torque-mode joint actuators (``robot0_torq_j1..7`` plus 2
        gripper). Without this controller, ``_apply_sim_action`` looks
        up GR00T's action keys by name in the model's actuator/joint
        tables, finds no match, and silently drops every action - the
        policy effectively sends zero torque (#168 verification).

        This installs a :class:`_LiberoOSCController` instance in
        ``world._backend_state["action_controller"]``. The rendering
        layer's :meth:`_apply_sim_action` checks for this key (via
        :meth:`_get_action_controller`) and dispatches to
        ``controller.apply(action_dict, model, data, robot_name)``
        which writes joint torques to ``data.ctrl``.

        The controller wraps RoboSuite's
        ``OperationalSpaceController`` for the arm (6-dim Cartesian
        PD with inverse-Jacobian) and a direct gripper-actuator
        write for the 7th channel.

        Best-effort: missing robosuite, missing site, missing actuator
        IDs - all log + skip without raising. The eval will run with
        actions silently dropped (the #168 status quo), which at
        least doesn't crash.

        Lifecycle: tied to the loaded scene's compiled model. Since
        ``_apply_canonical_state``'s init-state apply may have already
        run, the controller is built using stable model IDs at
        on_episode_start time. Subsequent ``load_scene`` calls
        invalidate the controller's IDs; the new world's
        ``_backend_state`` is fresh so callers must re-install via
        ``prewarm`` or another ``on_episode_start`` cycle.
        """
        try:
            controller = _LiberoOSCController.from_sim(
                sim,
                eef_site_name=f"{self._scene_gripper_prefix}grip_site",
                arm_prefix=self._scene_robot_prefix,
                gripper_prefix=self._scene_gripper_prefix,
            )
        except _ControllerInstallError as e:
            logger.warning(
                "LiberoAdapter._install_action_controller: %s. "
                "GR00T actions will silently no-op until this is resolved.",
                e,
            )
            return
        except Exception as e:  # noqa: BLE001 - never abort eval on controller failure
            logger.warning(
                "LiberoAdapter._install_action_controller: unexpected failure (%s); "
                "GR00T actions will silently no-op until this is resolved.",
                e,
            )
            return

        world = getattr(sim, "_world", None)
        if world is None:
            return
        backend_state = getattr(world, "_backend_state", None)
        if not isinstance(backend_state, dict):
            return
        backend_state["action_controller"] = controller
        logger.debug(
            "LiberoAdapter: installed OSC_POSE action_controller (eef_site=%r, arm_actuators=%d, gripper_actuators=%d)",
            controller.eef_site_name,
            len(controller.arm_actuator_ids),
            len(controller.gripper_actuator_ids),
        )

    def _install_libero_cameras(self, sim: SimEngine) -> None:
        """Inject the cameras the ``libero_panda`` data_config expects.

        Best-effort: the LIBERO ``Gr00tDataConfig`` declares
        ``video_keys = ["video.image", "video.wrist_image"]`` and the policy's
        ``_build_service_observation`` reads those from the robot observation
        as ``obs["image"]`` / ``obs["wrist_image"]``. Without these cameras
        in the sim, every direct-client call to a LIBERO server fails with
        ``Video key 'video.image' must be in observation`` (#148, Failure 1).

        Cameras already present in the sim are skipped silently. "Already
        present" means *either*:

        * the runtime camera registry on ``sim._world.cameras`` (added via
          a previous ``sim.add_camera`` call, including by this adapter on
          a prior episode), OR
        * the *compiled MuJoCo model* (declared via ``<camera>`` elements
          in a scene MJCF that ``sim.load_scene`` just loaded).

        The model-side check is critical for #166 because
        :meth:`Simulation.load_scene` creates a fresh ``SimWorld`` whose
        ``cameras`` registry starts empty even when the loaded MJCF
        declares cameras. Without the model-side check the install would
        re-add the same cameras on top of the scene's ones, triggering a
        spec recompile that resets qpos away from the canonical state we
        just restored in :meth:`_apply_canonical_state`.

        Other ``add_camera`` failures are logged at WARNING but never
        fatal - one missing camera shouldn't kill the whole eval.
        """
        add_camera = getattr(sim, "add_camera", None)
        if add_camera is None:
            logger.debug("LiberoAdapter: sim has no add_camera(); skipping camera install")
            return

        existing = self._existing_camera_names(sim)

        for cam_name, cam_kwargs in self._cameras.items():
            if cam_name in existing:
                logger.debug("LiberoAdapter: camera %r already in sim; skipping install", cam_name)
                # #168: even when we skip add_camera (because
                # the model-compiled camera is already there from
                # ``scene_camera_aliases`` rename), we still need to
                # publish the configured render dimensions to
                # ``world.cameras`` so :meth:`_get_sim_observation` reads
                # them via its ``cam_info.height/width`` lookup. Without
                # this step, model-side cameras fall through to the
                # 480×640 ``default_height``/``default_width`` of the
                # renderer mixin — which is a different aspect ratio AND
                # resolution from training (256×256), feeding GR00T
                # out-of-distribution images even after the #168
                # vertical-flip fix. Diagnostic
                # ``/tmp/opencode/eval-runs/diff_libero_obs.py`` showed
                # ``sim.get_observation()`` returned 480×640 while
                # ``sim.render(camera="image", width=256, height=256)``
                # correctly returned 256×256 — the divergence was
                # entirely in the get_observation path.
                self._publish_camera_dims_to_world(sim, cam_name, cam_kwargs)
                continue
            try:
                result = add_camera(name=cam_name, **cam_kwargs)
            except Exception as e:  # noqa: BLE001 - one bad camera shouldn't kill the eval
                logger.warning("LiberoAdapter: add_camera(%r) raised: %s", cam_name, e)
                continue
            if isinstance(result, dict) and result.get("status") == "error":
                msg = (result.get("content") or [{}])[0].get("text", "")
                # "already exists" is benign - the scene XML beat us to it.
                if "already exists" in msg.lower():
                    logger.debug("LiberoAdapter: camera %r already declared by scene", cam_name)
                    # Same publish step as the skip branch above —
                    # ``add_camera`` early-returned with "already exists"
                    # because the scene's MJCF declared the camera, so
                    # ``world.cameras[cam_name]`` is still empty. Inject
                    # a config-only SimCamera entry for dimension lookup.
                    self._publish_camera_dims_to_world(sim, cam_name, cam_kwargs)
                else:
                    logger.warning("LiberoAdapter: add_camera(%r) failed: %s", cam_name, msg)

    @staticmethod
    def _publish_camera_dims_to_world(sim: SimEngine, cam_name: str, cam_kwargs: dict[str, Any]) -> None:
        """Inject a config-only :class:`SimCamera` entry into ``world.cameras``.

        Used by :meth:`_install_libero_cameras` for cameras that already
        exist in the compiled MuJoCo model (typically renamed via
        :attr:`scene_camera_aliases`) so :meth:`_get_sim_observation`'s
        ``cam_info.height``/``cam_info.width`` lookup picks up the
        configured render dimensions instead of falling through to
        ``default_height``/``default_width`` (480×640).

        Idempotent: skips silently if ``world.cameras[cam_name]`` is
        already populated (from a prior call or an explicit
        ``add_camera`` from the user). Best-effort: skips silently if
        the sim has no ``_world`` or no ``cameras`` registry.

        Note: ``camera_id`` is left at the default ``-1`` because we
        don't know (and don't need) the model-side camera index here —
        :meth:`_get_sim_observation` looks it up via ``mj_name2id``.
        Only the ``height`` / ``width`` fields matter for this code
        path; the pose / FOV fields are not used (the model-compiled
        camera's pose / FOV from the MJCF wins, which is what we want).

        #168.
        """
        world = getattr(sim, "_world", None)
        if world is None:
            return
        cameras_attr = getattr(world, "cameras", None)
        if not isinstance(cameras_attr, dict):
            return
        if cam_name in cameras_attr:
            # Already published (e.g. an earlier episode + scene-recompile
            # cycle). Don't overwrite the user's possibly-tweaked entry.
            return
        height = int(cam_kwargs.get("height", 256))
        width = int(cam_kwargs.get("width", 256))
        cameras_attr[cam_name] = SimCamera(
            name=cam_name,
            width=width,
            height=height,
        )
        logger.debug(
            "LiberoAdapter: published render dims for model-side camera %r (%dx%d) to world.cameras",
            cam_name,
            width,
            height,
        )

    @staticmethod
    def _existing_camera_names(sim: SimEngine) -> set[str]:
        """Union of registry-side and model-side camera names known to ``sim``.

        Backends without a MuJoCo-compiled model (or with mujoco not
        importable) fall back to the registry-only check - that's the
        pre-#166-review behaviour, retained as a defensive fallback for
        non-MuJoCo engines that still want LIBERO eval.

        Critical for #166: :meth:`Simulation.load_scene` creates a fresh
        ``SimWorld`` whose ``cameras`` dict starts empty even when the
        loaded MJCF declares ``<camera>`` elements. Without enumerating
        the compiled model's cameras here, ``_install_libero_cameras``
        would unconditionally try to inject ``image`` / ``wrist_image``
        on top of scene-declared ones, triggering a spec recompile that
        in turn resets qpos and undoes :meth:`_apply_canonical_state`.
        """
        names: set[str] = set()
        world = getattr(sim, "_world", None)

        # Registry-side: cameras added via sim.add_camera() previously.
        cameras_attr = getattr(world, "cameras", None) if world is not None else None
        if isinstance(cameras_attr, dict):
            names.update(cameras_attr.keys())

        # Model-side: cameras declared in a loaded scene MJCF.
        model = getattr(world, "_model", None) if world is not None else None
        if model is None:
            return names
        try:
            import mujoco as _mj
        except ImportError:
            logger.debug("LiberoAdapter: mujoco not importable; skipping model-side camera check")
            return names
        try:
            ncam = int(getattr(model, "ncam", 0))
            for i in range(ncam):
                name = _mj.mj_id2name(model, _mj.mjtObj.mjOBJ_CAMERA, i)
                if name:
                    names.add(name)
        except Exception as e:  # noqa: BLE001 - never fatal during camera-existence check
            logger.debug("LiberoAdapter: model-side camera enumeration failed: %s", e)
        return names

    def _apply_canonical_state(self, sim: SimEngine, rng: random.Random | None = None) -> None:
        """Restore qpos / qvel to the scene's canonical home state.

        Three branches, in order of preference:

        1. **Init states** (``self._init_states is not None``): pick a row
           via ``rng`` (RNG-seeded so a given seed re-runs the same init
           across re-evaluations, fall back to ``random.Random()`` when
           no rng is provided), validate the width matches
           ``1 + model.nq + model.nv``, then write
           ``data.time / data.qpos / data.qvel`` directly. This is the
           branch that landed in #168 to fix bug I (success_rate=0
           because the robot started at qpos=0 instead of LIBERO's
           canonical "ready" pose). Width mismatches are raised loudly
           rather than silently sliced - they indicate the procedurally-
           generated MJCF diverges from upstream LIBERO's scene MJCF
           (e.g. missing ``(:objects ...)`` declarations).
        2. **Keyframe** (``model.nkey > 0``): call
           ``mujoco.mj_resetDataKeyframe(model, data, scene_keyframe_index)``.
           The MJCF carries the canonical pose explicitly via a
           ``<keyframe>`` element - LIBERO-authored hand-written scenes
           (the ones in upstream ``libero/libero/assets/scenes/``) ship one.
        3. **Snapshot-and-restore** (``model.nkey == 0``): cache
           ``data.qpos`` / ``data.qvel`` on the FIRST episode after a
           scene compile (after ``super().on_episode_start`` and
           ``_install_libero_cameras`` have run); restore the cached
           snapshot on every subsequent episode. The procedurally-
           generated MJCFs from :meth:`_generate_scene_from_bddl` (PR #165)
           don't carry a keyframe, so this branch fires when init_states
           is also None (e.g. adapters built directly from a BDDL file
           without going through :func:`load_libero_suite`).

        All three branches end with ``mj_forward`` so derived state
        (``xpos`` / ``xquat`` / sensor data) reflects the canonical
        ``qpos`` before the next ``get_observation`` / ``render`` call.

        Best-effort:

        * Sims without an exposed compiled MuJoCo model -> debug-log + skip.
        * ``scene_keyframe_index`` out of range when ``nkey > 0`` -> log
          at WARNING and skip (out-of-range is a config error).
        * ``mujoco`` not importable -> debug-log + skip.
        * Snapshot shape mismatches the current ``qpos`` (e.g. the
          model recompiled with a different ``nq`` between episodes,
          which is unusual) -> re-capture instead of restoring.

        Holds ``sim._lock`` if the sim exposes one to match the locking
        contract of :meth:`Simulation.reset` and :meth:`Simulation.send_action`
        - prevents racing a worker holding a stale qpos pointer.
        """
        world = getattr(sim, "_world", None)
        model = getattr(world, "_model", None) if world is not None else None
        data = getattr(world, "_data", None) if world is not None else None
        if model is None or data is None:
            logger.debug("LiberoAdapter: sim has no compiled MuJoCo model/data; skipping canonical-state apply")
            return

        try:
            import mujoco as _mj
        except ImportError:
            logger.debug("LiberoAdapter: mujoco not importable; skipping canonical-state apply")
            return

        nkey = int(getattr(model, "nkey", 0))
        lock = getattr(sim, "_lock", None)

        # Branch 1: init_states (highest priority)
        if self._init_states is not None:
            self._apply_init_state_branch(model, data, _mj, lock, rng=rng)
            return

        # Branch 2: keyframe
        if nkey > 0:
            self._apply_keyframe_branch(sim, model, data, _mj, lock, nkey)
        # Branch 3: snapshot-and-restore
        else:
            self._apply_snapshot_branch(sim, model, data, _mj, lock)

    def _apply_init_state_branch(
        self,
        model: Any,
        data: Any,
        mj: Any,
        lock: Any,
        *,
        rng: random.Random | None,
    ) -> None:
        """Init-state branch of :meth:`_apply_canonical_state` (#168).

        Picks one row from ``self._init_states`` (RNG-seeded), validates
        the width matches ``1 + model.nq + model.nv``, then writes
        ``data.time / data.qpos / data.qvel`` directly. Uses
        :func:`mujoco.mj_forward` to update derived state.

        Layout per
        ``/opt/conda/lib/python3.12/site-packages/robosuite/utils/binding_utils.py:213-241``
        (``MjSimState.from_flattened``): ``[time(1), qpos(nq), qvel(nv)]``,
        with ``na == 0`` asserted (no actuator state). LIBERO's
        ``OffScreenRenderEnv.set_init_state`` calls
        ``sim.set_state_from_flattened(state)`` which decomposes the
        same way, so applying directly to MuJoCo's ``data`` here matches
        the reference v0.1.1 LIBERO eval pipeline.

        Width mismatch is fatal (raises ``RuntimeError``) - the
        procedurally-generated MJCF must match upstream LIBERO's scene
        for init-state apply to make any sense. Silent slicing /
        padding would produce a deeply wrong physical state and mask
        a real scene-generation bug. Per AGENTS.md "no silent defaults
        on error".

        Episode 0 is pinned to ``init_states[0]`` deterministically;
        episodes 1+ are RNG-sampled. This matches v0.1.1 ``env_libero.py``'s
        ``env.set_init_state(init_states[0])`` pattern for the first
        episode and aligns with :meth:`prewarm`'s init-state apply
        (which also uses idx 0). The recorder's t=0.00 frame and the
        policy's first observation are then visually identical -
        critical for the visual-acceptance regression suite where
        users compare "first recorded frame" against "expected
        starting pose" (#168 bug D-residual).

        Per-episode RNG-seeded selection (episodes 1+): ``rng.randint(0, n_states-1)``.
        Re-running the same seed produces the same init state for the
        same episode index. ``rng=None`` falls back to a fresh
        ``random.Random()`` so direct calls (e.g. unit tests) still
        work.

        ``self._episode_count`` increments after every successful call
        so the next call is "episode 1+" and uses RNG sampling.
        """
        n_states = int(self._init_states.shape[0])  # type: ignore[union-attr]
        if n_states == 0:
            logger.debug("LiberoAdapter: empty init_states array; skipping init-state branch")
            return
        # Episode 0 = idx 0 (deterministic, matches v0.1.1 + prewarm).
        # Episodes 1+ = RNG-sampled.
        if self._episode_count == 0:
            idx = 0
        else:
            rng_local = rng if rng is not None else random.Random()
            idx = rng_local.randint(0, n_states - 1)
        state = self._init_states[idx]  # type: ignore[index]

        nq = int(model.nq)
        nv = int(model.nv)
        na = int(getattr(model, "na", 0))
        if na != 0:
            raise RuntimeError(
                f"LiberoAdapter: model has na={na} actuator state; init_state apply requires na=0. "
                f"LIBERO scenes don't carry actuator state and the flat-state layout assumes [time, qpos, qvel]."
            )

        expected_width = 1 + nq + nv
        actual_width = int(state.shape[0])
        if actual_width != expected_width:
            raise RuntimeError(
                f"LiberoAdapter: init_state width {actual_width} does not match compiled model "
                f"(1 + nq={nq} + nv={nv} = {expected_width}). The procedurally-generated MJCF "
                f"likely diverges from the upstream LIBERO scene MJCF for this BDDL task "
                f"(e.g. missing (:objects ...) declarations dropping free-joint bodies). "
                f"#168 bug I: silent slicing forbidden - fix the scene generator instead."
            )

        def _apply() -> None:
            data.time = float(state[0])
            np.copyto(data.qpos, state[1 : 1 + nq])
            np.copyto(data.qvel, state[1 + nq :])
            mj.mj_forward(model, data)

        if lock is not None:
            with lock:
                _apply()
        else:
            _apply()

        logger.debug(
            "LiberoAdapter: applied init_state[%d] (ep=%d, 1+nq+nv=%d, n_states=%d)",
            idx,
            self._episode_count,
            expected_width,
            n_states,
        )

        # Increment after successful apply so the next call is
        # "episode 1+" and gets RNG-sampled selection.
        self._episode_count += 1

    def _apply_keyframe_branch(
        self,
        sim: SimEngine,  # noqa: ARG002 - kept for symmetry with _apply_snapshot_branch
        model: Any,
        data: Any,
        mj: Any,
        lock: Any,
        nkey: int,
    ) -> None:
        """Keyframe branch of :meth:`_apply_canonical_state`."""
        if self._scene_keyframe_index < 0 or self._scene_keyframe_index >= nkey:
            logger.warning(
                "LiberoAdapter: scene_keyframe_index=%d out of range [0, %d); skipping",
                self._scene_keyframe_index,
                nkey,
            )
            return
        try:
            if lock is not None:
                with lock:
                    mj.mj_resetDataKeyframe(model, data, self._scene_keyframe_index)
                    mj.mj_forward(model, data)
            else:
                mj.mj_resetDataKeyframe(model, data, self._scene_keyframe_index)
                mj.mj_forward(model, data)
        except Exception as e:  # noqa: BLE001 - never fatal
            logger.warning(
                "LiberoAdapter: mj_resetDataKeyframe(%d) failed: %s",
                self._scene_keyframe_index,
                e,
            )
            return
        logger.debug(
            "LiberoAdapter: applied <keyframe> %d to canonical qpos",
            self._scene_keyframe_index,
        )

    def _apply_snapshot_branch(
        self,
        sim: SimEngine,  # noqa: ARG002 - kept for symmetry with _apply_keyframe_branch
        model: Any,
        data: Any,
        mj: Any,
        lock: Any,
    ) -> None:
        """Snapshot-and-restore branch of :meth:`_apply_canonical_state`.

        First episode: write the LIBERO-canonical Panda home pose into
        ``data.qpos[arm]`` (mirroring upstream ``Robot.reset(deterministic=True)``)
        so the snapshot captures arm joints at home pose, then capture
        ``data.qpos`` / ``data.qvel``. Subsequent episodes: restore the
        cached snapshot via ``np.copyto`` and ``mj_forward``.

        Procedurally-generated MJCFs (#165) hit this branch because they
        don't ship a ``<keyframe>`` AND callers may not pass
        ``init_states``. Without writing the home pose, ``data.qpos`` after
        ``load_scene`` is at MuJoCo's default (all zeros for arm joints,
        since the procedurally-generated MJCF doesn't set joint ``ref``
        attributes). Upstream's ``OffScreenRenderEnv`` instead writes
        ``MountedPanda.init_qpos`` (= ``[0, -0.161, 0, -2.444, 0, 2.227,
        π/4]``) inside its ``Robot.reset`` so the env starts at the
        ready pose. To match upstream's BDDL-default behaviour (#176
        sub-task 3d), we replicate that write here.

        #176 (sub-task 3d).
        """
        try:
            qpos = data.qpos
            qvel = data.qvel
        except AttributeError as e:
            logger.debug("LiberoAdapter: data has no qpos/qvel attrs: %s", e)
            return

        # First episode (or model recompile changed nq) -> write home
        # pose into arm joints, then capture; don't restore. The
        # snapshot is taken after super() + _install_libero_cameras
        # so it reflects the post-setup canonical state, with the arm
        # at the LIBERO-canonical ready pose.
        needs_capture = (
            self._canonical_qpos is None
            or self._canonical_qpos.shape != qpos.shape
            or self._canonical_qvel is None
            or self._canonical_qvel.shape != qvel.shape
        )
        if needs_capture:
            # #176 (sub-task 3d): write home pose to arm
            # joints before snapshotting. Without this the captured
            # snapshot is at MuJoCo's default qpos=0, which is
            # significantly off from upstream's ``MountedPanda.init_qpos``
            # post-``Robot.reset``. Best-effort: failures during home-
            # pose discovery / write fall through silently — the
            # snapshot captures whatever ``data.qpos`` happens to be.
            self._write_libero_arm_home_qpos(model, data, mj, lock)
            try:
                self._canonical_qpos = np.array(qpos, copy=True)
                self._canonical_qvel = np.array(qvel, copy=True)
            except Exception as e:  # noqa: BLE001 - capture is best-effort
                logger.debug("LiberoAdapter: snapshot capture failed: %s", e)
                self._canonical_qpos = None
                self._canonical_qvel = None
                return
            logger.debug(
                "LiberoAdapter: captured canonical qpos snapshot (nq=%d, nv=%d)",
                self._canonical_qpos.shape[0],
                self._canonical_qvel.shape[0],
            )
            return

        # Subsequent episode - restore the snapshot. The needs_capture
        # check above guarantees the snapshot fields are non-None here;
        # narrow for mypy.
        assert self._canonical_qpos is not None
        assert self._canonical_qvel is not None
        canonical_qpos = self._canonical_qpos
        canonical_qvel = self._canonical_qvel
        try:
            if lock is not None:
                with lock:
                    np.copyto(qpos, canonical_qpos)
                    np.copyto(qvel, canonical_qvel)
                    mj.mj_forward(model, data)
            else:
                np.copyto(qpos, canonical_qpos)
                np.copyto(qvel, canonical_qvel)
                mj.mj_forward(model, data)
        except Exception as e:  # noqa: BLE001 - never fatal
            logger.warning("LiberoAdapter: snapshot restore failed: %s", e)
            return
        logger.debug("LiberoAdapter: restored canonical qpos snapshot")

    def _write_libero_arm_home_qpos(self, model: Any, data: Any, mj: Any, lock: Any) -> None:
        """Write the LIBERO Panda + gripper ready pose into ``data.qpos``.

        Mirrors upstream ``robosuite.robots.robot.Robot.reset(deterministic=True)``
        which writes ``self.init_qpos`` (= ``MountedPanda.init_qpos`` for
        LIBERO) to the arm joint qpos addresses BEFORE constructing the
        OSC controller. Also mirrors ``SingleArm.reset`` which writes
        ``self.gripper.init_qpos`` (= ``PandaGripper.init_qpos`` =
        ``[0.020833, -0.020833]``) to the gripper finger joint qpos
        addresses. Used by :meth:`_apply_snapshot_branch` to bring the
        BDDL-default state into parity with upstream
        ``OffScreenRenderEnv``.

        Best-effort: any failure (no Panda joints, missing libero
        package, model has no ``arm joint``s) is logged at DEBUG and
        falls through silently, leaving ``data.qpos`` at whatever
        MuJoCo's default was. Caller's snapshot branch then captures
        the un-modified qpos.

        #176 (sub-task 3d).
        """
        # Resolve arm joint qpos addresses (same scan as
        # _LiberoOSCController.from_sim).
        arm_qpos_addrs: list[int] = []
        gripper_qpos_addrs: list[int] = []
        njnt = int(getattr(model, "njnt", 0))
        for i in range(njnt):
            jname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
            if not isinstance(jname, str):
                continue
            if jname.startswith(self._scene_robot_prefix) and not jname.startswith(self._scene_gripper_prefix):
                arm_qpos_addrs.append(int(model.jnt_qposadr[i]))
            elif jname.startswith(self._scene_gripper_prefix):
                # Filter to finger joints only — match #168 convention
                # (gripper0_finger_joint1, gripper0_finger_joint2).
                if "finger_joint" in jname:
                    gripper_qpos_addrs.append(int(model.jnt_qposadr[i]))
        if len(arm_qpos_addrs) != 7:
            logger.debug(
                "LiberoAdapter._write_libero_arm_home_qpos: expected 7 arm joints with prefix %r, "
                "found %d; skipping home-pose write",
                self._scene_robot_prefix,
                len(arm_qpos_addrs),
            )
            return

        home_qpos = _resolve_libero_arm_home_qpos(len(arm_qpos_addrs))
        if home_qpos is None:
            logger.debug(
                "LiberoAdapter._write_libero_arm_home_qpos: no LIBERO/robosuite home pose available; "
                "skipping home-pose write"
            )
            return

        # Resolve gripper init_qpos (PandaGripper). Same caching pattern
        # as arm home pose for cheap repeat lookup.
        gripper_init = _resolve_panda_gripper_init_qpos(len(gripper_qpos_addrs))

        def _do_write() -> None:
            for adr, v in zip(arm_qpos_addrs, home_qpos, strict=True):
                data.qpos[adr] = float(v)
            if gripper_init is not None and len(gripper_qpos_addrs) == len(gripper_init):
                for adr, v in zip(gripper_qpos_addrs, gripper_init, strict=True):
                    data.qpos[adr] = float(v)
            mj.mj_forward(model, data)

        try:
            if lock is not None:
                with lock:
                    _do_write()
            else:
                _do_write()
        except Exception as e:  # noqa: BLE001 - best-effort, never fatal
            logger.debug(
                "LiberoAdapter._write_libero_arm_home_qpos: write failed (%s); snapshot will capture pre-write qpos",
                e,
            )
            return
        logger.debug(
            "LiberoAdapter: wrote LIBERO Panda + gripper home pose into data.qpos (snapshot pre-capture); "
            "arm=%d, gripper=%d",
            len(arm_qpos_addrs),
            len(gripper_qpos_addrs),
        )

    def _apply_init_jitter(self, sim: SimEngine, rng: random.Random) -> None:
        """Apply ±jitter to xy of every body referenced by ``(:init (on A B))``.

        Best-effort: if the sim doesn't expose ``move_object`` / ``get_body_state``,
        or the body isn't in the scene, silently skip. This matches LIBERO's
        "small random perturbation per episode" convention without requiring
        full BDDL init semantics.
        """
        move_object = getattr(sim, "move_object", None)
        if move_object is None:
            logger.debug("LiberoAdapter: sim has no move_object(); skipping init jitter")
            return
        get_body_state = getattr(sim, "get_body_state", None)
        if get_body_state is None:
            return

        # Gather the set of bodies we want to jitter - BDDL init uses the same
        # Pred grammar, so (on cube_1 table_1) means "jitter cube_1".
        from strands_robots.benchmarks.libero.bddl_parser import Pred as _Pred

        seen: set[str] = set()
        for node in self.problem.init:
            for body in _extract_init_targets(node):
                seen.add(body)
        _ = _Pred  # referenced for clarity; actual test is inside _extract_init_targets

        for body in sorted(seen):
            try:
                state = get_body_state(body_name=body)
            except Exception as e:  # noqa: BLE001 - defensive
                logger.debug("jitter lookup for %r failed: %s", body, e)
                continue
            if not isinstance(state, dict) or state.get("status") != "success":
                continue
            pos = _extract_position(state)
            if pos is None:
                continue
            jx = rng.uniform(-self._init_jitter, self._init_jitter)
            jy = rng.uniform(-self._init_jitter, self._init_jitter)
            new_pos = [pos[0] + jx, pos[1] + jy, pos[2]]
            try:
                move_object(name=body, position=new_pos)
            except Exception as e:  # noqa: BLE001 - jitter failures are not fatal
                logger.debug("jitter apply for %r failed: %s", body, e)


def _extract_init_targets(node: Node) -> list[str]:
    """Return the first-arg body name of every leaf predicate in ``node``.

    Init clauses like ``(on cube_1 table_1)`` and ``(upright bottle_1)``
    share the convention that the first argument is the "subject" body -
    the thing whose position we may want to jitter. Nested
    ``and``/``or``/``not`` are traversed; non-predicates are ignored.
    """
    from strands_robots.benchmarks.libero.bddl_parser import And, Not, Or, Pred

    if isinstance(node, Pred):
        return [node.args[0]] if node.args else []
    if isinstance(node, (And, Or)):
        out: list[str] = []
        for c in node.clauses:
            out.extend(_extract_init_targets(c))
        return out
    if isinstance(node, Not):
        return _extract_init_targets(node.clause)
    return []


def _extract_position(state: dict[str, Any]) -> list[float] | None:
    """Pull ``{"json": {"position": [...]}}`` from a status-dict payload."""
    for block in state.get("content", []) or []:
        if isinstance(block, dict) and isinstance(block.get("json"), dict):
            pos = block["json"].get("position")
            if isinstance(pos, list) and len(pos) == 3 and all(isinstance(c, (int, float)) for c in pos):
                return [float(c) for c in pos]
    return None


def _walk_predicate_tree(node: Any, sim: SimEngine) -> list[tuple[str, bool]]:
    """Walk a BDDL goal tree and return ``(repr, verdict)`` for every leaf.

    #170 diagnostic helper. Originally used by
    :meth:`LiberoAdapter.is_success` to identify which sub-predicate
    disagreed with the (now-removed, see #178) ``env.check_success``
    delegation path. Kept for any future BDDL-evaluator debugging:
    compiles each ``Pred`` leaf in isolation via the existing
    ``compile_goal`` machinery and runs it against the live ``sim``.

    Each reported leaf string includes the actual body positions
    looked up via ``sim.get_body_state`` so we can see why a position-
    based predicate (``on``, ``inside``, etc.) returned its verdict —
    distinguishes between "name doesn't resolve" (None pos), "wrong
    geometric threshold" (positions present but predicate still False),
    and "true success" (positions present and predicate True).

    Returns a list of ``(human-readable predicate, verdict)`` tuples,
    one per leaf, in tree-traversal order. Combinators (``And`` / ``Or``
    / ``Not``) are reported as nested-string prefixes.
    """
    from strands_robots.benchmarks.libero.bddl_parser import And, Not, Or, Pred, compile_goal
    from strands_robots.simulation.predicates import _body_position

    out: list[tuple[str, bool]] = []

    def _arg_diag(arg: str) -> str:
        """Render a body name + its position for diagnostic output."""
        pos = _body_position(sim, arg)
        if pos is None:
            return f"{arg}=NONE"
        return f"{arg}=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"

    def visit(n: Any, prefix: str = "") -> None:
        if isinstance(n, Pred):
            try:
                verdict = bool(compile_goal(n)(sim))
            except Exception as e:  # noqa: BLE001
                out.append((f"{prefix}({n.name} {' '.join(n.args)})", False))
                logger.debug("predicate_log: leaf %r raised %s", n, e)
                return
            arg_diag = " ".join(_arg_diag(a) for a in n.args)
            out.append((f"{prefix}({n.name} {arg_diag})", verdict))
            return
        if isinstance(n, And):
            for c in n.clauses:
                visit(c, prefix + "AND/")
            return
        if isinstance(n, Or):
            for c in n.clauses:
                visit(c, prefix + "OR/")
            return
        if isinstance(n, Not):
            visit(n.clause, prefix + "NOT/")
            return

    visit(node)
    return out


def _extract_pose(state: dict[str, Any] | None) -> tuple[list[float] | None, list[float] | None]:
    """Pull ``(position, quaternion_wxyz)`` from a ``get_body_state`` payload.

    Both fields are optional; this returns ``(None, None)`` for any
    error / shape mismatch so the caller can selectively inject just
    the keys it has. The MuJoCo backend always reports both, so in
    the happy path you get both arrays back.
    """
    if not isinstance(state, dict) or state.get("status") != "success":
        return (None, None)
    pos: list[float] | None = None
    quat: list[float] | None = None
    for block in state.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        json_block = block.get("json")
        if not isinstance(json_block, dict):
            continue
        raw_pos = json_block.get("position")
        if isinstance(raw_pos, list) and len(raw_pos) == 3 and all(isinstance(c, (int, float)) for c in raw_pos):
            pos = [float(c) for c in raw_pos]
        raw_quat = json_block.get("quaternion")
        if isinstance(raw_quat, list) and len(raw_quat) == 4 and all(isinstance(c, (int, float)) for c in raw_quat):
            quat = [float(c) for c in raw_quat]
    return (pos, quat)


def _fmt_state_value(value: Any) -> str:
    """Format a state value for ``STATE_LOG`` output (#168).

    Returns ``"None"`` for missing keys, scalar floats rounded to 6dp,
    and list/tuple/ndarray values rounded element-wise. Matches the
    ``np.round(..., 6).tolist()`` style of #168's ACTION_LOG so
    a single grep yields parseable values across both diagnostic
    streams.
    """
    if value is None:
        return "None"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.6f}"
    if isinstance(value, (list, tuple)):
        return str([round(float(v), 6) if isinstance(v, (int, float)) else v for v in value])
    if isinstance(value, np.ndarray):
        return str(np.round(value, 6).tolist())
    return repr(value)


def _quat_wxyz_to_rpy_xyz(quat_wxyz: list[float]) -> tuple[float, float, float]:
    """MuJoCo ``(w, x, y, z)`` quaternion → extrinsic XYZ Euler ``(roll, pitch, yaw)``.

    Matches RoboSuite/LIBERO's ``mat2euler(..., axes='sxyz')`` convention
    byte-for-byte. RoboSuite implements ``sxyz`` as the extraction
    ``(atan2(M[2,1], M[2,2]), asin(-M[2,0]), atan2(M[1,0], M[0,0]))``
    where ``M`` is the rotation matrix in Hamilton convention.

    For unit quat ``q = (w, x, y, z)``, the rotation-matrix entries
    needed for the canonical extraction are (Hamilton convention,
    standard active rotation):

        M[0,0] = 1 - 2(y² + z²)
        M[1,0] = 2(xy + wz)
        M[2,0] = 2(xz - wy)
        M[2,1] = 2(yz + wx)
        M[2,2] = 1 - 2(x² + y²)

    Hence:

        roll  = atan2(M[2,1], M[2,2]) = atan2(2(wx + yz), 1 - 2(x² + y²))
        pitch = asin(-M[2,0])         = asin(2(wy - xz))
        yaw   = atan2(M[1,0], M[0,0]) = atan2(2(wz + xy), 1 - 2(y² + z²))

    Pure numpy / stdlib — **does not import scipy**, which is not a
    declared dependency of strands_robots. Equivalent to
    ``robosuite.utils.transform_utils.mat2euler(R, axes='sxyz')``
    on the rotation matrix derived from ``q``.

    **#176 (sub-task 3d)** corrected three sign errors in the
    pre-46 derivation. The pre-46 formula was based on
    ``R = R_x · R_y · R_z`` (intrinsic XYZ, NOT extrinsic), giving:

        roll  = atan2(2(wx - yz), ...)  [WRONG SIGN on yz]
        pitch = asin(2(wy + xz))         [WRONG SIGN on xz]
        yaw   = atan2(2(wz - xy), ...)  [WRONG SIGN on xy]

    These produced sign-flipped Euler angles for the EEF wrist body
    on the LIBERO ready pose (verified empirically against
    robosuite's ``mat2euler``: a quat=(0, 0.707, 0.707, 0) returns
    (π, 0, π/2) under sxyz but the pre-46 formula returned (-π, 0,
    -π/2)). The sign flip in yaw is genuinely a different rotation
    (left vs right twist about z). The state.yaw reported to the
    policy was therefore systematically inverted relative to
    training data, putting the policy out-of-distribution from the
    very first observation. (The state-parity test passed at canonical
    init because canonical-pose yaw is ~0.0005 rad and the test uses
    ``_angle_diff_mod_2pi`` which masked the sign flip; the test
    should also be made to forbid sign-flips for non-π rotations.)

    Gimbal lock (``|sin(pitch)| ≥ 1 - 1e-6``) collapses roll into yaw;
    we use the ``atan2(-M[1,2], M[1,1])`` resolution that matches
    scipy / robosuite.

    Returns:
        ``(roll, pitch, yaw)`` in **radians**, each in the principal
        range used by ``atan2`` / ``asin``: ``roll ∈ (-π, π]``,
        ``pitch ∈ [-π/2, π/2]``, ``yaw ∈ (-π, π]``.
    """
    import math

    w, x, y, z = quat_wxyz
    # Clamp asin argument to handle minor numerical drift on unit quats.
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - x * z)))
    pitch = math.asin(sin_pitch)
    if abs(sin_pitch) >= 1.0 - 1e-6:
        # Gimbal lock: roll absorbed into yaw. Use the same fallback
        # robosuite's ``mat2euler`` does (atan2(-M[1,2], M[1,1])).
        roll = 0.0
        # M[1,2] = 2(yz - wx); M[1,1] = 1 - 2(x² + z²).
        yaw = math.atan2(-2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + z * z))
    else:
        # M[2,1] = 2(yz + wx); M[2,2] = 1 - 2(x² + y²).
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        # M[1,0] = 2(xy + wz); M[0,0] = 1 - 2(y² + z²).
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (roll, pitch, yaw)


# Scene-generation helpers (#164)


def _default_scene_cache_dir() -> Path:
    """Filesystem location for cached LIBERO scene MJCFs.

    Uses :func:`strands_robots.utils.get_base_dir` so the cache lives
    under ``$STRANDS_BASE_DIR`` (typically ``~/.strands_robots/``)
    alongside other strands-robots state. Created on demand by
    :meth:`LiberoAdapter._generate_scene_from_bddl` - this helper just
    returns the path.
    """
    return get_base_dir() / "scene_cache" / "libero"


def _extract_compiled_mjcf(env: Any) -> str:
    """Pull the MJCF XML out of a ``libero`` ControlEnv.

    Prefers ``env.env.model.get_xml()`` (the **pre-compile** MJCF emitted by
    robosuite's ``ManipulationTask`` — what gets handed to
    ``MjModel.from_xml_string`` at construction time). Falls back to
    ``env.env.sim.model.get_xml()`` (a **post-compile** re-serialization via
    ``mj_saveLastXML``) if the pre-compile accessor isn't available.

    **Why pre-compile is preferred (#181).** ``mj_saveLastXML`` is a lossy
    round-trip — it doesn't preserve every ``<compiler>`` attribute. In
    particular ``inertiagrouprange="0 0"`` is dropped, so re-loading the
    saved XML re-computes body inertias from ALL geom groups (including
    visual meshes) rather than just collision geoms (group 0). Effect on
    ``libero-10/SCENE5``: re-loaded model has table_col mass 77.5 kg vs
    upstream's 34.5 kg, mug masses 4× heavier, etc.

    The pre-compile path preserves ``<compiler inertiagrouprange="0 0"/>``
    verbatim, so re-loading byte-identically reproduces upstream's runtime
    masses (table_col=34.54 kg, mug=0.0165 kg). Verified empirically by
    diffing ``env.env.sim.model._model`` (upstream runtime) vs
    ``mujoco.MjModel.from_xml_path(cached.xml)`` for both paths.

    Robosuite renamed accessors over the years; we try a small set of
    fallbacks before giving up. Never looks at any non-public attributes.
    """
    accessors = (
        # #181 — preferred: pre-compile MJCF from robosuite's
        # ``ManipulationTask``. Preserves ``<compiler>`` attributes that
        # ``mj_saveLastXML`` drops (notably ``inertiagrouprange``).
        lambda: env.env.model.get_xml(),
        # Older robosuite path: ``ManipulationEnv`` subclasses sometimes
        # expose the compiled XML via an explicit helper.
        lambda: env.env.model.get_model_xml(),  # type: ignore[attr-defined]
        # Last resort: post-compile re-serialization. Lossy on
        # ``<compiler>`` attributes (see #181); used only if the
        # pre-compile accessors aren't available (very old robosuite).
        lambda: env.env.sim.model.get_xml(),
    )
    last_err: Exception | None = None
    for accessor in accessors:
        try:
            xml = accessor()
        except Exception as e:  # noqa: BLE001 - try the next accessor
            last_err = e
            continue
        if isinstance(xml, str) and xml.strip():
            return xml
    raise RuntimeError(f"could not extract MJCF from libero env (last error: {last_err!r})")


# Match a complete ``<camera ... name="OLD" ...>`` declaration so the
# rename only touches camera definitions, not e.g. material names that
# happen to share a string. Anchored on the ``camera`` element name and
# guarded by ``\s+`` to avoid partial-word matches.
_CAMERA_NAME_RE = re.compile(r'(<camera\b[^>]*\bname=")([^"]+)(")')


def _rename_mjcf_cameras(xml: str, aliases: dict[str, str]) -> str:
    """Rename ``<camera name="OLD"...>`` → ``<camera name="NEW"...>`` per ``aliases``.

    Targeted regex only - we don't parse the whole MJCF. The rename is
    safe because MuJoCo doesn't allow duplicate ``<camera>`` names within
    a model, and camera references from external code (e.g.
    ``sim.render(camera_name=...)``) come from outside the XML so they
    aren't affected.

    Names not in ``aliases`` pass through unchanged.
    """
    if not aliases:
        return xml

    def _sub(match: re.Match[str]) -> str:
        head, name, tail = match.group(1), match.group(2), match.group(3)
        return head + aliases.get(name, name) + tail

    return _CAMERA_NAME_RE.sub(_sub, xml)


# Bumped whenever the BDDL -> MJCF transform pipeline changes its
# semantics. Hashed into the scene-cache key by
# :meth:`LiberoAdapter._scene_cache_key` so stale on-disk caches
# generated by prior versions get auto-invalidated when users pick up
# the upgrade.
#
# History:
# * ``v1``: implicit (pre-#168, alias map alone in cache key).
# * ``v2``: #168 (initial attempt) — applied ``_apply_libero_visual_fixes`` (rgba alpha=0 on
#   collision geoms + custom ``<visual>`` block with stacked ``<headlight>``).
#   Empirically wrong direction (washed out contrast); reverted in v3.
# * ``v3``: #168 (subsequent revert) — cached MJCF matches upstream ``OffScreenRenderEnv.sim.model.get_xml()``
#   verbatim except for the camera-name aliases. Visual fidelity is
#   handled at render time via ``world._backend_state["viz_option"]``
#   (set by :meth:`LiberoAdapter._install_render_options`), not via
#   MJCF rewrites.
# * ``v4``: #181 — switch ``_extract_compiled_mjcf`` to prefer
#   ``env.env.model.get_xml()`` (pre-compile) over
#   ``env.env.sim.model.get_xml()`` (post-compile via ``mj_saveLastXML``).
#   The post-compile path is lossy on ``<compiler>`` attributes, dropping
#   ``inertiagrouprange="0 0"`` which restricts inertia computation to
#   collision geoms (group 0). Without it, MuJoCo computes body masses
#   from ALL geom groups including visual meshes — table mass jumps from
#   34.5 kg to 77.5 kg, mug masses go up 4×, and ``mj_step`` produces
#   different ``qfrc_bias`` (Coriolis + gravity) than upstream's runtime
#   model. Effect on libero-10/SCENE5: end-to-end ``success_rate`` mean
#   over 5 master seeds × 5 episodes drops from ~0.72 (offscreen) to
#   ~0.52 (mujoco) entirely from this single ``<compiler>`` attr drop.
_LIBERO_MJCF_TRANSFORM_VERSION = "v4"


def _build_scene_robot_wrapper(
    mj: Any,
    model: Any,
    *,
    prefix: str,
    gripper_prefix: str | None = None,
) -> SimRobot | None:
    """Construct a :class:`SimRobot` for an existing scene-supplied Panda.

    Walks the compiled MuJoCo ``model`` looking for bodies / joints /
    actuators whose names start with ``prefix`` (default ``"robot0_"``,
    matching RoboSuite / LIBERO's canonical naming for the arm). When
    ``gripper_prefix`` is non-empty, joints AND actuators starting with
    *either* prefix are included in the wrapper - this is critical for
    RoboSuite-emitted scenes because the gripper has its own namespace
    (``gripper0_``) separate from the arm's (``robot0_``). Without the
    gripper prefix, ``gripper0_finger_joint{1,2}`` and the gripper
    actuator would be silently dropped from ``wrapper.joint_names`` and
    ``wrapper.actuator_ids``, the upstream observation pipeline would
    omit them from ``obs``, and downstream code looking up
    ``obs.get("gripper0_finger_joint1")`` (e.g.
    :meth:`LiberoAdapter.augment_observation` after
    :meth:`_resolve_scene_eef_and_gripper` resolves the gripper joint)
    would silently get ``None``. That manifests as
    ``state.gripper`` being omitted from the GR00T request and the
    server rejecting with ``State key 'state.gripper' must be in
    observation`` (#168 bug G).

    Body discovery uses ``prefix`` only - in RoboSuite/LIBERO MJCFs the
    gripper is mounted as a child body of the arm's last link, not a
    root-level body, so its bodies don't need to be in the joint-pool
    filter. We just need the joints / actuators that move it.

    Returns a :class:`SimRobot` whose IDs and namespace are filled in
    from the discovered names, or ``None`` when no body matches the arm
    prefix.

    The returned wrapper is only useful for **populating
    ``world.robots``** so ``BenchmarkProtocol.on_episode_start``'s
    ``list_robots()`` check returns non-empty. It is NOT a substitute
    for the wrapper that ``Simulation.add_robot`` builds via
    ``inject_robot_into_scene`` - that call also recompiles the spec
    and registers tendon / actuator side-effects we don't want here.
    The whole point of this discovery path is to avoid that recompile.

    Body / joint / actuator IDs are read from the *current* compiled
    model. If the spec is recompiled later (e.g. by
    ``_install_libero_cameras``), the IDs may no longer be valid; the
    adapter relies on its model-side camera detection (#167 / #166
    follow-up) to keep that recompile from firing.

    Returns ``None`` when:

    * No body name starts with ``prefix``.
    * ``model`` doesn't expose ``nbody`` / ``njnt`` / ``nu`` (e.g. a
      stub injected by tests).

    Discovery never raises - any unexpected MuJoCo error is caught at
    the call site in :meth:`LiberoAdapter._register_default_robot` and
    surfaced as a WARNING-and-continue.
    """
    nbody = int(getattr(model, "nbody", 0))
    njnt = int(getattr(model, "njnt", 0))
    nu = int(getattr(model, "nu", 0))
    if nbody == 0 or njnt == 0:
        return None

    # Find the root body of the scene-supplied robot - the first body
    # whose name starts with ``prefix`` and whose parent is the world
    # body (id 0). Falls back to the first match if no clear root is
    # found, which is acceptable for the wrapper's purposes (we only
    # need IDs for the compatibility check, not for kinematic queries).
    root_body_id = -1
    for i in range(nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        if not isinstance(name, str) or not name.startswith(prefix):
            continue
        if root_body_id < 0:
            root_body_id = i
        # Prefer a body whose parent is the world; treat that as canonical.
        body_parentid = getattr(model, "body_parentid", None)
        if body_parentid is not None:
            try:
                if int(body_parentid[i]) == 0:
                    root_body_id = i
                    break
            except (IndexError, TypeError):
                pass
    if root_body_id < 0:
        return None

    # Build the prefix-set for joint/actuator filtering. RoboSuite-style
    # scenes split the arm and gripper into TWO namespaces (e.g.
    # ``robot0_*`` for arm joints, ``gripper0_*`` for gripper finger
    # joints). The wrapper needs to include both so the upstream
    # observation pipeline surfaces ``state.gripper`` to the policy.
    joint_prefixes: tuple[str, ...] = (prefix,)
    if gripper_prefix:
        joint_prefixes = (prefix, gripper_prefix)

    def _starts_with_any(name: object) -> bool:
        return isinstance(name, str) and any(name.startswith(p) for p in joint_prefixes if p)

    joint_names: list[str] = []
    joint_ids: list[int] = []
    for i in range(njnt):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
        if _starts_with_any(name):
            joint_names.append(name)  # type: ignore[arg-type]
            joint_ids.append(i)

    actuator_ids: list[int] = []
    for i in range(nu):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i)
        if _starts_with_any(name):
            actuator_ids.append(i)

    return SimRobot(
        name="robot",  # registered key matches super()'s default
        urdf_path="",  # scene-supplied, no upstream URDF
        data_config="panda",  # LIBERO is Panda-only
        body_id=root_body_id,
        joint_names=joint_names,
        joint_ids=joint_ids,
        actuator_ids=actuator_ids,
        namespace=prefix,
    )


class _ControllerInstallError(RuntimeError):
    """Raised inside :meth:`_LiberoOSCController.from_sim` when the
    LIBERO action controller can't be built (missing robosuite,
    missing site/actuator IDs, etc.). Caught by
    :meth:`LiberoAdapter._install_action_controller` and converted to
    a WARNING log + silent fall-through to ``_apply_action_by_name``."""


class _LiberoOSCController:
    """OSC_POSE controller wrapper for GR00T-LIBERO action dispatch (#168).

    Converts task-space delta-EEF actions
    (``{x, y, z, roll, pitch, yaw, gripper}``) into joint torques
    via RoboSuite's ``OperationalSpaceController``. Wraps the 7th
    ``gripper`` channel separately as a direct write to gripper
    actuators (RoboSuite's OSC ignores the gripper).

    Architecture: holds a robosuite ``MjSim`` shim around the
    sim's compiled MuJoCo model + data, plus an ``OSC_POSE``
    controller bound to the arm's 7 joint indices and EEF site.
    The shim is constructed once at episode start and reused for
    every action call. ``apply()`` runs the controller with the
    incoming 6-dim Cartesian delta and writes the resulting
    torques to ``data.ctrl[arm_actuator_ids]``; the gripper
    channel writes to ``data.ctrl[gripper_actuator_ids]`` directly.

    Lifecycle: bound to one specific compiled model. If the spec
    is recompiled (via ``load_scene``, ``add_camera`` triggering a
    spec-rebuild, etc.) the controller's stored joint / actuator IDs
    become stale. ``LiberoAdapter._install_action_controller`` is
    called from prewarm + on_episode_start to keep the controller
    fresh per episode.
    """

    # Tells the SimEngine that this controller drives ``mj_step`` itself
    # (#168): one ``apply()`` advances physics by
    # ``physics_substeps_per_control`` steps, recomputing OSC torques each
    # step. Without this flag, ``_apply_sim_action`` would call ``mj_step``
    # again after ``apply()`` and double-step (or worse, leave a stale
    # ``data.ctrl`` write between policy steps).
    owns_stepping: bool = True

    # PandaGripper.format_action ramp constant (robosuite/models/grippers/
    # panda_gripper.py: ``self.speed = 0.01``). The gripper's normalized
    # ``current_action`` (2-vector in [-1, +1]) is incremented by
    # ``[-1, +1] * speed * sign(input)`` per substep, slowly ramping toward
    # full close (+1 → both fingers at clipped target) or full open (-1 →
    # opposite). #168: replicate this ramp instead of writing
    # the raw GR00T scalar to ``data.ctrl``, which previously caused one
    # finger to actuate in the wrong direction (finger1 has ctrlrange
    # [0, 0.04] where +1 clips to OPEN, but the format_action saturation
    # writes -1 = CLOSED to it).
    _GRIPPER_SPEED: ClassVar[float] = 0.01

    def __init__(
        self,
        controller: Any,
        sim_shim: Any,
        eef_site_name: str,
        arm_actuator_ids: list[int],
        gripper_actuator_ids: list[int],
        model: Any,
        data: Any,
        physics_substeps_per_control: int = 25,
        eef_site_id: int = -1,
        arm_qpos_addrs: list[int] | None = None,
    ) -> None:
        self.controller = controller
        self.sim_shim = sim_shim
        self.eef_site_name = eef_site_name
        self.eef_site_id = int(eef_site_id)
        self.arm_actuator_ids = list(arm_actuator_ids)
        self.arm_qpos_addrs = list(arm_qpos_addrs) if arm_qpos_addrs is not None else []
        self.gripper_actuator_ids = list(gripper_actuator_ids)
        self.model = model
        self.data = data
        # LIBERO trains at 20 Hz control with 500 Hz physics → 25 physics
        # substeps per policy action. Mismatch ⇒ the OSC controller
        # under-/over-shoots its delta target every step, manifesting as
        # `success_rate=0` even though motion looks correct in cameras.
        # See PR #168 (verification then fix in subsequent commits).
        self.physics_substeps_per_control = max(1, int(physics_substeps_per_control))

        # Stateful ``current_action`` for the gripper, mirroring
        # ``robosuite.models.grippers.panda_gripper.PandaGripper.current_action``
        # which is initialised to ``np.zeros(self.dof)`` (dof=1) and ramps
        # up to a 2-vector via numpy broadcasting on the first
        # ``format_action`` call. We init directly as a 2-vector since we
        # always have 2 fingers; semantics are identical.
        self._gripper_current_action: np.ndarray = np.zeros(2, dtype=np.float64)

        # Pre-compute bias / weight per gripper actuator for the
        # ``[-1, +1] → [ctrl_lo, ctrl_hi]`` rescaling done in
        # ``robosuite.robots.manipulator.Manipulator.grip_action``:
        #   bias = 0.5 * (hi + lo)
        #   weight = 0.5 * (hi - lo)
        #   data.ctrl[gripper] = bias + weight * format_action_output
        # Cached once at install time (ctrlrange is per-model immutable);
        # avoids re-reading model.actuator_ctrlrange at 25 Hz × 20 Hz.
        self._gripper_bias = np.array(
            [0.5 * (model.actuator_ctrlrange[gi, 1] + model.actuator_ctrlrange[gi, 0]) for gi in gripper_actuator_ids],
            dtype=np.float64,
        )
        self._gripper_weight = np.array(
            [0.5 * (model.actuator_ctrlrange[gi, 1] - model.actuator_ctrlrange[gi, 0]) for gi in gripper_actuator_ids],
            dtype=np.float64,
        )

        # #168 — diagnostic logging gate. Set
        # ``STRANDS_LIBERO_ACTION_LOG=1`` to emit one structured INFO
        # log line per ``apply()`` call for the first
        # ``STRANDS_LIBERO_ACTION_LOG_MAX`` (default 50) calls per
        # episode. Captures action keys, delta scale, gripper polarity,
        # EEF tracking, and qpos/ctrl deltas — answers the diagnostic
        # questions in PR #168 verification.
        self._action_log_enabled = os.environ.get("STRANDS_LIBERO_ACTION_LOG", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        try:
            self._action_log_max = int(os.environ.get("STRANDS_LIBERO_ACTION_LOG_MAX", "50"))
        except ValueError:
            logger.warning(
                "STRANDS_LIBERO_ACTION_LOG_MAX=%r is not an integer; defaulting to 50",
                os.environ.get("STRANDS_LIBERO_ACTION_LOG_MAX"),
            )
            self._action_log_max = 50
        self._action_log_step: int = 0

    def reset(self) -> None:
        """Reset stateful per-episode controller state.

        #168: the gripper's ``current_action`` is a stateful
        ramp accumulator (per ``PandaGripper.format_action``). Without a
        reset, the second episode starts with whatever finger position
        the first episode ended at — typically a partially-closed
        gripper, which biases every grasp attempt. Called from
        :meth:`LiberoAdapter._install_action_controller` (which itself is
        called from ``on_episode_start``) so each episode starts with a
        canonical ``current_action = [0, 0]``.

        #168: also resets the per-episode action-log step
        counter so each episode logs its own first N steps when
        ``STRANDS_LIBERO_ACTION_LOG=1`` is set.
        """
        self._gripper_current_action.fill(0.0)
        self._action_log_step = 0

    @classmethod
    def from_sim(
        cls,
        sim: SimEngine,
        *,
        eef_site_name: str,
        arm_prefix: str,
        gripper_prefix: str,
    ) -> _LiberoOSCController:
        """Build a controller bound to ``sim``'s loaded LIBERO scene.

        Discovers:
        - arm joints and qpos/qvel addresses (``robot0_joint1..7``)
        - arm actuator IDs (one per arm joint)
        - gripper actuator IDs (``gripper0_*`` actuators)
        - EEF site ID (e.g. ``gripper0_grip_site``)
        - actuator force/torque limits (``model.actuator_ctrlrange``)

        Raises :class:`_ControllerInstallError` with a diagnostic on
        any discovery failure. The caller (``_install_action_controller``)
        catches and logs at WARNING.
        """
        # Lazy imports - robosuite is a transitive dep via libero,
        # not pinned directly. Skip silently if either is unavailable.
        try:
            import mujoco as _mj
        except ImportError as e:
            raise _ControllerInstallError(f"mujoco not importable: {e}") from e
        try:
            from robosuite.controllers import (  # type: ignore[import-not-found]
                controller_factory,
                load_controller_config,
            )
            from robosuite.utils.binding_utils import MjSim  # type: ignore[import-not-found]
        except ImportError as e:
            raise _ControllerInstallError(f"robosuite not importable: {e}") from e

        world = getattr(sim, "_world", None)
        if world is None:
            raise _ControllerInstallError("sim has no _world")
        model = getattr(world, "_model", None)
        data = getattr(world, "_data", None)
        if model is None or data is None:
            raise _ControllerInstallError("sim._world has no compiled MuJoCo model/data")

        # 1. Discover arm joints (robot0_joint1..7).
        arm_joint_ids: list[int] = []
        arm_qpos_addrs: list[int] = []
        arm_qvel_addrs: list[int] = []
        njnt = int(getattr(model, "njnt", 0))
        for i in range(njnt):
            jname = _mj.mj_id2name(model, _mj.mjtObj.mjOBJ_JOINT, i)
            if not isinstance(jname, str) or not jname.startswith(arm_prefix):
                continue
            # Skip the gripper joints (different prefix; covered separately).
            if jname.startswith(gripper_prefix):
                continue
            arm_joint_ids.append(i)
            arm_qpos_addrs.append(int(model.jnt_qposadr[i]))
            # Each arm joint has 1 DoF (hinge), so qvel addr == joint id's
            # entry in jnt_dofadr. (For free joints this would be more
            # complex, but arm joints are hinges.)
            arm_qvel_addrs.append(int(model.jnt_dofadr[i]))
        if len(arm_joint_ids) != 7:
            raise _ControllerInstallError(
                f"expected 7 arm joints with prefix {arm_prefix!r}, found {len(arm_joint_ids)}"
            )

        # 2. Discover arm actuator IDs (one per arm joint).
        arm_actuator_ids: list[int] = []
        nu = int(getattr(model, "nu", 0))
        for jid in arm_joint_ids:
            for ai in range(nu):
                if int(model.actuator_trnid[ai, 0]) == jid:
                    arm_actuator_ids.append(ai)
                    break
            else:
                raise _ControllerInstallError(
                    f"no actuator found driving joint id={jid} (joint name "
                    f"{_mj.mj_id2name(model, _mj.mjtObj.mjOBJ_JOINT, jid)!r})"
                )

        # 3. Discover gripper actuator IDs (any actuator with gripper_prefix
        #    in its name).
        gripper_actuator_ids: list[int] = []
        for ai in range(nu):
            aname = _mj.mj_id2name(model, _mj.mjtObj.mjOBJ_ACTUATOR, ai)
            if isinstance(aname, str) and aname.startswith(gripper_prefix):
                gripper_actuator_ids.append(ai)
        if not gripper_actuator_ids:
            raise _ControllerInstallError(f"no gripper actuators with prefix {gripper_prefix!r}")

        # 4. Verify EEF site exists.
        site_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_SITE, eef_site_name)
        if site_id < 0:
            raise _ControllerInstallError(f"EEF site {eef_site_name!r} not found in model")

        # 5. Build robosuite MjSim shim around our model + data.
        # robosuite==1.4.0 ``MjSim.__init__(self, model)`` takes only
        # the model argument and creates a fresh internal
        # ``mujoco.MjData(model)`` accessible at ``sim_shim.data._data``.
        # That fresh data buffer is DISCONNECTED from our sim's
        # actual ``data`` - the controller would compute torques from
        # a stale, never-stepped buffer (#168 verification).
        # Hot-patch ``sim_shim.data._data`` to point at our actual
        # data so ``controller.run_controller()`` reads/writes the
        # same buffer the eval is stepping.
        sim_shim = MjSim(model)
        sim_shim.data._data = data

        # 6. Build OSC_POSE controller config + instance.
        controller_config = load_controller_config(default_controller="OSC_POSE")
        controller_config["robot_name"] = "Panda"
        controller_config["sim"] = sim_shim
        controller_config["eef_name"] = eef_site_name
        controller_config["joint_indexes"] = {
            "joints": arm_joint_ids,
            "qpos": arm_qpos_addrs,
            "qvel": arm_qvel_addrs,
        }
        # actuator_range is (low_arr, high_arr) for the ARM actuators
        # only - gripper is handled separately.
        ctrl_low = np.array(
            [float(model.actuator_ctrlrange[ai, 0]) for ai in arm_actuator_ids],
            dtype=np.float32,
        )
        ctrl_high = np.array(
            [float(model.actuator_ctrlrange[ai, 1]) for ai in arm_actuator_ids],
            dtype=np.float32,
        )
        controller_config["actuator_range"] = (ctrl_low, ctrl_high)

        # #176 — temporarily set ``data.qpos[arm]`` to the
        # LIBERO-canonical Panda "ready" pose around the
        # ``controller_factory`` call so the controller's
        # construction-time state capture matches upstream LIBERO
        # byte-for-byte.
        #
        # **Why this is needed.** Robosuite's ``BaseController.__init__``
        # ends with ``self.update_initial_joints(initial_joints)``,
        # which:
        #
        # 1. sets ``self.initial_joint = initial_joints``,
        # 2. calls ``self.update(force=True)`` reading ``ee_pos`` /
        #    ``ee_ori_mat`` / ``joint_pos`` from the *current*
        #    ``data.qpos``,
        # 3. stores those reads into ``self.initial_ee_pos`` and
        #    ``self.initial_ee_ori_mat``.
        #
        # ``OSC_POSE.__init__`` then sets
        # ``self.goal_pos = np.array(self.initial_ee_pos)`` and
        # ``self.goal_ori = np.array(self.initial_ee_ori_mat)``. So the
        # controller's *initial goal pose* is whatever ``data.qpos``
        # happened to be at construction time.
        #
        # Upstream LIBERO's ``Robot.reset(deterministic=True)``
        # (``robosuite/robots/robot.py``) writes ``self.init_qpos`` (=
        # ``MountedPanda.init_qpos`` = ``[0, -0.161, 0, -2.4446, 0,
        # 2.2268, π/4]``) into ``data.qpos[arm]`` BEFORE constructing
        # the controller, so ``initial_ee_pos`` / ``initial_ee_ori_mat``
        # / ``goal_pos`` / ``goal_ori`` all latch on to the ready-pose
        # EEF location. Then ``env.set_init_state(init_states[i])``
        # overwrites ``data.qpos`` with the per-episode canonical
        # state, but the controller's initial-state attributes are
        # frozen — leading to a non-trivial nullspace bias and a
        # ``goal_ori`` that's ~54 mrad different from the perturbed
        # canonical pose.
        #
        # Our ``LiberoAdapter._install_action_controller`` runs from
        # ``on_episode_start`` AFTER ``_apply_canonical_state`` has
        # already loaded the canonical ``init_states[i]`` into
        # ``data.qpos``, AND we never write the Panda ready pose
        # ourselves. The upstream-style ``Robot.reset`` step happens
        # implicitly via the LIBERO env construction in upstream's
        # path; we have no analog. So we replicate it here by
        # snapshotting ``data.qpos[arm]`` (the current canonical
        # values), writing the LIBERO Panda ready pose into those
        # addresses, calling ``mj_forward`` so site_xpos / site_xmat
        # reflect the home pose, building the controller (which
        # captures everything from the home-pose state), then
        # restoring ``data.qpos[arm]`` and calling ``mj_forward``
        # again. ``data.qvel`` is left as-is — upstream's
        # ``Robot.reset`` doesn't touch qvel, and our canonical
        # ``init_state`` is at zero velocity anyway.
        #
        # This fixes the ~50× torque divergence in #176 root cause: not
        # just the nullspace bias (``initial_joint``), but ALSO the
        # ``goal_ori`` initialization that without the swap would
        # latch on to the perturbed pose's EEF orientation.
        #
        # See ``probe_osc_internals_v3.py`` for the verification
        # showing torques match within tolerance after this swap.
        snapshot_qpos: np.ndarray | None = None
        home_qpos = _resolve_libero_arm_home_qpos(len(arm_qpos_addrs))
        if home_qpos is not None:
            try:
                # Snapshot current canonical arm qpos so we can restore.
                snapshot_qpos = np.array(
                    [float(data.qpos[adr]) for adr in arm_qpos_addrs],
                    dtype=np.float64,
                )
                # Write home pose into arm joint addresses.
                for adr, v in zip(arm_qpos_addrs, home_qpos, strict=True):
                    data.qpos[adr] = float(v)
                # Update derived state (site_xpos, site_xmat, jacobians)
                # so the controller's __init__ ``update(force=True)``
                # reads home-pose values.
                _mj.mj_forward(model, data)
            except (AttributeError, IndexError, TypeError, ValueError) as e:
                # Defensive: if we can't snapshot, don't proceed with
                # the swap (would leak home pose into the sim).
                logger.debug(
                    "_LiberoOSCController.from_sim: failed to swap qpos to home pose for "
                    "controller construction (%s); falling back to construction-time state",
                    e,
                )
                snapshot_qpos = None

        controller = controller_factory("OSC_POSE", controller_config)

        # Restore the per-episode canonical qpos and re-forward so
        # data is byte-identical to its pre-swap state. The
        # controller has already captured ``initial_*`` and ``goal_*``
        # at the home pose; subsequent ``update()`` calls will read
        # the restored canonical state, which is exactly upstream's
        # behavior post-``set_init_state``.
        if snapshot_qpos is not None:
            try:
                for adr, v in zip(arm_qpos_addrs, snapshot_qpos, strict=True):
                    data.qpos[adr] = float(v)
                _mj.mj_forward(model, data)
            except (AttributeError, IndexError, TypeError, ValueError) as e:
                # If restore fails the sim is in an unknown state;
                # raise loudly rather than silently leaving home pose.
                raise _ControllerInstallError(
                    f"failed to restore qpos after controller construction (snapshot was "
                    f"{snapshot_qpos.tolist()}): {e}. Sim may be in inconsistent state."
                ) from e

        # Compute physics-substeps-per-control from sim's actual timestep.
        # LIBERO trains at 20 Hz control rate. With dt=0.002 (default 500 Hz
        # physics), substeps = 25. This matches RoboSuite's standard step
        # loop in ``robosuite.environments.base.step``:
        #   for i in range(int(self.control_timestep / self.model_timestep)):
        #       self.sim.forward(); self._pre_action(...); self.sim.step()
        # which is what LIBERO's training data was generated with.
        dt = float(getattr(model.opt, "timestep", 0.002))
        substeps = max(1, int(round((1.0 / 20.0) / dt)))

        return cls(
            controller=controller,
            sim_shim=sim_shim,
            eef_site_name=eef_site_name,
            eef_site_id=int(site_id),
            arm_actuator_ids=arm_actuator_ids,
            arm_qpos_addrs=arm_qpos_addrs,
            gripper_actuator_ids=gripper_actuator_ids,
            model=model,
            data=data,
            physics_substeps_per_control=substeps,
        )

    def apply(
        self,
        action_dict: dict[str, Any],
        model: Any,
        data: Any,
        robot_name: str,  # noqa: ARG002 - kept for hook signature parity
    ) -> None:
        """Convert task-space delta-EEF action to joint torques + write data.ctrl.

        Reads from ``action_dict``: ``x, y, z, roll, pitch, yaw, gripper``.
        Writes to ``data.ctrl[arm_actuator_ids]`` (joint torques) and
        ``data.ctrl[gripper_actuator_ids]`` (gripper open/close).

        The OSC controller computes inverse Jacobian using
        ``data.xpos / xmat / qpos / qvel`` - which means
        :meth:`LiberoAdapter._forward_mj_data` must have run before
        this is first called (otherwise xpos/xmat are uninitialized).
        #168's ``mj_forward`` in ``Simulation.load_scene``
        guarantees this.

        **#168 control-rate fix.** LIBERO trains at 20 Hz control
        with 500 Hz physics → 25 physics substeps per policy action.
        We mirror RoboSuite's standard step loop
        (``robosuite.environments.base.Base.step``):

            set_goal(delta)                          # once
            for _ in range(physics_substeps):
                torques = controller.run_controller()
                data.ctrl[arm] = torques
                data.ctrl[gripper] = gripper_value
                mj_step(model, data)

        Each ``run_controller`` re-reads xpos/xmat/qpos/qvel/Jacobian
        via ``controller.update()`` (gated by the ``new_update`` flag
        which is set every iteration by the base class), so the OSC
        torques track the integrated state. Without the substep loop
        we ran OSC at 500 Hz (the physics rate) — the controller
        designed each torque profile for a 25-step horizon but only
        applied it for 1 step before the policy delivered a fresh
        delta, leading to the #168 "robot moves but never
        converges" symptom.

        ``owns_stepping = True`` tells the SimEngine not to call
        ``mj_step`` again after this returns; we've already advanced
        physics by the full control timestep.

        Best-effort against bad inputs: missing keys default to 0
        (no-op delta); shape mismatches log at WARNING and skip
        without raising.
        """
        # Refresh sim_shim's view of data (controller reads from
        # sim_shim.data.qpos / xpos / xmat). MjSim shim wraps our
        # data by reference, so this is a no-op in practice but
        # makes the assumption explicit.
        self.controller.update()

        # Pack 6-dim Cartesian delta: (dx, dy, dz, droll, dpitch, dyaw).
        # Missing keys default to 0 (no-op delta). Each per-key value
        # may be either a scalar or a 2-element list / array - GR00T-
        # LIBERO packs ALL action channels (x/y/z/roll/pitch/yaw/gripper)
        # to match the training-data shape, same convention PR #162
        # introduced for state.gripper. _to_scalar handles both forms
        # (and defensive against unexpected shapes) - #168 only
        # fixed gripper; #168 applies the same fix to every key.
        delta = np.array(
            [
                _to_scalar(action_dict.get("x", 0.0)),
                _to_scalar(action_dict.get("y", 0.0)),
                _to_scalar(action_dict.get("z", 0.0)),
                _to_scalar(action_dict.get("roll", 0.0)),
                _to_scalar(action_dict.get("pitch", 0.0)),
                _to_scalar(action_dict.get("yaw", 0.0)),
            ],
            dtype=np.float64,
        )
        # Default 0.5 (RLDS midway = no command) so an action dict
        # WITHOUT a gripper key produces no gripper movement. #168
        # added the RLDS→robosuite conversion ``-sign(2*v - 1)`` which
        # maps a default of 0.0 (RLDS close) to +1 (LIBERO close),
        # silently closing the gripper on every empty-dict
        # ``send_action({})`` call. With default 0.5 the conversion
        # gives 0 → no ramp → ``current_action`` stays at its previous
        # value → ctrl = bias (mid-range, holds open). #171 sub-task 3c
        # found this via the deep-rollout state parity test: gripper
        # drifted -19.6 mm in 50 zero-action steps while upstream stayed
        # at ~+20 mm.
        #
        # Production eval is unaffected — the policy always sends
        # ``gripper`` in the action chunk so the default never fires.
        gripper_value = _to_scalar(action_dict.get("gripper", 0.5))

        # #168 — convert RLDS gripper convention → robosuite/LIBERO
        # convention. The GR00T-N1.7-LIBERO checkpoint emits ``action.gripper``
        # in the RLDS dataloader's convention (``0 = close``, ``1 = open``);
        # robosuite's ``PandaGripper.format_action`` expects the opposite
        # (``+1 = close``, ``-1 = open``). NVIDIA bridges the two with two
        # helpers in ``Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_env.py``:
        #
        #   normalize_gripper_action(action, binarize=True):
        #     # [0, 1] → [-1, 1] → sign() → ±1
        #     action[..., -1] = 2 * action[..., -1] - 1
        #     action[..., -1] = np.sign(action[..., -1])
        #
        #   invert_gripper_action(action):
        #     # ±1 → ∓1 (RLDS → LIBERO sign convention)
        #     action[..., -1] = action[..., -1] * -1.0
        #
        # Combined: ``gripper_out = -np.sign(2 * gripper_in - 1)``. Concretely:
        #   gripper_in = 0.0 (RLDS close) → -sign(-1) = +1 (LIBERO close) ✓
        #   gripper_in = 0.5             → -sign(0)  =  0 (no motion)
        #   gripper_in = 1.0 (RLDS open)  → -sign(+1) = -1 (LIBERO open)  ✓
        #
        # Pre-#168 we passed the raw model output to our OSC's
        # ``np.sign(gripper_value)`` directly. Since the model's typical
        # outputs are in [0, 1], every "open" intent (model output ≈ 1)
        # collapsed to ``sign=+1``, which our OSC interprets as CLOSE — so
        # the gripper consistently went CLOSED for OPEN commands and
        # vice-versa. This is the action-side counterpart to #168's
        # observation-side V-flip bug; both rounds together close the
        # client-pipeline parity gap that left ``success_rate=0`` after
        # rounds 36-40 even though state pipeline was byte-equivalent
        # (#168) and image pipeline was within ``mean |Δ|=3-9/255``
        # (rounds 39+40).
        #
        # Diagnostic that surfaced this: NVIDIA's reference eval against
        # the SAME checkpoint+task got ``success_rate=1.0`` at 10s/ep,
        # while ours stayed at 0.0 at 120s/ep — clear ground-truth proof
        # the gap was on our side.
        gripper_value = -float(np.sign(2.0 * gripper_value - 1.0))

        # set_goal once per policy step. Subsequent run_controller
        # calls in the substep loop interpolate / hold this goal.
        try:
            self.controller.set_goal(delta)
        except Exception as e:  # noqa: BLE001 - log + skip rather than crash eval
            logger.warning(
                "_LiberoOSCController.apply: set_goal raised %s; this step's arm action will be no-op",
                e,
            )
            # Without a valid goal we still need to advance physics by the
            # full control timestep so the eval loop's timing is preserved
            # (otherwise sim time falls behind real time and benchmark
            # success criteria evaluated against ``cur_time`` go stale).
            import mujoco as mj

            for _ in range(self.physics_substeps_per_control):
                mj.mj_step(model, data)
            return

        # Cache mujoco module reference for the substep loop. Lazy import
        # is required because the OSC controller path is only exercised
        # under the `[sim-libero]` extra; the top-level adapter import
        # must work without mujoco available.
        import mujoco as mj

        n_arm = len(self.arm_actuator_ids)
        # Constant per-substep ramp for the gripper (#168). See
        # ``_GRIPPER_SPEED`` docstring above. Pre-compute the ramp
        # direction so the inner loop is just an in-place add + clip.
        # Sign of input dictates ramp direction: +1 (close) → finger
        # ``current_action`` ramps to ``[-1, +1]``; -1 (open) → ramps
        # to ``[+1, -1]``. The asymmetric direction is what causes the
        # "one finger goes the wrong way" bug if you write the raw
        # scalar to both finger ctrls.
        gripper_sign = float(np.sign(gripper_value))
        ramp_step = np.array([-1.0, 1.0]) * self._GRIPPER_SPEED * gripper_sign

        # #168 — capture pre-step state for diagnostic log.
        # Gated on ``STRANDS_LIBERO_ACTION_LOG=1`` so production eval
        # incurs zero cost (single bool check). The full diagnostic
        # answers PR #168's "what scale, what frame, what key
        # naming, does motion track delta" questions in one log line
        # per step.
        log_now = self._action_log_enabled and self._action_log_step < self._action_log_max
        if log_now:
            pre_eef_pos, pre_eef_quat = self._capture_eef_pose(data)
            pre_arm_ctrl = np.array([float(data.ctrl[ai]) for ai in self.arm_actuator_ids])
            pre_arm_qpos = (
                np.array([float(data.qpos[adr]) for adr in self.arm_qpos_addrs])
                if self.arm_qpos_addrs
                else np.zeros(n_arm)
            )
            pre_gripper_ctrl = np.array([float(data.ctrl[gi]) for gi in self.gripper_actuator_ids])
            pre_gripper_current = np.array(self._gripper_current_action)

        for _ in range(self.physics_substeps_per_control):
            # OSC: compute torques from current state (controller.update
            # is called inside run_controller via the new_update flag).
            try:
                torques = self.controller.run_controller()
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "_LiberoOSCController.apply: run_controller raised %s; "
                    "leaving previous data.ctrl in place for this substep",
                    e,
                )
                torques = None

            if torques is not None:
                torques_arr = np.asarray(torques, dtype=np.float64)
                if torques_arr.shape[0] != n_arm:
                    logger.warning(
                        "_LiberoOSCController.apply: torques shape %s != %d arm actuators; skipping arm ctrl write",
                        torques_arr.shape,
                        n_arm,
                    )
                else:
                    for ai, tq in zip(self.arm_actuator_ids, torques_arr, strict=True):
                        data.ctrl[ai] = float(tq)

            # Stateful gripper ramp + bias/weight rescale (#168
            # #168). Replicates the exact pipeline RoboSuite's
            # ``Manipulator.grip_action`` performs every substep:
            #   current_action = clip(current_action + [-1,+1]·speed·sign(input), -1, 1)
            #   ctrl = bias + weight * current_action
            # Without this, +1 (close) writes 0.04 to finger1 (range
            # [0, 0.04]) which actually OPENS finger1, breaking every
            # grasp. See PR #168 investigation.
            self._gripper_current_action = np.clip(
                self._gripper_current_action + ramp_step,
                -1.0,
                1.0,
            )
            applied_gripper = self._gripper_bias + self._gripper_weight * self._gripper_current_action
            for gi, val in zip(self.gripper_actuator_ids, applied_gripper, strict=True):
                data.ctrl[gi] = float(val)

            mj.mj_step(model, data)

        # #168 — emit one structured log line per apply()
        # while inside the captured-step window. Captures everything a
        # reviewer needs to bisect the residual bug: action key names,
        # delta scale, gripper polarity end-to-end, EEF tracking
        # (delta vs actual EEF motion), and qpos/ctrl deltas.
        if log_now:
            post_eef_pos, post_eef_quat = self._capture_eef_pose(data)
            post_arm_ctrl = np.array([float(data.ctrl[ai]) for ai in self.arm_actuator_ids])
            post_arm_qpos = (
                np.array([float(data.qpos[adr]) for adr in self.arm_qpos_addrs])
                if self.arm_qpos_addrs
                else np.zeros(n_arm)
            )
            post_gripper_ctrl = np.array([float(data.ctrl[gi]) for gi in self.gripper_actuator_ids])
            post_gripper_current = np.array(self._gripper_current_action)
            eef_pos_delta = post_eef_pos - pre_eef_pos
            logger.info(
                "ACTION_LOG step=%d "
                "action_keys=%s "
                "delta=%s gripper_value=%.4f "
                "eef_pos_pre=%s eef_pos_post=%s eef_pos_delta=%s "
                "eef_quat_pre=%s eef_quat_post=%s "
                "arm_ctrl_pre=%s arm_ctrl_post=%s "
                "arm_qpos_pre=%s arm_qpos_post=%s "
                "gripper_ctrl_pre=%s gripper_ctrl_post=%s "
                "gripper_current_pre=%s gripper_current_post=%s",
                self._action_log_step,
                sorted(action_dict.keys()),
                np.round(delta, 6).tolist(),
                gripper_value,
                np.round(pre_eef_pos, 6).tolist(),
                np.round(post_eef_pos, 6).tolist(),
                np.round(eef_pos_delta, 6).tolist(),
                np.round(pre_eef_quat, 4).tolist(),
                np.round(post_eef_quat, 4).tolist(),
                np.round(pre_arm_ctrl, 4).tolist(),
                np.round(post_arm_ctrl, 4).tolist(),
                np.round(pre_arm_qpos, 4).tolist(),
                np.round(post_arm_qpos, 4).tolist(),
                np.round(pre_gripper_ctrl, 6).tolist(),
                np.round(post_gripper_ctrl, 6).tolist(),
                np.round(pre_gripper_current, 4).tolist(),
                np.round(post_gripper_current, 4).tolist(),
            )
            self._action_log_step += 1

    def _capture_eef_pose(self, data: Any) -> tuple[np.ndarray, np.ndarray]:
        """Read EEF position + quaternion from ``data``.

        #168 helper for the diagnostic log path. Returns:
        - ``pos``: 3-vector ``data.site_xpos[eef_site_id]``
        - ``quat``: 4-vector unit quaternion (wxyz) computed from
          ``data.site_xmat[eef_site_id]`` via ``mju_mat2Quat``.

        Returns zero-filled arrays if ``eef_site_id < 0`` (e.g.
        controller built before the #168 changes; backwards
        compat for any pickled/test-injected instances).
        """
        if self.eef_site_id < 0:
            return np.zeros(3), np.zeros(4)
        pos = np.array(data.site_xpos[self.eef_site_id], dtype=np.float64)
        xmat = np.asarray(data.site_xmat[self.eef_site_id], dtype=np.float64).reshape(9)
        quat = np.zeros(4, dtype=np.float64)
        # Lazy import — adapter must be importable without mujoco.
        import mujoco as mj

        mj.mju_mat2Quat(quat, xmat)
        return pos, quat


# Module-level cache: resolving the LIBERO-canonical Panda home pose
# imports the libero robot module, which transitively imports robosuite,
# loads MJCF assets, and warns about missing macro files. The result is
# immutable per-process — cache it after the first lookup so calls 50×
# per second from ``_install_action_controller`` are free.
_CACHED_LIBERO_HOME_QPOS: np.ndarray | None = None
_CACHED_LIBERO_HOME_QPOS_RESOLVED: bool = False


def _resolve_libero_arm_home_qpos(n_arm: int) -> np.ndarray | None:
    """Return the LIBERO-canonical Panda 7-DoF home pose, or ``None``.

    Resolution order:

    1. ``libero.libero.envs.robots.mounted_panda.MountedPanda().init_qpos``
       — stock LIBERO layout (``pip install libero``).
    2. ``libero.envs.robots.mounted_panda.MountedPanda().init_qpos``
       — LIBERO-PRO layout (the LIBERO-PRO fork uses a flatter package
       structure). Same canonical ``init_qpos`` byte-for-byte.
    3. ``robosuite.models.robots.Panda().init_qpos`` (stock
       robosuite, slightly different — ``[0, π/16, 0, -π/2-π/3, 0,
       π-0.2, π/4]`` vs ``[0, -0.161, 0, -2.4446, 0, 2.2268, π/4]``).
       Only used when neither libero variant is importable.
    4. ``None`` if nothing is available — caller falls back to the
       controller's construction-time default.

    Both LIBERO layouts produce the same ``MountedPanda.init_qpos``,
    which is what upstream ``OffScreenRenderEnv``'s
    ``Robot.reset(deterministic=True)`` writes to ``data.qpos`` BEFORE
    constructing the OSC controller. That's the value upstream's
    ``controller.initial_joint`` ends up holding, and the value the
    #168 swap-and-restore in :meth:`_LiberoOSCController.from_sim`
    needs to write into ``data.qpos`` around ``controller_factory``.

    Refs #176.
    """
    global _CACHED_LIBERO_HOME_QPOS, _CACHED_LIBERO_HOME_QPOS_RESOLVED

    if _CACHED_LIBERO_HOME_QPOS_RESOLVED:
        if _CACHED_LIBERO_HOME_QPOS is None:
            return None
        if _CACHED_LIBERO_HOME_QPOS.shape != (n_arm,):
            # Different robot than the cached one (e.g. a 6-DoF arm).
            # Don't reuse the cached 7-DoF Panda value.
            return None
        return _CACHED_LIBERO_HOME_QPOS

    home: np.ndarray | None = None
    # Try both LIBERO package layouts. The class is identical between
    # them (same numerical init_qpos); only the import path differs.
    libero_module_paths = (
        "libero.libero.envs.robots.mounted_panda",  # stock libero
        "libero.envs.robots.mounted_panda",  # LIBERO-PRO
    )
    for module_path in libero_module_paths:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue
        mounted_panda_cls = getattr(module, "MountedPanda", None)
        if mounted_panda_cls is None:
            logger.debug(
                "%s exposes no MountedPanda; trying next libero layout",
                module_path,
            )
            continue
        try:
            # ``MountedPanda.init_qpos`` is a property that returns a
            # fresh ndarray on each access. Instantiating triggers
            # MJCF asset load (``robots/panda/robot.xml``); the property
            # itself reads from constants so the construction cost is
            # paid once per process.
            home = np.asarray(mounted_panda_cls().init_qpos, dtype=np.float64)
        except Exception as e:  # noqa: BLE001 - any failure soft-fall-through
            logger.debug(
                "MountedPanda(%s).init_qpos raised %s; trying next libero layout",
                module_path,
                e,
            )
            continue
        if home.shape != (n_arm,):
            logger.debug(
                "MountedPanda(%s).init_qpos shape %s does not match arm DoF %d; ignoring",
                module_path,
                home.shape,
                n_arm,
            )
            home = None
            continue
        # Successfully resolved.
        break

    if home is None:
        # Fall through to stock robosuite Panda. Its init_qpos differs
        # slightly from MountedPanda's (~30 mrad on j2/j6) but it's
        # closer to upstream than the perturbed canonical pose, so it's
        # still a useful fallback when libero is unavailable.
        try:
            from robosuite.models.robots.manipulators.panda_robot import (  # type: ignore[import-not-found]
                Panda,
            )

            home = np.asarray(Panda().init_qpos, dtype=np.float64)
            if home.shape != (n_arm,):
                logger.debug(
                    "robosuite Panda.init_qpos shape %s does not match arm DoF %d; ignoring",
                    home.shape,
                    n_arm,
                )
                home = None
        except ImportError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.debug("robosuite Panda init_qpos lookup raised %s; no fallback", e)

    _CACHED_LIBERO_HOME_QPOS = home
    _CACHED_LIBERO_HOME_QPOS_RESOLVED = True
    return home


# Module-level cache for the Panda gripper init_qpos (mirrors the arm
# home pose cache above). Used by
# ``LiberoAdapter._write_libero_arm_home_qpos`` to set finger joints to
# upstream's open canonical position.
_CACHED_PANDA_GRIPPER_INIT_QPOS: np.ndarray | None = None
_CACHED_PANDA_GRIPPER_INIT_QPOS_RESOLVED: bool = False


def _resolve_panda_gripper_init_qpos(n_finger: int) -> np.ndarray | None:
    """Return the LIBERO-canonical Panda gripper finger init qpos, or ``None``.

    Mirrors ``robosuite.models.grippers.panda_gripper.PandaGripper.init_qpos``,
    which is ``[0.020833, -0.020833]`` (≈ 2 cm fingertip separation,
    "open" position). Used by :meth:`LiberoAdapter._write_libero_arm_home_qpos`
    to initialise gripper finger qpos to upstream's canonical open
    pose, mirroring upstream ``SingleArm.reset`` which writes
    ``self.gripper.init_qpos`` to the gripper joint qpos addresses
    BEFORE constructing the OSC controller.

    Resolution order:

    1. ``robosuite.models.grippers.panda_gripper.PandaGripper().init_qpos``
       (the library function used by both stock libero and LIBERO-PRO).
    2. ``None`` — caller leaves gripper qpos at MuJoCo's default (zero,
       which is "fingers touching" — significantly different from the
       open canonical pose). The 14 mm divergence between
       ``data.qpos[gripper] = 0`` (closed) and ``data.qpos[gripper] =
       0.0208`` (open) is enough to put the policy out-of-distribution
       on ``state.gripper`` from the very first observation.

    Refs #176 sub-task 3d.
    """
    global _CACHED_PANDA_GRIPPER_INIT_QPOS, _CACHED_PANDA_GRIPPER_INIT_QPOS_RESOLVED

    if _CACHED_PANDA_GRIPPER_INIT_QPOS_RESOLVED:
        if _CACHED_PANDA_GRIPPER_INIT_QPOS is None:
            return None
        if _CACHED_PANDA_GRIPPER_INIT_QPOS.shape != (n_finger,):
            return None
        return _CACHED_PANDA_GRIPPER_INIT_QPOS

    init_qpos: np.ndarray | None = None
    try:
        from robosuite.models.grippers.panda_gripper import (  # type: ignore[import-not-found]
            PandaGripper,
        )

        init_qpos = np.asarray(PandaGripper().init_qpos, dtype=np.float64)
        if init_qpos.shape != (n_finger,):
            logger.debug(
                "PandaGripper.init_qpos shape %s does not match finger count %d; ignoring",
                init_qpos.shape,
                n_finger,
            )
            init_qpos = None
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001
        logger.debug("PandaGripper.init_qpos lookup raised %s; no fallback", e)

    _CACHED_PANDA_GRIPPER_INIT_QPOS = init_qpos
    _CACHED_PANDA_GRIPPER_INIT_QPOS_RESOLVED = True
    return init_qpos


def _to_scalar(value: Any) -> float:
    """Coerce a GR00T-LIBERO action channel to a scalar float.

    Handles GR00T's training-shape packing where each per-key
    value may be either a scalar (legacy) or a 2-element list /
    array / ndarray (current GR00T-LIBERO convention - same pattern
    PR #162 introduced for ``state.gripper``, applied to every
    action channel).

    An earlier commit in #168 fixed only ``gripper``; subsequent
    verification showed the same ``float(list)`` bug raises on
    ``x/y/z/roll/pitch/yaw`` too - GR00T sends ALL keys list-shaped.
    This helper centralises the coercion so all 7 action channels go
    through the same code path.

    * Scalar input → ``float(value)``
    * List / tuple / ndarray (non-empty) → ``float(value[0])``
    * Everything else (None, empty list, dict, etc.) → ``0.0`` after
      a single WARNING log per call (caller should filter spurious
      input shapes; here we just degrade gracefully).
    """
    try:
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
            return float(value[0])
        return float(value)
    except (TypeError, ValueError, IndexError) as e:
        logger.warning(
            "_LiberoOSCController._to_scalar: could not coerce action value %r to float (%s); "
            "treating as 0.0 for this step",
            value,
            e,
        )
        return 0.0


__all__ = [
    "BDDLParseError",
    "LiberoAdapter",
]
