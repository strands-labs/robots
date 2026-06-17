"""WBCPolicy - GR00T Whole-Body-Control (SONIC) locomotion for the Unitree G1.

A clean-room :class:`Policy` provider wrapping NVIDIA's GR00T Whole-Body-Control
ONNX controllers (the SONIC / decoupled-WBC family) for deploy-grade humanoid
locomotion. Upstream reference:
https://github.com/NVlabs/GR00T-WholeBodyControl
(``decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc.py``).

This is the locomotion / lower-body member of the non-VLA policy family
(alongside cuRobo and MoveIt2): ``requires_images = False`` and the goal arrives
through the well-known ``**kwargs`` locomotion keys rather than a natural-
language instruction. The controller drives the **15 leg+waist DOFs**; the arm
joints (16..n) are held at their nominal defaults. Layering an upper-body
manipulation policy on top is the job of a future ``CompositePolicy`` (#468),
deliberately out of scope here.

Scope: this targets the **non-gait** reference (``run_mujoco_gear_wbc.py`` +
``g1_gear_wbc.yaml``): single_obs_dim 86, a 7-wide command block, two policies
(main ``policy`` + ``walk_policy``) selected by velocity. The upstream repo also
ships a *gait-clock* variant (``run_mujoco_gear_wbc_gait.py``: single_obs_dim 95,
an 8-wide command with ``freq_cmd`` + a 2-dim clock signal + torso slots, single
policy) - that layout is a separate embodiment and is **not** implemented here.
The shipped weights this matches are ``GR00T-WholeBodyControl-Balance.onnx``
(main) and ``-Walk.onnx`` (walk), whose ONNX input is ``[batch, 516]`` (86 x 6)
and output ``[batch, 15]``.

Control contract (reproduced from the reference runner):

* **Two ONNX sessions** - a main ``policy_path`` and an optional
  ``walk_policy_path`` - loaded via ``onnxruntime.InferenceSession`` in
  ``__init__`` (centralised dependency check per AGENTS.md). Missing
  ``onnxruntime`` or a missing checkpoint raises ``RuntimeError`` - there is no
  silent zero-torque fallback (explicit project rule, AGENTS.md #5/#6).
* **Observation** - an 86-dim frame (command + base ang-vel + projected gravity
  + scaled qj/dqj + previous action), stacked over ``obs_history_len`` via a
  deque. See :mod:`strands_robots.policies.wbc.observation`.
* **Action** - the network emits a 15-dim joint-position *offset*. The policy
  forms absolute joint targets ``target_q = default_angles + action_scale *
  raw`` and returns them as a one-element ``list[dict]`` keyed by the G1
  leg+waist actuator names. Callers that drive torque-actuated MuJoCo convert
  to torque via :meth:`compute_torques` (the upstream PD law).

Usage through the sim backend::

    sim.run_policy(
        robot_name="unitree_g1", policy_provider="wbc",
        policy_config={"checkpoint": ".../GEAR-SONIC", "walk": True},
        policy_kwargs={"target_velocity": [0.5, 0.0, 0.0]},
        duration=10.0, control_frequency=50.0,
    )
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from strands_robots.policies.base import Policy
from strands_robots.utils import require_optional

from .config import WBCConfig
from .control import compute_targets, pd_control, projected_gravity
from .observation import ObservationHistory, build_single_frame

logger = logging.getLogger(__name__)


# Canonical Unitree G1 leg+waist actuator order (first 15 of the 29-DOF model).
# Sourced from the MuJoCo Menagerie ``unitree_g1`` model and kept identical to
# the ``unitree_g1`` state/action ordering already used by the lerobot_local
# embodiment map, so a value the rest of the stack accepts flows end-to-end.
# This is the EXPLICIT actuator <-> 15-dim WBC mapping required by #466
# ("no positional guessing"): WBC output index i drives WBC_G1_LEG_WAIST_JOINTS[i].
WBC_G1_LEG_WAIST_JOINTS: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)

# Full Unitree G1 29-DOF joint order (legs + waist + arms), matching the model's
# qpos[7:] layout and the upstream G1_29DOF_JOINT_NAMES. The OBSERVATION reads
# qj/dqj for ALL of these (n_obs_joints); the controller only DRIVES the first
# 15 (WBC_G1_LEG_WAIST_JOINTS). The two share the leg+waist prefix exactly.
WBC_G1_ALL_JOINTS: tuple[str, ...] = (
    *WBC_G1_LEG_WAIST_JOINTS,
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# The HF repo the checkpoint is fetched from when ``checkpoint`` is a bare
# model id rather than a local path. No weights are bundled; they are fetched
# at runtime under the NVIDIA Open Model License.
_DEFAULT_HF_REPO = "nvidia/GEAR-SONIC"

# Default per-session ONNX filenames inside a checkpoint directory.
_MAIN_POLICY_FILENAME = "policy.onnx"
_WALK_POLICY_FILENAME = "walk_policy.onnx"
_CONFIG_FILENAME = "config.json"

# Raw-velocity-norm threshold for walk-vs-main policy selection, matching the
# upstream reference (run_mujoco_gear_wbc.py: norm(loco_cmd) <= 0.05 -> main
# "standing" policy; above -> walk_policy).
_WALK_VELOCITY_THRESHOLD = 0.05

# HuggingFace repo id: "<org>/<repo>", each segment letters/digits/._- only.
# Used to decide whether a non-existent checkpoint string is an HF id worth
# downloading vs. a local path that should surface a clean not-found error.
_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def _looks_like_hf_repo_id(value: str) -> bool:
    """True if ``value`` is shaped like a HuggingFace ``org/repo`` model id.

    Rejects anything path-like - absolute paths, ``./`` or ``../`` prefixes,
    backslashes, ``.onnx`` files, ``..`` traversal, or more than one ``/`` -
    so a non-existent *local* path is never mistaken for a remote repo and sent
    to the network. A plain ``org/repo`` (the only ambiguous case vs. a relative
    dir) is treated as an id; the caller logs before downloading so that choice
    is visible.
    """
    if value.endswith(".onnx") or "\\" in value or ".." in value:
        return False
    if value.startswith((".", "/", "~")):
        return False
    return bool(_HF_REPO_ID_RE.match(value))


class WBCPolicy(Policy):
    """ONNX whole-body-control locomotion policy for the Unitree G1.

    Args:
        checkpoint: Either a local directory containing the ONNX files +
            ``config.json``, a direct path to the main ``.onnx`` file, or a
            HuggingFace model id (downloaded via ``huggingface_hub`` when the
            optional dep is present). No weights are bundled.
        config: A :class:`WBCConfig`, a path to a config JSON, or a dict. When
            ``None`` the policy looks for ``config.json`` inside the checkpoint
            directory.
        walk: When ``True`` (default) load and prefer the walk policy
            (``walk_policy_path``) for forward locomotion. When ``False`` only
            the main policy is loaded/used.
        target_velocity: Optional constructor-time default locomotion command
            ``[vx, vy, omega]`` (m/s, m/s, rad/s). Used when a call supplies no
            ``target_velocity`` kwarg - this is how a *static* walk works
            through the mesh ``tell()`` / ``policy_config`` path that forwards
            only constructor kwargs. Per-call ``target_velocity`` overrides it.
        allow_missing_models: Test/CI seam. When ``True`` the ONNX sessions are
            not loaded eagerly (so unit tests can inject a stub session via
            :attr:`policy_session` / :attr:`walk_session`). Production callers
            leave this ``False`` so a missing checkpoint fails loudly at
            construction.
        **kwargs: Forward-compatibility absorber for the smart-string / registry
            resolution path. Per the #300 contract, providers MUST ignore
            unknown kwargs rather than raising.

    Raises:
        RuntimeError: If ``onnxruntime`` is missing, or a checkpoint file is
            absent, when ``allow_missing_models`` is ``False``.
        ValueError: If the resolved config dimensions are inconsistent.
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        config: str | dict[str, Any] | WBCConfig | None = None,
        walk: bool = True,
        target_velocity: list[float] | None = None,
        allow_missing_models: bool = False,
        **kwargs: Any,
    ) -> None:
        self._walk = bool(walk)
        self._robot_state_keys: list[str] = []
        self._warned_no_velocity = False
        self._default_command = self._validate_velocity(target_velocity) if target_velocity is not None else None

        # Default to the canonical SONIC checkpoint when none is given, so
        # ``create_policy("wbc")`` resolves to a downloadable repo rather than
        # failing with a bare "checkpoint not found". No weights are bundled;
        # the HF download happens lazily in _load_sessions under the model
        # license. An explicit local path / id always wins.
        if checkpoint is None:
            checkpoint = _DEFAULT_HF_REPO

        # Resolve the config first - it tells us dims + default file layout.
        self._config = self._resolve_config(config, checkpoint)
        n = self._config.num_actions

        # The leg+waist joint names WBC reads/writes, in WBC output order.
        # set_robot_state_keys resolves these by name within the robot's joint
        # list; until then, default to the canonical mapping table so direct
        # get_actions calls (without set_robot_state_keys) still emit the right
        # keys. WBC_G1_LEG_WAIST_JOINTS has exactly 15 names, so n must not
        # exceed that (validated below).
        self._wbc_joint_names: list[str] = list(WBC_G1_LEG_WAIST_JOINTS[:n])

        # The G1 actuator-name mapping has exactly len(WBC_G1_LEG_WAIST_JOINTS)
        # entries. A config with more actions than the table can name would
        # silently truncate everywhere (state read, action keys, validation) and
        # fail late with a confusing zip/length error. Reject it at construction
        # (AGENTS.md #5: fail fast on a fatal config) - a different humanoid with
        # a different DOF count needs its own mapping table, not a silent slice.
        if n > len(WBC_G1_LEG_WAIST_JOINTS):
            raise ValueError(
                f"WBCConfig.num_actions={n} exceeds the {len(WBC_G1_LEG_WAIST_JOINTS)}-entry "
                "G1 leg+waist actuator mapping (WBC_G1_LEG_WAIST_JOINTS). WBCPolicy targets the "
                "G1 lower body; a different embodiment needs its own joint mapping table."
            )

        # The whole-body joints the qj/dqj observation blocks read (legs+waist+
        # arms), in WBC_G1_ALL_JOINTS order. set_robot_state_keys resolves these
        # by name; default to the canonical table until then.
        no = self._config.n_obs_joints
        if no > len(WBC_G1_ALL_JOINTS):
            raise ValueError(
                f"WBCConfig.n_obs_joints={no} exceeds the {len(WBC_G1_ALL_JOINTS)}-entry "
                "G1 whole-body joint mapping (WBC_G1_ALL_JOINTS)."
            )
        self._obs_joint_names: list[str] = list(WBC_G1_ALL_JOINTS[:no])

        # Pre-compute NumPy views of the per-joint vectors (once, not per tick).
        self._default_angles = (
            np.asarray(self._config.default_angles, dtype=np.float64)
            if self._config.default_angles
            else np.zeros(n, dtype=np.float64)
        )
        self._kps = np.asarray(self._config.kps, dtype=np.float64) if self._config.kps else np.ones(n, dtype=np.float64)
        self._kds = (
            np.asarray(self._config.kds, dtype=np.float64) if self._config.kds else np.zeros(n, dtype=np.float64)
        )

        # Previous action (15-dim) fed back into the observation; zeroed at reset.
        self._prev_action = np.zeros(n, dtype=np.float64)
        self._history = ObservationHistory(self._config)

        # ONNX sessions. Stub seam: when allow_missing_models is True we skip
        # the eager load so tests can assign fakes to policy_session/walk_session.
        self.policy_session: Any = None
        self.walk_session: Any = None
        if not allow_missing_models:
            self._load_sessions(checkpoint)

        if kwargs:
            logger.debug("WBCPolicy ignoring unknown constructor kwargs: %s", sorted(kwargs.keys()))

        logger.info(
            "WBCPolicy ready [num_actions=%d obs_history_len=%d walk=%s default_cmd=%s]",
            n,
            self._config.obs_history_len,
            self._walk,
            self._default_command.tolist() if self._default_command is not None else None,
        )

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "wbc"

    @property
    def requires_images(self) -> bool:
        """WBC controls from joint state + base IMU only, never images."""
        return False

    @property
    def config(self) -> WBCConfig:
        return self._config

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Resolve the G1 leg+waist joints BY NAME within the robot's key list.

        WBC output index ``i`` drives :data:`WBC_G1_LEG_WAIST_JOINTS`\\ ``[i]``.
        We locate each of those names inside ``robot_state_keys`` rather than
        assuming a fixed position, because the sim's joint list is not just the
        15 controllable joints in WBC order:

        * MuJoCo prepends the robot's free/floating-base joint (the G1 model
          names it ``floating_base_joint``), so the leg+waist joints are NOT
          at indices ``[0:15]``.
        * The full 29-DOF model also carries 14 arm joints interleaved after
          the waist.

        Resolving by name (#466: explicit mapping, no positional guessing)
        means the policy reads and writes exactly the right joints regardless
        of where the sim places them or what the free joint is called.

        Raises:
            ValueError: If any expected leg+waist joint name is absent from
                ``robot_state_keys`` - a mismatch that would otherwise actuate
                the wrong joints. The error lists the missing names.
        """
        keys = list(robot_state_keys)
        key_set = set(keys)

        # Controlled joints: the num_actions leg+waist joints WBC drives.
        expected = WBC_G1_LEG_WAIST_JOINTS[: self._config.num_actions]
        missing = [name for name in expected if name not in key_set]
        if missing:
            raise ValueError(
                "WBCPolicy: the robot's joint list is missing expected G1 "
                f"leg+waist joints: {missing}.\n"
                f"  expected (WBC order): {list(expected)}\n"
                f"  robot provided:       {keys}\n"
                "WBC drives these named joints; load the full unitree_g1 model "
                "(its leg+waist joints carry these exact names)."
            )

        # Observed joints: the n_obs_joints whole-body joints the qj/dqj blocks
        # read (legs+waist+arms). The controller observes the whole body even
        # though it only drives the leg+waist subset. Resolve these by name too
        # (in WBC_G1_ALL_JOINTS order), so the observation matches the model's
        # qpos[7:] layout regardless of how the sim orders / namespaces joints.
        obs_expected = WBC_G1_ALL_JOINTS[: self._config.n_obs_joints]
        obs_missing = [name for name in obs_expected if name not in key_set]
        if obs_missing:
            raise ValueError(
                "WBCPolicy: the robot's joint list is missing observed G1 joints "
                f"(qj/dqj read the whole body, n_obs_joints={self._config.n_obs_joints}): {obs_missing}.\n"
                f"  expected (observe order): {list(obs_expected)}\n"
                f"  robot provided:           {keys}\n"
                "Load the full unitree_g1 (29-DOF) model."
            )

        self._robot_state_keys = list(keys)
        self._wbc_joint_names = list(expected)  # the num_actions controlled joints
        self._obs_joint_names = list(obs_expected)  # the n_obs_joints observed joints

    def reset(self, seed: int | None = None) -> None:
        """Clear the observation history and previous-action feedback.

        The ``seed`` argument is accepted for API parity with the rest of the
        policy family; the ONNX controller is deterministic, so it is unused.
        """
        self._history.reset()
        self._prev_action = np.zeros(self._config.num_actions, dtype=np.float64)
        logger.debug("WBCPolicy.reset: cleared observation history + prev_action (seed=%r)", seed)

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Run one WBC inference step and return the 15-dim target joint positions.

        Reads the locomotion command from the well-known kwargs
        (``target_velocity = [vx, vy, omega]``, optional ``target_orientation``
        for roll/pitch/yaw, optional ``height``); ``instruction`` is ignored.
        Builds the stacked observation, runs the ONNX policy, converts the raw
        offset to absolute joint targets, and returns a single per-step action
        dict keyed by actuator name.

        Returns a one-element list (WBC is a closed-loop per-tick controller,
        not a chunked planner): the runner re-queries every control step.
        """
        command, raw_velocity = self._resolve_command(kwargs)

        qj, dqj, base_ang_vel, quat = self._extract_state(observation_dict)
        proj_grav = projected_gravity(quat)

        frame = build_single_frame(
            self._config,
            command=command,
            base_ang_vel=base_ang_vel,
            proj_gravity=proj_grav,
            qj=qj,
            dqj=dqj,
            prev_action=self._prev_action,
        )
        obs = self._history.push(frame)

        # Walk-selection uses the RAW (unscaled) velocity norm, matching the
        # upstream reference (run_mujoco_gear_wbc.py: norm(loco_cmd) <= 0.05).
        raw_action = self._run_session(obs, raw_velocity)
        self._prev_action = raw_action

        target_q = compute_targets(self._default_angles, raw_action, self._config.action_scale)
        keys = self._resolve_action_keys()
        return [{k: float(v) for k, v in zip(keys, target_q, strict=True)}]

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def compute_torques(
        self,
        target_pos: np.ndarray,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
    ) -> np.ndarray:
        """Expose the upstream PD law for callers driving torque actuators.

        ``tau = (target_pos - current_pos) * kp + (0 - current_vel) * kd``
        using the per-joint gains from the config. The desired velocity is
        zero (position hold between targets), matching the reference loop.
        """
        target_pos = np.asarray(target_pos, dtype=np.float64)
        current_pos = np.asarray(current_pos, dtype=np.float64)
        current_vel = np.asarray(current_vel, dtype=np.float64)
        zeros = np.zeros_like(target_pos)
        return pd_control(target_pos, current_pos, self._kps, zeros, current_vel, self._kds)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_command(self, kwargs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Build the 7-dim command block for this tick.

        Faithful to the upstream ``compute_observation`` layout
        (run_mujoco_gear_wbc.py)::

            command[0:3] = loco_cmd[:3] * cmd_scale   # velocity, scaled
            command[3]   = height_cmd                  # target base height
            command[4:7] = rpy_cmd                     # target roll/pitch/yaw

        Goal sources (per-call kwarg > constructor default > config default):
        - ``target_velocity = [vx, vy, omega]`` -> the raw velocity triple.
        - ``target_orientation = [roll, pitch, yaw]`` -> rpy command slots.
        - ``height`` -> the height command slot.

        Returns:
            ``(command, raw_velocity)`` where ``command`` is the ``command_dim``-
            wide (default 7) observation block with ``cmd_scale`` already applied
            to the velocity, and ``raw_velocity`` is the UNSCALED ``[vx, vy,
            omega]`` triple used for walk-vs-main policy selection (matching the
            upstream ``norm(loco_cmd) <= 0.05`` test on the raw command).
        """
        tv = kwargs.get("target_velocity")
        if tv is not None:
            vel_full = self._validate_velocity(tv)
        elif self._default_command is not None:
            vel_full = self._default_command.copy()
        else:
            vel_full = np.zeros(3, dtype=np.float64)
        # The raw velocity is the first three entries (vx, vy, omega); any extra
        # entries a caller packed in are ignored for the scaled command block.
        raw_velocity = vel_full[:3].copy()

        c = self._config.command_dim
        command = np.zeros(c, dtype=np.float64)

        # Slots [0:3]: velocity * cmd_scale (clamp the slice to whatever fits).
        cmd_scale = np.asarray(self._config.cmd_scale, dtype=np.float64).ravel()
        n_vel = min(3, c)
        scale = cmd_scale[:n_vel] if cmd_scale.shape[0] >= n_vel else np.ones(n_vel)
        command[:n_vel] = raw_velocity[:n_vel] * scale

        # Slot [3]: target base height (per-call ``height`` overrides the config).
        if c > 3:
            height = kwargs.get("height")
            command[3] = float(height) if height is not None else float(self._config.height_cmd)

        # Slots [4:7]: target roll/pitch/yaw (per-call ``target_orientation``
        # overrides the config ``rpy_cmd``).
        if c > 4:
            rpy_src = kwargs.get("target_orientation")
            rpy = np.asarray(rpy_src if rpy_src is not None else self._config.rpy_cmd, dtype=np.float64).ravel()
            n_rpy = min(c - 4, rpy.shape[0])
            command[4 : 4 + n_rpy] = rpy[:n_rpy]

        return command, raw_velocity

    def _extract_state(self, observation_dict: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pull (qj, dqj, base_ang_vel, base_quat) out of the observation dict.

        ``qj`` / ``dqj`` are the WHOLE-BODY joint positions / velocities
        (``n_obs_joints`` = 29: legs+waist+arms), read by name via
        :meth:`_read_joint_vector` in :data:`WBC_G1_ALL_JOINTS` order - matching
        the upstream ``qpos[7:7+n_joints]`` observation, which spans all joints,
        not just the 15 controlled ones. Base angular velocity and orientation
        come from well-known keys, defaulting to a level, still base when absent
        (an upright stance cue) rather than fabricating motion.

        Velocity availability: WBC is a velocity-feedback balance controller, so
        ``dqj`` and ``base_ang_vel`` are genuine inputs - not optional. The
        current MuJoCo backend's unified observation exposes joint *positions*
        only (no ``<name>.vel`` keys, no ``observation.velocity``), so a plain
        ``sim.run_policy`` rollout feeds WBC zero joint velocities. We emit a
        one-time warning when that happens (a dead velocity channel can
        destabilise the gait) rather than silently pretending the controller is
        fully observed. To supply real velocities, drive the policy from an
        observation that includes ``<name>.vel`` per-joint keys (or
        ``observation.velocity`` + ``base_ang_vel``), e.g. a teleop/IMU bridge
        or a future backend velocity field.
        """
        # qj/dqj observe the whole body (n_obs_joints), in WBC_G1_ALL_JOINTS order.
        obs_names = self._obs_joint_names
        qj = self._read_joint_vector(observation_dict, "position", obs_names)
        dqj = self._read_joint_vector(observation_dict, "velocity", obs_names)

        base_ang_vel = self._read_vec(observation_dict, ("base_ang_vel", "observation.base_ang_vel"), 3)
        if base_ang_vel is None:
            base_ang_vel = np.zeros(3, dtype=np.float64)

        # Warn once if BOTH velocity channels are absent - WBC is then running
        # open-loop on velocity, which the operator should know about.
        if not self._warned_no_velocity and not self._observation_has_velocity(observation_dict, base_ang_vel):
            self._warned_no_velocity = True
            logger.warning(
                "WBCPolicy: observation exposes no joint velocities or base angular "
                "velocity; feeding the balance controller zeros for dqj/base_ang_vel. "
                "Gait stability may degrade. Supply per-joint '<name>.vel' keys, "
                "'observation.velocity', and/or 'base_ang_vel' to close the loop."
            )

        quat = self._read_vec(observation_dict, ("base_quat", "observation.base_quat"), 4)
        if quat is None:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # upright (w=1)
        return qj, dqj, base_ang_vel, quat

    def _observation_has_velocity(self, obs: dict[str, Any], base_ang_vel: np.ndarray) -> bool:
        """True if the observation carries any joint or base velocity signal."""
        if obs.get("observation.velocity") is not None:
            return True
        if float(np.linalg.norm(base_ang_vel)) > 0.0:
            return True
        return any(f"{name}.vel" in obs for name in self._obs_joint_names)

    def _read_joint_vector(self, obs: dict[str, Any], kind: str, names: list[str]) -> np.ndarray:
        """Read positions/velocities for ``names`` (in order) from the observation.

        Reads BY NAME so each output index corresponds to ``names[i]`` -
        regardless of where the sim places the joint or how many other
        (free-base, other) joints surround it. ``names`` is the whole-body
        observed-joint list (:attr:`_obs_joint_names`) for the qj/dqj blocks.

        Two observation shapes are supported:

        * Per-joint scalars keyed by joint name (the unified sim observation):
          ``position`` reads ``obs[name]``; ``velocity`` reads ``obs[name + ".vel"]``.
        * A flat ``observation.state`` / ``observation.velocity`` vector paired
          with ``self._robot_state_keys`` for the index lookup (each name's slot
          in the robot's key list, NOT a positional slice). Without keys, the
          flat vector is consumed positionally (the cuRobo / MoveIt2 contract).

        Missing values default to zero (a still, nominal stance) - a
        *measured-state* default, distinct from the forbidden zero-*torque*
        fallback. NOTE: if the sim exposes no joint velocities (the current
        MuJoCo backend's unified observation is positions only, with no
        ``<name>.vel`` keys), ``dqj`` reads as zeros - see :meth:`_extract_state`
        for the consequence and the recommended teleop/IMU velocity source.
        """
        m = len(names)

        # Flat-vector form: index into the flat array by each joint's position.
        flat_key = "observation.state" if kind == "position" else "observation.velocity"
        flat = obs.get(flat_key)
        if flat is not None and hasattr(flat, "__len__"):
            arr = np.asarray(flat if not hasattr(flat, "tolist") else flat.tolist(), dtype=np.float64).ravel()
            if self._robot_state_keys:
                # Name-resolved indexing: map each observed joint to its slot in
                # the robot's key list (handles a free-base/interleaved layout).
                # First occurrence wins so a duplicated name can't shift the slot.
                index_of: dict[str, int] = {}
                for i, name in enumerate(self._robot_state_keys):
                    index_of.setdefault(name, i)
                out = np.zeros(m, dtype=np.float64)
                for i, name in enumerate(names):
                    j = index_of.get(name)
                    if j is not None and j < arr.shape[0]:
                        out[i] = arr[j]
                return out
            # No key mapping (direct-API / replay caller). Treat the flat vector
            # as already in `names` order - the same positional contract cuRobo /
            # MoveIt2 use for observation.state - so a provided state is USED.
            if arr.shape[0] >= m:
                return arr[:m].copy()
            out = np.zeros(m, dtype=np.float64)
            out[: arr.shape[0]] = arr
            return out

        # Per-joint scalar form (the unified sim observation).
        out = np.zeros(m, dtype=np.float64)
        for i, k in enumerate(names):
            v = obs.get(k) if kind == "position" else obs.get(f"{k}.vel")
            if v is not None:
                try:
                    out[i] = float(v)
                except (TypeError, ValueError):
                    # Non-numeric observation entry for this joint: leave the
                    # pre-zeroed default in place. The full per-joint qj/dqj
                    # block is validated downstream, so a single unparseable
                    # key degrades to its neutral value rather than aborting
                    # the whole observation build.
                    pass
        return out

    @staticmethod
    def _read_vec(obs: dict[str, Any], keys: tuple[str, ...], n: int) -> np.ndarray | None:
        for k in keys:
            v = obs.get(k)
            if v is not None and hasattr(v, "__len__") and len(v) == n:
                return np.asarray(v if not hasattr(v, "tolist") else v.tolist(), dtype=np.float64).ravel()
        return None

    def _run_session(self, obs: np.ndarray, raw_velocity: np.ndarray) -> np.ndarray:
        """Run the (walk or main) ONNX session and return the 15-dim raw action.

        Session selection matches the upstream reference
        (run_mujoco_gear_wbc.py): when the RAW (unscaled) velocity command norm
        is ``<= WALK_VELOCITY_THRESHOLD`` (0.05) the robot is "standing" and the
        main ``policy`` runs; above that the ``walk_policy`` runs. When
        ``walk=False`` or no walk session is loaded, the main policy always runs.

        The ONNX input is the stacked observation as a 1xN float32 batch; the
        output is squeezed to a 1-D ``num_actions`` vector.
        """
        session = self.policy_session
        moving = float(np.linalg.norm(raw_velocity[:3])) > _WALK_VELOCITY_THRESHOLD
        if self._walk and self.walk_session is not None and moving:
            session = self.walk_session
        if session is None:
            # Defensive: reachable only via the allow_missing_models test seam
            # if a caller forgot to inject a session. Never silently emit zeros.
            raise RuntimeError(
                "WBCPolicy has no ONNX session loaded. Construct with a valid "
                "checkpoint (allow_missing_models=False), or inject a stub via "
                "policy_session=/walk_session= in tests."
            )

        net_in = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        input_name = self._session_input_name(session)
        outputs = session.run(None, {input_name: net_in})
        raw = np.asarray(outputs[0], dtype=np.float64).ravel()
        n = self._config.num_actions
        if raw.shape[0] != n:
            raise RuntimeError(
                f"WBCPolicy ONNX output width {raw.shape[0]} != num_actions {n}; "
                "the checkpoint and config disagree on the action dimension."
            )
        return raw

    @staticmethod
    def _session_input_name(session: Any) -> str:
        """Resolve the ONNX session's first input name (duck-typed for stubs)."""
        get_inputs = getattr(session, "get_inputs", None)
        if get_inputs is not None:
            inputs = get_inputs()
            if inputs:
                return str(inputs[0].name)
        # Stub sessions may expose a plain ``input_name`` attribute.
        return str(getattr(session, "input_name", "obs"))

    def _resolve_action_keys(self) -> list[str]:
        """The leg+waist actuator names this policy emits, in WBC output order.

        Always the name-resolved WBC joint set (:attr:`_wbc_joint_names`), never
        a positional slice of the robot's full joint list - so the emitted
        targets are keyed by the actual leg+waist actuators even when the sim's
        joint list leads with a free-base joint or interleaves arm joints.
        """
        return list(self._wbc_joint_names[: self._config.num_actions])

    def _resolve_config(self, config: str | dict[str, Any] | WBCConfig | None, checkpoint: str | None) -> WBCConfig:
        if isinstance(config, WBCConfig):
            return config
        if isinstance(config, dict):
            return WBCConfig.from_dict(config)
        if isinstance(config, str):
            return WBCConfig.from_file(config)
        # config is None: look for config.json inside a checkpoint directory.
        if checkpoint:
            ckpt_path = Path(checkpoint).expanduser()
            cfg_path = ckpt_path / _CONFIG_FILENAME if ckpt_path.is_dir() else ckpt_path.parent / _CONFIG_FILENAME
            if cfg_path.is_file():
                return WBCConfig.from_file(cfg_path)
        # Fall back to a default config pointing at the conventional filenames.
        # The session loader will raise if the files are actually absent.
        main_path, walk_path = self._default_onnx_paths(checkpoint)
        return WBCConfig(policy_path=main_path, walk_policy_path=walk_path)

    @staticmethod
    def _default_onnx_paths(checkpoint: str | None) -> tuple[str, str | None]:
        if not checkpoint:
            return _MAIN_POLICY_FILENAME, _WALK_POLICY_FILENAME
        p = Path(checkpoint).expanduser()
        if p.is_dir():
            return str(p / _MAIN_POLICY_FILENAME), str(p / _WALK_POLICY_FILENAME)
        # checkpoint points directly at the main .onnx file.
        return str(p), str(p.parent / _WALK_POLICY_FILENAME)

    def _load_sessions(self, checkpoint: str | None) -> None:
        """Load the ONNX sessions, raising loudly on any missing dependency/file.

        Centralised dependency check per AGENTS.md: ``onnxruntime`` is imported
        once here (not scattered ``_ensure`` guards), so a consumer learns at
        construction time whether the policy is usable.

        Per the #466 acceptance criterion, a missing dependency or checkpoint
        raises ``RuntimeError`` (never a silent zero-torque fallback).
        ``require_optional`` raises ``ImportError`` with an actionable install
        hint; we re-raise it as ``RuntimeError`` (preserving the hint via the
        cause chain) so the failure type matches the rest of the WBC contract.
        """
        try:
            ort = require_optional(
                "onnxruntime",
                pip_install="onnxruntime",
                extra="wbc",
                purpose="WBCPolicy ONNX inference",
            )
        except ImportError as e:
            raise RuntimeError(f"WBCPolicy requires onnxruntime (the [wbc] extra) but it is not installed.\n{e}") from e

        # Resolve a HuggingFace model id (``org/repo``) to a local snapshot dir
        # so the path logic below sees ordinary files. A local path / dir is
        # left untouched. No weights are bundled - they download on first use
        # under the model's license and cache under the HF cache.
        checkpoint = self._maybe_download_checkpoint(checkpoint)

        main_path = self._resolve_onnx_path(self._config.policy_path, checkpoint, _MAIN_POLICY_FILENAME)
        if main_path is None or not Path(main_path).is_file():
            raise RuntimeError(
                f"WBCPolicy main ONNX checkpoint not found (resolved: {main_path!r}). "
                "Pass checkpoint='/path/to/GEAR-SONIC' (a dir with policy.onnx) or a "
                "direct .onnx path. No weights are bundled - download them under the "
                "NVIDIA Open Model License (e.g. nvidia/GEAR-SONIC)."
            )
        self.policy_session = ort.InferenceSession(main_path)  # type: ignore[attr-defined]

        if self._walk:
            walk_path = self._resolve_onnx_path(self._config.walk_policy_path, checkpoint, _WALK_POLICY_FILENAME)
            if walk_path is not None and Path(walk_path).is_file():
                self.walk_session = ort.InferenceSession(walk_path)  # type: ignore[attr-defined]
            else:
                logger.info(
                    "WBCPolicy walk=True but no walk policy found (resolved: %r); "
                    "using the main policy for locomotion too.",
                    walk_path,
                )

    @staticmethod
    def _maybe_download_checkpoint(checkpoint: str | None) -> str | None:
        """Resolve a HuggingFace model id to a local snapshot directory.

        Checkpoint resolution order (issue #466): local path | HF download | cache.
        A value that already exists on disk (file or dir) is returned unchanged.
        A bare ``org/repo`` id (e.g. the default ``nvidia/GEAR-SONIC``) is
        fetched via ``huggingface_hub.snapshot_download`` - which is itself a
        cache (repeat calls are offline-fast) - and the local dir is returned.

        Raises:
            RuntimeError: If the id looks like an HF repo but ``huggingface_hub``
                is not installed, or the download fails. Never silently proceeds
                with an unresolved checkpoint (the session load would then raise
                a less actionable error).
        """
        if not checkpoint:
            return checkpoint
        p = Path(checkpoint).expanduser()
        if p.exists():
            return checkpoint  # local path/dir - use as-is
        if not _looks_like_hf_repo_id(checkpoint):
            # Not HF-id-shaped (absolute path, ./ or ../ prefix, backslash,
            # extra slashes, an .onnx file, or non-HF charset). Return as-is so
            # the path resolver surfaces a clear "checkpoint not found" error
            # rather than attempting a confusing network download.
            return checkpoint
        try:
            hub = require_optional(
                "huggingface_hub",
                pip_install="huggingface_hub",
                extra="wbc",
                purpose="WBCPolicy checkpoint download",
            )
        except ImportError as e:
            raise RuntimeError(
                f"WBCPolicy checkpoint {checkpoint!r} looks like a HuggingFace model id, "
                f"but huggingface_hub is not installed to download it. If you meant a local "
                f"path, pass an existing directory or .onnx file.\n{e}"
            ) from e
        # Log BEFORE the network call so an unexpected download (e.g. a bare
        # create_policy("wbc") defaulting to nvidia/GEAR-SONIC, or a mistyped
        # local path that happens to be org/repo-shaped) is visible, not silent.
        logger.info(
            "WBCPolicy resolving checkpoint %r as a HuggingFace model id; downloading (cached after first use)...",
            checkpoint,
        )
        try:
            local_dir = hub.snapshot_download(  # type: ignore[attr-defined]
                repo_id=checkpoint, allow_patterns=["*.onnx", "*.json"]
            )
        except Exception as e:  # noqa: BLE001 - surface any hub failure as actionable RuntimeError
            raise RuntimeError(
                f"WBCPolicy failed to download checkpoint {checkpoint!r} from HuggingFace: {e}. "
                "If you meant a local checkpoint, pass an existing directory or .onnx path; "
                "otherwise check network / model-license access."
            ) from e
        logger.info("WBCPolicy downloaded checkpoint %r -> %s", checkpoint, local_dir)
        return str(local_dir)

    @staticmethod
    def _resolve_onnx_path(configured: str | None, checkpoint: str | None, filename: str) -> str | None:
        """Resolve a (possibly relative) ONNX path against the checkpoint dir."""
        if configured:
            p = Path(configured).expanduser()
            if p.is_absolute() or p.is_file():
                return str(p)
            # Relative path: resolve against the checkpoint directory.
            if checkpoint:
                base = Path(checkpoint).expanduser()
                base = base if base.is_dir() else base.parent
                return str(base / configured)
            return str(p)
        if checkpoint:
            base = Path(checkpoint).expanduser()
            base = base if base.is_dir() else base.parent
            return str(base / filename)
        return None

    @staticmethod
    def _validate_velocity(tv: Any) -> np.ndarray:
        """Validate a ``[vx, vy, omega]`` locomotion command (finite, len>=3)."""
        try:
            arr = np.asarray(tv, dtype=np.float64).ravel()
        except (TypeError, ValueError) as e:
            raise ValueError(f"target_velocity must be a numeric sequence, got {tv!r}") from e
        if arr.shape[0] < 3:
            raise ValueError(f"target_velocity must have at least 3 elements [vx, vy, omega], got {arr.shape[0]}")
        for i, v in enumerate(arr):
            if math.isnan(v) or math.isinf(v):
                raise ValueError(f"target_velocity[{i}]={v!r} must be finite")
        return arr


__all__ = ["WBCPolicy", "WBC_G1_LEG_WAIST_JOINTS", "WBC_G1_ALL_JOINTS"]
