"""WBCConfig - configuration for the GR00T Whole-Body-Control (SONIC) policy.

The upstream ``GearWbcController`` reference runner
(``decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc.py``) is config-driven:
a JSON/YAML file supplies the ONNX checkpoint paths, the per-joint PD gains,
the default joint angles, and the observation/action dimensions. This module
captures that contract as a frozen :class:`WBCConfig` dataclass plus a loader
that reads it from a JSON file or an in-memory dict.

Keeping the config as a typed dataclass (rather than passing a raw dict around)
means dimension/shape mistakes surface at construction with a clear message,
not as an opaque ONNX shape error mid-rollout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from strands_robots.utils import require_optional

# Upstream defaults from the GR00T-WholeBodyControl reference controller
# (decoupled_wbc/sim2mujoco: run_mujoco_gear_wbc.py + resources/robots/g1/
# g1_gear_wbc.yaml). The ``single_obs_dim`` is fixed by the controller's
# observation layout (compute_observation, single_obs_dim = 86):
#   command(7) + base_ang_vel(3) + projected_gravity(3)
#     + qj(n_obs_joints) + dqj(n_obs_joints) + action(num_actions)
# CRITICAL: qj/dqj observe ALL the robot's joints (upstream ``n_joints`` =
# nq - 7 = 29 for the G1), NOT just the 15 controlled leg+waist joints. The
# action block is the num_actions (15) leg+waist outputs. So:
#   7 + 3 + 3 + 29 + 29 + 15 = 86.  (Using 15 for qj/dqj would give 58 and put
# the data in the wrong slots - the network would see a malformed observation
# even though the total 516 width still loads.)
_DEFAULT_N_OBS_JOINTS = 29
#
# The command block (7) is NOT just zero-padded velocity - per
# compute_observation it is:
#   command[0:3] = loco_cmd[:3] * cmd_scale      (velocity, scaled)
#   command[3]   = height_cmd                    (target base height)
#   command[4:7] = rpy_cmd                       (target roll/pitch/yaw)
_DEFAULT_SINGLE_OBS_DIM = 86
_DEFAULT_NUM_ACTIONS = 15
# Upstream g1_gear_wbc.yaml: obs_history_len=6 (num_obs = 86*6 = 516).
_DEFAULT_OBS_HISTORY_LEN = 6
_DEFAULT_COMMAND_DIM = 7
# Upstream cmd_scale applied to [vx, vy, omega] and the default base-height
# command, from g1_gear_wbc.yaml.
_DEFAULT_CMD_SCALE = (2.0, 2.0, 0.5)
_DEFAULT_HEIGHT_CMD = 0.74


@dataclass(frozen=True)
class WBCConfig:
    """Typed configuration for :class:`~strands_robots.policies.wbc.policy.WBCPolicy`.

    Mirrors the fields the upstream ``GearWbcController`` config file supplies.
    All sequence fields are stored as plain ``list[float]`` (JSON-friendly);
    the policy converts them to NumPy arrays once at construction.

    Attributes:
        policy_path: Path to the main locomotion ONNX policy.
        walk_policy_path: Path to the walk ONNX policy. ``None`` when the
            checkpoint ships a single policy (``walk=False`` on the policy).
        xml_path: Optional MuJoCo XML the checkpoint was trained against.
            Informational only - the policy drives whatever robot the sim
            backend loaded; recorded here for provenance / validation.
        default_angles: Per-joint default (nominal stance) angles, length
            ``num_actions``. Subtracted from measured ``qj`` in the observation
            and added back to the ONNX target offset to form absolute targets.
        kps: Per-joint proportional gains, length ``num_actions``.
        kds: Per-joint derivative gains, length ``num_actions``.
        action_scale: Scale applied to the raw ONNX output before it becomes a
            joint-position offset (upstream ``action_scale``).
        obs_scales: Named scale factors applied to observation sub-vectors
            (``ang_vel`` / ``dof_pos`` / ``dof_vel``). Defaults match upstream
            g1_gear_wbc.yaml (ang_vel_scale=0.5, dof_pos_scale=1.0,
            dof_vel_scale=0.05).
        cmd_scale: Scale applied to the ``[vx, vy, omega]`` velocity command
            before it enters the observation's command block (upstream
            ``cmd_scale = [2.0, 2.0, 0.5]``).
        height_cmd: Default target base height written to command slot [3]
            (upstream ``height_cmd = 0.74``). Overridable per call via the
            ``height`` kwarg.
        rpy_cmd: Default target roll/pitch/yaw written to command slots [4:7]
            (upstream ``rpy_cmd = [0, 0, 0]``). Overridable per call via the
            ``target_orientation`` kwarg.
        single_obs_dim: Width of one observation frame (before history
            stacking). Default 86 (upstream GEAR-SONIC).
        obs_history_len: Number of frames stacked into the network input.
            Default 6 (upstream num_obs = 86 * 6 = 516).
        num_actions: Number of controllable joints (legs + waist). Default 15.
        command_dim: Width of the command sub-vector at the head of the
            observation. Default 7 (velocity[3] + height[1] + rpy[3]).
        n_obs_joints: Number of joints OBSERVED in the qj/dqj blocks - the
            robot's full joint count (upstream ``n_joints`` = nq - 7 = 29 for
            the G1), NOT ``num_actions``. The controller observes the whole body
            (legs + waist + arms) but only drives the first ``num_actions`` (15)
            leg+waist joints. Default 29.
    """

    policy_path: str
    walk_policy_path: str | None = None
    xml_path: str | None = None
    default_angles: list[float] = field(default_factory=list)
    kps: list[float] = field(default_factory=list)
    kds: list[float] = field(default_factory=list)
    action_scale: float = 0.25
    obs_scales: dict[str, float] = field(default_factory=lambda: {"ang_vel": 0.5, "dof_pos": 1.0, "dof_vel": 0.05})
    cmd_scale: list[float] = field(default_factory=lambda: list(_DEFAULT_CMD_SCALE))
    height_cmd: float = _DEFAULT_HEIGHT_CMD
    rpy_cmd: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    single_obs_dim: int = _DEFAULT_SINGLE_OBS_DIM
    obs_history_len: int = _DEFAULT_OBS_HISTORY_LEN
    num_actions: int = _DEFAULT_NUM_ACTIONS
    n_obs_joints: int = _DEFAULT_N_OBS_JOINTS
    command_dim: int = _DEFAULT_COMMAND_DIM

    def __post_init__(self) -> None:
        # Fail-fast on dimension mistakes (AGENTS.md #5: raise on fatal errors,
        # never warn-and-continue with a config that will misbehave later).
        if self.num_actions < 1:
            raise ValueError(f"WBCConfig.num_actions must be >= 1, got {self.num_actions}")
        if self.obs_history_len < 1:
            raise ValueError(f"WBCConfig.obs_history_len must be >= 1, got {self.obs_history_len}")
        if self.single_obs_dim < 1:
            raise ValueError(f"WBCConfig.single_obs_dim must be >= 1, got {self.single_obs_dim}")
        if self.command_dim < 3:
            # Need at least [vx, vy, omega].
            raise ValueError(f"WBCConfig.command_dim must be >= 3 (vx, vy, omega), got {self.command_dim}")
        if self.n_obs_joints < self.num_actions:
            # The controller observes all joints (legs+waist+arms) but drives the
            # first num_actions; observing fewer than it drives is impossible.
            raise ValueError(
                f"WBCConfig.n_obs_joints ({self.n_obs_joints}) must be >= num_actions ({self.num_actions}); "
                "qj/dqj observe the whole body, action drives the leg+waist subset."
            )

        # Per-joint vectors, when provided, must match num_actions. They are
        # allowed to be empty (the policy then falls back to zeros / unit gains
        # with a warning), but a *wrong* non-empty length is a hard error - it
        # almost certainly means the config was paired with the wrong checkpoint.
        for name in ("default_angles", "kps", "kds"):
            vec = getattr(self, name)
            if vec and len(vec) != self.num_actions:
                raise ValueError(
                    f"WBCConfig.{name} has length {len(vec)} but num_actions={self.num_actions}; "
                    "they must match (or leave the field empty to use defaults)."
                )

        # cmd_scale scales the [vx, vy, omega] velocity command, so it must have
        # exactly 3 entries when provided (upstream cmd_scale = [2.0, 2.0, 0.5]).
        # A wrong length is rejected rather than silently tolerated, matching the
        # per-joint vectors above.
        if self.cmd_scale and len(self.cmd_scale) != 3:
            raise ValueError(
                f"WBCConfig.cmd_scale must have exactly 3 entries [vx, vy, omega] scale, "
                f"got {len(self.cmd_scale)}: {self.cmd_scale}."
            )

    @property
    def num_obs(self) -> int:
        """Total network input width = ``single_obs_dim * obs_history_len``."""
        return self.single_obs_dim * self.obs_history_len

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WBCConfig:
        """Build a :class:`WBCConfig` from a plain dict.

        Only recognised keys are consumed; unknown keys are ignored (forward
        compatibility with the richer upstream config, which also carries
        ``simulation_dt`` / ``control_decimation`` / ``cmd_init`` / ``freq_cmd``
        / ``xml_path`` etc.). ``policy_path`` is required.

        The upstream ``g1_gear_wbc.yaml`` specifies the observation scales as
        FLAT keys (``ang_vel_scale`` / ``dof_pos_scale`` / ``dof_vel_scale``)
        rather than a nested ``obs_scales`` map. Those flat keys are normalised
        into ``obs_scales`` here so the upstream config loads unchanged. An
        explicit ``obs_scales`` map, if present, takes precedence.
        """
        if "policy_path" not in data:
            raise ValueError("WBCConfig requires a 'policy_path' entry")

        data = dict(data)  # shallow copy - don't mutate the caller's dict

        # Normalise upstream flat scale keys into the nested obs_scales map.
        _flat_scale_keys = {"ang_vel": "ang_vel_scale", "dof_pos": "dof_pos_scale", "dof_vel": "dof_vel_scale"}
        flat_scales = {
            short: float(data[flat]) for short, flat in _flat_scale_keys.items() if data.get(flat) is not None
        }
        if flat_scales:
            merged = dict(flat_scales)
            # An explicit obs_scales map wins over the flat keys it overlaps.
            if isinstance(data.get("obs_scales"), dict):
                merged.update(data["obs_scales"])
            data["obs_scales"] = merged

        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: str | Path) -> WBCConfig:
        """Load a :class:`WBCConfig` from a JSON or YAML file.

        ``.json`` parses with the stdlib; ``.yaml`` / ``.yml`` parse with
        ``pyyaml`` (optional - install ``strands-robots[wbc]`` or ``pyyaml``).
        YAML support lets the policy consume the upstream ``g1_gear_wbc.yaml``
        directly. The upstream YAML uses flat scale keys
        (``ang_vel_scale`` / ``dof_pos_scale`` / ``dof_vel_scale``) rather than a
        nested ``obs_scales`` map; :meth:`from_dict` normalises those.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the file is not valid JSON/YAML, has an unsupported
                extension, or is missing ``policy_path``.
            ImportError: If a YAML file is given but ``pyyaml`` is not installed.
        """
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"WBCConfig file not found: {p}")
        text = p.read_text()
        suffix = p.suffix.lower()
        if suffix == ".json":
            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(f"WBCConfig file {p} is not valid JSON: {e}") from e
        elif suffix in (".yaml", ".yml"):
            yaml = require_optional("yaml", pip_install="pyyaml", extra="wbc", purpose="WBCConfig YAML loading")
            data = yaml.safe_load(text)  # type: ignore[attr-defined]
        else:
            raise ValueError(f"WBCConfig file {p} has unsupported extension {suffix!r}; use .json, .yaml, or .yml.")
        if not isinstance(data, dict):
            raise ValueError(f"WBCConfig file {p} must contain a mapping, got {type(data).__name__}")
        return cls.from_dict(data)


__all__ = ["WBCConfig"]
