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
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from strands_robots.utils import (
    finite_number_error,
    positive_finite_number_error,
    require_optional,
    sequence_length,
)


def _non_negative_number_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a usable finite number ``>= 0``.

    The domain of a PD feedback gain. :func:`~strands_robots.utils.finite_number_error`
    decides numeric-ness and finiteness - the whole of the rule bar the floor -
    so only the floor is decided here, and a value the shared guard rejects is
    reported in its words. Zero is first-class: ``kp = 0`` with ``kd > 0`` is a
    pure-damping joint, which is a controller a caller may legitimately ask for.
    A NEGATIVE gain is not: ``tau = (target_q - q) * kp`` with ``kp < 0`` drives
    the joint AWAY from its target, so the feedback that exists to hold the
    stance accelerates the humanoid out of it.

    :func:`~strands_robots.utils.positive_finite_number_error` cannot express
    this domain (it refuses the first-class zero), and the library's only
    non-negative continuous guard sits under
    :mod:`strands_robots.simulation`, which :mod:`strands_robots.policies` must
    not depend on - hence a local binding over the shared numeric rule rather
    than a second copy of it.

    Args:
        value: The caller-supplied value.
        param: The parameter it came from, used in the message.
        context: Message prefix identifying the surface that received it.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    if error := finite_number_error(value, param, context):
        return error
    return None if float(value) >= 0.0 else f"{context}: {param} must be >= 0, got {value!r}."


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
# The observation scales from upstream g1_gear_wbc.yaml, and the single owner of
# them. Every consumer resolves an omitted key from THIS table, so a config that
# states some scales and leaves the rest to their documented default cannot end
# up scaled differently from one that states none: ``dof_vel`` is 0.05 either
# way. A second fallback number (a bare ``1.0``) would silently multiply the 29
# joint-velocity entries of the frame by 20 for exactly the configs that name a
# sibling key, which the network reads as a malformed observation.
_DEFAULT_OBS_SCALES: dict[str, float] = {"ang_vel": 0.5, "dof_pos": 1.0, "dof_vel": 0.05}
# Upstream cmd_scale applied to [vx, vy, omega] and the default base-height
# command, from g1_gear_wbc.yaml.
_DEFAULT_CMD_SCALE = (2.0, 2.0, 0.5)
_DEFAULT_HEIGHT_CMD = 0.74
# Upstream gait-variant step-frequency command (freq_cmd) from g1_gear_wbc.yaml.
# Only consumed by the gait-clock variant (WBCGaitPolicy), which carries an
# 8-wide command block with freq_cmd at slot [4]; the non-gait policy ignores it.
_DEFAULT_FREQ_CMD = 0.75


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
            joint-position offset (upstream ``action_scale``). Must be a finite
            number > 0: it is the only thing that carries the network's decision
            into the joint targets, so a zero (or a ``False``) discards the
            policy and holds the nominal stance, a negative value inverts every
            offset, and a ``nan``/``inf`` reaches ``data.ctrl`` as a poisoned
            torque on all ``num_actions`` joints.
        obs_scales: Named scale factors applied to observation sub-vectors
            (``ang_vel`` / ``dof_pos`` / ``dof_vel``). Defaults match upstream
            g1_gear_wbc.yaml (ang_vel_scale=0.5, dof_pos_scale=1.0,
            dof_vel_scale=0.05). A map that states only some of them is
            completed with those defaults at construction, so this attribute is
            always the full map the observation is built with - naming one scale
            never changes what an unnamed sibling is scaled by.
        cmd_scale: Scale applied to the ``[vx, vy, omega]`` velocity command
            before it enters the observation's command block (upstream
            ``cmd_scale = [2.0, 2.0, 0.5]``).
        height_cmd: Default target base height written to command slot [3]
            (upstream ``height_cmd = 0.74``). Overridable per call via the
            ``height`` kwarg.
        freq_cmd: Default step-frequency command for the gait-clock variant
            (upstream ``freq_cmd = 0.75``, written to the 8-wide command slot
            [4]). Only the gait variant
            (:class:`~strands_robots.policies.wbc.gait.WBCGaitPolicy`) reads it;
            the non-gait policy's 7-wide command has no frequency slot.
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
    obs_scales: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_OBS_SCALES))
    cmd_scale: list[float] = field(default_factory=lambda: list(_DEFAULT_CMD_SCALE))
    height_cmd: float = _DEFAULT_HEIGHT_CMD
    freq_cmd: float = _DEFAULT_FREQ_CMD
    rpy_cmd: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    single_obs_dim: int = _DEFAULT_SINGLE_OBS_DIM
    obs_history_len: int = _DEFAULT_OBS_HISTORY_LEN
    num_actions: int = _DEFAULT_NUM_ACTIONS
    n_obs_joints: int = _DEFAULT_N_OBS_JOINTS
    command_dim: int = _DEFAULT_COMMAND_DIM

    def __post_init__(self) -> None:
        # Fail-fast on dimension mistakes (AGENTS.md #5: raise on fatal errors,
        # never warn-and-continue with a config that will misbehave later).
        # Dimensions first, then values: a wrong length is the likelier root
        # cause (a config paired with the wrong checkpoint) and naming it first
        # keeps its message the one a mismatched pair reports.
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
            length = sequence_length(vec)
            if length is None:
                # A value carrying no readable component count cannot be a
                # per-joint vector. Asking first keeps a scalar or a 0-d array
                # from reaching ``len()`` as a bare TypeError, and a NumPy
                # vector from reaching ``if vec`` as the ambiguous-truth
                # ValueError - neither of which names the field.
                raise ValueError(
                    f"WBCConfig.{name} must be a sequence of {self.num_actions} numbers "
                    f"(or empty to use defaults), got {type(vec).__name__}."
                )
            if length and length != self.num_actions:
                raise ValueError(
                    f"WBCConfig.{name} has length {length} but num_actions={self.num_actions}; "
                    "they must match (or leave the field empty to use defaults)."
                )

        # cmd_scale scales the [vx, vy, omega] velocity command, so it must have
        # exactly 3 entries when provided (upstream cmd_scale = [2.0, 2.0, 0.5]).
        # A wrong length is rejected rather than silently tolerated, matching the
        # per-joint vectors above.
        cmd_scale_length = sequence_length(self.cmd_scale)
        if cmd_scale_length is None:
            raise ValueError(
                "WBCConfig.cmd_scale must be a sequence of 3 numbers [vx, vy, omega] scale, "
                f"got {type(self.cmd_scale).__name__}."
            )
        if cmd_scale_length and cmd_scale_length != 3:
            raise ValueError(
                f"WBCConfig.cmd_scale must have exactly 3 entries [vx, vy, omega] scale, "
                f"got {cmd_scale_length}: {self.cmd_scale}."
            )

        # Now the VALUES. The dimension rules above catch a config paired with
        # the wrong checkpoint; these catch one whose numbers cannot be honored
        # at all. Every field below is read verbatim into either the PD law that
        # writes ``data.ctrl`` or the observation the network sees, and none of
        # them is checked anywhere downstream: ``compute_targets`` is called
        # per-tick from ``get_actions``, so a non-real ``action_scale`` surfaces
        # as a bare ``TypeError`` from its ``float()`` after the ONNX sessions
        # have loaded and the rollout has started - the mid-rollout failure this
        # module exists to convert into a construction-time message - while a
        # ``nan`` surfaces as nothing at all and silently poisons every torque.
        if error := positive_finite_number_error(self.action_scale, "action_scale", "WBCConfig"):
            raise ValueError(error)
        for scalar_name in ("height_cmd", "freq_cmd"):
            if error := finite_number_error(getattr(self, scalar_name), scalar_name, "WBCConfig"):
                raise ValueError(error)
        for vector_name in ("default_angles", "cmd_scale", "rpy_cmd"):
            for index, component in enumerate(getattr(self, vector_name)):
                if error := finite_number_error(component, f"{vector_name}[{index}]", "WBCConfig"):
                    raise ValueError(error)
        for gain_name in ("kps", "kds"):
            for index, gain in enumerate(getattr(self, gain_name)):
                if error := _non_negative_number_error(gain, f"{gain_name}[{index}]", "WBCConfig"):
                    raise ValueError(error)
        if not isinstance(self.obs_scales, Mapping):
            raise ValueError(
                f"WBCConfig.obs_scales must be a mapping of scale name to number, got {type(self.obs_scales).__name__}."
            )
        for scale_name, scale in self.obs_scales.items():
            if error := finite_number_error(scale, f"obs_scales[{scale_name!r}]", "WBCConfig"):
                raise ValueError(error)
        # Fill the scales this config does not state from the upstream table, so
        # ``obs_scales`` is the complete map the observation is actually built
        # with. Done here - on the config, once - rather than per consumer, for
        # the reason WBCPolicy fills the per-joint SONIC vectors on the config:
        # the observation builder and the controller must see the same values.
        # A stated scale always wins; only an omitted one is filled. Frozen
        # dataclass, so the normalised map is installed with object.__setattr__.
        if any(name not in self.obs_scales for name in _DEFAULT_OBS_SCALES):
            object.__setattr__(self, "obs_scales", {**_DEFAULT_OBS_SCALES, **self.obs_scales})

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
        explicit ``obs_scales`` map, if present, takes precedence. A config that
        states only some of the scales keeps the documented default for the rest
        (:meth:`__post_init__` completes the map), so naming one scale does not
        change what an unnamed sibling is scaled by.
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
