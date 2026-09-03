"""Gait-clock WBC variant - the single-policy SONIC controller with a phase clock.

NVIDIA's GR00T-WholeBodyControl ships *two* MuJoCo reference runners for the
G1 (``decoupled_wbc/sim2mujoco/scripts``):

* ``run_mujoco_gear_wbc.py`` - the **non-gait** controller: an 86-wide
  observation, a 7-wide command block, and TWO ONNX policies (a balance
  ``policy`` + a ``walk_policy`` selected by velocity). That layout is
  implemented by :class:`~strands_robots.policies.wbc.policy.WBCPolicy`.
* ``run_mujoco_gear_wbc_gait.py`` - the **gait-clock** controller: a 95-wide
  observation, an 8-wide command block (a ``freq_cmd`` step-frequency slot
  inserted at index 4), a SINGLE ONNX policy, and a 2-dim bipedal **clock
  signal** appended to each frame. This module ports that variant.

The gait clock is the new ingredient. It is a small stateful phase generator
(:class:`GaitClock`) that turns the velocity command + step frequency into a
two-element ``[clock_FL, clock_FR]`` signal - a left/right-foot phase offset by
half a cycle - that the network consumes as its locomotion rhythm. The math is
a verbatim NumPy port of the upstream ``GearWbcController.compute_observation``
gait block (no torch dependency), so it is unit-testable against hand-computed
values on any machine.

The 95-dim frame layout (for ``no = n_obs_joints``, ``na = num_actions``,
``c = command_dim`` = 8)::

    [0      : c       ]  command  [vx*s, vy*s, omega*s, height, freq, roll, pitch, yaw]
    [c      : c+3     ]  base angular velocity  (scaled by obs_scales.ang_vel)
    [c+3    : c+6     ]  projected gravity       (orientation cue, unscaled)
    [c+6    : c+9     ]  torso angular velocity  (RESERVED - upstream writes zeros)
    [c+9    : c+12    ]  torso projected gravity (RESERVED - upstream writes zeros)
    [c+12   : c+12+no ]  joint positions qj      (defaults subtracted, * dof_pos)
    [c+12+no: c+12+2no]  joint velocities dqj    (scaled by obs_scales.dof_vel)
    [c+12+2no : +na   ]  previous action         (na-dim, the controlled set)
    [.. +na : +na+2   ]  clock signal            [clock_FL, clock_FR]

For the upstream G1 defaults (c=8, no=29, na=15) the populated width is
8 + 3 + 3 + 3 + 3 + 29 + 29 + 15 + 2 = 95 = ``single_obs_dim`` exactly.

CRITICAL (matching the non-gait builder): ``qj``/``dqj`` observe ALL the
robot's joints (upstream ``n_joints`` = nq-7 = 29 for the G1: legs+waist+arms),
NOT just the 15 controlled joints; the two **torso** blocks are reserved and
written as zeros by the upstream runner; and ``default_angles`` (length
``num_actions``) is zero-padded to ``no`` for the qj subtraction.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from strands_robots.utils import positive_finite_number_error

from .config import _DEFAULT_OBS_SCALES, WBCConfig
from .control import compute_targets, projected_gravity
from .policy import WBCPolicy

logger = logging.getLogger(__name__)

# Upstream gait-variant observation/command dimensions
# (run_mujoco_gear_wbc_gait.py: single_obs_dim hardcoded 95; command width 8).
GAIT_SINGLE_OBS_DIM = 95
GAIT_COMMAND_DIM = 8

# Control period of the upstream gait runner: simulation_dt (0.005) x
# control_decimation (4) = 0.02 s. The clock advances by ``dt * freq`` each
# control tick and ``just_started`` accumulates by ``dt``, so this is the
# reference period only - a rollout integrates at the rate the runtime states
# (:meth:`WBCGaitPolicy._control_dt`), because a period that is not the loop's
# own scales the realised step frequency by ``control_frequency / 50``.
_GAIT_CONTROL_DT = 0.02


class GaitClock:
    """Stateful bipedal phase-clock generator (NumPy port of the SONIC gait block).

    Reproduces the clock computation in NVIDIA's
    ``GearWbcController.compute_observation`` (``run_mujoco_gear_wbc_gait.py``)
    without torch. Each :meth:`update` advances the internal gait phase by
    ``dt * freq`` and returns the two-element ``[clock_FL, clock_FR]`` signal the
    network consumes - the left foot and right foot offset by half a cycle.

    Behaviour reproduced verbatim from upstream:

    * **Static hold**: when the (scaled) velocity command norm is below
      :data:`STATIC_VEL_THRESHOLD` the robot is "static"; the walk-start
      bookkeeping resets and, once a clock channel reaches its sine peak
      (> :data:`FREEZE_CLOCK_THRESHOLD`), that foot is *frozen* at ``1.0`` so the
      stance is held still rather than continuing to oscillate.
    * **Walk entry**: the first non-static tick reseeds ``gait_indices`` to
      ``-0.25`` and clears the freeze flags, so a fresh gait cycle starts cleanly.
    * **Warm-up**: for the first ``0.5 / freq`` seconds of walking the right-foot
      phase is pinned to ``0.25`` (upstream ``just_started`` ramp), easing the
      robot into the cycle.

    The generator is deterministic; :meth:`reset` returns it to the
    episode-start state.
    """

    PHASE = 0.5
    DURATION = 0.5
    STATIC_VEL_THRESHOLD = 0.1
    FREEZE_CLOCK_THRESHOLD = 0.98

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Return the clock to its episode-start state (call at episode boundaries)."""
        self.gait_indices: float = 0.0
        self.walking_mask: bool = False
        self.just_started: float = 0.0
        self.frozen_FL: bool = False
        self.frozen_FR: bool = False
        self.clock_inputs: NDArray[np.float64] = np.zeros(2, dtype=np.float64)

    def update(
        self,
        scaled_command_vel: NDArray[np.float64],
        freq: float,
        dt: float = _GAIT_CONTROL_DT,
    ) -> NDArray[np.float64]:
        """Advance the clock one control tick and return ``[clock_FL, clock_FR]``.

        Args:
            scaled_command_vel: The command velocity AFTER ``cmd_scale`` (the
                first three command-block entries ``[vx*s, vy*s, omega*s]``), as
                the upstream static test uses the scaled command norm.
            freq: Step frequency command (``freq_cmd``, the command-block slot
                [4]). Must be strictly positive - it sets both the phase
                increment and the warm-up window ``0.5 / freq``.
            dt: Control period in seconds (default 0.02 = upstream
                ``simulation_dt`` 0.005 x ``control_decimation`` 4).

        Returns:
            A length-2 ``float64`` array ``[clock_FL, clock_FR]``, a fresh copy
            each call.

        Raises:
            ValueError: If ``freq`` is not strictly positive (a zero/negative
                step frequency has no defined warm-up window or phase rate).
        """
        vel = np.asarray(scaled_command_vel, dtype=np.float64).ravel()
        if vel.shape[0] < 3:
            raise ValueError(f"scaled_command_vel must have at least 3 entries, got {vel.shape[0]}")
        if not math.isfinite(freq) or freq <= 0.0:
            raise ValueError(f"GaitClock.update: freq must be finite and > 0, got {freq!r}")

        is_static = bool(np.linalg.norm(vel[:3]) < self.STATIC_VEL_THRESHOLD)
        just_entered_walk = (not is_static) and (not self.walking_mask)
        self.walking_mask = not is_static

        if just_entered_walk:
            self.just_started = 0.0
            self.gait_indices = -0.25
        if not is_static:
            self.just_started += dt
            self.frozen_FL = False
            self.frozen_FR = False
        else:
            self.just_started = 0.0

        self.gait_indices = float((self.gait_indices + dt * freq) % 1.0)

        # Foot phases: right foot at the gait index, left foot half a cycle later.
        gait_FR = self.gait_indices
        gait_FL = (gait_FR + self.PHASE) % 1.0
        if self.just_started < (0.5 / freq):
            gait_FR = 0.25
        gait_pair = [gait_FL, gait_FR]

        # Stretch each phase into the [0, 1) clock domain about the duty point.
        stretched: list[float] = []
        for fi in gait_pair:
            if fi < self.DURATION:
                stretched.append(fi * (0.5 / self.DURATION))
            else:
                stretched.append(0.5 + (fi - self.DURATION) * (0.5 / (1.0 - self.DURATION)))

        clock = [math.sin(2.0 * math.pi * fi) for fi in stretched]

        # Static freeze: hold a foot at the sine peak so the stance stays still.
        for i, attr in enumerate(("frozen_FL", "frozen_FR")):
            frozen = getattr(self, attr)
            if is_static and (not frozen) and clock[i] > self.FREEZE_CLOCK_THRESHOLD:
                setattr(self, attr, True)
                clock[i] = 1.0
            if getattr(self, attr):
                clock[i] = 1.0

        self.clock_inputs = np.asarray(clock, dtype=np.float64)
        return self.clock_inputs.copy()


def build_gait_frame(
    config: WBCConfig,
    *,
    command: NDArray[np.float64],
    base_ang_vel: NDArray[np.float64],
    proj_gravity: NDArray[np.float64],
    qj: NDArray[np.float64],
    dqj: NDArray[np.float64],
    prev_action: NDArray[np.float64],
    clock: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Assemble one ``single_obs_dim``-wide gait-variant observation frame.

    Reproduces the layout written by the upstream
    ``GearWbcController.compute_observation`` gait runner (95-dim for the G1).
    The two **torso** blocks (``[c+6:c+9]`` torso angular velocity, ``[c+9:c+12]``
    torso projected gravity) are reserved and written as zeros, matching the
    upstream runner which sets ``single_obs[14:17] = 0`` and
    ``single_obs[17:20] = 0``. See the module docstring for the full layout.

    Args:
        config: The policy config (dims, scales, default angles). Must have
            ``command_dim == 8`` and a ``single_obs_dim`` large enough for the
            assembled width.
        command: Locomotion command, length ``command_dim`` (8): velocity[3],
            height[1], freq[1], rpy[3].
        base_ang_vel: Base angular velocity (rad/s), length 3.
        proj_gravity: Gravity direction in the body frame, length 3.
        qj: Measured joint positions for ALL observed joints, length
            ``n_obs_joints``.
        dqj: Measured joint velocities, length ``n_obs_joints``.
        prev_action: Previous network action, length ``num_actions``.
        clock: The two-element gait clock signal ``[clock_FL, clock_FR]`` from
            :meth:`GaitClock.update`.

    Returns:
        A ``(single_obs_dim,)`` float64 array.

    Raises:
        ValueError: If any sub-vector has the wrong length, or the assembled
            frame would overflow ``single_obs_dim``.
    """
    no = config.n_obs_joints
    na = config.num_actions
    c = config.command_dim

    command = np.asarray(command, dtype=np.float64).ravel()
    if command.shape[0] > c:
        raise ValueError(f"command length {command.shape[0]} exceeds command_dim {c}")
    if command.shape[0] < c:
        command = np.concatenate([command, np.zeros(c - command.shape[0], dtype=np.float64)])

    base_ang_vel = _require_len(base_ang_vel, 3, "base_ang_vel")
    proj_gravity = _require_len(proj_gravity, 3, "proj_gravity")
    qj = _require_len(qj, no, "qj")
    dqj = _require_len(dqj, no, "dqj")
    prev_action = _require_len(prev_action, na, "prev_action")
    clock = _require_len(clock, 2, "clock")

    defaults = np.zeros(no, dtype=np.float64)
    if config.default_angles:
        da = np.asarray(config.default_angles, dtype=np.float64)
        limit = min(da.shape[0], no)
        defaults[:limit] = da[:limit]
    # WBCConfig completes an incomplete obs_scales map at construction, so these
    # keys are present. The fallback resolves from the same upstream table rather
    # than from a bare 1.0, so a config assembled outside WBCConfig is scaled the
    # way the layout documents instead of 20x on the dqj block.
    ang_vel_scale = config.obs_scales.get("ang_vel", _DEFAULT_OBS_SCALES["ang_vel"])
    dof_pos_scale = config.obs_scales.get("dof_pos", _DEFAULT_OBS_SCALES["dof_pos"])
    dof_vel_scale = config.obs_scales.get("dof_vel", _DEFAULT_OBS_SCALES["dof_vel"])

    frame = np.zeros(config.single_obs_dim, dtype=np.float64)
    end = c + 12 + 2 * no + na + 2
    if end > config.single_obs_dim:
        raise ValueError(
            f"gait observation layout needs {end} values (command_dim={c}, n_obs_joints={no}, "
            f"num_actions={na}, +6 torso slots +2 clock) but single_obs_dim={config.single_obs_dim}; "
            "check the config (the gait variant expects single_obs_dim=95 for the G1)."
        )

    frame[0:c] = command
    frame[c : c + 3] = base_ang_vel * ang_vel_scale
    frame[c + 3 : c + 6] = proj_gravity
    # frame[c+6 : c+12] reserved torso blocks - left zero (upstream writes zeros).
    frame[c + 12 : c + 12 + no] = (qj - defaults) * dof_pos_scale
    frame[c + 12 + no : c + 12 + 2 * no] = dqj * dof_vel_scale
    frame[c + 12 + 2 * no : c + 12 + 2 * no + na] = prev_action
    frame[c + 12 + 2 * no + na : c + 12 + 2 * no + na + 2] = clock
    return frame


def _require_len(vec: NDArray[np.float64], n: int, name: str) -> NDArray[np.float64]:
    arr = np.asarray(vec, dtype=np.float64).ravel()
    if arr.shape[0] != n:
        raise ValueError(f"{name} must have length {n}, got {arr.shape[0]}")
    return arr


class WBCGaitPolicy(WBCPolicy):
    """Gait-clock SONIC locomotion variant for the Unitree G1 (single ONNX policy).

    Ports the upstream ``run_mujoco_gear_wbc_gait.py`` controller: a 95-dim
    observation with a ``freq_cmd`` step-frequency command slot, a 2-dim bipedal
    clock signal (:class:`GaitClock`), and a SINGLE ONNX policy (no walk/balance
    split). Everything else - the SONIC PD gains, the name-resolved G1 joint
    mapping, the checkpoint resolution, the torque-deploy hook - is inherited
    unchanged from :class:`WBCPolicy`, so a gait checkpoint plugs into the same
    sim plumbing (``sim.run_policy`` auto-installs the torque shim because this
    is a ``WBCPolicy`` subclass).

    The shipped ``GR00T-WholeBodyControl-Balance.onnx`` / ``-Walk.onnx`` weights
    are the *non-gait* family (516-wide input); this variant expects a
    gait-clock checkpoint whose ONNX input is ``[batch, 570]`` (95 x 6) and
    output ``[batch, 15]``. Provide it via ``checkpoint=`` exactly as the base
    policy does.

    Args:
        checkpoint: Local dir / ``.onnx`` path / HuggingFace id of the gait ONNX
            policy (see :class:`WBCPolicy`). Only a single ``policy.onnx`` is
            loaded - there is no walk policy in the gait variant.
        config: A :class:`WBCConfig`, config path, or dict. When ``None`` a
            gait-shaped default is used (``single_obs_dim=95``, ``command_dim=8``).
            An explicit config whose dims disagree with the gait layout is
            rejected at construction.
        target_velocity: Optional constructor-time default ``[vx, vy, omega]``.
        gait_frequency: Optional constructor-time default step frequency
            (``freq_cmd``), in steps/s. Per-call ``gait_frequency`` kwarg
            overrides it, which overrides ``config.freq_cmd``. Must be a finite
            number ``> 0`` (see :meth:`_validate_gait_frequency`).
        allow_missing_models: Test/CI seam (see :class:`WBCPolicy`).
        **kwargs: Forward-compatibility absorber (ignored unknown kwargs).
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        config: str | dict[str, Any] | WBCConfig | None = None,
        target_velocity: list[float] | None = None,
        gait_frequency: float | None = None,
        allow_missing_models: bool = False,
        **kwargs: Any,
    ) -> None:
        self._gait_clock = GaitClock()
        # Latch for the missing-rate warning below, so an unset control frequency
        # is reported once per policy rather than on every tick of the rollout.
        self._warned_missing_control_frequency = False
        # Before ``super().__init__`` - which loads the ONNX session(s) - so a
        # frequency the gait clock cannot honor is reported here rather than on
        # the first ``get_actions`` tick of a started rollout. Mirrors the
        # constructor-time ``target_velocity`` check the base policy already does.
        self._gait_frequency = self._validate_gait_frequency(gait_frequency) if gait_frequency is not None else None
        # The gait variant is a single-policy controller: force walk=False so the
        # base loader fetches only policy.onnx (no walk_policy.onnx).
        super().__init__(
            checkpoint=checkpoint,
            config=config,
            walk=False,
            target_velocity=target_velocity,
            allow_missing_models=allow_missing_models,
            **kwargs,
        )

    @property
    def provider_name(self) -> str:
        """Registry key for this provider (``"wbc_gait"``)."""
        return "wbc_gait"

    def _resolve_config(self, config: str | dict[str, Any] | WBCConfig | None, checkpoint: str | None) -> WBCConfig:
        """Resolve the config, defaulting to (and enforcing) the gait layout.

        When no config is supplied (and the checkpoint ships no ``config.json``)
        a gait-shaped default is built (``single_obs_dim=95``, ``command_dim=8``,
        single policy). Any resolved config whose dims disagree with the gait
        layout is rejected up front (AGENTS.md #5: fail fast on a fatal config) -
        loading a non-gait config into the gait observation builder would
        misplace the clock/torso slots. ``freq_cmd`` is held to the gait clock's
        stricter ``> 0`` domain here for the same reason: the config's own rule
        is finiteness, which is all the non-gait variant (with no step-frequency
        slot) can require.
        """
        cfg = super()._resolve_config(config, checkpoint)
        if config is None and cfg.single_obs_dim == 86 and cfg.command_dim == 7:
            # Base fell back to the non-gait default (no explicit config / no
            # config.json). Promote it to the gait layout and drop the walk path.
            import dataclasses

            cfg = dataclasses.replace(
                cfg,
                single_obs_dim=GAIT_SINGLE_OBS_DIM,
                command_dim=GAIT_COMMAND_DIM,
                walk_policy_path=None,
            )
        if cfg.single_obs_dim != GAIT_SINGLE_OBS_DIM or cfg.command_dim != GAIT_COMMAND_DIM:
            raise ValueError(
                f"WBCGaitPolicy requires the gait observation layout (single_obs_dim="
                f"{GAIT_SINGLE_OBS_DIM}, command_dim={GAIT_COMMAND_DIM}), but the resolved config has "
                f"single_obs_dim={cfg.single_obs_dim}, command_dim={cfg.command_dim}. "
                "Use WBCPolicy for the non-gait (86-dim, two-policy) family, or supply a gait config."
            )
        # The gait clock needs freq_cmd > 0 (WBCConfig only requires finite,
        # because the non-gait command block has no step-frequency slot to read).
        if error := positive_finite_number_error(cfg.freq_cmd, "config.freq_cmd", "WBCGaitPolicy"):
            raise ValueError(
                f"{error} The gait clock advances the phase by dt * freq_cmd and warms up over "
                "0.5 / freq_cmd, so a non-positive step frequency cannot be honored by this variant."
            )
        return cfg

    def reset(self, seed: int | None = None) -> None:
        """Clear the observation history, previous action, AND the gait clock."""
        super().reset(seed)
        self._gait_clock.reset()

    def _resolve_command(self, kwargs: dict[str, Any]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Build the 8-dim gait command block for this tick.

        Faithful to the upstream gait ``compute_observation``::

            command[0:3] = loco_cmd[:3] * cmd_scale   # velocity, scaled
            command[3]   = height_cmd                  # target base height
            command[4]   = freq_cmd                    # step frequency
            command[5:8] = rpy_cmd                     # target roll/pitch/yaw

        Note the ``freq_cmd`` slot at index 4 (absent in the non-gait 7-wide
        command) pushes rpy to slots [5:8]. Frequency precedence: per-call
        ``gait_frequency`` kwarg > constructor default > ``config.freq_cmd``.

        Every source in that chain is validated on the same domain, so the
        spelling that wins the precedence contest cannot accept a value the
        spelling that loses it would refuse: the per-call kwargs here, the
        constructor default in :meth:`__init__`, and ``config.freq_cmd`` in
        :meth:`_resolve_config`.

        Returns:
            ``(command, raw_velocity)`` - the 8-wide command block with
            ``cmd_scale`` applied to the velocity, and the UNSCALED ``[vx, vy,
            omega]`` triple.
        """
        tv = kwargs.get("target_velocity")
        if tv is not None:
            vel_full = self._validate_velocity(tv)
        elif self._default_command is not None:
            vel_full = self._default_command.copy()
        else:
            vel_full = np.zeros(3, dtype=np.float64)
        raw_velocity = vel_full[:3].copy()

        c = self._config.command_dim
        command = np.zeros(c, dtype=np.float64)

        cmd_scale = np.asarray(self._config.cmd_scale, dtype=np.float64).ravel()
        n_vel = min(3, c)
        scale = cmd_scale[:n_vel] if cmd_scale.shape[0] >= n_vel else np.ones(n_vel)
        command[:n_vel] = raw_velocity[:n_vel] * scale

        if c > 3:
            height = kwargs.get("height")
            command[3] = self._validate_height(height) if height is not None else float(self._config.height_cmd)

        if c > 4:
            freq = kwargs.get("gait_frequency")
            if freq is not None:
                freq = self._validate_gait_frequency(freq)
            elif self._gait_frequency is not None:
                freq = self._gait_frequency
            else:
                freq = self._config.freq_cmd
            command[4] = float(freq)

        if c > 5:
            rpy_src = kwargs.get("target_orientation")
            rpy = (
                self._validate_orientation(rpy_src)
                if rpy_src is not None
                else np.asarray(self._config.rpy_cmd, dtype=np.float64).ravel()
            )
            n_rpy = min(c - 5, rpy.shape[0])
            command[5 : 5 + n_rpy] = rpy[:n_rpy]

        return command, raw_velocity

    def _control_dt(self) -> float:
        """Control period the gait phase is integrated over, in seconds.

        The clock advances by ``dt * freq`` per tick, so ``dt`` has to be the
        period the executing loop actually queries this policy at - otherwise
        the commanded ``gait_frequency`` is not in steps per second. The runtime
        states that rate via
        :meth:`~strands_robots.policies.base.Policy.set_control_frequency`
        before the rollout loop starts, and this reads it there.

        Returns:
            ``1 / control_frequency`` when the runtime has stated a rate,
            otherwise the upstream reference period
            (:data:`_GAIT_CONTROL_DT`), which is what a caller driving
            ``get_actions`` directly gets.

        Notes:
            An unstated rate warns rather than assuming one silently, per the
            :attr:`~strands_robots.policies.base.Policy.control_frequency`
            contract: a wrong assumed period scales the realised step frequency
            by ``control_frequency / 50`` at every rate except the assumed one,
            which is a robot walking at a cadence nobody commanded. Warned once
            per policy - the condition cannot change mid-rollout, and a per-tick
            line at control rate is a flood.
        """
        if self.control_frequency is None:
            if not self._warned_missing_control_frequency:
                self._warned_missing_control_frequency = True
                logger.warning(
                    "WBCGaitPolicy: no control frequency stated, integrating the gait phase at the "
                    "upstream reference period (%.3f s / %.1f Hz). Call set_control_frequency(hz) - "
                    "PolicyRunner does - or the commanded gait_frequency is only in steps/s at %.1f Hz.",
                    _GAIT_CONTROL_DT,
                    1.0 / _GAIT_CONTROL_DT,
                    1.0 / _GAIT_CONTROL_DT,
                )
            return _GAIT_CONTROL_DT
        return 1.0 / self.control_frequency

    @staticmethod
    def _validate_gait_frequency(freq: Any) -> float:
        """Validate a step-frequency command (the ``freq_cmd`` override).

        :meth:`GaitClock.update` documents the domain - ``freq`` "must be
        strictly positive", because it sets both the phase increment and the
        warm-up window ``0.5 / freq`` - and enforces it, but only once the
        command block has been built and handed to it from inside
        :meth:`get_actions`. Deciding it where the caller's value arrives turns a
        mid-rollout :class:`ValueError` naming an internal method into one naming
        the parameter that was supplied.

        The rule is stricter than the ``finite_number_error`` domain
        :class:`~strands_robots.policies.wbc.config.WBCConfig` applies to
        ``freq_cmd``, and deliberately so: the non-gait 7-wide command block has
        no step-frequency slot, so a ``freq_cmd`` of ``0`` is inert for
        :class:`~strands_robots.policies.wbc.policy.WBCPolicy` and the config
        cannot demand positivity on its behalf. The gait variant reads the slot,
        so the positivity rule belongs here - the same place the gait layout
        itself is enforced.

        Args:
            freq: The caller-supplied step frequency (steps/s).

        Returns:
            The value as a ``float``.

        Raises:
            ValueError: If it is not a usable finite number ``> 0``.
        """
        if error := positive_finite_number_error(freq, "gait_frequency", "WBCGaitPolicy"):
            raise ValueError(error)
        return float(freq)

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Run one gait-variant inference step and return the 15-dim joint targets.

        Reads the locomotion command from the well-known kwargs
        (``target_velocity``, ``gait_frequency``, optional ``target_orientation``
        / ``height``); ``instruction`` is ignored. Advances the :class:`GaitClock`
        by the period the executing loop queries this policy at
        (:meth:`_control_dt`), so ``gait_frequency`` is steps per second at any
        control rate rather than only at the upstream reference 50 Hz,
        builds the 95-dim stacked observation, runs the single ONNX policy, and
        returns one per-step action dict keyed by leg+waist actuator name.
        """
        command, raw_velocity = self._resolve_command(kwargs)

        qj, dqj, base_ang_vel, quat = self._extract_state(observation_dict)
        proj_grav = projected_gravity(quat)

        clock = self._gait_clock.update(command[:3], float(command[4]), dt=self._control_dt())

        frame = build_gait_frame(
            self._config,
            command=command,
            base_ang_vel=base_ang_vel,
            proj_gravity=proj_grav,
            qj=qj,
            dqj=dqj,
            prev_action=self._prev_action,
            clock=clock,
        )
        obs = self._history.push(frame)

        raw_action = self._run_session(obs, raw_velocity)
        self._prev_action = raw_action

        target_q = compute_targets(self._default_angles, raw_action, self._config.action_scale)
        keys = self._resolve_action_keys()
        return [{k: float(v) for k, v in zip(keys, target_q, strict=True)}]


__all__ = ["GaitClock", "build_gait_frame", "WBCGaitPolicy", "GAIT_SINGLE_OBS_DIM", "GAIT_COMMAND_DIM"]
