"""Domain randomization and sensor-noise hooks for the Newton backend.

Mixed into :class:`~strands_robots.simulation.newton.simulation.NewtonSimEngine`.
Mirrors the MuJoCo backend's ``randomize`` contract (same keyword names and
defaults for the axes Newton supports) and adds ``set_obs_noise`` for additive
Gaussian sensor noise on joint encoders and camera frames -- the pieces a
sim2real workflow needs so datasets collected on the GPU backend do not overfit
to the default physics constants.

Where MuJoCo mutates a live ``mjModel`` in place, Newton finalises an immutable
``Model`` from a ``ModelBuilder``, so physics randomization (per-body mass and
per-shape friction) is applied to the builder arrays *before* finalisation. The
host's :meth:`NewtonSimEngine._rebuild` calls :meth:`_apply_domain_randomization`
at exactly that point. Lighting is applied at render time by steering the
directional light, and camera-frame jitter is a post-process on the rendered
RGB buffer.

**Coupling** (mirrors the MuJoCo mixin): this mixin reaches into the host's
``_world``, ``_lock``, ``_wp``, ``_rebuild``, and the domain-randomization /
sensor-noise state attributes initialised in ``NewtonSimEngine.__init__``. The
``TYPE_CHECKING`` stubs below are a documentary contract so mypy accepts those
lookups; they are not an enforceable protocol.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots.simulation.base import (
    finite_non_negative_error,
    randomization_range_error,
    randomization_seed_error,
    unknown_kwargs_error,
)
from strands_robots.utils import boolean_flag_error

logger = logging.getLogger(__name__)

# Parameter names ``randomize`` / ``set_obs_noise`` accept. Both declare
# ``**kwargs`` to match the ``**kwargs``-typed SimEngine base signature, and
# ``randomize`` reads ``randomize_positions`` out of it for the MuJoCo-parity
# error below - but nothing else there is used, so any other keyword is a caller
# mistake and is rejected instead of dropped. ``randomize_positions`` and
# ``position_noise`` stay accepted (not unknown) so code written against the
# MuJoCo signature keeps producing Newton's explicit unsupported-axis error
# rather than a confusing "unknown parameter".
_RANDOMIZE_PARAMS: tuple[str, ...] = (
    "randomize_colors",
    "randomize_lighting",
    "randomize_physics",
    "randomize_positions",
    "position_noise",
    "mass_range",
    "friction_range",
    "color_range",
    "seed",
)
_OBS_NOISE_PARAMS: tuple[str, ...] = (
    "joint_pos_std",
    "joint_vel_std",
    "camera_jitter_px",
    "seed",
)


class DomainRandomizationMixin:
    """Domain randomization + sensor-noise hooks for ``NewtonSimEngine``."""

    if TYPE_CHECKING:
        from strands_robots.simulation.models import SimWorld

        _lock: threading.RLock
        _world: SimWorld | None
        _wp: Any
        # Domain-randomization state (initialised in NewtonSimEngine.__init__).
        _dr: dict[str, Any] | None
        _dr_applied: dict[str, Any] | None
        _dr_light_dir: tuple[float, float, float] | None
        # Sensor-noise state.
        _obs_noise: dict[str, float] | None
        _obs_noise_rng: np.random.Generator | None

        def _rebuild(self) -> None: ...

    # Domain randomization

    def randomize(
        self,
        randomize_colors: bool = True,
        randomize_lighting: bool = True,
        randomize_physics: bool = False,
        mass_range: tuple[float, float] = (0.5, 2.0),
        friction_range: tuple[float, float] = (0.5, 1.5),
        color_range: tuple[float, float] = (0.1, 1.0),
        seed: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Apply domain randomization to the Newton scene.

        Keyword names and defaults mirror the MuJoCo backend so randomization
        code transfers across backends unchanged - including the shared
        boolean-flag domain each axis flag is checked on
        (:func:`~strands_robots.utils.boolean_flag_error`), so an axis is not
        turned off by ``"false"``, ``"no"``, ``"off"`` or ``"0"``. Each axis is
        opt-in:

          - ``randomize_colors=True``  - per-shape RGB resampled in ``color_range``.
          - ``randomize_lighting=True`` - directional-light orientation jittered.
          - ``randomize_physics=False`` - per-body mass (``mass_range``) and
            per-shape friction (``friction_range``) scaled; left untouched unless
            asked, matching MuJoCo's default.

        Physics randomization scales the builder's ``body_mass`` and
        ``body_inertia`` (inertia tracks mass for fixed geometry; Newton
        recomputes the inverse mass/inertia at finalisation) and the per-shape
        ``shape_material_mu`` friction coefficient, then rebuilds the model.
        Colors and the sampled light direction take effect on the next
        ``render`` call.

        Reproducibility: a fixed ``seed`` yields an identical multiplier
        sequence for a given scene, because the builder visits bodies and shapes
        in a deterministic order. The applied multipliers are returned in the
        ``json`` block (``mass_scales`` / ``friction_scales`` /
        ``light_direction``) so callers can assert reproducibility or log the
        per-episode physics.

        Args:
            randomize_colors: Resample per-shape RGB.
            randomize_lighting: Jitter the directional-light orientation.
            randomize_physics: Scale per-body mass and per-shape friction.
            mass_range: ``(lo, hi)`` multiplicative scale on body mass. Must
                be strictly positive: a zero multiplier leaves a massless body
                that ignores gravity rather than a lighter one.
            friction_range: ``(lo, hi)`` multiplicative scale on shape friction.
            color_range: ``(lo, hi)`` for uniform RGB sampling.
            seed: Optional seed for reproducible randomization; a non-negative
                integer, or None for fresh entropy.
            **kwargs: Tolerated for MuJoCo-signature parity: ``randomize_positions``
                and ``position_noise`` are accepted keywords, and
                ``randomize_positions=True`` returns an error (Newton does not
                yet support object-position randomization). It is checked on the
                same boolean-flag domain as the declared axes first, so that
                refusal cannot inherit a misread and blame an axis the caller
                asked to skip. Nothing else is
                forwarded, so any other keyword is rejected with an error naming
                the valid parameters instead of being silently dropped.

        Returns:
            Status dict. On success the ``json`` block carries the applied
            multipliers; an error dict is returned when a keyword is unknown, no
            world exists, an axis flag is not a boolean, a range is invalid, or
            an unsupported axis is requested.
        """
        if kwargs_error := unknown_kwargs_error("randomize", kwargs, _RANDOMIZE_PARAMS):
            return kwargs_error
        if self._world is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world (or load_scene) first."}]}
        # Each flag selects a posture, so each is checked on the shared domain
        # rather than read by truthiness - the same rule the MuJoCo backend's
        # keyword parity promises. ``randomize_positions`` is declared only
        # there, so here it arrives through ``**kwargs``; it is checked with the
        # rest and BEFORE the parity refusal below, which branches on it and so
        # would otherwise report an unsupported axis to a caller who had spelled
        # "do not randomize positions" as ``"false"``. The stored spec applies
        # ``bool()`` to each flag, so a misread would persist to every later
        # rebuild rather than to this call alone.
        axis_flags: list[tuple[str, Any]] = [
            ("randomize_colors", randomize_colors),
            ("randomize_lighting", randomize_lighting),
            ("randomize_physics", randomize_physics),
        ]
        if "randomize_positions" in kwargs:
            axis_flags.append(("randomize_positions", kwargs["randomize_positions"]))
        for flag_param, flag_value in axis_flags:
            if msg := boolean_flag_error(flag_value, flag_param, "randomize"):
                return {"status": "error", "content": [{"text": msg}]}
        if kwargs.get("randomize_positions"):
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "randomize_positions is not supported by the Newton backend yet. "
                            "Supported axes: randomize_colors, randomize_lighting, randomize_physics. "
                            "Use the MuJoCo backend for object-position randomization."
                        )
                    }
                ],
            }
        for label, rng_range, allow_zero in (
            # A zero MASS multiplier is not a lighter body, it is a massless one
            # that ignores gravity; zero friction and zero colour are both real
            # physical settings.
            ("mass_range", mass_range, False),
            ("friction_range", friction_range, True),
            ("color_range", color_range, True),
        ):
            if msg := randomization_range_error(rng_range, label, allow_zero=allow_zero):
                return {"status": "error", "content": [{"text": msg}]}
        # The seed is stored now and only reaches ``default_rng`` inside
        # ``_rebuild``, so an unusable one would otherwise raise after this call
        # already reported the randomization applied.
        if msg := randomization_seed_error(seed, "randomize"):
            return {"status": "error", "content": [{"text": msg}]}

        with self._lock:
            self._dr = {
                "randomize_colors": bool(randomize_colors),
                "randomize_lighting": bool(randomize_lighting),
                "randomize_physics": bool(randomize_physics),
                "mass_range": (float(mass_range[0]), float(mass_range[1])),
                "friction_range": (float(friction_range[0]), float(friction_range[1])),
                "color_range": (float(color_range[0]), float(color_range[1])),
                "seed": seed,
            }
            # _rebuild invokes _apply_domain_randomization, which samples the
            # multipliers and populates self._dr_applied / self._dr_light_dir.
            self._rebuild()
            applied = self._dr_applied or {}

        n_mass = len(applied.get("mass_scales", []))
        n_fric = len(applied.get("friction_scales", []))
        changes = []
        if randomize_colors:
            changes.append(f"Colors: {applied.get('n_colors', 0)} shapes randomized")
        if randomize_lighting:
            changes.append(f"Lighting: light direction = {applied.get('light_direction')}")
        if randomize_physics:
            changes.append(f"Physics: {n_mass} bodies mass-scaled, {n_fric} shapes friction-scaled")
        if not changes:
            changes.append("No axes enabled; nothing randomized.")

        return {
            "status": "success",
            "content": [
                {"text": "Domain randomization applied:\n" + "\n".join(changes)},
                {"json": applied},
            ],
        }

    def _apply_domain_randomization(self, builder: Any) -> None:
        """Apply the active randomization spec to a fresh ``ModelBuilder``.

        Called by :meth:`NewtonSimEngine._rebuild` after robots, objects, and
        the ground plane have been added but before ``builder.finalize``. A
        no-op when no randomization spec is active. Must be called with
        ``self._lock`` held. Samples deterministically from ``self._dr["seed"]``
        and records the applied multipliers in ``self._dr_applied``.

        Args:
            builder: The Newton ``ModelBuilder`` being assembled this rebuild.
        """
        dr = self._dr
        if not dr:
            return
        rng = np.random.default_rng(dr["seed"])
        applied: dict[str, Any] = {}

        if dr["randomize_colors"]:
            lo, hi = dr["color_range"]
            n = len(builder.shape_color)
            for i in range(n):
                builder.shape_color[i] = tuple(float(c) for c in rng.uniform(lo, hi, size=3))
            applied["n_colors"] = n

        if dr["randomize_physics"]:
            wp = self._wp
            mlo, mhi = dr["mass_range"]
            mass_scales: list[float] = []
            for i in range(len(builder.body_mass)):
                if builder.body_mass[i] > 0:
                    s = float(rng.uniform(mlo, mhi))
                    builder.body_mass[i] *= s
                    inertia = np.array(builder.body_inertia[i], dtype=np.float32).reshape(3, 3) * s
                    builder.body_inertia[i] = wp.mat33f(inertia)
                    mass_scales.append(s)
            flo, fhi = dr["friction_range"]
            friction_scales: list[float] = []
            for i in range(len(builder.shape_material_mu)):
                f = float(rng.uniform(flo, fhi))
                builder.shape_material_mu[i] *= f
                friction_scales.append(f)
            applied["mass_scales"] = mass_scales
            applied["friction_scales"] = friction_scales

        if dr["randomize_lighting"]:
            # Jitter the default directional light (normalized (-1, 1, -1)) and
            # renormalize so the renderer receives a unit direction.
            base = np.array([-1.0, 1.0, -1.0])
            jitter = rng.uniform(-0.6, 0.6, size=3)
            direction = base + jitter
            norm = float(np.linalg.norm(direction))
            if norm > 1e-6:
                direction = direction / norm
            light_dir = (float(direction[0]), float(direction[1]), float(direction[2]))
            self._dr_light_dir = light_dir
            applied["light_direction"] = light_dir
        else:
            self._dr_light_dir = None

        self._dr_applied = applied

    # Sensor noise

    def set_obs_noise(
        self,
        joint_pos_std: float = 0.0,
        joint_vel_std: float = 0.0,
        camera_jitter_px: float = 0.0,
        seed: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Configure additive Gaussian sensor noise on observations.

        Models real-encoder / real-camera measurement noise so policies trained
        on Newton data do not assume noise-free sensing. Once set, the noise is
        applied on every :meth:`get_observation` / :meth:`get_robot_state` and
        every rendered camera frame until reconfigured. Pass all-zero std to
        disable.

        Args:
            joint_pos_std: Std (radians) of Gaussian noise added to joint
                positions in ``get_observation`` and ``get_robot_state``.
            joint_vel_std: Std (rad/s) of Gaussian noise added to joint
                velocities in ``get_robot_state``.
            camera_jitter_px: Max integer pixel shift applied to rendered
                frames (uniform in ``[-px, px]`` per axis).
            seed: Optional seed for a reproducible noise stream; a non-negative
                integer, or None for fresh entropy.
            **kwargs: Declared only to match the ``**kwargs``-typed
                ``SimEngine.set_obs_noise`` signature; nothing is forwarded, so
                any keyword arriving here is rejected with an error naming the
                valid parameters rather than reporting an all-zero (no-op) noise
                configuration as success.

        Returns:
            Status dict echoing the configured noise, or an error dict when a
            keyword is unknown or a value is negative or non-finite.
        """
        if kwargs_error := unknown_kwargs_error("set_obs_noise", kwargs, _OBS_NOISE_PARAMS):
            return kwargs_error
        for label, value in (
            ("joint_pos_std", joint_pos_std),
            ("joint_vel_std", joint_vel_std),
            ("camera_jitter_px", camera_jitter_px),
        ):
            if msg := finite_non_negative_error(value, label, "set_obs_noise"):
                return {"status": "error", "content": [{"text": msg}]}
        # See the note in ``randomize``: the seed only reaches ``default_rng``
        # below, and on the noise path an unusable one would raise on the first
        # observation drawn rather than at this call.
        if msg := randomization_seed_error(seed, "set_obs_noise"):
            return {"status": "error", "content": [{"text": msg}]}

        with self._lock:
            self._obs_noise = {
                "joint_pos_std": float(joint_pos_std),
                "joint_vel_std": float(joint_vel_std),
                "camera_jitter_px": float(camera_jitter_px),
            }
            self._obs_noise_rng = np.random.default_rng(seed)
        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"Sensor noise: joint_pos_std={joint_pos_std}, "
                        f"joint_vel_std={joint_vel_std}, camera_jitter_px={camera_jitter_px}"
                    )
                }
            ],
        }

    def _apply_joint_pos_noise(self, obs: dict[str, float]) -> dict[str, float]:
        """Return ``obs`` with Gaussian noise added to each joint position.

        A no-op (returns the input unchanged) when no positive-std joint
        position noise is configured.

        Args:
            obs: Mapping of joint name to position (radians).

        Returns:
            New mapping with noise applied, or the original when disabled.
        """
        std = (self._obs_noise or {}).get("joint_pos_std", 0.0)
        rng = self._obs_noise_rng
        if std <= 0 or rng is None or not obs:
            return obs
        return {k: float(v) + float(rng.normal(0.0, std)) for k, v in obs.items()}

    def _apply_state_noise(self, state: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        """Return ``state`` with Gaussian noise added to positions and velocities.

        Used by :meth:`NewtonSimEngine.get_robot_state`, whose entries are
        ``{joint: {"position": p, "velocity": v}}``. Position noise uses
        ``joint_pos_std`` and velocity noise uses ``joint_vel_std`` from
        :meth:`set_obs_noise`. A no-op when neither std is positive.

        Args:
            state: Per-joint ``{"position", "velocity"}`` mapping.

        Returns:
            New mapping with noise applied, or the original when disabled.
        """
        cfg = self._obs_noise or {}
        pos_std = cfg.get("joint_pos_std", 0.0)
        vel_std = cfg.get("joint_vel_std", 0.0)
        rng = self._obs_noise_rng
        if rng is None or (pos_std <= 0 and vel_std <= 0) or not state:
            return state
        out: dict[str, dict[str, float]] = {}
        for jname, vals in state.items():
            pos = vals["position"] + (float(rng.normal(0.0, pos_std)) if pos_std > 0 else 0.0)
            vel = vals["velocity"] + (float(rng.normal(0.0, vel_std)) if vel_std > 0 else 0.0)
            out[jname] = {"position": pos, "velocity": vel}
        return out

    def _maybe_jitter_frame(self, frame: np.ndarray) -> np.ndarray:
        """Return ``frame`` shifted by a random integer pixel offset.

        Applies ``camera_jitter_px`` configured via :meth:`set_obs_noise` by
        rolling the image along both axes. A no-op when jitter is disabled.

        Args:
            frame: ``(H, W, 3)`` uint8 RGB array.

        Returns:
            Jittered array (a rolled view/copy), or the original when disabled.
        """
        px = (self._obs_noise or {}).get("camera_jitter_px", 0.0)
        rng = self._obs_noise_rng
        if px <= 0 or rng is None or frame.ndim < 2:
            return frame
        max_shift = int(px)
        if max_shift < 1:
            return frame
        dy = int(rng.integers(-max_shift, max_shift + 1))
        dx = int(rng.integers(-max_shift, max_shift + 1))
        return np.roll(frame, shift=(dy, dx), axis=(0, 1))
