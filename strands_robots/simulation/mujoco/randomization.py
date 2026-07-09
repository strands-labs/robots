"""Domain randomization mixin."""

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots.simulation.mujoco.backend import _NO_WORLD_MSG, _ensure_mujoco

logger = logging.getLogger(__name__)


class RandomizationMixin:
    """Domain randomization mixed into ``Simulation``.

    Recolors geoms, perturbs lighting, and scales body mass (with a matching
    inertia scale, so randomized bodies stay physically consistent) and geom
    friction by a random factor inside a user-supplied range.

    **Coupling** (see the :mod:`simulation` top-level docstring): mixin reaches
    into ``self._world``, ``self._lock``, and the host's
    ``_require_no_running_policy`` / ``_require_world`` helpers. ``TYPE_CHECKING``
    stubs below exist so mypy accepts those lookups; they are a
    documentary contract, not an enforceable protocol.
    """

    if TYPE_CHECKING:
        import threading

        from strands_robots.simulation.models import SimWorld

        _lock: "threading.RLock"
        _world: "SimWorld | None"
        _obs_noise: "dict[str, float] | None"
        _obs_noise_rng: "np.random.Generator | None"

        def _require_no_running_policy(
            self, action_name: str, robot_name: str | None = None
        ) -> dict[str, Any] | None: ...
        def _require_world(self) -> dict[str, Any] | None: ...

    def randomize(
        self,
        randomize_colors: bool = True,
        randomize_lighting: bool = True,
        randomize_physics: bool = False,
        randomize_positions: bool = False,
        position_noise: float = 0.02,
        color_range: tuple[float, float] = (0.1, 1.0),
        friction_range: tuple[float, float] = (0.5, 1.5),
        mass_range: tuple[float, float] = (0.5, 2.0),
        seed: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Apply domain randomization to the scene.

        Each flag is opt-in per-axis. Defaults:
          - ``randomize_colors=True`` - geom RGB re-sampled in ``color_range``.
          - ``randomize_lighting=True`` - light pos jittered ±0.5m, diffuse resampled.
          - ``randomize_physics=False`` - friction/mass left untouched unless asked.
          - ``randomize_positions=False`` - object qpos left untouched unless asked.

        "No flags" means "nothing is randomized" - the call is a no-op. This
        matches the LLM ergonomics principle: explicit is better than implicit.
        Randomization IS destructive (writes to ``model.geom_*`` / ``body_*``
        arrays and to ``data.qpos``); recompile the scene to undo.

        Args:
            randomize_colors:     Re-sample every non-ground geom's RGB (and
                                  its material colour, which overrides geom RGB
                                  in the renderer).
            randomize_lighting:   Jitter light positions + diffuse colour.
            randomize_physics:    Scale geom friction and body mass (body
                                  inertia is scaled by the same factor as the
                                  mass so each randomized body stays physically
                                  consistent).
            randomize_positions:  Add uniform noise to dynamic-object xyz.
            position_noise:       Max ± xyz offset in meters when randomising positions.
            color_range:          (lo, hi) for uniform RGB sampling.
            friction_range:       (lo, hi) multiplicative scale on friction[0].
            mass_range:           (lo, hi) multiplicative scale on body_mass.
            seed:                 Optional np.random seed for reproducibility.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        # domain randomization mutates model arrays; a running policy racing with it is UB
        if err := self._require_no_running_policy("randomize"):
            return err

        rng = np.random.default_rng(seed)
        mj = _ensure_mujoco()
        model = self._world._model
        data = self._world._data
        changes = []

        with self._lock:
            if randomize_colors:
                # Recolor every geom except the ground plane. Two correctness
                # points, both previously silent:
                #   1. Robot mesh geoms are typically UNNAMED, so a truthiness
                #      check on the name skipped them entirely - the robot kept
                #      its original colors while the call reported success.
                #   2. A geom that references a material draws its colour from
                #      that material in the renderer, NOT from geom_rgba, so the
                #      recolor is visually inert unless the material is updated
                #      too. Geoms sharing one material converge to the last
                #      colour written - acceptable for domain randomization.
                n_recolored = 0
                for i in range(model.ngeom):
                    if mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i) == "ground":
                        continue
                    color = rng.uniform(color_range[0], color_range[1], size=3)
                    model.geom_rgba[i, :3] = color
                    matid = int(model.geom_matid[i])
                    if matid >= 0:
                        model.mat_rgba[matid, :3] = color
                    n_recolored += 1
                changes.append(f"Colors: {n_recolored} geoms randomized")

            if randomize_lighting:
                for i in range(model.nlight):
                    model.light_pos[i] += rng.uniform(-0.5, 0.5, size=3)
                    model.light_diffuse[i] = rng.uniform(0.3, 1.0, size=3)
                changes.append(f"Lighting: {model.nlight} lights randomized")

            if randomize_physics:
                friction_scales = {}
                for i in range(model.ngeom):
                    gn = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
                    f = float(rng.uniform(*friction_range))
                    model.geom_friction[i, 0] *= f
                    friction_scales[gn] = f
                mass_scales = {}
                for i in range(model.nbody):
                    if model.body_mass[i] > 0:
                        bn = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
                        s = float(rng.uniform(*mass_range))
                        model.body_mass[i] *= s
                        # Inertia tracks mass for fixed geometry: scaling a
                        # rigid body's mass by ``s`` at constant shape (a uniform
                        # density change) scales its inertia tensor by the same
                        # ``s`` (I = integral of r^2 dm). Scaling mass alone
                        # leaves a physically inconsistent body - heavy in
                        # translation but with the light body's rotational
                        # resistance - which silently corrupts the dynamics the
                        # randomization is meant to perturb. Match the Newton
                        # backend, which scales both.
                        model.body_inertia[i] *= s
                        mass_scales[bn] = s
                changes.append(
                    f"Physics: {len(friction_scales)} geoms friction-scaled, {len(mass_scales)} bodies mass-scaled"
                )
                changes.append(f"   friction_scales={friction_scales}")
                changes.append(f"   mass_scales={mass_scales}")

            if randomize_positions:
                for obj_name, obj in self._world.objects.items():
                    if not obj.is_static:
                        jnt_name = f"{obj_name}_joint"
                        jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
                        if jnt_id >= 0:
                            qpos_addr = model.jnt_qposadr[jnt_id]
                            noise = rng.uniform(-position_noise, position_noise, size=3)
                            data.qpos[qpos_addr : qpos_addr + 3] += noise
                changes.append(f"Positions: ±{position_noise}m noise on dynamic objects")

            # Recompute derived state so the sim is left render-ready. Several
            # randomization axes mutate model arrays whose rendered/simulated
            # effect flows through data: light_pos -> data.light_xpos (the
            # array the renderer reads, NOT model.light_pos), and object qpos ->
            # body xpos. Without a forward the next render()/get_observation()
            # keeps stale derived values, so a light-position jitter is a silent
            # visual no-op until some later mj_step. Mirror the mutate-then-
            # forward contract already used by reset(), load_scene() and
            # move_object(). Guarded on ``changes`` so a no-flag call stays a
            # true no-op.
            if changes:
                mj.mj_forward(model, data)

        return {
            "status": "success",
            "content": [{"text": "Domain Randomization applied:\n" + "\n".join(changes)}],
        }

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
        on MuJoCo data do not assume noise-free sensing. Once set, the noise is
        applied on every :meth:`get_observation` / :meth:`get_robot_state` and
        every rendered camera frame (:meth:`render` and the camera frames in
        ``get_observation``) until reconfigured. Pass all-zero std to disable -
        with every std zero the noise path is an exact no-op, so leaving this
        unconfigured (the default) leaves every observation and render
        byte-for-byte unchanged. Mirrors :meth:`NewtonSimEngine.set_obs_noise`
        so an identical call behaves the same on both backends.

        Args:
            joint_pos_std: Std (radians) of Gaussian noise added to joint
                positions in ``get_observation`` and ``get_robot_state``.
            joint_vel_std: Std (rad/s) of Gaussian noise added to per-joint
                velocities - the ``<joint>.vel`` entries in ``get_observation``
                and the ``velocity`` field in ``get_robot_state``.
            camera_jitter_px: Max integer pixel shift applied to rendered
                frames (uniform in ``[-px, px]`` per axis).
            seed: Optional seed for a reproducible noise stream.
            **kwargs: Accepted for ``SimEngine.set_obs_noise`` signature
                compatibility; ignored by the MuJoCo backend.

        Returns:
            Status dict echoing the configured noise, or an error dict when any
            value is negative or non-finite.
        """
        for label, value in (
            ("joint_pos_std", joint_pos_std),
            ("joint_vel_std", joint_vel_std),
            ("camera_jitter_px", camera_jitter_px),
        ):
            try:
                fvalue = float(value)
            except (TypeError, ValueError):
                return {
                    "status": "error",
                    "content": [{"text": f"set_obs_noise: {label} must be a number, got {value!r}"}],
                }
            if not np.isfinite(fvalue) or fvalue < 0:
                return {
                    "status": "error",
                    "content": [
                        {"text": f"set_obs_noise: {label} must be a finite non-negative number, got {value!r}"}
                    ],
                }

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

    def _apply_obs_noise(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Return ``obs`` with configured sensor noise applied.

        ``get_observation`` returns a heterogeneous dict: scalar joint positions
        keyed by joint name, scalar per-joint velocities keyed ``<joint>.vel``,
        camera frames as ``(H, W, 3)`` uint8 arrays, and (for floating-base
        robots) ``base_quat`` / ``base_ang_vel`` list values. Position noise
        (``joint_pos_std``) applies to the position scalars, velocity noise
        (``joint_vel_std``) to the ``.vel`` scalars, and camera jitter
        (``camera_jitter_px``) to the image arrays. The floating-base list
        signals are left untouched (a quaternion would need renormalization;
        out of scope for additive scalar noise). A no-op returning the input
        unchanged when no noise is configured.
        """
        cfg = self._obs_noise or {}
        rng = self._obs_noise_rng
        if rng is None or not cfg:
            return obs
        pos_std = cfg.get("joint_pos_std", 0.0)
        vel_std = cfg.get("joint_vel_std", 0.0)
        px = cfg.get("camera_jitter_px", 0.0)
        if pos_std <= 0 and vel_std <= 0 and px <= 0:
            return obs
        out: dict[str, Any] = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                out[key] = self._maybe_jitter_frame(value) if px > 0 else value
            elif isinstance(value, float):
                if key.endswith(".vel"):
                    out[key] = value + (float(rng.normal(0.0, vel_std)) if vel_std > 0 else 0.0)
                else:
                    out[key] = value + (float(rng.normal(0.0, pos_std)) if pos_std > 0 else 0.0)
            else:
                out[key] = value
        return out

    def _apply_state_noise(self, state: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        """Return ``get_robot_state`` output with position + velocity noise.

        Entries are ``{joint: {"position": p, "velocity": v}}``. Position noise
        uses ``joint_pos_std`` and velocity noise uses ``joint_vel_std`` from
        :meth:`set_obs_noise`. A no-op when neither std is positive.
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

    def _maybe_jitter_frame(self, frame: "np.ndarray") -> "np.ndarray":
        """Return ``frame`` shifted by a random integer pixel offset.

        Applies ``camera_jitter_px`` configured via :meth:`set_obs_noise` by
        rolling the image along both axes. A no-op when jitter is disabled.
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
