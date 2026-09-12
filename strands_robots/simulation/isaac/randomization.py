"""Domain randomization + observation sensor noise for the Isaac backend.

The MuJoCo backend owns the reference semantics
(:mod:`strands_robots.simulation.mujoco.randomization`) and the Newton backend
mirrors them (:mod:`strands_robots.simulation.newton.randomization`); this
module is the Isaac third. Keyword names, defaults and parameter ORDER mirror
the MuJoCo signature exactly - a permuted shared name is invisible to every
gate and to Python, so a positional call would silently bind the wrong values
(AGENTS.md item 19; the cross-backend shared-parameter-order grader derives its
backend inventory from the ``create_simulation`` table, so this backend was
held to the rule the moment these methods appeared).

Until this module, ``randomize`` and ``set_obs_noise`` on Isaac were the
``SimEngine`` raising stubs - while ``docs/simulation/domain-randomization.md``
and ~30 example call sites drive ``randomize()`` and ~10 drive
``set_obs_noise()`` through the backend-agnostic surface, so the identical
script randomized on MuJoCo and Newton and raised ``NotImplementedError`` on
Isaac.

What each axis writes on THIS backend (each backend defines its own opt-in
axes; the flags, not the mechanisms, are the shared contract):

* ``randomize_colors`` - every registered object prim's USD
  ``primvars:displayColor`` is resampled per channel in ``color_range``. The
  object handles ship no color setter (measured on isaacsim 6.0.1:
  ``DynamicCuboid`` has ``apply_visual_material`` but no ``set_color``), so the
  write goes to the USD attribute the RTX renderer reads.
* ``randomize_lighting`` - every ``UsdLux`` light on the stage gets its
  intensity scaled in [0.5, 1.5] of its BASE intensity and its color resampled
  in ``color_range``. ``create_world``'s default ground plane ships one
  ``SphereLight`` at intensity 100000 (measured).
* ``randomize_physics`` - every DYNAMIC object's mass is scaled in
  ``mass_range`` of its base mass (``set_mass``), and each distinct physics
  material's static+dynamic friction is scaled in ``friction_range`` of its
  base. Materials are deduplicated by prim path first: ``add_object`` binds a
  material per object today, but scaling per *object* would compound on a
  shared material the day two objects share one.
* ``randomize_positions`` - every dynamic object is teleported to its BASE
  position plus a uniform xy offset in ``[-position_noise, +position_noise]``
  metres. z is preserved: objects rest on surfaces, and a z offset either
  buries them (penetration-recovery launch - measured on this backend, see the
  floating-base work) or drops them.

Anti-compounding, the property the MuJoCo axes document as load-bearing: every
scale and offset is measured from a BASE captured the first time an entity is
touched - before anything has been written - never from the live value, so
repeated calls draw independent samples inside the bound instead of walking
off. MuJoCo anchors on the compiled model's authored values; this backend has
no compiled model, so the anchor is the first-touch snapshot, which is the
authored value by construction (nothing else has written yet). An object added
AFTER a randomize call gets its base captured on the next call, unrandomized -
same effect as MuJoCo's mutate-then-randomize ordering note.

``set_obs_noise`` stores the config; :meth:`IsaacSimulation.get_observation`
applies it through :meth:`_apply_obs_noise`, suffix-keyed exactly as the
MuJoCo pass is (``joint_pos_std`` to position floats, ``joint_vel_std`` to
``.vel`` floats, ``camera_jitter_px`` as an integer-pixel roll on frames,
lists - the ``base_*`` signals - untouched).
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

#: Parameter names ``randomize`` accepts - the full MuJoCo set, because every
#: axis is implemented here. ``**kwargs`` exists to match the ``**kwargs``-typed
#: SimEngine base signature; anything not in this tuple is a caller mistake and
#: is rejected instead of dropped.
_RANDOMIZE_PARAMS: tuple[str, ...] = (
    "randomize_colors",
    "randomize_lighting",
    "randomize_physics",
    "randomize_positions",
    "position_noise",
    "color_range",
    "friction_range",
    "mass_range",
    "seed",
)

_OBS_NOISE_PARAMS: tuple[str, ...] = (
    "joint_pos_std",
    "joint_vel_std",
    "camera_jitter_px",
    "seed",
)

#: Bounds of the lighting intensity scale, fixed rather than a parameter:
#: neither sibling backend exposes a lighting range knob, and inventing one
#: here would be the mid-signature divergence item 19 exists to prevent.
_LIGHT_INTENSITY_SCALE: tuple[float, float] = (0.5, 1.5)


class IsaacRandomizationMixin:
    """``randomize`` / ``set_obs_noise`` for :class:`IsaacSimulation`.

    Expects the host class to provide ``_lock``, ``_world``, ``_world_created``,
    ``_objects`` and ``_config``. State this mixin owns (``_dr_base``,
    ``_obs_noise``, ``_obs_noise_rng``) is created lazily via ``getattr`` so the
    two dozen ``__new__``-skeleton test engines need no new seeds.
    """

    if TYPE_CHECKING:
        # Provided by the host IsaacSimulation; declared for the type checker
        # only (the same pattern the Newton mixin uses).
        _lock: threading.RLock
        _world: Any
        _world_created: bool
        _objects: dict[str, Any]
        _obs_noise: dict[str, float] | None
        _obs_noise_rng: Any
        _dr_base: dict[tuple[str, str], Any]

    # --- randomize -----------------------------------------------------------

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
        """Apply domain randomization to the scene. See the module docstring
        for what each axis writes on this backend.

        No flags means nothing is randomized - the call is a no-op, matching
        the sibling backends. Every flag is checked on the shared boolean-flag
        domain before anything is written: an axis is not turned off by
        ``"false"``/``"no"``/``"off"``/``"0"``, each of which is a truthy
        string that would have turned it ON.

        Returns:
            The standard envelope. On success the ``json`` block records what
            was sampled per axis, keyed by entity, so a run is reproducible
            from its own report plus the ``seed``.
        """
        if kwargs_error := unknown_kwargs_error("randomize", kwargs, _RANDOMIZE_PARAMS):
            return kwargs_error
        if not getattr(self, "_world_created", False) or getattr(self, "_world", None) is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world (or load_scene) first."}]}
        for flag_param, flag_value in (
            ("randomize_colors", randomize_colors),
            ("randomize_lighting", randomize_lighting),
            ("randomize_physics", randomize_physics),
            ("randomize_positions", randomize_positions),
        ):
            if msg := boolean_flag_error(flag_value, flag_param, "randomize"):
                return {"status": "error", "content": [{"text": msg}]}
        for range_param, range_value in (
            ("color_range", color_range),
            ("friction_range", friction_range),
            ("mass_range", mass_range),
        ):
            if msg := randomization_range_error(range_value, range_param):
                return {"status": "error", "content": [{"text": msg}]}
        if msg := finite_non_negative_error(position_noise, "position_noise", "randomize"):
            return {"status": "error", "content": [{"text": msg}]}
        if msg := randomization_seed_error(seed, "randomize"):
            return {"status": "error", "content": [{"text": msg}]}

        rng = np.random.default_rng(seed)
        applied: dict[str, Any] = {"seed": seed}
        changes: list[str] = []
        with self._lock:
            try:
                if randomize_colors:
                    n = self._randomize_colors(rng, color_range, applied)
                    changes.append(f"Colors: {n} object(s) resampled")
                if randomize_lighting:
                    n = self._randomize_lighting(rng, color_range, applied)
                    changes.append(f"Lighting: {n} light(s) rescaled")
                if randomize_physics:
                    n_mass, n_fric = self._randomize_physics(rng, mass_range, friction_range, applied)
                    changes.append(f"Physics: {n_mass} object(s) mass-scaled, {n_fric} material(s) friction-scaled")
                if randomize_positions:
                    n = self._randomize_positions(rng, float(position_noise), applied)
                    changes.append(f"Positions: {n} object(s) offset (xy, +/-{float(position_noise)} m)")
            except (RuntimeError, ValueError, OSError, AttributeError, TypeError, ImportError) as e:
                # An axis that failed mid-write leaves earlier axes applied;
                # say so rather than reporting a clean failure (the same
                # partial-progress honesty step's mid-run abort uses).
                done = "; ".join(changes) if changes else "nothing"
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"randomize failed ({type(e).__name__}: {e}). Applied before the "
                                f"failure: {done}. The scene may be partially randomized; "
                                f"re-run randomize (same seed re-samples identically from the "
                                f"recorded base values) or rebuild the scene."
                            )
                        }
                    ],
                }
        if not changes:
            changes.append("No axes enabled; nothing randomized.")
        return {
            "status": "success",
            "content": [
                {"text": "Domain randomization applied:\n" + "\n".join(changes)},
                {"json": applied},
            ],
        }

    # --- axis implementations (called with self._lock held) ------------------

    def _dr_base_for(self, kind: str, key: str, capture: Any) -> Any:
        """The anti-compounding anchor: first-touch snapshot per entity.

        ``capture`` is called exactly once per (kind, key) over the engine's
        lifetime; every later randomize samples from that recorded value, so
        repeated calls draw independent offsets inside the bound instead of
        compounding into a walk (the property the MuJoCo axes anchor on the
        authored model; this backend has no compiled model, and before the
        first write the live value IS the authored one).
        """
        # ``is None`` and NEVER ``or {}``: ``__init__`` pre-creates this dict
        # EMPTY, and an empty dict is falsy - the ``or {}`` spelling built a
        # fresh base on every call, so each randomize captured the value the
        # previous one had already scaled. Measured on the live backend before
        # this line was fixed: mass_range=(2,2) applied twice took a 0.105 kg
        # base to 0.421 kg (x4) instead of 0.211 (x2), and one seed stopped
        # reproducing one sample set because the recorded bases drifted. The
        # unit skeletons could not see it - built with ``__new__``, they have
        # no ``_dr_base`` at all, so the buggy spelling happened to store the
        # dict for them and only for them.
        base: dict[tuple[str, str], Any] | None = getattr(self, "_dr_base", None)
        if base is None:
            base = {}
            self._dr_base = base
        slot = (kind, key)
        if slot not in base:
            base[slot] = capture()
        return base[slot]

    def _object_prims(self) -> list[tuple[str, Any]]:
        import omni.usd  # type: ignore[import-not-found]

        stage = omni.usd.get_context().get_stage()
        out = []
        for name, st in getattr(self, "_objects", {}).items():
            prim = stage.GetPrimAtPath(st.prim_path)
            if prim and prim.IsValid():
                out.append((name, prim))
        return out

    def _randomize_colors(self, rng: Any, color_range: tuple[float, float], applied: dict[str, Any]) -> int:
        from pxr import Gf, UsdGeom  # type: ignore[import-not-found]

        lo, hi = float(color_range[0]), float(color_range[1])
        colors: dict[str, list[float]] = {}
        for name, prim in self._object_prims():
            rgb = [float(v) for v in rng.uniform(lo, hi, size=3)]
            # The displayColor primvar is what the RTX renderer reads for a
            # prim with no bound visual material; write the prim AND its geom
            # descendants, because add_object's primitives author the color on
            # the mesh child.
            for target in [prim, *prim.GetChildren()]:
                gprim = UsdGeom.Gprim(target)
                if gprim:
                    gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(*rgb)])
            colors[name] = rgb
        applied["colors"] = colors
        return len(colors)

    def _randomize_lighting(self, rng: Any, color_range: tuple[float, float], applied: dict[str, Any]) -> int:
        import omni.usd  # type: ignore[import-not-found]
        from pxr import Gf, UsdLux  # type: ignore[import-not-found]

        stage = omni.usd.get_context().get_stage()
        lo, hi = float(color_range[0]), float(color_range[1])
        lights: dict[str, dict[str, Any]] = {}
        for prim in stage.Traverse():
            if not (prim.HasAPI(UsdLux.LightAPI) or prim.GetTypeName().endswith("Light")):
                continue
            path = prim.GetPath().pathString
            light = UsdLux.LightAPI(prim)
            base_intensity = self._dr_base_for(
                "light_intensity", path, lambda light=light: float(light.GetIntensityAttr().Get() or 0.0)
            )
            scale = float(rng.uniform(*_LIGHT_INTENSITY_SCALE))
            rgb = [float(v) for v in rng.uniform(lo, hi, size=3)]
            light.GetIntensityAttr().Set(base_intensity * scale)
            light.GetColorAttr().Set(Gf.Vec3f(*rgb))
            lights[path] = {"intensity_scale": scale, "color": rgb}
        applied["lights"] = lights
        return len(lights)

    def _randomize_physics(
        self, rng: Any, mass_range: tuple[float, float], friction_range: tuple[float, float], applied: dict[str, Any]
    ) -> tuple[int, int]:
        masses: dict[str, dict[str, float]] = {}
        materials: dict[str, Any] = {}
        for name, st in getattr(self, "_objects", {}).items():
            handle = st.handle
            if st.is_static or handle is None or not hasattr(handle, "set_mass"):
                continue
            base_mass = self._dr_base_for("mass", name, lambda h=handle: float(h.get_mass()))
            scale = float(rng.uniform(*mass_range))
            handle.set_mass(base_mass * scale)
            masses[name] = {"base": base_mass, "scale": scale}
            material = (
                handle.get_applied_physics_material() if hasattr(handle, "get_applied_physics_material") else None
            )
            if material is not None and hasattr(material, "prim_path"):
                materials.setdefault(str(material.prim_path), material)
        frictions: dict[str, dict[str, float]] = {}
        for path, material in materials.items():
            base = self._dr_base_for(
                "friction",
                path,
                lambda m=material: (float(m.get_static_friction()), float(m.get_dynamic_friction())),
            )
            scale = float(rng.uniform(*friction_range))
            material.set_static_friction(base[0] * scale)
            material.set_dynamic_friction(base[1] * scale)
            frictions[path] = {"base_static": base[0], "base_dynamic": base[1], "scale": scale}
        applied["masses"] = masses
        applied["frictions"] = frictions
        return len(masses), len(frictions)

    def _randomize_positions(self, rng: Any, position_noise: float, applied: dict[str, Any]) -> int:
        moved: dict[str, list[float]] = {}
        for name, st in getattr(self, "_objects", {}).items():
            handle = st.handle
            if st.is_static or handle is None or not hasattr(handle, "set_world_pose"):
                continue
            base = self._dr_base_for(
                "position",
                name,
                lambda h=handle: [float(v) for v in np.asarray(h.get_world_pose()[0]).reshape(-1)],
            )
            offset = rng.uniform(-position_noise, position_noise, size=2)
            pos = [base[0] + float(offset[0]), base[1] + float(offset[1]), base[2]]
            handle.set_world_pose(position=np.asarray(pos, dtype=float))
            moved[name] = pos
        applied["positions"] = moved
        return len(moved)

    # --- set_obs_noise --------------------------------------------------------

    def set_obs_noise(
        self,
        joint_pos_std: float = 0.0,
        joint_vel_std: float = 0.0,
        camera_jitter_px: float = 0.0,
        seed: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Configure additive Gaussian sensor noise on observations.

        Signature, defaults and semantics mirror the MuJoCo and Newton
        backends: ``joint_pos_std`` (radians) on the position floats,
        ``joint_vel_std`` (rad/s) on the ``.vel`` floats, ``camera_jitter_px``
        as a random integer-pixel roll on rendered frames. All zeros clears the
        noise. Applied by :meth:`IsaacSimulation.get_observation` on every
        read; ``get_frame`` / ``render`` stay noise-free, matching MuJoCo.
        """
        if kwargs_error := unknown_kwargs_error("set_obs_noise", kwargs, _OBS_NOISE_PARAMS):
            return kwargs_error
        for param, value in (
            ("joint_pos_std", joint_pos_std),
            ("joint_vel_std", joint_vel_std),
            ("camera_jitter_px", camera_jitter_px),
        ):
            if msg := finite_non_negative_error(value, param, "set_obs_noise"):
                return {"status": "error", "content": [{"text": msg}]}
        if msg := randomization_seed_error(seed, "set_obs_noise"):
            return {"status": "error", "content": [{"text": msg}]}
        cfg = {
            "joint_pos_std": float(joint_pos_std),
            "joint_vel_std": float(joint_vel_std),
            "camera_jitter_px": float(camera_jitter_px),
        }
        with self._lock:
            if any(v > 0 for v in cfg.values()):
                self._obs_noise = cfg
                self._obs_noise_rng = np.random.default_rng(seed)
                text = (
                    f"Sensor noise: joint_pos_std={cfg['joint_pos_std']}, "
                    f"joint_vel_std={cfg['joint_vel_std']}, camera_jitter_px={cfg['camera_jitter_px']}"
                )
            else:
                self._obs_noise = None
                self._obs_noise_rng = None
                text = "Sensor noise cleared."
        return {"status": "success", "content": [{"text": text}, {"json": {**cfg, "seed": seed}}]}

    def _apply_obs_noise(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Return ``obs`` with the configured sensor noise applied.

        Suffix-keyed exactly as the MuJoCo pass: position noise to the plain
        float entries, velocity noise to the ``.vel`` floats, an integer-pixel
        roll to ndarray frames. List values (the ``base_*`` floating-base
        signals) are left untouched - a quaternion would need renormalization,
        out of scope for additive scalar noise. A no-op returning the input
        when no noise is configured.
        """
        cfg = getattr(self, "_obs_noise", None)
        rng = getattr(self, "_obs_noise_rng", None)
        if not cfg or rng is None or not obs:
            return obs
        pos_std = cfg.get("joint_pos_std", 0.0)
        vel_std = cfg.get("joint_vel_std", 0.0)
        px = cfg.get("camera_jitter_px", 0.0)
        out: dict[str, Any] = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                if px > 0:
                    dx, dy = (int(v) for v in rng.integers(-int(px), int(px) + 1, size=2))
                    out[key] = np.roll(value, shift=(dy, dx), axis=(0, 1))
                else:
                    out[key] = value
            elif isinstance(value, float):
                std = vel_std if key.endswith(".vel") else pos_std
                out[key] = value + (float(rng.normal(0.0, std)) if std > 0 else 0.0)
            else:
                out[key] = value
        return out
