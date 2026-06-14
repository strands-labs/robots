"""Rendering mixin - render, render_depth, get_contacts, observation helpers."""

import io
import logging
from typing import TYPE_CHECKING, Any

from strands_robots.simulation.mujoco.backend import _can_render, _ensure_mujoco

logger = logging.getLogger(__name__)


class RenderingMixin:
    """Rendering + observation helpers mixed into ``Simulation``.

    Owns ``render``, ``render_depth``, ``render_all``, ``get_contacts``, and
    the low-level ``_apply_sim_action`` (MuJoCo ``ctrl[]`` write + mj_step).

    **Coupling** (see simulation.py top-level docstring): mixin reaches
    into ``self._world``, ``self._renderer_tls``, ``self._renderer_model``,
    ``self.default_width`` / ``self.default_height``, ``self._lock`` and
    ``self._viewer_handle``. ``TYPE_CHECKING`` stubs below exist so mypy
    accepts those lookups; they are a documentary contract, not an
    enforceable protocol.

    Thread-safety note: MuJoCo ``Renderer`` uses thread-local GL contexts
    (CGL on macOS, GLX on Linux). A renderer created on thread A cannot be
    reused from thread B - we keep one per-thread via ``_renderer_tls``.
    """

    if TYPE_CHECKING:
        from strands_robots.simulation.models import SimWorld

        _world: "SimWorld | None"

        _renderer_model: Any
        _renderer_tls: Any  # threading.local() - per-thread renderer dict
        default_width: int
        default_height: int
        _lock: Any  # threading.RLock from Simulation

    def _validate_render_dims(self, width: int, height: int) -> dict[str, Any] | None:
        """reject non-positive render dims; convert MuJoCo's framebuffer
        overflow to a plain-English message that tells the LLM the actual cap.
        """
        if not isinstance(width, int) or not isinstance(height, int):
            return {
                "status": "error",
                "content": [
                    {"text": f"render: width/height must be int, got {type(width).__name__}/{type(height).__name__}."}
                ],
            }
        if width <= 0 or height <= 0:
            return {
                "status": "error",
                "content": [{"text": f"render: width and height must be > 0, got {width}x{height}."}],
            }
        # Hard absolute ceiling regardless of model config (OOM protection).
        _ABS_MAX = 4096
        if width > _ABS_MAX or height > _ABS_MAX:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"render: {width}x{height} exceeds absolute maximum offscreen framebuffer cap ({_ABS_MAX}x{_ABS_MAX}). Lower width/height or set offwidth/offheight in the model."
                    }
                ],
            }
        if self._world is not None and self._world._model is not None:
            max_w = int(getattr(self._world._model.vis.global_, "offwidth", 1280))
            max_h = int(getattr(self._world._model.vis.global_, "offheight", 960))
            if width > max_w or height > max_h:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"render: requested {width}x{height} exceeds the offscreen "
                                f"framebuffer cap ({max_w}x{max_h}). Lower width/height or "
                                f"rebuild the model with a larger <global offwidth='...' offheight='...'/>."
                            )
                        }
                    ],
                }
        return None

    def _get_renderer(self, width: int, height: int):
        """Get a cached MuJoCo renderer, creating one only if needed.

        Returns None if rendering is unavailable (headless without EGL/OSMesa).
        Callers must handle None return.

        Thread-safety: renderers are cached per-thread via ``threading.local``
        because ``mujoco.Renderer`` binds a GL context to the thread that
        creates it (CGL on macOS, GLX on Linux). Sharing renderers across
        threads would cause ``cgl.free()`` segfaults at cleanup time.
        """
        if not _can_render():
            return None
        mj = _ensure_mujoco()
        assert self._world is not None  # callers must check

        # Get or create per-thread renderer dict
        renderers = getattr(self._renderer_tls, "renderers", None)
        if renderers is None:
            renderers = {}
            self._renderer_tls.renderers = renderers
            self._renderer_tls.model = None

        # Invalidate this thread's cache if model changed (e.g. after recompile)
        if self._renderer_tls.model is not self._world._model:
            renderers.clear()
            self._renderer_tls.model = self._world._model
            # Keep the per-instance marker for compatibility with any remaining
            # read paths that checked self._renderer_model.
            self._renderer_model = self._world._model

        key = (width, height)
        if key not in renderers:
            # Bound the cache: max 4 resolutions per thread. Evict oldest
            # (first-inserted) to prevent unbounded GL context accumulation.
            _MAX_RENDERERS_PER_THREAD = 4
            if len(renderers) >= _MAX_RENDERERS_PER_THREAD:
                oldest_key = next(iter(renderers))
                try:
                    renderers[oldest_key].close()
                except Exception:
                    pass
                del renderers[oldest_key]
            renderers[key] = mj.Renderer(self._world._model, height=height, width=width)
        return renderers[key]

    def _get_viz_option(self) -> Any:
        """Return an ``mujoco.MjvOption`` from ``world._backend_state["viz_option"]``, or ``None``.

        The optional ``viz_option`` override lets benchmark adapters (e.g.
        :class:`~strands_robots.benchmarks.libero.adapter.LiberoAdapter`)
        configure render-time visualisation flags - things like
        ``mjvOption.geomgroup[0] = 0`` to hide collision geoms,
        ``sitegroup[*] = 0`` to hide site markers, ``mjVIS_JOINT/mjVIS_ACTUATOR/mjVIS_COM = 0``
        to hide joint/actuator/COM debug widgets - without changing the
        loaded MJCF or affecting non-LIBERO callers. RoboSuite /
        ``OffScreenRenderEnv`` set these in their viewer; when adapters
        running through ``MuJoCoSimulation`` need parity, they populate
        ``_backend_state["viz_option"]`` and the render path here threads
        the option through to ``Renderer.update_scene(..., scene_option=...)``.

        Returns ``None`` (the default) when no adapter has set the
        override. ``Renderer.update_scene`` accepts ``scene_option=None``
        as the no-op meaning, so non-LIBERO callers see zero behaviour
        change.

        Storing the option on ``world._backend_state`` (per the convention
        documented at :class:`~strands_robots.simulation.models.SimWorld`)
        ties its lifecycle to the loaded scene: a subsequent
        :meth:`Simulation.load_scene` replaces ``self._world`` and the
        option goes with it. Matches the lifecycle of the other state
        keys in ``_backend_state`` (``spec``, ``xml``, ``scene_loaded``,
        etc.).
        """
        if self._world is None:
            return None
        state = getattr(self._world, "_backend_state", None)
        if not isinstance(state, dict):
            return None
        return state.get("viz_option")

    def _get_sim_observation(self, robot_name: str, *, skip_images: bool = False) -> dict[str, Any]:
        """Get observation from sim: joint state + cameras (unless skipped).

        Implements :meth:`SimEngine.get_observation`'s schema.

        Multi-robot note: when the injected robot XML was namespaced
        (e.g. ``arm0/shoulder_pan`` in MuJoCo to allow multiple same-config
        robots), we look up the prefixed MuJoCo name but return the short
        name in the observation dict so the policy sees a stable, config-level
        schema regardless of how many robots are in the scene.
        """
        mj = _ensure_mujoco()
        assert self._world is not None  # callers must check
        model, data = self._world._model, self._world._data
        robot = self._world.robots[robot_name]
        pfx = robot.namespace or ""

        obs = {}
        for jnt_name in robot.joint_names:
            # Try namespaced name first (multi-robot), fall back to raw.
            lookup = pfx + jnt_name if pfx else jnt_name
            jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, lookup)
            if jnt_id < 0 and pfx:
                jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
            if jnt_id >= 0:
                obs[jnt_name] = float(data.qpos[model.jnt_qposadr[jnt_id]])

        if skip_images:
            return obs

        # Render every camera defined on the model plus any python-side cameras.
        # Individual camera failures are logged but do not drop joint state.
        cameras_to_render = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
        for pycam_name in self._world.cameras:
            if pycam_name not in cameras_to_render:
                cameras_to_render.append(pycam_name)

        for cname in cameras_to_render:
            if not cname:
                continue
            cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, cname)
            cam_info = self._world.cameras.get(cname)
            h = cam_info.height if cam_info else self.default_height
            w = cam_info.width if cam_info else self.default_width
            try:
                renderer = self._get_renderer(w, h)
                if renderer is None:
                    continue
                viz_option = self._get_viz_option()
                if cam_id >= 0:
                    renderer.update_scene(data, camera=cam_id, scene_option=viz_option)
                else:
                    renderer.update_scene(data, scene_option=viz_option)
                obs[cname] = renderer.render().copy()
            except (RuntimeError, ValueError) as e:
                # Individual camera failure shouldn't stop joint state collection.
                # Common cause: camera ID invalid after scene recompile.
                logger.debug("Camera render failed for %s: %s", cname, e)

        return obs

    def _apply_sim_action(self, robot_name: str, action_dict: dict[str, Any], n_substeps: int = 1) -> None:
        """Apply action dict to sim (same interface as robot.send_action).

        Multi-robot note: action keys are *short* names (e.g. ``shoulder_pan``).
        We look up the namespaced MuJoCo actuator/joint name for this
        specific ``robot_name`` so the same action dict routes to the right
        physical actuator when multiple same-config robots exist.

        Action-controller hook (#168): when a benchmark adapter
        has installed a custom action controller via
        ``world._backend_state["action_controller"]`` (mirroring the
        ``viz_option`` pattern from #168), dispatch to it
        instead of the actuator/joint-name lookup loop. Used by
        :class:`LiberoAdapter` to convert GR00T's task-space delta-EEF
        actions (7-dim ``{x, y, z, roll, pitch, yaw, gripper}``) into
        the LIBERO scene's torque-mode joint actuators (9-dim
        ``robot0_torq_j1..7`` + gripper) via RoboSuite's
        ``OperationalSpaceController`` (OSC_POSE). Without this hook,
        ``_apply_sim_action`` would silently drop every key (no name
        match), the policy would effectively send 0 torque, and any
        observed motion would be gravity / drift only.

        Default (no controller installed) preserves the existing
        actuator/joint-name lookup path verbatim. Non-LIBERO callers
        and existing tests see zero behaviour change.

        Owns-stepping flag (#168): controllers may declare
        ``owns_stepping = True`` on the controller object to signal
        that ``apply()`` itself advances physics by the correct number
        of substeps for the policy step (LIBERO: 25 mj_step calls per
        ``apply()`` so OSC torques recompute every physics step at
        500 Hz while policy commands arrive at 20 Hz). When the flag
        is true the outer ``mj_step`` loop here is skipped to avoid
        double-stepping. The default (flag absent / False) preserves
        the original 1-substep-per-apply contract.
        """
        mj = _ensure_mujoco()
        assert self._world is not None  # callers must check
        model, data = self._world._model, self._world._data
        robot = self._world.robots.get(robot_name)
        pfx = robot.namespace if robot else ""

        # Action-controller fast path: adapter-installed transform
        # from action_dict (e.g. task-space deltas) to data.ctrl
        # writes (joint torques). When set, the controller takes
        # full responsibility for the data.ctrl update; the
        # actuator/joint-name lookup loop is skipped.
        controller = self._get_action_controller()
        controller_handled_stepping = False
        if controller is not None:
            try:
                controller.apply(action_dict, model, data, robot_name)
                # #168: some controllers (e.g. LIBERO's
                # OSC_POSE wrapper) need to advance physics themselves
                # at a controller-defined rate (e.g. 25 substeps per
                # policy step at 20 Hz LIBERO control / 500 Hz physics).
                # When the controller declares ``owns_stepping = True``,
                # skip the outer ``mj_step`` loop below - the controller
                # has already advanced ``data.time`` by the full control
                # timestep. Without this, we'd double-step (the outer
                # loop would run an extra mj_step on top of the
                # controller's substeps), corrupting trajectories.
                controller_handled_stepping = bool(getattr(controller, "owns_stepping", False))
            except Exception as e:  # noqa: BLE001 - never abort eval on a controller failure
                logger.warning(
                    "_apply_sim_action: action_controller.apply raised %s; falling through to "
                    "name-lookup path (action may be dropped)",
                    e,
                )
                self._unresolved_action_keys = self._apply_action_by_name(model, data, action_dict, pfx, mj)
        else:
            self._unresolved_action_keys = self._apply_action_by_name(model, data, action_dict, pfx, mj)

        if not controller_handled_stepping:
            for _ in range(max(1, n_substeps)):
                mj.mj_step(model, data)

        assert self._world is not None
        self._world.sim_time = data.time
        # When the controller advanced physics itself, ``step_count``
        # should reflect the actual number of mj_step calls (typically
        # 25 for LIBERO @ 20 Hz / 500 Hz), not the policy-step count.
        if controller_handled_stepping:
            self._world.step_count = int(getattr(self._world, "step_count", 0)) + int(
                getattr(controller, "physics_substeps_per_control", n_substeps)
            )
        else:
            self._world.step_count += n_substeps

        if hasattr(self, "_viewer_handle") and self._viewer_handle is not None:
            self._viewer_handle.sync()

    def _get_action_controller(self) -> Any:
        """Return an installed action-controller or ``None``.

        Mirrors :meth:`_get_viz_option`. The controller (if present)
        is set by a benchmark adapter via
        ``world._backend_state["action_controller"]`` and is expected
        to expose an ``apply(action_dict, model, data, robot_name)``
        method that writes to ``data.ctrl``. See
        :meth:`LiberoAdapter._install_action_controller` for the
        canonical use case.

        Returns ``None`` (the default) when no adapter has set the
        override. The actuator/joint-name lookup loop in
        :meth:`_apply_sim_action` is the fallback in that case.
        """
        if self._world is None:
            return None
        state = getattr(self._world, "_backend_state", None)
        if not isinstance(state, dict):
            return None
        return state.get("action_controller")

    def _apply_action_by_name(
        self,
        model: Any,
        data: Any,
        action_dict: dict[str, Any],
        pfx: str,
        mj: Any,
    ) -> list[str]:
        """Default action-application: look up actuator / joint by name.

        Extracted from :meth:`_apply_sim_action` so the
        ``action_controller`` fast path can fall back to it on
        controller failure (the same path non-LIBERO callers use).

        Returns:
            List of action keys that could not be resolved to any
            actuator or joint (empty list when all keys applied).
        """

        def _lookup(obj_type: Any, name: str) -> int:
            """Try namespaced lookup first, fall back to raw."""
            if pfx:
                i = mj.mj_name2id(model, obj_type, pfx + name)
                if i >= 0:
                    return i
            return int(mj.mj_name2id(model, obj_type, name))

        unresolved: list[str] = []
        for key, value in action_dict.items():
            act_id = _lookup(mj.mjtObj.mjOBJ_ACTUATOR, key)
            if act_id >= 0:
                data.ctrl[act_id] = float(value)
                continue

            # Fallback: key is a joint name. Find the actuator that drives
            # this joint, handling BOTH transmission types:
            #   * JOINT / JOINTINPARENT - actuator_trnid[ai, 0] == jnt_id
            #   * TENDON               - the joint participates in a tendon
            #     (via wrap entries) whose tendon id == actuator_trnid[ai, 0]
            # Tendon grippers (e.g. the Franka/Panda ``split`` actuator that
            # drives finger_joint1/2) were silently dropped before this branch
            # because their actuator_trnid points at the *tendon*, not the
            # finger joint - see issue #318.
            jnt_id = _lookup(mj.mjtObj.mjOBJ_JOINT, key)
            if jnt_id < 0:
                # #367: an action key that resolves to neither an actuator nor
                # a joint is silently dropped today. Silent gripper drops are
                # exactly the failure mode #318 was filed to fix, so surface it
                # -- once per (prefix, key) to avoid per-step log spam at 50Hz.
                self._warn_unresolved_action_key(pfx, key, "no actuator or joint")
                unresolved.append(key)
                continue

            ai = self._actuator_for_joint(model, jnt_id, mj)
            if ai < 0:
                self._warn_unresolved_action_key(pfx, key, "joint has no driving actuator")
                unresolved.append(key)
                continue

            # Scale a logical command into the actuator's ctrlrange when the
            # transmission is a tendon (gripper ctrlrange is e.g. [0, 255]
            # tendon units, not a finger-joint position). Direct JOINT
            # actuators keep the raw value (positions/torques in joint units).
            data.ctrl[ai] = self._scale_ctrl_for_actuator(model, ai, float(value), mj)

        return unresolved

    def _warn_unresolved_action_key(self, pfx: str, key: str, reason: str) -> None:
        """Warn once per (prefix, key) that an action key could not be applied.

        #367: replaces the prior silent ``continue`` on unresolved action keys.
        De-duplicated via a per-world set so a 50Hz control loop does not spam
        the log -- the operator sees the missing key once and can act on it.

        Includes the actual actuator/joint names from the model so the user
        knows exactly which keys the scene accepts.
        """
        warned = getattr(self, "_warned_unresolved_keys", None)
        if warned is None:
            warned = set()
            self._warned_unresolved_keys = warned
        dedup = (pfx, key)
        if dedup in warned:
            return
        warned.add(dedup)
        # Surface the valid actuator/joint names from the loaded model so
        # users can self-correct without inspecting the MJCF by hand.
        valid_names = self._get_valid_action_keys(pfx)
        hint = f" Valid keys for this robot: {valid_names}" if valid_names else ""
        logger.warning(
            "[sim] action key %r (prefix=%r) could not be applied: %s. The value was dropped.%s",
            key,
            pfx,
            reason,
            hint,
        )

    def _get_valid_action_keys(self, pfx: str) -> list[str]:
        """Return actuator names available under the given namespace prefix.

        When ``pfx`` is set (multi-robot), strips the prefix from returned
        names so the caller sees the short form that ``send_action`` expects.
        """
        world = getattr(self, "_world", None)
        if world is None or getattr(world, "_model", None) is None:
            return []
        mj = _ensure_mujoco()
        model = world._model
        names: list[str] = []
        for i in range(model.nu):
            raw = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            if not raw:
                continue
            if pfx and raw.startswith(pfx):
                names.append(raw[len(pfx) :])
            elif not pfx:
                names.append(raw)
        return names

    @staticmethod
    def _actuator_for_joint(model: Any, jnt_id: int, mj: Any) -> int:
        """Return the id of the actuator that drives ``jnt_id``, or -1.

        Matches direct joint-transmission actuators first, then falls back to
        tendon-transmission actuators whose tendon wraps ``jnt_id`` (the
        Panda/Franka gripper case from issue #318).
        """
        # 1. Direct joint transmission (JOINT / JOINTINPARENT).
        joint_trn = {int(mj.mjtTrn.mjTRN_JOINT)}
        if hasattr(mj.mjtTrn, "mjTRN_JOINTINPARENT"):
            joint_trn.add(int(mj.mjtTrn.mjTRN_JOINTINPARENT))
        for ai in range(model.nu):
            if int(model.actuator_trntype[ai]) in joint_trn and model.actuator_trnid[ai, 0] == jnt_id:
                return ai

        # 2. Tendon transmission: find tendons whose JOINT wrap entries
        #    include jnt_id, then the actuator driving that tendon.
        tendon_trn = int(mj.mjtTrn.mjTRN_TENDON)
        wrap_joint = int(mj.mjtWrap.mjWRAP_JOINT)
        tendons_with_joint: set[int] = set()
        for t in range(int(model.ntendon)):
            adr = int(model.tendon_adr[t])
            num = int(model.tendon_num[t])
            for w in range(adr, adr + num):
                if int(model.wrap_type[w]) == wrap_joint and int(model.wrap_objid[w]) == jnt_id:
                    tendons_with_joint.add(t)
                    break
        if tendons_with_joint:
            for ai in range(model.nu):
                if (
                    int(model.actuator_trntype[ai]) == tendon_trn
                    and int(model.actuator_trnid[ai, 0]) in tendons_with_joint
                ):
                    return ai
        return -1

    @staticmethod
    def _scale_ctrl_for_actuator(model: Any, ai: int, value: float, mj: Any) -> float:
        """Scale ``value`` into the actuator's ctrlrange for tendon drives.

        Tendon-gripper actuators expose a ctrlrange in tendon units (e.g.
        ``[0, 255]``) that does not match a finger-joint position. When the
        caller passes a small logical value (a normalised ``[0, 1]`` open/close
        fraction, or a finger position within the joint range), map it onto the
        actuator ctrlrange so the gripper actually moves. A value already
        inside the ctrlrange is passed through unchanged.

        Direct JOINT actuators return ``value`` untouched (positions/torques
        are already in the correct units).
        """
        if int(model.actuator_trntype[ai]) != int(mj.mjtTrn.mjTRN_TENDON):
            return value
        lo, hi = float(model.actuator_ctrlrange[ai, 0]), float(model.actuator_ctrlrange[ai, 1])
        if not bool(model.actuator_ctrllimited[ai]) or hi <= lo:
            return value
        span = hi - lo
        # #367 item 1a: a ctrlrange that spans zero (e.g. [-1, 1]) is itself the
        # normalised command space -- the caller passes the command verbatim and
        # we must NOT re-map it onto [lo, hi] (which would clip a symmetric
        # -0.5 to 0.0 -> -1.0). Treat lo < 0 as "already normalised, pass
        # through clamped to the range".
        if lo < 0.0:
            return min(hi, max(lo, value))
        # A normalised [0, 1] open/close fraction is the conventional gripper
        # command from VLA policies. When the actuator ctrlrange is much wider
        # than unit scale (e.g. the Panda tendon's [0, 255]), a value within
        # [lo, lo + 1] is overwhelmingly likely to be such a fraction rather
        # than a literal tendon-unit command, so we map it onto the full range.
        # If the caller already passes a clearly in-range value (> lo + 1 and
        # <= hi), we trust it verbatim.
        #
        # #367 item 1b: use a small epsilon on the boundary so a normalised
        # 1.0 + FP-noise (from a quantised VLA head) is still treated as the
        # fraction 1.0 (-> hi) rather than slipping into the verbatim branch and
        # writing ~1.0 onto a [0, 255] range (a nearly-closed gripper).
        if span > 1.0 and value > (lo + 1.0 + 1e-6) and value <= hi:
            return value
        # Treat the incoming value as a normalised [0, 1] open/close fraction.
        frac = min(1.0, max(0.0, value))
        return lo + frac * span

    def render(
        self, camera_name: str = "default", width: int | None = None, height: int | None = None
    ) -> dict[str, Any]:
        """Render a camera view as base64 PNG image."""
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world (or load_scene) first."}]}

        mj = _ensure_mujoco()
        # treat `None` as "use default", but `0` / negative values must
        # still hit the validator (bool coercion would swallow them silently).
        w = self.default_width if width is None else width
        h = self.default_height if height is None else height
        if err := self._validate_render_dims(w, h):
            return err

        try:
            renderer = self._get_renderer(w, h)
            if renderer is None:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                " Rendering unavailable (no OpenGL context). "
                                "Install EGL or OSMesa for offscreen rendering: "
                                "apt-get install libosmesa6-dev"
                            )
                        }
                    ],
                }
            # strict camera validation - no silent fallback to default.
            # Special 'default' / 'free' tokens route to the free camera; any
            # other name MUST resolve or we error (prevents the LLM from
            # believing it rendered viewpoint X while actually getting free-cam).
            if camera_name in (None, "", "default", "free"):
                cam_id = -1
                label = "free (default)"
            else:
                cam_id = mj.mj_name2id(self._world._model, mj.mjtObj.mjOBJ_CAMERA, camera_name)
                if cam_id < 0:
                    return {
                        "status": "error",
                        "content": [
                            {"text": f"Camera '{camera_name}' not found. Available: {self._list_camera_names()}"}
                        ],
                    }
                label = camera_name

            if cam_id >= 0:
                renderer.update_scene(self._world._data, camera=cam_id, scene_option=self._get_viz_option())
            else:
                renderer.update_scene(self._world._data, scene_option=self._get_viz_option())

            img = renderer.render().copy()

            from PIL import Image

            pil_img = Image.fromarray(img)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            png_bytes = buffer.getvalue()

            # base64-encode for transport through the Strands tool-response
            # serializer (which expects strings, not raw bytes, for image
            # content in the Bedrock/Anthropic multimodal format).
            import base64 as _b64

            png_b64 = _b64.b64encode(png_bytes).decode("ascii")

            # summary stats so render_all can flag empty-looking frames
            # without decoding the PNG a second time.
            import numpy as _np

            pixel_var = float(_np.var(img))
            pixel_mean = float(_np.mean(img))

            return {
                "status": "success",
                "content": [
                    {"text": f"📸 {w}x{h} from '{label}' at t={self._world.sim_time:.3f}s"},
                    {"image": {"format": "png", "source": {"bytes": png_b64}}},
                    {"json": {"pixel_variance": pixel_var, "pixel_mean": pixel_mean, "camera": label}},
                ],
            }
        except Exception as e:
            return {"status": "error", "content": [{"text": f"Render failed: {e}"}]}

    def render_depth(
        self, camera_name: str = "default", width: int | None = None, height: int | None = None
    ) -> dict[str, Any]:
        """Render depth map from a camera."""
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world (or load_scene) first."}]}

        mj = _ensure_mujoco()
        # see note in render() re: None vs 0/negative.
        w = self.default_width if width is None else width
        h = self.default_height if height is None else height
        if err := self._validate_render_dims(w, h):
            return err

        try:
            # strict camera validation (same policy as render())
            if camera_name in (None, "", "default", "free"):
                cam_id = -1
                label = "free (default)"
            else:
                cam_id = mj.mj_name2id(self._world._model, mj.mjtObj.mjOBJ_CAMERA, camera_name)
                if cam_id < 0:
                    return {
                        "status": "error",
                        "content": [
                            {"text": f"Camera '{camera_name}' not found. Available: {self._list_camera_names()}"}
                        ],
                    }
                label = camera_name

            renderer = self._get_renderer(w, h)
            if renderer is None:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                " Depth rendering unavailable (no OpenGL context). "
                                "Install EGL or OSMesa for offscreen rendering."
                            )
                        }
                    ],
                }
            if cam_id >= 0:
                renderer.update_scene(self._world._data, camera=cam_id, scene_option=self._get_viz_option())
            else:
                renderer.update_scene(self._world._data, scene_option=self._get_viz_option())
            # MuJoCo prints a one-time ARB_clip_control warning on macOS
            # when depth precision is reduced. Capture stderr on the first
            # depth render so we can surface the warning in the response
            # text (the LLM otherwise never hears about it).
            clip_warn = getattr(self, "_depth_warn_text", None)
            if clip_warn is None:
                import contextlib as _ctx
                import io as _io
                import sys as _sys

                buf = _io.StringIO()
                with _ctx.redirect_stderr(buf):
                    renderer.enable_depth_rendering()
                    depth = renderer.render()
                    renderer.disable_depth_rendering()
                captured = buf.getvalue()
                # Also forward to the real stderr so logs don't vanish.
                if captured and _sys.__stderr__ is not None:
                    try:
                        _sys.__stderr__.write(captured)
                    except Exception:
                        pass
                if "ARB_clip_control" in captured:
                    # ARB_clip_control missing → OpenGL depth buffer uses
                    # default [0,1] range with compressed far-plane precision.
                    # After linearization below, Min/Max are still in meters,
                    # but their precision (especially for distant pixels) is
                    # degraded vs. a GPU with ARB_clip_control. Downstream
                    # consumers should treat these values as approximate.
                    self._depth_warn_text = (
                        "⚠️ Depth accuracy limited on this GPU (missing ARB_clip_control). "
                        "Linearized Min/Max are in meters but precision is degraded "
                        "(especially for far-plane pixels) - treat as approximate."
                    )
                else:
                    self._depth_warn_text = ""
                clip_warn = self._depth_warn_text
            else:
                renderer.enable_depth_rendering()
                depth = renderer.render()
                renderer.disable_depth_rendering()

            # Linearize OpenGL depth buffer to metric depth (meters).
            # MuJoCo renderer returns normalized values in [0, 1] where 0 = near,
            # 1 = far plane. Convert: z = znear*zfar / (zfar - d*(zfar - znear))
            #
            # On MuJoCo >= 3.0, `model.vis.map.{znear,zfar}` are fractions of
            # `model.stat.extent` (the model's bounding scale), NOT absolute
            # meters - multiply by extent to get real clip-plane distances.
            # pyproject.toml pins mujoco>=3.2, so this convention is safe here.
            import numpy as _np

            extent = float(self._world._model.stat.extent)
            znear = float(self._world._model.vis.map.znear) * extent
            zfar = float(self._world._model.vis.map.zfar) * extent
            # Avoid division by zero for pixels at exactly the far plane
            denom = zfar - depth * (zfar - znear)
            denom = _np.where(denom == 0, 1e-10, denom)
            depth_m = znear * zfar / denom
            # Clamp: pixels at far plane (depth==1) → zfar
            depth_m = _np.clip(depth_m, znear, zfar)

            text = (
                f"📸 Depth {w}x{h} from '{label}'\nMin: {float(depth_m.min()):.4f}m, Max: {float(depth_m.max()):.4f}m"
            )
            if clip_warn:
                text += f"\n{clip_warn}"
            return {
                "status": "success",
                "content": [
                    {"text": text},
                    {"json": {"depth_min": float(depth_m.min()), "depth_max": float(depth_m.max())}},
                ],
            }
        except Exception as e:
            return {"status": "error", "content": [{"text": f"Depth render failed: {e}"}]}

    def _list_camera_names(self) -> list[str]:
        """helper to list all camera names (model-defined + SimCamera aliases)
        for error messages when an unknown camera_name is requested."""
        import mujoco as _mj

        names: list[str] = []
        if self._world is not None and self._world._model is not None:
            for cid in range(self._world._model.ncam):
                raw = _mj.mj_id2name(self._world._model, _mj.mjtObj.mjOBJ_CAMERA, cid)
                if raw:
                    names.append(raw)
        # Include SimCamera registry keys (may match model names; dedupe)
        for k in self._world.cameras.keys() if self._world else ():
            if k not in names:
                names.append(k)
        return names

    def get_contacts(self) -> dict[str, Any]:
        """Return the list of active geom-geom contacts at the current step.

        We run ``mj_forward`` first so the contact list reflects the
        current qpos/qvel even immediately after ``reset`` or ``add_robot``
        (without this, stale contacts from the previous step / uninitialised
        memory can appear as phantom penetrations at t=0).
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world (or load_scene) first."}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data
        # Lock while running mj_forward + snapshotting contacts so a policy
        # thread's mj_step can't mutate data.ncon / data.contact[] between our
        # forward pass and the iteration. We copy the contact records under
        # the lock; name resolution can then run lock-free.
        with self._lock:
            mj.mj_forward(model, data)
            ncon = int(data.ncon)
            contact_snapshot = [
                {
                    "geom1": int(data.contact[i].geom1),
                    "geom2": int(data.contact[i].geom2),
                    "dist": float(data.contact[i].dist),
                    "pos": data.contact[i].pos.tolist(),
                }
                for i in range(ncon)
            ]

        def _resolve_geom(gid: int) -> str:
            """Prefer the geom name; fall back to its parent body name; then id."""
            gn = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gid)
            if gn:
                return gn
            # Walk to the parent body name.
            try:
                bid = int(model.geom_bodyid[gid])
                bn = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid)
                if bn:
                    return f"{bn}/geom_{gid}"
            except (IndexError, AttributeError):
                pass
            return f"geom_{gid}"

        contacts = []
        for c in contact_snapshot:
            g1 = _resolve_geom(c["geom1"])
            g2 = _resolve_geom(c["geom2"])
            contacts.append({"geom1": g1, "geom2": g2, "dist": c["dist"], "pos": c["pos"]})

        text = f"💥 {len(contacts)} contacts" if contacts else "No contacts."
        if contacts:
            for c in contacts[:10]:
                text += f"\n  • {c['geom1']} ↔ {c['geom2']} (d={c['dist']:.4f})"

        return {
            "status": "success",
            "content": [{"text": text}, {"json": {"contacts": contacts}}],
        }

    # Multi-camera capture - Session recording for simulation

    #
    # Design:
    #  - render_all(cameras=None, width=, height=) - single-shot snapshot
    #    of every camera at current sim_time. One PNG per camera.
    #  - start_cameras_recording(...) - daemon thread, one imageio writer
    #    per camera, appends frames at fps.
    #  - stop_cameras_recording() - flushes writers, returns paths + sizes.
    #  - get_cameras_recording_status() - frame counts, elapsed, per-cam.
    #
    # Thread safety: _get_renderer is thread-local (threading.local), so the
    # background thread creates its own GL context. No shared state with
    # main dispatch thread.

    def _active_camera_list(self, cameras):
        """Resolve cameras to concrete camera names currently in the world.

        Handles namespaced camera names (e.g. 'arm0/wrist_cam') by also
        checking the short suffix form ('wrist_cam').

        Returns
        -------
        resolved : list[str]
            Camera names that resolved to real model cameras.
        unresolved_inputs : list[str]
            User-supplied camera names that could NOT be resolved (empty
            list when cameras is None or when every input matched).
        """
        if self._world is None or self._world._model is None:
            return [], []
        mj = _ensure_mujoco()
        model = self._world._model
        from_model = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
        from_model = [c for c in from_model if c]
        py_side = list(self._world.cameras.keys()) if self._world else []
        all_cams = list(dict.fromkeys(from_model + py_side))
        if cameras is None:
            return all_cams, []
        # Try to resolve unknown names via namespace prefix matching.
        resolved: list[str] = []
        unresolved: list[str] = []
        for c in cameras:
            if c in all_cams:
                resolved.append(c)
            else:
                # Try suffix match: 'side' → 'arm0/side'
                matches = [ac for ac in all_cams if ac.endswith("/" + c)]
                if len(matches) == 1:
                    resolved.append(matches[0])
                    logger.debug("Camera '%s' resolved to namespaced '%s'", c, matches[0])
                else:
                    unresolved.append(c)
                    logger.warning(
                        "Camera '%s' not found. Available: %s",
                        c,
                        ", ".join(all_cams) or "(none)",
                    )
        return resolved, unresolved

    def render_all(self, cameras=None, width=None, height=None):
        """Render every (or a subset of) camera in one call.

        Counterpart to ``render()`` for multi-view workflows - e.g. stereo,
        overhead + wrist, or all cameras in a 4-view grid. Each camera ships
        as its own ``{"image": {...}}`` block in the response.

        Args:
            cameras: list of camera names; None = every camera.
            width:   per-camera width (defaults to camera's configured width).
            height:  per-camera height (same).

        Returns:
            ``{"status", "content": [{"text": summary},
                                     {"text": "📸 cam1"}, {"image": {...}},
                                     {"text": "📸 cam2"}, {"image": {...}}, ...]}``
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world (or load_scene) first."}]}
        names, unresolved = self._active_camera_list(cameras)
        if cameras is not None and unresolved:
            return {
                "status": "error",
                "content": [{"text": f"Camera(s) not found: {unresolved}. Available: {self._list_camera_names()}"}],
            }
        if not names:
            return {"status": "error", "content": [{"text": "No cameras in scene."}]}
        content = []
        ok, failed = 0, 0
        low_var_warnings: list[str] = []
        for cam_name in names:
            r = self.render(camera_name=cam_name, width=width, height=height)
            if r.get("status") == "success":
                ok += 1
                img_block = None
                stats = None
                for block in r.get("content", []):
                    if isinstance(block, dict):
                        if "image" in block and img_block is None:
                            img_block = block
                        if "json" in block and stats is None:
                            stats = block["json"]
                if img_block is not None:
                    label = f"📸 {cam_name}"
                    # flag near-uniform frames (all black / all clear).
                    if stats and float(stats.get("pixel_variance", 99)) < 1.0:
                        warn = f"⚠️ camera '{cam_name}': image appears empty (variance < 1)"
                        label = f"{label}  {warn}"
                        low_var_warnings.append(warn)
                    content.append({"text": label})
                    content.append(img_block)
            else:
                failed += 1
                err = r.get("content", [{}])[0].get("text", "?")
                content.append({"text": f"{cam_name}: {err}"})
        warn_suffix = f", {len(low_var_warnings)} low-variance" if low_var_warnings else ""
        summary = (
            f"📸 Multi-camera snapshot at t={self._world.sim_time:.3f}s: "
            f"{ok} ok, {failed} failed, {len(names)} requested{warn_suffix}"
        )
        return {
            "status": "success" if ok else "error",
            "content": [{"text": summary}, *content],
        }

    def start_cameras_recording(
        self,
        cameras=None,
        output_dir=None,
        fps=30,
        width=None,
        height=None,
        name=None,
        max_frames_per_camera=3000,
    ):
        """Start background capture of one ndarray buffer per camera.

        Strategy: the background thread collects raw RGB frames in memory
        (one list per camera). ``stop_cameras_recording`` then flushes each
        list to an MP4 on the main thread. This avoids a long-lived ffmpeg
        subprocess pipe that would break under concurrent imageio writes +
        policy-loop timing jitter.

        Memory cost: H*W*3 bytes * fps * duration * n_cams. For a 2s / 4-cam /
        320x240 / 15fps rollout: ~27 MB. Bounded by ``max_frames_per_camera``.

        Args:
            cameras: list of camera names; None = every camera.
            output_dir: where to write ``{tag}__{cam}.mp4``.
            fps: capture rate.
            width/height: per-frame size.
            name: filename tag (auto if None).
            max_frames_per_camera: safety cap on in-memory buffers.
        """
        import os as _os
        import tempfile as _tempfile
        import threading as _threading
        import time as _time
        import uuid as _uuid

        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world (or load_scene) first."}]}

        if getattr(self, "_cams_rec_state", None) and self._cams_rec_state.get("running"):
            cur = self._cams_rec_state["name"]
            return {
                "status": "error",
                "content": [{"text": f"Already recording '{cur}'. Call stop_cameras_recording() first."}],
            }

        names, unresolved = self._active_camera_list(cameras)
        # Strict validation: if user specified cameras, error on any unresolved names
        # (same policy as render() and render_depth() - fail loudly, don't silently drop).
        # NOTE: `unresolved` contains the raw user inputs that didn't map, so the
        # namespace-suffix resolution path (e.g. 'side' → 'arm0/side') is preserved.
        if cameras is not None and unresolved:
            return {
                "status": "error",
                "content": [{"text": (f"Camera(s) not found: {unresolved}. Available: {self._list_camera_names()}")}],
            }
        if not names:
            return {"status": "error", "content": [{"text": "No cameras to record."}]}

        out_dir = _os.path.abspath(output_dir or _os.path.join(_tempfile.gettempdir(), "strands_robots", "recordings"))
        _os.makedirs(out_dir, exist_ok=True)
        tag = name or f"rec_{_uuid.uuid4().hex[:8]}"

        buffers = {cam: [] for cam in names}
        paths = {cam: _os.path.join(out_dir, f"{tag}__{cam}.mp4") for cam in names}

        # ``ready`` is set by the recorder thread once its GL context is warm
        # and it has entered the capture loop. ``start`` blocks on it below so
        # that "start returned success" guarantees frames are being captured -
        # callers that stop after a short sleep (e.g. tests, brief clips) no
        # longer race the ~0.5s fresh-thread EGL warmup and get an empty buffer.
        state = {
            "running": True,
            "name": tag,
            "cameras": names,
            "fps": fps,
            "width": width,
            "height": height,
            "buffers": buffers,
            "paths": paths,
            "errors": dict.fromkeys(names, 0),
            "output_dir": out_dir,
            "started_at": _time.time(),
            "thread": None,
            "max_frames": max_frames_per_camera,
            "ready": _threading.Event(),
        }

        def _loop():
            from strands_robots.simulation.policy_runner import _extract_frame_ndarray

            # Warm up the recorder thread's GL context BEFORE the
            # timing loop starts capturing into buffers. MuJoCo's
            # ``mujoco.GLContext.make_current()`` is thread-bound:
            # ``mujoco.egl.GLContext`` allocates a fresh EGL context
            # per calling thread. A main-thread ``sim.render()`` call
            # warms only the main thread's context; this daemon
            # thread starts cold. Without warmup, the first ~15
            # render calls per camera return the GL clear-colour
            # gradient before the context settles.
            #
            # History: rounds 11/12/13 added thread-side warmup; round
            # 14 reverted because the load-scene-without-mj_forward
            # bug was bigger. #168 fixed mj_forward in load_scene,
            # which made warmup unnecessary IN THE SLOW PATH. Round
            # 17's prewarm-fresh-ep0 fast-path skips load_scene,
            # leaving no per-recorder-thread render before capture.
            # #168 tried main-thread warmup (thread-isolation
            # made it ineffective). #168 re-applied the
            # 2-pass thread-side warmup. #168 verification showed
            # 2 passes was insufficient: image channel stayed cold for
            # ~15 frames while wrist cleared at frame 3 - per-camera
            # warmup latency varies across cameras (likely GPU
            # command-buffer flush ordering).
            #
            # #168 (this code): replace fixed-pass warmup with an
            # adaptive warmup loop. Render each camera until it
            # produces output with column-stddev above the cold-
            # gradient threshold. The cold gradient artifact is uniform
            # skybox blue->grey with col-std ~0.6; real geometry has
            # col-std > 25 (background plane + objects + textures).
            # Threshold of 5.0 cleanly separates the two regimes
            # without false-positives on legitimately uniform scenes
            # (those would still be > 1.0 from JPEG/encoding noise
            # if they're real renders, not the GL clear-colour).
            #
            # Cap: 30 attempts per camera. At 30 fps that's 1.0 s of
            # wall-time worst-case before the timing loop starts
            # capturing - invisible vs the 250+ s eval wall-time.
            # Common case: ~3-5 attempts per camera, total ~100-200 ms
            # bounded by the slowest-warming camera in the rotation.
            #
            # Errors during warmup are swallowed at DEBUG. Persistent
            # render failures will resurface as
            # ``state["errors"][cam]`` accumulating in the timing
            # loop below (visible via
            # :meth:`get_cameras_recording_status`).
            _max_warmup_attempts = 30
            _cold_std_threshold = 5.0
            _warm: dict[str, bool] = dict.fromkeys(names, False)
            for _attempt in range(_max_warmup_attempts):
                if all(_warm.values()):
                    break
                for cam in names:
                    if _warm[cam]:
                        continue
                    try:
                        r = self.render(camera_name=cam, width=width, height=height)
                        arr = _extract_frame_ndarray(r)
                    except Exception as e:  # noqa: BLE001 - warmup failures non-fatal
                        logger.debug("recorder thread warmup render failed for %s: %s", cam, e)
                        continue
                    if arr is None:
                        continue
                    # arr.std(axis=0) is per-column std-dev; .mean()
                    # collapses to a scalar. Cold gradients have
                    # near-zero values; real geometry > 5.
                    col_std = float(arr.std(axis=0).mean())
                    if col_std > _cold_std_threshold:
                        _warm[cam] = True
                        logger.debug(
                            "recorder thread warmup: %r warmed at attempt %d (col_std=%.2f)",
                            cam,
                            _attempt + 1,
                            col_std,
                        )
            if not all(_warm.values()):
                cold = [c for c, w in _warm.items() if not w]
                logger.warning(
                    "recorder thread warmup: %d cameras still cold after %d attempts: %s. "
                    "First captured frames may show gradient artifact.",
                    len(cold),
                    _max_warmup_attempts,
                    cold,
                )

            # Warmup done (or capped) - capture loop is about to run. Unblock
            # the caller waiting in start_cameras_recording so the success
            # return coincides with the first captured frame, not the cold
            # thread launch.
            state["ready"].set()

            interval = 1.0 / fps
            while state["running"]:
                t0 = _time.time()
                for cam in names:
                    if not state["running"]:
                        break
                    if len(state["buffers"][cam]) >= state["max_frames"]:
                        continue
                    try:
                        r = self.render(camera_name=cam, width=width, height=height)
                        arr = _extract_frame_ndarray(r)
                        if arr is not None:
                            state["buffers"][cam].append(arr)
                        else:
                            state["errors"][cam] += 1
                    except Exception as e:
                        state["errors"][cam] += 1
                        logger.debug("camera recorder (%s) error: %s", cam, e)
                lag = _time.time() - t0
                if lag < interval:
                    _time.sleep(interval - lag)

        state["thread"] = _threading.Thread(target=_loop, daemon=True)
        state["thread"].start()
        self._cams_rec_state = state

        # Wait for the recorder thread to warm its GL context and enter the
        # capture loop before reporting success. Worst case is the 30-attempt
        # warmup cap (~1s/cam at 64x48, more for larger frames) plus a small
        # margin; the common case is ~0.5s. If warmup somehow stalls we still
        # return after the timeout rather than blocking forever - the thread
        # keeps trying and ``get_cameras_recording_status`` exposes errors.
        _ready_timeout = 5.0 + 1.0 * len(names)
        if not state["ready"].wait(timeout=_ready_timeout):
            logger.warning(
                "camera recorder '%s' not ready after %.1fs; returning anyway (first frames may be delayed)",
                tag,
                _ready_timeout,
            )

        msg = (
            f"🎬 Recording {len(names)} camera(s) @ {fps} FPS → {out_dir}\n"
            f"   tag: {tag}\n"
            f"   cameras: {', '.join(names)}"
        )
        return {"status": "success", "content": [{"text": msg}]}

    def stop_cameras_recording(self):
        """Stop capture, flush buffers to MP4 on the MAIN thread.

        Runs ``imageio.get_writer``/``append_data``/``close`` here instead of
        the recording thread so the ffmpeg pipe doesn't race with policy
        timing jitter. Returns per-camera frame counts and paths.

        Idempotent and safe whichever ``start_cameras_recording*`` variant
        was used:

        * Daemon-thread (``start_cameras_recording``) → flips
          ``state["running"] = False``, joins the thread, then flushes.
        * Synchronous (``start_cameras_recording_synchronous``) → no
          thread to join; the ``finalize`` callable returned alongside
          ``on_frame`` is the preferred entry point but
          ``stop_cameras_recording`` works equivalently for callers that
          don't keep the closure handle.
        """
        state = getattr(self, "_cams_rec_state", None)
        if not state or not state.get("running"):
            # idempotent - 'already stopped' is a success, not an error.
            return {"status": "success", "content": [{"text": "Was not recording cameras."}]}

        state["running"] = False
        thread = state.get("thread")
        if thread is not None:
            thread.join(timeout=5.0)

        result = self._flush_cameras_recording_state(state)
        self._cams_rec_state = None
        return result

    def _flush_cameras_recording_state(self, state: dict) -> dict:
        """Encode ``state["buffers"]`` to MP4 + return the standard result dict.

        Shared by :meth:`stop_cameras_recording` (daemon-thread path) and
        the ``finalize`` callable returned by
        :meth:`start_cameras_recording_synchronous`. ``state`` is mutated
        in place - ``running`` should already be ``False`` before this
        runs, and the daemon thread (if any) already joined.

        Best-effort: per-camera flush failures are reported in the result
        dict's text + JSON (``frames`` / ``errors`` / ``size_kb``) but
        never raise, so a partial encode still yields a structured
        success response with the surviving artifacts.
        """
        import os as _os
        import time as _time

        try:
            import imageio.v2 as imageio
        except ImportError:
            return {
                "status": "error",
                "content": [{"text": "imageio not installed. pip install imageio imageio-ffmpeg"}],
            }

        elapsed = _time.time() - state["started_at"]
        lines = [
            f"🎬 Stopped '{state['name']}' after {elapsed:.1f}s",
            f"   output_dir: {state['output_dir']}",
        ]
        artifacts = []
        for cam in state["cameras"]:
            frames_buffer = state["buffers"][cam]
            path = state["paths"][cam]
            errors = state["errors"][cam]
            frames_written = 0
            size_kb = 0.0
            if frames_buffer:
                writer = imageio.get_writer(path, fps=state["fps"], quality=8, macro_block_size=1)
                try:
                    for arr in frames_buffer:
                        writer.append_data(arr)
                        frames_written += 1
                finally:
                    writer.close()
                if _os.path.exists(path):
                    size_kb = _os.path.getsize(path) / 1024
            lines.append(
                f"   📹 {cam:20s} {frames_written:>5d} frames  {size_kb:>7.1f} KB  "
                f"({errors} errors)  → {_os.path.basename(path)}"
            )
            artifacts.append(
                {
                    "camera": cam,
                    "path": path,
                    "frames": frames_written,
                    "errors": errors,
                    "size_kb": size_kb,
                }
            )

        return {
            "status": "success",
            "content": [
                {"text": "\n".join(lines)},
                {"json": {"recording": state["name"], "artifacts": artifacts}},
            ],
        }

    def start_cameras_recording_synchronous(
        self,
        cameras=None,
        output_dir=None,
        fps=30,
        width=None,
        height=None,
        name=None,
        max_frames_per_camera=3000,
    ):
        """Synchronous-mode counterpart to :meth:`start_cameras_recording`.

        Returns ``(on_frame, finalize)`` callables instead of spawning a
        daemon thread. The eval driver wires ``on_frame`` into
        :meth:`~strands_robots.simulation.SimEngine.evaluate_benchmark`'s
        new ``on_frame=`` kwarg (#191), and rendering happens on the eval
        thread - eliminating the cross-thread ``mjData`` race the daemon
        recorder hits under multi-threaded eval (Strands ``Agent`` tool
        dispatch under asyncio, where the eval runs on a worker thread
        distinct from the script main).

        Symptoms of the daemon-thread bug this fixes (#191):
        ``run_mujoco_agent.py --policy=groot`` measured 2-3% frame
        capture rate vs the programmatic single-thread driver, with
        visible greenish GL clear-colour gradient frames at episode
        boundaries. The synchronous mode trades the daemon thread for a
        per-step render call; the eval thread already holds a warm GL
        context (the renderer is per-thread; the policy loop drives
        ``sim.render`` on its own thread for the policy obs), so no
        warmup loop is needed.

        Caller pattern::

            on_frame, finalize = sim.start_cameras_recording_synchronous(
                cameras=["image", "wrist_image"],
                output_dir=video_dir,
                name=rec_name,
            )
            try:
                sim.evaluate_benchmark(
                    benchmark_name=task,
                    n_episodes=5,
                    seed=42,
                    policy_provider="groot",
                    policy_config={...},
                    on_frame=on_frame,
                )
            finally:
                finalize()

        Args:
            cameras: list of camera names; ``None`` = every camera.
            output_dir: where to write ``{tag}__{cam}.mp4``. Defaults to
                ``$TMPDIR/strands_robots/recordings``.
            fps: encoded MP4 frame rate (and target capture rate when
                ``on_frame`` fires more often than ``fps``).
            width, height: per-frame size; defaults to the renderer's
                native resolution.
            name: filename tag (auto-generated UUID prefix when ``None``).
            max_frames_per_camera: safety cap on in-memory buffers.
                Frames beyond the cap are silently dropped (status
                visible via :meth:`get_cameras_recording_status`).

        Returns:
            On success: ``{"status": "success", "content": [{"text": ...},
            {"json": {"on_frame": <callable>, "finalize": <callable>}}]}``.
            The closures aren't natively JSON-serializable; consumers in
            Python code unpack them via the JSON block. Tool-spec callers
            that can't reach Python closures can use the daemon-thread
            variant instead.

            On error: ``{"status": "error", "content": [{"text": ...}]}``
            (no world, already-recording, unresolved camera names, etc.).
        """
        import os as _os
        import tempfile as _tempfile
        import time as _time
        import uuid as _uuid

        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world (or load_scene) first."}]}

        if getattr(self, "_cams_rec_state", None) and self._cams_rec_state.get("running"):
            cur = self._cams_rec_state["name"]
            return {
                "status": "error",
                "content": [{"text": f"Already recording '{cur}'. Call stop_cameras_recording() first."}],
            }

        names, unresolved = self._active_camera_list(cameras)
        if cameras is not None and unresolved:
            return {
                "status": "error",
                "content": [{"text": (f"Camera(s) not found: {unresolved}. Available: {self._list_camera_names()}")}],
            }
        if not names:
            return {"status": "error", "content": [{"text": "No cameras to record."}]}

        out_dir = _os.path.abspath(output_dir or _os.path.join(_tempfile.gettempdir(), "strands_robots", "recordings"))
        _os.makedirs(out_dir, exist_ok=True)
        tag = name or f"rec_{_uuid.uuid4().hex[:8]}"

        buffers: dict[str, list] = {cam: [] for cam in names}
        paths = {cam: _os.path.join(out_dir, f"{tag}__{cam}.mp4") for cam in names}

        state: dict[str, Any] = {
            "running": True,
            "name": tag,
            "cameras": names,
            "fps": fps,
            "width": width,
            "height": height,
            "buffers": buffers,
            "paths": paths,
            "errors": dict.fromkeys(names, 0),
            "output_dir": out_dir,
            "started_at": _time.time(),
            # No daemon thread in synchronous mode; left as None so
            # ``stop_cameras_recording`` can detect this and skip the
            # join.
            "thread": None,
            "max_frames": max_frames_per_camera,
            # Sync mode is opt-in: the on_frame closure renders from the
            # eval thread, no daemon thread is spawned. Tracked in state
            # so introspection / status surfaces can distinguish the two.
            "mode": "synchronous",
        }
        self._cams_rec_state = state

        def _on_frame(_step: int, _observation: dict, _action: dict) -> None:
            """Per-step capture: render each camera + append to the buffer.

            Errors are absorbed into ``state["errors"][cam]`` so a single
            bad frame doesn't abort the rollout (matches the daemon-thread
            policy). Stops capturing once the per-camera cap is hit.
            """
            from strands_robots.simulation.policy_runner import _extract_frame_ndarray

            if not state["running"]:
                return
            for cam in state["cameras"]:
                if len(state["buffers"][cam]) >= state["max_frames"]:
                    continue
                try:
                    r = self.render(camera_name=cam, width=width, height=height)
                    arr = _extract_frame_ndarray(r)
                    if arr is not None:
                        state["buffers"][cam].append(arr)
                    else:
                        state["errors"][cam] += 1
                except Exception as e:  # noqa: BLE001 - per-frame failures non-fatal
                    state["errors"][cam] += 1
                    logger.debug("synchronous recorder (%s) error: %s", cam, e)

        def _finalize() -> dict:
            """Flush buffers to MP4 + clear sim state. Idempotent.

            Returns the same standard result dict as
            :meth:`stop_cameras_recording` so callers can log artifacts
            uniformly. Calling ``finalize()`` after the first call is a
            no-op success ("Was not recording cameras.") - matching the
            ``stop_cameras_recording`` idempotency contract.
            """
            current = getattr(self, "_cams_rec_state", None)
            if current is not state or not state.get("running"):
                return {"status": "success", "content": [{"text": "Was not recording cameras."}]}
            state["running"] = False
            result = self._flush_cameras_recording_state(state)
            self._cams_rec_state = None
            return result

        msg = (
            f"🎬 Recording {len(names)} camera(s) @ {fps} FPS → {out_dir} (synchronous mode)\n"
            f"   tag: {tag}\n"
            f"   cameras: {', '.join(names)}\n"
            "   wire on_frame= into evaluate_benchmark / PolicyRunner.evaluate"
        )
        return {
            "status": "success",
            "content": [
                {"text": msg},
                {"json": {"on_frame": _on_frame, "finalize": _finalize, "name": tag, "output_dir": out_dir}},
            ],
        }

    def get_cameras_recording_status(self):
        """Cheap introspection of an ongoing multi-camera recording."""
        import time as _time

        state = getattr(self, "_cams_rec_state", None)
        if not state or not state.get("running"):
            return {"status": "success", "content": [{"text": "⚪ No active camera recording."}]}

        elapsed = _time.time() - state["started_at"]
        lines = [f"🟢 Recording '{state['name']}' for {elapsed:.1f}s  @ {state['fps']} FPS"]
        for cam in state["cameras"]:
            frames = len(state["buffers"][cam])
            lines.append(f"   📹 {cam:20s} {frames:>5d} frames  ({state['errors'][cam]} errors)")
        return {"status": "success", "content": [{"text": "\n".join(lines)}]}
