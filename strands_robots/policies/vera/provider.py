"""VERA policy provider - :class:`Policy` implementation for strands-robots.

VERA (Video-to-Embodied Robot Action, MIT/CSAIL) is a two-stage closed-loop
video-to-action policy: an embodiment-agnostic **video planner** (DFoT / WAN)
dreams future frames from the current observation, and an embodiment-specific
**Jacobian IDM** translates the dream into robot actions. Both stages live in a
websocket policy server (``vera.server.start_vera_server``); this provider is a
typed websocket client (:class:`VeraWebsocketClient`) plus an optional managed
server subprocess (:class:`VeraServerRunner`), mirroring the ``cosmos3`` service
pattern.

Observation flow
----------------
``SimEngine.get_observation`` returns a **flat** dict::

    {"<joint_name>": float, ..., "<camera_name>": np.ndarray(H, W, 3)}

VERA is *video-first*: it consumes the camera frame(s) only (proprio is read
server-side from its own sim/IDM where needed). The provider keeps a rolling
**context window** of the last ``context_frames`` camera frames (width-concat
across views, matching the server's ``view_keys`` order) and calls the server's
chunked ``infer`` when its local action queue drains - exactly the
``RemotePolicy`` contract from VERA's own eval harness.

Action flow
-----------
The server returns ``{"action": np.ndarray[H, D]}``. Actions are queued and
popped one per :meth:`get_actions` chunk request; each ``D``-vector is mapped to
robot actuator names via ``action_mapping`` (or the embodiment's default action
column names). Values are coerced to python ``float`` / ``list[float]`` per the
:class:`Policy` contract - never raw ``np.ndarray``.
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from typing import Any

import numpy as np

from strands_robots.policies.base import Policy
from strands_robots.utils import finite_number_error, name_list_error, positive_finite_number_error

from .client import VeraWebsocketClient
from .config import VeraConfig
from .server_runner import VeraServerRunner, make_server_runner

logger = logging.getLogger(__name__)


#: Upper bound (exclusive) on the IK output-smoothing coefficient. ``alpha``
#: weights the *previous* joint target in the EMA
#: ``target = (1 - alpha) * solved + alpha * previous``, so ``1.0`` weights the
#: previous value alone and the arm never leaves the pose it was first solved
#: to; anything above ``1.0`` puts a negative weight on the IK solution and the
#: targets diverge away from it, growing without bound down the chunk.
MAX_IK_SMOOTHING = 1.0


def _ik_smoothing_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` is not a usable IK output-smoothing coefficient.

    The EMA blending each IK solution toward the previous target only has a
    meaning for ``alpha`` in ``[0, 1)``: ``0`` disables the smoothing (the
    documented default) and a value approaching ``1`` damps harder. Outside that
    interval the blend is not a smoother:

    * ``1.0`` (and ``True``, which is ``1.0``) makes every commanded target the
      previous one, so the arm freezes at the pose the first solve produced and
      the whole chunk is discarded.
    * above ``1.0`` the weight on the IK solution is negative, so each target
      extrapolates *away* from where the solver put it - measured at ``-5.9x``
      the solved joint travel for ``1.5`` and ``-35.3x`` for ``2.0``.
    * a negative value and ``nan`` both fail the ``alpha > 0`` test the EMA is
      gated on, so the smoothing the caller asked for is silently not applied.
    * ``inf`` makes every target after the first non-finite.

    Only the interval is decided here; numeric-ness, ``bool`` rejection and
    finiteness are :func:`~strands_robots.utils.finite_number_error`'s, whose
    domain this is a bounded subset of. The rule is private to this module
    because the interval is a property of this EMA rather than of a shared
    quantity - :func:`~strands_robots.utils.positive_finite_number_error` is the
    nearest shared domain and its ``> 0`` floor is wrong here, since ``0`` is
    the documented "no smoothing" default.

    Args:
        value: The caller-supplied coefficient.
        param: The parameter it came from, used in the message.
        context: Message prefix identifying the surface that received it.

    Returns:
        An error message, or ``None`` when the value is usable.
    """
    if (err := finite_number_error(value, param, context)) is not None:
        return err
    if not 0.0 <= float(value) < MAX_IK_SMOOTHING:
        return (
            f"{context}: {param} must be in [0, {MAX_IK_SMOOTHING}) - 0 disables "
            f"the smoothing, and {MAX_IK_SMOOTHING} would weight only the previous "
            f"target so the arm never leaves its first solved pose. Got {value!r}."
        )
    return None


def _is_image_value(value: Any) -> bool:
    """Heuristic: is this observation value a camera frame ``(H, W, 3)``?"""
    arr = np.asarray(value) if not isinstance(value, np.ndarray) else value
    return arr.ndim == 3 and arr.shape[-1] == 3


def _resize_frame(frame: np.ndarray, width: int) -> np.ndarray:
    """Resize a ``(H, W, 3) uint8`` frame to ``(width, width, 3)`` (square).

    The VERA WAN/DFoT planner expects each view at a fixed per-view width
    (``VeraConfig.render_width``); the sim may render at any resolution, so the
    provider resizes here so ``sum(view_widths)`` matches the concatenated rgb.
    Uses PIL (Pillow is already a sim dep) to avoid a hard cv2 dependency.
    """
    if frame.shape[0] == width and frame.shape[1] == width:
        return frame
    try:
        from PIL import Image

        return np.asarray(
            Image.fromarray(frame).resize((width, width), getattr(Image, "Resampling", Image).BILINEAR), dtype=np.uint8
        )
    except Exception:
        # Nearest-neighbour numpy fallback (no PIL): index-based resample.
        h, w = frame.shape[:2]
        ys = (np.linspace(0, h - 1, width)).astype(np.int64)
        xs = (np.linspace(0, w - 1, width)).astype(np.int64)
        return np.ascontiguousarray(frame[ys][:, xs])


def _to_uint8_frame(value: Any) -> np.ndarray:
    """Coerce a camera frame to a contiguous ``(H, W, 3) uint8`` array."""
    arr = np.asarray(value)
    if arr.ndim == 4:  # (1, H, W, 3) -> (H, W, 3)
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"camera frame must be (H, W, 3); got {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


class VeraPolicy(Policy):
    """VERA video-to-action policy (service mode via ``vera.server``).

    Args:
        embodiment: ``"pusht"`` | ``"mimicgen"`` | ``"allegro"`` | ``"droid"``.
        server_port: Policy-server websocket port. ``None`` applies the
            per-embodiment default; any other value must be an ``int`` in
            ``[1, 65535]``.
        vis_port: MJPEG live-viewer port. ``None`` applies the per-embodiment
            default; ``0`` disables the viewer; any other value must be an
            ``int`` in ``[1, 65535]``.
        algo_config: WAN planner ``algo_config.yaml`` (point at omni to swap).
        text_prompt: Optional text conditioning for the video planner.
        ckpt_root: Root of downloaded VERA checkpoints (``VERA_CKPT_ROOT``).
        auto_launch_server: Launch + manage the server subprocess on first use.
        dynamics_run_id: Jacobian/IDM checkpoint id (per-embodiment default).
        tracker_backend: IDM point-tracker backend override.
        motion_plan_scale: IDM motion-plan scale (applied live via ``configure``).
        host: Server hostname.
        image_keys: Explicit ordered camera keys to width-concat. When ``None``
            the server's ``view_keys`` (from the connect handshake) are used,
            matched against the observation's image keys. Must be a list of
            distinct non-blank names; a single key passed as a bare string is
            refused rather than read one key per character.
        action_mapping: ``{action_column_name: robot_actuator_name}`` rename of
            the server's action columns to robot actuator names. When ``None``
            columns keep their server names (``action_0``, ``action_1``, …).
        prompt: Default instruction used when ``get_actions`` is called with an
            empty ``instruction`` and the server needs a prompt.
        client: Pre-built client (dependency injection for tests).
        server_runner: Pre-built runner (dependency injection for tests).
        config: Pre-built :class:`VeraConfig` (overrides the kwargs above).

    Notes:
        * Needs camera frames - ``requires_images`` is ``True``.
        * Latency is chunked (a diffusion video planner), not 500 Hz servo;
          one infer returns ``action_horizon`` steps.
    """

    def __init__(
        self,
        embodiment: str = "pusht",
        server_port: int | None = None,
        vis_port: int | None = None,
        algo_config: Any = None,
        text_prompt: str | None = None,
        ckpt_root: Any = None,
        auto_launch_server: bool = True,
        dynamics_run_id: str | None = None,
        tracker_backend: str | None = None,
        motion_plan_scale: float | None = None,
        ik_smoothing: float = 0.0,
        server_mode: str = "subprocess",
        docker_image: str | None = None,
        docker_gpus: str | None = None,
        host: str = "127.0.0.1",
        image_keys: list[str] | None = None,
        action_mapping: dict[str, str] | None = None,
        prompt: str = "",
        client: VeraWebsocketClient | None = None,
        server_runner: VeraServerRunner | None = None,
        config: VeraConfig | None = None,
    ) -> None:
        # Refused before any server or config work: image_keys names the camera
        # keys read out of the observation, and a bare string is iterable per
        # character, so an unchecked one raises KeyError mid-rollout - after the
        # policy server has been launched and the model loaded.
        if image_keys and (err := name_list_error(image_keys, "image_keys", "VeraPolicy")):
            raise ValueError(err)
        # Same reason, same place: the smoothing coefficient blends every
        # commanded joint target with the previous one inside get_actions, so
        # a value outside [0, 1) freezes or diverges the arm mid-rollout
        # instead of damping it.
        if (err := _ik_smoothing_error(ik_smoothing, "ik_smoothing", "VeraPolicy")) is not None:
            raise ValueError(err)
        self.config = config or VeraConfig(
            embodiment=embodiment,  # type: ignore[arg-type]
            host=host,
            server_port=server_port,
            vis_port=vis_port,
            algo_config=algo_config,
            text_prompt=text_prompt,
            ckpt_root=ckpt_root,
            auto_launch_server=auto_launch_server,
            dynamics_run_id=dynamics_run_id,
            tracker_backend=tracker_backend,
            motion_plan_scale=motion_plan_scale,
            server_mode=server_mode,
            docker_image=docker_image or "strands-vera-server:latest",
            docker_gpus=docker_gpus or "all",
        )
        self.image_keys = list(image_keys) if image_keys else None
        self.action_mapping = dict(action_mapping) if action_mapping else None
        self.prompt = prompt
        self._robot_state_keys: list[str] = []

        # --- action binding / IK state ------------------------------------
        # For eef_delta / cartesian_delta embodiments (mimicgen / droid) the
        # server returns a 6-DoF end-effector DELTA, not joint targets. To drive
        # a MuJoCo arm we must IK each delta into joint-space targets keyed by the
        # robot's real joint names. The IK bridge is built lazily on first use
        # from a MjModel + ee_frame supplied via get_actions(**kwargs) or set on
        # the policy (set_ik_target). joint_position embodiments (allegro) need
        # NO IK: the action columns map directly onto robot_state_keys.
        self._mj_model: Any = None  # mujoco.MjModel (injected)
        self._ee_frame_name: str | None = None  # e.g. "hand" / "attachment_site"
        self._ee_frame_type: str = "body"
        self._rotation_dim: int = 3  # axis-angle by default; 6 = rot6d
        self._ik_smoothing: float = float(ik_smoothing)
        self._ik_prev_q: dict[str, float] = {}
        self._translation_scale: float = 1.0
        self._ik_bridge: Any = None  # lazily built MinkIKBridge
        self._sim_namespace: str | None = None  # robot namespace for ee-frame scoping
        self._warned_unbound: bool = False

        self._client = client or VeraWebsocketClient(self.config.host, int(self.config.server_port or 0))
        self._runner = server_runner
        if self._runner is None and self.config.auto_launch_server:
            self._runner = make_server_runner(self.config)

        # Episode state (mirrors VERA's RemotePolicy).
        self._server_meta: dict[str, Any] | None = None
        self._window: deque[np.ndarray] = deque()
        self._queue: deque[dict[str, Any]] = deque()  # robot-actuator dicts, ready to send
        self._session = str(uuid.uuid4())
        self._started = False

    # -- Policy ABC ---------------------------------------------------------

    @property
    def provider_name(self) -> str:
        """Registry key for this provider (``"vera"``)."""
        return "vera"

    @property
    def requires_images(self) -> bool:
        """VERA conditions on camera frames - always needs images."""
        return True

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Record the robot's ordered joint/state keys used to build the state vector.

        Raises:
            ValueError: If ``robot_state_keys`` is not an ordered list of
                distinct non-blank names, per
                :func:`~strands_robots.utils.name_list_error`. A single name
                passed as a bare string is the mistake this catches: ``str`` is
                iterable per character, so it would bind one joint per letter.
        """
        if robot_state_keys and (
            error := name_list_error(robot_state_keys, "robot_state_keys", "set_robot_state_keys")
        ):
            raise ValueError(error)
        self._robot_state_keys = list(robot_state_keys)

    def set_ik_target(
        self,
        mj_model: Any,
        ee_frame_name: str,
        ee_frame_type: str = "body",
        *,
        rotation_dim: int | None = None,
        translation_scale: float | None = None,
    ) -> None:
        """Configure the IK bridge for eef/cartesian-delta embodiments.

        Required to drive a MuJoCo arm with ``mimicgen``/``droid`` (the server
        emits end-effector deltas, not joint targets). Not needed for
        ``allegro`` (joint_position) or ``pusht`` (planar). Resets any existing
        bridge so a later model/frame change rebuilds it.

        Args:
            mj_model: The arm's ``mujoco.MjModel``.
            ee_frame_name: End-effector frame the IK tracks (e.g. ``"hand"``).
            ee_frame_type: ``"body"`` | ``"site"`` | ``"geom"``.
            rotation_dim: Override the delta rotation encoding. ``None`` (the
                default) keeps the embodiment's convention; a supplied one must
                be 3 (axis-angle) or 6 (rot6d), the two encodings the decoder
                implements. Any other width is refused here rather than stored:
                it reaches the decoder mid-rollout, inside ``get_actions``, long
                after this call returned.
            translation_scale: Multiplier on the translation delta, composed
                on top of the OSC position scale. ``None`` (the default)
                leaves the current value; a supplied one must be a positive
                finite number, since it scales every translation delta the
                policy converts to joint targets.
        """
        # Validate before mutating any state, so a refused call leaves the
        # policy untouched (guard-before-mutation discipline).
        #
        # The width selects which rotation parameterization the decoder reads out
        # of each delta, and it implements exactly two. An unusable one is not
        # refused here-and-now by anything downstream: it is stored, then raises
        # from inside ``get_actions`` on the first inference - after the server
        # handshake and the IK bridge build - and an ``int()`` coercion truncated
        # a fractional value first, so that refusal named a width the caller
        # never supplied. Imported here rather than at module scope, matching the
        # other :mod:`~strands_robots.policies.vera.sim_ik` uses in this class.
        from .sim_ik import coerce_rotation_dim

        resolved_rotation_dim: int | None = None
        if rotation_dim is not None:
            resolved_rotation_dim, dim_err = coerce_rotation_dim(rotation_dim, "rotation_dim", "set_ik_target")
            if resolved_rotation_dim is None:
                raise ValueError(dim_err)
        if translation_scale is not None:
            if (
                err := positive_finite_number_error(translation_scale, "translation_scale", "set_ik_target")
            ) is not None:
                raise ValueError(err)
        self._mj_model = mj_model
        self._ee_frame_name = ee_frame_name
        self._ee_frame_type = ee_frame_type
        if resolved_rotation_dim is not None:
            self._rotation_dim = resolved_rotation_dim
        if translation_scale is not None:
            self._translation_scale = float(translation_scale)
        self._ik_bridge = None  # force rebuild

    def autoconfigure_ik(self, mj_model: Any, namespace: str | None = None, *, force: bool = False) -> bool:
        """Zero-config IK setup: discover the ee-frame from the MjModel.

        Called by the simulation alongside ``set_robot_state_keys`` so eef/
        cartesian-delta embodiments (mimicgen / droid) need NO manual wiring.
        Discovers an end-effector frame (site/body) from the compiled model
        (namespace-scoped) and configures the IK target. Idempotent: skips when
        already configured unless ``force``. Picks the rotation encoding from the
        server's ``action_space`` once the metadata handshake has happened
        (``cartesian_delta`` => axis-angle dim 3; ``eef_delta`` => axis-angle dim 3
        by default - override via ``set_ik_target(rotation_dim=...)``).

        Returns:
            True when an ee-frame was resolved + configured, else False.
        """
        if mj_model is None:
            return False
        if self._ee_frame_name is not None and not force:
            return True  # already configured (explicit or prior auto)
        from .ee_frame import discover_ee_frame

        found = discover_ee_frame(mj_model, namespace)
        if found is None:
            return False
        frame_name, frame_type = found
        self.set_ik_target(mj_model, frame_name, frame_type)
        logger.info(
            "VeraPolicy: auto-configured IK target ee=%s/%s (namespace=%s)",
            frame_type,
            frame_name,
            namespace,
        )
        return True

    def set_sim_context(self, mj_model: Any, namespace: str | None = None) -> None:
        """Hook the simulation calls (next to set_robot_state_keys) to pass the
        world model + this robot's namespace. Triggers IK auto-config for
        eef/cartesian-delta embodiments; a no-op for joint_position / pos.

        Safe to call before the server handshake - auto-config is also retried
        lazily on first eef-delta infer if the action_space turns out to need it.
        """
        self._mj_model = mj_model
        self._sim_namespace = namespace
        # If we already know the action space and it needs IK, configure now.
        meta = self._server_meta or {}
        action_space = str(meta.get("action_space", "")).lower()
        if action_space in ("eef_delta", "cartesian_delta", "cartesian_position", "eef_pose"):
            self.autoconfigure_ik(mj_model, namespace)

    def reset(self, seed: int | None = None) -> None:
        """Clear local context + queue and reset the server's episode state."""
        self._window.clear()
        self._queue.clear()
        self._ik_prev_q = {}
        self._session = str(uuid.uuid4())
        reset_info: dict[str, Any] = {"session_id": self._session, "reason": "eval_episode"}
        if seed is not None:
            reset_info["seed"] = int(seed)
        try:
            self._client.reset(reset_info)
        except Exception as e:  # noqa: BLE001 - reset is best-effort
            logger.info("VeraPolicy.reset best-effort failed: %s", e)

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Return the next VERA action chunk as ``list[dict]`` (one per step).

        Appends the current camera frame to the rolling context window and, when
        the local action queue is empty, calls the server's chunked ``infer``;
        the returned ``[H, D]`` chunk is mapped to robot actuator-name dicts.
        """
        self._ensure_started()
        meta = self._server_meta or {}

        frame = self._extract_frame(observation_dict, meta)
        ctx_max = int(meta.get("context_frames", 9))
        if self._window.maxlen != ctx_max:
            self._window = deque(self._window, maxlen=ctx_max)
        self._window.append(frame)

        if not self._queue:
            chunk = self._infer(observation_dict, instruction, meta)
            # Convert the raw [H, D] chunk into a queue of robot-actuator dicts.
            # Routing depends on the server's action_space:
            #   joint_position  -> map columns directly onto robot joints
            #   eef/cartesian_delta -> IK the whole chunk to joint targets
            #   pos / unknown   -> column-name passthrough (action_mapping/raw)
            for action_dict in self._chunk_to_action_dicts(chunk, observation_dict, meta):
                self._queue.append(action_dict)

        if not self._queue:
            return []
        return [self._queue.popleft()]

    # -- internals ----------------------------------------------------------

    def _ensure_started(self) -> None:
        """Launch the server (once) and complete the metadata handshake."""
        if self._started:
            return
        if self._runner is not None:
            self._runner.start()
        self._server_meta = self._client.get_server_metadata()
        # Apply live-tunable knobs that don't need a model rebuild.
        if self.config.motion_plan_scale is not None:
            try:
                self._client.configure({"motion_plan_scale": float(self.config.motion_plan_scale)})
            except Exception as e:  # noqa: BLE001
                logger.info("VeraPolicy live configure(motion_plan_scale) skipped: %s", e)
        self._started = True

    def _resolve_view_keys(self, observation_dict: dict[str, Any], meta: dict[str, Any]) -> list[str]:
        """Ordered camera keys to width-concat: explicit > server views > discovered."""
        if self.image_keys:
            return self.image_keys
        obs_image_keys = [k for k, v in observation_dict.items() if _is_image_value(v)]
        server_views = [str(v) for v in meta.get("view_keys", [])]
        # Match server views to observation keys when names line up; otherwise
        # fall back to the discovered image keys in dict order.
        matched = [k for k in server_views if k in observation_dict]
        if matched:
            return matched
        return obs_image_keys

    def _extract_frame(self, observation_dict: dict[str, Any], meta: dict[str, Any]) -> np.ndarray:
        """Build one width-concatenated ``(H, W, 3) uint8`` frame from all views."""
        view_keys = self._resolve_view_keys(observation_dict, meta)
        if not view_keys:
            raise ValueError(
                "VeraPolicy requires at least one camera frame in the observation "
                f"(keys: {list(observation_dict)}); none look like (H, W, 3) images."
            )
        # ``VeraConfig.__post_init__`` holds ``render_width`` to the shared media
        # pixel domain and normalizes it to a plain ``int``, so there is nothing
        # left to coerce or to fall back to here. This line used to read
        # ``int(self.config.render_width or 128)``, which substituted 128 for a
        # width of 0 and truncated a ``2.7`` to a 2-pixel view, both under a
        # success result.
        rw = self.config.render_width
        assert rw is not None  # guaranteed by VeraConfig.__post_init__
        frames = [_resize_frame(_to_uint8_frame(observation_dict[k]), rw) for k in view_keys]
        if len(frames) == 1:
            return frames[0]
        # Width-concat across views, each already render_width wide.
        return np.ascontiguousarray(np.concatenate(frames, axis=1))

    def _infer(self, observation_dict: dict[str, Any], instruction: str, meta: dict[str, Any]) -> np.ndarray:
        """Pack the rolling context window and call the server's ``infer``."""
        view_keys = self._resolve_view_keys(observation_dict, meta)
        context_rgb = np.stack(list(self._window), axis=0)  # (T, H, W, 3) uint8
        # Each view was resized to render_width before concat, so view_widths is
        # simply render_width per view (sum == concatenated rgb width). Read from
        # the config rather than divided back out of the frame: the second
        # fallback this line used to carry (``context_rgb.shape[2] // n_views``)
        # was a second definition of the same falsy case that ``_extract_frame``
        # spelled as 128, and the invariant above is what made the two agree
        # rather than anything asserting they must.
        n_views = max(1, len(view_keys))
        per_w = self.config.render_width
        assert per_w is not None  # guaranteed by VeraConfig.__post_init__
        view_widths = [per_w] * n_views
        req: dict[str, Any] = {
            "context_rgb": context_rgb,
            "view_keys": list(view_keys),
            "view_widths": view_widths,
            "session_id": self._session,
        }
        if meta.get("needs_prompt"):
            req["prompt"] = instruction or self.prompt or ""
        out = self._client.infer(req)
        action = np.asarray(out["action"], dtype=np.float32)
        if action.ndim == 1:
            action = action[None, :]
        return action

    # -- action binding -----------------------------------------------------

    def _chunk_to_action_dicts(
        self, chunk: np.ndarray, observation_dict: dict[str, Any], meta: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Convert a raw ``[H, D]`` VERA chunk into a list of robot-actuator dicts.

        Routes on the server's ``action_space`` so each provider output maps onto
        the robot's REAL joint/actuator names (what ``send_action`` resolves) -
        never bare ``action_i`` keys (which the sim drops as unresolved):

        * ``joint_position`` (allegro): columns map directly onto
          ``robot_state_keys`` (positional joint targets).
        * ``eef_delta`` / ``cartesian_delta`` (mimicgen / droid): the chunk is a
          6-DoF end-effector delta - IK the whole chunk to joint targets keyed by
          ``robot_state_keys`` (needs ``set_ik_target`` / a ``mj_model`` kwarg).
        * ``pos`` / unknown: keep server column names (or ``action_mapping``).
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        action_space = str(meta.get("action_space", "")).lower()
        gripper_idx = int(meta.get("gripper_dim_index", -1))
        gripper_is_raw = bool(meta.get("gripper_is_raw", True))

        # eef / cartesian delta -> IK to joints
        if action_space in ("eef_delta", "cartesian_delta", "cartesian_position", "eef_pose"):
            ik_dicts = self._ik_chunk_to_action_dicts(chunk, observation_dict, meta, gripper_idx, gripper_is_raw)
            if ik_dicts is not None:
                return ik_dicts
            # IK not configured -> fall through to (warned) raw mapping so the
            # failure is loud + the caller can still inspect the output.

        # joint_position -> direct column->joint binding
        if action_space in ("joint_position", "joint_velocity", "pos") or self._robot_state_keys:
            return [self._vector_to_action_dict(row, meta, gripper_idx, gripper_is_raw) for row in chunk]

        # unknown -> raw column names (+ optional action_mapping)
        return [self._vector_to_action_dict(row, meta, gripper_idx, gripper_is_raw) for row in chunk]

    def _action_column_names(self, action_dim: int, meta: dict[str, Any]) -> list[str]:
        """Resolve action-column names, binding to the robot's joints when possible.

        Priority: explicit ``action_mapping`` > robot joint names
        (``robot_state_keys``, the names ``send_action`` resolves) > server
        default ``action_{i}`` (last-resort; warned once - the sim would drop it).
        """
        # 1) explicit caller-provided rename.
        if self.action_mapping:
            base = [f"action_{i}" for i in range(action_dim)]
            return [self.action_mapping.get(n, n) for n in base]

        # 2) bind to the robot's real joint names. For positional action spaces
        #    column i is joint i. When the action carries a trailing gripper that
        #    the robot's joint list does not, keep extra columns as action_i.
        keys = list(self._robot_state_keys)
        if keys and len(keys) >= action_dim:
            return keys[:action_dim]
        if keys and len(keys) < action_dim:
            # joints + (gripper / extra) columns: bind what we can, name the rest.
            extra = [f"action_{i}" for i in range(len(keys), action_dim)]
            return keys + extra

        # 3) nothing to bind to -> raw names; warn ONCE (these get dropped by the
        #    sim's send_action as unresolved keys -> robot won't move).
        if not self._warned_unbound:
            logger.warning(
                "VeraPolicy: no robot_state_keys set and no action_mapping; emitting "
                "raw 'action_i' keys which the simulator will treat as UNRESOLVED "
                "(robot will not move). Call set_robot_state_keys(...) (the sim does "
                "this automatically before rollout) or pass action_mapping=..."
            )
            self._warned_unbound = True
        return [f"action_{i}" for i in range(action_dim)]

    def _vector_to_action_dict(
        self,
        vec: np.ndarray,
        meta: dict[str, Any],
        gripper_idx: int | None = None,
        gripper_is_raw: bool | None = None,
    ) -> dict[str, Any]:
        """Map one ``D``-vector to ``{actuator_name: float}`` (gripper binarized).

        Honours the server's ``gripper_dim_index`` + ``gripper_is_raw`` contract:
        a raw gripper float is binarized at >0.5 -> close (1.0).
        """
        vec = np.asarray(vec, dtype=np.float32).ravel()
        names = self._action_column_names(vec.shape[0], meta)
        if gripper_idx is None:
            gripper_idx = int(meta.get("gripper_dim_index", -1))
        if gripper_is_raw is None:
            gripper_is_raw = bool(meta.get("gripper_is_raw", True))
        out: dict[str, Any] = {}
        for i, name in enumerate(names):
            val = float(vec[i])
            if i == gripper_idx and gripper_is_raw:
                val = 1.0 if val > 0.5 else 0.0
            out[name] = val
        return out

    # -- eef/cartesian-delta IK --------------------------------------------

    def _resolve_ik_inputs(self, observation_dict: dict[str, Any]) -> tuple[Any, str, np.ndarray] | None:
        """Gather (mj_model, ee_frame, q_init) for IK, or None when unavailable.

        ``mj_model`` / ``ee_frame`` come from ``set_ik_target`` (preferred) or
        from per-call kwargs the sim may pass. ``q_init`` is the robot's current
        joint configuration, read from the observation in ``robot_state_keys``
        order (the same keys the sim seeds the policy with).
        """
        mj_model = self._mj_model
        # Lazy zero-config: if a model is present but no ee-frame yet, discover it.
        if mj_model is not None and self._ee_frame_name is None:
            self.autoconfigure_ik(mj_model, self._sim_namespace)
        ee_frame = self._ee_frame_name
        if mj_model is None or ee_frame is None:
            return None
        # Current joint configuration as the IK seed. mink.Configuration works in
        # FULL model qpos space (nq), which includes non-robot DOFs (e.g. free
        # bodies). Seed from the model rest pose (qpos0) and write the robot's
        # joint values into their qpos addresses, so IK perturbs only the arm.
        keys = self._robot_state_keys
        if not keys:
            return None
        addr = self._joint_qpos_addr(mj_model)  # {state_key: qpos_index}
        try:
            import numpy as _np

            q_full = _np.array(mj_model.qpos0, dtype=_np.float64).copy()
        except Exception:
            q_full = np.zeros(int(mj_model.nq), dtype=np.float64)
        for k in keys:
            if k not in observation_dict:
                return None
            val = float(np.asarray(observation_dict[k]).reshape(-1)[0])
            if k in addr:
                q_full[addr[k]] = val
        return mj_model, ee_frame, q_full

    def _joint_qpos_addr(self, mj_model: Any) -> dict[str, int]:
        """Map each robot_state_key -> its qpos address in the full model.

        Robot joints are namespaced in the compiled model (``<ns>/<joint>``);
        ``robot_state_keys`` are unqualified. Match by suffix so the IK seed and
        output read/write the correct qpos slots regardless of other DOFs (free
        bodies, multiple robots) present in the scene.

        Cached against both inputs the mapping is derived from - the model and
        ``robot_state_keys`` - because they move independently and the model is
        not the one that moves on a rebind. ``SimEngine`` hands the single
        compiled world model to whichever robot it binds
        (``bind_policy_sim_context``), immediately after setting that robot's
        keys, so a second robot in the same scene arrives with new keys and the
        *identical* model. A model-only key served it the previous robot's
        addresses, and neither consumer can report that: both skip a key the
        mapping does not carry, so the IK seed silently keeps the rest pose
        (:meth:`_resolve_ik_inputs`) and the decoded targets silently omit every
        arm joint (:meth:`_ik_chunk_to_action_dicts`). The keys are the whole
        domain of this mapping, so a stale one is not a near miss.

        The model is held and compared by identity, not by ``id()``: an ``id()``
        is unique only while its object is alive and an int key keeps nothing
        alive, so a freed model's address can be reused by the model that
        replaces it and collide. Holding it pins nothing new - ``self._mj_model``
        already holds the same object for as long as it is the active model, and
        this single-slot cache drops a superseded one on the next miss.
        """
        keys = tuple(self._robot_state_keys)
        cache = getattr(self, "_qpos_addr_cache", None)
        if cache is not None and cache[0] is mj_model and cache[1] == keys:
            return cache[2]

        addr: dict[str, int] = {}
        # Real MjModel: map robot_state_keys -> qpos addresses by (namespaced)
        # joint name. Falls back to a positional identity map when the model
        # lacks MuJoCo joint introspection (e.g. a test stub) - state_key i -> i.
        try:
            import mujoco

            for j in range(int(mj_model.njnt)):
                name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
                if not name:
                    continue
                short = name.split("/")[-1]
                qadr = int(mj_model.jnt_qposadr[j])
                for k in keys:
                    if k in (short, name):
                        addr[k] = qadr
        except (AttributeError, ImportError, TypeError):
            addr = {}
        if not addr:
            addr = {k: i for i, k in enumerate(keys)}
        self._qpos_addr_cache = (mj_model, keys, addr)
        return addr

    def _ensure_ik_bridge(self, mj_model: Any, ee_frame: str):
        """Lazily build (and cache) the MinkIKBridge."""
        if self._ik_bridge is None:
            from .sim_ik import MinkIKBridge

            self._ik_bridge = MinkIKBridge(mj_model, ee_frame, self._ee_frame_type)
        return self._ik_bridge

    def _ik_chunk_to_action_dicts(
        self,
        chunk: np.ndarray,
        observation_dict: dict[str, Any],
        meta: dict[str, Any],
        gripper_idx: int,
        gripper_is_raw: bool,
    ) -> list[dict[str, Any]] | None:
        """IK an eef/cartesian-delta chunk into joint-keyed action dicts.

        Returns ``None`` (so the caller can fall back + warn) when the IK target
        is not configured (no mj_model/ee_frame, or q_init unreadable).
        """
        inputs = self._resolve_ik_inputs(observation_dict)
        if inputs is None:
            if not self._warned_unbound:
                logger.warning(
                    "VeraPolicy: action_space=%r emits end-effector DELTAS, but no IK "
                    "target is configured (mj_model + ee_frame). Call "
                    "policy.set_ik_target(mj_model, ee_frame_name=...) before rollout "
                    "(mimicgen/droid). Falling back to raw action_i keys, which the "
                    "simulator will DROP as unresolved (robot will not move).",
                    meta.get("action_space"),
                )
                self._warned_unbound = True
            return None

        mj_model, ee_frame, q_init = inputs
        bridge = self._ensure_ik_bridge(mj_model, ee_frame)
        from .sim_ik import decode_vera_delta_chunk_to_targets

        # Gripper presence: VERA uses gripper_dim_index == -1 to mean "the gripper
        # is the LAST column" (Python-style index), NOT "no gripper". Detect a
        # gripper column from the chunk width vs the pose dims (3 trans + rot):
        # any trailing column beyond the pose block is the gripper.
        pose_dims = 3 + self._rotation_dim
        chunk_dim = int(chunk.shape[1]) if chunk.ndim == 2 else len(chunk)
        has_gripper = chunk_dim > pose_dims
        eff_gripper_idx = gripper_idx if (gripper_idx is not None and gripper_idx >= 0) else (chunk_dim - 1)
        result = decode_vera_delta_chunk_to_targets(
            chunk,
            bridge,
            q_init,
            rotation_dim=self._rotation_dim,
            has_gripper=has_gripper,
            gripper_dim_index=eff_gripper_idx if has_gripper else -1,
            translation_scale=self._translation_scale,
        )
        qpos = np.asarray(result["qpos"], dtype=np.float32)  # [T, nq]
        grip = result.get("gripper")
        keys = self._robot_state_keys
        addr = self._joint_qpos_addr(mj_model)
        # Gripper: a robot joint whose name hints at the gripper receives the
        # binarized gripper command (IK does not solve the gripper DOFs).
        gripper_keys = [k for k in keys if "gripper" in k.lower() or "finger" in k.lower()]
        arm_keys = [k for k in keys if k not in gripper_keys]

        dicts: list[dict[str, Any]] = []
        for t in range(qpos.shape[0]):
            row = qpos[t]
            # Read each arm joint back from its qpos address (robust to free bodies).
            d: dict[str, Any] = {k: float(row[addr[k]]) for k in arm_keys if k in addr and addr[k] < row.shape[0]}
            # EMA smoothing across steps to damp IK jitter (Cosmos3 reasoner
            # flagged jittery Panda motion). alpha in [0,1); 0 disables.
            a_sm = self._ik_smoothing
            if a_sm > 0.0:
                for jk in list(d):
                    prev = self._ik_prev_q.get(jk)
                    if prev is not None:
                        d[jk] = (1.0 - a_sm) * d[jk] + a_sm * prev
                    self._ik_prev_q[jk] = d[jk]
            if grip is not None and t < len(grip):
                gval = float(grip[t])
                if gripper_is_raw:
                    gval = 1.0 if gval > 0.5 else 0.0
                for gk in gripper_keys:
                    d[gk] = gval
            dicts.append(d)
        logger.debug(
            "VeraPolicy IK: %d steps, tracking mean=%.1fmm max=%.1fmm",
            len(dicts),
            result["tracking_error"]["mean_mm"],
            result["tracking_error"]["max_mm"],
        )
        return dicts

    def close(self) -> None:
        """Close the client and stop the managed server subprocess."""
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass
        if self._runner is not None:
            self._runner.stop()
