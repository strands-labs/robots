"""Native provider for the Pollen Microduck 14-DOF locomotion policies.

The Microduck (Pollen Robotics' open 14-DOF biped) ships a family of ONNX
policies - ``alpha_walking``, ``alpha_stand``, ``alpha_sitstand``, ``roulade``,
``ball_kick_*``, ``roller*``, ``alpha_ground_pick`` - each an actor with the
input normaliser fused into the graph. :class:`MicroduckPolicy` adapts one such
export to the :class:`~strands_robots.policies.base.Policy` interface so it runs
through the standard ``Robot(...).run_policy`` seam in MuJoCo or on hardware.

Two things make this provider almost configuration-free:

* **The ONNX is self-describing.** ``session.get_modelmeta().custom_metadata_map``
  carries ``joint_names``, ``default_joint_pos``, ``action_scale`` and
  ``command_names``; the policy reads them on first use, so pointing it at a
  different weight file reconfigures it. Explicit constructor arguments win over
  the metadata when supplied.
* **Normalisation is baked in.** The observation vector is fed RAW to the graph
  (see :mod:`.observation`); the provider NEVER re-normalises.

Decode is ``motor_target = DEFAULT_POSE + action * action_scale``, and the RAW
action (pre-decode) is what feeds the next tick's ``last_action`` observation
block - matching Pollen's reference deployment exactly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from strands_robots.policies.base import Policy
from strands_robots.utils import (
    finite_vector_error,
    name_list_error,
    positive_finite_number_error,
    require_optional,
)

from . import observation as obs_builder
from ._session import MicroduckSession

logger = logging.getLogger(__name__)

#: The 14 actuated joints, in CONTRACT order. Never permute - the ONNX obs and
#: action tensors are index-based against exactly this order.
MICRODUCK_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)

#: Neutral pose (rad), same order. Baked into every shipped ONNX's
#: ``default_joint_pos`` metadata; kept here as the fallback when a session
#: without metadata is injected (e.g. a test stub).
MICRODUCK_DEFAULT_POSE: tuple[float, ...] = (
    0.0,
    -0.0873,
    -0.4579,
    -0.0049,
    0.4530,
    0.3491,
    0.3491,
    0.0,
    0.0,
    0.0,
    0.0873,
    0.4579,
    0.0049,
    -0.4530,
)

#: Unified command width (twist 3 + head_pose 4 + body_pose 6) for the shipped
#: alpha policies. Legacy twist-only policies use 3. Overridden from the
#: ``command_names`` metadata when the export declares narrower commands.
_DEFAULT_COMMAND_WIDTH = 13

#: Per-command-name component counts, used to size the command vector from the
#: ONNX ``command_names`` metadata (dead-weight rule: slots stay present+zero).
_COMMAND_COMPONENTS = {"twist": 3, "head_pose": 4, "body_pose": 6}

#: Observation components that are NOT the command block: ``base_ang_vel`` (3)
#: and ``projected_gravity`` (3). The rest of the fixed part is three per-joint
#: blocks (``joint_pos``, ``joint_vel``, ``last_action``), so the full non-command
#: width is ``_BASE_OBS_WIDTH + 3 * len(joint_names)`` - 48 for the 14-DOF biped.
_BASE_OBS_WIDTH = 6


#: The component counts :meth:`MicroduckPolicy.get_actions` documents for the
#: well-known ``target_velocity`` kwarg: ``[vx, vy, omega]`` and ``[vx, vy]``.
#: Single-sourced so the refusal and the docstring cannot drift apart. The
#: two-component form is deliberately kept: this policy's command vector
#: persists across ticks, so "set vx and vy, leave omega" is a coherent request
#: - which is why the family's other readers, whose command is rebuilt per call,
#: require three.
TARGET_VELOCITY_WIDTHS: frozenset[int] = frozenset({2, 3})


def _action_scale_error(value: Any, source: str) -> str | None:
    """Return why ``value`` cannot be an action scale, or ``None`` if it can.

    ``action_scale`` is the only path from the network's raw output to the joint
    targets - ``motor_target = default_pose + raw_action * action_scale`` - so a
    value that is not a positive finite number does not fail, it silently
    changes what the policy commands. A scale of ``0`` (or ``False``, an ``int``
    subclass) makes every target exactly ``default_pose``: the network's
    decision is discarded and the biped holds its nominal stance while the
    rollout reports success. A non-finite scale makes all fourteen targets
    ``nan``. Neither is checked downstream - :func:`observation.decode_action`
    runs per tick inside :meth:`MicroduckPolicy.get_actions`, so a non-real
    value surfaced as a bare ``TypeError`` from its own ``float()`` only after
    the session had loaded and the rollout had started.

    The scale reaches the decode two ways - the constructor kwarg and the ONNX
    ``action_scale`` metadata - and both consult this one domain, so a scale a
    caller is refused cannot arrive through the file instead. ``source`` names
    which route the value came from so the refusal points at the thing to fix.

    Args:
        value: The candidate scale, from a caller or from ONNX metadata.
        source: Human-readable route, e.g. ``"constructor"``.

    Returns:
        The refusal text, or ``None`` when ``value`` is usable.
    """
    return positive_finite_number_error(value, "action_scale", f"MicroduckPolicy ({source})")


class MicroduckPolicy(Policy):
    """ONNX locomotion policy for the Pollen Microduck 14-DOF biped.

    Args:
        onnx_path: Path to a shipped Microduck ``.onnx`` policy (e.g.
            ``alpha_walking.onnx``). Required unless ``session=`` is injected.
        session: An injected ONNX-like session (see
            :class:`~strands_robots.policies.microduck._session.MicroduckSession`)
            for tests / non-onnxruntime hosts.
        providers: Ordered onnxruntime execution providers. Defaults to
            ``["CPUExecutionProvider"]`` (the biped is a tiny MLP; CPU is ample).
        command: Initial command vector. Width is taken from the ONNX
            ``command_names`` metadata; defaults to all-zero (stand in place).
            Per-tick overrides arrive via ``get_actions(command=...)`` or the
            well-known ``target_velocity`` kwarg (writes the twist block).
        joint_names / default_pose / command_names: Explicit config overrides.
            When omitted, read from the ONNX metadata on first inference.
            Explicit values always win.
        action_scale: Scale applied to the raw ONNX output before it becomes a
            joint-position offset. Must be a positive finite number: a scale of
            ``0`` discards the network's decision and holds ``default_pose``,
            and a non-finite one makes every target ``nan``. When omitted, read
            from the ONNX ``action_scale`` metadata and held to the same domain.
    """

    #: Proprioceptive locomotion - no cameras.
    requires_images = False

    def __init__(
        self,
        *,
        onnx_path: str | Path | None = None,
        session: MicroduckSession | None = None,
        providers: list[str] | None = None,
        command: NDArray[np.float32] | list[float] | None = None,
        joint_names: list[str] | None = None,
        default_pose: NDArray[np.float32] | list[float] | None = None,
        action_scale: float | None = None,
        command_names: list[str] | None = None,
    ) -> None:
        if session is None and onnx_path is None:
            raise ValueError(
                "MicroduckPolicy requires either `onnx_path` (real weights) or "
                "`session` (injected for tests). Neither was supplied."
            )

        self._session: MicroduckSession | None = session
        self._onnx_path: Path | None = Path(onnx_path) if onnx_path else None
        self._providers = providers or ["CPUExecutionProvider"]
        self._input_name: str | None = None

        # Config; None means "read from ONNX metadata on first inference".
        self._joint_names: list[str] | None = list(joint_names) if joint_names else None
        self._default_pose: NDArray[np.float32] | None = (
            np.asarray(default_pose, dtype=np.float32) if default_pose is not None else None
        )
        if action_scale is not None and (error := _action_scale_error(action_scale, "constructor")):
            raise ValueError(error)
        self._action_scale: float | None = float(action_scale) if action_scale is not None else None
        self._command_names: list[str] | None = list(command_names) if command_names else None
        # ``gravity_source`` is training-time and baked into the ONNX (Pollen's
        # ``use_projected_gravity``); resolved from ``custom_metadata_map`` on
        # first inference, matching the ``joint_names``/``default_pose``/etc.
        # metadata-first pattern.  Kept ``None`` here so ``_ensure_config`` is
        # the single owner of "which slot-two branch this checkpoint expects".
        self._gravity_source: str | None = None
        self._configured = False

        self._initial_command = np.asarray(command, dtype=np.float32) if command is not None else None
        self._command: NDArray[np.float32] | None = None
        self._last_action: NDArray[np.float32] | None = None
        self._robot_state_keys: list[str] = list(MICRODUCK_JOINT_NAMES)

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        """Registry key for this provider (``"microduck"``)."""
        return "microduck"

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Record the robot's ordered joint keys.

        The policy indexes its obs/action tensors by joint NAME (read from the
        ONNX metadata), so it does not depend on this ordering; the list is
        validated for shape only, through the shared
        :func:`~strands_robots.utils.name_list_error` domain, so a single joint
        name passed as a bare string (iterable per character) is refused rather
        than bound one joint per letter.

        Raises:
            ValueError: If ``robot_state_keys`` is truthy but not an ordered
                list of distinct non-blank names.
        """
        if robot_state_keys and (
            error := name_list_error(robot_state_keys, "robot_state_keys", "set_robot_state_keys")
        ):
            raise ValueError(error)
        self._robot_state_keys = list(robot_state_keys) if robot_state_keys else list(MICRODUCK_JOINT_NAMES)

    def reset(self, seed: int | None = None) -> None:
        """Restore per-episode state (last action + command) at episode start.

        ``last_action`` is cleared and rebuilt lazily by :meth:`get_actions`.
        The command has no such lazy rebuild - ``_ensure_config`` builds it once
        and early-returns thereafter - so it is restored here, to the same
        vector a first episode starts from. Clearing it instead would give
        ``reset`` two meanings: before the first ``get_actions`` it restores the
        constructor's command (``_ensure_config`` has yet to run), after it the
        next tick would find no command at all.
        """
        self._last_action = None
        self._command = self._episode_start_command() if self._configured else None

    async def get_actions(
        self,
        observation_dict: dict[str, Any],
        instruction: str,  # unused - locomotion policy is not VLA
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return the next per-joint motor targets from the ONNX policy.

        Builds the raw 61-D (or command-sized) observation, runs the fused
        actor+normaliser graph, records the RAW action as ``last_action`` for
        the next tick, and decodes to motor targets.

        Recognised ``**kwargs``:

        * ``command``: full command vector override for this tick. Must be
          ``command_names``-wide and finite.
        * ``target_velocity``: ``[vx, vy, omega]`` (or ``[vx, vy]``, which leaves
          ``omega`` at its current value) written into the twist slots of the
          command vector - the well-known locomotion goal kwarg. Must carry
          finite numbers and one of the two component counts above; any other
          width is refused rather than truncated or partially written.

        Note that what the twist slots MEAN is a property of the loaded weights,
        not of this method. Pollen's locomotion exports read them as a velocity,
        which is what ``target_velocity`` writes; other exports in the family read
        the same three slots differently, and for those a caller supplies the
        slots wholesale through ``command``.

        Raises:
            ValueError: If ``command`` or ``target_velocity`` carries a
                non-numeric or non-finite component, if ``command`` is not
                ``command_names``-wide, or if ``target_velocity``'s component
                count is not in :data:`TARGET_VELOCITY_WIDTHS`.
        """
        self._ensure_config()
        assert self._joint_names is not None and self._default_pose is not None
        assert self._action_scale is not None and self._command is not None

        self._apply_command_kwargs(kwargs)

        if self._last_action is None:
            self._last_action = np.zeros(len(self._joint_names), dtype=np.float32)

        vector = obs_builder.build_observation(
            observation_dict,
            joint_names=self._joint_names,
            default_pose=self._default_pose,
            last_action=self._last_action,
            command=self._command,
            gravity_source=self._gravity_source or obs_builder.GRAVITY_SOURCE_PROJECTED,
        )

        raw_action = self.infer_raw(vector)
        # The graph's own output width is the third width contract in this class,
        # and it was the one not checked: ``default_pose`` is held to
        # ``len(joint_names)`` in ``_ensure_config`` and a ``command`` override to
        # the width ``command_names`` declares in ``_apply_command_kwargs``. This
        # one differs from both in being fed BACK - ``last_action`` is this array,
        # so the observation the graph is handed on the NEXT tick is only
        # ``48 + len(command)`` wide while this holds. A width that happens to
        # broadcast against ``default_pose`` (1) therefore decoded silently and
        # then changed the observation width from tick 2 onward, reporting a full
        # action dict throughout; any other width raised
        # ``operands could not be broadcast together`` from inside numpy's decode,
        # naming neither this policy nor the graph. Refuse here, where both the
        # expected width and its source are still in hand.
        if raw_action.shape[0] != len(self._joint_names):
            raise ValueError(
                f"{type(self).__name__}: the ONNX graph returned {raw_action.shape[0]} action "
                f"value(s) but there are {len(self._joint_names)} joints "
                f"(from joint_names). The action feeds both the joint targets and "
                f"the next tick's last_action observation block, so a width other "
                f"than the joint count cannot be used."
            )
        # A non-finite component is refused for the same reason the sibling
        # scene-construction guards refuse one: it is written straight out (here to
        # every joint target) AND fed back into the next observation, so the
        # rollout reports success while commanding nan and poisoning the vector the
        # graph sees next. ``EmpiricalNormalization`` is baked into the graph, so
        # nothing downstream sanitises it.
        if error := finite_vector_error(f"{type(self).__name__}.get_actions", "the ONNX action", raw_action):
            raise ValueError(error)
        self._last_action = raw_action.copy()

        motor_target = obs_builder.decode_action(
            raw_action, default_pose=self._default_pose, action_scale=self._action_scale
        )
        action_dict: dict[str, float] = {name: float(motor_target[i]) for i, name in enumerate(self._joint_names)}
        return [action_dict]

    # ------------------------------------------------------------------
    # Inference seam
    # ------------------------------------------------------------------

    def infer_raw(self, obs_vector: NDArray[np.float32]) -> NDArray[np.float32]:
        """Run the ONNX graph on a RAW observation vector, return the raw action.

        This is the byte-compatibility seam: it feeds the vector straight to the
        session (no normalisation - that is fused into the graph) exactly as
        Pollen's ``infer_policy.py`` does, so an identical obs yields an
        identical action. ``get_actions`` calls this internally.
        """
        session = self._get_session()
        assert self._input_name is not None  # set by _get_session on first call
        batch = np.asarray(obs_vector, dtype=np.float32).reshape(1, -1)
        outputs = session.run(None, {self._input_name: batch})
        return np.asarray(outputs[0], dtype=np.float32).squeeze(0)

    # ------------------------------------------------------------------
    # Config / session lifecycle
    # ------------------------------------------------------------------

    def _apply_command_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Fold per-call command overrides into the running command vector."""
        assert self._command is not None
        if kwargs.get("command") is not None:
            # Asked before the coercion, for the same two reasons as the sibling
            # ``target_velocity`` below: a non-numeric element otherwise surfaced
            # as a bare ``could not convert string to float`` from numpy, naming
            # neither this policy nor the parameter; and a nan/inf one was
            # assigned to ``self._command`` and only refused afterwards by
            # ``build_observation``, so a caller that handled that error and kept
            # ticking carried the poisoned command into every later tick and every
            # later episode.
            if error := finite_vector_error(f"{type(self).__name__}.get_actions", "command", kwargs["command"]):
                raise ValueError(error)
            new = np.asarray(kwargs["command"], dtype=np.float32).reshape(-1)
            if new.shape[0] != self._command.shape[0]:
                raise ValueError(
                    f"MicroduckPolicy: command override has width {new.shape[0]}, "
                    f"expected {self._command.shape[0]} (from command_names)."
                )
            self._command = new
        tv = kwargs.get("target_velocity")
        if tv is not None:
            # Two sibling providers reading this same well-known goal key hold it
            # to a domain that names it - ``WBCPolicy._validate_velocity`` and the
            # ``param_name="target_velocity"`` guard MotionBricks applies on both
            # its constructor and per-call paths - and this one did not. A
            # non-finite component WAS still refused, but downstream by
            # ``build_observation``, which names ``command`` and the assembled
            # observation rather than the parameter the caller passed. Ask here,
            # where the caller's own parameter name is still in hand.
            if error := finite_vector_error(f"{type(self).__name__}.get_actions", "target_velocity", tv):
                raise ValueError(error)
            tv = np.asarray(tv, dtype=np.float32).reshape(-1)
            # This method documents exactly two spellings for the kwarg,
            # ``[vx, vy, omega]`` and ``[vx, vy]``, and the write accepted any
            # width. A longer one was silently truncated to its first three
            # components; a shorter one wrote only the slots it covered, and the
            # command vector persists across ticks, so ``target_velocity=[0.3]``
            # left the PREVIOUS tick's lateral and yaw components commanding the
            # robot under a reported success. The sibling ``command`` override in
            # this same method already refuses a width it cannot honor, naming the
            # expected width and its source; this one now does too.
            if tv.shape[0] not in TARGET_VELOCITY_WIDTHS:
                raise ValueError(
                    f"MicroduckPolicy: target_velocity has {tv.shape[0]} component(s), "
                    f"expected {' or '.join(f'{w}' for w in sorted(TARGET_VELOCITY_WIDTHS, reverse=True))} "
                    f"([vx, vy, omega] or [vx, vy])."
                )
            n = min(tv.shape[0], self._command.shape[0])
            self._command[:n] = tv[:n]

    def _ensure_config(self) -> None:
        """Resolve joint_names / default_pose / action_scale / command from ONNX metadata."""
        if self._configured:
            return
        meta = self._read_metadata()

        if self._joint_names is None:
            names = meta.get("joint_names")
            self._joint_names = [s.strip() for s in names.split(",")] if names else list(MICRODUCK_JOINT_NAMES)
        if self._default_pose is None:
            dp = meta.get("default_joint_pos")
            self._default_pose = (
                np.array([float(x) for x in dp.split(",")], dtype=np.float32)
                if dp
                else np.array(MICRODUCK_DEFAULT_POSE, dtype=np.float32)
            )
        if self._action_scale is None:
            declared = meta.get("action_scale", 1.0)
            try:
                parsed = float(declared)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"MicroduckPolicy: ONNX metadata action_scale={declared!r} is not a number.") from exc
            if error := _action_scale_error(parsed, "ONNX metadata"):
                raise ValueError(error)
            self._action_scale = parsed
        if self._command_names is None:
            cn = meta.get("command_names")
            self._command_names = [s.strip() for s in cn.split(",")] if cn else None

        # ``gravity_source`` selects which base block feeds slot two of the
        # observation vector.  ``projected_gravity`` (default, every shipped
        # alpha policy) reads ``base_quat`` and rotates world ``-Z`` into the
        # base frame; ``raw_accel`` reads ``base_acc`` verbatim (older exports,
        # backlash variants).  Refuse any other spelling here rather than at
        # the builder seam so a mistyped metadata entry is caught once at
        # first-inference configuration - the builder does the same check
        # every tick, but this one names the SOURCE that supplied the value.
        if self._gravity_source is None:
            gravity_declared: str | None = meta.get("gravity_source")
            source = gravity_declared.strip() if gravity_declared else obs_builder.GRAVITY_SOURCE_PROJECTED
            if source not in obs_builder._GRAVITY_SOURCES:
                raise ValueError(
                    f"MicroduckPolicy: ONNX metadata gravity_source={gravity_declared!r} is not one "
                    f"of {list(obs_builder._GRAVITY_SOURCES)}. This value is a training-time "
                    f"flag baked into the export (Pollen's use_projected_gravity); the two "
                    f"branches read different base keys (base_quat vs base_acc), so a third "
                    f"spelling has no defined slot-two contract."
                )
            self._gravity_source = source

        if self._command is None:
            self._command = self._episode_start_command()

        if len(self._default_pose) != len(self._joint_names):
            raise ValueError(
                f"MicroduckPolicy: default_pose has {len(self._default_pose)} entries "
                f"but there are {len(self._joint_names)} joints."
            )
        self._configured = True

    def _episode_start_command(self) -> NDArray[np.float32]:
        """The command vector an episode starts from: the constructor's, else zeros.

        Single-sourced so the first episode (through ``_ensure_config``) and
        every later one (through :meth:`reset`) cannot drift apart, and so the
        width check lives in one place.

        Returns a COPY: ``_apply_command_kwargs`` writes the twist slots in
        place, and ``np.asarray(...).reshape(-1)`` on an already-``float32``
        array shares memory with ``_initial_command``, so handing it out
        directly would let one tick's ``target_velocity`` become the command
        every later episode restores to.

        Raises:
            ValueError: If a constructor-supplied command's width does not
                match the width ``command_names`` declares.
        """
        width = self._command_width()
        if self._initial_command is None:
            return np.zeros(width, dtype=np.float32)
        cmd = np.asarray(self._initial_command, dtype=np.float32).reshape(-1)
        if cmd.shape[0] != width:
            raise ValueError(
                f"MicroduckPolicy: initial command width {cmd.shape[0]} != "
                f"expected {width} (from command_names={self._command_names})."
            )
        return cmd.copy()

    def _command_width(self) -> int:
        """Command vector width the graph will accept.

        The graph's own declared input width is the authority when it declares
        one: it is a hard constraint, and ``command_names`` is not a width. The
        metadata names which command slots a skill READS, and Pollen's reference
        runner emits ONE unified 13-component command for every skill in a
        bundle, leaving the slots a skill ignores present and zero (the
        dead-weight rule this module's observation builder documents). Seven of
        the nine shipped Pollen exports declare a narrower ``command_names``
        than their graph consumes - ``roulade`` declares ``twist`` (3) against
        an ``obs`` input of 61 - so summing the names built a 51-wide vector for
        a 61-wide graph and onnxruntime refused it with ``Got: 51 Expected:
        61``, making those seven policies unrunnable.

        Falls back to the ``command_names`` sum, then the 13-D default, when the
        session declares no usable width (an injected stub, or a graph with a
        dynamic first-axis symbol rather than an integer).
        """
        declared = self._declared_command_width()
        if declared is not None:
            return declared
        if not self._command_names:
            return _DEFAULT_COMMAND_WIDTH
        return sum(_COMMAND_COMPONENTS.get(name, 0) for name in self._command_names)

    def _declared_command_width(self) -> int | None:
        """Command width implied by the graph's declared ``obs`` input, else ``None``.

        Returns ``None`` rather than raising whenever the width cannot be read as
        a positive integer - no declared shape, a dynamic-axis symbol, a shape
        that does not exceed the fixed blocks - so an injected stub keeps the
        ``command_names`` behaviour instead of being held to a shape it never
        declared.
        """
        if self._joint_names is None:
            return None
        get_inputs = getattr(self._session, "get_inputs", None)
        if not callable(get_inputs):
            return None
        try:
            shape = list(get_inputs()[0].shape)
            total = shape[-1]
        except Exception:  # noqa: BLE001 - a stub need not declare a shape
            return None
        if not isinstance(total, int) or isinstance(total, bool):
            return None
        fixed = _BASE_OBS_WIDTH + 3 * len(self._joint_names)
        width = total - fixed
        return width if width > 0 else None

    def _read_metadata(self) -> dict[str, str]:
        """Best-effort read of the ONNX ``custom_metadata_map`` (empty if absent)."""
        try:
            session = self._get_session()
            meta = session.get_modelmeta()
            return dict(getattr(meta, "custom_metadata_map", {}) or {})
        except Exception:  # noqa: BLE001 - metadata is optional; fall back to constants
            return {}

    def _get_session(self) -> MicroduckSession:
        """Return the cached session, building it from ``onnx_path`` on first call."""
        if self._session is None:
            assert self._onnx_path is not None  # validated in __init__
            self._session = self._build_onnx_session(self._onnx_path, self._providers)
        if self._input_name is None:
            self._input_name = self._detect_input_name(self._session)
        return self._session

    @staticmethod
    def _detect_input_name(session: MicroduckSession) -> str:
        """Read the graph's single input name (``"obs"`` on shipped exports)."""
        get_inputs = getattr(session, "get_inputs", None)
        if callable(get_inputs):
            try:
                return str(get_inputs()[0].name)
            except Exception:  # noqa: BLE001
                pass
        return "obs"

    @staticmethod
    def _build_onnx_session(onnx_path: Path, providers: list[str]) -> MicroduckSession:
        """Build a real onnxruntime session - deferred so tests need not import ORT."""
        ort = require_optional(
            "onnxruntime",
            extra="microduck",
            purpose="running the Microduck locomotion graph (pass `session=` to inject a stub)",
        )
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"Microduck ONNX policy not found: {onnx_path}. Shipped weights live in "
                "Pollen's microduck repo under policies/*.onnx (e.g. alpha_walking.onnx)."
            )
        sess = ort.InferenceSession(str(onnx_path), providers=providers)  # type: ignore[attr-defined]
        logger.info("Microduck ONNX session ready: %s (providers=%s)", onnx_path.name, sess.get_providers())
        return sess  # onnxruntime.InferenceSession satisfies MicroduckSession structurally
