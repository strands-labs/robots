"""KimodoPolicy - text-to-motion diffusion for the Unitree G1.

Clean-room :class:`Policy` provider wrapping NVIDIA's Kimodo
(``nvidia/Kimodo-G1-RP-v1``) text-conditioned motion diffusion model. Kimodo
takes a natural-language prompt (e.g. ``"walking forward with confident
strides"``) and samples per-frame full-body ``qpos`` sequences for the Unitree
G1 via a diffusion process. This policy adapts that sequence into the
per-tick action-dict contract the rest of ``strands_robots`` expects.

Where it sits in the stack (identical seat to
:class:`~strands_robots.policies.motionbricks.MotionBricksPolicy`):

* Kimodo emits **motion targets** (kinematic ``qpos``) for all 29 leg + waist +
  arm joints - a whole-body reference, not a joint subset.
* Standalone, calling the policy in a MuJoCo sim without a tracker sets those
  targets directly - the faithful visualisation of a kinematic generator, and
  what this provider supports today.
* Closing the loop through physics needs a controller that TRACKS this
  reference: the 29 targets are its input, so generator and tracker run in
  series over the same joints. That is a cascade, not a composition, and
  :class:`~strands_robots.policies.composite.CompositePolicy` does not express
  it - it merges two policies over DISJOINT joint groups (see its module
  docstring). Composing Kimodo with
  :class:`~strands_robots.policies.wbc.WBCPolicy` in particular cannot track a
  reference at all: WBC's only command input is a target base velocity, it has
  no reference-pose input, and it drives 15 of the 29 joints Kimodo already
  drives.

Contract:

* ``requires_images = False`` - the sampler is driven by a text prompt,
  never a camera frame.
* ``get_actions(obs, instruction, **kwargs)`` reads the prompt from the
  well-known keys (``instruction`` / ``text_prompt``) and sampling knobs from
  ``kwargs`` (``diffusion_steps``, ``guidance_scale``, ``seed``). The instance
  synthesises ONCE on first call (or when the prompt changes) then streams
  per-frame targets on subsequent calls, one frame per call, synchronously.
* Changing the prompt mid-rollout is how a long-horizon sequence is driven:
  each new prompt samples the next segment and the stream continues. A fresh
  sample starts at its own canonical start pose, unrelated to wherever the
  previous segment left the robot, so the segment is eased off the pose last
  commanded over ``config.transition_frames`` native frames - the same
  transition length Kimodo's own sampler applies to a multi-prompt sequence.
  Without it the seam commands every joint to step at once, which is not a
  reference any tracker can follow.
* Output is a dict of joint-name -> target angle for the G1's 29 leg + waist
  + arm joints, keyed by :data:`KIMODO_G1_JOINTS` (the canonical WBC ordering)
  so the tracker sees identical names.

Model injection seam: pass a ``motion_agent`` implementing
:class:`KimodoMotionAgent` to unit-test the frame -> action-dict mapping
WITHOUT diffusers, weights, or CUDA. A missing install or checkpoint raises
``RuntimeError`` with an install hint - no silent fallback (AGENTS.md #5/#6).

Requires the ``[kimodo]`` extra; weights are fetched on demand from
HuggingFace under the NVIDIA Open Model License.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from strands_robots.policies.base import Policy
from strands_robots.policies.wbc.policy import WBC_G1_ALL_JOINTS

from .config import KimodoConfig

logger = logging.getLogger(__name__)


# The 29 leg+waist+arm joints Kimodo drives, in the sampler's ``qpos[7:]``
# order. Verified identical to the canonical WBC/MotionBricks ordering so a
# Kimodo reference and a WBC/PD tracker name the same joints - they compose
# without a remapping table.
KIMODO_G1_JOINTS: tuple[str, ...] = WBC_G1_ALL_JOINTS


# qpos layout: [root_pos(3), root_quat(4), joint_angles(njoints)]. First 7
# entries are the free-flying base (not actuator targets).
_NUM_ROOT_QPOS = 7


@runtime_checkable
class KimodoMotionAgent(Protocol):
    """Injection seam for the Kimodo sampler (``motion_agent=`` arg).

    The real agent (built from diffusers weights) and unit-test stubs both
    satisfy this protocol, so the policy's mapping logic is testable without
    torch/diffusers/CUDA/checkpoints.
    """

    def sample(
        self,
        prompt: str,
        num_frames: int,
        diffusion_steps: int,
        guidance_scale: float,
        seed: int | None,
    ) -> NDArray[np.float32]:
        """Sample a motion. Returns ``(num_frames, 7+29)`` float32 qpos."""
        ...


def _slerp_quat(
    q0: NDArray[np.float32],
    q1: NDArray[np.float32],
    t: float | np.floating[Any],
) -> NDArray[np.float32]:
    """Spherically interpolate between two wxyz quaternions.

    Shared by the native -> tracker upsample and the segment-transition ease so
    both walk the same arc: a quaternion has two representations for one
    rotation, and picking the shorter arc (the sign flip below) is what stops an
    interpolation from taking the long way round.

    Args:
        q0: Start quaternion, wxyz. Assumed unit-norm.
        q1: End quaternion, wxyz. Assumed unit-norm.
        t: Interpolation weight, 0.0 returns ``q0`` and 1.0 returns ``q1``. A
            numpy scalar is accepted as-is rather than widened to a Python
            float, so the caller's precision decides the arithmetic precision.

    Returns:
        The interpolated unit quaternion, wxyz.
    """
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        # Nearly parallel - the arc is shorter than float precision resolves,
        # so lerp+normalise avoids dividing by a vanishing sin(theta).
        q = q0 + t * (q1 - q0)
    else:
        theta_0 = float(np.arccos(dot))
        sin_theta_0 = float(np.sin(theta_0))
        theta = theta_0 * t
        s0 = float(np.sin(theta_0 - theta)) / sin_theta_0
        s1 = float(np.sin(theta)) / sin_theta_0
        q = s0 * q0 + s1 * q1
    return q / max(float(np.linalg.norm(q)), 1e-8)  # type: ignore[no-any-return]


def _ease_onto_previous_pose(
    qpos: NDArray[np.float32],
    previous_pose: NDArray[np.float32],
    frames: int,
) -> NDArray[np.float32]:
    """Ease a freshly sampled motion off the pose that was last commanded.

    Kimodo samples every motion from its own canonical start pose, so the first
    frame of a new segment bears no relation to the last frame of the previous
    one. Emitting it unmodified steps every joint in a single tick. This decays
    that pose offset to zero across the first ``frames`` frames, leaving the
    rest of the segment untouched: the motion keeps its own shape and velocity
    and only its starting offset is absorbed.

    The offset decays by ``1 / (frames + 1)`` per frame, so the residual step at
    the seam is the full offset divided by ``frames + 1`` rather than the offset
    itself. The first frame is deliberately not pinned exactly onto
    ``previous_pose``: that would command zero motion for one frame, trading a
    position step for a velocity stall.

    Args:
        qpos: The freshly sampled ``(n, 7 + njoints)`` motion, native rate.
        previous_pose: The last commanded ``(7 + njoints,)`` qpos frame.
        frames: Number of frames to spread the offset over. Clamped to the
            length of ``qpos``.

    Returns:
        A new array of the same shape, with the leading frames eased. ``qpos``
        is not modified.
    """
    count = int(min(frames, qpos.shape[0]))
    if count <= 0:
        return qpos
    out = qpos.copy()
    # Root orientation interpolates on the quaternion arc; root position and
    # joint angles are ordinary linear channels.
    linear = list(range(3)) + list(range(_NUM_ROOT_QPOS, qpos.shape[1]))
    offset = previous_pose[linear] - qpos[0, linear]
    q_prev = previous_pose[3:7]
    q_prev = q_prev / max(float(np.linalg.norm(q_prev)), 1e-8)
    for i in range(count):
        weight = 1.0 - (i + 1) / (count + 1)
        out[i, linear] = qpos[i, linear] + weight * offset
        q_i = qpos[i, 3:7]
        q_i = q_i / max(float(np.linalg.norm(q_i)), 1e-8)
        out[i, 3:7] = _slerp_quat(q_i, q_prev, weight)
    return out


def _slerp_upsample(
    qpos: NDArray[np.float32],
    native_fps: int,
    target_fps: int,
) -> NDArray[np.float32]:
    """Upsample a qpos sequence from ``native_fps`` to ``target_fps``.

    Linear interpolation on joint angles, SLERP on the root quaternion (indices
    3-6). Position (indices 0-2) is linearly interpolated. Kimodo native is
    30 Hz and the G1 tracker consumes at 50 Hz, so this widens 120 frames ->
    ~200 frames without introducing joint discontinuities.
    """
    if target_fps == native_fps:
        return qpos
    n_in = qpos.shape[0]
    duration_s = (n_in - 1) / native_fps if n_in > 1 else 0.0
    n_out = max(1, int(round(duration_s * target_fps)) + 1)
    t_in = np.linspace(0.0, 1.0, n_in, dtype=np.float32)
    t_out = np.linspace(0.0, 1.0, n_out, dtype=np.float32)
    out = np.zeros((n_out, qpos.shape[1]), dtype=np.float32)
    # Position + joints - linear.
    for j in list(range(3)) + list(range(7, qpos.shape[1])):
        out[:, j] = np.interp(t_out, t_in, qpos[:, j])
    # Quaternion - SLERP. Vectorised across output frames.
    q_in = qpos[:, 3:7]
    # Normalise safety.
    q_in = q_in / np.linalg.norm(q_in, axis=1, keepdims=True).clip(min=1e-8)
    for i, t in enumerate(t_out):
        idx = np.clip(np.searchsorted(t_in, t) - 1, 0, n_in - 2)
        alpha = (t - t_in[idx]) / max(t_in[idx + 1] - t_in[idx], 1e-8)
        out[i, 3:7] = _slerp_quat(q_in[idx], q_in[idx + 1], alpha)
    return out


class KimodoPolicy(Policy):
    """Text-to-motion diffusion policy for the Unitree G1.

    Args:
        config: A :class:`KimodoConfig` instance. Constructed via
            ``KimodoConfig()`` for defaults or ``KimodoConfig.from_dict(...)``
            from user config.
        motion_agent: Optional :class:`KimodoMotionAgent` (injection seam for
            tests). If ``None``, the real diffusers-backed agent is lazily
            constructed on first call - which requires the ``[kimodo]`` extra
            plus a working CUDA runtime.
        **kwargs: Passed to :class:`Policy` base class (e.g. ``robot_name``).

    Example (in a MuJoCo sim):

        >>> from strands_robots import Robot
        >>> sim = Robot("g1", mesh=False)
        >>> sim.run_policy(
        ...     robot_name="g1",
        ...     policy_provider="kimodo",
        ...     policy_config={
        ...         "diffusion_steps": 100,
        ...         "guidance_scale": 7.5,
        ...         "num_frames": 120,
        ...     },
        ...     instruction="a person walking forward with confident strides",
        ...     n_steps=200,
        ...     control_frequency=50,
        ... )
    """

    #: This policy does not consume image observations - text prompt drives it.
    requires_images: bool = False

    def __init__(
        self,
        config: KimodoConfig | dict[str, Any] | None = None,
        *,
        motion_agent: KimodoMotionAgent | None = None,
        model_id: str | None = None,
        diffusion_steps: int | None = None,
        guidance_scale: float | None = None,
        num_frames: int | None = None,
        transition_frames: int | None = None,
        native_fps: int | None = None,
        tracker_fps: int | None = None,
        device: str | None = None,
        dtype: str | None = None,
        seed: int | None = None,
    ) -> None:
        """Build the policy from a config object and/or per-field overrides.

        Every field the registry advertises in ``config_keys`` is an explicit
        parameter here, so ``create_policy("kimodo", diffusion_steps=25)``
        configures the sampler instead of failing on an unexpected keyword.
        There is deliberately no ``**kwargs``: an unknown knob raises
        ``TypeError`` at construction rather than being swallowed by a
        parameter nothing reads.

        Args:
            config: A :class:`KimodoConfig`, a plain dict of its fields, or
                ``None`` for the dataclass defaults.
            motion_agent: Injected sampler, for driving the frame -> action-dict
                mapping without diffusers, weights, or CUDA.
            model_id: HuggingFace model id override.
            diffusion_steps: Denoising steps override.
            guidance_scale: Classifier-free-guidance weight override.
            num_frames: Motion length override, in native frames.
            transition_frames: Override for the number of native frames a newly
                sampled segment is eased off the last commanded pose over.
            native_fps: Sampler output rate override.
            tracker_fps: Tracker consumption rate override.
            device: torch device string override.
            dtype: Sampler dtype override (``"fp16"``/``"bf16"``/``"fp32"``).
            seed: Sampling seed override.

        Raises:
            ValueError: If a resolved field is outside its domain, as validated
                by :class:`KimodoConfig`.
        """
        self.config: KimodoConfig = self._resolve_config(
            config,
            model_id=model_id,
            diffusion_steps=diffusion_steps,
            guidance_scale=guidance_scale,
            num_frames=num_frames,
            transition_frames=transition_frames,
            native_fps=native_fps,
            tracker_fps=tracker_fps,
            device=device,
            dtype=dtype,
            seed=seed,
        )
        self._motion_agent: KimodoMotionAgent | None = motion_agent

        # Streaming state - filled by _synthesise() and drained by get_actions().
        # The buffer caches ONE sampler run; _buffer_key records the inputs that
        # produced it so a changed prompt/knob/seed re-enters the sampler instead
        # of draining a buffer those inputs never generated.
        self._buffer_key: tuple[str, int, float, int | None] | None = None
        self._motion_buffer: NDArray[np.float32] | None = None
        self._frame_cursor: int = 0
        # The last qpos frame handed out. A re-sample eases onto it so a new
        # segment continues from where the robot actually is; ``None`` means
        # nothing has been commanded yet, so there is no seam to ease.
        self._last_frame: NDArray[np.float32] | None = None
        # The sampler's own output, kept so the buffer can be rebuilt without
        # easing (and without re-sampling) when an episode boundary means the
        # transition it was built for no longer applies.
        self._raw_sample: NDArray[np.float32] | None = None
        self._buffer_eased: bool = False
        self._joint_names: tuple[str, ...] = KIMODO_G1_JOINTS
        self._robot_state_keys: list[str] = list(KIMODO_G1_JOINTS)

    # -----------------------------------------------------------------
    # Policy interface (abstract members from base.Policy)
    # -----------------------------------------------------------------
    @property
    def provider_name(self) -> str:
        """Registry key for this provider (``"kimodo"``)."""
        return "kimodo"

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Resolve the G1's 29 leg+waist+arm joints BY NAME in the robot's key list.

        Kimodo emits ``qpos[7:]`` in :data:`KIMODO_G1_JOINTS` order; we locate
        each of those names inside ``robot_state_keys`` rather than assuming a
        fixed position (the sim may prepend the free floating-base joint and
        namespace joints). Resolving by name means the action-dict keys match
        the robot's actuators regardless of ordering.

        Raises:
            ValueError: If any expected G1 joint name is absent.
        """
        keys = list(robot_state_keys)
        key_set = set(keys)
        missing = [name for name in KIMODO_G1_JOINTS if name not in key_set]
        if missing:
            raise ValueError(
                "KimodoPolicy: the robot's joint list is missing expected G1 joints: "
                f"{missing}.\n"
                f"  expected (qpos[7:] order): {list(KIMODO_G1_JOINTS)}\n"
                f"  robot provided:            {keys}\n"
                "Kimodo drives the 29-DOF G1; load the full unitree_g1 model."
            )
        self._robot_state_keys = keys
        self._joint_names = tuple(KIMODO_G1_JOINTS)

    async def get_actions(
        self,
        observation_dict: dict[str, Any],
        instruction: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return the next per-frame G1 joint targets for the current prompt.

        Args:
            observation_dict: Sim observation (ignored - Kimodo is open-loop
                kinematic).
            instruction: Text prompt for motion synthesis. If it changes across
                calls, the sampler re-runs and the frame cursor resets.
            **kwargs: Optional overrides for this call:
                ``text_prompt`` (alias for ``instruction``),
                ``diffusion_steps``, ``guidance_scale``, ``seed``. Each is an
                input to the sampler, so supplying one that differs from the
                value that produced the buffered motion re-samples; supplying
                the same values drains the existing buffer rather than paying
                for a run that would return identical frames.

        Returns:
            A single-element list containing a dict mapping each of
            :data:`KIMODO_G1_JOINTS` to its target angle (float, radians).
            The single-element list matches the base ``Policy.get_actions``
            contract (chunked policies return more than one). At the end of the
            sampled buffer the last frame is held (tracker keeps servoing).
        """
        prompt = kwargs.get("text_prompt") or instruction
        if not prompt or not prompt.strip():
            raise ValueError(
                "KimodoPolicy requires a non-empty 'instruction' or 'text_prompt' "
                "(natural-language motion description)."
            )

        # (Re)sample on the first call, or whenever any input that determines
        # the motion differs from the one that produced the buffer we hold.
        diffusion_steps = int(kwargs.get("diffusion_steps", self.config.diffusion_steps))
        guidance_scale = float(kwargs.get("guidance_scale", self.config.guidance_scale))
        seed = kwargs.get("seed", self.config.seed)
        key = self._sample_key(prompt, diffusion_steps, guidance_scale, seed)
        if self._motion_buffer is None or key != self._buffer_key:
            self._synthesise(
                prompt=prompt,
                diffusion_steps=diffusion_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )

        assert self._motion_buffer is not None  # narrowed by _synthesise
        idx = min(self._frame_cursor, self._motion_buffer.shape[0] - 1)
        frame = self._motion_buffer[idx]
        self._frame_cursor += 1

        # Record the emitted pose before returning: it is the seam a later
        # re-sample eases onto.
        self._last_frame = frame.copy()

        joint_angles = frame[_NUM_ROOT_QPOS:]
        action = {name: float(v) for name, v in zip(self._joint_names, joint_angles)}
        return [action]

    def reset(self, seed: int | None = None) -> None:
        """Rewind to the first frame, and re-seed the sampler when given a seed.

        ``PolicyRunner.evaluate`` derives a distinct seed per episode and
        forwards it here so a stochastic policy samples afresh each episode
        while staying reproducible across re-runs at the same master seed. A
        diffusion sample IS this policy's per-episode state, so a new seed has
        to reach the sampler: it is recorded on the config and the next
        :meth:`get_actions` re-samples, because the seed is part of the key
        identifying the buffered motion.

        Rewinding also forgets the last commanded pose: episodes are
        independent, so the next segment opens at the motion's own start pose
        rather than being eased onto where the previous episode finished.

        Args:
            seed: Sampling seed for the next episode. ``None`` rewinds only,
                replaying the motion already held. A seed equal to the one that
                produced the current buffer also replays it rather than
                re-running the sampler for identical frames.
        """
        self._frame_cursor = 0
        # A new episode starts the robot afresh, so there is no previously
        # commanded pose to stay continuous with. Forgetting the pose is not
        # enough on its own: a buffer already eased onto it would still be
        # replayed from frame 0, opening the new episode on a transition built
        # for the previous one. Rebuilding from the held sample undoes that
        # without paying for another diffusion run.
        self._last_frame = None
        if self._buffer_eased and self._raw_sample is not None:
            self._rebuild_buffer(None)
        if seed is not None:
            # Store on config-shadow so next _synthesise picks it up.
            object.__setattr__(self.config, "seed", seed)

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------
    @staticmethod
    def _sample_key(
        prompt: str,
        diffusion_steps: int,
        guidance_scale: float,
        seed: int | None,
    ) -> tuple[str, int, float, int | None]:
        """Identify a sampler run by every input that determines its output.

        The buffered motion is a cache of one sampler run, and this is its cache
        key. Comparing the full key - rather than the prompt alone - is what
        makes ``diffusion_steps`` / ``guidance_scale`` / ``seed`` behave as the
        per-call overrides :meth:`get_actions` documents, and what lets
        :meth:`reset` re-seed an episode.

        Both the producer (:meth:`_synthesise`) and the consumer
        (:meth:`get_actions`) build the key here, so the two cannot disagree
        about which inputs identify a motion.

        Args:
            prompt: Text prompt the motion was sampled for.
            diffusion_steps: Denoising steps used.
            guidance_scale: Classifier-free-guidance weight used.
            seed: Sampling seed, or ``None`` for an unseeded sample.

        Returns:
            A hashable tuple identifying the run.
        """
        return (prompt, diffusion_steps, guidance_scale, None if seed is None else int(seed))

    def _synthesise(
        self,
        prompt: str,
        diffusion_steps: int,
        guidance_scale: float,
        seed: int | None,
    ) -> None:
        """Sample a fresh motion buffer for ``prompt`` and reset the cursor."""
        agent = self._motion_agent or self._build_real_agent()
        raw = agent.sample(
            prompt=prompt,
            num_frames=self.config.num_frames,
            diffusion_steps=diffusion_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        if raw.ndim != 2 or raw.shape[1] != _NUM_ROOT_QPOS + len(KIMODO_G1_JOINTS):
            raise RuntimeError(
                f"Kimodo agent returned qpos with shape {raw.shape}, expected "
                f"(*, {_NUM_ROOT_QPOS + len(KIMODO_G1_JOINTS)})"
            )
        self._raw_sample = raw.astype(np.float32)
        self._rebuild_buffer(self._last_frame)
        self._buffer_key = self._sample_key(prompt, diffusion_steps, guidance_scale, seed)
        assert self._motion_buffer is not None  # narrowed by _rebuild_buffer
        # The prompt is caller-supplied, so it is identified by digest rather
        # than echoed: interpolating the text would let a prompt containing a
        # newline forge an additional log record. The digest still correlates
        # repeated samples of the same prompt across a run.
        logger.info(
            "Kimodo: sampled %d frames @ %dHz for prompt sha256:%s (%d chars)",
            self._motion_buffer.shape[0],
            self.config.tracker_fps,
            hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12],
            len(prompt),
        )

    def _rebuild_buffer(self, previous_pose: NDArray[np.float32] | None) -> None:
        """Build the streaming buffer from the held sample and rewind the cursor.

        Separated from :meth:`_synthesise` because the buffer depends on two
        things - the sample and the pose it must continue from - and only the
        first costs a diffusion run. An episode boundary changes the second, so
        it rebuilds here rather than re-sampling.

        Args:
            previous_pose: The pose the segment must continue from, or ``None``
                to emit the sample as the sampler produced it.
        """
        assert self._raw_sample is not None  # only called with a held sample
        sampled = self._raw_sample
        # Ease at the native rate so ``transition_frames`` counts the same
        # frames Kimodo's own ``num_transition_frames`` does, and before the
        # upsample so the interpolation smooths the eased frames with the rest.
        if previous_pose is not None:
            sampled = _ease_onto_previous_pose(
                sampled,
                previous_pose,
                self.config.transition_frames,
            )
        self._buffer_eased = previous_pose is not None
        # Upsample native 30Hz -> tracker 50Hz for smooth control.
        self._motion_buffer = _slerp_upsample(
            sampled,
            native_fps=self.config.native_fps,
            target_fps=self.config.tracker_fps,
        )
        self._frame_cursor = 0

    @staticmethod
    def _resolve_config(
        config: KimodoConfig | dict[str, Any] | None,
        **overrides: Any,
    ) -> KimodoConfig:
        """Merge a base config with per-field overrides, explicit values winning.

        Precedence is override > ``config`` field > dataclass default. The merge
        goes back through :class:`KimodoConfig` rather than
        ``dataclasses.replace`` so ``__post_init__`` re-validates the merged
        result: an override is checked exactly like a directly-constructed
        field instead of bypassing the domain.

        Args:
            config: Base config, a dict of its fields, or ``None`` for defaults.
            **overrides: Per-field values, where ``None`` means "not supplied".

        Returns:
            The merged, validated config.
        """
        if config is None:
            base: dict[str, Any] = {}
        elif isinstance(config, dict):
            base = dict(config)
        else:
            base = {name: getattr(config, name) for name in config.__dataclass_fields__}
        base.update({key: value for key, value in overrides.items() if value is not None})
        return KimodoConfig.from_dict(base)

    def _build_real_agent(self) -> KimodoMotionAgent:
        """Lazy-construct the real diffusers-backed sampler agent.

        Kept out of ``__init__`` so unit tests can construct the policy with an
        injected ``motion_agent`` without needing torch/diffusers/CUDA on the
        import path.
        """
        try:
            from ._diffusers_agent import DiffusersKimodoAgent  # local, optional
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "KimodoPolicy requires the '[kimodo]' extra. Install with:\n"
                "  pip install 'strands-robots[kimodo]'\n"
                "or provide a custom motion_agent= for offline testing."
            ) from exc
        agent = DiffusersKimodoAgent(self.config)
        self._motion_agent = agent
        return agent
