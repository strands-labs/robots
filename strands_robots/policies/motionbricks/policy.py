"""MotionBricksPolicy - generative kinematic motion for the Unitree G1.

A clean-room :class:`Policy` provider wrapping NVIDIA's MotionBricks generative
motion model (the ``motionbricks/`` subproject of
`GR00T-WholeBodyControl <https://github.com/NVlabs/GR00T-WholeBodyControl>`_).
MotionBricks is a *kinematic motion generator*: given a **style** (a clip mode
such as ``walk`` / ``stealth_walk`` / ``walk_boxing``) and a movement/facing
command it synthesises per-frame full-body ``qpos`` for the G1, faster than
real time. Upstream reference runner:
``motionbricks/scripts/interactive_demo_g1.py``.

Where it sits in the stack: MotionBricks emits the per-frame **motion targets**
for all 29 leg + waist + arm joints - a whole-body reference. Standalone, that
output is a kinematic reference (set ``qpos`` + forward kinematics), which is the
faithful way to visualise a kinematic generator and what this provider supports
today. Closing the loop through physics needs a controller that TRACKS the
reference, taking the 29 targets as its input: generator and tracker run in
series over the same joints. That is a cascade, not a composition, so
:class:`~strands_robots.policies.composite.CompositePolicy` (which merges
DISJOINT joint groups) does not express it, and
:class:`~strands_robots.policies.wbc.WBCPolicy` cannot serve as the tracker -
its only command input is a target base velocity, with no reference-pose input.

This is a non-VLA member of the policy family (like ``wbc`` / cuRobo / MoveIt2):

* ``requires_images = False`` - the generator is driven by a style + direction
  command, never camera frames.
* ``get_actions`` reads the goal from the well-known ``**kwargs`` keys
  (``style`` / ``mode``, ``target_velocity``, ``target_heading``) rather than a
  natural-language instruction.
* The output is a single per-tick action dict mapping the G1's 29 leg+waist+arm
  joint names (:data:`MOTIONBRICKS_G1_JOINTS`, reusing the canonical WBC
  ordering) to their target angles. Synthesis advances **one frame per call,
  synchronously - no threads** (issue note): at >1kHz-equivalent synthesis the
  per-call cost is small, and overlapping is the tracking layer's job.

Model injection seam (the analogue of WBC's ``allow_missing_models``): pass a
``motion_agent`` implementing :class:`MotionAgent` to unit-test the
frame -> action-dict mapping WITHOUT checkpoints, GPU, or the ``motionbricks``
install. A missing install or checkpoint raises ``RuntimeError`` with an install
hint - there is no silent fallback (AGENTS.md #5/#6).

Requires the ``[motionbricks]`` extra and the upstream checkpoints (git-LFS,
NVIDIA Open Model License); no weights are bundled.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from strands_robots.policies.base import Policy
from strands_robots.policies.wbc.policy import WBC_G1_ALL_JOINTS
from strands_robots.utils import require_optional

from .config import MotionBricksConfig
from .observation import build_control_signals, resolve_locomotion_style, resolve_mode

logger = logging.getLogger(__name__)

# The 29 leg+waist+arm joints MotionBricks drives, in the generator's
# ``qpos[7:]`` order. Verified IDENTICAL to the upstream G1 model joint order
# (floating_base_joint then this sequence), so reusing the canonical WBC
# ordering means a MotionBricks reference and a WBC tracker name the same joints
# - they compose without a remapping table.
MOTIONBRICKS_G1_JOINTS: tuple[str, ...] = WBC_G1_ALL_JOINTS

# qpos layout: [root_pos(3), root_quat(4), joint_angles(njoints)]. The first 7
# entries are the free floating base (not actuator targets).
_NUM_ROOT_QPOS = 7


@runtime_checkable
class MotionAgent(Protocol):
    """Injection seam for the MotionBricks generator (the ``motion_agent=`` arg).

    The real agent (:class:`_MotionBricksAgentAdapter`, built from checkpoints)
    and the unit-test stub both satisfy this protocol, so the
    frame -> action-dict mapping is testable without GPU / checkpoints.

    Attributes:
        clip_keys: Ordered clip mode names (the generator's ``CLIPS`` keys).
        clip_token_specs: Per-mode explicit ``allowed_pred_num_tokens`` masks
            (``None`` where a mode declares none), indexed by mode.
        min_token: Generator minimum token count.
        max_token: Generator maximum token count.
    """

    clip_keys: list[str]
    clip_token_specs: list[list[int] | None]
    min_token: int
    max_token: int

    def reset(self) -> None:
        """Reset the generator to its idle warm-start (episode boundary)."""
        ...

    def next_qpos(self, control_signals: dict[str, Any], controller_dt: float) -> NDArray[np.float64]:
        """Return the current full-body ``qpos`` and queue the next generation.

        Mirrors the upstream demo loop: read the current frame's ``qpos`` via
        ``get_next_frame()``, then advance the generator with the supplied
        control signals + the motion context.

        Args:
            control_signals: Plain (torch-free) control dict from
                :func:`~strands_robots.policies.motionbricks.observation.build_control_signals`.
            controller_dt: Per-regeneration integration horizon.

        Returns:
            The full-body ``qpos`` (``[root_pos(3), root_quat(4), joints]``).
        """
        ...


class MotionBricksPolicy(Policy):
    """Generative kinematic motion policy for the Unitree G1.

    Args:
        config: A :class:`MotionBricksConfig`, a path to a config JSON, a dict,
            or ``None``. Required to build the real generator; may be ``None``
            when ``motion_agent`` is injected (unit tests).
        motion_agent: Optional :class:`MotionAgent` to use instead of building
            the real generator from checkpoints (the test/CI seam). When
            ``None``, the generator is built from ``config`` (needs the
            ``[motionbricks]`` extra + checkpoints).
        result_dir: Convenience shortcut for ``config`` - when ``config`` is
            ``None`` and ``result_dir`` is given, a :class:`MotionBricksConfig`
            is built from it (plus ``style`` / ``device``).
        style: Default motion style (mode index or name) used when a call does
            not pass ``style`` / ``mode``. Overrides ``config.style``.
        target_velocity: Optional default movement direction ``[vx, vy]`` used
            when a call passes none.
        device: Torch device for the generator (when building from ``config``).
        style_map: Optional overrides merged over the built-in
            ``locomotion_style`` -> clip-name map
            (:data:`~strands_robots.policies.motionbricks.observation.LOCOMOTION_STYLE_TO_G1_CLIP`).
            Used to translate a caller-supplied ``locomotion_style`` goal kwarg
            to this generator's clip set; merged over (and taking precedence
            over) any ``config.style_map``.

    Raises:
        ValueError: If neither ``config``/``result_dir`` nor ``motion_agent`` is
            provided.
        RuntimeError: If building the real generator fails because the
            ``[motionbricks]`` extra or the checkpoints are missing.
    """

    def __init__(
        self,
        config: str | dict[str, Any] | MotionBricksConfig | None = None,
        *,
        motion_agent: MotionAgent | None = None,
        result_dir: str | None = None,
        style: int | str | None = None,
        target_velocity: list[float] | None = None,
        device: str | None = None,
        style_map: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._robot_state_keys: list[str] = []
        self._config = self._resolve_config(config, result_dir, style, device)

        # Default goal: style (per-call kwarg wins) and an optional standing
        # movement direction.
        self._default_style: int | str = (
            style if style is not None else (self._config.style if self._config is not None else "walk")
        )
        self._default_velocity = list(target_velocity) if target_velocity is not None else None

        # locomotion_style -> clip-name overrides: config.style_map is the base,
        # the constructor arg takes precedence; both overlay the built-in
        # LOCOMOTION_STYLE_TO_G1_CLIP defaults inside resolve_locomotion_style.
        merged_style_map: dict[str, str] = {}
        if self._config is not None and self._config.style_map:
            merged_style_map.update(self._config.style_map)
        if style_map:
            merged_style_map.update(style_map)
        self._style_map: dict[str, str] | None = merged_style_map or None

        if motion_agent is not None:
            self._agent: MotionAgent = motion_agent
        else:
            if self._config is None:
                raise ValueError(
                    "MotionBricksPolicy needs either a config/result_dir (to build the generator) "
                    "or a motion_agent (the injection seam). Pass result_dir=<path to out/> or "
                    "config={'result_dir': ...}, or inject motion_agent=<stub> for tests."
                )
            self._agent = self._build_agent(self._config)

        # The integration horizon per regeneration: from config when present,
        # else the upstream default ((8/30)*2.0).
        self._controller_dt = self._config.controller_dt if self._config is not None else (8 / 30.0) * 2.0

        # The 29 joints we emit, in qpos[7:] order (== WBC_G1_ALL_JOINTS).
        # set_robot_state_keys resolves them by NAME within the robot's joint
        # list; until then default to the canonical ordering so direct
        # get_actions calls still emit the right keys.
        self._joint_names: list[str] = list(MOTIONBRICKS_G1_JOINTS)

        # Full body qpos ([root(7), joints]) from the latest synthesis frame,
        # exposed via `last_qpos` for kinematic visualisation (the root pose is
        # not an actuator target, so it is not in the action dict).
        self._last_qpos: NDArray[np.float64] | None = None

        if kwargs:
            logger.debug("MotionBricksPolicy ignoring unknown constructor kwargs: %s", sorted(kwargs.keys()))

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_config(
        config: str | dict[str, Any] | MotionBricksConfig | None,
        result_dir: str | None,
        style: int | str | None,
        device: str | None,
    ) -> MotionBricksConfig | None:
        """Normalise the various config inputs into a :class:`MotionBricksConfig` or ``None``."""
        if isinstance(config, MotionBricksConfig):
            return config
        if isinstance(config, dict):
            return MotionBricksConfig.from_dict(config)
        if isinstance(config, str):
            return MotionBricksConfig.from_file(config)
        if config is None and result_dir is not None:
            overrides: dict[str, Any] = {"result_dir": result_dir}
            if style is not None:
                overrides["style"] = style
            if device is not None:
                overrides["device"] = device
            return MotionBricksConfig.from_dict(overrides)
        if config is None:
            return None
        raise ValueError(
            f"Unsupported config type {type(config).__name__}; pass a MotionBricksConfig, dict, path, or None"
        )

    def _build_agent(self, config: MotionBricksConfig) -> MotionAgent:
        """Build the real MotionBricks generator from checkpoints (heavy path).

        Raises:
            RuntimeError: If the ``[motionbricks]`` extra is not installed or
                the checkpoint tree is missing/incomplete.
        """
        require_optional(
            "motionbricks",
            pip_install="-e <GR00T-WholeBodyControl>/motionbricks",
            extra="motionbricks",
            purpose="MotionBricks generative motion synthesis",
        )
        result_dir = Path(config.result_dir).expanduser().resolve()
        if not result_dir.is_dir():
            raise RuntimeError(
                f"MotionBricks result_dir not found: {result_dir}\n"
                "Point it at the upstream 'out/' checkpoint tree (fetch with git-LFS):\n"
                '  git lfs pull --include="motionbricks/out/**" --exclude=""'
            )
        return _MotionBricksAgentAdapter.build(config, result_dir)

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------
    @property
    def provider_name(self) -> str:
        """Registry key for this provider (``"motionbricks"``)."""
        return "motionbricks"

    @property
    def requires_images(self) -> bool:
        """MotionBricks is driven by a style + direction command, never images."""
        return False

    @property
    def config(self) -> MotionBricksConfig | None:
        """Resolved :class:`MotionBricksConfig`, or ``None`` when unconfigured."""
        return self._config

    @property
    def last_qpos(self) -> NDArray[np.float64] | None:
        """Full-body ``qpos`` ([root_pos(3), root_quat(4), joints]) of the latest frame.

        ``None`` before the first :meth:`get_actions` call. Unlike the action
        dict (which carries only the 29 actuator joint targets), this includes
        the free floating-base pose, so a kinematic visualiser can place the
        root each frame and render the synthesised motion faithfully (the
        upstream demo's "set qpos + forward kinematics" loop). It is a copy, so
        mutating it does not perturb the policy.
        """
        return None if self._last_qpos is None else self._last_qpos.copy()

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Resolve the G1's 29 leg+waist+arm joints BY NAME within the robot's key list.

        MotionBricks emits ``qpos[7:]`` in :data:`MOTIONBRICKS_G1_JOINTS` order;
        we locate each of those names inside ``robot_state_keys`` rather than
        assuming a fixed position (the sim prepends the free floating-base joint
        and may namespace joints). Resolving by name means the action dict keys
        match the robot's joints regardless of ordering.

        Raises:
            ValueError: If any expected G1 joint name is absent - a mismatch
                that would otherwise drive the wrong joints. The message lists
                the missing names.
        """
        keys = list(robot_state_keys)
        key_set = set(keys)
        missing = [name for name in MOTIONBRICKS_G1_JOINTS if name not in key_set]
        if missing:
            raise ValueError(
                "MotionBricksPolicy: the robot's joint list is missing expected G1 joints: "
                f"{missing}.\n"
                f"  expected (qpos[7:] order): {list(MOTIONBRICKS_G1_JOINTS)}\n"
                f"  robot provided:            {keys}\n"
                "MotionBricks drives the 29-DOF G1; load the full unitree_g1 model "
                "(its joints carry these exact names)."
            )
        self._robot_state_keys = keys
        self._joint_names = list(MOTIONBRICKS_G1_JOINTS)

    def reset(self, seed: int | None = None) -> None:
        """Reset the generator to its idle warm-start.

        The ``seed`` is accepted for API parity with the policy family; the
        generator manages its own RNG, so it is logged but not threaded into the
        upstream agent (which seeds per-clip internally).
        """
        self._agent.reset()
        logger.debug("MotionBricksPolicy.reset: generator re-seeded to idle warm-start (seed=%r)", seed)

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Synthesise one motion frame and return the 29-dim target joint angles.

        Reads the goal from the well-known kwargs (``style`` / ``mode``,
        ``target_velocity``, ``target_heading``); ``instruction`` is ignored.
        Advances the generator exactly one frame **synchronously** and returns a
        single per-step action dict keyed by G1 joint name.

        Returns a one-element list (MotionBricks emits one frame per tick, not a
        chunked plan): the runner re-queries every control step.
        """
        # Style precedence: an explicit style=/mode= kwarg pins the clip (direct
        # callers). Otherwise a caller-supplied locomotion_style goal kwarg is
        # translated to this generator's clip set, so the goal channel steers
        # the gait. Falls back to the configured default style.
        explicit_style = kwargs.get("style", kwargs.get("mode"))
        if explicit_style is not None:
            style: int | str = explicit_style
        elif kwargs.get("locomotion_style") is not None:
            style = resolve_locomotion_style(kwargs["locomotion_style"], self._agent.clip_keys, self._style_map)
        else:
            style = self._default_style
        mode_idx = resolve_mode(style, self._agent.clip_keys)

        # Inject the default movement direction when the call passes none.
        call_kwargs = dict(kwargs)
        if call_kwargs.get("target_velocity") is None and self._default_velocity is not None:
            call_kwargs["target_velocity"] = self._default_velocity

        control_signals = build_control_signals(
            mode_idx=mode_idx,
            clip_token_specs=self._agent.clip_token_specs,
            min_token=self._agent.min_token,
            max_token=self._agent.max_token,
            kwargs=call_kwargs,
        )

        qpos = np.asarray(self._agent.next_qpos(control_signals, self._controller_dt), dtype=np.float64).ravel()
        self._last_qpos = qpos
        keys = self._joint_names
        needed = _NUM_ROOT_QPOS + len(keys)
        if qpos.shape[0] < needed:
            raise RuntimeError(
                f"MotionBricks generator returned qpos of length {qpos.shape[0]}, but {needed} are needed "
                f"(7 root + {len(keys)} joints). The checkpoint/skeleton does not match the {len(keys)}-DOF G1."
            )
        joint_targets = qpos[_NUM_ROOT_QPOS : _NUM_ROOT_QPOS + len(keys)]
        return [{k: float(v) for k, v in zip(keys, joint_targets, strict=True)}]


class _MotionBricksAgentAdapter:
    """Real :class:`MotionAgent` wrapping the upstream ``full_navigation_agent``.

    Confines the torch + ``motionbricks`` imports and the GPU/CPU device
    handling to the heavy path so :mod:`policy` and :mod:`observation` stay
    import-light and unit-testable. Built via :meth:`build`.
    """

    def __init__(
        self,
        full_agent: Any,
        clip_keys: list[str],
        clip_token_specs: list[list[int] | None],
        min_token: int,
        max_token: int,
        device: str,
    ) -> None:
        self._fa = full_agent
        self.clip_keys = clip_keys
        self.clip_token_specs = clip_token_specs
        self.min_token = int(min_token)
        self.max_token = int(max_token)
        self._device = device

    @classmethod
    def build(cls, config: MotionBricksConfig, result_dir: Path) -> _MotionBricksAgentAdapter:
        """Construct the upstream generator from the checkpoint tree.

        The upstream configs reference the skeleton with a path relative to the
        process CWD (``out/...``), so the heavy load runs with CWD set to the
        parent of ``result_dir`` (restored afterwards). When ``device='cpu'`` on
        a CUDA-only host the upstream ``test()`` would crash pinning a GPU; we
        neutralise that pin for the duration of the build.
        """
        from types import SimpleNamespace

        import torch as t  # heavy path only
        from motionbricks.exp_setup.experiment import test  # type: ignore[import-not-found]
        from motionbricks.motion_backbone.demo.clips import clip_holder_G1  # type: ignore[import-not-found]
        from motionbricks.motion_backbone.demo.full_agent import full_navigation_agent  # type: ignore[import-not-found]
        from motionbricks.motion_backbone.inference.motion_inference import (
            motion_inference,  # type: ignore[import-not-found]
        )

        device = config.device
        skeleton_xml = config.skeleton_xml or str(result_dir.parent / "assets" / "skeletons" / "g1" / "g1.xml")
        clips_ckpt = result_dir / "G1-clip.ckpt"
        prev_cwd = os.getcwd()
        prev_set_device = t.cuda.set_device
        try:
            os.chdir(result_dir.parent)
            if device == "cpu":
                # Stop upstream test() from pinning a CUDA device on a CPU run.
                t.cuda.set_device = lambda *a, **k: None  # type: ignore[assignment]
            args = SimpleNamespace(
                result_dir=result_dir.name,
                EXP=config.exp,
                return_model_configs=True,
                return_dataloader=False,
                recording_dir=None,
                data_root=None,
                explicit_dataset_folder=None,
            )
            models, confs = test(args)
            map_loc = "cpu" if device == "cpu" else device
            for name in ("pose", "root"):
                state_dict = t.load(confs[name].ckpt_path, map_location=map_loc)["state_dict"]
                models[name].load_state_dict(state_dict)
            inferencer = motion_inference(models, models["pose"].args, device=device)
            full_agent = full_navigation_agent(
                inferencer,
                None,
                device=device,
                speed_scale=list(config.speed_scale),
                target_root_realignment=True,
                source_root_realignment=True,
                force_canonicalization=True,
                skeleton_xml=skeleton_xml,
                skip_ending_target_cond=False,
                filter_qpos=True,
                clips=config.clips,
                ckpt_path=str(clips_ckpt) if clips_ckpt.is_file() else None,
                reprocess_clips=False,
                val_dataloader=None,
            ).to(device)
        finally:
            os.chdir(prev_cwd)
            t.cuda.set_device = prev_set_device  # type: ignore[assignment]

        clip_keys = list(clip_holder_G1.CLIPS.keys())
        clip_token_specs: list[list[int] | None] = [
            clip_holder_G1.CLIPS[k].get("allowed_pred_num_tokens") for k in clip_keys
        ]
        min_token = int(inferencer._args["min_tokens"])
        max_token = int(inferencer._args["max_tokens"])
        return cls(full_agent, clip_keys, clip_token_specs, min_token, max_token, device)

    def reset(self) -> None:
        self._fa.reset()

    def next_qpos(self, control_signals: dict[str, Any], controller_dt: float) -> NDArray[np.float64]:
        import torch as t  # heavy path only

        qpos = self._fa.get_next_frame()
        context = self._fa.get_context_mujoco_qpos()
        torch_cs = {
            "movement_direction": t.tensor([control_signals["movement_direction"]], dtype=t.float32),
            "facing_direction": t.tensor([control_signals["facing_direction"]], dtype=t.float32),
            "mode": t.tensor([[int(control_signals["mode"])]], dtype=t.long),
            "allowed_pred_num_tokens": t.tensor(control_signals["allowed_pred_num_tokens"], dtype=t.int).view([1, -1]),
            "context_mujoco_qpos": context,
        }
        with t.no_grad():
            self._fa.generate_new_frames(torch_cs, controller_dt)
        return np.asarray(qpos, dtype=np.float64)


__all__ = ["MotionBricksPolicy", "MotionAgent", "MOTIONBRICKS_G1_JOINTS"]
