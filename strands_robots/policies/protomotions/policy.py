"""ProtoMotionsPolicy - ONNX Generalist Tracking Policy provider.

Wraps ``cagataydev/protomotions-gtp-unitree-g1``'s ``unified_pipeline.onnx``
into the :class:`~strands_robots.policies.base.Policy` interface. Given a
reference
:class:`~strands_robots.policies.protomotions.motion_utils.MotionPlayer`
and a per-tick observation dict, emits PD joint targets for the G1's 29
actuators.

Contract:

* ``requires_images = False`` - the tracker reads root/anchor rotation and
  joint pos/vel only, never an image.
* ``get_actions(obs, instruction, **kwargs)`` reads:

  - ``observation.state`` (well-known key) OR the joint names in
    :data:`~strands_robots.policies.protomotions.config.GTP_G1_JOINT_NAMES`
    directly on the obs dict - either shape works.
  - Optional per-call knobs on ``kwargs``:

    * ``motion``: A new :class:`MotionPlayer`, cache dict, or ``.pt`` /
      ``.npz`` path to swap in without re-instantiating the policy.
    * ``anchor_rot_xyzw`` / ``root_ang_vel_local``: If the caller already
      derived these (e.g. from an IMU on hardware), pass them directly; else
      the policy computes them from ``observation_dict``.

* Output is a per-frame dict ``{joint_name: target_radians, ...}`` in the
  action-value convention (``float`` per DOF, never ``np.ndarray``).

Injection seam: pass ``session=`` implementing :class:`ProtoMotionsSession`
to unit-test the observation -> future-window -> action-dict mapping without
onnxruntime, weights, or CUDA.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from strands_robots.policies.base import Policy
from strands_robots.utils import positive_whole_number_error, require_optional

from .config import (
    ProtoMotionsConfig,
    load_config_from_yaml,
)
from .motion_utils import MotionPlayer
from .state_utils import (
    compute_anchor_rot,
    compute_root_local_ang_vel,
    mujoco_wxyz_to_xyzw,
)

logger = logging.getLogger(__name__)

__all__ = ["ProtoMotionsPolicy", "ProtoMotionsSession"]

#: Canonical GTP checkpoint on HuggingFace. Nothing in this package downloads
#: it - ``onnx_path``/``yaml_path`` take local files - so this is quoted in the
#: not-found remedy rather than being resolved for the caller.
_GTP_G1_HF_REPO = "cagataydev/protomotions-gtp-unitree-g1"
_GTP_G1_ONNX_FILENAME = "unified_pipeline.onnx"


@runtime_checkable
class ProtoMotionsSession(Protocol):
    """Injection seam for the ONNX tracker session.

    A caller stubs this for tests to avoid importing onnxruntime,
    downloading weights, or needing a CUDA runtime. The real session (``onnxruntime.
    InferenceSession``) satisfies this shape via a thin adapter built in
    :meth:`ProtoMotionsPolicy._build_onnx_session`.
    """

    def run(self, output_names: list[str] | None, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Run the tracker on a single input batch.

        Args:
            output_names: Names of outputs to return (or ``None`` for all,
                in the order the session declares).
            inputs: Named input arrays. Shapes documented in
                :attr:`ProtoMotionsConfig.onnx_in_names`.

        Returns:
            A list of output arrays in the requested order.
        """
        ...


class ProtoMotionsPolicy(Policy):
    """ONNX Generalist Tracking Policy for the Unitree G1.

    See the module docstring for the full contract. All fields the registry
    advertises are explicit constructor parameters - no ``**kwargs`` - so a
    typo raises ``TypeError`` at build time rather than being swallowed.

    Args:
        onnx_path: Path to the ONNX artifact (``unified_pipeline.onnx`` or an
            equivalent GTP export). Required unless ``session=`` is passed.
        yaml_path: Path to the sidecar YAML (``unified_pipeline.yaml``).
            Falls back to the pinned :class:`ProtoMotionsConfig` defaults if
            omitted, which matches the shipped weights.
        motion: Initial reference motion. One of:
            - :class:`MotionPlayer` instance,
            - cache dict (from :func:`qpos_to_motion_data` or ``as_cache()``),
            - ``.npz`` / ``.pt`` path (loaded lazily).
            May be ``None`` at build time and set later via
            ``get_actions(motion=...)`` or :meth:`load_motion`.
        session: Injected ONNX-like session (see :class:`ProtoMotionsSession`).
            Used for tests / non-CUDA hosts.
        providers: Ordered list of onnxruntime execution providers. Defaults to
            ``["CUDAExecutionProvider", "CPUExecutionProvider"]`` if
            ``onnxruntime`` is installed.
        history_length: Length of the historical-actions rolling buffer that
            feeds the ``historical_processed_actions`` ONNX input. The upstream
            checkpoint uses ``1`` (single previous action). Larger values pad
            with zeros until the buffer fills. A positive whole number: the
            value is a buffer dimension, so a fractional or boolean spelling
            cannot be honored as the window length it reads as.
    """

    #: Non-VLA - the tracker is proprioceptive, no cameras.
    requires_images = False

    def __init__(
        self,
        *,
        onnx_path: str | Path | None = None,
        yaml_path: str | Path | None = None,
        motion: MotionPlayer | dict[str, Any] | str | Path | None = None,
        session: ProtoMotionsSession | None = None,
        providers: list[str] | None = None,
        history_length: int = 1,
    ) -> None:
        if session is None and onnx_path is None:
            raise ValueError(
                "ProtoMotionsPolicy requires either `onnx_path` (real weights) "
                "or `session` (injected for tests). Neither was supplied."
            )
        # The shared whole-number domain runs BEFORE the ``int()`` normalisation
        # below, for the reason ProtoMotionsConfig.__post_init__ states about its
        # own indices: that conversion is what turns a config's
        # ``history_length: 2.7`` into a two-frame buffer and a ``true`` into a
        # one-frame one, each silently narrowing the window the tracker reads.
        if error := positive_whole_number_error(history_length, "history_length", "ProtoMotionsPolicy"):
            raise ValueError(error)

        self._config: ProtoMotionsConfig = (
            load_config_from_yaml(yaml_path) if yaml_path is not None else ProtoMotionsConfig()
        )

        self._session: ProtoMotionsSession | None = session
        self._onnx_path: Path | None = Path(onnx_path) if onnx_path else None
        self._providers = providers or [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        # Rolling history buffer, shape [history_length, num_dofs].
        # Initialised BEFORE load_motion so its reset path can zero it in-place.
        self._history_length = int(history_length)
        self._action_history: NDArray[np.float32] = np.zeros(
            (self._history_length, self._config.num_dofs), dtype=np.float32
        )

        # Per-episode playhead - advances on each get_actions() call, unless the
        # caller passes a new motion (which resets it).
        self._frame_cursor: int = 0

        self._motion_player: MotionPlayer | None = None
        if motion is not None:
            self.load_motion(motion)

        # Cache of joint names -> their index inside a caller-supplied
        # ``observation.state`` (list-of-values). Recomputed lazily whenever
        # the robot's state-key list changes.
        self._state_index_cache: dict[str, int] | None = None
        self._robot_state_keys: list[str] = list(self._config.joint_names)

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        """Registry key for this provider (``"protomotions"``)."""
        return "protomotions"

    @property
    def config(self) -> ProtoMotionsConfig:
        """The pinned typed config in force for this instance."""
        return self._config

    @property
    def required_bodies(self) -> tuple[str, ...]:
        """The anchor link whose world orientation the tracker consumes.

        ``torso_link`` on the shipped G1 config. Declaring it here is what
        makes the policy runnable through the standard runtime: the observation
        feed carries the floating base only (``base_quat`` is the PELVIS), and
        the anchor differs from it by the three waist joints, so the tracker
        cannot derive this signal itself. The runtime resolves the name once
        per rollout and merges ``body.<name>.quat`` into every observation.
        """
        return (self._config.anchor_body_name,)

    @property
    def num_dofs(self) -> int:
        """Number of joint DOFs the tracker drives (29 for the G1 GTP)."""
        return self._config.num_dofs

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Resolve the 29 GTP joint names by NAME in the robot's key list.

        The tracker's ONNX input is dof-index-based, so we build a lookup from
        joint name -> the caller's ``observation.state`` index and cache it.
        Resolving by name means the tracker's output aligns with the robot's
        actuators regardless of sim-specific joint ordering or namespacing.

        Args:
            robot_state_keys: The robot's own joint-name list (as reported by
                the runtime's observation feed).

        Raises:
            ValueError: If any of the 29 expected G1 joint names is absent
                from ``robot_state_keys``.
        """
        keys = list(robot_state_keys)
        key_set = set(keys)
        missing = [n for n in self._config.joint_names if n not in key_set]
        if missing:
            raise ValueError(
                "ProtoMotionsPolicy: robot's joint list is missing expected G1 "
                f"joints: {missing}.\n"
                f"  expected: {list(self._config.joint_names)}\n"
                f"  robot provided: {keys}\n"
                "The GTP tracker drives the 29-DOF G1; load unitree_g1."
            )
        self._robot_state_keys = keys
        self._state_index_cache = {name: keys.index(name) for name in keys}

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def load_motion(self, motion: MotionPlayer | dict[str, Any] | str | Path) -> None:
        """Swap in a new reference motion and reset the playhead.

        Args:
            motion: Same union as the constructor. A string / Path is loaded
                lazily by :class:`MotionPlayer` (``.npz`` fast path or ``.pt``
                via torch).
        """
        if isinstance(motion, MotionPlayer):
            self._motion_player = motion
        elif isinstance(motion, dict):
            self._motion_player = MotionPlayer(motion, control_dt=self._config.control_dt)
        elif isinstance(motion, (str, Path)):
            self._motion_player = MotionPlayer(str(motion), control_dt=self._config.control_dt)
        else:
            raise TypeError(f"motion must be a MotionPlayer, cache dict, or path, got {type(motion).__name__}.")

        # New clip -> reset playhead and history.
        self._frame_cursor = 0
        self._action_history[:] = 0.0
        logger.info(
            "ProtoMotionsPolicy loaded motion: %d frames @ %.0f Hz",
            self._motion_player.total_frames,
            1.0 / self._motion_player.control_dt,
        )

    def reset(self, seed: int | None = None) -> None:
        """Reset the playhead and action history at the start of an episode.

        The signature matches :meth:`Policy.reset`, which the runtime calls as
        ``policy.reset(seed=...)`` once per episode. A narrower override would
        raise ``TypeError`` there; the runtime catches it and continues, so the
        playhead would silently carry over and every episode after the first
        would replay from wherever the previous one stopped.

        Args:
            seed: Accepted for interface parity and ignored - the tracker is
                deterministic given its reference motion and observation, so it
                holds no RNG state to reseed.
        """
        del seed  # deterministic tracker: no RNG state to reseed
        self._frame_cursor = 0
        self._action_history[:] = 0.0

    # ------------------------------------------------------------------
    # get_actions
    # ------------------------------------------------------------------

    async def get_actions(
        self,
        observation_dict: dict[str, Any],
        instruction: str,  # unused - tracker is not VLA
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return the next per-frame G1 joint targets from the ONNX tracker.

        Args:
            observation_dict: Per-tick observation. Must supply the current
                anchor rotation, root angular velocity, joint positions and
                joint velocities. Two shapes are accepted:

                1. A canonical ``observation.state`` list plus explicit
                   ``anchor_rot_xyzw`` / ``root_ang_vel_local`` on ``kwargs``.
                2. Per-joint entries under the GTP joint names (with ``.vel``
                   suffix for velocities), plus ``anchor_rot_xyzw`` /
                   ``root_ang_vel_local`` on the obs dict itself. Each
                   of those two is also read under its
                   ``observation.``-prefixed spelling
                   (``observation.anchor_rot_xyzw`` /
                   ``observation.root_ang_vel_local``), which is the
                   shape a runtime that namespaces its observation keys
                   produces.

            instruction: Ignored - the tracker follows the loaded motion, not
                a text prompt.
            **kwargs: Optional per-call knobs:

                * ``motion``: Swap in a new reference on this call
                  (see :meth:`load_motion`).
                * ``anchor_rot_xyzw``: Shape ``[4]`` (xyzw) explicit override.
                * ``root_ang_vel_local``: Shape ``[3]`` explicit override.
                * ``dof_pos`` / ``dof_vel``: Shape ``[num_dofs]`` explicit
                  overrides. When present, they win over ``observation_dict``.

        Returns:
            A single-frame list ``[{joint_name: target_radians, ...}]``
            matching the base ``Policy.get_actions`` list-of-dicts contract.

        Raises:
            RuntimeError: If no motion has been loaded yet.
            KeyError: If a required piece of state is absent from both the
                obs dict and ``kwargs``.
        """
        if "motion" in kwargs and kwargs["motion"] is not None:
            self.load_motion(kwargs["motion"])

        if self._motion_player is None:
            raise RuntimeError(
                "ProtoMotionsPolicy.get_actions: no reference motion loaded. "
                "Pass `motion=` to the constructor, call load_motion(), or "
                "pass `motion=...` on get_actions()."
            )

        # 1. Build the future-reference window from the MotionPlayer.
        future = self._motion_player.get_future_references(self._frame_cursor, list(self._config.future_step_indices))
        num_future = self._config.num_future_steps

        mimic_future_anchor_rot = (
            future["body_rot"][:, self._config.anchor_body_index, :].reshape(1, num_future, 4).astype(np.float32)
        )
        mimic_future_dof_pos = future["dof_pos"].reshape(1, num_future, self._config.num_dofs).astype(np.float32)
        mimic_future_dof_vel = future["dof_vel"].reshape(1, num_future, self._config.num_dofs).astype(np.float32)

        # 2. Derive the current-state inputs from the observation.
        current_anchor_rot = self._extract_anchor_rot(observation_dict, kwargs)
        current_root_local_ang_vel = self._extract_root_local_ang_vel(observation_dict, kwargs)
        current_dof_pos = self._extract_dof_pos(observation_dict, kwargs)
        current_dof_vel = self._extract_dof_vel(observation_dict, kwargs)

        # 3. Historical actions buffer (shape [1, history_length, num_dofs]).
        historical = self._action_history.reshape(1, self._history_length, self._config.num_dofs).astype(np.float32)

        inputs = {
            "current_anchor_rot": current_anchor_rot.reshape(1, 4).astype(np.float32),
            "current_dof_pos": current_dof_pos.reshape(1, self._config.num_dofs).astype(np.float32),
            "current_dof_vel": current_dof_vel.reshape(1, self._config.num_dofs).astype(np.float32),
            "current_root_local_ang_vel": current_root_local_ang_vel.reshape(1, 3).astype(np.float32),
            "historical_processed_actions": historical,
            "mimic_future_anchor_rot": mimic_future_anchor_rot,
            "mimic_future_dof_pos": mimic_future_dof_pos,
            "mimic_future_dof_vel": mimic_future_dof_vel,
        }

        session = self._get_session()
        requested = list(self._config.onnx_out_names)
        outputs = session.run(requested, inputs)
        # Pair outputs to names via the list we asked for, so a config that
        # declares the checkpoint's outputs in a different order still reads
        # the right tensor. Positional indexing would silently feed the PD
        # loop `stiffness_targets` after such a reorder.
        by_name = dict(zip(requested, outputs, strict=False))
        missing = [name for name in ("actions", "joint_pos_targets") if name not in by_name]
        if missing:
            raise RuntimeError(
                "ProtoMotionsPolicy: the tracker session returned no "
                f"{missing} output. config.onnx_out_names declares "
                f"{requested}; the session answered with "
                f"{len(outputs)} array(s). The GTP export must expose "
                "`actions` (fed back into the history buffer) and "
                "`joint_pos_targets` (sent to the robot's PD loop)."
            )
        actions_out = by_name["actions"].reshape(-1)[: self._config.num_dofs]
        joint_pos_targets = by_name["joint_pos_targets"].reshape(-1)[: self._config.num_dofs]

        # 4. Update history + advance the playhead.
        self._action_history = np.roll(self._action_history, shift=-1, axis=0)
        self._action_history[-1] = actions_out.astype(np.float32)
        self._frame_cursor += 1

        # 5. Wrap as the per-joint action-dict expected by the runtime.
        action_dict: dict[str, float] = {
            name: float(joint_pos_targets[i]) for i, name in enumerate(self._config.joint_names)
        }
        return [action_dict]

    # ------------------------------------------------------------------
    # ONNX session lifecycle
    # ------------------------------------------------------------------

    def _get_session(self) -> ProtoMotionsSession:
        """Return the cached session, building it from ``onnx_path`` on first call."""
        if self._session is not None:
            return self._session
        assert self._onnx_path is not None  # validated in __init__
        self._session = self._build_onnx_session(self._onnx_path, self._providers)
        return self._session

    @staticmethod
    def _build_onnx_session(onnx_path: Path, providers: list[str]) -> ProtoMotionsSession:
        """Build a real onnxruntime session - deferred so tests don't import ORT."""
        ort = require_optional(
            "onnxruntime",
            extra="protomotions",
            purpose="running the GTP tracker graph (pass `session=` to inject a stub instead)",
        )

        if not onnx_path.exists():
            # ``onnx_path`` takes a local file, and nothing here resolves a
            # model id (unlike WBCPolicy, which downloads its checkpoint). The
            # old wording read "Download from <repo> on HuggingFace", which is
            # circular for the caller who passed that repo id: it names the
            # argument back instead of the step that produces a local path.
            raise FileNotFoundError(
                f"ONNX artifact not found: {onnx_path}. This parameter takes a local file, "
                f"and this policy does not download one. Fetch the checkpoint first:\n"
                f'  python -c "from huggingface_hub import hf_hub_download; '
                f"print(hf_hub_download('{_GTP_G1_HF_REPO}', '{_GTP_G1_ONNX_FILENAME}'))\"\n"
                f"then pass the path it prints as onnx_path."
            )

        sess = ort.InferenceSession(  # type: ignore[attr-defined]
            str(onnx_path), providers=providers
        )
        logger.info(
            "ProtoMotions ONNX session ready: %s (providers=%s)",
            onnx_path.name,
            sess.get_providers(),
        )
        return sess  # onnxruntime.InferenceSession already satisfies the Protocol

    # ------------------------------------------------------------------
    # Observation extraction helpers
    # ------------------------------------------------------------------

    def _extract_anchor_rot(self, obs: dict[str, Any], kwargs: dict[str, Any]) -> np.ndarray:
        """Pull the anchor-body ``xyzw`` rotation from obs or an explicit kwarg."""
        if "anchor_rot_xyzw" in kwargs and kwargs["anchor_rot_xyzw"] is not None:
            return np.asarray(kwargs["anchor_rot_xyzw"], dtype=np.float32)
        # The runtime's answer to `required_bodies`: body.<anchor>.quat, wxyz.
        anchor_key = f"body.{self._config.anchor_body_name}.quat"
        if anchor_key in obs:
            return mujoco_wxyz_to_xyzw(np.asarray(obs[anchor_key], dtype=np.float32))
        # Well-known observation keys. Every observation-key fallback ladder in
        # this package pairs a bare key with ``observation.<that key>``: the
        # sibling ladder in :meth:`_extract_root_local_ang_vel` below does, and
        # so do both of ``WBCPolicy``'s. A caller who follows that convention
        # therefore writes ``observation.anchor_rot_xyzw``, and that spelling
        # was absent while the suffix-less ``observation.anchor_rot`` was
        # accepted -- so one observation dict whose keys were both spelled to
        # the convention had its angular velocity resolve and its anchor
        # rotation raise. The suffix-less form is kept because it is accepted
        # today; it is also the weaker name, since it does not say which
        # component order it carries and this rung does not reorder.
        for key in ("anchor_rot_xyzw", "observation.anchor_rot_xyzw", "observation.anchor_rot"):
            if key in obs:
                return np.asarray(obs[key], dtype=np.float32)
        # Fallback: derive from a full body-rotation batch.
        if "body_rot_xyzw" in obs:
            arr = np.asarray(obs["body_rot_xyzw"], dtype=np.float32)
            return compute_anchor_rot(arr, self._config.anchor_body_index)
        # The floating-base quaternion is the ROOT. It is the anchor rotation
        # only when the config anchors on the root itself; on the G1 the torso
        # differs from the pelvis by the waist joints, so substituting it would
        # silently feed the tracker the wrong frame.
        if self._config.anchor_is_root:
            for key in ("base_quat", "root_quat_wxyz"):
                if key in obs:
                    return mujoco_wxyz_to_xyzw(np.asarray(obs[key], dtype=np.float32))
        raise KeyError(
            "ProtoMotionsPolicy: could not resolve the anchor rotation. The "
            f"tracker anchors on {self._config.anchor_body_name!r}, so it needs "
            f"{anchor_key!r} in the observation. Through a simulation rollout "
            "the runtime supplies that key from this policy's "
            "`required_bodies`; a caller assembling observations by hand can "
            "instead pass `anchor_rot_xyzw=[x,y,z,w]` via kwargs, or supply it "
            "on the observation as `anchor_rot_xyzw` (or its prefixed spelling "
            "`observation.anchor_rot_xyzw`), or supply "
            "`body_rot_xyzw`. Note that `base_quat` is the floating base "
            f"({self._config.root_body_name!r}), NOT the anchor."
        )

    def _extract_root_local_ang_vel(self, obs: dict[str, Any], kwargs: dict[str, Any]) -> np.ndarray:
        """Pull the root-body local-frame angular velocity."""
        if "root_ang_vel_local" in kwargs and kwargs["root_ang_vel_local"] is not None:
            return np.asarray(kwargs["root_ang_vel_local"], dtype=np.float32)
        for key in ("root_ang_vel_local", "observation.root_ang_vel_local"):
            if key in obs:
                return np.asarray(obs[key], dtype=np.float32)
        # Derive from full body-rot + world-frame body-angvel arrays.
        if "body_rot_xyzw" in obs and "body_ang_vel_world" in obs:
            rots = np.asarray(obs["body_rot_xyzw"], dtype=np.float32)
            avels = np.asarray(obs["body_ang_vel_world"], dtype=np.float32)
            return compute_root_local_ang_vel(rots, avels, self._config.root_body_index)
        # The runtime's floating-base angular velocity is the root freejoint's
        # qvel[3:6], which MuJoCo already reports in the BODY frame - exactly
        # the local-frame quantity this input wants.
        for key in ("base_ang_vel", "root_ang_vel_body"):
            if key in obs:
                return np.asarray(obs[key], dtype=np.float32)
        raise KeyError(
            "ProtoMotionsPolicy: could not resolve the root angular velocity "
            "in the root's local frame. A simulation rollout supplies it as "
            "`base_ang_vel` (the floating base's body-frame qvel); a caller "
            "assembling observations by hand can pass "
            "`root_ang_vel_local=[wx,wy,wz]` via kwargs or supply "
            "`body_rot_xyzw`+`body_ang_vel_world`."
        )

    def _extract_dof_pos(self, obs: dict[str, Any], kwargs: dict[str, Any]) -> np.ndarray:
        """Pull the 29 joint positions in :attr:`config.joint_names` order."""
        if "dof_pos" in kwargs and kwargs["dof_pos"] is not None:
            return np.asarray(kwargs["dof_pos"], dtype=np.float32)
        return self._pack_by_name(obs, suffix="")

    def _extract_dof_vel(self, obs: dict[str, Any], kwargs: dict[str, Any]) -> np.ndarray:
        """Pull the 29 joint velocities in :attr:`config.joint_names` order."""
        if "dof_vel" in kwargs and kwargs["dof_vel"] is not None:
            return np.asarray(kwargs["dof_vel"], dtype=np.float32)
        return self._pack_by_name(obs, suffix=".vel")

    def _pack_by_name(self, obs: dict[str, Any], suffix: str, fill: float | None = None) -> np.ndarray:
        """Pack per-joint values from the obs dict in canonical joint order.

        Tries three obs conventions in order:
        1. ``observation.state`` plus a matching ``state_keys`` list on ``obs``.
        2. Per-joint keys ``<name><suffix>`` directly on ``obs``.
        3. A single ``observation.state`` array whose ordering matches
           ``self._robot_state_keys`` (set via :meth:`set_robot_state_keys`).

        ``observation.state`` is flattened before either convention indexes it,
        so a batched ``(1, D)`` state reads identically to a flat ``(D,)`` one.
        """
        # ``observation.state`` is read by conventions 1 and 3 below, and both
        # index it positionally, so it is normalized once here rather than in
        # each branch. A runtime that batches the state feeds ``(1, D)`` -
        # LeRobot's own ``AddBatchDimensionObservationStep`` does exactly that -
        # and flattening in only one branch made the two conventions disagree
        # about the same observation.
        raw_state = obs.get("observation.state")
        state_arr = None if raw_state is None else np.asarray(raw_state, dtype=np.float32).reshape(-1)

        # 1. observation.state + explicit joint-key list on obs.
        keys_list = obs.get("state_keys")
        if state_arr is not None and keys_list is not None:
            out = np.zeros(self._config.num_dofs, dtype=np.float32)
            for i, name in enumerate(self._config.joint_names):
                key = name + suffix
                if key in keys_list:
                    out[i] = float(state_arr[list(keys_list).index(key)])
                elif fill is None:
                    raise KeyError(f"ProtoMotionsPolicy: joint {key!r} missing from observation.state's state_keys.")
                else:
                    out[i] = fill
            return out

        # 2. Per-joint keys directly on the obs dict.
        direct_hits = sum(1 for name in self._config.joint_names if (name + suffix) in obs)
        if direct_hits > 0:
            out = np.zeros(self._config.num_dofs, dtype=np.float32)
            for i, name in enumerate(self._config.joint_names):
                key = name + suffix
                if key in obs:
                    out[i] = float(obs[key])
                elif fill is None:
                    raise KeyError(
                        f"ProtoMotionsPolicy: joint {key!r} missing from obs "
                        f"(found {direct_hits}/{self._config.num_dofs})."
                    )
                else:
                    out[i] = fill
            return out

        # 3. observation.state matches self._robot_state_keys order.
        if state_arr is not None and self._robot_state_keys:
            out = np.zeros(self._config.num_dofs, dtype=np.float32)
            for i, name in enumerate(self._config.joint_names):
                key = name + suffix
                if key in self._robot_state_keys:
                    out[i] = float(state_arr[self._robot_state_keys.index(key)])
                elif fill is None:
                    raise KeyError(
                        f"ProtoMotionsPolicy: joint {key!r} missing from "
                        f"self._robot_state_keys (call set_robot_state_keys)."
                    )
                else:
                    out[i] = fill
            return out

        # Nothing matched.
        if fill is None:
            raise KeyError(
                f"ProtoMotionsPolicy: could not resolve dof values with suffix "
                f"{suffix!r}. Provide `observation.state`+`state_keys`, "
                f"per-joint keys, or explicit `dof_pos`/`dof_vel` kwargs."
            )
        return np.full(self._config.num_dofs, fill, dtype=np.float32)
