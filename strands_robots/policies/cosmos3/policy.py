"""Cosmos 3 policy - NVIDIA omnimodal VLA policy via Cosmos Framework.

Implements :class:`~strands_robots.policies.base.Policy` for the Cosmos 3
**Generator action surface** (``nvidia/Cosmos3-Nano-Policy-DROID`` and friends).

The Cosmos 3 ``policy`` action mode takes ``image + instruction`` and returns an
``[T, D]`` action chunk + rollout video - a 1:1 match for the robots policy
contract. We talk to the Cosmos Framework RoboLab WebSocket policy server
(``cosmos_framework.scripts.action_policy_server_robolab``) over a
self-contained msgpack+NumPy WebSocket protocol (no ``openpi-client``
dependency - see :mod:`strands_robots.policies.cosmos3.client`), mirroring
:class:`~strands_robots.policies.groot.Gr00tPolicy` service mode.

Observation flow
----------------
The robots ``SimEngine.get_observation`` returns a **flat** dict::

    {"<joint_name>": float, ..., "<camera_name>": np.ndarray(H, W, 3)}

We translate that into the server's OpenPI observation::

    {
        "prompt": instruction,
        "observation/wrist_image_left":      np.ndarray(H, W, 3) uint8,
        "observation/exterior_image_1_left": ...,
        "observation/exterior_image_2_left": ...,
        "observation/joint_position":  np.ndarray(1, 7) float32,
        "observation/gripper_position": np.ndarray(1, 1) float32,
    }

via an explicit ``observation_mapping`` (robot key → server key), with a
sensible auto-mapping fallback.

Action flow
-----------
The server returns ``{"action": np.ndarray(T, D)}``. Each of the ``D`` columns
is named by the embodiment's ``action_layout`` (e.g. DROID joint_pos =
``[joint_0..joint_6, gripper]``). We emit ``list[dict]`` - one dict per
timestep - optionally remapping column names to robot actuator names via
``action_mapping``.

Example::

    from strands_robots.policies import create_policy

    # robot="panda" applies the built-in DROID-layout -> Panda actuator mapping
    # (joint_0..joint_6 -> joint1..joint7, gripper -> finger_joint1), so the
    # per-step dicts use real MuJoCo Panda actuator names without manual mapping.
    policy = create_policy(
        "cosmos3",
        embodiment="droid",
        host="localhost",
        port=8000,
        robot="panda",
    )
    chunk = policy.get_actions_sync(observation, "pick up the cube")
    # chunk == [{"joint1": .., ..., "finger_joint1": ..}, ...]  (one per timestep)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots.policies._state_keys import drop_velocity_siblings
from strands_robots.policies.base import Policy
from strands_robots.utils import name_list_error, tcp_port_error

from .client import Cosmos3WebsocketClient
from .embodiments import (
    Cosmos3Embodiment,
    get_embodiment,
    get_robot_action_mapping,
    list_robot_action_mappings,
)

if TYPE_CHECKING:
    from .policy_diffusers import Cosmos3DiffusersBackend

logger = logging.getLogger(__name__)


_IMAGE_KEY_HINTS = ("image", "rgb", "cam")


def _is_image_key(server_key: str) -> bool:
    """Heuristic: does an OpenPI server key name a camera image?"""
    low = server_key.lower()
    return any(h in low for h in _IMAGE_KEY_HINTS)


def _to_image_uint8(value: Any) -> np.ndarray:
    """Coerce a camera frame to a contiguous ``(H, W, 3) uint8`` array."""
    arr = np.asarray(value)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"camera frame must be (H, W, 3); got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


class Cosmos3Policy(Policy):
    """NVIDIA Cosmos 3 VLA policy (service mode via Cosmos Framework).

    Args:
        embodiment: Embodiment key/alias (``"droid"``, ``"umi"``, ``"av"``,
            ``"bridge"``, ``"openarm"``). Selects domain, action layout, and
            defaults. ``"openarm"`` is a post-training-only embodiment: it maps
            a checkpoint post-trained on OpenArm episodes (see
            :mod:`strands_robots.policies.cosmos3.embodiments`), not a released
            zero-shot model.
        host: Policy-server hostname.
        port: Policy-server WebSocket port, an ``int`` in ``[1, 65535]``.
            Read only when this constructor builds the client; an injected
            ``client`` owns its own address. A value outside the range is
            refused rather than interpolated into ``ws://<host>:<port>``.
        action_space: ``"joint_pos"`` or ``"midtrain"`` - must match how the
            server was launched (DROID default = ``joint_pos``).
        observation_mapping: ``{robot_obs_key: "observation/<server_key>"}``.
            Maps robot camera + state keys onto the server's OpenPI keys.
            When ``None``, a default mapping is used (see :meth:`_default_obs_mapping`).
            The mapping MUST cover the embodiment's **full** camera set
            (:attr:`Cosmos3Embodiment.camera_keys`): the RoboLab server composes
            its conditioning from all declared views and rejects a partial
            observation. For the DROID embodiment (``joint_pos``/RoboArena) that
            is all three keys - ``observation/wrist_image_left``,
            ``observation/exterior_image_1_left``, and
            ``observation/exterior_image_2_left``. A mapping that omits any
            required key (or whose source camera is absent at runtime) raises a
            client-side ``ValueError`` naming the missing keys, before any
            request reaches the server.
        action_mapping: ``{action_column_name: robot_actuator_name}``. Renames
            the embodiment's action-layout columns to robot actuator names.
            Keys are validated against the active layout at construction.
            When ``None``, columns keep their layout names.
        robot: Convenience - name of a known robot (``"panda"``/``"franka"``)
            whose built-in DROID-layout action mapping is applied when
            ``action_mapping`` is not given. Explicit ``action_mapping`` wins.
        prompt: Default instruction used when ``get_actions`` is called with an
            empty instruction.
        api_key: Optional bearer token for the server.
        client: Pre-built client (dependency injection for tests).
        backend: ``"service"`` (default) talks to the Cosmos Framework RoboLab
            WebSocket policy server (unchanged behaviour). ``"diffusers"`` runs
            Cosmos 3 **in-process** via native Hugging Face ``diffusers``
            (the upstream ``Cosmos3OmniPipeline``); one forward pass returns the
            predicted world video + sound + action chunk. The diffusers backend
            imports ``diffusers`` + ``torch`` lazily and raises an actionable
            install error when the stack is missing. NOTE: the diffusers backend
            emits the model's raw unified action (e.g. DROID = 9D end-effector
            pose + 1D gripper), not the service server's ``joint_pos`` (8D)
            layout - use ``backend="service"`` when you need joint commands.
        mode: Cosmos 3 physics mode (``"policy"`` default, ``"forward_dynamics"``,
            ``"inverse_dynamics"``). Only the ``diffusers`` backend supports the
            non-default modes; a non-``"policy"`` mode under ``backend="service"``
            raises (the RoboLab server serves only the policy action surface).
        model: HF repo id / local path for the in-process diffusers checkpoint
            (default ``"nvidia/Cosmos3-Nano"``). Ignored by the service backend
            (the server selects the checkpoint via ``--checkpoint-path``).
        diffusers_backend: Pre-built
            :class:`~strands_robots.policies.cosmos3.policy_diffusers.Cosmos3DiffusersBackend`.
            Injecting one skips the heavy import (how the tests drive this
            backend), and it is also the supported route to the backend's own
            load and sampling knobs: this constructor forwards only
            ``embodiment``, ``model`` and ``mode``, so ``resolution_tier``,
            ``view_point``, ``device``, ``dtype``, ``num_inference_steps``,
            ``guidance_scale``, ``enable_sound`` and ``enable_safety_checker``
            are set on the backend and reach the policy through this parameter.

    Notes:
        * This policy needs camera frames **and** robot state in the
          observation - ``requires_images`` is ``True``. Every camera key in
          the embodiment's :attr:`~Cosmos3Embodiment.camera_keys` must be
          mapped and present (the DROID embodiment requires all three RoboArena
          views); a partial set fails fast client-side.
        * Latency is chunked (a diffusion policy), not 500 Hz servo. One
          inference returns a chunk of ~``action_chunk_size`` steps.
        * The predicted world video/sound (diffusers backend) are surfaced on
          :attr:`last_rollout` after each ``get_actions`` call, leaving the
          ``list[dict]`` return type (the Policy ABC contract) unchanged.
    """

    def __init__(
        self,
        embodiment: str = "droid",
        host: str = "localhost",
        port: int = 8000,
        action_space: str | None = None,
        observation_mapping: dict[str, str] | None = None,
        action_mapping: dict[str, str] | None = None,
        robot: str | None = None,
        prompt: str = "",
        api_key: str | None = None,
        client: Cosmos3WebsocketClient | None = None,
        transport: str = "raw",
        pretrained_name_or_path: str | None = None,
        backend: str = "service",
        mode: str = "policy",
        diffusers_backend: Cosmos3DiffusersBackend | None = None,
        model: str | None = None,
    ) -> None:
        self.embodiment: Cosmos3Embodiment = get_embodiment(embodiment)
        self.host = host
        self.port = port
        # ``pretrained_name_or_path`` is injected by the registry's model-id
        # resolver (e.g. create_policy("nvidia/Cosmos3-Nano-Policy-DROID")).
        # Cosmos 3 service mode picks the checkpoint server-side (via
        # --checkpoint-path), so this kwarg is informational only. We store it
        # for introspection and log a hint so the user knows the value isn't
        # being silently dropped.
        self.pretrained_name_or_path = pretrained_name_or_path
        if pretrained_name_or_path is not None:
            logger.info(
                "Cosmos3Policy: pretrained_name_or_path=%r noted. "
                "Service mode selects the checkpoint server-side "
                "(--checkpoint-path). Ensure the server is running the "
                "expected model.",
                pretrained_name_or_path,
            )
        self.action_space = action_space or self.embodiment.default_action_space
        if self.action_space not in self.embodiment.action_layouts:
            raise ValueError(
                f"embodiment {self.embodiment.name!r} has no action_space "
                f"{self.action_space!r}; available: {sorted(self.embodiment.action_layouts)}"
            )
        self.default_prompt = prompt
        self._obs_mapping = observation_mapping or self._default_obs_mapping()
        # ``robot=`` sugar: apply a built-in DROID-layout -> actuator mapping
        # (e.g. robot="panda" -> joint_0..6->joint1..7, gripper->finger_joint1)
        # unless the caller supplied an explicit action_mapping. Unknown robot
        # values are rejected up-front so a typo'd or unsupported name cannot
        # silently fall through to the raw DROID layout (whose keys the user's
        # robot will then ignore in send_action). See AGENTS.md key convention
        # #6 "No silent defaults on error".
        if action_mapping is None and robot is not None:
            action_mapping = get_robot_action_mapping(robot)
            if action_mapping is None:
                raise ValueError(
                    f"Unknown robot {robot!r}. Available built-in mappings: "
                    f"{list_robot_action_mappings()}. Pass an explicit "
                    f"action_mapping= or omit robot=."
                )
        self._action_mapping = action_mapping or {}
        self.robot_state_keys: list[str] = []
        # Auxiliary world outputs (predicted video / sound) from the last
        # get_actions call, surfaced WITHOUT changing the Policy ABC return
        # type. None until the first inference, and always None for the service
        # backend (the RoboLab server's "video" field is not consumed here).
        self.last_rollout: dict[str, Any] | None = None

        self._client: Cosmos3WebsocketClient | None = None
        self._diffusers: Cosmos3DiffusersBackend | None = None
        if backend not in ("service", "diffusers"):
            raise ValueError(f"Unknown Cosmos 3 backend {backend!r}. Available: ['service', 'diffusers'].")
        self.backend = backend
        self.mode = mode
        # Validate action_mapping keys name real columns of the ACTIVE layout so
        # a typo'd rename can't silently emit a key the robot never consumes.
        # The two backends emit different action layouts: ``service`` returns the
        # RoboLab server's post-processed ``action_space`` layout (joint_pos /
        # midtrain); ``diffusers`` returns the model's raw unified action named
        # by ``raw_action_layout``. Validate against whichever is active.
        layout_cols = set(self._active_action_layout())
        bad = [k for k in self._action_mapping if k not in layout_cols]
        if bad:
            raise ValueError(
                f"action_mapping keys {bad} are not in the {self.embodiment.name!r} "
                f"{backend!r}-backend action layout. Valid columns: {sorted(layout_cols)}"
            )
        # ``mode`` (policy / forward_dynamics / inverse_dynamics) is a diffusers-
        # only physics surface (CosmosActionCondition.mode). The service RoboLab
        # server only does the policy action surface, so a non-default mode under
        # backend="service" is an error, not a silent no-op (AGENTS.md #6).
        if backend == "service" and mode != "policy":
            raise ValueError(
                f"mode={mode!r} is only available with backend='diffusers'. The "
                "service backend (Cosmos Framework RoboLab server) serves only the "
                "policy action surface. Pass backend='diffusers' for "
                "forward_dynamics / inverse_dynamics."
            )

        if backend == "diffusers":
            # In-process Cosmos 3 via native diffusers (the heavy diffusers +
            # torch import lives lazily inside Cosmos3DiffusersBackend, which
            # loads Cosmos3OmniPipeline.from_pretrained). No service client.
            if diffusers_backend is not None:
                self._diffusers = diffusers_backend
            else:
                from .policy_diffusers import Cosmos3DiffusersBackend as _Backend

                self._diffusers = _Backend(
                    embodiment=self.embodiment,
                    model=model or pretrained_name_or_path,
                    mode=mode,
                )
            logger.info(
                "Cosmos3Policy ready [embodiment=%s domain=%s action_space=%s chunk=%d backend=diffusers mode=%s]",
                self.embodiment.name,
                self.embodiment.domain_name,
                self.action_space,
                self.embodiment.action_chunk_size,
                mode,
            )
        else:
            # ``port`` addresses the RoboLab policy server this client dials, so
            # a value that cannot name one is refused before it reaches
            # ``ws://<host>:<port>``. An injected ``client`` owns its own
            # address, so the port is validated only when this constructor is
            # the one that builds the endpoint.
            if not client and (port_error := tcp_port_error(port, "port", type(self).__name__)) is not None:
                raise ValueError(port_error)
            self._client = client or Cosmos3WebsocketClient(host=host, port=port, api_key=api_key, transport=transport)
            logger.info(
                "Cosmos3Policy ready [embodiment=%s domain=%s action_space=%s chunk=%d backend=service ws://%s:%d]",
                self.embodiment.name,
                self.embodiment.domain_name,
                self.action_space,
                self.embodiment.action_chunk_size,
                host,
                port,
            )

    @property
    def provider_name(self) -> str:
        """Registry key for this provider (``"cosmos3"``)."""
        return "cosmos3"

    @property
    def requires_images(self) -> bool:
        """Cosmos 3 conditions on camera frames - always needs images."""
        return True

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Record the robot's ordered joint/state keys.

        Used (a) as the fallback gripper/joint source when no explicit
        ``observation_mapping`` names them, and (b) as default action actuator
        names when no ``action_mapping`` is supplied and the layout is generic.

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
        self.robot_state_keys = list(robot_state_keys)

    def reset(self, seed: int | None = None) -> None:
        """Per-episode reset.

        Forwards a best-effort ``reset`` hint to the policy server and reseeds
        the local NumPy RNG when ``seed`` is given.

        .. note::
            **The ``seed`` is NOT forwarded to the server's diffusion sampler.**
            The Cosmos Framework RoboLab server's ``reset`` endpoint (and
            OpenPI's ``WebsocketClientPolicy.reset()``) take no arguments, so
            the server-side per-episode RNG is not re-seeded from here. As
            documented in :meth:`Policy.reset` (the #187 reproducibility
            caveat), rollouts are therefore **not** byte-reproducible across
            re-runs purely by passing ``seed``. To get deterministic server
            rollouts, launch the server with ``--deterministic-seed`` (and a
            fixed ``--seed``), or extend the robolab server to accept a
            per-request seed (tracked as an upstream feature request).
        """
        if self.backend == "diffusers":
            assert self._diffusers is not None
            self._diffusers.reset()
        else:
            assert self._client is not None
            self._client.reset()
        # #331: reseed via the shared helper so Cosmos3Policy reaches RNG parity
        # with Gr00tPolicy (Python random + NumPy + torch CPU/CUDA + cuDNN
        # determinism), not just the global NumPy RNG. Same set_eval_seed
        # behaviour across both providers for #187 reproducibility.
        from strands_robots.policies._rng import reseed_client_rngs

        reseed_client_rngs(seed)

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Query Cosmos 3 for an action chunk.

        Args:
            observation_dict: Flat robots observation (joint floats + camera
                ndarrays), per the ``SimEngine.get_observation`` schema.
            instruction: Natural-language task instruction.
            **kwargs: Extra inference options, honored on the ``diffusers``
                backend only - they are forwarded verbatim to the world model's
                ``infer`` call. The ``service`` backend calls the remote client
                without them, so the same keyword is silently dropped there; the
                ABC requires a provider to ignore what it cannot honor rather
                than raise (:meth:`~strands_robots.policies.base.Policy.get_actions`),
                and this provider's answer depends on the ``backend`` it was
                constructed with. None of the well-known keys the ABC lists
                (``target_pose``, ``target_joints``, ``world_update``) is read by
                either backend.

        Returns:
            ``list[dict]`` - one action dict per predicted timestep.
        """
        prompt = instruction or self.default_prompt
        obs = self._build_server_observation(observation_dict, prompt)
        if self.backend == "diffusers":
            assert self._diffusers is not None  # set in __init__ for backend=diffusers
            result = self._diffusers.infer(obs, **kwargs)
            # Surface the predicted world video/sound on a non-breaking channel
            # so the ABC return type stays list[dict]. None for forward_dynamics
            # action, but the world video is still captured.
            self.last_rollout = {
                "action": result.get("action"),
                "video": result.get("video"),
                "sound": result.get("sound"),
            }
            action_arr = result.get("action")
            if action_arr is None:
                # mode="forward_dynamics" predicts world video only - there is no
                # action chunk to return. Surfaced via last_rollout["video"].
                return []
            return self._unpack_actions(np.asarray(action_arr))
        assert self._client is not None  # set in __init__ for backend=service
        result = self._client.infer(obs)
        action = np.asarray(result["action"])
        return self._unpack_actions(action)

    def _default_obs_mapping(self) -> dict[str, str]:
        """Identity-ish default: assume robot obs already uses server keys.

        Falls back to mapping the embodiment's expected camera keys onto
        themselves; callers with differently-named robot cameras should pass an
        explicit ``observation_mapping``.
        """
        return {k: k for k in self.embodiment.camera_keys}

    def _build_server_observation(self, robot_obs: dict[str, Any], prompt: str) -> dict[str, Any]:
        """Translate the flat robot observation into the server's OpenPI dict."""
        obs: dict[str, Any] = {"prompt": prompt}

        # Images: map robot camera keys → server image keys.
        for robot_key, server_key in self._obs_mapping.items():
            if robot_key in robot_obs and server_key.startswith("observation/"):
                val = robot_obs[robot_key]
                if isinstance(val, np.ndarray) or hasattr(val, "__array__"):
                    arr = np.asarray(val)
                    if arr.ndim == 3:
                        obs[server_key] = _to_image_uint8(arr)
                        continue
                # Non-image mapped values pass straight through (e.g. state).
                obs[server_key] = val

        # State for joint_pos action space: joint_position (1,7) + gripper (1,1).
        if self.action_space == "joint_pos":
            self._attach_joint_state(robot_obs, obs)

        # Camera-set guard: the served embodiment conditions on a fixed set of
        # camera frames, and the RoboLab server hard-requires the *full* set -
        # e.g. the DROID joint_pos / RoboArena path composes its views from all
        # three keys (wrist_image_left + exterior_image_1_left +
        # exterior_image_2_left) and rejects a partial observation server-side
        # with an opaque error. Validate every required key is present *before*
        # sending so a partial observation_mapping (or a camera missing at
        # runtime) fails fast client-side, naming exactly which keys are missing
        # instead of an opaque server-side rejection (AGENTS.md key convention
        # #6: no opaque server-side failures; fail fast with a clear message).
        required = list(self.embodiment.camera_keys)
        if required:
            missing = [k for k in required if k not in obs]
            if missing:
                covered = sorted({v for v in self._obs_mapping.values() if _is_image_key(v)})
                raise ValueError(
                    f"Cosmos3Policy(embodiment={self.embodiment.name!r}, "
                    f"action_space={self.action_space!r}) requires camera frames "
                    f"for: {required}. Missing: {missing}. observation_mapping "
                    f"covered: {covered}. Available observation keys: "
                    f"{sorted(robot_obs)}. Map a sim camera to each required key."
                )
        elif not any(k.startswith("observation/") and _is_image_key(k) for k in obs):
            # Degenerate embodiment with no declared camera_keys: still enforce
            # requires_images (always True) - at least one camera frame.
            raise ValueError(
                "Cosmos3Policy requires at least one camera frame, but none of the "
                f"mapped camera keys {sorted(self._obs_mapping)} were found in the "
                f"observation. Available observation keys: {sorted(robot_obs)}"
            )

        return obs

    def _attach_joint_state(self, robot_obs: dict[str, Any], obs: dict[str, Any]) -> None:
        """Build ``observation/joint_position`` + ``observation/gripper_position``.

        Priority:
            1. Explicit keys already present in robot_obs / via obs_mapping.
            2. ``robot_state_keys`` (first 7 = joints, a 'gripper'-named key).
            3. When no ``robot_state_keys`` was declared, the observation's own
               scalar keys in insertion order. That ordering is position-only:
               :func:`~strands_robots.policies._state_keys.drop_velocity_siblings`
               drops each ``<joint>.vel`` whose ``<joint>`` is present, so a sim
               observation - which emits a velocity companion beside every joint
               position - does not put a velocity in every other slot of the
               7-joint request and truncate the trailing joints away.
        """
        if "observation/joint_position" in obs and "observation/gripper_position" in obs:
            return  # already provided via mapping

        joints: list[float] = []
        gripper: float | None = None

        # Use declared state-key order when available; an ordering inferred from
        # the observation is position-only (see drop_velocity_siblings).
        state_keys = self.robot_state_keys or drop_velocity_siblings(
            [k for k, v in robot_obs.items() if np.isscalar(v) or np.ndim(v) == 0]
        )
        present = [k for k in state_keys if k in robot_obs]
        # First pass: pull any explicitly gripper/finger-named key as the gripper.
        gripper_keys = [k for k in present if ("gripper" in k.lower() or "finger" in k.lower())]
        if gripper_keys:
            gripper = float(np.asarray(robot_obs[gripper_keys[0]]).reshape(-1)[0])
        # Joints = the first 7 non-gripper state keys.
        for k in present:
            if k in gripper_keys:
                continue
            if len(joints) < 7:
                joints.append(float(np.asarray(robot_obs[k]).reshape(-1)[0]))
        # Fallback: if no gripper/finger-named key but we have an extra 8th
        # joint-like state key, treat it as the gripper.
        if gripper is None:
            non_gripper = [k for k in present if k not in gripper_keys]
            if len(non_gripper) >= 8:
                gripper = float(np.asarray(robot_obs[non_gripper[7]]).reshape(-1)[0])

        # joint_pos requires BOTH joints(7) and gripper - the server applies
        # `1 - gripper` and conditions on it. Never fabricate a silent default
        # (AGENTS.md key convention #6); surface a clear, actionable error.
        if "observation/joint_position" not in obs:
            if len(joints) < 7:
                raise ValueError(
                    "Cosmos3Policy(action_space='joint_pos') needs 7 joint state "
                    f"values but found {len(joints)}. Set robot_state_keys (7 joints "
                    "+ gripper) or pass an observation_mapping. "
                    f"Available observation keys: {sorted(robot_obs)}"
                )
            obs["observation/joint_position"] = np.asarray(joints[:7], dtype=np.float32).reshape(1, 7)
        if "observation/gripper_position" not in obs:
            if gripper is None:
                raise ValueError(
                    "Cosmos3Policy(action_space='joint_pos') could not find a "
                    "gripper state key (names containing 'gripper'/'finger', or an "
                    "8th state key). Set robot_state_keys with a gripper entry. "
                    f"Available observation keys: {sorted(robot_obs)}"
                )
            obs["observation/gripper_position"] = np.asarray([[gripper]], dtype=np.float32)

    def _active_action_layout(self) -> list[str]:
        """Column names for the action layout the active backend emits.

        The ``service`` backend returns the RoboLab server's post-processed
        ``action_space`` layout (``joint_pos`` / ``midtrain``); the ``diffusers``
        backend returns the model's raw unified action (``raw_action_layout``,
        e.g. DROID = 9D end-effector pose + 1D gripper). Returning the matching
        layout keeps ``_unpack_actions`` and ``action_mapping`` validation
        correct for both backends without duplicating either.
        """
        if self.backend == "diffusers":
            return list(self.embodiment.raw_action_layout)
        return list(self.embodiment.action_layouts.get(self.action_space, []))

    def _action_column_names(self, width: int) -> list[str]:
        """Resolve the per-column action names for the active backend's layout."""
        layout = self._active_action_layout()
        names = list(layout[:width])
        # Pad / fall back if the run returns a different width than expected.
        for i in range(len(names), width):
            names.append(f"action_{i}")
        return names

    def _unpack_actions(self, action: np.ndarray) -> list[dict[str, Any]]:
        """Split an ``[T, D]`` chunk into per-timestep actuator dicts."""
        if action.ndim == 1:
            action = action[None, :]
        if action.ndim != 2:
            raise ValueError(f"expected action chunk [T, D]; got shape {action.shape}")

        horizon, width = action.shape
        col_names = self._action_column_names(width)
        # Apply optional rename: layout column name → robot actuator name.
        out_names = [self._action_mapping.get(name, name) for name in col_names]

        steps: list[dict[str, Any]] = []
        for t in range(horizon):
            row = action[t]
            steps.append({out_names[d]: float(row[d]) for d in range(width)})
        return steps
