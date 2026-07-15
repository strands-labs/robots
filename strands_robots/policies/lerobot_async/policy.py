"""Remote lerobot inference over gRPC - client to a lerobot ``PolicyServer``.

:class:`LerobotAsyncPolicy` is a drop-in :class:`~strands_robots.policies.base.Policy`
that offloads inference to a remote lerobot ``PolicyServer`` (lerobot's native
async-inference gRPC transport, ``lerobot.async_inference.policy_server``).
Every :meth:`get_actions` call forwards the current observation to the server,
which runs the configured lerobot policy (ACT, SmolVLA, diffusion, pi0/pi0.5,
VQBeT, ...) on its own GPU and streams back an action chunk. Because it
satisfies the ``Policy`` ABC it works anywhere a local policy does:
``sim.run_policy(policy_provider="lerobot_async", ...)``,
``sim.eval_policy(...)``, or a hardware control loop that calls
:func:`~strands_robots.policies.create_policy`.

Unlike ``lerobot_local`` (which loads the checkpoint in-process, tying the
robot loop to the GPU that holds the weights), the async client keeps the robot
side light: it sends observations and applies returned actions while the
heavyweight model lives on a separate server. This mirrors lerobot's own
``robot_client`` / ``policy_server`` split and is the right shape for edge
robots (e.g. a Jetson) driven by a policy running on a datacentre GPU.

Start the server first (on the GPU host)::

    pip install 'lerobot[async]'
    python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080

Then point the client at it::

    from strands_robots import create_policy

    policy = create_policy(
        "lerobot_async",
        server_address="gpu-box:8080",
        policy_type="act",
        pretrained_name_or_path="lerobot/act_so101",
    )
    # or via a smart string (server_address parsed from the URL):
    policy = create_policy(
        "grpc://gpu-box:8080",
        policy_type="act",
        pretrained_name_or_path="lerobot/act_so101",
    )

The client selects which checkpoint the server loads (via lerobot's
``SendPolicyInstructions`` handshake), so ``policy_type`` and
``pretrained_name_or_path`` are required. The connection is established lazily
on first :meth:`get_actions`, so constructing the policy does not require the
server to be up yet.
"""

from __future__ import annotations

import logging
import pickle  # nosec B403 - lerobot's async transport serializes obs/actions with pickle
import threading
import time
from typing import Any

import numpy as np

from strands_robots.policies.base import Policy
from strands_robots.utils import require_optional

logger = logging.getLogger(__name__)

#: Policy types a lerobot ``PolicyServer`` can serve, mirroring
#: ``lerobot.async_inference.constants.SUPPORTED_POLICIES``. Validated
#: client-side so a typo fails fast with the valid set instead of only
#: surfacing after the gRPC handshake.
SUPPORTED_POLICY_TYPES: tuple[str, ...] = (
    "act",
    "smolvla",
    "diffusion",
    "tdmpc",
    "vqbet",
    "pi0",
    "pi05",
    "groot",
)

#: Default gRPC handshake timeout (seconds).
DEFAULT_CONNECT_TIMEOUT = 10.0
#: Default per-request timeout (seconds). A remote VLA chunk can take a while.
DEFAULT_REQUEST_TIMEOUT = 60.0


class LerobotAsyncPolicy(Policy):
    """Client-side policy that runs inference on a remote lerobot ``PolicyServer``.

    Args:
        server_address: Server address as ``host:port`` (gRPC). When given it
            takes precedence over ``host``/``port``. A ``grpc://`` scheme is
            stripped if present.
        host: Server host (used when ``server_address`` is not given).
        port: Server port (used when ``server_address`` is not given).
        policy_type: lerobot policy type the server should load (one of
            :data:`SUPPORTED_POLICY_TYPES`). Required.
        pretrained_name_or_path: HuggingFace model id or path the server loads.
            Required.
        device: Device string for **server-side** inference (e.g. ``"cuda"`` or
            ``"cpu"``). Defaults to ``"cuda"`` - the async provider exists to
            offload inference to a GPU host; override with ``device="cpu"`` for a
            CPU server.
        actions_per_chunk: Max number of actions the server returns per chunk.
        actions_per_step: Number of actions the consumer executes from one chunk
            before re-querying (the :attr:`execution_horizon`). Defaults to
            ``actions_per_chunk`` so a chunked open-loop policy (ACT, diffusion)
            executes the whole chunk per network round-trip instead of paying a
            round-trip per control step.
        connect_timeout: Seconds to wait for the gRPC ``Ready`` handshake.
        request_timeout: Seconds to wait for each observation/action RPC.
        rename_map: Optional ``{robot_obs_key: model_feature_key}`` map forwarded
            to the server's ``RemotePolicyConfig.rename_map``. The server applies
            it as a ``RenameObservationsProcessorStep`` (renaming each matching
            observation key to its mapped name) before the policy sees the
            observation - the async analog of the ``lerobot_local`` provider's
            ``obs_rename``. Use it when the checkpoint expects camera/state keys
            that differ from the ones the robot exposes (e.g.
            ``{"observation.images.front": "observation.images.laptop"}``); keys
            not present in the map pass through unchanged.

    Unrecognized kwargs are ignored (for forward-compatible ``policy_config``
    passthrough via :func:`~strands_robots.policies.create_policy`) but logged
    at WARNING, so a mistyped connection kwarg does not silently leave the
    client pointed at the default address.

    Raises:
        ValueError: If ``policy_type`` / ``pretrained_name_or_path`` are missing
            or ``policy_type`` is not server-supported.
        ConnectionError: On first use, if the server cannot be reached.
    """

    def __init__(
        self,
        server_address: str | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
        policy_type: str | None = None,
        pretrained_name_or_path: str | None = None,
        device: str = "cuda",
        actions_per_chunk: int = 50,
        actions_per_step: int | None = None,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        rename_map: dict[str, str] | None = None,
        **ignored_kwargs: Any,
    ) -> None:
        address = server_address or f"{host}:{port}"
        if "://" in address:
            address = address.split("://", 1)[1]
        self.server_address = address

        if not policy_type:
            raise ValueError(
                "lerobot_async requires policy_type=... (the lerobot policy the "
                f"server loads); one of {SUPPORTED_POLICY_TYPES}."
            )
        if policy_type not in SUPPORTED_POLICY_TYPES:
            raise ValueError(
                f"policy_type {policy_type!r} is not served by a lerobot PolicyServer; "
                f"choose one of {SUPPORTED_POLICY_TYPES}."
            )
        if not pretrained_name_or_path:
            raise ValueError(
                "lerobot_async requires pretrained_name_or_path=... (the checkpoint "
                "the server loads via SendPolicyInstructions)."
            )

        self.policy_type = policy_type
        self.pretrained_name_or_path = pretrained_name_or_path
        self.device = device
        self.actions_per_chunk = int(actions_per_chunk)
        self.actions_per_step = int(actions_per_step) if actions_per_step is not None else self.actions_per_chunk
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout
        if rename_map is not None and not isinstance(rename_map, dict):
            raise ValueError(
                "lerobot_async: rename_map must be a dict mapping each robot observation "
                f"key to the model's expected feature key, got {type(rename_map).__name__}."
            )
        self.rename_map: dict[str, str] = dict(rename_map) if rename_map else {}

        if ignored_kwargs:
            logger.warning(
                "LerobotAsyncPolicy ignoring unexpected constructor kwarg(s) %s; "
                "connecting to %s. Set the server via server_address= (or host=/port=); "
                "server-side policy config belongs on the PolicyServer, not the client.",
                sorted(ignored_kwargs),
                self.server_address,
            )

        self.robot_state_keys: list[str] = []

        # Lazily initialised gRPC state (typed Any: modules imported on demand).
        self._grpc: Any = None
        self._pb2: Any = None
        self._pb2_grpc: Any = None
        self._channel: Any = None
        self._stub: Any = None

        self._instructions_sent = False
        self._timestep = 0
        self._lock = threading.Lock()

    # -- Policy metadata ------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "lerobot_async"

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self.robot_state_keys = list(robot_state_keys)

    def reset(self, seed: int | None = None) -> None:
        """Reset for a new episode.

        Restarts the client timestep counter and, if already connected, calls
        the server's ``Ready`` RPC. ``Ready`` flushes the server's observation
        queue and predicted-timestep set (its per-episode state) while keeping
        the loaded policy resident - so fresh observations in the next episode
        are not deduped against the previous one's timesteps.

        Args:
            seed: Accepted for interface compatibility; the loaded policy's RNG
                lives server-side and is not seedable through this transport.
        """
        self._timestep = 0
        if self._stub is not None:
            try:
                self._stub.Ready(self._pb2.Empty(), timeout=self.request_timeout)
            except self._grpc.RpcError as exc:  # pragma: no cover - network dependent
                logger.warning("lerobot_async: server Ready() during reset failed: %s", exc)

    # -- Connection lifecycle -------------------------------------------------

    def _ensure_connected(self) -> None:
        """Open the gRPC channel and complete the ``Ready`` handshake (idempotent)."""
        if self._stub is not None:
            return

        grpc: Any = require_optional(
            "grpc",
            pip_install="grpcio",
            extra="lerobot-async",
            purpose="the lerobot async-inference gRPC transport",
        )
        from lerobot.transport import services_pb2, services_pb2_grpc
        from lerobot.transport.utils import grpc_channel_options

        self._grpc = grpc
        self._pb2 = services_pb2
        self._pb2_grpc = services_pb2_grpc
        self._channel = grpc.insecure_channel(self.server_address, grpc_channel_options())
        stub = services_pb2_grpc.AsyncInferenceStub(self._channel)
        try:
            stub.Ready(services_pb2.Empty(), timeout=self.connect_timeout)
        except grpc.RpcError as exc:
            self._channel.close()
            self._channel = None
            raise ConnectionError(
                f"LerobotAsyncPolicy could not reach a lerobot PolicyServer at "
                f"{self.server_address}. Start one first, e.g.:\n"
                f"  python -m lerobot.async_inference.policy_server "
                f"--host=0.0.0.0 --port={self.server_address.rsplit(':', 1)[-1]}\n"
                f"Underlying error: {type(exc).__name__}: {exc}"
            ) from exc
        self._stub = stub
        logger.info("LerobotAsyncPolicy connected to lerobot PolicyServer at %s", self.server_address)

    def _send_instructions(self, lerobot_features: dict[str, Any]) -> None:
        """Tell the server which checkpoint to load (lerobot ``SendPolicyInstructions``)."""
        from lerobot.async_inference.helpers import RemotePolicyConfig

        cfg = RemotePolicyConfig(
            policy_type=self.policy_type,
            pretrained_name_or_path=self.pretrained_name_or_path,
            lerobot_features=lerobot_features,
            actions_per_chunk=self.actions_per_chunk,
            device=self.device,
            rename_map=self.rename_map,
        )
        self._stub.SendPolicyInstructions(
            self._pb2.PolicySetup(data=pickle.dumps(cfg)),  # nosec B301
            timeout=self.request_timeout,
        )
        self._instructions_sent = True
        logger.info(
            "lerobot_async: server loading %s (%s) on %s",
            self.pretrained_name_or_path,
            self.policy_type,
            self.device,
        )

    def close(self) -> None:
        """Close the gRPC channel. Safe to call more than once."""
        with self._lock:
            if self._channel is not None:
                try:
                    self._channel.close()
                finally:
                    self._channel = None
                    self._stub = None

    # -- Observation / action wire conversion ---------------------------------

    def _camera_items(self, observation_dict: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
        """Return ``(key, HWC array)`` pairs for RGB/depth camera entries."""
        cams: list[tuple[str, np.ndarray]] = []
        for key, value in observation_dict.items():
            if key in self.robot_state_keys or key == "task":
                continue
            arr = np.asarray(value)
            if arr.ndim == 3 and arr.shape[2] in (1, 3):
                cams.append((key, arr))
        return cams

    def _build_lerobot_features(self, observation_dict: dict[str, Any]) -> dict[str, Any]:
        """Build the lerobot feature spec the server uses to decode observations.

        Mirrors lerobot's ``hw_to_dataset_features`` over a hardware-feature map
        assembled from the declared joint state keys (scalars concatenated into
        ``observation.state``) and the camera keys present in the observation
        (``observation.images.<name>``).
        """
        from lerobot.utils.constants import OBS_STR
        from lerobot.utils.feature_utils import hw_to_dataset_features

        if not self.robot_state_keys:
            raise RuntimeError(
                "lerobot_async: robot_state_keys is empty; call set_robot_state_keys() "
                "with the robot's joint/motor names before inference."
            )

        hw_features: dict[str, Any] = {key: float for key in self.robot_state_keys}
        for key, arr in self._camera_items(observation_dict):
            hw_features[key] = tuple(int(d) for d in arr.shape)
        return hw_to_dataset_features(hw_features, OBS_STR, use_video=False)

    def _to_raw_observation(self, observation_dict: dict[str, Any], instruction: str) -> dict[str, Any]:
        """Build the lerobot ``RawObservation`` the server expects."""
        raw: dict[str, Any] = {}
        for key in self.robot_state_keys:
            if key not in observation_dict:
                raise RuntimeError(
                    f"lerobot_async: observation is missing declared state key {key!r}. "
                    f"Declared keys: {self.robot_state_keys}; observation keys: "
                    f"{sorted(observation_dict)}."
                )
            raw[key] = float(observation_dict[key])
        for key, arr in self._camera_items(observation_dict):
            raw[key] = arr
        if instruction:
            raw["task"] = instruction
        return raw

    def _request_action_chunk(self, raw_obs: dict[str, Any]) -> list[Any]:
        """Send one observation and return the server's ``list[TimedAction]`` chunk."""
        from lerobot.async_inference.helpers import TimedObservation
        from lerobot.transport.utils import send_bytes_in_chunks

        timed = TimedObservation(
            timestamp=time.time(),
            timestep=self._timestep,
            observation=raw_obs,
            must_go=True,
        )
        self._timestep += 1
        payload = pickle.dumps(timed)  # nosec B301

        self._stub.SendObservations(
            send_bytes_in_chunks(payload, self._pb2.Observation),
            timeout=self.request_timeout,
        )
        actions = self._stub.GetActions(self._pb2.Empty(), timeout=self.request_timeout)
        data = getattr(actions, "data", b"")
        if not data:
            raise RuntimeError(
                "lerobot_async: server returned no actions for the observation. The "
                "observation was filtered out or server-side inference failed; check "
                "the PolicyServer logs."
            )
        return pickle.loads(data)  # nosec B301

    def _chunk_to_action_dicts(self, chunk: list[Any]) -> list[dict[str, Any]]:
        """Convert the server's ``list[TimedAction]`` into per-tick action dicts.

        Each ``TimedAction.action`` is a 1D tensor over the policy's action
        dimensions; values are mapped to :attr:`robot_state_keys` by index and
        the chunk is capped at :attr:`execution_horizon` (the re-query interval
        the consumer drives).
        """
        if not self.robot_state_keys:
            raise RuntimeError(
                "lerobot_async: robot_state_keys is empty; call set_robot_state_keys() before inference."
            )

        result: list[dict[str, Any]] = []
        for timed_action in chunk[: self.execution_horizon]:
            action = timed_action.get_action() if hasattr(timed_action, "get_action") else timed_action.action
            values = np.asarray(action.detach().cpu().numpy() if hasattr(action, "detach") else action).flatten()
            result.append(
                {key: (float(values[i]) if i < len(values) else 0.0) for i, key in enumerate(self.robot_state_keys)}
            )
        if not result:
            raise RuntimeError("lerobot_async: server returned an empty action chunk.")
        return result

    # -- Inference ------------------------------------------------------------

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Forward the observation to the remote server and return its action chunk.

        On the first call the client connects, completes the ``Ready``
        handshake, and sends the checkpoint-load instructions (built from the
        declared state keys plus the cameras present in this observation).

        Args:
            observation_dict: Robot observation (per-joint scalars keyed by name
                plus bare camera-name RGB/depth arrays).
            instruction: Natural-language task, forwarded to the server policy.
            **kwargs: Ignored (forward-compatible passthrough).

        Returns:
            List of action dicts, each mapping every robot state key to a python
            ``float`` target for that tick.

        Raises:
            ConnectionError: If the server cannot be reached on first use.
            RuntimeError: If state keys are undeclared, the observation is
                missing a declared key, or the server returns no actions.
        """
        with self._lock:
            self._ensure_connected()
            if not self._instructions_sent:
                self._send_instructions(self._build_lerobot_features(observation_dict))
            raw_obs = self._to_raw_observation(observation_dict, instruction)
            chunk = self._request_action_chunk(raw_obs)
        return self._chunk_to_action_dicts(chunk)
