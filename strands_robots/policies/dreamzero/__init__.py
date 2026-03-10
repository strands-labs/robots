#!/usr/bin/env python3
"""
DreamZero Policy Provider — World Action Model inference via WebSocket.

DreamZero (NVIDIA GEAR Lab) is a 14B World Action Model that jointly predicts
actions and videos. It uses a Wan2.1 video DiT backbone with a flow matching
action head, achieving zero-shot performance on unseen tasks.

This provider connects to a running DreamZero inference server over WebSocket,
using the same protocol as the official eval client (msgpack-serialized numpy).

Architecture:
    Camera Images → [WebSocket] → DreamZero Server (multi-GPU) → Actions + Video

Requirements:
    - Running DreamZero server (see: github.com/dreamzero0/dreamzero)
    - pip install websockets msgpack-numpy (auto-installed with dreamzero)

Usage:
    from strands_robots.policies import create_policy

    policy = create_policy(
        "dreamzero",
        host="localhost",
        port=8000,
        instruction="pick up the red cube",
    )
    actions = await policy.get_actions(observation_dict, "pick up the red cube")

Server setup (requires multi-GPU):
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \\
        --standalone --nproc_per_node=2 \\
        socket_test_optimized_AR.py --port 8000 \\
        --enable-dit-cache --model-path <checkpoint>
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from strands_robots.policies import Policy

logger = logging.getLogger(__name__)


class DreamzeroPolicy(Policy):
    """DreamZero World Action Model — WebSocket inference client.

    Connects to a DreamZero server that jointly predicts actions and future
    video frames. The server runs the 14B model across multiple GPUs.

    Protocol (msgpack over WebSocket):
        1. Connect → server sends PolicyServerConfig (image_resolution, cameras, etc.)
        2. Send obs dict (numpy arrays + prompt + session_id) → receive actions (N, 8)
        3. Send reset → server saves video and resets state

    The action output is (N, 8): 7 joint positions + 1 gripper.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        instruction: str = "",
        session_id: Optional[str] = None,
        image_resize: Optional[tuple] = None,
        action_horizon: int = 24,
        **kwargs,
    ):
        """Initialize DreamZero policy client.

        Args:
            host: DreamZero server hostname
            port: DreamZero server port
            instruction: Default language instruction for the policy
            session_id: Session ID for tracking (auto-generated if None)
            image_resize: Override image resize (default: use server config)
            action_horizon: Expected action horizon from server
        """
        self._host = host
        self._port = port
        self._instruction = instruction
        self._session_id = session_id or str(uuid.uuid4())
        self._image_resize = image_resize
        self._action_horizon = action_horizon
        self._robot_state_keys: List[str] = []

        # Connection state (lazy connect)
        self._ws = None
        self._packer = None
        self._server_config = None
        self._connected = False
        self._step = 0

        logger.info(f"🌊 DreamZero policy initialized: ws://{host}:{port} " f"session={self._session_id[:8]}...")

    @property
    def provider_name(self) -> str:
        return "dreamzero"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        self._robot_state_keys = robot_state_keys
        logger.info(f"🔧 DreamZero robot state keys: {self._robot_state_keys}")

    # Maximum allowed image resolution from server config (prevents OOM from malicious configs)
    _MAX_IMAGE_RESOLUTION = (4096, 4096)

    def _validate_server_config(self, config: dict) -> dict:
        """Validate server config values to prevent resource exhaustion.

        Bounds-checks image_resolution to prevent a malicious or misconfigured
        server from causing excessive memory allocation.
        """
        if not isinstance(config, dict):
            logger.warning(f"Server config is not a dict ({type(config)}), using defaults")
            return {}

        img_res = config.get("image_resolution")
        if img_res is not None:
            try:
                h, w = int(img_res[0]), int(img_res[1])
                max_h, max_w = self._MAX_IMAGE_RESOLUTION
                if h <= 0 or w <= 0:
                    raise ValueError(f"image_resolution must be positive, got ({h}, {w})")
                if h > max_h or w > max_w:
                    logger.warning(f"Server image_resolution ({h}, {w}) exceeds max " f"({max_h}, {max_w}), clamping")
                    config["image_resolution"] = (min(h, max_h), min(w, max_w))
            except (TypeError, IndexError, ValueError) as e:
                logger.warning(f"Invalid image_resolution in server config: {img_res} ({e}), removing")
                config.pop("image_resolution", None)

        return config

    def _ensure_connected(self):
        """Lazily connect to the DreamZero server."""
        if self._connected:
            return

        try:
            import websockets.sync.client
        except ImportError:
            raise ImportError("DreamZero requires websockets: pip install websockets")

        try:
            from openpi_client import msgpack_numpy
        except ImportError:
            try:
                import msgpack_numpy
            except ImportError:
                raise ImportError("DreamZero requires msgpack-numpy: pip install msgpack-numpy")

        uri = f"ws://{self._host}:{self._port}"
        logger.info(f"🌊 Connecting to DreamZero server at {uri}...")

        try:
            self._ws = websockets.sync.client.connect(
                uri,
                compression=None,
                max_size=None,
                ping_interval=60,
                ping_timeout=600,
            )

            # Server sends config on connect
            raw_config = self._ws.recv()
            self._server_config = self._validate_server_config(msgpack_numpy.unpackb(raw_config))
            self._packer = msgpack_numpy.Packer()
            self._connected = True

            logger.info(f"🌊 Connected! Server config: {self._server_config}")

        except ConnectionRefusedError:
            raise ConnectionError(
                f"DreamZero server not running at ws://{self._host}:{self._port}. "
                f"Start it with: torchrun --nproc_per_node=2 "
                f"socket_test_optimized_AR.py --port {self._port}"
            )
        except Exception as e:
            # Try wss://
            uri_s = f"wss://{self._host}:{self._port}"
            logger.info(f"ws:// failed, trying {uri_s}...")
            try:
                self._ws = websockets.sync.client.connect(
                    uri_s,
                    compression=None,
                    max_size=None,
                    ping_interval=60,
                    ping_timeout=600,
                )
                raw_config = self._ws.recv()
                self._server_config = self._validate_server_config(msgpack_numpy.unpackb(raw_config))
                self._packer = msgpack_numpy.Packer()
                self._connected = True
                logger.info(f"🌊 Connected via wss! Config: {self._server_config}")
            except Exception:
                raise ConnectionError(f"Cannot connect to DreamZero server at {self._host}:{self._port}: {e}")

    def _build_observation(
        self,
        observation_dict: Dict[str, Any],
        instruction: str,
    ) -> Dict[str, Any]:
        """Convert strands-robots observation format to DreamZero protocol.

        DreamZero expects:
            - observation/exterior_image_{i}_left: (H, W, 3) uint8
            - observation/wrist_image_left: (H, W, 3) uint8
            - observation/joint_position: (7,) float32
            - observation/cartesian_position: (6,) float32
            - observation/gripper_position: (1,) float32
            - prompt: str
            - session_id: str
            - endpoint: "infer"

        strands-robots provides:
            - camera images as numpy arrays (various keys)
            - joint states as floats or arrays
        """
        obs = {}

        # Get server config for expected format
        cfg = self._server_config or {}
        img_res = self._image_resize or cfg.get("image_resolution", (180, 320))
        # Bounds check: cap resolution to prevent excessive memory allocation from
        # untrusted server config (max 4096x4096 is generous for any VLA use case)
        if isinstance(img_res, (list, tuple)) and len(img_res) == 2:
            img_res = (min(max(1, img_res[0]), 4096), min(max(1, img_res[1]), 4096))
        n_ext_cams = min(cfg.get("n_external_cameras", 2), 16)  # cap camera count
        needs_wrist = cfg.get("needs_wrist_camera", True)

        # Map camera images
        camera_keys = sorted([k for k in observation_dict if "camera" in k.lower() or "image" in k.lower()])

        ext_cam_idx = 0
        for key in camera_keys:
            img = observation_dict[key]
            if not isinstance(img, np.ndarray):
                continue

            # Resize if needed
            if img_res and img.shape[:2] != tuple(img_res):
                try:
                    from PIL import Image

                    pil_img = Image.fromarray(img)
                    pil_img = pil_img.resize((img_res[1], img_res[0]))
                    img = np.array(pil_img)
                except Exception:
                    pass

            # Assign to DreamZero camera slots
            if "wrist" in key.lower() and needs_wrist:
                obs["observation/wrist_image_left"] = img.astype(np.uint8)
            elif ext_cam_idx < n_ext_cams:
                obs[f"observation/exterior_image_{ext_cam_idx}_left"] = img.astype(np.uint8)
                ext_cam_idx += 1

        # Fill missing cameras with zeros
        h, w = img_res if img_res else (180, 320)
        for i in range(ext_cam_idx, n_ext_cams):
            obs[f"observation/exterior_image_{i}_left"] = np.zeros((h, w, 3), dtype=np.uint8)
        if needs_wrist and "observation/wrist_image_left" not in obs:
            obs["observation/wrist_image_left"] = np.zeros((h, w, 3), dtype=np.uint8)

        # Map joint state
        joint_pos = self._extract_joint_state(observation_dict, "joint_position", 7)
        cart_pos = self._extract_joint_state(observation_dict, "cartesian_position", 6)
        gripper_pos = self._extract_joint_state(observation_dict, "gripper_position", 1)

        obs["observation/joint_position"] = joint_pos
        obs["observation/cartesian_position"] = cart_pos
        obs["observation/gripper_position"] = gripper_pos

        # Language prompt
        obs["prompt"] = instruction or self._instruction
        obs["session_id"] = self._session_id
        obs["endpoint"] = "infer"

        return obs

    def _extract_joint_state(
        self,
        observation_dict: Dict[str, Any],
        key_hint: str,
        expected_dim: int,
    ) -> np.ndarray:
        """Extract joint state from observation dict, matching by key name."""
        # Direct key match
        for key in observation_dict:
            if key_hint in key.lower():
                val = observation_dict[key]
                if isinstance(val, np.ndarray):
                    return val.astype(np.float32).flatten()[:expected_dim]
                elif isinstance(val, (list, tuple)):
                    return np.array(val, dtype=np.float32)[:expected_dim]
                elif isinstance(val, (int, float)):
                    arr = np.zeros(expected_dim, dtype=np.float32)
                    arr[0] = float(val)
                    return arr

        # Try building from robot_state_keys
        if self._robot_state_keys and "joint" in key_hint:
            values = []
            for sk in self._robot_state_keys:
                if sk in observation_dict:
                    values.append(float(observation_dict[sk]))
            if values:
                arr = np.array(values, dtype=np.float32)
                if len(arr) >= expected_dim:
                    return arr[:expected_dim]
                else:
                    padded = np.zeros(expected_dim, dtype=np.float32)
                    padded[: len(arr)] = arr
                    return padded

        return np.zeros(expected_dim, dtype=np.float32)

    async def get_actions(
        self,
        observation_dict: Dict[str, Any],
        instruction: str,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Get actions from DreamZero server.

        Sends camera images + joint state + instruction via WebSocket,
        receives (N, 8) action array (7 joints + 1 gripper).

        Args:
            observation_dict: Robot observation (cameras + state)
            instruction: Natural language task instruction
            **kwargs: Additional options

        Returns:
            List of action dicts, one per timestep in the action horizon
        """
        self._ensure_connected()

        # Build DreamZero-format observation
        obs = self._build_observation(observation_dict, instruction)

        # Send and receive
        try:
            from openpi_client import msgpack_numpy
        except ImportError:
            import msgpack_numpy

        data = self._packer.pack(obs)
        self._ws.send(data)

        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"DreamZero server error:\n{response}")

        actions_array = msgpack_numpy.unpackb(response)

        if not isinstance(actions_array, np.ndarray):
            raise ValueError(f"Expected numpy array, got {type(actions_array)}")

        # Validate shape to prevent OOM from malicious server payloads
        if actions_array.ndim != 2:
            raise ValueError(f"Expected 2D actions array, got shape {actions_array.shape}")
        if actions_array.shape[0] > 1000 or actions_array.shape[1] > 100:
            raise ValueError(f"Actions array shape {actions_array.shape} exceeds safety bounds (1000, 100)")

        # Convert (N, 8) numpy array to list of action dicts
        actions = []
        for i in range(actions_array.shape[0]):
            action_vec = actions_array[i]
            action_dict = {}

            # Map to robot state keys if available
            if self._robot_state_keys:
                for j, key in enumerate(self._robot_state_keys):
                    if j < len(action_vec) - 1:  # Last dim is gripper
                        action_dict[key] = float(action_vec[j])
                action_dict["gripper"] = float(action_vec[-1])
            else:
                # Generic joint naming
                for j in range(min(7, len(action_vec) - 1)):
                    action_dict[f"joint_{j}"] = float(action_vec[j])
                action_dict["gripper"] = float(action_vec[-1])

            actions.append(action_dict)

        self._step += 1
        logger.debug(
            f"🌊 DreamZero step {self._step}: "
            f"received {len(actions)} actions, "
            f"range [{actions_array.min():.3f}, {actions_array.max():.3f}]"
        )

        return actions

    def reset(self):
        """Reset the DreamZero session (triggers video save on server)."""
        if not self._connected or self._ws is None:
            return

        try:
            from openpi_client import msgpack_numpy
        except ImportError:
            import msgpack_numpy  # noqa: F401

        reset_msg = {"endpoint": "reset"}
        self._ws.send(self._packer.pack(reset_msg))
        self._ws.recv()  # "reset successful"

        # New session
        self._session_id = str(uuid.uuid4())
        self._step = 0
        logger.info(f"🌊 DreamZero reset. New session: {self._session_id[:8]}...")

    def close(self):
        """Close the WebSocket connection."""
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
            self._connected = False
            logger.info("🌊 DreamZero connection closed")

    def __del__(self):
        self.close()


# Alias for PEP8-style name (DreamZeroPolicy is the preferred public name)
DreamZeroPolicy = DreamzeroPolicy

__all__ = ["DreamzeroPolicy", "DreamZeroPolicy"]
