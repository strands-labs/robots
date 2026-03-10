#!/usr/bin/env python3
"""
Auto-resolve a HuggingFace model ID or shorthand to (provider, kwargs).

Maps human-friendly policy strings to the correct provider + config:
    "lerobot/act_aloha_sim"  → ("lerobot_local", {"pretrained_name_or_path": "lerobot/act_aloha_sim"})
    "mock"                   → ("mock", {})
    "localhost:8080"         → ("lerobot_async", {"server_address": "localhost:8080"})
    "ws://gpu:9000"          → ("dreamzero", {"host": "gpu", "port": 9000})
"""

import logging
import re
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)

__all__ = ["resolve_policy"]

# ─────────────────────────────────────────────────────────────────────
# Prefix → provider mapping
# ─────────────────────────────────────────────────────────────────────

_HF_ORG_TO_PROVIDER: Dict[str, str] = {
    # LeRobot models → lerobot_local (direct HF inference)
    "lerobot": "lerobot_local",
    # OpenVLA
    "openvla": "openvla",
    # Microsoft
    "microsoft": "magma",  # Magma-8B
    # InternRobotics
    "internrobotics": "internvla",
    # Robotics Diffusion Transformer
    "robotics-diffusion-transformer": "rdt",
    # Unitree
    "unitreerobotics": "unifolm",
    # BAAI
    "baai": "robobrain",
    # NVIDIA
    "nvidia": "groot",  # GR00T models default
    # CogACT
    "cogact": "cogact",
    # Dream-org
    "dream-org": "dreamgen",
    # AgiBot World
    "agibot-world": "go1",
}

# Explicit model ID → provider overrides (for ambiguous orgs like nvidia)
_MODEL_ID_OVERRIDES: Dict[str, str] = {
    "nvidia/gr00t": "groot",
    "nvidia/groot": "groot",
    "nvidia/alpamayo": "alpamayo",
    "nvidia/dreamzero": "dreamzero",
    "nvidia/cosmos-predict": "cosmos_predict",
    "nvidia/cosmos-predict2": "cosmos_predict",
    "nvidia/cosmos-predict2.5-2b": "cosmos_predict",
    "nvidia/cosmos-predict2.5-14b": "cosmos_predict",
    "nvidia/cosmos-policy": "cosmos_predict",
    # AgiBot World GO-1
    "agibot-world/go-1": "go1",
    "agibot-world/go-1-air": "go1",
}

# Shorthand names → provider
_SHORTHAND_TO_PROVIDER: Dict[str, str] = {
    "mock": "mock",
    "random": "mock",
    "test": "mock",
    "groot": "groot",
    "dreamgen": "dreamgen",
    "dreamzero": "dreamzero",
    "cosmos": "cosmos_predict",
    "cosmos_predict": "cosmos_predict",
    "go1": "go1",
}


def resolve_policy(
    policy: str,
    **extra_kwargs,
) -> Tuple[str, Dict[str, Any]]:
    """Resolve a policy string to (provider_name, provider_kwargs).

    Accepts:
        - HuggingFace model IDs: "lerobot/act_aloha_sim_transfer_cube_human"
        - Server addresses: "localhost:8080", "ws://gpu:9000"
        - Shorthand names: "mock", "groot"
        - Explicit provider: "lerobot_local", "openvla", etc.

    Returns:
        (provider_name, kwargs_dict) ready for create_policy()

    Examples:
        >>> resolve_policy("lerobot/act_aloha_sim_transfer_cube_human")
        ("lerobot_local", {"pretrained_name_or_path": "lerobot/act_aloha_sim_transfer_cube_human"})

        >>> resolve_policy("mock")
        ("mock", {})

        >>> resolve_policy("localhost:8080")
        ("lerobot_async", {"server_address": "localhost:8080"})

        >>> resolve_policy("ws://gpu-server:9000")
        ("dreamzero", {"host": "gpu-server", "port": 9000})

        >>> resolve_policy("openvla/openvla-7b")
        ("openvla", {"model_id": "openvla/openvla-7b"})
    """
    policy = policy.strip()
    kwargs: Dict[str, Any] = {}

    # ── 1. Server addresses ──
    # ws:// or wss:// → dreamzero
    if policy.startswith("ws://") or policy.startswith("wss://"):
        match = re.match(r"wss?://([^:]+):(\d+)", policy)
        if match:
            kwargs["host"] = match.group(1)
            kwargs["port"] = int(match.group(2))
        else:
            kwargs["host"] = policy.replace("ws://", "").replace("wss://", "").split(":")[0]
            kwargs["port"] = 8000
        kwargs.update(extra_kwargs)
        return "dreamzero", kwargs

    # host:port pattern (no /) → lerobot_async gRPC
    if re.match(r"^[\w.\-]+:\d+$", policy) and "/" not in policy:
        kwargs["server_address"] = policy
        kwargs.update(extra_kwargs)
        return "lerobot_async", kwargs

    # grpc:// prefix → lerobot_async
    if policy.startswith("grpc://"):
        kwargs["server_address"] = policy.replace("grpc://", "")
        kwargs.update(extra_kwargs)
        return "lerobot_async", kwargs

    # zmq:// prefix → groot
    if policy.startswith("zmq://"):
        match = re.match(r"zmq://([^:]+):(\d+)", policy)
        if match:
            kwargs["host"] = match.group(1)
            kwargs["port"] = int(match.group(2))
        kwargs.update(extra_kwargs)
        return "groot", kwargs

    # ── 2. Shorthand names ──
    if policy.lower() in _SHORTHAND_TO_PROVIDER:
        provider = _SHORTHAND_TO_PROVIDER[policy.lower()]
        kwargs.update(extra_kwargs)
        return provider, kwargs

    # ── 3. HuggingFace model IDs (org/model) ──
    if "/" in policy:
        org = policy.split("/")[0].lower()

        # Check explicit overrides first
        policy_lower = policy.lower()
        for prefix, provider in _MODEL_ID_OVERRIDES.items():
            if policy_lower.startswith(prefix):
                kwargs["pretrained_name_or_path"] = policy
                kwargs.update(extra_kwargs)
                return provider, kwargs

        # Check org → provider mapping
        if org in _HF_ORG_TO_PROVIDER:
            provider = _HF_ORG_TO_PROVIDER[org]

            # Different providers need the model ID in different kwargs
            if provider in ("lerobot_local", "lerobot_async"):
                kwargs["pretrained_name_or_path"] = policy
            elif provider in ("openvla", "internvla", "magma", "rdt", "unifolm", "robobrain", "cogact", "go1"):
                kwargs["model_id"] = policy
            elif provider in ("groot", "dreamgen"):
                kwargs["pretrained_name_or_path"] = policy
            else:
                kwargs["pretrained_name_or_path"] = policy

            kwargs.update(extra_kwargs)
            return provider, kwargs

        # Unknown org but has / → assume HF model, try lerobot_local
        logger.info(f"Unknown HF org '{org}', defaulting to lerobot_local for '{policy}'")
        kwargs["pretrained_name_or_path"] = policy
        kwargs.update(extra_kwargs)
        return "lerobot_local", kwargs

    # ── 4. Bare provider name ──
    # Check if it's a registered provider name
    try:
        from strands_robots.policies import list_providers

        all_providers = list_providers()
        if policy.lower() in all_providers:
            kwargs.update(extra_kwargs)
            return policy.lower(), kwargs
    except ImportError:
        pass

    # ── 5. Fallback: assume it's a local path or model name ──
    kwargs["pretrained_name_or_path"] = policy
    kwargs.update(extra_kwargs)
    return "lerobot_local", kwargs
