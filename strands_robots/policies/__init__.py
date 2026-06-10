"""Policy Abstraction for VLA, motion planners, MPC, and scripted controllers.

Plugin-based registry - all provider definitions live in registry/policies.json.
No hardcoded if/elif chains. New providers are auto-discovered or registered at runtime.

The :class:`Policy` ABC is intentionally agnostic about *how* actions are
produced, so the same interface fits VLA-style providers (consume images +
instruction) and non-VLA providers (cuRobo, MoveIt2, MPC, pure-IK / scripted
trajectories).  Non-VLA providers typically set ``requires_images=False`` and
read their goal from the well-known ``**kwargs`` keys (``target_pose``,
``target_joints``, ``world_update``) documented on
:meth:`Policy.get_actions`.

Built-in providers (see policies.json for full list):
    - mock: Sinusoidal test actions (non-VLA reference, ``requires_images=False``)
    - groot: NVIDIA GR00T via ZMQ
    - lerobot_local: Direct HuggingFace inference (ACT, Pi0, SmolVLA, Diffusion, ...)

Usage::

    from strands_robots.policies import create_policy, Policy

    # By provider name
    policy = create_policy("groot", port=5555)
    policy = create_policy("lerobot_local",
        pretrained_name_or_path="lerobot/act_aloha_sim_transfer_cube_human")

    # By smart string (auto-resolves provider)
    policy = create_policy("lerobot/act_aloha_sim")
    policy = create_policy("zmq://localhost:5555")
    policy = create_policy("mock")

    # Custom provider
    register_policy("my_provider", lambda: MyPolicy, aliases=["my"])
"""

from strands_robots.policies.base import Policy

# Cosmos3Policy is import-safe: it depends only on numpy. The WebSocket
# client uses a self-contained msgpack+websockets transport (no
# ``openpi-client`` dependency).
from strands_robots.policies.cosmos3 import Cosmos3Policy
from strands_robots.policies.factory import (
    UntrustedRemoteCodeError,
    create_policy,
    list_providers,
    register_policy,
)
from strands_robots.policies.mock import MockPolicy

__all__ = [
    "Policy",
    "MockPolicy",
    "Cosmos3Policy",
    "create_policy",
    "register_policy",
    "list_providers",
    "UntrustedRemoteCodeError",
]
