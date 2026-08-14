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

from typing import TYPE_CHECKING

from strands_robots.policies.base import (
    ChunkedPolicy,
    Policy,
    align_action_values,
    chunk_count_error,
    resolve_chunk_length,
)

# Cosmos3Policy is import-safe: it depends only on numpy. The WebSocket
# client uses a self-contained msgpack+websockets transport (no
# ``openpi-client`` dependency).
from strands_robots.policies.composite import CompositePolicy
from strands_robots.policies.cosmos3 import Cosmos3Policy
from strands_robots.policies.factory import (
    UntrustedRemoteCodeError,
    create_policy,
    list_providers,
    policy_mapping_error,
    policy_provider_error,
    preflight_policy,
    register_policy,
)
from strands_robots.policies.mock import MockPolicy
from strands_robots.policies.persistent import (
    PersistentPolicy,
    evict,
    list_cached,
    preload,
)

__all__ = [
    "Policy",
    "ChunkedPolicy",
    "resolve_chunk_length",
    "align_action_values",
    "chunk_count_error",
    "MockPolicy",
    "Cosmos3Policy",
    "CompositePolicy",
    "create_policy",
    "preflight_policy",
    "policy_provider_error",
    "policy_mapping_error",
    "register_policy",
    "list_providers",
    "list_policy_types",
    "UntrustedRemoteCodeError",
    "PersistentPolicy",
    "preload",
    "list_cached",
    "evict",
]


if TYPE_CHECKING:
    # Static-analysis import so ``list_policy_types`` (resolved lazily below)
    # is a defined export for type-checkers / CodeQL py/undefined-export.
    from strands_robots.policies.lerobot_local.resolution import list_policy_types


def __getattr__(name: str) -> object:
    """Lazily expose the ``lerobot_local`` policy-type discovery surface.

    ``list_policy_types()`` answers "which ``policy_type`` strings can I pass to
    ``create_policy('lerobot_local', policy_type=...)``?" -- the natural follow-up
    to ``list_providers()``. It lives in the ``lerobot_local`` package, whose
    eager import chain pulls in torch, so it is resolved on first access (PEP
    562) rather than at ``import strands_robots.policies`` time. This keeps the
    package import torch-free while still exposing the discovery peer next to
    ``list_providers``.
    """
    if name == "list_policy_types":
        from strands_robots.policies.lerobot_local.resolution import (
            list_policy_types as _list_policy_types,
        )

        globals()["list_policy_types"] = _list_policy_types
        return _list_policy_types
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
