"""Does this policy fit that robot, and what can the run form ask for.

Two halves. :func:`policy_fit` compares a checkpoint's declared features with
what a peer announces (joints, cameras, normalisation tags) and names every
mismatch in the operator's terms. :func:`policy_catalog` is the provider
catalog the run form is generated from: every registered provider, split by
which of its inputs the mesh wire can actually carry.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "camera_keys",
    "state_dim",
    "action_dim",
    "policy_fit",
    "policy_catalog",
    "WIRE_CMD_KEYS",
    "WIRE_KEY_TYPES",
]

logger = logging.getLogger(__name__)

#: lerobot's prefix for image observations. Everything after it is the camera's name.
_IMAGE_PREFIX = "observation.images."


def _shape(entry: Any) -> list[int] | None:
    if not isinstance(entry, Mapping):
        return None
    shape = entry.get("shape")
    if isinstance(shape, (list, tuple)) and all(isinstance(n, int) for n in shape):
        return [int(n) for n in shape]
    return None


def camera_keys(features: Mapping[str, Any] | None) -> list[str]:
    """The camera NAMES a policy expects, in declaration order."""
    out: list[str] = []
    for key in features or {}:
        if isinstance(key, str) and key.startswith(_IMAGE_PREFIX):
            name = key[len(_IMAGE_PREFIX) :].strip()
            if name and name not in out:
                out.append(name)
    return out


def state_dim(features: Mapping[str, Any] | None) -> int | None:
    """How many state values the policy was trained to read, or None when not stated."""
    shape = _shape((features or {}).get("observation.state"))
    return shape[0] if shape else None


def action_dim(features: Mapping[str, Any] | None) -> int | None:
    """How many values the policy emits per step, or None when not stated."""
    shape = _shape((features or {}).get("action"))
    return shape[0] if shape else None


def policy_fit(
    *,
    input_features: Mapping[str, Any] | None = None,
    output_features: Mapping[str, Any] | None = None,
    joints: Iterable[str] | None = None,
    cameras: Iterable[str] | None = None,
    physical: bool = True,
    norm_tag: str | None = None,
    declared_norm_tags: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compare a checkpoint's declared features with what the target peer announces. Returns ``{ok,
    blocking, problems, checked}``.
    """
    joint_names = [j for j in (joints or []) if isinstance(j, str)]
    camera_names = [c for c in (cameras or []) if isinstance(c, str)]
    problems: list[dict[str, str]] = []
    checked: list[str] = []

    metal = "a real arm" if physical else "the simulated robot"

    tags = [t for t in (declared_norm_tags or []) if isinstance(t, str)]
    wanted = (norm_tag or "").strip()
    if wanted and tags:
        if wanted not in tags:
            problems.append(
                {
                    "kind": "norm_tag",
                    "detail": (
                        f"this checkpoint declares no normalisation stats for {wanted!r}, so its inputs "
                        f"would be scaled by the wrong statistics and its actions would drive {metal} to "
                        f"the wrong places - pick one of the tags it does declare: {', '.join(tags)}"
                    ),
                }
            )
        else:
            checked.append("norm_tag")

    sd = state_dim(input_features)
    ad = action_dim(output_features)
    needed = camera_keys(input_features)

    if joint_names:
        n = len(joint_names)
        if sd is not None:
            checked.append("state")
            if sd != n:
                problems.append(
                    {
                        "kind": "state_dim",
                        "detail": (
                            f"this policy reads a {sd}-value state and this robot reports {n} joints "
                            f"({', '.join(joint_names)}). It was trained on different hardware: the "
                            f"observation cannot be assembled, and the run fails after {metal} has "
                            f"already been energised and parked"
                        ),
                    }
                )
        if ad is not None:
            checked.append("action")
            if ad != n:
                problems.append(
                    {
                        "kind": "action_dim",
                        "detail": (
                            f"this policy emits {ad} value(s) per step and this robot has {n} joints. "
                            f"Those numbers cannot be joint commands for this arm - at best the run "
                            f"errors with {metal} torqued, at worst the values land on the wrong joints"
                        ),
                    }
                )

    if needed and camera_names:
        checked.append("cameras")
        missing = [c for c in needed if c not in camera_names]
        if missing:
            problems.append(
                {
                    "kind": "cameras",
                    "detail": (
                        f"this policy was trained with camera(s) {', '.join(missing)} and this robot "
                        f"announces {', '.join(camera_names)}. Without that view the policy sees no "
                        f"image for it, so it acts on a blank frame rather than on the scene - and it "
                        f"will not say so"
                    ),
                }
            )

    return {
        "ok": not problems,
        # None of these are forceable: a tick cannot make 2 numbers drive 6 joints.
        "blocking": bool(problems),
        "problems": problems,
        "checked": checked,
    }


# ---------------------------------------------------------------------------
# the provider catalog
# ---------------------------------------------------------------------------

#: Keys ``mesh.security.validate_command()`` admits for execute/start, minus the ones the
#: dashboard sets itself (action/instruction/policy_provider/duration).
WIRE_CMD_KEYS: tuple[str, ...] = (
    "policy_host",
    "policy_port",
    "policy_type",
    "server_address",
    "model_path",
    "pretrained_name_or_path",
    "robot_name",
    "target_pose",
    "target_joints",
    "world_update",
    "control_frequency",
    "action_horizon",
    "fast_mode",
    "n_steps",
)

#: How each wire key should be rendered / parsed by the run form.
WIRE_KEY_TYPES: dict[str, str] = {
    "policy_host": "string",
    "policy_port": "int",
    "policy_type": "string",
    "server_address": "string",
    "model_path": "string",
    "pretrained_name_or_path": "string",
    "robot_name": "string",
    "target_pose": "json",
    "target_joints": "json",
    "world_update": "json",
    "control_frequency": "float",
    "action_horizon": "int",
    "fast_mode": "bool",
    "n_steps": "int",
}

#: Registry key -> the wire key that actually carries it. The registry names a
#: provider's constructor kwargs; the wire schema names its own fields, and the
#: two only partly overlap.
_WIRE_ALIASES = {
    "port": "policy_port",
    "host": "policy_host",
    "checkpoint": "model_path",
    "policy_path": "model_path",
    "repo_id": "pretrained_name_or_path",
    "server_address": "server_address",
}


def policy_catalog() -> list[dict[str, Any]]:
    """Full provider objects from ``registry/policies.json``, in registry order."""
    try:
        from strands_robots.registry.policies import get_policy_provider, list_policy_providers
    except Exception as exc:  # noqa: BLE001 - a broken registry is an empty catalog, reported by the route
        logger.warning("policy registry unavailable: %s", exc)
        return []

    try:
        from strands_robots.mesh.security import is_safe_policy_provider
    except Exception:  # noqa: BLE001 - without the mesh gate nothing is locked

        def is_safe_policy_provider(_name: str) -> bool:  # type: ignore[misc]
            return True

    out: list[dict[str, Any]] = []
    for name in list_policy_providers():
        spec = get_policy_provider(name) or {}
        requires = list(spec.get("requires") or [])
        config_keys = list(spec.get("config_keys") or [])
        # Split the provider's inputs by what the wire will actually carry, so
        # the form can render the deliverable fields and say that the rest
        # only work when the policy is built locally.
        wire_fields: list[dict[str, Any]] = []
        unsettable: list[str] = []
        for key in dict.fromkeys(requires + config_keys):
            wire_key = _WIRE_ALIASES.get(key, key)
            if wire_key in WIRE_CMD_KEYS:
                wire_fields.append(
                    {
                        "key": key,
                        "wire_key": wire_key,
                        "type": WIRE_KEY_TYPES.get(wire_key, "string"),
                        "required": key in requires,
                        "default": (spec.get("defaults") or {}).get(key),
                    }
                )
            else:
                unsettable.append(key)
        out.append(
            {
                "name": name,
                "description": spec.get("description", ""),
                "requires": requires,
                "config_keys": config_keys,
                "defaults": dict(spec.get("defaults") or {}),
                "shorthands": list(spec.get("shorthands") or []),
                "url_patterns": list(spec.get("url_patterns") or []),
                "extra": spec.get("extra"),
                "trainable": bool(spec.get("trainer")),
                "wire_fields": wire_fields,
                "unsettable_over_mesh": unsettable,
                # False -> the mesh security gate rejects it; the card shows a lock
                # and points at STRANDS_MESH_POLICY_TYPE_ALLOW rather than letting
                # the operator discover it as a wire rejection.
                "wire_safe": bool(is_safe_policy_provider(name)),
                # Hardware peers cannot build checkpoint policies over the wire
                # (they only accept {port, host, data_config}).
                "server_based": bool({"port", "policy_port", "server_address", "host"} & set(requires)),
            }
        )
    return out
