"""Pin: shipped ACL example templates scope robot subscribe (no blanket ``**``).

Both ``examples/mesh_acl_example.json5`` and
``examples/mesh_acl_strict_per_peer.json5`` must:

1. Load + pass the loader's shape validation (so an operator who copies
   them verbatim gets a working, fail-closed ACL).
2. NOT grant the robot role a ``declare_subscriber`` rule whose
   ``key_exprs`` contains the bare ``**`` wildcard. A robot subscribing
   to ``**`` can observe every peer's cmd / state / camera streams --
   the cross-peer telemetry exposure these templates exist to prevent.

The operator role MAY keep a broad ``**`` subscribe (operators monitor
the whole fleet); these tests assert the distinction holds.
"""

from __future__ import annotations

from pathlib import Path

from strands_robots.mesh import _acl_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE = _REPO_ROOT / "examples" / "mesh_acl_example.json5"
_STRICT = _REPO_ROOT / "examples" / "mesh_acl_strict_per_peer.json5"


def _load(path: Path) -> dict:
    # _load_acl_file runs the full shape validation + fail-closed checks.
    return _acl_config._load_acl_file(path)


def _rule_ids_for_subject(acl: dict, subject_id: str) -> set[str]:
    """Return the set of rule ids bound to *subject_id* via policies."""
    out: set[str] = set()
    for pol in acl.get("policies", []):
        if subject_id in (pol.get("subjects") or []):
            out.update(pol.get("rules") or [])
    return out


def _subscribe_key_exprs(acl: dict, rule_ids: set[str]) -> list[str]:
    """Flatten key_exprs of every declare_subscriber rule in *rule_ids*."""
    kes: list[str] = []
    for rule in acl.get("rules", []):
        if rule.get("id") not in rule_ids:
            continue
        if "declare_subscriber" not in (rule.get("messages") or []):
            continue
        kes.extend(rule.get("key_exprs") or [])
    return kes


def test_example_loads_and_validates() -> None:
    acl = _load(_EXAMPLE)
    assert acl["enabled"] is True
    assert acl["default_permission"] == "deny"


def test_strict_loads_and_validates() -> None:
    acl = _load(_STRICT)
    assert acl["enabled"] is True
    assert acl["default_permission"] == "deny"


def test_example_robot_subscribe_is_not_wildcard() -> None:
    acl = _load(_EXAMPLE)
    robot_rules = _rule_ids_for_subject(acl, "robot_peer")
    robot_subs = _subscribe_key_exprs(acl, robot_rules)
    assert robot_subs, "robot_peer must have at least one scoped subscribe rule"
    assert "**" not in robot_subs, (
        "robot_peer subscribe grants bare '**' -- a robot must not be able to "
        "subscribe to every peer's topics. Scope by topic class instead."
    )


def test_example_operator_subscribe_may_be_broad() -> None:
    acl = _load(_EXAMPLE)
    op_rules = _rule_ids_for_subject(acl, "operator_peer")
    op_subs = _subscribe_key_exprs(acl, op_rules)
    # Operators legitimately observe the fleet; the template grants '**'.
    assert "**" in op_subs, "operator_peer is expected to retain broad observe"


def test_strict_robot_subscribe_is_per_peer_scoped() -> None:
    acl = _load(_STRICT)
    for robot_subject in ("robot_a", "robot_b"):
        rules = _rule_ids_for_subject(acl, robot_subject)
        subs = _subscribe_key_exprs(acl, rules)
        assert subs, f"{robot_subject} must have scoped subscribe rules"
        assert "**" not in subs, (
            f"{robot_subject} subscribe grants bare '**' -- strict template must "
            "enumerate per-peer own-topic prefixes only."
        )
