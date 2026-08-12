"""Shared capability-manifest schema for the fleet examples.

A capability manifest is a per-robot, per-site declaration of what a robot can
physically do. It is the boundary between business vocabulary (work orders:
material, operation, qty) and robot vocabulary (skills, payload, fixtures,
zones). One manifest per robot per site:

    {"robot": "lekiwi-a1", "site": "site-a",
     "skills": [{"name": "transport", "payload_kg": 8.0,
                 "fixture": "tote_clamp", "zones": ["litho", "etch"]}]}

Matching is deterministic hard-constraint filtering only: site equality, skill
name equality, payload threshold, fixture equality, zone coverage. There is no
model in the loop. An LLM agent may choose AMONG the robots this filter admits
(see ``05_work_order_dispatch.py``); it can never add one.

Consumed by example 01 (skill dispatch, #2180) and example 05 (work-order
dispatch, #2185). Part of the fleet suite (epic #2179).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Robot / site / skill / fixture / zone identifiers share one conservative
# charset: they cross the mesh wire (``robot_name`` allows ``[A-Za-z0-9_.-]+``)
# and appear in audit payloads, so nothing fancier is accepted here.
_IDENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class Skill:
    """One capability a robot offers at its site.

    Attributes:
        name: Skill identifier (e.g. ``"transport"``, ``"handle"``).
        payload_kg: Maximum payload this skill can carry, in kilograms.
        fixture: The single fixture type this skill can hold (e.g.
            ``"smif_pod"``). A robot with several fixtures declares one
            skill entry per fixture.
        zones: Zones (within the robot's site) this skill can serve.
    """

    name: str
    payload_kg: float
    fixture: str
    zones: tuple[str, ...]


@dataclass(frozen=True)
class CapabilityManifest:
    """Everything one robot can do at one site."""

    robot: str
    site: str
    skills: tuple[Skill, ...]


@dataclass(frozen=True)
class StepRequirement:
    """The hard constraints one work-order step places on a robot.

    Produced by the deterministic order-to-steps translation in example 05;
    consumed by :func:`feasible_robots`.
    """

    skill: str
    site: str
    payload_kg: float
    fixture: str
    zones: tuple[str, ...]


def _require_ident(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _IDENT_RE.match(value):
        raise ValueError(f"{field} must be a non-empty string matching [A-Za-z0-9][A-Za-z0-9_.-]* (got {value!r})")
    return value


def skill_from_dict(raw: Any) -> Skill:
    """Validate one skill dict and return a :class:`Skill`.

    Raises ValueError naming the offending field; a manifest that cannot be
    validated must never silently shrink to the fields that happened to parse.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"skill must be an object (got {type(raw).__name__})")
    unknown = set(raw) - {"name", "payload_kg", "fixture", "zones"}
    if unknown:
        raise ValueError(f"skill has unknown fields {sorted(unknown)}")
    name = _require_ident(raw.get("name"), "skill.name")
    payload = raw.get("payload_kg")
    if isinstance(payload, bool) or not isinstance(payload, int | float) or not payload > 0:
        raise ValueError(f"skill.payload_kg must be a positive number (got {payload!r})")
    fixture = _require_ident(raw.get("fixture"), "skill.fixture")
    zones = raw.get("zones")
    if not isinstance(zones, list | tuple) or not zones:
        raise ValueError(f"skill.zones must be a non-empty list (got {zones!r})")
    return Skill(
        name=name,
        payload_kg=float(payload),
        fixture=fixture,
        zones=tuple(_require_ident(z, "skill.zones[]") for z in zones),
    )


def manifest_from_dict(raw: Any) -> CapabilityManifest:
    """Validate one manifest dict and return a :class:`CapabilityManifest`."""
    if not isinstance(raw, dict):
        raise ValueError(f"manifest must be an object (got {type(raw).__name__})")
    unknown = set(raw) - {"robot", "site", "skills"}
    if unknown:
        raise ValueError(f"manifest has unknown fields {sorted(unknown)}")
    robot = _require_ident(raw.get("robot"), "manifest.robot")
    site = _require_ident(raw.get("site"), "manifest.site")
    skills = raw.get("skills")
    if not isinstance(skills, list | tuple) or not skills:
        raise ValueError(f"manifest.skills must be a non-empty list (got {skills!r})")
    return CapabilityManifest(robot=robot, site=site, skills=tuple(skill_from_dict(s) for s in skills))


def _reject_reason(manifest: CapabilityManifest, req: StepRequirement) -> dict[str, Any] | None:
    """Why this manifest cannot serve this step, or None if it can.

    The reason is machine-readable: it names the first failing constraint in a
    fixed check order (site, skill, fixture, payload_kg, zones) together with
    the required and actual values, so a NACK consumer can act on it without
    parsing prose.
    """
    if manifest.site != req.site:
        return {"robot": manifest.robot, "constraint": "site", "required": req.site, "actual": manifest.site}
    named = [s for s in manifest.skills if s.name == req.skill]
    if not named:
        return {
            "robot": manifest.robot,
            "constraint": "skill",
            "required": req.skill,
            "actual": sorted({s.name for s in manifest.skills}),
        }
    # A skill entry serves the step only if every remaining constraint holds.
    # When several entries share the name, report the failure of the entry
    # that got FURTHEST through the checks (fixture, then payload, then
    # zones): "your tote_clamp skill is 4 kg short" is actionable where
    # "your smif_pod skill is the wrong fixture" is noise.
    _DEPTH = {"fixture": 0, "payload_kg": 1, "zones": 2}
    best_failure: dict[str, Any] | None = None
    for skill in named:
        if skill.fixture != req.fixture:
            failure = {
                "robot": manifest.robot,
                "constraint": "fixture",
                "required": req.fixture,
                "actual": skill.fixture,
            }
        elif skill.payload_kg < req.payload_kg:
            failure = {
                "robot": manifest.robot,
                "constraint": "payload_kg",
                "required": req.payload_kg,
                "actual": skill.payload_kg,
            }
        elif not set(req.zones) <= set(skill.zones):
            failure = {
                "robot": manifest.robot,
                "constraint": "zones",
                "required": sorted(req.zones),
                "actual": sorted(skill.zones),
            }
        else:
            return None  # this skill entry satisfies every constraint
        if best_failure is None or _DEPTH[failure["constraint"]] > _DEPTH[best_failure["constraint"]]:
            best_failure = failure
    return best_failure


def feasible_robots(
    manifests: list[CapabilityManifest] | tuple[CapabilityManifest, ...],
    req: StepRequirement,
) -> tuple[list[CapabilityManifest], list[dict[str, Any]]]:
    """Deterministic hard-constraint filter: stage one of the translation.

    Returns ``(feasible, rejections)`` where ``feasible`` is sorted by robot
    name (so the result is stable across runs) and ``rejections`` carries one
    machine-readable reason per excluded robot. An empty ``feasible`` list is
    the caller's cue to NACK the order - never to guess.
    """
    feasible: list[CapabilityManifest] = []
    rejections: list[dict[str, Any]] = []
    for manifest in sorted(manifests, key=lambda m: m.robot):
        reason = _reject_reason(manifest, req)
        if reason is None:
            feasible.append(manifest)
        else:
            rejections.append(reason)
    return feasible, rejections
