"""Tests for ``Simulation``'s tool_spec AgentTool interface.

Two concerns:

1. ``_dispatch_action`` forwards ``policy_config`` nested-dict correctly and
   drops unknown top-level keys (no ``**kwargs`` passthrough).
2. ``tool_spec.json`` every action resolves to a *public* method (the DX
   contract: no ``sim._private_thing`` behind an alias).
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest

# Skip the whole module if mujoco isn't available (dev env without [sim-mujoco]).
pytest.importorskip("mujoco")

import json
import re
from pathlib import Path

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim() -> Generator[Simulation, None, None]:
    s = Simulation(tool_name="dispatch_test", mesh=False)
    yield s
    s.cleanup()


def _capture_kwargs(captured: dict[str, Any], sim: Simulation, method_name: str):
    """Build a replacement that preserves the original signature so the
    schema-driven dispatcher binds the kwargs correctly."""
    import inspect
    from functools import wraps

    original = getattr(sim, method_name)

    @wraps(original)
    def fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
        # Bind positional args to parameter names for uniform capture
        sig = inspect.signature(original)
        bound = sig.bind_partial(*args, **kwargs)
        captured.clear()
        captured.update(bound.arguments)
        return {"status": "success", "content": [{"text": "ok"}]}

    return fake


class TestDispatcherForwardsPolicyConfig:
    """Nested ``policy_config`` routes verbatim to the method."""

    def test_run_policy_forwards_policy_config_as_single_dict(self, sim):
        captured: dict[str, Any] = {}
        cfg = {
            "observation_mapping": {
                "front": "video.front",
                "wrist": "video.wrist",
                "joint_position": "state.single_arm",
            },
            "action_mapping": {"action.single_arm": "joint_position"},
            "device": "mps",
        }
        with patch.object(sim, "run_policy", _capture_kwargs(captured, sim, "run_policy")):
            sim._dispatch_action(
                "run_policy",
                {
                    "robot_name": "so100",
                    "policy_provider": "mock",
                    "instruction": "pick up the red cube",
                    "duration": 3.0,
                    "policy_config": cfg,
                },
            )
        assert captured["robot_name"] == "so100"
        assert captured["policy_provider"] == "mock"
        assert captured["instruction"] == "pick up the red cube"
        assert captured["duration"] == 3.0
        # policy_config reaches the method as a single opaque dict
        assert captured["policy_config"] == cfg

    def test_eval_policy_forwards_policy_config(self, sim):
        captured: dict[str, Any] = {}
        cfg = {
            "pretrained_name_or_path": "lerobot/smolvla_base",
            "device": "mps",
            "trust_remote_code": True,
            "actions_per_step": 4,
        }
        with patch.object(sim, "eval_policy", _capture_kwargs(captured, sim, "eval_policy")):
            sim._dispatch_action(
                "eval_policy",
                {
                    "robot_name": "so100",
                    "policy_provider": "lerobot_local",
                    "n_episodes": 2,
                    "max_steps": 100,
                    "policy_config": cfg,
                },
            )
        assert captured["robot_name"] == "so100"
        assert captured["policy_provider"] == "lerobot_local"
        assert captured["n_episodes"] == 2
        assert captured["max_steps"] == 100
        assert captured["policy_config"] == cfg

    def test_start_policy_forwards_policy_config(self, sim):
        captured: dict[str, Any] = {}
        cfg = {
            "host": "localhost",
            "port": 5555,
            "api_token": "dummy-token",
            "observation_mapping": {"front": "video.front"},
            "action_mapping": {"action.single_arm": "joint_position"},
        }
        with patch.object(sim, "start_policy", _capture_kwargs(captured, sim, "start_policy")):
            sim._dispatch_action(
                "start_policy",
                {
                    "robot_name": "so100",
                    "policy_provider": "groot",
                    "instruction": "tidy the desk",
                    "policy_config": cfg,
                },
            )
        assert captured["policy_provider"] == "groot"
        assert captured["instruction"] == "tidy the desk"
        assert captured["policy_config"] == cfg


class TestDispatcherRejectsUnknownTopLevelKeys:
    """T1: Unknown top-level keys must be REJECTED with a friendly error."""

    def test_run_policy_rejects_legacy_top_level_policy_kwargs(self, sim):
        """Legacy policy kwargs at the top level must be rejected, not silently dropped."""
        result = sim._dispatch_action(
            "run_policy",
            {
                "robot_name": "so100",
                "policy_provider": "mock",
                "observation_mapping": {"x": "y"},  # not a top-level param anymore
            },
        )
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "Unknown parameter 'observation_mapping'" in text
        assert "run_policy" in text

    def test_non_policy_action_rejects_unknown_kwargs(self, sim):
        result = sim._dispatch_action(
            "set_gravity",
            {"gravity": [0, 0, -9.81], "device": "mps"},
        )
        assert result["status"] == "error"
        assert "Unknown parameter 'device'" in result["content"][0]["text"]


class TestToolSpecIsClean:
    """tool_spec.json must advertise ``policy_config`` and NOT the old leaked keys."""

    def test_tool_spec_declares_policy_config(self):
        import json
        from pathlib import Path

        spec_path = Path(__file__).resolve().parents[3] / "strands_robots" / "simulation" / "mujoco" / "tool_spec.json"
        spec = json.loads(spec_path.read_text())
        props = spec["properties"]

        # policy_config must be present as an object
        assert "policy_config" in props, "tool_spec.json missing 'policy_config'"
        assert props["policy_config"]["type"] == "object"

        # Legacy top-level policy fields must NOT be advertised
        for leaked in (
            "observation_mapping",
            "action_mapping",
            "host",
            "port",
            "api_token",
            "policy_host",
            "policy_port",
            "pretrained_name_or_path",
            "trust_remote_code",
            "actions_per_step",
            "use_processor",
            "processor_overrides",
            "device",
            "model_path",
        ):
            assert leaked not in props, (
                f"tool_spec.json must not advertise top-level '{leaked}' - it belongs under policy_config"
            )


# Public-method DX contract

# Extract live alias table


_src = (Path(__file__).resolve().parents[3] / "strands_robots/simulation/mujoco/simulation.py").read_text()
_m = re.search(r"_ALIASES\s*=\s*\{([^}]+)\}", _src)
_LIVE_ALIASES = {}
if _m:
    for _line in _m.group(1).splitlines():
        _mm = re.match(r'\s*"([^"]+)":\s*"([^"]+)"', _line.strip().rstrip(","))
        if _mm:
            _LIVE_ALIASES[_mm.group(1)] = _mm.group(2)


def test_every_tool_spec_action_has_a_public_method_or_documented_alias():
    """DevX contract: every action in tool_spec.json resolves to either
    a PUBLIC method ``sim.<action>()`` or to a PUBLIC method via the
    dispatcher's documented ``_ALIASES`` table. No private leading-underscore
    fallbacks are allowed.
    """
    spec_path = Path(__file__).resolve().parents[3] / "strands_robots/simulation/mujoco/tool_spec.json"
    spec = json.loads(spec_path.read_text())
    actions = spec["properties"]["action"]["enum"]

    offenders = []
    for action in actions:
        resolved = _LIVE_ALIASES.get(action, action)
        method = getattr(Simulation, resolved, None)
        if method is None:
            offenders.append(f"{action!r} → method {resolved!r} does not exist")
        elif resolved.startswith("_"):
            offenders.append(f"{action!r} → PRIVATE method {resolved!r} (leaky DX)")

    assert not offenders, "tool_spec actions must resolve to PUBLIC methods:\n  - " + "\n  - ".join(offenders)


def test_tool_spec_declares_create_world_curriculum_knobs() -> None:
    """The LLM-facing tool_spec must advertise create_world's world/terrain knobs.

    The router accepts any create_world signature param at runtime (it validates
    against the method signature), but an LLM only forms tool calls from the
    tool_spec schema it is handed. ``terrain`` + its curriculum companion
    ``difficulty`` (the terrain-elevation curriculum knob) must both be
    discoverable there, alongside ``ground_plane``, or an agent driving the sim
    tool cannot spawn a robot on non-flat / curriculum-scaled ground.
    """
    spec_path = Path(__file__).resolve().parents[3] / "strands_robots/simulation/mujoco/tool_spec.json"
    props = json.loads(spec_path.read_text())["properties"]
    for knob in ("ground_plane", "terrain", "difficulty"):
        assert knob in props, f"tool_spec.json must advertise create_world's {knob!r} knob"
    assert props["difficulty"]["type"] == "number"


# Schema-load performance contract


def test_tool_spec_schema_cached_at_module_load(sim: Simulation) -> None:
    """tool_spec property must not re-open/parse the 357-line JSON per access.

    The property is called on every strands agent LLM invocation (hot path).
    The cached ``_TOOL_SPEC_SCHEMA`` dict must be the exact object returned
    under ``inputSchema.json`` across repeated accesses, proving there's no
    reload in the property body.
    """
    from strands_robots.simulation.mujoco.simulation import _TOOL_SPEC_SCHEMA

    spec_a = sim.tool_spec
    spec_b = sim.tool_spec
    # Identity check - same dict object, not just equal content
    assert spec_a["inputSchema"]["json"] is _TOOL_SPEC_SCHEMA
    assert spec_b["inputSchema"]["json"] is _TOOL_SPEC_SCHEMA
    assert spec_a["inputSchema"]["json"] is spec_b["inputSchema"]["json"]


def test_tool_spec_schema_has_expected_shape() -> None:
    """Cached schema must still expose the canonical JSON-schema top keys."""
    from strands_robots.simulation.mujoco.simulation import _TOOL_SPEC_SCHEMA

    assert isinstance(_TOOL_SPEC_SCHEMA, dict)
    assert "type" in _TOOL_SPEC_SCHEMA
    assert "properties" in _TOOL_SPEC_SCHEMA
    assert "required" in _TOOL_SPEC_SCHEMA


# Description vs. enum drift contract
#
# The ``tool_spec`` description string is on the LLM hot path: an agent
# discovers the available actions from this text, so an action that is in the
# enum (and therefore dispatchable) but absent from the description is
# effectively invisible. This is exactly how the three [Benchmark] actions went
# undiscoverable while the "Actions (N total)" count drifted. These two checks
# pin the description to the enum so the next added action fails CI until it is
# documented.


def test_tool_spec_description_mentions_every_enum_action(sim: Simulation) -> None:
    """Every action in the enum must appear by name in the tool_spec description.

    Catches the drift where a dispatchable action (e.g. the [Benchmark] trio) is
    added to tool_spec.json + a handler but never surfaced in the human/LLM
    description, leaving it undiscoverable.
    """
    description = sim.tool_spec["description"]
    enum = sim.tool_spec["inputSchema"]["json"]["properties"]["action"]["enum"]

    # Longest-name-first so a substring action (e.g. "render") does not mask a
    # genuinely-missing longer action (e.g. "render_all") during the membership
    # scan. We assert exact whole-token presence via word boundaries.
    missing = [a for a in enum if not re.search(rf"\b{re.escape(a)}\b", description)]
    assert not missing, (
        f"tool_spec description must name every dispatchable enum action; undocumented: {sorted(missing)}"
    )


def test_tool_spec_description_action_count_matches_enum(sim: Simulation) -> None:
    """The "Actions (N total)" count in the description must equal len(enum)."""
    description = sim.tool_spec["description"]
    enum = sim.tool_spec["inputSchema"]["json"]["properties"]["action"]["enum"]

    m = re.search(r"Actions \((\d+) total\)", description)
    assert m is not None, "tool_spec description must state 'Actions (N total)'"
    stated = int(m.group(1))
    assert stated == len(enum), (
        f"tool_spec description says {stated} actions but the enum has {len(enum)}; "
        "update the count when adding/removing an action."
    )
