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

import inspect
import json
import re
from pathlib import Path

from strands.types.tools import AgentTool  # noqa: E402

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


# The converse direction: a described action the enum does not publish
#
# The two checks above pin ``enum -> description``: every enum entry is named in
# the text, and the stated count equals ``len(enum)``. Neither can observe the
# other direction, and the description carried a name the enum never offered -
# ``save_episode``, listed under ``[Recording]`` beside ``start_recording`` and
# ``stop_recording`` while absent from the enum and declared deliberately
# Python-only in ``_PYTHON_ONLY_ACTIONS`` below.
#
# Both existing checks passed throughout. The membership scan only ever asks
# whether each *enum* entry appears in the text, so a surplus name is not a
# value it iterates. The count check compared the stated ``77`` against
# ``len(enum)``, which agreed - the sentence was consistent with the enum while
# introducing a list of 78 names, so the one number that would have contradicted
# it was the one nobody counted.
#
# The direction matters because the two failures are not symmetric. An enum entry
# missing from the text is undiscoverable but selectable. A name in the text that
# the enum lacks is the reverse: the model is told to use an action the
# schema-constrained decoder cannot emit, so following the documentation produces
# a rejected call.
#
# ``describe()`` is deliberately wider than both and is not checked here - it
# lists 90 methods, 13 of them Python-only, because it is the developer-facing
# inventory. The description is not: it enumerates the *action* surface under a
# count, which is what makes a name in it that the enum lacks a defect rather
# than breadth.


def _described_actions(description: str) -> list[str]:
    """The capability names the description's ``Actions (...)`` sentence lists.

    Parsed structurally rather than by harvesting every identifier-shaped word.
    The categories carry parenthesised asides holding prose and ``;``
    (``move_to (Cartesian EE transport via IK; not collision-aware)``), and the
    final category is followed by a closing sentence that names ``destroy()``
    a second time, so a word-harvesting scan would read both as list items - and
    would fail this for a prose word that happened to match a Python-only method
    name, which is not a drift. Asides are stripped, the trailing sentence is
    cut, and only a bare identifier standing alone as a list item counts.
    """
    tail = description[description.index("Actions (") :]
    names: list[str] = []
    for segment in re.findall(r"\[[^\]]+\]([^\[]*)", tail):
        segment = re.sub(r"\([^()]*\)", "", segment)
        segment = segment.split(". ")[0]
        for item in re.split(r"[;,]", segment):
            item = item.strip().rstrip(".").strip()
            if re.fullmatch(r"[a-z_][a-z0-9_]*", item):
                names.append(item)
    return names


def test_tool_spec_description_names_no_action_the_enum_omits(sim: Simulation) -> None:
    """Prose may not promise an action the decoder cannot select.

    This is the direction that broke. ``save_episode`` stays reachable from
    Python and stays in ``_PYTHON_ONLY_ACTIONS``; what it may not do is appear in
    the model-facing action list. ``run_policy(n_episodes=N)`` is the published
    multi-episode path, so nothing an agent could reach was lost by dropping it.
    """
    described = set(_described_actions(sim.tool_spec["description"]))
    enum = set(sim.tool_spec["inputSchema"]["json"]["properties"]["action"]["enum"])

    unpublished = sorted(described - enum)
    assert not unpublished, (
        "the tool_spec description lists these actions but the enum does not offer them, so a "
        "model following the description emits a value the schema rejects; publish them in the "
        f"enum or drop them from the description: {unpublished}"
    )


def test_tool_spec_description_action_count_matches_the_names_it_lists(sim: Simulation) -> None:
    """``Actions (N total)`` is a claim about the list, not only about the enum.

    The companion check above compares the same literal against ``len(enum)``.
    Both are needed and neither implies the other: the stated count was ``77``
    and the enum held 77, so that check passed while the sentence listed 78
    names. Counting the list is what turns the literal into a claim that can
    contradict the text carrying it.
    """
    description = sim.tool_spec["description"]
    described = _described_actions(description)

    m = re.search(r"Actions \((\d+) total\)", description)
    assert m is not None, "tool_spec description must state 'Actions (N total)'"
    stated = int(m.group(1))
    assert stated == len(described), (
        f"tool_spec description says {stated} actions but its category lists name {len(described)}: {sorted(described)}"
    )


def test_tool_spec_description_lists_no_action_twice(sim: Simulation) -> None:
    """The count check compares lengths, so a repeat would mask an omission.

    ``destroy`` appears twice in the description - once as a ``[World]`` action
    and once in the closing ``Call destroy() at session end`` sentence - and if
    the parser read the second occurrence the totals would still agree while one
    action went unlisted. Pinned on the parser rather than on the text so the
    sentence stays free to mention an action again in prose.
    """
    described = _described_actions(sim.tool_spec["description"])
    repeated = sorted({name for name in described if described.count(name) > 1})
    assert not repeated, f"the description's action lists name these more than once: {repeated}"


def test_a_described_action_absent_from_the_enum_is_reported(sim: Simulation) -> None:
    """Planted-defect meta-test: the parity check above is not vacuous.

    It asserts a set is empty, which is the shape that also passes when the
    parser finds nothing at all - and the parser is the part most likely to stop
    matching if the description is reworded. A description carrying one extra
    name must report exactly that name, and the real one must stay clean.
    """
    description = sim.tool_spec["description"]
    enum = set(sim.tool_spec["inputSchema"]["json"]["properties"]["action"]["enum"])

    assert len(_described_actions(description)) > 1, (
        "the parser found no action list to check; the description's "
        "'Actions (N total): [Category] a, b, c' shape has changed"
    )

    planted = description.replace("[Recording] start_recording,", "[Recording] teleport_everything, start_recording,")
    assert set(_described_actions(planted)) - enum == {"teleport_everything"}
    assert not set(_described_actions(description)) - enum, "the real description must stay published"


# Vector-arity contract
#
# ``_validate_and_build_kwargs`` step 2 refuses any vector param whose component
# count is not in ``_VECTOR_PARAM_LENGTHS`` - "Parameter 'orientation' must be a
# list of 4 numbers, got 3." That refusal is the router's, and until the bounds
# below were declared the schema said only ``{"type": "array", "items": {"type":
# "number"}}``: a model forms its call from the schema and nothing else, so the
# one number it needed was the one number not published, and the arity was
# discoverable only by being rejected.
#
# ``minItems`` / ``maxItems`` are the machine-readable form of that count. Prose
# is not a substitute: seven of the ten carried the shape in a description
# (``[x, y, z]``) while three - ``position``, ``gravity``, ``orientation`` - had
# no description at all, and prose is not read by a schema-constrained decoder.
#
# Keyed on the live table rather than a literal copy of it, so a param added
# there fails here until it is published.


def _tool_spec_properties() -> dict[str, Any]:
    spec_path = Path(__file__).resolve().parents[3] / "strands_robots/simulation/mujoco/tool_spec.json"
    return json.loads(spec_path.read_text())["properties"]


class TestTheSchemaPublishesTheArityTheRouterEnforces:
    """Every router-validated vector param declares its component count."""

    @staticmethod
    def _schema_name(param: str) -> str:
        """The name a caller spells ``param`` as in the schema.

        The router rewrites ``_FIELD_ALIASES`` before validating, so a table
        entry can be validated under a name no caller ever writes.
        """
        props = _tool_spec_properties()
        if param in props:
            return param
        aliases_by_param = {target: field for field, target in Simulation._FIELD_ALIASES.items()}
        return aliases_by_param.get(param, param)

    def test_every_router_validated_vector_param_is_reachable_in_the_schema(self) -> None:
        """Each table entry is a schema property, directly or under a field alias.

        Skipping an entry that is absent from the schema would make this class
        vacuous exactly when it matters: a property renamed out of the schema
        would stop being checked instead of failing. ``torque`` is the one entry
        with no property of its own - the router rewrites the schema's
        ``torque_vec`` to it via ``_FIELD_ALIASES`` before validating - so the
        alias table is consulted rather than the entry being passed over.
        """
        props = _tool_spec_properties()

        unreachable = [param for param in Simulation._VECTOR_PARAM_LENGTHS if self._schema_name(param) not in props]
        assert not unreachable, (
            "every _VECTOR_PARAM_LENGTHS entry must be advertised as a tool_spec property, "
            f"directly or via _FIELD_ALIASES; unreachable: {sorted(unreachable)}"
        )

    def test_two_spellings_of_one_vector_param_accept_the_same_count(self) -> None:
        """``torque`` and ``torque_vec`` are one wire field, so one count.

        A field alias means two table entries can resolve to a single schema
        property, and a property has one pair of bounds. If the entries ever
        disagreed no bounds could be correct for both, and the next test would
        report the property twice with contradictory expectations rather than
        naming the contradiction.
        """
        by_property: dict[str, set[tuple[int, ...]]] = {}
        for param, accepted_lens in Simulation._VECTOR_PARAM_LENGTHS.items():
            by_property.setdefault(self._schema_name(param), set()).add(tuple(accepted_lens))

        disagreeing = {name: sorted(lens) for name, lens in by_property.items() if len(lens) > 1}
        assert not disagreeing, f"params sharing one tool_spec property must accept the same counts: {disagreeing}"

    def test_every_router_validated_vector_param_declares_min_and_max_items(self) -> None:
        """The published bounds equal the counts the router will accept."""
        props = _tool_spec_properties()

        accepted_by_property: dict[str, set[int]] = {}
        for param, accepted_lens in Simulation._VECTOR_PARAM_LENGTHS.items():
            accepted_by_property.setdefault(self._schema_name(param), set()).update(accepted_lens)

        offenders: list[str] = []
        for name, accepted in sorted(accepted_by_property.items()):
            schema = props.get(name)
            if schema is None:
                continue  # reported by the reachability test above
            if schema.get("type") != "array":
                offenders.append(f"{name!r} is validated as a vector but advertised as {schema.get('type')!r}")
                continue
            if schema.get("minItems") != min(accepted) or schema.get("maxItems") != max(accepted):
                offenders.append(
                    f"{name!r} accepts {sorted(accepted)} components but advertises "
                    f"minItems={schema.get('minItems')} maxItems={schema.get('maxItems')}"
                )

        assert not offenders, "tool_spec must publish the component counts the router enforces:\n  - " + "\n  - ".join(
            offenders
        )

    def test_orientation_publishes_the_quaternion_component_order(self) -> None:
        """Arity alone cannot pin ``orientation``, so the order is stated.

        The other nine params are fully described by a count: a rejected length
        is the only way to get them wrong. ``orientation`` is not. Both
        ``[w, x, y, z]`` and ``[x, y, z, w]`` are four components, ``add_object``
        assigns the value straight to ``body.quat``, and MuJoCo reads that
        scalar-first - so the wrong order passes every check the router has and
        applies a different rotation under ``status="success"``. A silent wrong
        answer is the one failure mode the bounds above do not cover, which is
        why the convention is published rather than left to the count.
        """
        orientation = _tool_spec_properties()["orientation"]
        description = orientation.get("description", "")
        assert "quaternion" in description.lower(), "orientation must say it is a quaternion, not a 4-vector"
        assert "[w, x, y, z]" in description, (
            "orientation must publish the scalar-first component order; an [x, y, z, w] "
            f"vector is otherwise indistinguishable to a caller. Got: {description!r}"
        )


# Router-vs-enum breadth contract
#
# ``_dispatch_action`` resolves an action with ``getattr(self, name)`` and no
# allowlist, so every public method is dispatchable while the ``action`` enum
# advertises a curated subset. The enum -> method direction is already pinned
# above (``test_every_tool_spec_action_has_a_public_method_or_documented_alias``);
# this section pins the direction that was never measured, which is the one that
# drifts silently. Adding a public method to the engine or any mixin made it
# agent-dispatchable-but-unadvertised, and nothing could tell that apart from a
# deliberate omission - so the omissions and the accidents looked identical.
#
# Membership in ``_PYTHON_ONLY_ACTIONS`` means "dispatchable from Python,
# deliberately not advertised to a model". It does NOT mean "must stay
# unadvertised": publishing any of these is a curation judgement, and whether the
# router should instead refuse a non-enum action is the open question in #2093.
# This guard settles neither. It only makes the set unable to change in silence -
# a new public method fails here until someone writes down which side it is on.
#
# Nothing is asserted about the count, deliberately: a number in an assertion
# message is stale the first time the set legitimately changes, and the failure
# names the methods either way.

# Grouped by declaring class, which is the unit a reviewer checks against.
_PYTHON_ONLY_ACTIONS = frozenset(
    {
        # MuJoCoSimEngine - lifecycle and introspection, plus the two
        # move-the-robot paths the motion primitives and run_policy front.
        "bind_policy_sim_context",
        "cleanup",
        "describe",
        "get_observation",
        "physics_timestep",
        "robot_action_keys",
        "robot_joint_names",
        "run_multi_policy",
        "send_action",
        # RenderingMixin - return arrays / camera intrinsics rather than text.
        "get_camera_params",
        "get_frame",
        "start_cameras_recording_synchronous",
        # DatasetRecordingMixin
        "save_episode",
        "stream_dataset",
        # SimEngine
        "verify_dataset_episodes",
        # ManipulationMixin
        "attachment_involving",
        # TeleopMixin - drives real hardware from a host input device.
        "attach_teleop",
        "detach_teleop",
        "get_teleoperate_status",
        "list_teleops",
        "stop_teleoperate",
        "teleoperate",
        "tool_name_label",
    }
)


def _dispatchable_public_methods(cls: type = Simulation) -> set[str]:
    """Names ``_dispatch_action`` will resolve on ``cls``.

    Mirrors the router's own reachability rule rather than restating it: it takes
    ``getattr`` then ``inspect.signature``, so a public *callable* is the unit,
    and the leading-underscore names it refuses are excluded here too.
    """
    return {name for name, _value in inspect.getmembers(cls, callable) if not name.startswith("_")}


def _framework_surface() -> set[str]:
    """The tool framework's own public surface, which is not a sim capability.

    ``get_display_properties`` / ``mark_dynamic`` / ``stream`` reach the engine by
    inheriting from ``AgentTool``, so listing them as deliberate sim omissions
    would be wrong twice: they are not capabilities anyone curated, and a
    strands-agents upgrade that adds a protocol method would fail the inventory
    below for a reason that has nothing to do with this schema. Derived from the
    live base class so that upgrade is absorbed instead of reported.
    """
    return {name for name in dir(AgentTool) if not name.startswith("_")}


def _enum_targets() -> set[str]:
    """Every method name the enum reaches, directly or through an action alias."""
    enum = _tool_spec_properties()["action"]["enum"]
    return set(enum) | {_LIVE_ALIASES.get(action, action) for action in enum}


class TestTheRouterDispatchesOnlyPublishedOrDeclaredActions:
    """The gap between what dispatches and what is advertised is declared."""

    @staticmethod
    def _unaccounted(cls: type = Simulation) -> set[str]:
        return _dispatchable_public_methods(cls) - _enum_targets() - _framework_surface() - _PYTHON_ONLY_ACTIONS

    def test_every_dispatchable_method_is_published_or_declared_python_only(self) -> None:
        """A public method is either in the enum or named as Python-only.

        This is the forcing function, and the reason it is an equality rather
        than a subset check is in the companion test below: a method that is
        quietly deleted should also fail, or the inventory decays into a list of
        names that no longer mean anything.
        """
        unaccounted = self._unaccounted()
        assert not unaccounted, (
            "these methods are dispatchable via sim(action=...) but neither advertised in the "
            "tool_spec enum nor declared in _PYTHON_ONLY_ACTIONS - publish the ones an agent "
            f"should reach and add the rest to the inventory: {sorted(unaccounted)}"
        )

    def test_the_inventory_names_only_methods_that_still_exist(self) -> None:
        """A renamed or removed method leaves the inventory, rather than rotting.

        Without this, the set accumulates names that pin nothing: the test above
        passes on a subset, so a stale entry is invisible there, and the next
        reader cannot tell a live deliberate omission from a dead one.
        """
        stale = _PYTHON_ONLY_ACTIONS - _dispatchable_public_methods()
        assert not stale, f"_PYTHON_ONLY_ACTIONS names methods that no longer exist: {sorted(stale)}"

    def test_the_inventory_never_names_an_action_the_enum_publishes(self) -> None:
        """The two sets are disjoint, so neither can mask the other.

        An entry that is also in the enum would be a contradiction the first test
        cannot report - subtracting the inventory would hide a published action
        from the accounting, and the schema would be describing something the
        inventory calls Python-only.
        """
        both = _PYTHON_ONLY_ACTIONS & _enum_targets()
        assert not both, f"an action cannot be both published and declared Python-only: {sorted(both)}"

    def test_the_framework_surface_hides_no_published_action(self) -> None:
        """Excluding ``AgentTool``'s surface must not exempt a real capability.

        ``_framework_surface`` is subtracted before the accounting, so a
        framework method that ever shares a name with a published action would
        stop being checked here without any test failing. Pinned as an emptiness
        claim about the overlap so that collision is a failure, not a silence.
        """
        collision = _framework_surface() & _enum_targets()
        assert not collision, (
            "a tool_spec action shares a name with the AgentTool protocol surface, so the "
            f"breadth accounting would skip it: {sorted(collision)}"
        )

    def test_a_newly_added_public_method_is_reported(self) -> None:
        """Planted-defect meta-test: the accounting is not vacuous.

        Every assertion above is that a set is empty, which is exactly the shape
        that passes when the measurement silently finds nothing. A subclass
        carrying one new public method is the condition the first test exists to
        catch, so it must be reported for that subclass and absent for the real
        one.
        """

        class _EngineWithANewCapability(Simulation):
            def teleport_everything(self) -> dict[str, Any]:
                raise NotImplementedError

        assert self._unaccounted(_EngineWithANewCapability) == {"teleport_everything"}
        assert not self._unaccounted(), "the real engine must stay accounted for"


# The geom-property payload contract
#
# ``set_geom_properties`` is the one published action whose entire purpose is
# writing geom payload vectors, and the flat property dict described it as if
# ``add_object`` were the only consumer. Two consequences, both invisible to the
# router: it validates a call against the *method signature*, so a payload the
# schema never published still dispatches when a Python caller passes it, and a
# schema-constrained decoder cannot emit it at all.
#
# ``friction`` was published nowhere, so the coefficients the method validates,
# documents and applies were unreachable from the model-facing surface.
#
# ``size`` is worse than unreachable, because two actions consume that one wire
# field under different conventions: ``add_object`` takes full extents, and
# ``set_geom_properties`` takes the compiled geom's own ``geom_size``
# components, which are half-extents for a box. The published description named
# only ``add_object`` and stated the other convention was explicitly not what
# the field meant, so a caller who follows it either doubles a box under
# ``status="success"`` or is refused for a component the same text calls unused.


def _geom_property_params() -> set[str]:
    """The payload parameters ``set_geom_properties`` accepts."""
    signature = inspect.signature(Simulation.set_geom_properties)
    return {name for name in signature.parameters if name != "self"}


def _compiled_geom_id(sim: Simulation, geom_name: str) -> int:
    """The live model's id for ``geom_name``, so a write can be read back."""
    import mujoco

    geom_id = int(mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name))
    assert geom_id >= 0, f"no geom named {geom_name!r}"
    return geom_id


@pytest.fixture
def world_sim(sim: Simulation) -> Simulation:
    """A session with a world, ready for scene mutation."""
    assert sim.create_world()["status"] == "success"
    return sim


class TestTheGeomPropertyPayloadIsPublished:
    """Every payload ``set_geom_properties`` accepts is reachable from the schema."""

    def test_every_geom_property_parameter_is_reachable_in_the_schema(self) -> None:
        """A payload absent from the schema is unreachable for a model.

        The router binds against the method signature rather than the schema, so
        an unpublished payload is not refused - it is simply never emitted, and
        the capability reads as missing rather than as undiscoverable. This is
        the parameter-level form of the action-level accounting above, scoped to
        the one action whose whole purpose is writing these vectors.
        """
        published = set(_tool_spec_properties())
        aliases_by_param = {target: field for field, target in Simulation._FIELD_ALIASES.items()}

        unreachable = sorted(
            param for param in _geom_property_params() if not ({param, aliases_by_param.get(param, param)} & published)
        )
        assert not unreachable, (
            "set_geom_properties accepts these payloads but tool_spec publishes no property for "
            f"them, so a schema-constrained decoder cannot pass them: {unreachable}"
        )

    def test_the_published_friction_carries_the_arity_and_order_the_geom_defines(self) -> None:
        """Three coefficients in a fixed order, so a count alone cannot pin it.

        MuJoCo's friction triple is ordered (sliding, torsional, rolling) and the
        three are not interchangeable: swapping them passes every arity check and
        applies a different contact model under ``status="success"``. Published in
        the same shape ``orientation`` uses for its component order, and for the
        same reason.
        """
        friction = _tool_spec_properties()["friction"]
        assert friction["type"] == "array"
        assert friction.get("minItems") == 3 and friction.get("maxItems") == 3, (
            "friction takes exactly three coefficients; publish the count rather than leaving a "
            f"decoder to guess it. Got: {friction!r}"
        )
        description = friction.get("description", "")
        for coefficient in ("sliding", "torsional", "rolling"):
            assert coefficient in description, (
                f"friction must publish its component order; {coefficient!r} is missing from {description!r}"
            )

    def test_a_friction_the_schema_publishes_is_applied(self, world_sim: Simulation) -> None:
        """The published payload reaches the model, end to end through the router."""
        assert (
            world_sim(action="add_object", name="crate", shape="box", position=[0, 0, 0.1], size=[0.1, 0.1, 0.1])[
                "status"
            ]
            == "success"
        )

        result = world_sim(action="set_geom_properties", geom_name="crate", friction=[0.6, 0.01, 0.001])

        assert result["status"] == "success", result
        geom_id = _compiled_geom_id(world_sim, "crate_geom")
        applied = [float(component) for component in world_sim.mj_model.geom_friction[geom_id][:3]]
        assert applied == [0.6, 0.01, 0.001], f"the published payload must reach the model, got {applied}"

    def test_a_partial_friction_is_refused_with_the_count_it_wanted(self, world_sim: Simulation) -> None:
        """The published bounds match the refusal, so the schema is honest.

        Bounds a caller can satisfy and still be refused would be worse than no
        bounds, so the arity the property advertises is the arity the action
        enforces.
        """
        world_sim(action="add_object", name="crate", shape="box", position=[0, 0, 0.1], size=[0.1, 0.1, 0.1])

        result = world_sim(action="set_geom_properties", geom_name="crate", friction=[0.6, 0.01])

        assert result["status"] == "error"
        assert "3 component" in result["content"][0]["text"]


class TestTheSharedSizeFieldPublishesBothConventions:
    """One wire field, two scales - so the schema names both."""

    @staticmethod
    def _geom_size(sim: Simulation, geom_name: str) -> list[float]:
        """The compiled ``geom_size`` of ``geom_name``, read from the live model."""
        geom_id = _compiled_geom_id(sim, geom_name)
        return [float(component) for component in sim.mj_model.geom_size[geom_id][:3]]

    def test_one_size_vector_scales_a_box_differently_through_the_two_actions(self, world_sim: Simulation) -> None:
        """The divergence, measured, and the disclosure it requires.

        ``size=[0.2, 0.2, 0.2]`` is a 20 cm box through ``add_object`` and a 40 cm
        box through ``set_geom_properties``, because one takes full extents and the
        other takes the geom's own half-extents. Nothing refuses the mismatch -
        both calls report ``status="success"`` - so the only place a caller can
        learn it is the property that carries the field.
        """
        world_sim(action="add_object", name="crate", shape="box", position=[0, 0, 0.5], size=[0.2, 0.2, 0.2])
        built = self._geom_size(world_sim, "crate_geom")

        assert world_sim(action="set_geom_properties", geom_name="crate", size=[0.2, 0.2, 0.2])["status"] == "success"
        resized = self._geom_size(world_sim, "crate_geom")

        assert built == [0.1, 0.1, 0.1], f"premise: add_object takes full extents, got {built}"
        assert resized == [0.2, 0.2, 0.2], f"premise: set_geom_properties takes half-extents, got {resized}"

        description = _tool_spec_properties()["size"]["description"]
        assert "add_object" in description and "set_geom_properties" in description, (
            "size means different things to two actions, so the property must name both rather "
            f"than describing one of them. Got: {description!r}"
        )
        assert "half-extent" in description.lower(), (
            "the resize convention is half-extents; a description that only states the full-extent "
            f"convention doubles every box it is followed for. Got: {description!r}"
        )

    def test_the_capsule_layout_the_field_publishes_for_add_object_is_refused_by_the_resize(
        self, world_sim: Simulation
    ) -> None:
        """A refusal that blames a value the other convention calls unused.

        ``add_object`` spells a capsule ``[diameter, unused, height]``, and the
        middle component is genuinely ignored there. The resize wants
        ``[radius, half-length]``, so following the published layout is refused
        for the zero in the slot the layout itself calls unused - a caller cannot
        get from that message to the two-component form.
        """
        world_sim(action="add_object", name="rod", shape="capsule", position=[0, 0, 0.5], size=[0.1, 0.0, 0.4])

        refused = world_sim(action="set_geom_properties", geom_name="rod", size=[0.1, 0.0, 0.4])
        accepted = world_sim(action="set_geom_properties", geom_name="rod", size=[0.1, 0.4])

        assert refused["status"] == "error", "premise: the add_object capsule layout is not accepted here"
        assert accepted["status"] == "success", accepted

        description = _tool_spec_properties()["size"]["description"]
        assert "radius" in description, (
            "the resize takes a capsule as [radius, half-length]; publish it so the add_object "
            f"triple is not the only layout a caller can read. Got: {description!r}"
        )

    def test_a_shared_field_both_actions_scale_alike_needs_no_disclosure(self, world_sim: Simulation) -> None:
        """``color`` is the control: one convention, so one description.

        Shared by the same two actions and meaning the same thing in both, which
        is why its description names neither. Without this the class above would
        read as a rule about every shared field rather than about the one whose
        meaning changes with the action.
        """
        world_sim(action="add_object", name="crate", shape="box", position=[0, 0, 0.1], size=[0.1, 0.1, 0.1])

        assert world_sim(action="set_geom_properties", geom_name="crate", color=[0.8, 0.2, 0.2])["status"] == "success"

        description = _tool_spec_properties()["color"]["description"]
        assert "add_object" not in description and "set_geom_properties" not in description, (
            "color means one thing to both actions, so it needs no per-action split; a mention "
            f"here would say the conventions diverge when they do not. Got: {description!r}"
        )
