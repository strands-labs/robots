"""A router refusal names the field by a spelling the schema publishes.

``_dispatch_action`` rewrites ``_FIELD_ALIASES`` to method parameter names
*before* validating, so every refusal downstream of that rewrite reports the
parameter it validated rather than the field the caller wrote. One of those
canonical names has no property of its own: ``apply_force``'s torque is
published as ``torque_vec`` and validated as ``torque``, and the schema has no
``torque`` property at all.

The consequence is a refusal a schema-constrained model cannot act on. Sending
``torque_vec="up"`` was answered ``Parameter 'torque' must be a list of 3
numbers``, and an unknown field on the same action was answered ``Valid:
['body_name', 'force', 'point', 'torque']`` -- both naming a property the model
is not permitted to emit, so its only route back was to guess that ``torque``
and ``torque_vec`` are one field.

The string-type check, thirty lines above the vector one in the same function,
already resolved the spelling back to what the caller sent; its comment records
why ("reporting the canonical parameter would send them looking for a key their
payload does not contain"). ``test_tool_spec.py`` independently keeps a
``_schema_name`` helper for the same mapping, to grade the very table these
refusals report from. What was missing was one owner for the rule, consulted by
every site that names a field.

The sweeps below are derived from ``_FIELD_ALIASES`` and from the schema, so an
alias added later is graded without a second list to keep true.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import (  # noqa: E402
    MuJoCoSimEngine,
    Simulation,
)

_SPEC = json.loads((Path(inspect.getfile(MuJoCoSimEngine)).parent / "tool_spec.json").read_text())
_PUBLISHED: frozenset[str] = frozenset(_SPEC["properties"]) - {"action"}

# Method parameters the schema reaches only under a different name. The
# dispatcher is deliberately wider than the schema - ``stop_recording`` accepts
# ``bucket`` / ``run_id`` from Python without publishing either - so an
# unpublished name is not by itself a defect. What a refusal may not do is name
# THESE, because the same field has a published spelling the caller can supply.
_ALIASED_AWAY: frozenset[str] = frozenset(
    param for wire, param in MuJoCoSimEngine._FIELD_ALIASES.items() if wire in _PUBLISHED and param not in _PUBLISHED
)

# A bad value for each published JSON type, chosen so the router refuses before
# the method body runs. These sweeps measure the refusal, not the action.
_BAD_BY_TYPE: dict[str, Any] = {
    "array": [1, 2],  # short vector: refused by the arity check
    "string": 7,
    "object": "notadict",
}

# Actions whose method takes ``**kwargs``: the unknown-parameter check is
# skipped for them by design (they forward or reject the residual keys
# themselves), so they publish no "Valid:" list to grade.
_VAR_KEYWORD_ACTIONS: frozenset[str] = frozenset(
    action
    for action in _SPEC["properties"]["action"]["enum"]
    if (method := getattr(MuJoCoSimEngine, MuJoCoSimEngine._ACTION_ALIASES.get(action, action), None)) is not None
    and any(p.kind is inspect.Parameter.VAR_KEYWORD for p in inspect.signature(method).parameters.values())
)


def _required_parameters(action: str) -> list[str]:
    """Named parameters of *action*'s method that carry no default."""
    method = getattr(MuJoCoSimEngine, MuJoCoSimEngine._ACTION_ALIASES.get(action, action), None)
    if method is None:  # pragma: no cover - every published action resolves today
        return []
    return [
        name
        for name, p in inspect.signature(method).parameters.items()
        if name != "self"
        and p.default is inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    ]


# Actions the router refuses before the method body runs, because the payload is
# missing a parameter the signature requires. Only these may be dispatched bare:
# an action whose parameters all carry defaults BINDS, so ``sim(action=X)``
# executes it on the live world instead of producing the refusal under test --
# which for ``start_recording`` means creating a dataset in the shared cache
# (rule 15) and for ``open_viewer`` means reaching ``launch_passive``.
_REQUIRED_PARAM_ACTIONS: frozenset[str] = frozenset(
    action for action in _SPEC["properties"]["action"]["enum"] if _required_parameters(action)
)


@pytest.fixture
def sim() -> Generator[Simulation, None, None]:
    """A world holding one body, so a refusal is the only thing under test."""
    engine = Simulation(tool_name="refusal_spelling")
    try:
        assert engine(action="create_world")["status"] == "success"
        built = engine(action="add_object", name="cube", shape="box", position=[0.2, 0.0, 0.1], size=[0.05] * 3)
        assert built["status"] == "success", built
        yield engine
    finally:
        engine.destroy()


def _text(result: dict[str, Any]) -> str:
    """The refusal text of a structured tool result."""
    return " ".join(block.get("text", "") for block in result.get("content", []))


def _valid_list(text: str) -> list[str]:
    """The names advertised by an ``Unknown parameter ... Valid: [...]`` refusal."""
    match = re.search(r"Valid: \[(.*?)\]", text)
    assert match is not None, f"refusal carried no Valid: list: {text}"
    return re.findall(r"'([^']+)'", match.group(1))


def _aliases_by_kind(kind: str) -> list[tuple[str, str]]:
    """``(wire, param)`` alias pairs whose published property has JSON type *kind*."""
    return sorted(
        (wire, param)
        for wire, param in MuJoCoSimEngine._FIELD_ALIASES.items()
        if _SPEC["properties"].get(wire, {}).get("type") == kind
    )


class TestTheAliasMapIsWhatMakesThisReachable:
    """Premise: the sweeps below are graded against a real asymmetry."""

    def test_at_least_one_canonical_parameter_name_is_unpublished(self) -> None:
        """Without one, every sweep here would hold for a router that ignores aliases.

        ``torque`` is that name today. If the schema ever publishes it, this test
        is the one that says the sweeps have gone vacuous.
        """
        unpublished = sorted(set(MuJoCoSimEngine._FIELD_ALIASES.values()) - _PUBLISHED)
        assert unpublished, (
            "premise: some _FIELD_ALIASES target must be absent from the schema, "
            "or a refusal naming the target is indistinguishable from one naming the wire field"
        )
        assert "torque" in unpublished, unpublished
        assert _ALIASED_AWAY == frozenset({"torque"}), sorted(_ALIASED_AWAY)

    def test_every_alias_wire_name_is_itself_published(self) -> None:
        """A wire name is what a model emits, so the spelling reported must exist.

        The fallback in the reporting rule keeps a Python-only parameter
        reportable under its own name; this pins that the aliases in play today
        never need it.
        """
        unpublished_wires = sorted(set(MuJoCoSimEngine._FIELD_ALIASES) - _PUBLISHED)
        assert unpublished_wires == ["camera_names", "joint_positions"], unpublished_wires
        for wire, param in MuJoCoSimEngine._FIELD_ALIASES.items():
            assert wire in _PUBLISHED or param in _PUBLISHED, (wire, param)


class TestAVectorRefusalNamesTheFieldTheCallerSent:
    """The arity and element checks report the spelling the payload carried."""

    @pytest.mark.parametrize(("wire", "param"), _aliases_by_kind("array"))
    def test_a_short_vector_sent_under_a_wire_name_is_refused_under_that_name(
        self, sim: Simulation, wire: str, param: str
    ) -> None:
        """Pre-fix ``torque_vec=[1, 2]`` was answered "Parameter 'torque' ...".

        The caller's payload has no ``torque`` key and the schema has no
        ``torque`` property, so neither the request nor the contract offered a
        way to act on that name.
        """
        result = sim(action="apply_force", body_name="cube", **{wire: _BAD_BY_TYPE["array"]})
        assert result["status"] == "error"
        text = _text(result)
        assert f"'{wire}'" in text, text
        assert f"'{param}'" not in text or param == wire, text

    @pytest.mark.parametrize(("wire", "param"), _aliases_by_kind("array"))
    def test_a_non_numeric_component_sent_under_a_wire_name_is_refused_under_that_name(
        self, sim: Simulation, wire: str, param: str
    ) -> None:
        """The per-component refusal reports from the same table, so it needs the same rule."""
        result = sim(action="apply_force", body_name="cube", **{wire: [1, 2, "up"]})
        assert result["status"] == "error"
        text = _text(result)
        assert f"'{wire}'[2]" in text, text
        assert f"'{param}'" not in text or param == wire, text

    def test_a_vector_with_no_alias_keeps_its_own_name(self, sim: Simulation) -> None:
        """Most vectors are spelled the same on both sides; that path is unchanged."""
        result = sim(action="apply_force", body_name="cube", force=[1, 2])
        assert result["status"] == "error"
        assert "Parameter 'force' must be a list of 3 numbers, got 2." in _text(result)

    def test_a_python_caller_who_writes_the_canonical_name_is_named_by_it(self, sim: Simulation) -> None:
        """The rule reports what the caller wrote, not what the schema prefers.

        ``torque`` is reachable from Python even though the schema does not
        publish it, and rewriting that refusal to ``torque_vec`` would name a key
        the payload does not carry - the same mistake in the other direction.
        """
        result = sim(action="apply_force", body_name="cube", torque=[1, 2])
        assert result["status"] == "error"
        assert "Parameter 'torque' must be a list of 3 numbers, got 2." in _text(result)


class TestTheBareDispatchSweepCannotExecuteAnAction:
    """A sweep that dispatches bare may only name actions the router refuses.

    ``sim(action=X)`` with no other key is a refusal probe, but it is only a
    probe for an action whose signature requires something: when every parameter
    carries a default the payload BINDS, and the method runs against the live
    world. Two of those reach outside the test process -- ``start_recording``
    resolves ``repo_id="local/sim_recording"`` with no ``root`` and creates it
    under ``$HF_LEROBOT_HOME`` (AGENTS.md rule 15, whose measured cost is one
    planted dataset turning 133 passes into 22 ``FileExistsError`` failures in
    unrelated modules), and ``open_viewer`` reaches
    ``mujoco.viewer.launch_passive`` on any host that is not headless. Neither is
    visible on CI, where ``DISPLAY`` is unset and no dataset exists yet.

    So the filter is a property of the sweep, not a convenience, and these tests
    fail if it is widened back to the published enum.
    """

    def test_no_graded_action_can_be_called_with_no_arguments(self) -> None:
        """The mechanical form of "the router refuses it before the body runs".

        Binding the signature with nothing but ``self`` is what
        ``_dispatch_action`` effectively attempts for a bare payload; a method
        that binds is a method that executes.
        """
        bindable = []
        for action in sorted(_REQUIRED_PARAM_ACTIONS):
            method = getattr(MuJoCoSimEngine, MuJoCoSimEngine._ACTION_ALIASES.get(action, action))
            try:
                inspect.signature(method).bind(object())
            except TypeError:
                continue
            bindable.append(action)
        assert not bindable, f"bare dispatch would execute these on the live world: {bindable}"

    @pytest.mark.parametrize(
        ("action", "effect"),
        [
            ("start_recording", "creates a dataset under $HF_LEROBOT_HOME (rule 15)"),
            ("open_viewer", "reaches mujoco.viewer.launch_passive on a host with a display"),
        ],
    )
    def test_an_action_with_a_side_effect_outside_the_process_is_not_swept(self, action: str, effect: str) -> None:
        """Named individually, because these are the two the filter exists for."""
        assert action in _SPEC["properties"]["action"]["enum"], f"premise: {action} is published"
        assert action not in _REQUIRED_PARAM_ACTIONS, (
            f"{action} would be dispatched bare by the sweep above, which {effect}"
        )

    def test_the_bare_dispatch_sweep_is_parametrized_over_the_filtered_set(self) -> None:
        """Pins the decorator, not just the set it is derived from.

        The two tests above grade what :data:`_REQUIRED_PARAM_ACTIONS` contains,
        which stays true if the sweep is widened back to the published enum
        beside them. This reads the sweep's own parametrization, so that edit is
        what fails rather than the next developer's cache.
        """
        suite = TestARefusalAdvertisesOnlyPublishedSpellings
        sweep = suite.test_a_missing_required_parameter_is_named_by_a_published_spelling
        marks = [m for m in getattr(sweep, "pytestmark", []) if m.name == "parametrize"]
        assert len(marks) == 1, marks
        argnames, argvalues = marks[0].args
        assert argnames == "action", argnames
        assert sorted(argvalues) == sorted(_REQUIRED_PARAM_ACTIONS), sorted(argvalues)

    def test_the_filter_still_grades_most_of_the_published_surface(self) -> None:
        """Non-vacuity: a derivation that collapsed to nothing would report clean."""
        assert len(_REQUIRED_PARAM_ACTIONS) >= 20, sorted(_REQUIRED_PARAM_ACTIONS)
        assert _REQUIRED_PARAM_ACTIONS <= set(_SPEC["properties"]["action"]["enum"])


class TestARefusalAdvertisesOnlyPublishedSpellings:
    """Whatever a refusal offers as a remedy, a model has to be able to emit it."""

    @pytest.mark.parametrize(
        "action",
        sorted(set(_SPEC["properties"]["action"]["enum"]) - _VAR_KEYWORD_ACTIONS),
    )
    def test_the_valid_list_names_only_fields_the_schema_publishes(self, sim: Simulation, action: str) -> None:
        """Pre-fix ``apply_force`` advertised ``torque``, which is not a property.

        The list is built from the method signature, so a parameter whose
        published spelling differs reached the caller under a name the schema
        does not carry - while the spelling that works went unmentioned.
        """
        payload: dict[str, Any] = {"action": action, "definitely_not_a_field": 1}
        result = sim(**payload)
        assert result["status"] == "error"
        text = _text(result)
        assert "Unknown parameter" in text, text
        offered = set(_valid_list(text))
        assert not (offered & _ALIASED_AWAY), (
            f"action {action!r} advertised {sorted(offered & _ALIASED_AWAY)}, which the schema "
            f"reaches only under another name: {text}"
        )

    def test_a_parameter_with_no_published_spelling_is_still_advertised(self, sim: Simulation) -> None:
        """The dispatcher's deliberate width is not what this narrows.

        ``stop_recording`` takes ``bucket`` / ``run_id`` from Python and publishes
        neither, and rewriting the reporting rule to name only schema properties
        would drop them from the list of what the action accepts. Only a
        parameter the schema reaches under ANOTHER name is substituted.
        """
        result = sim(action="stop_recording", definitely_not_a_field=1)
        assert result["status"] == "error"
        offered = set(_valid_list(_text(result)))
        assert {"bucket", "run_id"} <= offered, sorted(offered)
        assert not ({"bucket", "run_id"} & _PUBLISHED), "premise: neither may be a published property"

    @pytest.mark.parametrize("action", sorted(_REQUIRED_PARAM_ACTIONS))
    def test_a_missing_required_parameter_is_named_by_a_published_spelling(self, sim: Simulation, action: str) -> None:
        """The required-parameter refusal reports a signature name too.

        No required parameter is aliased away today, so this sweep passes either
        way - it is here so that the day one is, the refusal names a field the
        caller can supply rather than the day the report is read.

        Dispatched bare, so it is parametrized over the actions the router
        refuses before the method body runs; see :data:`_REQUIRED_PARAM_ACTIONS`
        for why the rest are not swept here.
        """
        result = sim(action=action)
        assert result["status"] == "error", result
        text = _text(result)
        match = re.search(r"requires parameter '([A-Za-z_][A-Za-z0-9_]*)'", text)
        assert match is not None, f"action {action!r} was refused for another reason: {text}"
        assert match.group(1) not in _ALIASED_AWAY, text


class TestTheStringCheckKeepsTheBehaviourItAlreadyHad:
    """The one site that already resolved the spelling now shares the rule."""

    @pytest.mark.parametrize(("wire", "param"), _aliases_by_kind("string"))
    def test_a_non_string_sent_under_a_wire_name_is_refused_under_that_name(
        self, sim: Simulation, wire: str, param: str
    ) -> None:
        """``checkpoint_name`` was already reported correctly; it still is."""
        result = sim(action="save_state", **{wire: 7})
        assert result["status"] == "error"
        text = _text(result)
        assert f"'{wire}' must be a string" in text, text

    def test_a_string_with_no_alias_keeps_its_own_name(self, sim: Simulation) -> None:
        """Folding the two sites into one rule must not move the common case."""
        result = sim(action="load_scene", scene_path=7)
        assert result["status"] == "error"
        assert "'scene_path' must be a string" in _text(result)


class TestOneOwnerForTheReportingRule:
    """Root cause: five refusal sites, one of which resolved the spelling."""

    def test_no_refusal_in_the_validator_formats_a_raw_parameter_name(self) -> None:
        """Every site names the field through the shared rule, not the loop variable.

        A site that interpolates the parameter it is iterating over reports the
        post-rewrite name, which is what made ``torque`` reachable in four
        messages at once.
        """
        source = inspect.getsource(MuJoCoSimEngine._validate_and_build_kwargs)
        for raw in ("{vparam}", "{param_name}'."):
            assert raw not in source, f"a refusal still formats the raw name {raw!r}"
        assert source.count("_reported_param_name(") >= 4, source.count("_reported_param_name(")

    def test_the_reported_spelling_prefers_the_payload_then_the_schema(self) -> None:
        """The rule's three branches, stated directly on the helper."""
        from strands_robots.simulation.mujoco.simulation import _reported_param_name

        aliases = {"torque_vec": "torque"}
        assert _reported_param_name("torque", aliases, {"torque": [1]}) == "torque"
        assert _reported_param_name("torque", aliases, {"torque_vec": [1]}) == "torque_vec"
        assert _reported_param_name("torque", aliases, {}) == "torque_vec"
        assert _reported_param_name("force", aliases, {}) == "force"
        assert _reported_param_name("bucket", aliases, {}) == "bucket"
