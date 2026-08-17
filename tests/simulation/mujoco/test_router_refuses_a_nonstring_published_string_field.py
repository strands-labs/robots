"""A field published as a string is refused unless it is one.

``_dispatch_action``'s documented validation layer refuses an unknown parameter,
refuses a missing required one "(no raw Python ``TypeError``)", and checks a
vector's length and dtype "before the value reaches numpy / MuJoCo". A scalar
string field had no such layer, so a non-string value was carried into the method
body and failed wherever that body first assumed a string -- reaching the caller
as ``AttributeError: 'int' object has no attribute 'lower'``,
``TypeError: stat: path should be string, bytes, os.PathLike or integer, not
list``, or ``TypeError: unhashable type: 'list'``. None of those names the
parameter, and an agent cannot act on any of them.

The tests below sweep the published surface for that shape, then bound the fix:
the type is the only thing refused, the schema is what the enforced set is
derived from, and the one field whose method accepts a wider domain
(``keyframe``, a name *or* an index) is still free to use it.
"""

from __future__ import annotations

import inspect
import json
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

# Actions that run a rollout, spawn a viewer, or tear the session down. This
# sweep measures a *refusal*, which happens before the method runs, so nothing
# here needs to execute -- excluding them keeps the sweep to seconds.
_NOT_SWEPT = frozenset(
    {
        "run_policy",
        "start_policy",
        "stop_policy",
        "eval_policy",
        "replay_episode",
        "evaluate_benchmark",
        "run_multi_policy",
        "start_multi_policy",
        "open_viewer",
        "destroy",
        "cleanup",
    }
)

# Three shapes a JSON payload can carry instead of a string. The list and dict
# are not incidental: they are what produced the ``unhashable type`` failures.
_NON_STRINGS: tuple[Any, ...] = (7, ["x"], {"k": 1})

_KEYFRAME_MJCF = """
<mujoco model="kf">
  <worldbody>
    <body name="link" pos="0 0 0.2">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom name="g" type="capsule" fromto="0 0 0 0.12 0 0" size="0.02" mass="0.3"/>
    </body>
  </worldbody>
  <actuator><position name="a" joint="j" kp="30"/></actuator>
  <keyframe><key name="home" qpos="0.4" ctrl="0.4"/></keyframe>
</mujoco>
"""


def _fresh() -> Simulation:
    s = Simulation(tool_name="string_field_test", mesh=False)
    assert s.create_world()["status"] == "success"
    return s


@pytest.fixture
def sim() -> Generator[Simulation, None, None]:
    s = _fresh()
    yield s
    s.cleanup()


def _published_string_fields() -> set[str]:
    """Wire names the schema declares as a string, excluding ``action``."""
    return {
        wire
        for wire, prop in _SPEC["properties"].items()
        if wire != "action" and (prop.get("type") == "string" or prop.get("type") == ["string"])
    }


def _swept_sites() -> list[tuple[str, str]]:
    """``(action, wire_field)`` pairs the sweep drives.

    A field applies to an action when the action's method declares the parameter
    it maps to, which is how the dispatcher itself decides what an action accepts.
    """
    aliases = MuJoCoSimEngine._FIELD_ALIASES
    sites: list[tuple[str, str]] = []
    for action in _SPEC["properties"]["action"]["enum"]:
        if action in _NOT_SWEPT:
            continue
        method = getattr(MuJoCoSimEngine, MuJoCoSimEngine._ACTION_ALIASES.get(action, action), None)
        if method is None:
            continue
        try:
            params = {n for n in inspect.signature(method).parameters if n != "self"}
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
        for wire in sorted(_published_string_fields()):
            if aliases.get(wire, wire) in params:
                sites.append((action, wire))
    return sites


class TestEveryPublishedStringFieldIsRefusedWhenItIsNotAString:
    """No published string field carries a non-string into the method body."""

    def test_no_published_string_field_escapes_as_a_raw_exception(self) -> None:
        """A non-string must produce a refusal, never an exception.

        The dispatcher is the agent boundary: its result dict is the only channel
        a bad input is reported on, so an exception raised past it is not a worse
        message -- it is no message at all, and it takes the caller's ``except``
        clauses and any open recording with it.
        """
        sites = _swept_sites()
        assert len(sites) >= 30, f"premise: the sweep must cover the published surface, got {len(sites)} sites"

        sim = _fresh()
        escaped: list[str] = []
        try:
            for action, wire in sites:
                for bad in _NON_STRINGS:
                    try:
                        result = sim(action=action, **{wire: bad})
                    except Exception as exc:  # noqa: BLE001 - the failure under test
                        escaped.append(f"{action}({wire}={bad!r}) -> {type(exc).__name__}: {exc}")
                        sim.cleanup()
                        sim = _fresh()
                        continue
                    if not isinstance(result, dict) or "status" not in result:
                        escaped.append(
                            f"{action}({wire}={bad!r}) -> returned {type(result).__name__}, not a result dict"
                        )
                    elif result["status"] != "error":
                        escaped.append(f"{action}({wire}={bad!r}) -> status={result['status']!r}, expected 'error'")
        finally:
            sim.cleanup()

        assert not escaped, "a non-string reached the method body:\n  " + "\n  ".join(escaped[:12])

    def test_the_refusal_names_the_field_the_value_and_its_type(self, sim: Simulation) -> None:
        """A refusal a caller cannot act on is a dead end.

        Naming all three is what separates this from the raw exception it
        replaces: ``'int' object has no attribute 'lower'`` names neither the
        field the caller sent nor the value it sent.
        """
        result = sim(action="add_robot", data_config=7)
        assert result["status"] == "error"
        text = " ".join(block.get("text", "") for block in result["content"])
        assert "data_config" in text, text
        assert "7" in text, text
        assert "int" in text, text
        assert "must be a string" in text, text


class TestAStateSavedUnderANonStringNameCannotBeReloaded:
    """The silent form of the same defect, and the worse one.

    ``save_state`` keys its checkpoints by the name it is given, and a number is
    a perfectly good dict key -- so nothing raised. The save reported success,
    and because every agent-tool call carries the name as a string, no payload
    the caller could ever send would address that checkpoint again. The refusal
    they got instead listed it: ``Checkpoint '1' not found. Available: [1]``.
    """

    def test_a_numeric_checkpoint_name_is_refused_rather_than_stored(self, sim: Simulation) -> None:
        """A name only the Python path can produce must not be accepted from the wire."""
        saved = sim(action="save_state", checkpoint_name=1)
        assert saved["status"] == "error", saved

    def test_the_refusal_names_the_field_the_caller_actually_sent(self, sim: Simulation) -> None:
        """``checkpoint_name`` is an alias for ``name``, and the caller sent the alias.

        Reporting the canonical parameter would send them looking for a key their
        payload does not contain, which is the dead end this whole layer exists
        to remove.
        """
        text = " ".join(b.get("text", "") for b in sim(action="save_state", checkpoint_name=1)["content"])
        assert "checkpoint_name" in text, text

        canonical = " ".join(b.get("text", "") for b in sim(action="save_state", name=1)["content"])
        assert "'name'" in canonical, canonical

    def test_the_save_and_reload_round_trip_closes_on_a_string_name(self, sim: Simulation) -> None:
        """After the refusal, the retry a caller can actually make has to work.

        This is the behaviour the silent success replaced: the scene is restored,
        rather than the caller holding a checkpoint reference that resolves to
        nothing for the rest of the session.
        """
        assert (
            sim(action="add_object", name="cube", shape="box", size=[0.1, 0.1, 0.1], position=[0.3, 0, 0.05])["status"]
            == "success"
        )
        assert sim(action="save_state", checkpoint_name="1")["status"] == "success"
        assert sim(action="step", n_steps=40)["status"] == "success"
        restored = sim(action="load_state", checkpoint_name="1")
        assert restored["status"] == "success", restored


class TestTheCreationSiteGuardIsStillTheGuardOnThePythonPath:
    """The two layers agree on the verdict and differ only in which reports first.

    ``entity_name_error`` refuses three values at the creation sites; this layer
    refuses one of them, for every published field, at the boundary. Calling a
    method directly bypasses the dispatcher deliberately -- that is the Python
    path -- so the creation-site guard is not made unreachable by this.
    """

    @pytest.mark.parametrize("bad", _NON_STRINGS)
    def test_the_python_path_still_reaches_the_creation_site_guard(self, sim: Simulation, bad: Any) -> None:
        """A direct call does not pass through the dispatcher, so its guard reports."""
        result = sim.add_object(name=bad, shape="box", size=[0.05, 0.05, 0.05])
        assert result["status"] == "error"
        text = " ".join(b.get("text", "") for b in result["content"])
        assert "must be a non-empty string" in text, text

    def test_an_empty_name_is_still_refused_by_the_creation_site(self, sim: Simulation) -> None:
        """Emptiness is not a type error, so it stays the creation site's call.

        The boundary layer passes ``""`` through precisely because it is a valid
        string; the creation site is what knows it cannot address an entity.
        """
        result = sim(action="add_object", name="", shape="box", size=[0.05, 0.05, 0.05])
        assert result["status"] == "error"
        text = " ".join(b.get("text", "") for b in result["content"])
        assert "must be a non-empty string" in text, text


class TestTheRefusalStaysInsideItsOwnDomain:
    """Only the type is refused, and only for the fields the schema publishes."""

    def test_the_documented_integer_keyframe_index_is_still_accepted(self, sim: Simulation, tmp_path: Path) -> None:
        """``keyframe`` takes a name *or* an index, and both must stay reachable.

        ``add_robot`` annotates it ``str | int`` and documents both forms, so a
        rule derived from a schema that published only ``"string"`` would refuse
        a documented capability. The schema is what has to be true here, not the
        check.
        """
        model = tmp_path / "kf.xml"
        model.write_text(_KEYFRAME_MJCF)

        by_index = sim(action="add_robot", name="byindex", urdf_path=str(model), keyframe=0)
        assert by_index["status"] == "success", by_index

        by_name = sim(action="add_robot", name="byname", urdf_path=str(model), keyframe="home")
        assert by_name["status"] == "success", by_name

        state = sim(action="get_robot_state", robot_name="byindex")
        assert state["status"] == "success", state

    def test_an_empty_string_is_not_refused(self, sim: Simulation) -> None:
        """``""`` is a routing token here, not a type error.

        ``render(camera_name="")`` selects the free camera and an empty
        ``instruction`` is the documented rollout default, so the emptiness rule
        belongs at the creation sites ``entity_name_error`` guards.
        """
        result = sim(action="render", camera_name="", width=64, height=64)
        assert result["status"] == "success", result

    def test_a_str_subclass_is_accepted(self, sim: Simulation) -> None:
        """A ``str`` subclass is a string by every operation that follows."""

        class Label(str):
            pass

        added = sim(action="add_object", name=Label("cube"), shape="box", size=[0.1, 0.1, 0.1])
        assert added["status"] == "success", added

    def test_none_still_means_unset(self, sim: Simulation) -> None:
        """``None`` is the absence of a value, so the parameter default applies."""
        assert sim(action="add_object", name="probe", shape="box", mesh_path=None)["status"] == "success"

    def test_an_unknown_field_still_gets_the_unknown_parameter_refusal(self, sim: Simulation) -> None:
        """The unknown-parameter check runs first, so it keeps its own message.

        A field the action does not accept is a different mistake from a field
        sent with the wrong type, and reporting the second for the first would
        send the caller after a type that was never the problem.
        """
        result = sim(action="step", bogus_field=7)
        assert result["status"] == "error"
        text = " ".join(block.get("text", "") for block in result["content"])
        assert "Unknown parameter" in text, text
        assert "must be a string" not in text, text


class TestTheEnforcedSetIsDerivedFromThePublishedSchema:
    """The set follows the schema, so a field published later is covered."""

    def test_the_enforced_set_is_exactly_the_published_string_fields(self) -> None:
        """A second hand-written list would be a second thing to keep true."""
        aliases = MuJoCoSimEngine._FIELD_ALIASES
        expected = {aliases.get(wire, wire) for wire in _published_string_fields()}
        assert expected, "premise: the schema must publish string fields"
        assert MuJoCoSimEngine._PUBLISHED_STRING_PARAMS == frozenset(expected)

    def test_a_parameter_the_schema_does_not_publish_is_not_type_enforced(self) -> None:
        """The dispatcher is deliberately wider than the schema for Python callers.

        ``stop_recording`` accepts ``bucket`` / ``run_id`` from Python without
        publishing them, and enforcing a published type on an unpublished field
        would enforce a contract nobody was shown.
        """
        params = set(inspect.signature(MuJoCoSimEngine.stop_recording).parameters) - {"self"}
        unpublished = params - set(_SPEC["properties"])
        assert "bucket" in unpublished, f"premise: bucket must stay unpublished, got {sorted(unpublished)}"
        assert not (unpublished & MuJoCoSimEngine._PUBLISHED_STRING_PARAMS)

    def test_the_keyframe_field_publishes_the_integer_index_its_method_accepts(self) -> None:
        """The schema has to admit every form the method does.

        Narrowing it back to ``"string"`` would make the dispatcher refuse the
        documented index form, so the two are pinned together here.
        """
        assert set(_SPEC["properties"]["keyframe"]["type"]) == {"string", "integer"}
        annotation = inspect.signature(MuJoCoSimEngine.add_robot).parameters["keyframe"].annotation
        assert int in getattr(annotation, "__args__", ()), annotation
