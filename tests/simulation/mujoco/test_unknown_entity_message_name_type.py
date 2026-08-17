"""The unknown-entity messages list what IS registered whatever the name's type.

``registered()`` is deliberately total (#1776): a name that cannot be a registry
key - a list, a dict, an int, ``None`` - resolves to "no such entity" so it is
*reported* through the unknown-entity message rather than raising
``TypeError: unhashable type`` out of a method whose only documented failure
channel is the agent-tool dict. Its docstring says so: such a name "resolves to
``False``, which lets the caller report it with the message it already has".

Those messages could not honor that handoff. Each gated its whole tail on one
condition, ``if known and isinstance(requested, str)``, but only the
:func:`difflib.get_close_matches` suggestion needs a string. So for every
non-``str`` name the availability listing and the discovery action - facts about
the world, not about the requested name - were suppressed too, and three of the
five helpers then fell into an ``else`` that asserted a falsehood:

    Robot '['front']' not found. No robots in the scene; add one with action='add_robot'.

with ``arm`` registered. The remaining two returned the bare dead-end
``"<Kind> 'X' not found."`` that #1299/#1303/#1306 introduced these helpers to
eliminate - ``test_missing_entity_error_messages`` states that contract: the
message "names the entity, offers a close match, lists the available names, and
points at the discovery action".

A non-``str`` name is not exotic here: ``robot_name`` is the *first positional*
parameter of ``move_to`` / ``set_gripper`` / ``rotate_wrist``, so
``sim.move_to([0.2, 0.0, 0.1])`` lands a position list in it.

These tests pin the split: what is registered is reported for every name type,
the empty-scene claim is only made when the scene really is empty, and a
``str`` name's message is unchanged. GL-free (``mesh=False``, no rendering).
"""

import ast
import inspect
import pathlib
from typing import Any

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.base import SimEngine, close_match_hint  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

ARM_MJCF = """<mujoco model="arm"><compiler angle="radian"/>
 <worldbody><body name="base" pos="0 0 0.05"><geom type="box" size="0.05 0.05 0.05"/>
  <body name="link" pos="0 0 0.06"><joint name="pan" type="hinge" axis="0 0 1" range="-2 2" damping="4"/>
   <geom type="capsule" fromto="0 0 0 0.18 0 0" size="0.02"/>
   <site name="tcp" pos="0.18 0 0" size="0.005"/></body></body></worldbody>
 <actuator><position name="a_pan" joint="pan" kp="50" ctrlrange="-2 2"/></actuator>
 <sensor><framepos name="tcp_pos" objtype="site" objname="tcp"/></sensor></mujoco>
"""

# Names that cannot be a registry key, so registered() routes them here.
NOT_A_NAME: list[Any] = [
    pytest.param(["arm"], id="list"),
    pytest.param({"name": "arm"}, id="dict"),
    pytest.param(7, id="int"),
    pytest.param(None, id="None"),
    pytest.param(2.5, id="float"),
]

# ``get_sensor_data`` reads ``sensor_name`` behind ``if sensor_name and ...``, so
# ``None`` is its documented "every sensor" spelling rather than a bad name.
# Derived, so a value added above cannot quietly escape the strict subset.
NOT_A_NAME_STRICT = [case for case in NOT_A_NAME if case.id != "None"]

# The falsehood each of the three "else" branches used to assert.
FALSE_EMPTY_CLAIMS = ("No robots in the scene", "No objects in the scene")


@pytest.fixture
def populated(tmp_path):
    """A scene that really does hold a robot, an object and a camera."""
    arm = tmp_path / "arm.xml"
    arm.write_text(ARM_MJCF, encoding="utf-8")
    sim = Simulation(tool_name="test_unknown_entity_name_type", mesh=False)
    assert sim.create_world(gravity=[0, 0, 0])["status"] == "success"
    assert sim.add_robot(name="arm", urdf_path=str(arm))["status"] == "success"
    assert (
        sim.add_object(name="crate", shape="box", size=[0.06, 0.06, 0.06], position=[0.3, 0, 0.03])["status"]
        == "success"
    )
    assert sim.add_camera(name="look", position=[0.5, -0.4, 0.3], target=[0, 0, 0.1])["status"] == "success"
    yield sim
    sim.cleanup()


@pytest.fixture
def empty_world():
    """A world with no robots and no objects, so the empty-scene claim is true."""
    sim = Simulation(tool_name="test_unknown_entity_name_type_empty", mesh=False)
    assert sim.create_world(gravity=[0, 0, 0])["status"] == "success"
    yield sim
    sim.cleanup()


def _messages(sim: Simulation, requested: Any) -> dict[str, str]:
    """Every unknown-entity message the helpers produce for one requested name."""
    return {
        "base.robot": SimEngine._unknown_robot_msg(sim, requested),
        "mujoco.robot": sim._unknown_robot_msg(requested),
        "mujoco.object": sim._unknown_object_msg(requested),
        "mujoco.camera": sim._unknown_camera_msg(requested),
        "entity.Body": sim._unknown_mj_entity_msg("Body", requested),
        "entity.Joint": sim._unknown_mj_entity_msg("Joint", requested),
        "entity.Sensor": sim._unknown_mj_entity_msg("Sensor", requested),
    }


def _text(result: dict[str, Any]) -> str:
    assert result["status"] == "error", result
    return next(block["text"] for block in result["content"] if "text" in block)


class TestTheListingIsNotGatedOnTheNameType:
    """What is registered is a fact about the world, not about the requested name."""

    @pytest.mark.parametrize("requested", NOT_A_NAME)
    def test_every_helper_still_lists_what_is_registered(self, populated, requested):
        for label, msg in _messages(populated, requested).items():
            assert "Available" in msg, f"{label} gave a dead end for {requested!r}: {msg}"

    @pytest.mark.parametrize("requested", NOT_A_NAME)
    def test_no_helper_claims_the_populated_scene_is_empty(self, populated, requested):
        for label, msg in _messages(populated, requested).items():
            for claim in FALSE_EMPTY_CLAIMS:
                assert claim not in msg, f"{label} asserted {claim!r} for {requested!r}: {msg}"

    @pytest.mark.parametrize("requested", NOT_A_NAME)
    def test_every_helper_still_names_the_requested_value(self, populated, requested):
        """The caller has to be able to see which value was rejected."""
        for label, msg in _messages(populated, requested).items():
            assert f"'{requested}'" in msg, f"{label} did not name {requested!r}: {msg}"

    @pytest.mark.parametrize("requested", NOT_A_NAME)
    def test_no_suggestion_is_invented_for_a_name_with_no_characters_to_match(self, populated, requested):
        """difflib is the one part that genuinely needs a string, so it stays out."""
        for label, msg in _messages(populated, requested).items():
            assert "Did you mean" not in msg, f"{label} suggested a match for {requested!r}: {msg}"


class TestThePublicSurfacesReportIt:
    """Driven through the public API, not the private helpers."""

    def test_a_positional_position_lands_in_robot_name_and_is_reported_usefully(self, populated):
        """``move_to``'s first positional is ``robot_name`` - the realistic mistake."""
        text = _text(populated.move_to([0.2, 0.0, 0.1], position=[0.2, 0.0, 0.1], max_steps=2))
        assert "Available robots: ['arm']" in text, text
        assert "No robots in the scene" not in text, text

    @pytest.mark.parametrize(
        ("label", "call", "expected"),
        [
            ("set_gripper", lambda s: s.set_gripper(["arm"], state="open", steps=2), "Available robots: ['arm']"),
            (
                "rotate_wrist",
                lambda s: s.rotate_wrist(["arm"], target_yaw=0.1, max_steps=2),
                "Available robots: ['arm']",
            ),
            ("send_action", lambda s: s.send_action({"a_pan": 0.1}, robot_name=["arm"]), "Available robots: ['arm']"),
            ("move_object", lambda s: s.move_object(["crate"], position=[0.1, 0, 0.1]), "Available objects: ['crate']"),
            ("remove_object", lambda s: s.remove_object(["crate"]), "Available objects: ['crate']"),
            ("get_body_state", lambda s: s.get_body_state(["arm/base"]), "Available bodies:"),
            ("get_sensor_data", lambda s: s.get_sensor_data(["tcp_pos"]), "Available sensors:"),
        ],
    )
    def test_the_surface_lists_what_is_registered(self, populated, label, call, expected):
        text = _text(call(populated))
        assert expected in text, f"{label}: {text}"
        for claim in FALSE_EMPTY_CLAIMS:
            assert claim not in text, f"{label} asserted {claim!r}: {text}"


class TestAStringNameIsUnaffected:
    """Over-reach control: the dominant path keeps its close match and listing."""

    def test_a_typo_still_gets_a_close_match_and_the_listing(self, populated):
        for label, msg in _messages(populated, "arm0").items():
            assert "Available" in msg, f"{label}: {msg}"
        assert "Did you mean: arm?" in populated._unknown_robot_msg("arm0")
        assert "Did you mean: crate?" in populated._unknown_object_msg("crat")

    def test_the_robot_message_is_the_documented_shape(self, populated):
        assert populated._unknown_robot_msg("arm0") == (
            "Robot 'arm0' not found. Did you mean: arm? Available robots: ['arm']. Use action='list_robots' to see all."
        )


class TestTheEmptySceneClaimStillHolds:
    """Non-vacuity: the ``else`` branch is only reached when the scene IS empty."""

    @pytest.mark.parametrize("requested", [pytest.param("arm", id="str"), *NOT_A_NAME])
    def test_an_empty_world_is_reported_as_empty(self, empty_world, requested):
        assert "No robots in the scene" in empty_world._unknown_robot_msg(requested)
        assert "No objects in the scene" in empty_world._unknown_object_msg(requested)
        assert "No robots in the scene" in SimEngine._unknown_robot_msg(empty_world, requested)


class TestCloseMatchHint:
    """The one place that needs a ``str`` - and the only thing it decides."""

    def test_a_close_string_is_suggested(self):
        assert close_match_hint("crat", ["crate", "table"]) == " Did you mean: crate?"

    def test_several_matches_are_ordered_and_capped(self):
        hint = close_match_hint("arm", ["arm0", "arm1", "arm2", "arm3"])
        assert hint.startswith(" Did you mean: ")
        assert hint.count(",") == 2, hint  # n=3

    @pytest.mark.parametrize("requested", NOT_A_NAME)
    def test_a_name_with_no_characters_to_match_yields_nothing(self, requested):
        assert close_match_hint(requested, ["crate"]) == ""

    def test_an_empty_registry_yields_nothing(self):
        assert close_match_hint("crate", []) == ""

    def test_a_string_with_no_close_match_yields_nothing(self):
        assert close_match_hint("zzzzzzzz", ["crate", "table"]) == ""

    def test_the_fragment_is_appendable_as_is(self):
        """It carries its own leading space so a caller appends it unconditionally."""
        assert close_match_hint("crat", ["crate"]).startswith(" ")
        assert close_match_hint(["crate"], ["crate"]) == ""


class TestTheReportIsReachedAtAll:
    """Two lookups raised before their own report, so the message never ran.

    ``registered()`` makes the world registries total, but two dict membership
    tests built locally were still bare: ``sensor_name not in sensors`` in
    ``get_sensor_data`` and ``name not in checkpoints`` in ``load_state``. Both
    sit directly in front of a "not found. Available: [...]" report, so an
    unhashable name raised ``TypeError: unhashable type`` out of a method whose
    only documented failure channel is the agent-tool dict - the exact escape
    ``registered()`` exists to close, one statement earlier.
    """

    @pytest.mark.parametrize("requested", NOT_A_NAME_STRICT)
    def test_get_sensor_data_reports_instead_of_raising(self, populated, requested):
        text = _text(populated.get_sensor_data(requested))
        assert "Available sensors:" in text, text
        assert f"'{requested}'" in text, text

    def test_an_omitted_sensor_name_still_reads_every_sensor(self, populated):
        """``None`` is the documented "no filter" spelling, not a bad name."""
        result = populated.get_sensor_data(None)
        assert result["status"] == "success", result
        payload = next(b["json"] for b in result["content"] if "json" in b)
        assert set(payload["sensors"]) == {"arm/tcp_pos"}, payload

    @pytest.mark.parametrize("requested", NOT_A_NAME)
    def test_load_state_reports_instead_of_raising(self, populated, requested):
        assert populated.save_state(name="cp1")["status"] == "success"
        text = _text(populated.load_state(requested))
        assert "Checkpoint" in text and "not found" in text, text
        assert "Available: ['cp1']" in text, text

    def test_a_real_sensor_name_still_reads_it(self, populated):
        """Over-reach control: the honored lookup is unchanged."""
        result = populated.get_sensor_data("arm/tcp_pos")
        assert result["status"] == "success", result
        payload = next(b["json"] for b in result["content"] if "json" in b)
        assert "arm/tcp_pos" in payload["sensors"], payload

    def test_a_real_checkpoint_still_loads(self, populated):
        """Over-reach control: the honored lookup is unchanged."""
        assert populated.save_state(name="cp1")["status"] == "success"
        assert populated.load_state("cp1")["status"] == "success"


# -- structural guard ------------------------------------------------------

# Derived from a symbol this module already imports, so a rename cannot
# silently point the sweep at a directory that holds no helpers.
_SCAN_ROOT = pathlib.Path(inspect.getfile(close_match_hint)).parent

_EXPECTED_HELPERS = {
    "base.py::_unknown_robot_msg",
    "mujoco/manipulation.py::_unknown_robot_msg",
    "mujoco/motion_primitives.py::_unknown_robot_msg",
    "mujoco/physics.py::_unknown_mj_entity_msg",
    "mujoco/physics.py::_unknown_robot_msg",
    "mujoco/simulation.py::_unknown_action_msg",
    "mujoco/simulation.py::_unknown_camera_msg",
    "mujoco/simulation.py::_unknown_model_msg",
    "mujoco/simulation.py::_unknown_object_msg",
    "mujoco/simulation.py::_unknown_robot_msg",
}


def _unknown_entity_helpers(source: str) -> dict[str, ast.FunctionDef]:
    """Map ``name`` -> node for every ``_unknown_*_msg`` helper in *source*."""
    tree = ast.parse(source)
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_unknown_") and node.name.endswith("_msg")
    }


def _gates_on_the_name_type(node: ast.FunctionDef) -> bool:
    """True when the helper tests the requested name's type itself."""
    for call in ast.walk(node):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "isinstance"):
            continue
        if call.args and isinstance(call.args[0], ast.Name) and call.args[0].id == "requested":
            return True
    return False


def _discovered() -> dict[str, ast.FunctionDef]:
    found: dict[str, ast.FunctionDef] = {}
    for path in sorted(_SCAN_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for name, node in _unknown_entity_helpers(source).items():
            found[f"{path.relative_to(_SCAN_ROOT).as_posix()}::{name}"] = node
    return found


class TestNoHelperGatesTheListingOnTheNameType:
    """A sixth helper cannot ship the conflation this change removed."""

    def test_the_scan_found_the_known_helpers(self):
        """Non-vacuity: a mis-rooted scan must not report a clean sweep."""
        assert set(_discovered()) == _EXPECTED_HELPERS, set(_discovered())

    def test_no_helper_tests_the_requested_names_type(self):
        offenders = sorted(qualname for qualname, node in _discovered().items() if _gates_on_the_name_type(node))
        assert not offenders, (
            "An unknown-entity message must not decide what to list from the requested "
            "name's type: only the difflib suggestion needs a string, and gating the "
            "availability listing on it suppresses a fact about the world for every "
            "non-str name. Use close_match_hint(requested, known) and branch on "
            f"`known` alone. Offenders: {offenders}"
        )

    def test_the_scanner_detects_a_planted_conflation(self):
        planted = (
            "def _unknown_thing_msg(self, requested: object) -> str:\n"
            '    """Planted."""\n'
            "    known = self.things()\n"
            "    msg = f\"Thing '{requested}' not found.\"\n"
            "    if known and isinstance(requested, str):\n"
            '        msg += f" Available: {known}."\n'
            "    return msg\n"
        )
        helpers = _unknown_entity_helpers(planted)
        assert set(helpers) == {"_unknown_thing_msg"}
        assert _gates_on_the_name_type(helpers["_unknown_thing_msg"])

    def test_the_scanner_accepts_the_fixed_shape(self):
        fixed = (
            "def _unknown_thing_msg(self, requested: object) -> str:\n"
            '    """Fixed."""\n'
            "    known = self.things()\n"
            "    msg = f\"Thing '{requested}' not found.\"\n"
            "    if known:\n"
            "        msg += close_match_hint(requested, known)\n"
            '        msg += f" Available: {known}."\n'
            "    return msg\n"
        )
        helpers = _unknown_entity_helpers(fixed)
        assert not _gates_on_the_name_type(helpers["_unknown_thing_msg"])
