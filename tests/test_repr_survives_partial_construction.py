"""``__repr__`` must survive an instance whose ``__init__`` never finished.

``repr`` is what a traceback, a debugger and a failing assertion render. A class
that validates its own arguments raises *before* it assigns the attributes its
``__repr__`` reads, and the raising frame keeps that half-built instance alive -
so rendering it reports ``[AttributeError ... raised in repr()]`` naming an
attribute that has nothing to do with the refusal under investigation. Pytest
shows it as::

    assert <[AttributeError("'RosBridgedRobot' object has no attribute
    'node_name'") raised in repr()] RosBridgedRobot object at 0x...> is None

which sends the reader after ``node_name`` when the real failure was
``ValueError: invalid node_name: 'bad!'`` - the value they already passed.

This module pins the contract behaviourally rather than structurally, because a
static check cannot tell a repr that *would* raise from one whose attributes
happen to be assigned early enough: the survey below constructs the worst case
of a half-built instance (``cls.__new__(cls)``, where no attribute exists at
all) and renders it. That needs no heuristic and therefore no exemption list.

Three properties are pinned:

* every class in the package that defines ``__repr__`` renders a half-built
  instance without raising, and still identifies its own type;
* a real, documented constructor refusal leaves an instance a reader can render,
  reporting the lifecycle fact and naming no attribute;
* a fully constructed instance is unaffected - it still names its fields, and
  never claims to be partially constructed.

The fallback wording has one owner,
:func:`strands_robots.utils.partial_construction_repr`, so the phrase a reader
learns to recognise in a traceback cannot diverge between the transport
bridges, the teleop input streams, the dataset recorder, the peer registry and
the simulation engines.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import textwrap
from typing import Any

import pytest

import strands_robots
from strands_robots.dataset_recorder import DatasetRecorder
from strands_robots.hardware_rtps_bridge import HardwareRtpsBridge
from strands_robots.mesh.core import Mesh
from strands_robots.mesh.input import InputPublisher, InputReceiver
from strands_robots.mesh.ros_bridge import RosBridgedRobot
from strands_robots.mesh.rosbridge_robot import RosbridgeRobot
from strands_robots.mesh.rtps_robot import RtpsRobot
from strands_robots.mesh.security import ValidationError
from strands_robots.mesh.session import PeerInfo
from strands_robots.policies.lerobot_local.processor import ProcessorBridge
from strands_robots.utils import partial_construction_repr

#: The phrase the shared fallback reports. Pinned here so a rewording has to be
#: a deliberate edit in two places rather than a silent drift in one.
FALLBACK_PHRASE = "partially constructed"

#: Every class in the package that defines ``__repr__``, as
#: ``"<module path relative to the package>::<class>"``. Asserted to equal what
#: the survey discovers, so a class that grows a ``__repr__`` later has to be
#: triaged here rather than joining the untested set silently.
EXPECTED_REPR_CLASSES = frozenset(
    {
        "dataset_recorder::DatasetRecorder",
        "hardware_rtps_bridge::HardwareRtpsBridge",
        "mesh/core::Mesh",
        "mesh/input::InputPublisher",
        "mesh/input::InputReceiver",
        "mesh/ros_bridge::RosBridgedRobot",
        "mesh/rosbridge_robot::RosbridgeRobot",
        "mesh/rtps_robot::RtpsRobot",
        "mesh/session::PeerInfo",
        "policies/lerobot_local/processor::ProcessorBridge",
        "simulation/isaac/simulation::IsaacSimulation",
    }
)


def _package_root() -> pathlib.Path:
    """Locate the installed package from the imported module, never a path literal."""
    return pathlib.Path(inspect.getfile(strands_robots)).parent


def _classes_defining_repr() -> dict[str, type]:
    """Discover, and import, every class in the package that defines ``__repr__``."""
    root = _package_root()
    found: dict[str, type] = {}
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "def __repr__" not in source:
            continue
        rel = path.relative_to(root).with_suffix("")
        dotted = "strands_robots." + str(rel).replace("/", ".")
        dotted = dotted.removesuffix(".__init__")
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ClassDef):
                continue
            if not any(isinstance(f, ast.FunctionDef) and f.name == "__repr__" for f in node.body):
                continue
            key = f"{str(rel).removesuffix('/__init__')}::{node.name}"
            found[key] = getattr(importlib.import_module(dotted), node.name)
    return found


def _renders_a_half_built_instance(cls: type) -> str | None:
    """Return the failure for ``repr`` on a half-built ``cls``, or ``None`` if it survives.

    ``__new__`` is called directly, and the class bound through ``Any`` to say so:
    skipping ``__init__`` is the point, because it produces the worst case of the
    state a refusal leaves behind - an instance on which no attribute exists yet.

    The handler catches ``Exception`` and stops there. A failure inside ``repr``
    is the answer being collected - it is the defect this module pins - so it has
    to become a verdict rather than abort the survey. An interrupt is not an
    answer about a ``repr``, though: swallowing ``BaseException`` would record an
    operator's Ctrl-C as "this class cannot render a half-built instance", fail
    the assertion naming the wrong cause, and carry on to the next class.
    ``pytest``'s own ``skip`` and ``fail`` outcomes derive from ``BaseException``
    for the same reason, so they have to reach the runner instead of becoming a
    verdict about the class under survey.
    """
    factory: Any = cls
    obj = factory.__new__(factory)
    try:
        rendered = repr(obj)
    except Exception as exc:  # noqa: BLE001 - a repr failure is a verdict, control flow is not
        return f"{type(exc).__name__}: {exc}"
    if cls.__name__ not in rendered:
        return f"does not identify its type: {rendered!r}"
    return None


def _raising_repr(exc: BaseException) -> type:
    """A class whose ``__repr__`` raises, so the classifier's handler width is observable."""

    class _Raising:
        def __repr__(self) -> str:
            raise exc

    return _Raising


DISCOVERED = _classes_defining_repr()


def _live_self(exc: BaseException, cls: type) -> Any:
    """Return the half-built ``cls`` the raising frame still holds, as a reader sees it."""
    tb, found = exc.__traceback__, None
    while tb is not None:
        candidate = tb.tb_frame.f_locals.get("self")
        if type(candidate) is cls:
            found = candidate
        tb = tb.tb_next
    return found


class _Mesh:
    """The members the input streams touch before they validate their arguments."""

    peer_id = "arm"
    alive = True

    def subscribe(self, *args: Any, **kwargs: Any) -> None:
        return None

    def unsubscribe(self, *args: Any, **kwargs: Any) -> None:
        return None


def _fake_mesh() -> Any:
    """Return the structural mesh stand-in the input streams accept.

    Annotated ``Any`` because a real :class:`strands_robots.mesh.core.Mesh` would
    need a live session, and neither stream reads more than the members above
    before the validation this module drives.
    """
    return _Mesh()


class _Robot:
    tool_name_str = "arm"


class _Dataset:
    repo_id = "user/dataset"
    root = pathlib.Path("/tmp")


#: ``(class, the refusal it raises, a call that triggers it, the attribute the
#: old repr reported)``. Every refusal here is the class's own validation of the
#: caller's arguments - the reward for which used to be a traceback naming the
#: attribute in the fourth column instead of the value in the message.
REFUSALS: list[tuple[type, type[Exception], Any, str]] = [
    (
        RosBridgedRobot,
        ValueError,
        lambda: RosBridgedRobot(node_name="bad name!", cmd_vel_topic="/cmd_vel", odom_topic="/odom"),
        "node_name",
    ),
    (
        RosbridgeRobot,
        ValueError,
        lambda: RosbridgeRobot(
            node_name="bad name!",
            cmd_vel_topic="/cmd_vel",
            odom_topic="/odom",
            host="127.0.0.1",
            port=9090,
        ),
        "node_name",
    ),
    (RtpsRobot, ValueError, lambda: RtpsRobot(node_name="bad name!", cmd_vel_topic="/cmd_vel"), "node_name"),
    (HardwareRtpsBridge, ValueError, lambda: HardwareRtpsBridge(None, domain_id=99999), "_robot_name"),
    (
        InputPublisher,
        ValueError,
        lambda: InputPublisher(_fake_mesh(), object(), device_name="leader", hz=0),
        "_running",
    ),
    (InputReceiver, ValidationError, lambda: InputReceiver(_fake_mesh(), object(), source_peer_id="**"), "_running"),
]

#: ``(label, a call that succeeds, a field the repr must name)``.
#: ``HardwareRtpsBridge`` is absent because a working instance needs cyclonedds;
#: its refusal path above needs none, which is the half this module is about.
BUILDABLE: list[tuple[str, Any, str]] = [
    (
        "RosBridgedRobot",
        lambda: RosBridgedRobot(node_name="/turtle1", cmd_vel_topic="/cmd_vel", odom_topic="/odom"),
        "/turtle1",
    ),
    (
        "RosbridgeRobot",
        lambda: RosbridgeRobot(
            node_name="/turtle1", cmd_vel_topic="/cmd_vel", odom_topic="/odom", host="127.0.0.1", port=9090
        ),
        "9090",
    ),
    ("RtpsRobot", lambda: RtpsRobot(node_name="/arm", cmd_vel_topic="/cmd_vel"), "/arm"),
    ("Mesh", lambda: Mesh(_Robot(), peer_id="arm", peer_type="robot"), "arm"),
    ("PeerInfo", lambda: PeerInfo(peer_id="arm", peer_type="robot", last_seen_mono=0.0), "arm"),
    ("DatasetRecorder", lambda: DatasetRecorder(dataset=_Dataset()), "user/dataset"),
    ("InputPublisher", lambda: InputPublisher(_fake_mesh(), object(), device_name="leader", hz=50.0), "leader"),
    (
        "InputReceiver",
        lambda: InputReceiver(_fake_mesh(), object(), source_peer_id="peer", device_name="leader"),
        "peer",
    ),
    ("ProcessorBridge", lambda: ProcessorBridge(None, None), "pre=None"),
]


class TestEveryReprSurvivesAHalfBuiltInstance:
    """The contract, over every class in the package that defines ``__repr__``."""

    @pytest.mark.parametrize("key", sorted(DISCOVERED))
    def test_a_half_built_instance_renders(self, key: str) -> None:
        problem = _renders_a_half_built_instance(DISCOVERED[key])
        assert problem is None, (
            f"{key}: repr on an instance whose __init__ did not finish {problem}. "
            "Wrap the body in try/except AttributeError and return "
            "strands_robots.utils.partial_construction_repr(self)."
        )

    def test_the_survey_covers_the_classes_this_module_names(self) -> None:
        assert set(DISCOVERED) == set(EXPECTED_REPR_CLASSES)

    def test_the_survey_detects_a_planted_defect(self) -> None:
        class _Raises:
            def __repr__(self) -> str:
                return f"_Raises({self.missing})"  # type: ignore[attr-defined]

        assert "AttributeError" in (_renders_a_half_built_instance(_Raises) or "")

    def test_the_survey_detects_a_repr_that_hides_its_type(self) -> None:
        class _Anonymous:
            def __repr__(self) -> str:
                return "<something>"

        problem = _renders_a_half_built_instance(_Anonymous)
        assert problem is not None and "does not identify its type" in problem


class TestARefusedConstructorStaysDiagnosable:
    """A validation refusal must not be hidden by the repr of what it refused."""

    @pytest.mark.parametrize(
        ("cls", "refusal", "make", "attribute"),
        REFUSALS,
        ids=[cls.__name__ for cls, _, _, _ in REFUSALS],
    )
    def test_the_half_built_instance_reports_the_lifecycle_fact(
        self, cls: type, refusal: type[Exception], make: Any, attribute: str
    ) -> None:
        with pytest.raises(refusal) as excinfo:
            make()
        half_built = _live_self(excinfo.value, cls)
        assert half_built is not None, "the raising frame no longer holds the instance"
        rendered = repr(half_built)
        assert FALLBACK_PHRASE in rendered
        assert cls.__name__ in rendered
        assert attribute not in rendered, (
            f"the repr names {attribute!r}, which is what sent readers chasing an "
            "attribute rather than reading the refusal"
        )

    @pytest.mark.parametrize(
        ("cls", "refusal", "make", "attribute"),
        REFUSALS,
        ids=[cls.__name__ for cls, _, _, _ in REFUSALS],
    )
    def test_the_refusal_itself_still_describes_the_argument(
        self, cls: type, refusal: type[Exception], make: Any, attribute: str
    ) -> None:
        with pytest.raises(refusal) as excinfo:
            make()
        message = str(excinfo.value)
        assert message, "the refusal must say something"
        assert FALLBACK_PHRASE not in message, "the refusal is about the argument, not the lifecycle"


class TestAFullyBuiltInstanceIsUnaffected:
    """The tolerance must not swallow a working repr - the over-reach control."""

    @pytest.mark.parametrize(("label", "make", "field"), BUILDABLE, ids=[b[0] for b in BUILDABLE])
    def test_it_still_names_its_fields(self, label: str, make: Any, field: str) -> None:
        rendered = repr(make())
        assert field in rendered
        assert FALLBACK_PHRASE not in rendered


class TestTheFallbackWordingHasOneOwner:
    """One rule, one wording - so a traceback reads the same in every layer."""

    def test_every_tolerant_repr_delegates_to_the_shared_helper(self) -> None:
        root = _package_root()
        adrift: list[str] = []
        tolerant = 0
        for path in sorted(root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            if "def __repr__" not in source:
                continue
            for node in ast.walk(ast.parse(source)):
                if not isinstance(node, ast.ClassDef):
                    continue
                for fn in node.body:
                    if not (isinstance(fn, ast.FunctionDef) and fn.name == "__repr__"):
                        continue
                    handlers = [h for t in ast.walk(fn) if isinstance(t, ast.Try) for h in t.handlers]
                    if not handlers:
                        continue
                    tolerant += 1
                    body = ast.unparse(fn)
                    if "partial_construction_repr(self)" not in body:
                        adrift.append(f"{path.relative_to(root)}::{node.name}")
        assert tolerant == len(EXPECTED_REPR_CLASSES), tolerant
        assert adrift == [], f"these reprs re-implement the fallback: {adrift}"

    def test_no_repr_spells_the_phrase_itself(self) -> None:
        root = _package_root()
        offenders: list[str] = []
        for path in sorted(root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            if "def __repr__" not in source:
                continue
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.FunctionDef) and node.name == "__repr__":
                    if FALLBACK_PHRASE in ast.unparse(node):
                        offenders.append(f"{path.relative_to(root)}:{node.lineno}")
        assert offenders == [], f"the phrase must come from the shared helper: {offenders}"

    def test_the_helper_names_no_attribute_and_identifies_the_type(self) -> None:
        class _Example:
            pass

        rendered = partial_construction_repr(_Example())
        assert rendered.startswith("_Example(")
        assert FALLBACK_PHRASE in rendered
        assert "id=0x" in rendered

    def test_the_helper_is_the_only_definition_of_the_wording(self) -> None:
        source = textwrap.dedent(inspect.getsource(partial_construction_repr))
        assert FALLBACK_PHRASE in source


class TestTheReprClassifierCollectsFailuresWithoutSwallowingControlFlow:
    """``_renders_a_half_built_instance`` must catch a repr failure and only that.

    The classifier turns "what did this class's ``repr`` do with a half-built
    instance" into a verdict string so the survey above can compare every class
    in the package. A raise is one of the answers it has to collect - the
    escaping ``AttributeError`` is the defect this file pins - so a failure
    inside ``repr`` cannot be allowed to abort the survey. An interrupt is not
    an answer about a ``repr``, though: recording one as "this class cannot
    render a half-built instance" would fail the assertion naming the wrong
    cause and carry on to the next class. Both halves are pinned here because
    the correct handler is the one that satisfies both.
    """

    @pytest.mark.parametrize(
        "exc",
        [
            AttributeError("'RosBridgedRobot' object has no attribute 'node_name'"),
            TypeError("unsupported format string passed to NoneType.__format__"),
            RuntimeError("Rendering unavailable (no OpenGL context)"),
        ],
        ids=["AttributeError", "TypeError", "RuntimeError"],
    )
    def test_a_repr_failure_is_collected_as_a_verdict(self, exc: Exception) -> None:
        """The escape this file exists to pin must reach the survey, not the runner."""
        problem = _renders_a_half_built_instance(_raising_repr(exc))
        assert problem == f"{type(exc).__name__}: {exc}"

    @pytest.mark.parametrize(
        "exc",
        [
            KeyboardInterrupt(),
            SystemExit(1),
            pytest.skip.Exception("the optional dependency this repr reads is absent"),
            pytest.fail.Exception("the fixture for this class is incomplete"),
        ],
        ids=["KeyboardInterrupt", "SystemExit", "pytest.skip", "pytest.fail"],
    )
    def test_control_flow_reaches_the_runner_instead_of_becoming_a_verdict(self, exc: BaseException) -> None:
        # Executable premise: each of these derives from BaseException without
        # deriving from Exception, which is the whole reason the handler width
        # is observable at all.
        assert not isinstance(exc, Exception), f"{type(exc).__name__} no longer tests the handler width"
        with pytest.raises(type(exc)):
            _renders_a_half_built_instance(_raising_repr(exc))
