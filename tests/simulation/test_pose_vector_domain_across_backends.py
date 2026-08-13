"""Regression tests: every backend validates a scene-placement pose vector.

``coerce_pose_vector`` documents the invariant these tests pin - "membership, not
truthiness" - and the MuJoCo backend has routed ``add_object`` / ``add_robot`` /
``move_object`` / ``add_camera`` and ``move_to`` through it since the pose
parameters were hardened there. The Newton and Isaac backends applied it to
``add_camera`` only, so their remaining placement methods read the caller's
vector directly. Measured on the same 9-value probe set, one ``add_object``
per case, with no ``newton`` / ``isaacsim`` installed:

* Newton read the vector for TRUTHINESS (``position or [0.0, 0.0, 0.0]``), which
  is wrong twice over. ``np.array([0.4, 0.2, 0.3])`` - what pose arithmetic
  produces, and what the Args advertise - raised a bare
  ``ValueError: truth value of an array with more than one element is
  ambiguous`` straight through the structured envelope these methods document as
  their only failure channel. And ``[]`` is falsy, so it read as *omitted*: the
  object was placed at the default origin and the call reported success.
* Everything else was stored verbatim on the registry entry and handed to the
  solver rebuild: ``[1.0, 2.0]`` became a 2-component position, ``[nan, 0, 0]``
  and ``[inf, 0, 0]`` a non-finite one, ``[True, 0, 0]`` read ``True`` as the
  coordinate 1.0, and the bare string ``"abc"`` was stored AS the position.
  A 3-component ``orientation`` was accepted as a quaternion.
* Isaac coerced with ``list(position)``, which validated nothing either - the
  same wrong-length / non-finite / ``bool`` / string values reached the prim
  constructor, and ``"abc"`` was split per character into a 3-component
  "position". Its ``move_object`` / ``set_robot_pose`` additionally SLICED with
  ``position[:3]`` / ``orientation[:4]``, so a 5-component request was written as
  its first 3 under a success result, and ``move_object`` tested the vector for
  truthiness on top of that.

MuJoCo refused all 16 unusable values in that set and accepted all 3 NumPy
vectors. Every message here is the shared one, so a pose one backend refuses is
refused by all of them.

``TestNoPlacementMethodDrifts`` keeps it that way structurally: every public
method of a backend engine class that takes a ``position`` / ``orientation`` /
``target`` parameter must route it through the shared helper. The scope is
class methods deliberately - ``scene_ops.reposition_body_in_scene`` is a
module-level spec helper reached only from the already-validated
``move_object``, and it returns ``bool`` rather than the agent-tool envelope.

What is NOT in scope, and is asserted to be unchanged: the shape-dependent half
of ``size`` - the per-shape component count, the short-vector fallback the Isaac
``add_object`` docstring promises and neither other backend offers, and the
positivity of a consumed extent. Those need one contract decision rather than a
helper default, and #1858 tracks them. ``color``, ``mass`` and the
shape-independent half of ``size`` were all in that list until their domains were
settled - on the shared ``coerce_rgba``, ``SimEngine._validate_mass`` and
``coerce_size_vector`` respectively - and are now pinned in
``tests/simulation/test_color_domain_across_backends.py``,
``tests/simulation/test_object_mass_domain_across_backends.py`` and
``tests/simulation/test_object_size_domain_across_backends.py``.

These tests are GL-free and need neither ``newton``/``warp`` nor ``isaacsim``
nor a GPU: every guard runs before its method touches a solver or a stage, so
calling the unbound method with a small stand-in for ``self`` exercises it in
every environment (the pattern
``tests/simulation/test_entity_name_domain_at_creation.py`` uses).
"""

from __future__ import annotations

import ast
import inspect
import math
import pathlib
import sys
import threading
import types
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.isaac.simulation import IsaacConfig, IsaacSimulation
from strands_robots.simulation.models import SimObject, SimWorld
from strands_robots.simulation.newton.simulation import NewtonSimEngine
from strands_robots.utils import coerce_pose_vector

NAN = float("nan")
INF = float("inf")

#: Vectors no placement call can honor. Each is a 3-component probe: an empty
#: vector (the one a truthiness read swallowed as *omitted*), two wrong lengths,
#: the two non-finite components, a ``bool`` (an ``int`` subclass, so
#: ``float(True)`` would silently mean the coordinate 1.0), a bare string (an
#: iterable of 1-character strings), and a non-iterable scalar.
UNUSABLE_POSITIONS: tuple[Any, ...] = (
    [],
    [1.0, 2.0],
    [0.1, 0.2, 0.3, 0.4],
    [NAN, 0.0, 0.0],
    [INF, 0.0, 0.0],
    [True, 0.0, 0.0],
    "abc",
    0.38,
    [None, 0.0, 0.0],
)

#: Quaternions no placement call can honor - same classes, at length 4.
UNUSABLE_ORIENTATIONS: tuple[Any, ...] = (
    [],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [NAN, 0.0, 0.0, 0.0],
    [INF, 0.0, 0.0, 0.0],
    [True, 0.0, 0.0, 0.0],
    "abcd",
    1.0,
)

#: Accepted spellings of a usable pose. The NumPy forms are the point: pose
#: arithmetic produces them and every Args block advertises them.
GOOD_POSITIONS: tuple[Any, ...] = (
    [0.4, 0.2, 0.3],
    (0.4, 0.2, 0.3),
    np.array([0.4, 0.2, 0.3]),
    np.array([0.4, 0.2, 0.3], dtype=np.float32),
    [np.float64(0.4), 0.2, 0.3],
)

GOOD_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0])


# --------------------------------------------------------------------------- #
# The shared domain                                                           #
# --------------------------------------------------------------------------- #
class TestTheSharedDomain:
    """``coerce_pose_vector`` is the single definition every call site shares."""

    @pytest.mark.parametrize("vec", UNUSABLE_POSITIONS)
    def test_unusable_position_is_refused(self, vec):
        value, err = coerce_pose_vector("add_object", "position", vec, 3)
        assert err is not None, vec
        assert value is None
        assert err.startswith("add_object: 'position'")

    @pytest.mark.parametrize("vec", UNUSABLE_ORIENTATIONS)
    def test_unusable_orientation_is_refused(self, vec):
        _value, err = coerce_pose_vector("add_object", "orientation", vec, 4)
        assert err is not None, vec

    @pytest.mark.parametrize("vec", GOOD_POSITIONS)
    def test_a_usable_position_normalizes_to_plain_floats(self, vec):
        """Accepted, and the NumPy scalars do not outlive the boundary.

        The pose is stored on :class:`SimObject` / :class:`SimRobot` (both
        annotated ``list[float]``) and echoed in the agent-visible status text,
        so a surviving ``np.float64`` would leak ``np.float64(0.4)`` into it.
        """
        value, err = coerce_pose_vector("add_object", "position", vec, 3)
        assert err is None, vec
        assert value == [0.4, 0.2, 0.3] or value == pytest.approx([0.4, 0.2, 0.3])
        assert all(type(v) is float for v in value)

    def test_omitted_is_not_refused(self):
        """``None`` means omitted - the caller applies its own documented default."""
        assert coerce_pose_vector("add_object", "position", None, 3) == (None, None)


# --------------------------------------------------------------------------- #
# Newton stand-ins (no newton / warp / GPU: the guards precede the solver)     #
# --------------------------------------------------------------------------- #
def _newton_stub() -> Any:
    stub = types.SimpleNamespace(
        _world=SimWorld(),
        _model=types.SimpleNamespace(body_label=["ground"]),
        _lock=threading.RLock(),
        _robot_joint_map={},
        _rebuild=lambda: None,
        # Inherited from SimEngine, so a real engine always has it. add_object
        # routes its ``mass`` through it, and a stand-in that omitted it would
        # make that guard look absent rather than unexercised.
        _validate_mass=SimEngine._validate_mass,
    )
    return stub


def _newton_stub_with_object() -> Any:
    stub = _newton_stub()
    stub._world.objects["crate"] = SimObject(
        name="crate", shape="box", position=[0.0, 0.0, 0.0], orientation=[1.0, 0.0, 0.0, 0.0]
    )
    return stub


class TestNewtonAddObject:
    @pytest.mark.parametrize("vec", UNUSABLE_POSITIONS)
    def test_unusable_position_is_refused(self, vec):
        stub = _newton_stub()
        result = NewtonSimEngine.add_object(stub, "crate", position=vec)
        assert result["status"] == "error", (vec, result)
        assert "'position'" in result["content"][0]["text"]

    @pytest.mark.parametrize("vec", UNUSABLE_ORIENTATIONS)
    def test_unusable_orientation_is_refused(self, vec):
        stub = _newton_stub()
        result = NewtonSimEngine.add_object(stub, "crate", orientation=vec)
        assert result["status"] == "error", (vec, result)
        assert "'orientation'" in result["content"][0]["text"]

    @pytest.mark.parametrize("vec", UNUSABLE_POSITIONS)
    def test_a_refused_pose_registers_no_object(self, vec):
        """No half-placed object: the registry is untouched.

        Pre-fix ``[]`` reported success and registered the crate at the default
        origin, and ``[nan, 0, 0]`` registered it with a non-finite position.
        """
        stub = _newton_stub()
        NewtonSimEngine.add_object(stub, "crate", position=vec)
        assert dict(stub._world.objects) == {}

    def test_a_numpy_pose_is_accepted_and_normalized(self):
        """Pre-fix this raised ``ValueError: truth value of an array ...``."""
        stub = _newton_stub()
        result = NewtonSimEngine.add_object(
            stub, "crate", position=np.array([0.4, 0.2, 0.3]), orientation=GOOD_ORIENTATION
        )
        assert result["status"] == "success", result
        stored = stub._world.objects["crate"]
        assert stored.position == [0.4, 0.2, 0.3]
        assert all(type(v) is float for v in stored.position)
        assert all(type(v) is float for v in stored.orientation)

    def test_an_omitted_pose_still_takes_the_documented_default(self):
        stub = _newton_stub()
        assert NewtonSimEngine.add_object(stub, "crate")["status"] == "success"
        stored = stub._world.objects["crate"]
        assert stored.position == [0.0, 0.0, 0.0]
        assert stored.orientation == [1.0, 0.0, 0.0, 0.0]


class TestNewtonAddRobot:
    @pytest.mark.parametrize("vec", UNUSABLE_POSITIONS)
    def test_unusable_position_is_refused(self, vec, tmp_path):
        stub = _newton_stub()
        result = NewtonSimEngine.add_robot(stub, "arm", urdf_path=str(tmp_path / "arm.urdf"), position=vec)
        assert result["status"] == "error", (vec, result)
        assert "'position'" in result["content"][0]["text"]
        assert dict(stub._world.robots) == {}

    @pytest.mark.parametrize("vec", UNUSABLE_ORIENTATIONS)
    def test_unusable_orientation_is_refused(self, vec, tmp_path):
        stub = _newton_stub()
        result = NewtonSimEngine.add_robot(stub, "arm", urdf_path=str(tmp_path / "arm.urdf"), orientation=vec)
        assert result["status"] == "error", (vec, result)
        assert "'orientation'" in result["content"][0]["text"]

    def test_the_refusal_precedes_the_asset_resolution(self):
        """A bad pose is reported as such, not as a missing asset.

        With no ``urdf_path`` the method resolves an asset from the registry,
        which is the expensive step and the one that would otherwise report
        first. Pinning the order keeps the caller's actual mistake in the message.
        """
        stub = _newton_stub()
        result = NewtonSimEngine.add_robot(stub, "definitely_not_a_robot", position=[NAN, 0.0, 0.0])
        assert result["status"] == "error"
        assert "'position'" in result["content"][0]["text"]


class TestNewtonMoveObject:
    @pytest.mark.parametrize("vec", UNUSABLE_POSITIONS)
    def test_unusable_position_is_refused(self, vec):
        stub = _newton_stub_with_object()
        result = NewtonSimEngine.move_object(stub, "crate", position=vec)
        assert result["status"] == "error", (vec, result)
        assert "'position'" in result["content"][0]["text"]

    @pytest.mark.parametrize("vec", UNUSABLE_POSITIONS)
    def test_a_refused_move_leaves_the_object_where_it_was(self, vec):
        stub = _newton_stub_with_object()
        NewtonSimEngine.move_object(stub, "crate", position=vec)
        assert stub._world.objects["crate"].position == [0.0, 0.0, 0.0]

    def test_a_numpy_move_is_accepted_and_normalized(self):
        stub = _newton_stub_with_object()
        result = NewtonSimEngine.move_object(stub, "crate", position=np.array([1.0, 2.0, 3.0]))
        assert result["status"] == "success", result
        assert stub._world.objects["crate"].position == [1.0, 2.0, 3.0]

    def test_the_status_text_names_the_pose_without_testing_it(self):
        """The message itself used ``position or 'same'`` - the same defect.

        A NumPy position reached that f-string even when the write succeeded, so
        the ambiguous-truth ``ValueError`` escaped from the *reporting* step.
        """
        stub = _newton_stub_with_object()
        result = NewtonSimEngine.move_object(stub, "crate", position=np.array([1.0, 2.0, 3.0]))
        assert "1.0, 2.0, 3.0" in result["content"][0]["text"]

    def test_an_omitted_pose_still_means_leave_it(self):
        stub = _newton_stub_with_object()
        result = NewtonSimEngine.move_object(stub, "crate")
        assert result["status"] == "success"
        assert "same" in result["content"][0]["text"]
        assert stub._world.objects["crate"].position == [0.0, 0.0, 0.0]


# --------------------------------------------------------------------------- #
# Isaac stand-ins (no isaacsim / omni / GPU needed)                            #
# --------------------------------------------------------------------------- #
class _FakeHandle:
    """Records the pose actually written, so a truncation is observable."""

    def __init__(self) -> None:
        self.poses: list[tuple[Any, Any]] = []

    def set_world_pose(self, position=None, orientation=None):
        self.poses.append((position, orientation))


def _isaac_stub() -> Any:
    return types.SimpleNamespace(
        _lock=threading.RLock(),
        _world_created=True,
        _config=IsaacConfig(),
        _objects={},
        _robots={},
        _cameras={},
        _replicated=False,
        _prim_registry=[],
        _world=types.SimpleNamespace(scene=types.SimpleNamespace(add=lambda handle: None)),
        _construct_shape_prim=lambda **kwargs: (object(), kwargs.get("size")),
        _validate_mass=SimEngine._validate_mass,
    )


@pytest.fixture
def isaac_usd_modules(monkeypatch):
    """Stand in for the ``omni``/``pxr`` modules a real Isaac install provides.

    After the pose is written, ``move_object`` also mirrors it onto the prim's USD
    xform ops so a render-only tick cannot show a stale transform. That block is
    best-effort and needs ``omni.usd`` + ``pxr``, which only ship with Isaac Sim -
    so the accepted path is exercised with a stand-in whose stage reports no prim,
    which is the branch a real install takes for an unmapped path.
    """
    stage = types.SimpleNamespace(GetPrimAtPath=lambda path: None)
    omni = types.ModuleType("omni")
    omni_usd = types.ModuleType("omni.usd")
    omni_usd.get_context = lambda: types.SimpleNamespace(get_stage=lambda: stage)  # type: ignore[attr-defined]
    omni.usd = omni_usd  # type: ignore[attr-defined]
    pxr = types.ModuleType("pxr")
    pxr.Gf = types.SimpleNamespace(Vec3d=tuple, Quatf=tuple)  # type: ignore[attr-defined]
    pxr.UsdGeom = types.SimpleNamespace()  # type: ignore[attr-defined]
    for name, module in (("omni", omni), ("omni.usd", omni_usd), ("pxr", pxr)):
        monkeypatch.setitem(sys.modules, name, module)


def _isaac_stub_with_object() -> Any:
    stub = _isaac_stub()
    handle = _FakeHandle()
    stub._objects["crate"] = types.SimpleNamespace(
        name="crate", prim_path="/World/Objects/crate", handle=handle, shape="box", is_static=False
    )
    return stub


def _isaac_stub_with_robot() -> Any:
    stub = _isaac_stub()
    handle = _FakeHandle()
    stub._robots["arm"] = types.SimpleNamespace(
        name="arm", prim_path="/World/Robots/arm", articulation=handle, joint_names=["j"]
    )
    return stub


class TestIsaacAddObject:
    @pytest.mark.parametrize("vec", UNUSABLE_POSITIONS)
    def test_unusable_position_is_refused(self, vec):
        stub = _isaac_stub()
        result = IsaacSimulation.add_object(stub, "crate", position=vec)
        assert result["status"] == "error", (vec, result)
        assert "'position'" in result["content"][0]["text"]
        assert stub._objects == {}
        assert stub._prim_registry == []

    @pytest.mark.parametrize("vec", UNUSABLE_ORIENTATIONS)
    def test_unusable_orientation_is_refused(self, vec):
        stub = _isaac_stub()
        result = IsaacSimulation.add_object(stub, "crate", orientation=vec)
        assert result["status"] == "error", (vec, result)
        assert "'orientation'" in result["content"][0]["text"]

    def test_a_string_position_is_no_longer_read_per_character(self):
        """``list("abc")`` produced the 3-"component" position ``['a','b','c']``.

        This asserted ``"must be numbers"``, which was the ELEMENT read's verdict
        and so held only for a string whose length happened to equal the component
        count: ``"abc"`` reached that read, while ``"cube"`` was refused one gate
        earlier as a wrong-length *vector*. A string is now refused on its type
        before its length is consulted, so the property this test is named for -
        the characters are never read as components - is checked on the property
        itself rather than on whichever downstream guard the length routed it to.
        """
        stub = _isaac_stub()
        for text in ("abc", "cube", "0.1,0.2,0.3"):
            result = IsaacSimulation.add_object(stub, "crate", position=text)
            assert result["status"] == "error", (text, result)
            message = result["content"][0]["text"]
            assert "'position'" in message
            assert "str" in message
            # No character of the string may appear as a component, and no count of
            # them as a component count.
            assert "-element vector" not in message
            assert "must be numbers" not in message
            assert stub._objects == {}

    def test_a_numpy_pose_is_accepted_and_echoed_as_plain_floats(self):
        stub = _isaac_stub()
        result = IsaacSimulation.add_object(stub, "crate", position=np.array([0.4, 0.2, 0.3]))
        assert result["status"] == "success", result
        echoed = result["content"][0]["json"]["position"]
        assert echoed == [0.4, 0.2, 0.3]
        assert all(type(v) is float for v in echoed)

    def test_an_omitted_pose_still_takes_the_documented_default(self):
        stub = _isaac_stub()
        result = IsaacSimulation.add_object(stub, "crate")
        assert result["status"] == "success", result
        assert result["content"][0]["json"]["position"] == [0.0, 0.0, 0.5]


class TestIsaacAddRobot:
    @pytest.mark.parametrize("vec", UNUSABLE_POSITIONS)
    def test_unusable_position_is_refused(self, vec):
        stub = _isaac_stub()
        result = IsaacSimulation.add_robot(stub, "arm", data_config="panda", position=vec)
        assert result["status"] == "error", (vec, result)
        assert "'position'" in result["content"][0]["text"]
        assert stub._robots == {}
        assert stub._prim_registry == []

    @pytest.mark.parametrize("vec", (UNUSABLE_ORIENTATIONS[0], UNUSABLE_ORIENTATIONS[1], "abcd"))
    def test_a_wrong_length_quaternion_reports_its_real_defect(self, vec):
        """Not "orientation is not applied on the Isaac backend".

        That reject exists for a well-formed NON-IDENTITY quaternion, which the
        spawn path genuinely ignores. A wrong-length or non-numeric value is a
        different mistake, so the pose domain has to be checked first or the
        caller is told to omit a parameter they mis-spelled instead.
        """
        stub = _isaac_stub()
        result = IsaacSimulation.add_robot(stub, "arm", data_config="panda", orientation=vec)
        assert result["status"] == "error", (vec, result)
        assert "'orientation'" in result["content"][0]["text"]
        assert "not applied on the Isaac" not in result["content"][0]["text"]

    def test_the_identity_only_reject_still_reports_its_own_reason(self):
        """A well-formed non-identity quaternion keeps the unsupported-feature error."""
        stub = _isaac_stub()
        result = IsaacSimulation.add_robot(stub, "arm", data_config="panda", orientation=[0.0, 1.0, 0.0, 0.0])
        assert result["status"] == "error"
        assert "not applied on the Isaac" in result["content"][0]["text"]

    def test_a_numpy_position_and_identity_quaternion_still_spawn(self):
        stub = _isaac_stub()
        result = IsaacSimulation.add_robot(
            stub, "arm", data_config="panda", position=np.array([0.4, 0.2, 0.0]), orientation=GOOD_ORIENTATION
        )
        assert result["status"] == "success", result
        assert list(stub._robots) == ["arm"]


class TestIsaacMoveObject:
    @pytest.mark.parametrize("vec", UNUSABLE_POSITIONS)
    def test_unusable_position_is_refused(self, vec):
        stub = _isaac_stub_with_object()
        result = IsaacSimulation.move_object(stub, "crate", position=vec)
        assert result["status"] == "error", (vec, result)
        assert "'position'" in result["content"][0]["text"]

    @pytest.mark.parametrize("vec", UNUSABLE_POSITIONS)
    def test_a_refused_move_writes_no_pose(self, vec):
        """The refusal precedes the transform write, so nothing half-moves."""
        stub = _isaac_stub_with_object()
        IsaacSimulation.move_object(stub, "crate", position=vec)
        assert stub._objects["crate"].handle.poses == []

    def test_an_over_long_position_is_refused_not_truncated(self):
        """``np.array(position[:3])`` silently wrote the first 3 of 5.

        A caller who passed a 5-component vector asked for something this API
        cannot express; answering with a 3-component move under a success result
        reports a pose they never requested.
        """
        stub = _isaac_stub_with_object()
        result = IsaacSimulation.move_object(stub, "crate", position=[1.0, 2.0, 3.0, 4.0, 5.0])
        assert result["status"] == "error"
        assert "3-element" in result["content"][0]["text"]
        assert stub._objects["crate"].handle.poses == []

    def test_a_numpy_move_is_accepted(self, isaac_usd_modules):
        stub = _isaac_stub_with_object()
        result = IsaacSimulation.move_object(stub, "crate", position=np.array([1.0, 2.0, 3.0]))
        assert result["status"] == "success", result
        written, _ori = stub._objects["crate"].handle.poses[0]
        assert list(written) == [1.0, 2.0, 3.0]

    def test_an_omitted_pose_still_means_leave_it(self, isaac_usd_modules):
        stub = _isaac_stub_with_object()
        result = IsaacSimulation.move_object(stub, "crate")
        assert result["status"] == "success", result
        assert "same" in result["content"][0]["text"]


class TestIsaacSetRobotPose:
    @pytest.mark.parametrize("vec", UNUSABLE_POSITIONS)
    def test_unusable_position_is_refused(self, vec):
        stub = _isaac_stub_with_robot()
        result = IsaacSimulation.set_robot_pose(stub, "arm", position=vec)
        assert result["status"] == "error", (vec, result)
        assert "'position'" in result["content"][0]["text"]
        assert stub._robots["arm"].articulation.poses == []

    @pytest.mark.parametrize("vec", UNUSABLE_ORIENTATIONS)
    def test_unusable_orientation_is_refused(self, vec):
        stub = _isaac_stub_with_robot()
        result = IsaacSimulation.set_robot_pose(stub, "arm", orientation=vec)
        assert result["status"] == "error", (vec, result)
        assert "'orientation'" in result["content"][0]["text"]

    def test_an_over_long_pose_is_refused_not_truncated(self):
        stub = _isaac_stub_with_robot()
        result = IsaacSimulation.set_robot_pose(stub, "arm", position=[1.0, 2.0, 3.0, 4.0])
        assert result["status"] == "error"
        assert stub._robots["arm"].articulation.poses == []

    def test_a_numpy_pose_is_accepted(self):
        stub = _isaac_stub_with_robot()
        result = IsaacSimulation.set_robot_pose(
            stub, "arm", position=np.array([1.0, 2.0, 3.0]), orientation=GOOD_ORIENTATION
        )
        assert result["status"] == "success", result
        written, ori = stub._robots["arm"].articulation.poses[0]
        assert list(written) == [1.0, 2.0, 3.0]
        assert list(ori) == [1.0, 0.0, 0.0, 0.0]


# --------------------------------------------------------------------------- #
# Cross-backend parity                                                        #
# --------------------------------------------------------------------------- #
class TestEveryBackendGivesTheSameVerdict:
    """A pose one backend refuses is refused by all of them, with one message."""

    @pytest.fixture
    def mj_sim(self):
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import Simulation

        sim = Simulation(tool_name="test_pose_domain_parity_sim", mesh=False)
        assert sim.create_world()["status"] == "success"
        yield sim
        sim.cleanup()

    @pytest.mark.parametrize("vec", UNUSABLE_POSITIONS)
    def test_add_object_position_verdicts_match(self, mj_sim, vec):
        mj = mj_sim.add_object("crate", position=vec)
        nt = NewtonSimEngine.add_object(_newton_stub(), "crate", position=vec)
        ic = IsaacSimulation.add_object(_isaac_stub(), "crate", position=vec)
        assert mj["status"] == nt["status"] == ic["status"] == "error", (vec, mj, nt, ic)
        texts = {mj["content"][0]["text"], nt["content"][0]["text"], ic["content"][0]["text"]}
        assert len(texts) == 1, texts

    @pytest.mark.parametrize("vec", UNUSABLE_ORIENTATIONS)
    def test_add_object_orientation_verdicts_match(self, mj_sim, vec):
        mj = mj_sim.add_object("crate", orientation=vec)
        nt = NewtonSimEngine.add_object(_newton_stub(), "crate", orientation=vec)
        ic = IsaacSimulation.add_object(_isaac_stub(), "crate", orientation=vec)
        assert mj["status"] == nt["status"] == ic["status"] == "error", (vec, mj, nt, ic)
        texts = {mj["content"][0]["text"], nt["content"][0]["text"], ic["content"][0]["text"]}
        assert len(texts) == 1, texts

    @pytest.mark.parametrize("vec", GOOD_POSITIONS)
    def test_a_usable_pose_is_accepted_everywhere(self, mj_sim, vec):
        """The parity is two-way: no backend refuses a pose another honors."""
        assert mj_sim.add_object("crate", position=vec)["status"] == "success"
        assert NewtonSimEngine.add_object(_newton_stub(), "crate", position=vec)["status"] == "success"
        assert IsaacSimulation.add_object(_isaac_stub(), "crate", position=vec)["status"] == "success"

    @pytest.mark.parametrize("vec", (UNUSABLE_POSITIONS[0], [NAN, 0.0, 0.0], "abc"))
    def test_move_object_position_verdicts_match(self, mj_sim, vec):
        assert mj_sim.add_object("crate", position=[0.0, 0.0, 0.5])["status"] == "success"
        mj = mj_sim.move_object("crate", position=vec)
        nt = NewtonSimEngine.move_object(_newton_stub_with_object(), "crate", position=vec)
        ic = IsaacSimulation.move_object(_isaac_stub_with_object(), "crate", position=vec)
        assert mj["status"] == nt["status"] == ic["status"] == "error", (vec, mj, nt, ic)
        texts = {mj["content"][0]["text"], nt["content"][0]["text"], ic["content"][0]["text"]}
        assert len(texts) == 1, texts


# --------------------------------------------------------------------------- #
# The out-of-scope parameters are pinned as unchanged                          #
# --------------------------------------------------------------------------- #
class TestSizeIsOutOfScope:
    """This change is the pose axis only; ``size`` keeps its own contract.

    Three axes have since left this list. ``color`` counts are settled on the
    shared ``coerce_rgba`` domain, pinned in
    ``tests/simulation/test_color_domain_across_backends.py``; ``mass`` is
    settled on the shared ``SimEngine._validate_mass`` domain, pinned in
    ``tests/simulation/test_object_mass_domain_across_backends.py`` - including
    the ``mass=0`` "make it static" spelling Newton documents, which is the one
    place the three backends' accepted masses legitimately differ. The
    shape-independent half of ``size`` is settled on the shared
    ``coerce_size_vector`` domain, pinned in
    ``tests/simulation/test_object_size_domain_across_backends.py``.

    What remains out of scope for the pose axis is the part of ``size`` that
    depends on the shape: the per-shape component count, whether a short vector
    may be completed from trailing defaults, and whether a consumed extent must
    be positive. #1858 tracks those three, and
    ``test_object_size_domain_across_backends.py`` carries their boundary pins
    beside the domain they bound - one class, not two.

    The pin that used to live here asserted that Newton read ``size=[]`` as an
    omission and applied its default extent. That was a correct statement of the
    behaviour at the time and is now the defect the shared domain removes, so it
    is replaced rather than deleted: the property worth keeping is that the pose
    change did not quietly alter what an omitted ``size`` does.
    """

    def test_an_omitted_size_still_takes_the_backend_default(self):
        """Membership, not truthiness - and ``None`` still means omitted."""
        stub = _newton_stub()
        assert NewtonSimEngine.add_object(stub, "crate", size=None)["status"] == "success"
        assert stub._world.objects["crate"].size == [0.05, 0.05, 0.05]


# --------------------------------------------------------------------------- #
# Structural guard                                                            #
# --------------------------------------------------------------------------- #
#: Parameters that name a pose vector written into a transform.
_POSE_PARAMS = frozenset({"position", "orientation", "target"})

#: Every public engine-class method that takes one, as ``(backend, method)``.
#: Asserted exactly, so a scan root that resolved elsewhere fails loudly instead
#: of reporting a clean sweep over nothing.
_KNOWN_PLACEMENT_METHODS = {
    ("mujoco", "add_camera"),
    ("mujoco", "add_object"),
    ("mujoco", "add_robot"),
    ("mujoco", "move_object"),
    ("mujoco", "move_to"),
    ("newton", "add_camera"),
    ("newton", "add_object"),
    ("newton", "add_robot"),
    ("newton", "move_object"),
    ("isaac", "add_camera"),
    ("isaac", "add_object"),
    ("isaac", "add_robot"),
    ("isaac", "move_object"),
    ("isaac", "move_to"),
    ("isaac", "set_robot_pose"),
}


def _scan_placement_methods(root: pathlib.Path) -> tuple[set[tuple[str, str]], list[str]]:
    """Return ``(found, unguarded)`` over every backend engine class under ``root``."""
    found: set[tuple[str, str]] = set()
    unguarded: list[str] = []
    for backend in ("mujoco", "newton", "isaac"):
        for path in sorted((root / backend).glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
                # Direct children only: a nested helper belongs to its own scope.
                for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
                    if fn.name.startswith("_"):
                        continue
                    args = fn.args.args + fn.args.kwonlyargs
                    if not any(a.arg in _POSE_PARAMS for a in args):
                        continue
                    found.add((backend, fn.name))
                    routed = any(
                        isinstance(node, ast.Call)
                        and (
                            (isinstance(node.func, ast.Name) and node.func.id == "coerce_pose_vector")
                            # Routing through the backend-agnostic core counts:
                            # MotionPrimitivesCore._validate_move_to_args wraps
                            # coerce_pose_vector (pinned non-vacuously by
                            # test_the_shared_core_validator_uses_the_shared_helper).
                            or (isinstance(node.func, ast.Attribute) and node.func.attr == "_validate_move_to_args")
                        )
                        for node in ast.walk(fn)
                    )
                    if not routed:
                        unguarded.append(f"{backend}/{path.name}:{fn.lineno} {cls.name}.{fn.name}")
    return found, unguarded


class TestNoPlacementMethodDrifts:
    """A new backend method taking a pose must route it through the shared helper."""

    def test_every_placement_method_routes_through_the_shared_helper(self):
        root = pathlib.Path(inspect.getfile(SimEngine)).parent
        _found, unguarded = _scan_placement_methods(root)
        assert unguarded == [], (
            "these methods take a pose vector without routing it through "
            f"strands_robots.utils.coerce_pose_vector: {unguarded}"
        )

    def test_the_scan_sees_the_methods_it_is_meant_to_cover(self):
        """Non-vacuity: an empty or mis-rooted scan cannot pass the test above."""
        root = pathlib.Path(inspect.getfile(SimEngine)).parent
        found, _unguarded = _scan_placement_methods(root)
        assert found == _KNOWN_PLACEMENT_METHODS, found.symmetric_difference(_KNOWN_PLACEMENT_METHODS)

    def test_the_scan_flags_a_planted_omission(self, tmp_path):
        """And it really detects one, so a clean sweep means clean sources."""
        for backend in ("mujoco", "newton", "isaac"):
            (tmp_path / backend).mkdir()
        (tmp_path / "newton" / "simulation.py").write_text(
            "class Engine:\n    def place(self, name, position=None):\n        return {'status': 'success'}\n",
            encoding="utf-8",
        )
        found, unguarded = _scan_placement_methods(tmp_path)
        assert found == {("newton", "place")}
        assert len(unguarded) == 1
        assert "Engine.place" in unguarded[0]

    def test_the_shared_core_validator_uses_the_shared_helper(self):
        """The indirection the scan accepts must end at ``coerce_pose_vector``.

        ``move_to`` routes its pose parameters through
        ``MotionPrimitivesCore._validate_move_to_args`` (the backend-agnostic
        core extracted in #2153), so the scan accepts that call as routed.
        That acceptance is only sound while the core validator itself calls
        the shared helper - this pins the second hop, so the chain cannot be
        broken by editing the module the scan does not walk.
        """
        from strands_robots.simulation.motion_primitives_base import MotionPrimitivesCore

        source = inspect.getsource(MotionPrimitivesCore._validate_move_to_args)
        assert "coerce_pose_vector" in source

    def test_a_module_level_spec_helper_is_out_of_scope_by_construction(self):
        """``reposition_body_in_scene`` is not a public method and takes no envelope.

        It edits the MuJoCo spec for a static body and returns ``bool``; its only
        production caller is ``move_object``, which validates first. Scoping the
        sweep to engine-class methods excludes it without a name-based exemption
        list, which is where the next hole would hide.
        """
        from strands_robots.simulation.mujoco import scene_ops

        sig = inspect.signature(scene_ops.reposition_body_in_scene)
        assert "self" not in sig.parameters
        # ``from __future__ import annotations`` there, so the annotation is a string.
        assert sig.return_annotation == "bool"


class TestNonFiniteIsTheReasonNotJustTheRule:
    """Premise for refusing ``nan``/``inf``: they poison the transform silently."""

    def test_a_nan_never_compares_out_of_range(self):
        """Every downstream bounds check on a ``nan`` component passes."""
        assert not (NAN > 1.0)
        assert not (NAN < -1.0)
        assert not (abs(NAN - 0.0) < 1e-9)

    def test_a_nan_quaternion_has_no_normalizable_norm(self):
        assert math.isnan(math.sqrt(NAN * NAN + 0.0 + 0.0 + 0.0))
