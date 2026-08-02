"""Regression tests: a creation site refuses a name that cannot address its entity.

``add_object`` / ``add_camera`` / ``add_robot`` claim a name. Until this fix they
accepted three values that produce an entity some part of this same API cannot
then address, and each failed differently and late (measured on MuJoCo 3.11.0,
one ``create_world`` then ``add_object(<name>, shape="box", size=[0.06]*3)``):

* ``""`` -> ``status="success"``, registry key ``''``, and the body compiled
  under MuJoCo's own sentinel for *unnamed*: ``mj_name2id(model, BODY, "")``
  returns ``-1``, so ``get_body_state(body_name="")`` answered
  ``Body '' not found``. Through ``add_camera`` it is worse - ``render`` routes
  ``camera_name in FREE_CAMERA_TOKENS`` to the free camera by an explicit
  token check, so a camera created as ``""`` can never be rendered from. The
  other two members of that set are addressable strings and so are outside this
  domain; they are refused as reserved names, in
  ``test_reserved_camera_name_at_creation.py``.
* ``"a\\x00b"`` -> ``status="success"``, registry key ``'a\\x00b'``, compiled
  body ``'a'``. MuJoCo compares names only up to the NUL, so
  ``mj_name2id(..., "a\\x00b")`` and ``mj_name2id(..., "a")`` returned the SAME
  id: two names, one entity, each layer believing a different one. Through
  ``add_robot`` the NUL took the namespace separator with it - the bodies
  compiled as ``'abase'`` / ``'alink'`` instead of under the ``'a\\x00b/'``
  prefix ``list_bodies`` looks for.
* ``7`` -> the int key was entered in the registry and only *then* did the spec
  build reach ``add_body()`` and raise ``TypeError``, so the raise escaped the
  agent-tool dict these methods document as their only failure channel AND left
  the world holding an entry for a body that does not exist. ``["x"]`` raised out
  of the duplicate-name test itself (``unhashable type``). Through ``add_robot``
  the int name did compile - under registry key ``7``, which the tool surface,
  where a name always arrives as a JSON string, can never address - and any
  OTHER falsy value (``0``, ``[]``) fell into the ``if not name`` derive branch
  and reported success under a label the caller never asked for.

The lookup half of this was closed separately (#1774/#1776: a name that cannot
be a registry key resolves to "absent"). A creation site cannot reuse that
answer - it has to refuse - so the domain lives in one shared
:func:`~strands_robots.utils.entity_name_error` and all three backends' three
methods route through it, keeping them from drifting.

On the Isaac backend the same three values were accepted, and there the name is
also interpolated into the USD prim path (``{stage_path}/Robots/{name}``), which
turns an unaddressable name into cross-entity corruption. Measured with three
procedural robots and no Isaac Sim installed: ``add_robot("")`` reported
``status="success"`` at the prim path ``/World/Robots/`` - the *container* scope
for every robot rather than a child prim - so ``remove_robot("")``, which prunes
the cleanup registry with ``p.startswith(prim_path)``, pruned EVERY robot's prim,
leaving two live robots with zero tracked prims to release at teardown.
``add_robot(7)`` / ``add_object(7)`` succeeded under the int registry key ``7``,
and an unhashable name raised ``TypeError`` out of the duplicate-name test
instead of returning the structured envelope those methods document.

What is deliberately NOT refused is pinned too (``TestTheDomainIsNotAnAllowlist``):
this is not the MJCF-interpolation allowlist ``^[a-zA-Z0-9_-]+$``. A namespaced,
dotted, spaced or non-ASCII name is addressable, and narrowing to an allowlist is
a separate decision from refusing a name that demonstrably cannot address its
entity.

These tests are GL-free (``mesh=False``, no render) and the Newton half needs
neither ``newton``/``warp`` nor a GPU: the guard runs before either method
touches a solver, so calling the unbound method with a small stand-in for
``self`` exercises it in every environment (the pattern
``tests/simulation/newton/test_add_camera_numeric_validation.py`` uses).
"""

from __future__ import annotations

import os
import tempfile
import threading
import types

import pytest

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.isaac.simulation import IsaacConfig, IsaacSimulation
from strands_robots.simulation.newton.simulation import NewtonSimEngine
from strands_robots.utils import FREE_CAMERA_TOKENS, entity_name_error

# --------------------------------------------------------------------------- #
# Probe sets                                                                  #
# --------------------------------------------------------------------------- #

#: Names that are not a string. ``True`` is included because ``bool`` is an
#: ``int`` subclass; ``0``/``[]``/``{}`` because a falsy non-string is what the
#: MuJoCo ``add_robot`` derive branch used to swallow; ``["x"]``/``{"a": 1}``/
#: ``{1}`` because those are not hashable, so the duplicate-name test that was
#: the first thing to touch the name raised instead of answering.
_NON_STRING_NAMES = (7, 0, -1, 1.5, True, False, None, ["x"], {"a": 1}, {1}, ("a",), b"cube")

#: Strings carrying a NUL, in each position: leading, interior and trailing.
_NUL_NAMES = ("\x00", "\x00cube", "a\x00b", "cube\x00", "a\x00\x00b")

#: Names that must keep working. Everything here is addressable: MuJoCo resolves
#: it with ``mj_name2id`` and it round-trips through a JSON tool call.
_GOOD_NAMES = ("cube", "so101_gripper", "camera-1", "table.top", "obj 2", "küche", "x" * 200)


class _StrSubclass(str):
    """A ``str`` subclass is a string by every operation here and by the registry."""


# --------------------------------------------------------------------------- #
# The shared domain                                                           #
# --------------------------------------------------------------------------- #
class TestTheDomain:
    """``entity_name_error`` is the single definition the six call sites share."""

    @pytest.mark.parametrize("name", _NON_STRING_NAMES)
    def test_non_string_is_refused(self, name):
        err = entity_name_error("add_object", "name", name)
        assert err is not None, name
        assert "non-empty string" in err
        assert type(name).__name__ in err

    def test_empty_string_is_refused(self):
        err = entity_name_error("add_object", "name", "")
        assert err is not None
        assert "non-empty string" in err

    @pytest.mark.parametrize("name", _NUL_NAMES)
    def test_nul_is_refused_in_any_position(self, name):
        err = entity_name_error("add_object", "name", name)
        assert err is not None, name
        assert "NUL" in err

    @pytest.mark.parametrize("name", _GOOD_NAMES)
    def test_addressable_name_is_accepted(self, name):
        assert entity_name_error("add_object", "name", name) is None

    def test_str_subclass_is_accepted(self):
        """A ``str`` subclass keys the registry and compiles like any other string."""
        assert entity_name_error("add_object", "name", _StrSubclass("cube")) is None

    def test_message_names_the_method_and_the_parameter(self):
        """The caller sees which call and which argument was refused, not a bare type name."""
        err = entity_name_error("add_camera", "name", 7)
        assert err is not None
        assert err.startswith("add_camera: 'name'")

    def test_message_is_ascii(self):
        """Project rule: no non-ASCII in a user-facing string. A NUL name must not leak itself."""
        for probe in (*_NON_STRING_NAMES, "", *_NUL_NAMES):
            err = entity_name_error("add_object", "name", probe)
            assert err is not None
            err.encode("ascii")  # raises UnicodeEncodeError on a smuggled byte


class TestTheDomainIsNotAnAllowlist:
    """Pin the scope: only names that cannot address their entity are refused.

    Tightening to the MJCF-interpolation allowlist ``^[a-zA-Z0-9_-]+$`` would
    also refuse every name here, and each of these IS addressable - so that
    narrowing is a separate decision, not a side effect of this one.
    """

    @pytest.mark.parametrize("name", ["so101/gripper", "table.top", "obj 2", "küche", "cube\ttab", "a\nb"])
    def test_unusual_but_addressable_name_is_accepted(self, name):
        assert entity_name_error("add_object", "name", name) is None


# --------------------------------------------------------------------------- #
# MuJoCo                                                                      #
# --------------------------------------------------------------------------- #
_ARM_XML = """
<mujoco model="arm">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <light name="l" pos="0 0 3" dir="0 0 -1"/>
    <body name="base" pos="0 0 0">
      <geom name="torso" type="box" size="0.05 0.05 0.05"/>
      <body name="link" pos="0 0 0.1">
        <joint name="j" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom name="arm" type="capsule" size="0.02 0.05"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="j_act" joint="j"/></actuator>
</mujoco>
"""


@pytest.fixture
def arm_path():
    """The robot model on disk, so ``add_robot`` needs no asset download."""
    path = os.path.join(tempfile.mkdtemp(), "arm.xml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_ARM_XML)
    return path


@pytest.fixture
def sim():
    pytest.importorskip("mujoco")
    from strands_robots.simulation.mujoco.simulation import Simulation

    s = Simulation(tool_name="test_entity_name_domain_sim", mesh=False)
    assert s.create_world()["status"] == "success"
    yield s
    s.cleanup()


def _body_names(sim) -> list[str]:
    import mujoco

    model = sim._world._model
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]


class TestMujocoAddObject:
    @pytest.mark.parametrize("name", (*_NON_STRING_NAMES, "", *_NUL_NAMES))
    def test_unaddressable_name_is_refused(self, sim, name):
        """An error dict, not a raise and not a success.

        Pre-fix: ``""`` and a NUL name returned success, ``7`` and ``["x"]``
        raised ``TypeError`` through the tool-result contract.
        """
        result = sim.add_object(name, shape="box", size=[0.06, 0.06, 0.06], position=[0.0, 0.0, 0.5])
        assert result["status"] == "error", (name, result)
        assert "'name'" in result["content"][0]["text"]

    @pytest.mark.parametrize("name", (*_NON_STRING_NAMES, "", *_NUL_NAMES))
    def test_refusal_leaves_no_registry_entry(self, sim, name):
        """The orphan half: a refused creation must not claim the name either.

        Pre-fix ``add_object(7, ...)`` left ``7`` in ``world.objects`` with no
        body in the model - a world holding an entry for something that does not
        exist.
        """
        sim.add_object(name, shape="box", size=[0.06, 0.06, 0.06])
        assert list(sim._world.objects) == []

    @pytest.mark.parametrize("name", ("", "a\x00b"))
    def test_refusal_compiles_no_body(self, sim, name):
        """The model is untouched, so nothing unaddressable is left in it.

        Pre-fix the compiled body list was ``['world', '', 'a']`` - one body
        named with MuJoCo's unnamed sentinel and one under a truncated name.
        """
        before = _body_names(sim)
        sim.add_object(name, shape="box", size=[0.06, 0.06, 0.06])
        assert _body_names(sim) == before

    def test_an_addressable_object_still_round_trips(self, sim):
        """The happy path is unchanged: create it, then address it by that name."""
        assert (
            sim.add_object("cube", shape="box", size=[0.06, 0.06, 0.06], position=[0.0, 0.0, 0.5])["status"]
            == "success"
        )
        assert "cube" in sim._world.objects
        assert sim.get_body_state(body_name="cube")["status"] == "success"

    def test_the_duplicate_name_error_is_unchanged(self, sim):
        """A name that IS addressable and taken keeps its own error, not this one."""
        sim.add_object("cube", shape="box")
        result = sim.add_object("cube", shape="box")
        assert result["status"] == "error"
        assert "exists" in result["content"][0]["text"]


class TestMujocoAddCamera:
    @pytest.mark.parametrize("name", (*_NON_STRING_NAMES, "", *_NUL_NAMES))
    def test_unaddressable_name_is_refused(self, sim, name):
        result = sim.add_camera(name, position=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.0])
        assert result["status"] == "error", (name, result)
        assert "'name'" in result["content"][0]["text"]

    @pytest.mark.parametrize("name", (*_NON_STRING_NAMES, "", *_NUL_NAMES))
    def test_refusal_leaves_the_camera_registry_alone(self, sim, name):
        before = dict(sim._world.cameras)
        sim.add_camera(name, position=[1.0, 1.0, 1.0])
        assert sim._world.cameras == before

    def test_the_empty_name_collides_with_a_render_routing_token(self, sim):
        """Pin the premise for refusing ``""`` on a camera specifically.

        ``render``/``get_frame`` select the FREE camera for ``camera_name`` in
        this token set, so a camera registered under ``""`` is unreachable by
        construction - not merely awkward to address.

        This replaces a pin that asserted only ``"" in (None, "", "default",
        "free")`` against a copy of the tuple written here. That was true by
        inspection and therefore vacuous, and by restating the set locally it
        recorded the routing rule in a place that could not notice the rule
        moving. It now reads the one definition the routing itself reads, so the
        premise is checked rather than transcribed.

        The premise is *wider* than the conclusion drawn here: ``"default"`` and
        ``"free"`` are the same kind of unreachable and are refused as reserved
        names, which ``entity_name_error`` cannot express because they are
        perfectly addressable strings. That half lives in
        ``test_reserved_camera_name_at_creation.py``; this test keeps only the
        part that belongs to the addressability domain.
        """
        assert "" in FREE_CAMERA_TOKENS
        # And the domain under test here is what refuses it, not the reserved rule.
        assert entity_name_error("add_camera", "name", "") is not None

    def test_an_addressable_camera_still_registers(self, sim):
        assert sim.add_camera("wrist", position=[0.5, 0.0, 0.4], target=[0.0, 0.0, 0.0])["status"] == "success"
        assert "wrist" in sim._world.cameras


class TestMujocoAddRobot:
    @pytest.mark.parametrize("name", (7, 0, -1, 1.5, True, False, ["x"], {"a": 1}, {1}, b"arm", *_NUL_NAMES))
    def test_unaddressable_name_is_refused(self, sim, arm_path, name):
        """Includes the falsy non-strings the derive branch used to swallow.

        Pre-fix: ``7`` compiled bodies under the ``7/`` namespace but keyed the
        registry with the int; ``0`` silently derived ``'arm'`` from the model
        filename and reported success; ``["x"]`` raised ``unhashable type``;
        ``"a\\x00b"`` compiled ``'abase'``/``'alink'``, losing the namespace
        separator to the NUL.
        """
        result = sim.add_robot(name, urdf_path=arm_path)
        assert result["status"] == "error", (name, result)
        assert "'name'" in result["content"][0]["text"]
        assert list(sim._world.robots) == []

    @pytest.mark.parametrize("name", (None, ""))
    def test_the_documented_derive_short_form_still_derives(self, sim, arm_path, name):
        """``None`` and ``""`` are documented inputs, not refusals.

        This is the one place the domain is not applied verbatim: ``add_robot``
        advertises an omitted name as "derive a label from the model", and ``""``
        has always taken that branch. Refusing it here would break a documented
        call, so both pass through - and the label they derive is asserted, so a
        future tightening cannot quietly turn a derive into an error.
        """
        result = sim.add_robot(name, urdf_path=arm_path)
        assert result["status"] == "success", result
        assert list(sim._world.robots) == ["arm"]

    def test_an_explicit_addressable_label_still_works(self, sim, arm_path):
        assert sim.add_robot("arm1", urdf_path=arm_path)["status"] == "success"
        assert sim.get_robot_state("arm1")["status"] == "success"

    def test_a_refused_robot_leaves_the_model_alone(self, sim, arm_path):
        """No half-added robot: the compiled body list is untouched."""
        before = _body_names(sim)
        sim.add_robot("a\x00b", urdf_path=arm_path)
        assert _body_names(sim) == before


# --------------------------------------------------------------------------- #
# Newton (no newton / warp / GPU needed - the guard runs before the solver)    #
# --------------------------------------------------------------------------- #
def _newton_stub() -> types.SimpleNamespace:
    """A stand-in for ``self`` carrying only what the three guards read."""
    return types.SimpleNamespace(
        _world=types.SimpleNamespace(objects={}, cameras={}, robots={}),
        _model=types.SimpleNamespace(body_label=["ground"]),
        _lock=threading.RLock(),
        # Inherited from SimEngine, and read by ``add_object`` for its ``mass``.
        _validate_mass=SimEngine._validate_mass,
    )


class TestNewtonRefusesTheSameNames:
    """The Newton backend's three creation sites share the domain, so it cannot drift.

    ``entity_name_error`` documents the invariant: a name one backend refuses
    must be refused by the others. Without a guard here, the same ``add_object("")``
    that MuJoCo now refuses would still register an unaddressable entity on
    Newton.
    """

    @pytest.mark.parametrize("name", (*_NON_STRING_NAMES, "", *_NUL_NAMES))
    def test_add_object(self, name):
        stub = _newton_stub()
        result = NewtonSimEngine.add_object(stub, name)  # type: ignore[arg-type]
        assert result["status"] == "error", (name, result)
        assert "'name'" in result["content"][0]["text"]
        assert stub._world.objects == {}

    @pytest.mark.parametrize("name", (*_NON_STRING_NAMES, "", *_NUL_NAMES))
    def test_add_camera(self, name):
        stub = _newton_stub()
        result = NewtonSimEngine.add_camera(stub, name)  # type: ignore[arg-type]
        assert result["status"] == "error", (name, result)
        assert "'name'" in result["content"][0]["text"]
        assert stub._world.cameras == {}

    @pytest.mark.parametrize("name", (*_NON_STRING_NAMES, "", *_NUL_NAMES))
    def test_add_robot(self, name):
        stub = _newton_stub()
        result = NewtonSimEngine.add_robot(stub, name)  # type: ignore[arg-type]
        assert result["status"] == "error", (name, result)
        assert "'name'" in result["content"][0]["text"]
        assert stub._world.robots == {}

    def test_an_addressable_name_gets_past_the_guard(self):
        """The guard is not a blanket refusal: a usable name reaches the next check.

        ``add_object("cube")`` with no ``shape`` support behind it still fails,
        but on the shape/solver path - not on the name - which is what pins that
        the guard did not simply swallow every call.
        """
        stub = _newton_stub()
        result = NewtonSimEngine.add_object(stub, "cube", shape="not_a_shape")  # type: ignore[arg-type]
        assert result["status"] == "error"
        assert "'name'" not in result["content"][0]["text"]


# --------------------------------------------------------------------------- #
# Isaac (no Isaac Sim / GPU needed - the guard runs before the stage is touched) #
# --------------------------------------------------------------------------- #
def _isaac_stub() -> types.SimpleNamespace:
    """A stand-in for ``self`` carrying only what the three guards read."""
    return types.SimpleNamespace(
        _lock=threading.RLock(),
        _world_created=True,
        _config=IsaacConfig(),
        _robots={},
        _objects={},
        _cameras={},
        _replicated=False,
        _prim_registry=[],
    )


class TestIsaacRefusesTheSameNames:
    """The Isaac backend's three creation sites share the domain, so it cannot drift.

    ``entity_name_error`` documents the invariant: a name one backend refuses
    must be refused by the others. Isaac's ``add_camera`` docstring already made
    the same promise for ``position`` / ``target`` / ``fov`` / ``width`` /
    ``height`` ("a camera configuration one backend refuses is refused by all
    three") while the name - the thing that identifies the entity - went
    unchecked.
    """

    @pytest.mark.parametrize("name", (*_NON_STRING_NAMES, "", *_NUL_NAMES))
    def test_add_object(self, name):
        stub = _isaac_stub()
        result = IsaacSimulation.add_object(stub, name)  # type: ignore[arg-type]
        assert result["status"] == "error", (name, result)
        assert "'name'" in result["content"][0]["text"]
        assert stub._objects == {}

    @pytest.mark.parametrize("name", (*_NON_STRING_NAMES, "", *_NUL_NAMES))
    def test_add_camera(self, name):
        stub = _isaac_stub()
        result = IsaacSimulation.add_camera(stub, name)  # type: ignore[arg-type]
        assert result["status"] == "error", (name, result)
        assert "'name'" in result["content"][0]["text"]
        assert stub._cameras == {}

    @pytest.mark.parametrize("name", (*_NON_STRING_NAMES, "", *_NUL_NAMES))
    def test_add_robot(self, name):
        stub = _isaac_stub()
        result = IsaacSimulation.add_robot(stub, name)  # type: ignore[arg-type]
        assert result["status"] == "error", (name, result)
        assert "'name'" in result["content"][0]["text"]
        assert stub._robots == {}

    @pytest.mark.parametrize("name", (*_NON_STRING_NAMES, "", *_NUL_NAMES))
    def test_a_refused_name_reserves_no_prim_path(self, name):
        """The refusal precedes the stage: nothing is queued for cleanup.

        The prim path is interpolated from the name, so a refusal that landed
        after the append would leave a path in the teardown registry for a prim
        that was never created.
        """
        stub = _isaac_stub()
        IsaacSimulation.add_robot(stub, name)  # type: ignore[arg-type]
        assert stub._prim_registry == []

    def test_an_addressable_name_gets_past_the_guard(self):
        """The guard is not a blanket refusal: a usable name reaches the next check.

        ``add_object("cube", shape="not_a_shape")`` still fails, but on the shape
        path - not on the name - which pins that the guard did not simply swallow
        every call.
        """
        stub = _isaac_stub()
        result = IsaacSimulation.add_object(stub, "cube", shape="not_a_shape")  # type: ignore[arg-type]
        assert result["status"] == "error"
        assert "'name'" not in result["content"][0]["text"]

    def test_an_addressable_robot_name_reaches_the_procedural_lookup(self):
        """An addressable name still resolves a procedural robot and registers it.

        The procedural branch needs no Isaac Sim, so this is the positive half of
        the contract: the guard rejects exactly the unaddressable names and
        nothing else.
        """
        stub = _isaac_stub()
        result = IsaacSimulation.add_robot(stub, "arm", data_config="panda")  # type: ignore[arg-type]
        assert result["status"] == "success", result
        assert list(stub._robots) == ["arm"]
        assert stub._prim_registry == ["/World/Robots/arm"]

    def test_removing_one_robot_keeps_every_other_robots_prim(self):
        """Regression for the corruption an empty name used to cause.

        ``/World/Robots/`` prefixes every robot's prim path, and
        ``remove_robot`` prunes the cleanup registry by prefix - so removing the
        empty-named robot used to drop EVERY robot's prim while leaving the
        robots themselves registered. With the name refused at creation the
        empty name never enters the registry, so the prune stays scoped.
        """
        stub = _isaac_stub()
        for label in ("arm", "helper"):
            assert (
                IsaacSimulation.add_robot(stub, label, data_config="panda")["status"]  # type: ignore[arg-type]
                == "success"
            )
        assert IsaacSimulation.add_robot(stub, "", data_config="panda")["status"] == "error"  # type: ignore[arg-type]

        stub._action_controllers = {}
        assert IsaacSimulation.remove_robot(stub, "")["status"] == "error"  # type: ignore[arg-type]

        assert stub._prim_registry == ["/World/Robots/arm", "/World/Robots/helper"]
        assert sorted(stub._robots) == ["arm", "helper"]


class TestEveryBackendGivesTheSameVerdict:
    """All three backends refuse the same name with the same message."""

    @pytest.fixture
    def mj_sim(self):
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import Simulation

        s = Simulation(tool_name="test_entity_name_domain_parity_sim", mesh=False)
        s.create_world()
        yield s
        s.cleanup()

    @pytest.mark.parametrize("name", (7, ["x"], "", "a\x00b"))
    def test_add_object_verdicts_match(self, mj_sim, name):
        mj = mj_sim.add_object(name, shape="box")
        nt = NewtonSimEngine.add_object(_newton_stub(), name)  # type: ignore[arg-type]
        ic = IsaacSimulation.add_object(_isaac_stub(), name)  # type: ignore[arg-type]
        assert mj["status"] == nt["status"] == ic["status"] == "error", (mj, nt, ic)
        assert mj["content"][0]["text"] == nt["content"][0]["text"] == ic["content"][0]["text"]

    @pytest.mark.parametrize("name", (7, ["x"], "", "a\x00b"))
    def test_add_camera_verdicts_match(self, mj_sim, name):
        mj = mj_sim.add_camera(name, position=[1.0, 1.0, 1.0])
        nt = NewtonSimEngine.add_camera(_newton_stub(), name)  # type: ignore[arg-type]
        ic = IsaacSimulation.add_camera(_isaac_stub(), name)  # type: ignore[arg-type]
        assert mj["status"] == nt["status"] == ic["status"] == "error", (mj, nt, ic)
        assert mj["content"][0]["text"] == nt["content"][0]["text"] == ic["content"][0]["text"]
