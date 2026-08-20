# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: ``add_camera`` refuses a name its own ``render`` resolves past.

A backend whose render entry points select the FREE camera for a token cannot
also let ``add_camera`` claim that token as a camera *name*. Until this fix the
MuJoCo backend did both, and the two halves disagreed silently - measured on
MuJoCo 3.11.0, one ``create_world`` then
``add_camera("free", position=[8, 8, 8], target=[0, 0, 0])``:

* ``status="success"``, text ``Camera 'free' added at [8.0, 8.0, 8.0]``;
* ``world.cameras`` held ``'free'`` and ``mj_name2id(model, CAMERA, "free")``
  resolved it, so the camera really was in the compiled model;
* ``list_cameras()`` answered ``['default', 'free']``, offering the name as
  renderable;
* and ``render(camera_name="free")`` took the ``cam_id = -1`` free-camera branch
  with label ``"free (default)"``. The camera at ``[8, 8, 8]`` was unreachable
  through the very API that created it, with no error at any point.

``"default"`` reached the same end by a worse route. It was refused, but only as
a *duplicate* - ``create_world`` registers the built-in free view under that
name - and that refusal prescribed the remedy that completed the defect:

```
add_camera('default')       -> error | camera 'default' already exists. Remove it first.
remove_camera('default')    -> success
add_camera('default') again  -> success | Camera 'default' added at [9.0, 9.0, 9.0]
   registry ['default'], compiled MJCF ['default'], list_cameras ['default']
   render(camera_name='default') -> still the free camera
```

So following the error message's own instruction replaced the advertised
free-view alias with an unreachable camera.

The Newton backend already refused the whole set, so this was a one-backend
parity gap against a rule that was settled and tested
(``test_multi_camera.py::test_reserved_name_rejected``). The fix states the token
set once, in :data:`~strands_robots.utils.FREE_CAMERA_TOKENS`, and refuses it
through the shared :func:`~strands_robots.utils.reserved_camera_name_error`, so
the side that *routes* a token and the side that *refuses* it read the same
definition. Eleven copies of that tuple literal were in the tree - seven in
``mujoco/rendering.py``, three in ``newton/simulation.py``, one in ``base.py``
written in a different order - and MuJoCo's ``add_camera`` had the set in a
comment but not in code, which is precisely how the two halves came to disagree.
``TestTheTokenSetHasOneDefinition`` pins that no twelfth copy can appear.

The Isaac backend deliberately stays out of the rule: its ``get_frame`` looks the
camera up in ``self._cameras`` directly with no token check, so ``"default"``
there is an ordinary addressable name - and is that backend's documented
signature default. ``TestIsaacDoesNotRouteAndSoDoesNotRefuse`` pins that, so a
later "make the backends consistent" change cannot break a documented call by
applying a rule whose premise Isaac does not share.

These tests need no GL: ``add_camera`` compiles the spec but renders nothing, and
the Newton half needs neither ``newton``/``warp`` nor a GPU because the guard runs
before the method touches a solver (the unbound-method stand-in pattern
``tests/simulation/newton/test_add_camera_numeric_validation.py`` uses).
"""

from __future__ import annotations

import ast
import pathlib
import threading
import types
from typing import Any, cast

import pytest

from strands_robots.simulation.newton.simulation import NewtonSimEngine
from strands_robots.utils import FREE_CAMERA_TOKENS, reserved_camera_name_error

#: The tokens that are also *addressable* strings, so only the reserved-name rule
#: can refuse them. ``None`` and ``""`` are refused earlier by
#: ``entity_name_error`` (they are unaddressable regardless of routing), which is
#: why they are not the subject of these tests.
_ADDRESSABLE_TOKENS = ("default", "free")

#: Names that must keep working - none of them is a routing token.
_GOOD_NAMES = ("wrist", "front", "defaults", "freely", "free_cam", "Free", "DEFAULT", "cam-1")


# --------------------------------------------------------------------------- #
# The shared domain                                                           #
# --------------------------------------------------------------------------- #
class TestTheDomain:
    @pytest.mark.parametrize("name", FREE_CAMERA_TOKENS)
    def test_every_routing_token_is_refused(self, name: Any) -> None:
        """A non-``str`` token is not this guard's business, so ``None`` passes through."""
        err = reserved_camera_name_error("add_camera", "name", name)
        if name is None:
            assert err is None, "an unaddressable name is entity_name_error's domain"
        else:
            assert err is not None, name
            assert "reserved" in err

    @pytest.mark.parametrize("name", _GOOD_NAMES)
    def test_a_name_that_merely_resembles_a_token_is_accepted(self, name: str) -> None:
        """Membership, not a prefix or a case-insensitive match.

        ``"Free"`` and ``"defaults"`` are not routing tokens, so the render
        entry points look them up like any other name and they must register.
        """
        assert reserved_camera_name_error("add_camera", "name", name) is None

    @pytest.mark.parametrize("name", (7, 0, None, ["free"], {"default": 1}, b"free"))
    def test_a_non_string_is_left_to_the_addressability_domain(self, name: Any) -> None:
        """Two guards, two questions - this one must not also answer the first."""
        assert reserved_camera_name_error("add_camera", "name", name) is None

    def test_the_message_names_the_method_the_value_and_the_reason(self) -> None:
        err = reserved_camera_name_error("add_camera", "name", "free")
        assert err is not None
        assert err.startswith("add_camera: 'free' is reserved")
        assert "pick a distinct camera name" in err
        # The reason, not just the verdict: why this name and not another.
        assert "free camera" in err

    def test_the_message_is_ascii(self) -> None:
        """Project rule: no non-ASCII in a user-facing string."""
        for token in FREE_CAMERA_TOKENS:
            err = reserved_camera_name_error("add_camera", "name", token)
            if err is not None:
                err.encode("ascii")


# --------------------------------------------------------------------------- #
# MuJoCo - the backend this fix changes                                       #
# --------------------------------------------------------------------------- #
@pytest.fixture
def sim():
    pytest.importorskip("mujoco")
    from strands_robots.simulation.mujoco.simulation import Simulation

    s = Simulation(tool_name="test_reserved_camera_name_sim", mesh=False)
    assert s.create_world()["status"] == "success"
    yield s
    s.cleanup()


def _compiled_camera_names(sim) -> list[str]:
    import mujoco

    model = sim._world._model
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]


class TestMujocoAddCamera:
    @pytest.mark.parametrize("name", _ADDRESSABLE_TOKENS)
    def test_a_routing_token_is_refused(self, sim, name: str) -> None:
        """Pre-fix: ``"free"`` returned success, ``"default"`` said "already exists"."""
        result = sim.add_camera(name, position=[8.0, 8.0, 8.0], target=[0.0, 0.0, 0.0])
        assert result["status"] == "error", (name, result)
        assert "reserved" in result["content"][0]["text"]

    @pytest.mark.parametrize("name", _ADDRESSABLE_TOKENS)
    def test_the_refusal_names_the_routing_as_the_reason(self, sim, name: str) -> None:
        """Not a bare "invalid name": the caller is told why this name cannot work."""
        text = sim.add_camera(name, position=[8.0, 8.0, 8.0], target=[0.0, 0.0, 0.0])["content"][0]["text"]
        assert "free camera" in text
        assert "pick a distinct camera name" in text

    def test_the_refusal_compiles_no_camera(self, sim) -> None:
        """The model is untouched - the pre-fix path really did inject a ``<camera>``."""
        before = _compiled_camera_names(sim)
        sim.add_camera("free", position=[8.0, 8.0, 8.0], target=[0.0, 0.0, 0.0])
        assert _compiled_camera_names(sim) == before
        assert "free" not in sim._world.cameras

    def test_the_refusal_does_not_advertise_the_name(self, sim) -> None:
        """``list_cameras`` offered ``'free'`` as renderable while render ignored it."""
        sim.add_camera("free", position=[8.0, 8.0, 8.0], target=[0.0, 0.0, 0.0])
        assert "free" not in [c for c in sim.list_cameras() if c != "default"]

    def test_the_prescribed_remedy_no_longer_completes_the_defect(self, sim) -> None:
        """The whole pre-fix ``"default"`` sequence, in one test.

        The duplicate-name refusal said "Remove it first."; doing that and
        retrying succeeded and left an unreachable camera at the removed
        alias's name. Both steps must now refuse for the same stated reason.
        """
        first = sim.add_camera("default", position=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.0])
        assert first["status"] == "error"
        assert "reserved" in first["content"][0]["text"]
        # The old message's remedy, followed literally: it is refused for the
        # same stated reason, so the sequence cannot be completed at all.
        removed = sim.remove_camera("default")
        assert removed["status"] == "error", removed
        assert "reserved" in removed["content"][0]["text"]
        second = sim.add_camera("default", position=[9.0, 9.0, 9.0], target=[0.0, 0.0, 0.0])
        assert second["status"] == "error", second
        assert "reserved" in second["content"][0]["text"]
        # The advertised alias and the compiled camera both survived every
        # step. Pre-fix the removal deleted the compiled camera the scene
        # ships with (this assertion read ``[]``) and nothing could restore it.
        assert "default" in sim._world.cameras
        assert _compiled_camera_names(sim) == ["default"]

    def test_the_reserved_check_precedes_the_duplicate_test(self, sim) -> None:
        """Ordering is the property: "already exists" was the misleading answer.

        ``create_world`` registers the built-in free view as ``"default"``, so
        the duplicate test can answer for that name - and its remedy is wrong.
        The reserved rule has to win.
        """
        assert "default" in sim._world.cameras
        text = sim.add_camera("default", position=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.0])["content"][0]["text"]
        assert "reserved" in text
        assert "already exists" not in text

    @pytest.mark.parametrize("name", _GOOD_NAMES)
    def test_an_addressable_camera_still_registers(self, sim, name: str) -> None:
        """The rule is membership - a name that merely resembles a token works."""
        assert sim.add_camera(name, position=[0.5, 0.0, 0.4], target=[0.0, 0.0, 0.0])["status"] == "success"
        assert name in sim._world.cameras
        assert name in _compiled_camera_names(sim)

    def test_a_registered_camera_is_reachable_by_the_name_it_was_given(self, sim) -> None:
        """The invariant behind the rule, stated positively and GL-free.

        Every name ``add_camera`` accepts must resolve in the compiled model,
        which is what ``render``'s non-token branch does with it. A routing
        token could not satisfy this, because render never reaches the lookup.
        """
        import mujoco

        assert sim.add_camera("wrist", position=[0.5, 0.0, 0.4], target=[0.0, 0.0, 0.0])["status"] == "success"
        cam_id = mujoco.mj_name2id(sim._world._model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
        assert cam_id >= 0
        assert "wrist" not in FREE_CAMERA_TOKENS


# --------------------------------------------------------------------------- #
# Newton - the backend the rule came from                                     #
# --------------------------------------------------------------------------- #
def _newton_stub() -> types.SimpleNamespace:
    """A stand-in for ``self`` carrying only what ``add_camera`` reads.

    ``add_camera`` touches ``self._world`` (its ``cameras`` dict), ``self._model``
    (its ``body_label`` list) and ``self._lock``. None of those need Newton, so a
    namespace with the three lets the guard run without the optional
    ``newton`` / ``warp`` packages or a GPU.
    """
    return types.SimpleNamespace(
        _world=types.SimpleNamespace(cameras={}),
        _model=types.SimpleNamespace(body_label=("ground", "ball")),
        _lock=threading.RLock(),
    )


def _newton_add_camera(stub: types.SimpleNamespace, name: Any) -> dict[str, Any]:
    """Call the unbound ``add_camera`` with the stand-in for ``self``.

    The stand-in is deliberately not a ``NewtonSimEngine``: the point is to reach
    the guard without the optional solver packages, and the guard runs before the
    method touches one. Funnelling every call through one boundary states that
    once - the shape the sibling
    ``tests/simulation/newton/test_add_camera_numeric_validation.py`` uses -
    instead of repeating it at each call site. A narrow ``cast`` rather than a
    suppression, so the argument stays checked at every real call.
    """
    # Unbound, with the stand-in supplied as ``self`` - a bound call would look
    # ``add_camera`` up on the namespace, which does not carry it. The ``cast`` is
    # a no-op at runtime and only tells the checker what the argument stands in
    # for, which keeps the boundary explicit without suppressing the check.
    return NewtonSimEngine.add_camera(
        cast(NewtonSimEngine, stub), name, position=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.0]
    )


class TestSameVerdictAsTheNewtonSibling:
    """Both backends route the tokens, so both must refuse them identically.

    Newton's refusal was an inline literal and MuJoCo's did not exist. Asserting
    the *text* rather than only the verdict is what makes the shared domain
    load-bearing: a second copy of the rule would drift in wording first.
    """

    @pytest.mark.parametrize("name", _ADDRESSABLE_TOKENS)
    def test_newton_refuses_it(self, name: str) -> None:
        stub = _newton_stub()
        result = _newton_add_camera(stub, name)
        assert result["status"] == "error", (name, result)
        assert "reserved" in result["content"][0]["text"]
        assert stub._world.cameras == {}

    @pytest.mark.parametrize("name", _ADDRESSABLE_TOKENS)
    def test_the_two_refusals_are_the_same_sentence(self, sim, name: str) -> None:
        stub = _newton_stub()
        newton = _newton_add_camera(stub, name)
        mujoco_result = sim.add_camera(name, position=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.0])
        assert newton["status"] == mujoco_result["status"] == "error"
        assert newton["content"][0]["text"] == mujoco_result["content"][0]["text"]

    @pytest.mark.parametrize("name", _GOOD_NAMES)
    def test_both_accept_the_same_good_names(self, sim, name: str) -> None:
        stub = _newton_stub()
        newton = _newton_add_camera(stub, name)
        mujoco_result = sim.add_camera(name, position=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.0])
        assert newton["status"] == mujoco_result["status"] == "success", (name, newton, mujoco_result)


# --------------------------------------------------------------------------- #
# Isaac - deliberately out of the rule                                        #
# --------------------------------------------------------------------------- #
class _FakeCameraHandle:
    """Stand-in for the Isaac ``Camera`` sensor handle."""


def _isaac_engine():
    from strands_robots.simulation.isaac.simulation import IsaacConfig, IsaacSimulation

    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._config = IsaacConfig()
    engine._lock = threading.RLock()
    engine._world = None
    engine._world_created = True
    engine._robots = {}
    engine._objects = {}
    engine._cameras = {}
    engine._prim_registry = []
    engine._cam_out_size = {}
    engine._camera_warmup_steps = 0
    engine._sim_time = 0.0
    engine._step_count = 0
    engine._main_tid = threading.get_ident()

    def _create_camera_prim(**kwargs: Any) -> tuple[Any, float]:
        return _FakeCameraHandle(), 24.0

    engine._create_camera_prim = _create_camera_prim  # type: ignore[method-assign]
    return engine


class TestIsaacDoesNotRouteAndSoDoesNotRefuse:
    """The premise of the rule, pinned where it does not hold.

    Applying the reserved-name rule to Isaac "for consistency" would break a
    documented call, because ``"default"`` is Isaac's own signature default. The
    rule follows from routing, and Isaac does not route.
    """

    def test_isaac_get_frame_looks_the_name_up_instead_of_routing_it(self) -> None:
        """The premise: no token check on the read path, so no token is special."""
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        source = pathlib.Path(inspect_file(IsaacSimulation)).read_text(encoding="utf-8")
        tree = ast.parse(source)
        get_frame = _find_method(tree, "IsaacSimulation", "get_frame")
        assert get_frame is not None, "IsaacSimulation.get_frame not found"
        body = ast.get_source_segment(source, get_frame) or ""
        assert "_cameras[camera_name]" in body, "Isaac resolves the name directly"
        # No membership test against the token set anywhere on the read path.
        # ``camera_name: str = "default"`` in the signature is the *default* this
        # backend documents, not a route, so the scan looks for the comparison
        # rather than for the word.
        assert "FREE_CAMERA_TOKENS" not in body, "Isaac get_frame now routes the tokens"
        for node in ast.walk(get_frame):
            if isinstance(node, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
                rendered = ast.get_source_segment(source, node) or ""
                assert "free" not in rendered, f"Isaac get_frame now routes: {rendered}"

    @pytest.mark.parametrize("name", _ADDRESSABLE_TOKENS)
    def test_isaac_still_accepts_the_name(self, name: str) -> None:
        engine = _isaac_engine()
        result = engine.add_camera(name, position=[2.0, 2.0, 2.0], target=[0.0, 0.0, 0.0])
        assert result["status"] == "success", (name, result)
        assert name in engine._cameras

    def test_isaacs_documented_signature_default_still_works(self) -> None:
        """``add_camera()`` with no name is a documented call on this backend."""
        import inspect as _inspect

        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        assert _inspect.signature(IsaacSimulation.add_camera).parameters["name"].default == "default"
        engine = _isaac_engine()
        result = engine.add_camera(position=[2.0, 2.0, 2.0], target=[0.0, 0.0, 0.0])
        assert result["status"] == "success", result
        assert "default" in engine._cameras


# --------------------------------------------------------------------------- #
# Structural: the token set has exactly one definition                        #
# --------------------------------------------------------------------------- #
def inspect_file(obj) -> str:
    import inspect as _inspect

    return _inspect.getsourcefile(obj) or ""


def _find_method(tree: ast.Module, class_name: str, method: str) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method:
                    return item
    return None


def _package_root() -> pathlib.Path:
    import strands_robots

    return pathlib.Path(strands_robots.__file__).parent


class TestTheTokenSetHasOneDefinition:
    """A twelfth copy of the tuple is how the two halves drifted the first time.

    The defect was not that the rule was wrong, but that the routing side and
    the creation side each carried their own spelling of the set - and one of
    them was a comment. Scanning for the literal keeps that from recurring.
    """

    def test_no_module_rewrites_the_literal(self) -> None:
        offenders = []
        for path in sorted(_package_root().rglob("*.py")):
            if path.name == "utils.py" and path.parent == _package_root():
                continue  # the one definition
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Tuple) or len(node.elts) != 4:
                    continue
                values = []
                for elt in node.elts:
                    if isinstance(elt, ast.Constant):
                        values.append(elt.value)
                if set(values) == {None, "", "default", "free"}:
                    offenders.append(f"{path.relative_to(_package_root())}:{node.lineno}")
        assert not offenders, f"re-spelled FREE_CAMERA_TOKENS instead of importing it: {offenders}"

    def test_the_definition_is_non_vacuous(self) -> None:
        """Guard the guard: a scan for a set nobody writes would pass trivially."""
        assert set(FREE_CAMERA_TOKENS) == {None, "", "default", "free"}

    def test_both_sides_of_the_contract_read_it(self) -> None:
        """The routing side and the refusing side, each importing the one name."""
        routing = (_package_root() / "simulation" / "mujoco" / "rendering.py").read_text(encoding="utf-8")
        refusing = (_package_root() / "utils.py").read_text(encoding="utf-8")
        assert "FREE_CAMERA_TOKENS" in routing
        assert "def reserved_camera_name_error" in refusing
        assert "FREE_CAMERA_TOKENS" in refusing

    def test_every_backend_that_routes_also_refuses(self) -> None:
        """The invariant, stated so a fourth routing backend cannot skip the guard."""
        sim_root = _package_root() / "simulation"
        for backend in ("mujoco", "newton"):
            files = list((sim_root / backend).rglob("*.py"))
            blob = "\n".join(p.read_text(encoding="utf-8") for p in files)
            assert "FREE_CAMERA_TOKENS" in blob, f"{backend} no longer routes the tokens"
            assert "reserved_camera_name_error" in blob, f"{backend} routes the tokens but does not refuse them"
