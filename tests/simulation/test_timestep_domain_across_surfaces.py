"""Every surface that installs a physics timestep applies the one shared domain.

The integration timestep is the ``dt`` each physics substep advances by, so a
value the integrator cannot honor poisons the whole world rather than one call.
:meth:`~strands_robots.simulation.base.SimEngine._validate_timestep` is the
shared domain that says so, and its docstring names the surface it was lifted
from: *"This is the same contract ``MuJoCoSimEngine.set_timestep`` already
enforces, so the value cannot be set at world creation on terms the setter would
refuse."*

All three backends' ``create_world`` route through it, and so does the MuJoCo
setter. The Newton setter did not - it carried a hand-rolled
``float()``/``math.isfinite()`` pair with no ``bool`` arm - even though its own
docstring says it *"Mirrors the MuJoCo backend"* and the module that pins it
says the same. Measured on one ``create_world()``, then ``set_timestep(<value>)``,
comparing the two backends over fifteen values:

* ``True`` / ``numpy.True_`` / ``numpy.bool_(True)`` -> Newton
  ``status="success"``, ``Timestep: 1.0s (1Hz)``, and ``world.timestep == 1.0``:
  a one-second step, 500x the 0.002 default, installed by a value that is not a
  number. MuJoCo refused all three, and so did Newton's *own*
  ``create_world(timestep=True)``, so one backend held two domains for one field.
* ``False`` was refused by both - but on Newton only by accident, via
  ``float(False) == 0.0`` failing the ``> 0`` test rather than by being a
  boolean. Half-handled by coincidence is why nothing noticed the other half.
* the twelve remaining values (``nan``, ``inf``, ``-inf``, ``0``, ``0.0``,
  ``-0.002``, ``None``, ``[0.002]``, ``"0.002"``, ``numpy.float64(0.002)``,
  ``0.002``) already agreed, so the divergence was exactly the boolean family:
  three cells of fifteen.

A one-second ``dt`` is not merely a coarse simulation. Newton advances
``dt = timestep / substeps`` per solver step, so with the default ten substeps
that value makes one ``step()`` call cover a full second of simulated time
instead of 0.002 s. Replaying both step sizes in MuJoCo with a 1 kg 0.12 m box
released 0.60 m above the floor:

============================== ================= =================
quantity                       dt from ``True``  dt from ``0.002``
============================== ================= =================
solver dt                      0.1 s             0.0002 s
sim time per ``step()``         1.0 s            0.002 s
control steps spanning the fall 1                1500
settled height (rest = 0.06 m) 0.05508 m         0.05989 m
penetration into the floor     4.92 mm           0.11 mm
============================== ================= =================

So the whole 0.54 m fall happens between two consecutive observations - there is
no trajectory for a policy to act on - and the contact is resolved 45x worse,
the box coming to rest almost 5 mm inside the ground plane. Both under
``status="success"``, reported as ``Timestep: 1.0s (1Hz)``.

The world builders were pinned the same way - by claim rather than by measurement.
Every backend's ``create_world`` validates the *effective* dt (the argument, or
the engine default when the argument is omitted) and names whichever knob it came
from: ``default_timestep`` on MuJoCo and Newton, ``physics_dt`` on Isaac. That is
six cells, and only the two MuJoCo ones were driven - by
``tests/simulation/mujoco/test_create_world_physics_param_validation.py``, which
is ``importorskip("mujoco")``-gated and MuJoCo-only by construction. The scan
below covered ``set_timestep`` alone, so the sentence above ("all three
backends' ``create_world`` route through it") was structurally unenforced for
every backend and behaviourally unmeasured for two.

The effective-dt check is load-bearing there rather than defensive, because
neither of those two backends fully validates its engine default at construction:
``NewtonSimEngine.__init__`` stores ``default_timestep`` raw, and
``IsaacConfig.__post_init__`` tests ``physics_dt <= 0`` - a bare comparison,
which is False for ``nan`` and ``inf`` and lets a boolean through. So
``IsaacConfig(physics_dt=float("nan"))`` constructs, and ``create_world()`` is
the only thing between that object and a world built on a dt no integrator can
advance by.

Solver-free: ``NewtonSimEngine.set_timestep`` validates and writes before it
touches the solver, so the engine here is built via ``__new__`` with only the
attributes that path reads. The pre-existing Newton pins for this method live in
``tests/simulation/newton/test_gravity_timestep.py``, which is skipped whenever
Newton/Warp are absent and asserts only ``"positive" in text`` - a substring both
domains satisfy - so it could not have caught this on either count.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import threading
import types
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.models import SimWorld
from strands_robots.simulation.newton.simulation import NewtonSimEngine

_DEFAULT_DT = 0.002

# Values no integrator can honor. ``False`` is here because it is a boolean, not
# because it is zero: the coincidence that ``float(False) == 0.0`` is what made
# the missing bool arm invisible.
UNUSABLE = [
    True,
    False,
    np.True_,
    np.bool_(True),
    float("nan"),
    float("inf"),
    float("-inf"),
    0,
    0.0,
    -0.002,
    None,
    [0.002],
]

# Values the domain accepts, so a refusal here would be a regression rather than
# a fix. ``"0.002"`` and the NumPy scalar are accepted because the shared domain
# coerces anything ``float()`` accepts - see its docstring.
USABLE = [0.002, 0.5, np.float64(0.002), "0.002"]

BOOLEANS = [True, np.True_, np.bool_(True)]

# ``None`` is the documented "use the engine default" sentinel for the
# ``create_world(timestep=...)`` ARGUMENT, so it is not unusable there - the
# MuJoCo module pinning the same builder excludes it for the same reason. It
# stays unusable as the engine DEFAULT (there is nothing further to fall back
# to), and as a setter argument (the setter has no sentinel); both asymmetries
# are pinned in TestTheEngineDefaultSentinelIsArgumentOnly.
UNUSABLE_ARGUMENTS = [value for value in UNUSABLE if value is not None]


def _newton_engine() -> NewtonSimEngine:
    """A Newton engine holding a created world, without a Newton install.

    ``__init__`` imports Newton/Warp and builds a solver. ``set_timestep``
    validates, takes the lock and writes ``world.timestep``; none of that is
    physics, so the engine is built via ``__new__`` with just those attributes -
    the harness ``tests/simulation/newton/test_free_base_is_not_an_actuator.py``
    uses. ``_model`` is the non-``None`` sentinel for "world created".
    """
    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    engine._world = SimWorld(timestep=_DEFAULT_DT, gravity=[0.0, 0.0, -9.81])
    engine._model = object()
    engine._lock = threading.RLock()
    return engine


def _stored_timestep(engine: NewtonSimEngine) -> float:
    """The dt the world currently holds.

    ``_world`` is declared ``SimWorld | None`` on the engine and the fixture
    above always builds one, so the narrowing happens here once instead of at
    every assertion.
    """
    world = engine._world
    assert world is not None
    return float(world.timestep)


def _text(result: dict[str, Any]) -> str:
    return " ".join(block["text"] for block in result.get("content", []) if "text" in block)


def _set(engine: NewtonSimEngine, value: Any) -> dict[str, Any]:
    """Call the setter with a deliberately off-type value.

    Routed through one funnel so the off-domain values, which the annotation
    ``float`` does not describe, need a single documented ``Any`` rather than a
    suppression at every call site.
    """
    return engine.set_timestep(value)


class TestTheNewtonSetterRefusesWhatNoIntegratorCanHonor:
    @pytest.mark.parametrize("value", UNUSABLE, ids=repr)
    def test_an_unusable_timestep_is_refused(self, value: Any) -> None:
        engine = _newton_engine()
        result = _set(engine, value)
        assert result["status"] == "error", f"{value!r} was accepted"
        assert "set_timestep" in _text(result)

    @pytest.mark.parametrize("value", UNUSABLE, ids=repr)
    def test_a_refused_timestep_is_not_installed(self, value: Any) -> None:
        """The world keeps its dt, so a refused call cannot half-apply.

        The write is ``world.timestep = timestep`` under the lock, and Newton
        reads that live on every step, so a value that reached it would be in
        force for the rest of the session.
        """
        engine = _newton_engine()
        _set(engine, value)
        assert _stored_timestep(engine) == pytest.approx(_DEFAULT_DT)

    @pytest.mark.parametrize("value", BOOLEANS, ids=repr)
    def test_a_boolean_is_named_as_a_boolean(self, value: Any) -> None:
        """``True`` is refused for being a boolean, not for being out of range.

        ``float(True)`` is ``1.0``, which is finite and positive, so a domain
        that only checks the number accepts it. The message has to say which
        mistake was made or the caller reads "must be positive" against a value
        that is.
        """
        result = _set(_newton_engine(), value)
        assert result["status"] == "error"
        assert "bool" in _text(result).lower()

    @pytest.mark.parametrize("value", USABLE, ids=repr)
    def test_a_usable_timestep_is_still_accepted(self, value: Any) -> None:
        engine = _newton_engine()
        result = _set(engine, value)
        assert result["status"] == "success", _text(result)
        assert _stored_timestep(engine) == pytest.approx(float(value))

    def test_a_large_but_usable_timestep_still_warns_rather_than_refusing(self) -> None:
        """The warn-not-reject arm above 0.1 s is unchanged by the new domain."""
        result = _set(_newton_engine(), 0.5)
        assert result["status"] == "success"
        assert "unusually large" in _text(result)


class TestTheSetterAndTheWorldBuilderAgree:
    """A dt the world builder refuses cannot be installed afterwards.

    ``create_world`` calls the shared domain directly, so its verdict is the
    staticmethod's verdict. Comparing the setter against it is what stops the
    two drifting again on this backend - which is the failure this module was
    written for: ``create_world(timestep=True)`` refused while
    ``set_timestep(True)`` installed a 1-second step.
    """

    @pytest.mark.parametrize("value", UNUSABLE + USABLE, ids=repr)
    def test_the_setter_matches_the_creation_domain(self, value: Any) -> None:
        creation_refuses = SimEngine._validate_timestep(value, "create_world") is not None
        setter_refuses = _set(_newton_engine(), value)["status"] == "error"
        assert setter_refuses == creation_refuses, (
            f"{value!r}: create_world refuses={creation_refuses}, set_timestep refuses={setter_refuses}"
        )


class TestBothBackendsSetTimestepAgree:
    """The Newton setter answers as the MuJoCo setter it says it mirrors does.

    Newton's docstring claims to mirror MuJoCo, and the module pinning it says
    the same; neither claim was checked against the MuJoCo verdict, and the two
    disagreed on three values.
    """

    @pytest.mark.parametrize("value", UNUSABLE + USABLE, ids=repr)
    def test_the_two_backends_return_the_same_verdict(self, value: Any) -> None:
        pytest.importorskip("mujoco")
        from strands_robots import Simulation

        newton_refuses = _set(_newton_engine(), value)["status"] == "error"

        sim = Simulation(backend="mujoco", mesh=False)
        try:
            sim.create_world()
            mujoco_refuses = sim.set_timestep(value)["status"] == "error"
        finally:
            sim.cleanup()

        assert newton_refuses == mujoco_refuses, (
            f"{value!r}: newton refuses={newton_refuses}, mujoco refuses={mujoco_refuses}"
        )


def _backend_dir() -> pathlib.Path:
    """The simulation package, derived from a symbol rather than a path literal."""
    return pathlib.Path(inspect.getfile(SimEngine)).parent


# Every public surface that installs a physics timestep. ``create_world`` writes
# the same field the setter does - before any setter can be called - so the two
# belong to one domain, which is this module's subject.
_TIMESTEP_SURFACES = ("set_timestep", "create_world")


def _surfaces_missing_the_shared_domain(root: pathlib.Path) -> dict[str, list[str]]:
    """Timestep-installing methods that do not call the shared domain.

    Keyed by ``<backend>/<module>.py`` so a failure names the file to fix, with
    ``<Class>.<method>`` naming the surface.
    """
    found: dict[str, list[str]] = {}
    for backend in ("mujoco", "newton", "isaac"):
        for module in sorted((root / backend).glob("*.py")):
            tree = ast.parse(module.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                for member in ast.iter_child_nodes(node):
                    if not isinstance(member, ast.FunctionDef) or member.name not in _TIMESTEP_SURFACES:
                        continue
                    calls = {
                        call.func.attr
                        for call in ast.walk(member)
                        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    }
                    if "_validate_timestep" not in calls:
                        found.setdefault(f"{backend}/{module.name}", []).append(f"{node.name}.{member.name}")
    return found


def _surfaces_present(root: pathlib.Path) -> set[str]:
    present = set()
    for backend in ("mujoco", "newton", "isaac"):
        for module in sorted((root / backend).glob("*.py")):
            text = module.read_text(encoding="utf-8")
            for surface in _TIMESTEP_SURFACES:
                if f"def {surface}(" in text:
                    present.add(f"{backend}/{module.name}::{surface}")
    return present


class TestNoBackendCanShipAnUnsharedTimestepDomain:
    def test_every_timestep_surface_calls_the_shared_domain(self) -> None:
        assert _surfaces_missing_the_shared_domain(_backend_dir()) == {}

    def test_the_scan_sees_the_surfaces_it_claims_to_cover(self) -> None:
        """Non-vacuity: name the surfaces, so a mis-rooted scan cannot pass.

        This is the matrix the module asserts in prose: three world builders and
        two setters. Isaac exposes no ``set_timestep``; if it gains one it joins
        this set and the guard above starts checking it.
        """
        assert _surfaces_present(_backend_dir()) == {
            "mujoco/simulation.py::create_world",
            "mujoco/simulation.py::set_timestep",
            "newton/simulation.py::create_world",
            "newton/simulation.py::set_timestep",
            "isaac/simulation.py::create_world",
        }

    def test_the_scan_detects_a_surface_that_hand_rolls_the_domain(self, tmp_path: pathlib.Path) -> None:
        """A planted copy of the defect must be found, or a clean result is luck."""
        for backend in ("mujoco", "newton", "isaac"):
            (tmp_path / backend).mkdir()
        (tmp_path / "newton" / "simulation.py").write_text(
            "import math\n"
            "class NewtonSimEngine:\n"
            "    def set_timestep(self, timestep):\n"
            "        if not math.isfinite(float(timestep)) or timestep <= 0:\n"
            '            return {"status": "error"}\n'
            "        self._world.timestep = timestep\n",
            encoding="utf-8",
        )
        assert _surfaces_missing_the_shared_domain(tmp_path) == {
            "newton/simulation.py": ["NewtonSimEngine.set_timestep"]
        }


def _isaac_engine(physics_dt: Any) -> Any:
    """An Isaac engine whose config carries ``physics_dt``, without an Isaac install.

    ``create_world`` reads exactly ``self._config.physics_dt`` before the shared
    timestep domain, and the refusal returns before the lock and before any
    stage work, so the engine is built via ``__new__`` with only that attribute.
    The config is a real :class:`IsaacConfig` wherever it will construct one, so
    the value under test is one a caller can really hold.
    """
    from strands_robots.simulation.isaac.config import IsaacConfig
    from strands_robots.simulation.isaac.simulation import IsaacSimulation

    engine = IsaacSimulation.__new__(IsaacSimulation)
    try:
        engine._config = IsaacConfig(physics_dt=physics_dt)
    except (TypeError, ValueError):
        # ``IsaacConfig`` refuses part of the domain itself (see
        # TestTheConfigGuardCannotSeeEveryUnusableDefault). Where it does, hold
        # the value directly so ``create_world`` is still measured on it.
        engine._config = types.SimpleNamespace(physics_dt=physics_dt)
    return engine


def _newton_world_builder(default_timestep: Any) -> NewtonSimEngine:
    """A Newton engine with no world yet, carrying ``default_timestep``.

    ``__init__`` stores ``default_timestep`` raw (nothing validates it there) and
    then imports Newton/Warp to build a solver. ``create_world`` validates the
    effective dt before it takes the lock, so ``__new__`` plus the two attributes
    that path reads is enough - and mirrors what ``__init__`` assigns.
    """
    engine = NewtonSimEngine.__new__(NewtonSimEngine)
    engine._world = None
    engine._lock = threading.RLock()
    engine.default_timestep = default_timestep
    return engine


def _create_world(engine: Any, **kwargs: Any) -> dict[str, Any]:
    """Call ``create_world`` with deliberately off-type values.

    One funnel so the off-domain values, which the ``float`` annotation does not
    describe, need a single documented ``Any`` instead of a suppression per call.
    """
    return engine.create_world(**kwargs)


class TestEveryWorldBuilderRefusesWhatNoIntegratorCanHonor:
    """The explicit ``timestep=`` argument, on each backend's ``create_world``.

    The module docstring's claim is that all three route through the shared
    domain. Only MuJoCo's refusal was ever driven
    (``tests/simulation/mujoco/test_create_world_physics_param_validation.py``,
    which is MuJoCo-only by construction), so the two backends whose builders
    were added later were pinned structurally and never behaviourally.
    """

    @pytest.mark.parametrize("value", UNUSABLE_ARGUMENTS, ids=repr)
    def test_newton_refuses_the_argument(self, value: Any) -> None:
        result = _create_world(_newton_world_builder(_DEFAULT_DT), timestep=value)
        assert result["status"] == "error"
        assert "timestep" in _text(result)

    @pytest.mark.parametrize("value", UNUSABLE_ARGUMENTS, ids=repr)
    def test_isaac_refuses_the_argument(self, value: Any) -> None:
        result = _create_world(_isaac_engine(_DEFAULT_DT), timestep=value)
        assert result["status"] == "error"
        assert "timestep" in _text(result)

    @pytest.mark.parametrize("value", UNUSABLE_ARGUMENTS, ids=repr)
    def test_the_three_backends_return_the_same_verdict(self, value: Any) -> None:
        """A dt one builder refuses cannot be accepted by another.

        The shared domain exists so the accepted set is one set; this compares
        the two skeleton-backed builders against the MuJoCo builder that has
        always been pinned, rather than against the staticmethod they all call.
        """
        pytest.importorskip("mujoco")
        from strands_robots import Simulation

        newton_refuses = _create_world(_newton_world_builder(_DEFAULT_DT), timestep=value)["status"] == "error"
        isaac_refuses = _create_world(_isaac_engine(_DEFAULT_DT), timestep=value)["status"] == "error"

        sim = Simulation(backend="mujoco", mesh=False)
        try:
            mujoco_refuses = _create_world(sim, timestep=value)["status"] == "error"
        finally:
            sim.destroy()

        assert newton_refuses == mujoco_refuses == isaac_refuses, (
            f"{value!r}: mujoco refuses={mujoco_refuses}, "
            f"newton refuses={newton_refuses}, isaac refuses={isaac_refuses}"
        )


class TestAnUnusableEngineDefaultIsNamedUnderItsOwnKnob:
    """``create_world()`` validates the EFFECTIVE dt, not just the argument.

    Each backend's comment says so in the same words - *"so an unusable engine
    default is reported under its own name instead of compiling into the
    world"* - and each names a different knob: ``default_timestep`` on MuJoCo and
    Newton, ``physics_dt`` on Isaac. A caller who never passed ``timestep`` must
    be told which knob is wrong, so the message naming the knob is the contract,
    not an incidental detail. Only the MuJoCo half was pinned.
    """

    @pytest.mark.parametrize("value", UNUSABLE, ids=repr)
    def test_newton_names_default_timestep(self, value: Any) -> None:
        result = _create_world(_newton_world_builder(value))
        assert result["status"] == "error"
        assert "default_timestep" in _text(result)

    @pytest.mark.parametrize("value", UNUSABLE, ids=repr)
    def test_isaac_names_physics_dt(self, value: Any) -> None:
        result = _create_world(_isaac_engine(value))
        assert result["status"] == "error"
        assert "physics_dt" in _text(result)
        # The knob the caller did not touch must not be blamed instead.
        assert "default_timestep" not in _text(result)


class TestTheConfigGuardCannotSeeEveryUnusableDefault:
    """Why the effective-dt check is load-bearing rather than defensive.

    Neither non-MuJoCo backend fully validates its engine default at
    construction: ``NewtonSimEngine.__init__`` stores ``default_timestep`` raw,
    and ``IsaacConfig.__post_init__`` tests ``physics_dt <= 0`` - a bare
    comparison, which is False for ``nan`` and for ``inf``, and True-for-neither
    of the booleans. So an unusable default really can be held by a constructed
    object, and ``create_world``'s effective-dt check is the only thing between
    it and a world built on a dt no integrator can advance by.
    """

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), True], ids=repr)
    def test_isaac_config_admits_a_default_create_world_then_refuses(self, value: Any) -> None:
        from strands_robots.simulation.isaac.config import IsaacConfig

        config = IsaacConfig(physics_dt=value)  # constructs: the `<= 0` test cannot see it
        # Identity rather than equality: the claim is that the field is stored with
        # no coercion at all, which `nan == nan` being False would otherwise hide.
        assert config.physics_dt is value

        engine = _isaac_engine(value)
        result = _create_world(engine)
        assert result["status"] == "error"
        assert "physics_dt" in _text(result)

    def test_the_newton_constructor_stores_the_default_unvalidated(self) -> None:
        """Source fact behind the fixture: ``__init__`` only assigns it."""
        source = inspect.getsource(NewtonSimEngine.__init__)
        assert "self.default_timestep = default_timestep" in source
        assert "_validate_timestep" not in source


class TestARefusedWorldBuilderCostsNoSolverWork:
    """A refusal returns before the lock; an accepted value proceeds past it.

    Both skeletons deliberately omit the attributes the post-guard body reads
    (Newton's Warp handle, Isaac's lock). A refused value therefore returns a
    structured error, while a value the domain accepts raises ``AttributeError``
    reaching for that state - which is what shows the guard runs first rather
    than after the world is under construction.
    """

    @pytest.mark.parametrize("value", UNUSABLE_ARGUMENTS, ids=repr)
    def test_a_refused_value_never_reaches_the_solver(self, value: Any) -> None:
        assert _create_world(_newton_world_builder(_DEFAULT_DT), timestep=value)["status"] == "error"
        assert _create_world(_isaac_engine(_DEFAULT_DT), timestep=value)["status"] == "error"

    def test_an_accepted_value_proceeds_past_the_guard(self) -> None:
        with pytest.raises(AttributeError):
            _create_world(_newton_world_builder(_DEFAULT_DT), timestep=0.004)
        with pytest.raises(AttributeError, match="_lock"):
            _create_world(_isaac_engine(_DEFAULT_DT), timestep=0.004)


class TestTheEngineDefaultSentinelIsArgumentOnly:
    """``timestep=None`` means "use the engine default" - and only as an argument.

    The same value therefore has three different verdicts on one field, which is
    why it needs pinning rather than assuming: as a ``create_world`` argument it
    selects the engine default and the call proceeds; as the engine default it is
    refused (nothing remains to fall back to); as a ``set_timestep`` argument it
    is refused (that surface has no sentinel).
    """

    def test_none_as_the_argument_selects_the_engine_default(self) -> None:
        # Accepted: the call proceeds past the guard into solver state the
        # skeletons deliberately lack, rather than returning a refusal.
        with pytest.raises(AttributeError):
            _create_world(_newton_world_builder(_DEFAULT_DT), timestep=None)
        with pytest.raises(AttributeError, match="_lock"):
            _create_world(_isaac_engine(_DEFAULT_DT), timestep=None)

    def test_none_as_the_engine_default_is_refused_under_its_own_name(self) -> None:
        newton = _create_world(_newton_world_builder(None))
        assert newton["status"] == "error"
        assert "default_timestep" in _text(newton)

        isaac = _create_world(_isaac_engine(None))
        assert isaac["status"] == "error"
        assert "physics_dt" in _text(isaac)

    def test_none_is_still_refused_by_the_setter(self) -> None:
        """The setter has no sentinel, so its domain is the stricter one."""
        assert _set(_newton_engine(), None)["status"] == "error"
