"""``send_action`` must refuse a substep count it cannot advance, before it writes.

Companion to ``test_step_count_domain_across_backends.py``, which settled
``step(n_steps)``. ``send_action(action, robot_name, n_substeps)`` is the second
public stepping surface and it was left out of that change deliberately, because
it has a contract of its own: it *writes an actuator target and then advances*,
so a count it cannot honor is not merely a bad number of steps - a refusal that
arrives after the write leaves the robot commanded and the world un-advanced.

## What was measured on ``182fcc6``

No backend validated the count, and the three did not even agree on what a
``0`` meant. Real MuJoCo engine (mujoco 3.11.0, one position actuator), Newton
on its real inherited ``_advance`` solver-free, Isaac on its verbatim substep
loop:

| ``n_substeps`` | MuJoCo | Newton | Isaac |
| --- | --- | --- | --- |
| ``0`` | 1 ``mj_step``, ``step_count`` **+0** | 1 control step | **0 steps**, success |
| ``-5`` | 1 ``mj_step``, ``step_count`` **-5** | 1 control step | 0 steps, success |
| ``False`` | 1 ``mj_step``, ``step_count`` +0 | 1 control step | 0 steps, success |
| ``True`` | 1 ``mj_step`` | 1 control step | 1 step |
| ``2.7`` | **raises ``TypeError``** | ``step_count`` = **2.7** | raises ``TypeError`` |
| ``3.0`` | **raises ``TypeError``** | 3 steps, ``step_count`` = 3.0 | raises ``TypeError`` |
| ``nan`` | 1 ``mj_step``, ``step_count`` = **nan** | 1 control step | raises ``TypeError`` |
| ``inf`` | raises ``TypeError`` | ``step_count`` = **inf** | raises ``TypeError`` |
| ``"3"`` / ``None`` / ``[3]`` | raises ``TypeError`` | raises ``TypeError`` | raises ``TypeError`` |

Four findings the issue (#1870) did not have, each pinned below:

1. **The counter desynchronizes from the physics.** MuJoCo floors its loop at
   ``max(1, n_substeps)`` but adds the *raw* count to ``step_count``, so a ``0``
   ran one ``mj_step`` and recorded none, and a ``-5`` ran one and moved the
   counter **backwards**.
2. **``nan`` and ``inf`` poison the counter permanently.** ``step_count`` became
   ``nan`` on MuJoCo and ``inf`` on Newton - not one bad call, but every later
   reader of the world's step count for the rest of its life.
3. **``3.0`` - an integral float, the most innocuous value here - raised
   ``TypeError``** on two backends, *after* the target was written, straight past
   the documented ``{status, content}`` envelope. Its sibling ``step(3.0)``
   accepts it and advances 3 (``USABLE_COUNTS``), so the two surfaces disagreed
   about the same number.
4. **The backends disagreed about ``0``.** MuJoCo and Newton floored it to one
   step; Isaac, which has no floor at all, advanced none. So the same call
   already meant two different things, and "honor 0" would have been picking
   Isaac's *absence of a floor* over the reference backend's explicit one.

## Why the floor is ``1`` here and ``0`` on ``step``

The issue asked for this to be settled before code, listing "honor ``0`` (write
but do not advance)" against "refuse ``0`` and name ``step``". It is settled as
**refuse**, on evidence rather than taste: both producers of this count already
refuse anything below 1, each with its own raise -
``PolicyRunner._control_substeps`` returns ``>= 1`` and raises otherwise, with a
docstring recording that clamping ``0`` to 1 reinstated the exact
under-integration it exists to prevent, and ``training.rl.env.SimEnv`` refuses
an ``n_substeps`` below 1. ``send_action`` was the only member of that chain
without the guarantee, which is the "a sibling states the invariant and this
caller skips it" shape. Pinned in ``TestTheFloorIsADecisionWithEvidence``.

The domain is therefore ``positive_whole_number_error``: the same scalar policy
as ``step``'s ``non_negative_whole_number_error`` with the floor moved to 1.
``positive_count_error`` was the other candidate and is wrong here - it admits
only a true ``int``, so it would refuse ``3.0``, ``np.int64(3)`` and
``np.uint8(2)``, all of which the reference backend honors today and all of
which ``step`` honors by documented design.
"""

from __future__ import annotations

import ast
import inspect
import math
import pathlib
import textwrap
import threading
import types
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import IsaacSimulation
from strands_robots.simulation.models import SimRobot, SimWorld
from strands_robots.simulation.newton.simulation import NewtonSimEngine
from strands_robots.simulation.policy_runner import PolicyRunner
from strands_robots.utils import (
    non_negative_whole_number_error,
    positive_count_error,
    positive_whole_number_error,
)

pytest.importorskip("mujoco")

NAN = float("nan")
INF = float("inf")

#: An ``int`` wider than the float range: ``float(10**400)`` raises
#: ``OverflowError``. Refused by this guard, unlike ``step``'s - see the
#: magnitude paragraph in ``positive_whole_number_error``.
BEYOND_FLOAT_RANGE = 10**400
#: An ``int`` whose own ``repr`` raises (wider than
#: ``sys.get_int_max_str_digits()``), so rendering the refusal must not.
BEYOND_INT_STR_LIMIT = 10**5000

#: Every substep count no backend can honor. Each row is a measured pre-fix
#: acceptance, a silent floor, a poisoned counter or a bare raise on at least
#: one backend - see the table in the module docstring.
UNUSABLE_SUBSTEPS: tuple[Any, ...] = (
    0,
    -5,
    True,
    False,
    np.bool_(True),
    2.7,
    "3",
    NAN,
    INF,
    -INF,
    None,
    [3],
    np.array([3]),
    BEYOND_FLOAT_RANGE,
    BEYOND_INT_STR_LIMIT,
)

#: Counts every backend must honor, paired with the steps each must advance.
#: ``3.0`` is the load-bearing row: MuJoCo and Isaac *raised* on it pre-fix
#: while ``step(3.0)`` accepted it, so honoring it is what removes the
#: disagreement between the two surfaces rather than a new liberty.
USABLE_SUBSTEPS: tuple[tuple[Any, int], ...] = (
    (1, 1),
    (3, 3),
    (3.0, 3),
    (np.int64(3), 3),
    (np.uint8(2), 2),
    (np.float64(4.0), 4),
)


def _substep_id(value: Any) -> str | None:
    """Test ID for a probe, or ``None`` to let pytest name it.

    The outsized integers need naming to keep the module collectable: pytest
    derives an ID for an ``int`` parameter with ``str()``, which raises for one
    wider than ``sys.get_int_max_str_digits()``. That is a collection error
    taking down every class in the file, which is the same defect this module
    pins on the guard (rendering a value it was only asked to classify) reappearing
    in the harness.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        if value == BEYOND_FLOAT_RANGE:
            return "beyond_float_range"
        if value == BEYOND_INT_STR_LIMIT:
            return "beyond_int_str_limit"
    return None


_ARM_XML = """
<mujoco model="one_drive_arm">
  <compiler angle="radian"/>
  <option gravity="0 0 0" timestep="0.002"/>
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <joint name="lift" type="hinge" axis="0 1 0" range="-2 2" limited="true" damping="4"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
    </body>
  </worldbody>
  <actuator>
    <position name="lift_act" joint="lift" kp="30" ctrlrange="-2 2"/>
  </actuator>
</mujoco>
"""

_TIMESTEP = 0.002


@pytest.fixture
def sim(tmp_path):  # noqa: ANN001, ANN201 - the engine types _world as optional
    """A one-actuator arm from an inline MJCF, so no asset is fetched."""
    from strands_robots.simulation import Simulation

    path = tmp_path / "one_drive_arm.xml"
    path.write_text(_ARM_XML, encoding="utf-8")
    engine = Simulation(backend="mujoco", tool_name="substep_domain_test", mesh=False)
    engine.create_world()
    added = engine.add_robot(name="arm", urdf_path=str(path))
    assert added["status"] == "success", added
    yield engine
    engine.cleanup()


def _text(result: dict[str, Any]) -> str:
    return next(block["text"] for block in result["content"] if "text" in block)


def _observed(engine: Any) -> tuple[float, Any, float]:
    """The world state a refused call must leave untouched: time, count, ctrl."""
    data = engine._world._data
    return float(data.time), engine._world.step_count, float(data.ctrl[0])


# --------------------------------------------------------------------------- #
# The shared domain                                                           #
# --------------------------------------------------------------------------- #
class TestTheSharedDomain:
    """``positive_whole_number_error`` is the single definition all three share."""

    @pytest.mark.parametrize("count", UNUSABLE_SUBSTEPS, ids=_substep_id)
    def test_an_unusable_count_is_refused(self, count: Any) -> None:
        error = positive_whole_number_error(count, "n_substeps", "send_action")
        assert error is not None, count
        assert "n_substeps" in error

    @pytest.mark.parametrize(("count", "_expected"), USABLE_SUBSTEPS)
    def test_a_usable_count_is_accepted(self, count: Any, _expected: int) -> None:
        assert positive_whole_number_error(count, "n_substeps", "send_action") is None, count

    def test_an_accepted_count_survives_the_int_coercion_it_is_paired_with(self) -> None:
        """The guard exists so the ``int()`` each backend then applies cannot raise."""
        for count, expected in USABLE_SUBSTEPS:
            assert positive_whole_number_error(count, "n_substeps", "send_action") is None
            assert int(count) == expected

    def test_the_message_names_the_parameter_and_the_value(self) -> None:
        error = positive_whole_number_error(0, "n_substeps", "send_action")
        assert error == "send_action: n_substeps must be a positive whole number, got 0."

    def test_the_message_is_ascii_and_never_raises_while_rendering(self) -> None:
        for count in UNUSABLE_SUBSTEPS:
            error = positive_whole_number_error(count, "n_substeps", "send_action")
            assert error is not None
            error.encode("ascii")


# --------------------------------------------------------------------------- #
# The floor: the one thing the issue said was a decision                      #
# --------------------------------------------------------------------------- #
class TestTheFloorIsADecisionWithEvidence:
    """``0`` is refused here and honored by ``step``, and the reason is measured.

    #1870 asked for this to be settled before code. The evidence is that both
    producers of this count already refuse anything below 1, so refusing at the
    consumer makes the chain uniform instead of introducing a new rule.
    """

    def test_the_two_surfaces_differ_only_in_the_floor(self) -> None:
        assert positive_whole_number_error(0, "n_substeps", "send_action") is not None
        assert non_negative_whole_number_error(0, "n_steps", "step") is None
        for count, _expected in USABLE_SUBSTEPS:
            assert positive_whole_number_error(count, "n_substeps", "send_action") is None
            assert non_negative_whole_number_error(count, "n_steps", "step") is None

    def test_the_runner_that_produces_this_count_already_refuses_a_non_positive(self) -> None:
        """``_control_substeps`` is the sole producer on the rollout path.

        Its docstring records that clamping with ``max(1, int(override))`` let
        ``0``/``-5`` collapse to a single physics step, "reinstating the exact
        under-integration this helper exists to prevent". That is this issue's
        reasoning one layer up, and its return contract is ``>= 1``.
        """
        sim_double: Any = types.SimpleNamespace(physics_timestep=lambda: _TIMESTEP)
        runner = PolicyRunner(sim_double)
        for bad in (0, -5, 2.7, True):
            with pytest.raises(ValueError, match="control_substeps"):
                runner._control_substeps(30.0, bad)  # type: ignore[arg-type]
        assert runner._control_substeps(30.0, 4) == 4
        assert runner._control_substeps(30.0) >= 1

    def test_the_rl_env_that_produces_this_count_also_refuses_a_non_positive(self) -> None:
        """The second producer, refusing the same parameter name by its own raise.

        Read from the source rather than by constructing a ``SimEnv`` (which
        needs a live engine): the point being pinned is that the floor exists in
        a second place, so a reader cannot conclude ``1`` was invented here.
        """
        from strands_robots.training.rl import env as rl_env

        source = inspect.getsource(rl_env)
        assert "n_substeps must be >= 1" in source

    def test_advance_nothing_still_has_a_spelling_and_it_is_named(self) -> None:
        """Refusing ``0`` removes nothing a caller can express, and says so.

        ``step(0)`` is the documented no-op, and the refusal has to point at it -
        otherwise the caller is told what is wrong and not what to do.
        """
        assert non_negative_whole_number_error(0, "n_steps", "step") is None
        for doc in (
            IsaacSimulation.send_action.__doc__,
            NewtonSimEngine.send_action.__doc__,
        ):
            assert doc is not None
            assert "step" in doc


# --------------------------------------------------------------------------- #
# MuJoCo: the reference backend, on a real engine                             #
# --------------------------------------------------------------------------- #
class TestMuJoCoSendAction:
    """Measured on a real compiled model, so the ordering claim is observable."""

    @pytest.mark.parametrize("count", UNUSABLE_SUBSTEPS, ids=_substep_id)
    def test_an_unusable_count_is_refused_without_writing_or_stepping(self, sim, count: Any) -> None:
        before = _observed(sim)
        result = sim.send_action({"lift_act": 0.5}, robot_name="arm", n_substeps=count)
        assert result["status"] == "error", count
        assert "n_substeps" in _text(result)
        assert _observed(sim) == before, f"{count!r} changed the world it was refused for"

    @pytest.mark.parametrize(("count", "expected"), USABLE_SUBSTEPS)
    def test_a_usable_count_advances_exactly_that_many_steps(self, sim, count: Any, expected: int) -> None:
        t0, c0, _ctrl0 = _observed(sim)
        result = sim.send_action({"lift_act": 0.5}, robot_name="arm", n_substeps=count)
        assert result["status"] == "success", _text(result)
        t1, c1, ctrl1 = _observed(sim)
        assert round((t1 - t0) / _TIMESTEP) == expected
        assert c1 - c0 == expected
        assert ctrl1 == pytest.approx(0.5)

    def test_an_integral_float_no_longer_raises_past_the_envelope(self, sim) -> None:
        """``3.0`` raised ``TypeError`` from ``range()`` pre-fix, after the write.

        The single most innocuous value in the probe set - a count read from a
        config, or ``duration / dt`` - and the surface it reached raised rather
        than answering, having already commanded the robot. Its sibling
        ``step(3.0)`` accepted the same number.
        """
        result = sim.send_action({"lift_act": 0.5}, robot_name="arm", n_substeps=3.0)
        assert result["status"] == "success"
        assert sim._world.step_count == 3
        assert sim.step(n_steps=3.0)["status"] == "success"


class TestTheStepCounterCanNoLongerDesynchronizeOrBePoisoned:
    """``step_count`` tracked the raw count while the loop tracked a floored one.

    The loop was ``range(max(1, n_substeps))`` and the counter was
    ``step_count += n_substeps``, so the two disagreed for every count below 1 -
    and for ``nan`` the disagreement was permanent. Closed by the guard alone,
    because it was only reachable with a value outside the domain; pinned here so
    that stays true if either line is touched.
    """

    def test_a_zero_no_longer_steps_once_while_recording_none(self, sim) -> None:
        assert sim.send_action({"lift_act": 0.5}, robot_name="arm", n_substeps=0)["status"] == "error"
        assert sim._world.step_count == 0
        assert float(sim._world._data.time) == 0.0

    def test_a_negative_count_no_longer_moves_the_counter_backwards(self, sim) -> None:
        sim.send_action({"lift_act": 0.5}, robot_name="arm", n_substeps=4)
        assert sim._world.step_count == 4
        assert sim.send_action({"lift_act": 0.5}, robot_name="arm", n_substeps=-5)["status"] == "error"
        assert sim._world.step_count == 4

    def test_a_non_finite_count_no_longer_poisons_the_counter_for_good(self, sim) -> None:
        """``step_count`` became ``nan``, and every later reader inherited it."""
        assert sim.send_action({"lift_act": 0.5}, robot_name="arm", n_substeps=NAN)["status"] == "error"
        assert isinstance(sim._world.step_count, int)
        assert not math.isnan(float(sim._world.step_count))
        assert sim.send_action({"lift_act": 0.5}, robot_name="arm", n_substeps=2)["status"] == "success"
        assert sim._world.step_count == 2


# --------------------------------------------------------------------------- #
# Newton and Isaac: the guard precedes every target write, solver and stage   #
# --------------------------------------------------------------------------- #
def _newton_stub() -> tuple[Any, dict[str, list[Any]]]:
    """A Newton stand-in recording whether it was written to or advanced.

    ``_write_targets`` and ``_advance`` are recorders rather than the real
    methods, because what is being measured is that neither is *reached* - a
    stand-in that ran them could only show the end state, not the ordering.
    """
    calls: dict[str, list[Any]] = {"write": [], "advance": []}
    world = SimWorld()
    world.robots["arm"] = SimRobot(name="arm", urdf_path="<inline>", joint_names=["lift"])
    stub: Any = types.SimpleNamespace(
        _world=world,
        _model=types.SimpleNamespace(body_label=["ground"]),
        _lock=threading.RLock(),
        _targets={},
        substeps=1,
        _resolve_single_robot=lambda name: name or "arm",
        _coerce_action=lambda action, robot: (dict(action), None),
        _write_targets=lambda: calls["write"].append(True),
        _advance=lambda n: calls["advance"].append(n),
    )
    # ``send_action`` resolves the keys it will accept through
    # ``robot_action_keys`` (a floating base's free joint is a joint but not a
    # commandable scalar), so the stand-in resolves it through the real
    # implementation rather than restating the rule. Restating it here would put
    # a second copy of the vocabulary in the tree, which is the drift this
    # indirection exists to remove.
    stub.robot_joint_names = lambda name: NewtonSimEngine.robot_joint_names(stub, name)
    stub.robot_action_keys = lambda name: NewtonSimEngine.robot_action_keys(stub, name)
    return stub, calls


def _isaac_stub() -> tuple[Any, dict[str, int]]:
    """An Isaac stand-in counting world ticks, with no stage and no RTX."""
    calls = {"n": 0}
    stub: Any = types.SimpleNamespace(
        _lock=threading.RLock(),
        _world_created=True,
        _config=IsaacConfig(),
        _sim_time=0.0,
        _step_count=0,
        _robots={},
        _action_controllers={},
        _world=types.SimpleNamespace(step=lambda render=False: calls.__setitem__("n", calls["n"] + 1)),
    )
    return stub, calls


class TestNewtonSendAction:
    @pytest.mark.parametrize("count", UNUSABLE_SUBSTEPS, ids=_substep_id)
    def test_an_unusable_count_is_refused_before_any_target_is_written(self, count: Any) -> None:
        stub, calls = _newton_stub()
        result = NewtonSimEngine.send_action(stub, {"lift": 0.5}, robot_name="arm", n_substeps=count)
        assert result["status"] == "error", count
        assert "n_substeps" in _text(result)
        assert calls["write"] == [], f"{count!r} wrote targets it was refused for"
        assert calls["advance"] == []
        assert stub._targets == {}

    @pytest.mark.parametrize(("count", "expected"), USABLE_SUBSTEPS)
    def test_a_usable_count_reaches_advance_already_coerced(self, count: Any, expected: int) -> None:
        """The ``int()`` is the guard's other half: ``_advance`` gets a true ``int``.

        Pre-fix a ``3.0`` reached ``range(max(1, 3.0))`` and raised, and a
        ``2.7`` was added to ``step_count`` verbatim.
        """
        stub, calls = _newton_stub()
        result = NewtonSimEngine.send_action(stub, {"lift": 0.5}, robot_name="arm", n_substeps=count)
        assert result["status"] == "success", _text(result)
        assert calls["advance"] == [expected]
        assert isinstance(calls["advance"][0], int)
        assert calls["write"] == [True]


class TestIsaacSendAction:
    @pytest.mark.parametrize("count", UNUSABLE_SUBSTEPS, ids=_substep_id)
    def test_an_unusable_count_is_refused_before_the_lock_and_any_tick(self, count: Any) -> None:
        stub, calls = _isaac_stub()
        result = IsaacSimulation.send_action(stub, {"lift": 0.5}, robot_name="arm", n_substeps=count)
        assert result["status"] == "error", count
        assert "n_substeps" in _text(result)
        assert calls["n"] == 0, f"{count!r} ticked a world it was refused for"
        assert stub._step_count == 0
        assert stub._sim_time == 0.0


# --------------------------------------------------------------------------- #
# The three backends no longer disagree                                       #
# --------------------------------------------------------------------------- #
class TestTheThreeBackendsAgree:
    """One domain, one refusal - byte-identical, not merely equivalent."""

    @pytest.mark.parametrize("count", UNUSABLE_SUBSTEPS, ids=_substep_id)
    def test_the_refusal_is_the_same_text_on_every_backend(self, sim, count: Any) -> None:
        mujoco_text = _text(sim.send_action({"lift_act": 0.5}, robot_name="arm", n_substeps=count))
        newton_text = _text(
            NewtonSimEngine.send_action(_newton_stub()[0], {"lift": 0.5}, robot_name="arm", n_substeps=count)
        )
        isaac_text = _text(
            IsaacSimulation.send_action(_isaac_stub()[0], {"lift": 0.5}, robot_name="arm", n_substeps=count)
        )
        assert mujoco_text == newton_text == isaac_text

    def test_the_zero_the_backends_used_to_disagree_about(self, sim) -> None:
        """The clearest reason this was not merely an unvalidated input.

        Pre-fix ``n_substeps=0`` advanced one step on MuJoCo and Newton and none
        on Isaac, so the same call meant two different things depending on the
        backend. "Honor 0" would have adopted Isaac's *missing* floor over the
        reference backend's explicit one.
        """
        results = (
            sim.send_action({"lift_act": 0.5}, robot_name="arm", n_substeps=0),
            NewtonSimEngine.send_action(_newton_stub()[0], {"lift": 0.5}, robot_name="arm", n_substeps=0),
            IsaacSimulation.send_action(_isaac_stub()[0], {"lift": 0.5}, robot_name="arm", n_substeps=0),
        )
        assert [r["status"] for r in results] == ["error"] * 3
        assert len({_text(r) for r in results}) == 1
        assert sim._world.step_count == 0


# --------------------------------------------------------------------------- #
# Drift: no send_action surface may skip the guard                            #
# --------------------------------------------------------------------------- #
_BACKEND_MODULES = (
    "strands_robots/simulation/mujoco/simulation.py",
    "strands_robots/simulation/newton/simulation.py",
    "strands_robots/simulation/isaac/simulation.py",
)


def _send_action_defs() -> list[tuple[str, ast.FunctionDef]]:
    """Every concrete ``send_action`` definition in the backend modules."""
    found: list[tuple[str, ast.FunctionDef]] = []
    root = pathlib.Path(__file__).resolve().parents[2]
    for relative in _BACKEND_MODULES:
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "send_action":
                found.append((relative, node))
    return found


class TestNoSendActionSurfaceSkipsTheGuard:
    """A fourth backend, or a rewrite of one of these three, fails here.

    The scan is what makes this the last pass rather than a first: the defect was
    not that one backend forgot the guard, it was that all three did, and each
    diverged differently. A behavioural test proves today's three; the scan
    states the rule.
    """

    def test_the_scan_finds_every_backend(self) -> None:
        """A positive control: an empty scan must not read as a clean one."""
        paths = {relative for relative, _node in _send_action_defs()}
        assert paths == set(_BACKEND_MODULES)

    @pytest.mark.parametrize("relative", _BACKEND_MODULES)
    def test_every_send_action_validates_its_substep_count(self, relative: str) -> None:
        defs = [node for path, node in _send_action_defs() if path == relative]
        assert defs, relative
        for node in defs:
            args = [a.arg for a in node.args.args]
            assert "n_substeps" in args, f"{relative}: send_action lost its n_substeps parameter"
            calls = [
                child
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "positive_whole_number_error"
            ]
            assert calls, f"{relative}: send_action does not call positive_whole_number_error"
            assert any(
                isinstance(arg, ast.Constant) and arg.value == "n_substeps" for call in calls for arg in call.args
            ), f"{relative}: the guard is called but not for n_substeps"

    @pytest.mark.parametrize("relative", _BACKEND_MODULES)
    def test_the_guard_precedes_every_write_and_every_lock(self, relative: str) -> None:
        """Ordering, structurally: a refusal after the write is the failure mode.

        Compared by line number rather than by running the method, because the
        property is "no path can write first" and a behavioural test only ever
        exercises the paths it thought of.
        """
        for path, node in _send_action_defs():
            if path != relative:
                continue
            guard_lines = [
                child.lineno
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "positive_whole_number_error"
            ]
            assert guard_lines, relative
            first_guard = min(guard_lines)
            withs = [child.lineno for child in ast.walk(node) if isinstance(child, ast.With)]
            assert all(first_guard < line for line in withs), (
                f"{relative}: the substep guard runs inside or after a lock, so a refusal "
                f"can arrive once the world is already held"
            )


# --------------------------------------------------------------------------- #
# Boundary                                                                    #
# --------------------------------------------------------------------------- #
class TestNeighbouringSubstepSurfacesStayOutOfScope:
    """Pins of behaviour left unchanged, so the boundary is stated not omitted.

    Replace these when the surfaces they describe are settled, rather than
    deleting them: the scope statement stays useful and simply narrows.
    """

    def test_there_is_still_no_per_call_ceiling_on_a_substep_count(self) -> None:
        """MuJoCo's ``_MAX_STEPS_PER_CALL`` guards ``step`` and not this surface.

        So a count ``step`` refuses is still accepted here, on every backend.
        That is the resource policy tracked as #1871 - a decision about a number,
        not an input domain - and it is not smuggled in with the domain. Asserted
        structurally rather than by running 100_001 physics steps.
        """
        from strands_robots.simulation.mujoco import rendering

        source = inspect.getsource(rendering.RenderingMixin._apply_sim_action)
        assert "_MAX_STEPS_PER_CALL" not in source
        assert "_STEPS_PER_BATCH" not in source
        over = 100_001
        assert positive_whole_number_error(over, "n_substeps", "send_action") is None

    def test_the_private_floors_are_retained_and_are_now_unreachable(self) -> None:
        """Both are defensive no-ops once the public surfaces refuse.

        Kept rather than deleted so a future internal caller cannot advance a
        negative count, and pinned so the pair stays legible: removing a floor
        and removing the guard are not independently safe.
        """
        from strands_robots.simulation.mujoco import rendering

        mujoco_source = inspect.getsource(rendering.RenderingMixin._apply_sim_action)
        assert "max(1, n_substeps)" in mujoco_source
        newton_source = textwrap.dedent(inspect.getsource(NewtonSimEngine._advance))
        assert "max(1, n_steps)" in newton_source

    def test_the_magnitude_boundary_is_the_float_edge_not_a_resource_ceiling(self) -> None:
        """``1e300`` is accepted and ``10**400`` refused, as on every caller of this guard.

        Recorded because the two read alike and are not: refusing the second is
        the float64 edge the domain already had, at which the guard used to raise
        ``OverflowError`` rather than answer. Choosing a ceiling that would refuse
        the first is #1871.
        """
        assert positive_whole_number_error(1e300, "n_substeps", "send_action") is None
        assert positive_whole_number_error(BEYOND_FLOAT_RANGE, "n_substeps", "send_action") is not None

    def test_the_narrower_candidate_domain_was_rejected_for_a_measured_reason(self) -> None:
        """``positive_count_error`` would have regressed the reference backend.

        It admits only a true ``int``, so it refuses every NumPy and integral-float
        row of ``USABLE_SUBSTEPS`` - counts MuJoCo honors today and ``step``
        honors by documented design. Pinned so the choice between the two guards
        is a measurement a later reader can re-run, not a preference.
        """
        for count, _expected in USABLE_SUBSTEPS:
            assert positive_whole_number_error(count, "n_substeps", "send_action") is None
        refused_by_the_narrower_guard = [
            count for count, _expected in USABLE_SUBSTEPS if positive_count_error(count, "n", "ctx") is not None
        ]
        assert refused_by_the_narrower_guard == [3.0, np.int64(3), np.uint8(2), np.float64(4.0)]

    def test_the_rl_env_still_raises_rather_than_returning_a_structured_error(self) -> None:
        """``SimEnv`` refuses the same parameter with a ``ValueError`` of its own.

        Left alone: it is a gym-style constructor rather than an agent tool, so a
        raise is its documented channel and it has no ``{status, content}`` to
        return. Recorded because the two refusals now differ in wording for the
        same parameter name, which is a real inconsistency and a deliberate one.
        """
        from strands_robots.training.rl import env as rl_env

        source = inspect.getsource(rl_env)
        assert "n_substeps must be >= 1" in source
        assert "must be a positive whole number" not in source
