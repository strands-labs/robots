"""Isaac articulation read/write surfaces - the fallback limit source and every I/O failure.

:mod:`strands_robots.simulation.isaac.motion_primitives` lists what its adapter
half owns, and the articulation layer runs through three of those surfaces:
resolving the gripper / wrist DOFs against the articulation's own limits,
asserting PD position targets on them, and reading the articulation's own base
pose - the frame ``move_to`` maps every world-frame target through. Each is
documented to tolerate more than one surface and to answer a surface that
cannot be read *loudly*:

  * :meth:`_articulation_dof_limits` documents TWO sources - the
    ``dof_properties`` structured array (authoritative, honoring ``hasLimits``
    when that field is present) and the view-shaped ``get_dof_limits()``
    fallback - and reports ``None`` for a DOF whose bounds are absent,
    non-finite or degenerate, so the caller refuses instead of mapping
    ``open``/``close`` onto a range that does not exist;
  * :meth:`_read_joint_positions` documents the plain-array and torch-tensor
    surfaces and returns ``None`` for "could not be read", which callers "must
    answer loudly, never by substituting zeros";
  * :meth:`_apply_position_targets` documents a narrow exception set and turns
    a failed write into a structured error;
  * :meth:`_articulation_base_pose` documents the plain-array and torch-tensor
    surfaces, seven "could not be read" routes and a normalized quaternion, and
    returns ``None`` for a pose callers "must answer loudly, never by
    substituting an origin base (a wrong base makes every world-frame target
    silently wrong)".

``tests/simulation/isaac/test_motion_primitives.py`` pins the contracts its own
docstring enumerates - resolution, convergence, timeout and abort - and its
``_FakeArticulation`` always supplies ``dof_properties``, always answers a read,
and carries no ``get_world_pose`` at all. So the *authoritative* limit source is
exercised and the *fallback* source is not, every read/write failure report is
unreached, and of the base-pose readback's seven routes
``tests/simulation/isaac/test_move_to_ik.py`` drives one - a pose that answers
``None``. This module drives the other source and every failure arm, on the
plain-data surfaces plus through the primitives that consume them.

Like its sibling this needs no NVIDIA Isaac Sim: the articulation, the world and
the one lazily imported ``ArticulationAction`` type are faked, so every cell
runs on any host.

Out of scope, and each its own contract from the adapter's ownership list: the
world/robot resolution guards and the Kit-pump threading marshal.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import textwrap
import types
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.isaac.motion_primitives import IsaacMotionPrimitivesMixin
from strands_robots.simulation.isaac.simulation import IsaacSimulation, _RobotState

from .test_motion_primitives import (  # noqa: F401 - fake_articulation_action is an autouse fixture
    ARM_JOINTS,
    ARM_LIMITS,
    _FakeArticulation,
    _FakeWorld,
    _json_block,
    _make_sim,
    fake_articulation_action,
)

# ---------------------------------------------------------------------------
# Articulations whose limit SOURCE, or whose I/O, is configurable.
# ---------------------------------------------------------------------------


class _LimitSourceArticulation:
    """Articulation exposing a chosen one of the two documented limit surfaces.

    A wrapper rather than a ``_FakeArticulation`` subclass: what varies here
    *is* an attribute the fake's own ``__init__`` writes, so a subclass can only
    vary it by deleting and re-assigning what its base just set - which reads
    exactly like an accidental shadow of an inherited attribute, and is one.
    Wrapping states each surface once: ``dof_properties`` and
    ``get_dof_limits`` exist on the wrapper only for the cases that say they
    do, and every other articulation call delegates to the wrapped fake.

    ``source`` selects which surface exists:

      ``"fallback"``         only ``get_dof_limits()``;
      ``"props_unreadable"`` a ``dof_properties`` that is not a structured
                             array (so ``props["lower"]`` raises) plus the
                             fallback, i.e. the authoritative source present
                             but unreadable;
      ``"no_has_limits"``    ``dof_properties`` carrying ``lower``/``upper``
                             but no ``hasLimits`` field;
      ``"raising_fallback"`` no ``dof_properties`` and a fallback that raises;
      ``"none"``             neither surface.

    ``view_shaped`` returns the ``(num_envs, num_dofs, 2)`` shape an Isaac
    ``ArticulationView`` reports rather than a plain ``(num_dofs, 2)``;
    ``as_tensor`` wraps it in the ``.cpu().numpy()`` surface a GPU pipeline
    hands back; ``rows`` truncates the table so it is shorter than the
    articulation.
    """

    def __init__(
        self,
        joint_names: list[str],
        limits: list[tuple[float, float] | None],
        *,
        source: str,
        view_shaped: bool = False,
        as_tensor: bool = False,
        rows: int | None = None,
        **kwargs: Any,
    ):
        spans = [s for s in limits if s is not None]
        assert len(spans) == len(limits), "a fallback table cannot express hasLimits=False"
        self._inner = _FakeArticulation(joint_names, limits, **kwargs)
        self._raises = source == "raising_fallback"

        if source != "none":
            table: Any = np.array(spans[: rows if rows is not None else len(spans)], dtype=np.float64)
            if view_shaped:
                table = np.array([table])
            self._table = _TorchTensor(table) if as_tensor else table
            # Bound as an instance attribute, so ``source="none"`` exposes no
            # such surface at all rather than one that refuses when called -
            # the adapter probes for the fallback with getattr.
            self.get_dof_limits = self._fallback_limits

        if source == "no_has_limits":
            fields = np.zeros(len(joint_names), dtype=[("lower", "f8"), ("upper", "f8")])
            for i, (lo, hi) in enumerate(spans):
                fields["lower"][i], fields["upper"][i] = lo, hi
            self.dof_properties = fields
        elif source == "props_unreadable":
            self.dof_properties = np.zeros(len(joint_names))  # not a structured array
        # Every other source leaves the authoritative surface absent: see
        # ``__getattr__``, which does not resolve it on the wrapped fake.

    def _fallback_limits(self):
        if self._raises:
            raise RuntimeError("articulation view was torn down")
        return self._table

    def __getattr__(self, name):
        # Reached only for names this wrapper did not set. The two limit
        # surfaces are what this module varies, so an absent one stays absent
        # instead of resolving on the wrapped fake; ``_inner`` is listed so a
        # missing wrapper state reports itself rather than recursing here.
        if name in ("dof_properties", "get_dof_limits", "_inner"):
            raise AttributeError(name)
        return getattr(self._inner, name)


class _TorchTensor:
    """The ``.cpu().numpy()`` surface a GPU articulation hands back."""

    def __init__(self, arr):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FailingIoArticulation(_FakeArticulation):
    """Articulation whose position read or target write fails.

    ``read_fails_after=0`` raises on every read; ``=1`` lets a setup read
    through and fails from the first in-loop read (the mid-run abort).
    ``read_returns_none`` models the surface that answers ``None`` rather than
    raising. ``apply_fails`` raises from ``apply_action``, the write failure.
    """

    def __init__(
        self,
        joint_names: list[str],
        limits: list[tuple[float, float] | None],
        *,
        read_fails_after: int | None = None,
        read_returns_none: bool = False,
        apply_fails: bool = False,
        **kwargs: Any,
    ):
        super().__init__(joint_names, limits, **kwargs)
        self.read_fails_after = read_fails_after
        self.read_returns_none = read_returns_none
        self.apply_fails = apply_fails
        self.reads = 0

    def get_joint_positions(self):
        self.reads += 1
        if self.read_returns_none:
            return None
        if self.read_fails_after is not None and self.reads > self.read_fails_after:
            raise RuntimeError("articulation was torn down")
        return super().get_joint_positions()

    def apply_action(self, action) -> None:
        if self.apply_fails:
            raise RuntimeError("physics view is invalid")
        super().apply_action(action)


def _sim_with(art, joint_names: list[str] = ARM_JOINTS, robot_name: str = "arm", data_config: str | None = None):
    """An ``IsaacSimulation`` whose world and robot state both hold *art*."""
    sim = IsaacSimulation()
    sim._world = _FakeWorld(art)
    sim._world_created = True
    sim._robots[robot_name] = _RobotState(
        name=robot_name,
        prim_path=f"/World/Robots/{robot_name}",
        joint_names=list(joint_names),
        articulation=art,
        data_config=data_config,
    )
    return sim


REAL_LIMITS: list[tuple[float, float]] = [s for s in ARM_LIMITS if s is not None]
_limits = IsaacMotionPrimitivesMixin._articulation_dof_limits
_read = IsaacMotionPrimitivesMixin._read_joint_positions
_base_pose = IsaacMotionPrimitivesMixin._articulation_base_pose


# ---------------------------------------------------------------------------
# The two documented limit sources.
# ---------------------------------------------------------------------------


class TestLimitsResolveFromEitherDocumentedSource:
    """``dof_properties`` is authoritative; ``get_dof_limits()`` is the fallback."""

    def test_the_authoritative_source_is_the_reference(self):
        art = _FakeArticulation(ARM_JOINTS, REAL_LIMITS)
        assert _limits(art, len(ARM_JOINTS)) == REAL_LIMITS

    @pytest.mark.parametrize(
        ("view_shaped", "as_tensor"),
        [(False, False), (True, False), (False, True), (True, True)],
        ids=["plain-(n,2)", "view-(1,n,2)", "tensor-(n,2)", "tensor-(1,n,2)"],
    )
    def test_the_fallback_source_reports_the_same_spans(self, view_shaped, as_tensor):
        # An articulation with no dof_properties at all: the fallback is the
        # only surface, in each shape Isaac reports it in.
        art = _LimitSourceArticulation(
            ARM_JOINTS, REAL_LIMITS, source="fallback", view_shaped=view_shaped, as_tensor=as_tensor
        )
        # Without this the spans below are equally true of the authoritative
        # source, so the case would pass while measuring the wrong surface.
        assert not hasattr(art, "dof_properties")
        assert _limits(art, len(ARM_JOINTS)) == REAL_LIMITS

    def test_an_unreadable_authoritative_source_falls_through_to_the_fallback(self):
        art = _LimitSourceArticulation(ARM_JOINTS, REAL_LIMITS, source="props_unreadable")
        assert art.dof_properties is not None  # present, just not a structured array
        assert _limits(art, len(ARM_JOINTS)) == REAL_LIMITS

    def test_properties_without_a_has_limits_field_are_still_read(self):
        art = _LimitSourceArticulation(ARM_JOINTS, REAL_LIMITS, source="no_has_limits")
        assert "hasLimits" not in (art.dof_properties.dtype.names or ())
        assert _limits(art, len(ARM_JOINTS)) == REAL_LIMITS


# ---------------------------------------------------------------------------
# Every "no usable bounds" outcome.
# ---------------------------------------------------------------------------


class TestLimitsReportNoneWhenNoBoundsAreUsable:
    """A DOF whose bounds are absent, non-finite or degenerate reports ``None``."""

    def test_neither_source_present(self):
        art = _LimitSourceArticulation(ARM_JOINTS, REAL_LIMITS, source="none")
        assert not hasattr(art, "dof_properties")
        assert not hasattr(art, "get_dof_limits")
        assert _limits(art, len(ARM_JOINTS)) == [None] * len(ARM_JOINTS)

    def test_a_fallback_that_raises(self):
        art = _LimitSourceArticulation(ARM_JOINTS, REAL_LIMITS, source="raising_fallback")
        assert _limits(art, len(ARM_JOINTS)) == [None] * len(ARM_JOINTS)

    def test_a_table_shorter_than_the_articulation(self):
        art = _LimitSourceArticulation(ARM_JOINTS, REAL_LIMITS, source="fallback", rows=2)
        assert _limits(art, len(ARM_JOINTS)) == [REAL_LIMITS[0], REAL_LIMITS[1], None, None, None]

    @pytest.mark.parametrize(
        "span",
        [(0.0, float("inf")), (float("-inf"), 0.0), (float("nan"), 1.0), (0.0, float("nan"))],
        ids=["upper-inf", "lower-inf", "lower-nan", "upper-nan"],
    )
    def test_a_non_finite_bound(self, span):
        art = _FakeArticulation(ARM_JOINTS, [span, *REAL_LIMITS[1:]])
        assert _limits(art, len(ARM_JOINTS))[0] is None

    @pytest.mark.parametrize("span", [(1.0, 1.0), (2.0, 0.5)], ids=["equal", "inverted"])
    def test_a_degenerate_bound(self, span):
        art = _FakeArticulation(ARM_JOINTS, [span, *REAL_LIMITS[1:]])
        assert _limits(art, len(ARM_JOINTS))[0] is None


# ---------------------------------------------------------------------------
# The fallback source is load-bearing, not merely parsed.
# ---------------------------------------------------------------------------


class TestThePrimitivesDriveThroughTheFallbackSource:
    """An articulation reporting limits only through ``get_dof_limits()`` still works."""

    def test_set_gripper_maps_open_onto_a_fallback_sourced_span(self):
        art = _LimitSourceArticulation(ARM_JOINTS, REAL_LIMITS, source="fallback", view_shaped=True)
        sim = _sim_with(art)
        result = sim.set_gripper(robot_name="arm", state="open", steps=6)
        assert result["status"] == "success", result
        # ``targets`` is keyed by the joint name; "open" is the HIGH end of
        # the fallback-sourced span.
        assert _json_block(result)["targets"]["jaw"] == pytest.approx(REAL_LIMITS[4][1])

    def test_rotate_wrist_bounds_the_target_against_a_fallback_sourced_span(self):
        art = _LimitSourceArticulation(ARM_JOINTS, REAL_LIMITS, source="fallback")
        sim = _sim_with(art)
        outside = REAL_LIMITS[3][1] + 1.0
        result = sim.rotate_wrist(robot_name="arm", target_yaw=outside)
        assert result["status"] == "error"
        assert "outside joint" in result["content"][0]["text"]
        assert art.applied == []  # refused before any write


# ---------------------------------------------------------------------------
# The plain-data read / write surfaces.
# ---------------------------------------------------------------------------


class TestReadJointPositions:
    """The documented surfaces, and ``None`` for one that cannot be read."""

    def test_a_torch_tensor_is_read_through_cpu_numpy(self):
        art = _FakeArticulation(ARM_JOINTS, REAL_LIMITS)
        art.get_joint_positions = lambda: _TorchTensor(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        assert _read(art).tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])

    @pytest.mark.parametrize(
        "exc",
        [RuntimeError("torn down"), ValueError("bad shape"), AttributeError("surface drift"), TypeError("wrong type")],
        ids=["RuntimeError", "ValueError", "AttributeError", "TypeError"],
    )
    def test_a_raising_read_reports_none_rather_than_zeros(self, exc):
        art = _FakeArticulation(ARM_JOINTS, REAL_LIMITS)

        def _raise():
            raise exc

        art.get_joint_positions = _raise
        assert _read(art) is None

    def test_a_read_that_answers_none_reports_none(self):
        art = _FailingIoArticulation(ARM_JOINTS, REAL_LIMITS, read_returns_none=True)
        assert _read(art) is None


class TestApplyPositionTargets:
    """A failed write is a structured error naming the action and the robot."""

    def test_a_successful_write_commands_only_the_indexed_dofs(self):
        art = _FakeArticulation(ARM_JOINTS, REAL_LIMITS)
        sim = _sim_with(art)
        assert sim._apply_position_targets("set_gripper", "arm", art, {4: 1.5}) is None
        assert np.asarray(art.applied[-1].joint_indices).tolist() == [4]

    def test_a_raising_write_is_reported_not_raised(self):
        art = _FailingIoArticulation(ARM_JOINTS, REAL_LIMITS, apply_fails=True)
        sim = _sim_with(art)
        error = sim._apply_position_targets("set_gripper", "arm", art, {4: 1.5})
        assert error is not None and error["status"] == "error"
        text = error["content"][0]["text"]
        assert "set_gripper" in text and "'arm'" in text and "physics view is invalid" in text


# ---------------------------------------------------------------------------
# The base-pose readback: every documented "could not be read" route.
# ---------------------------------------------------------------------------

GOOD_BASE_POS = np.array([0.4, -0.2, 0.1])
GOOD_BASE_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
# Inside the workspace sanity box measured from GOOD_BASE_POS.
WORLD_TARGET = [0.45, -0.2, 0.25]


def _pose_raising(exc: BaseException) -> Any:
    """A ``get_world_pose`` that raises - the torn-down articulation."""

    def _get_world_pose() -> Any:
        raise exc

    return _get_world_pose


def _pose_returning(value: Any) -> Any:
    """A ``get_world_pose`` that answers *value*."""

    def _get_world_pose() -> Any:
        return value

    return _get_world_pose


# One entry per documented "could not be read" route. The count is pinned
# against the readback's own arms below, so a route added to the production
# helper fails this module until it is driven here.
_UNREADABLE_BASE_POSES: dict[str, Any] = {
    "read-raises": _pose_raising(RuntimeError("physics view is invalid")),
    "pose-is-None": _pose_returning(None),
    "pose-does-not-unpack": _pose_returning((GOOD_BASE_POS, GOOD_BASE_QUAT, GOOD_BASE_QUAT)),
    "a-component-is-None": _pose_returning((GOOD_BASE_POS, None)),
    "wrong-component-count": _pose_returning((GOOD_BASE_POS[:2], GOOD_BASE_QUAT)),
    "a-component-is-non-finite": _pose_returning((GOOD_BASE_POS, np.array([np.inf, 0.0, 0.0, 0.0]))),
    "quaternion-has-no-direction": _pose_returning((GOOD_BASE_POS, np.zeros(4))),
}
_ROUTES = sorted(_UNREADABLE_BASE_POSES)


class _BasePoseArticulation:
    """Articulation whose base-pose surface is stated here, wrapping the fake.

    Composed rather than subclassed for the reason
    :class:`TestTheLimitSourceFakeComposesRatherThanInherits` records, and for
    one more: the joint-space fake carries no ``get_world_pose`` at all, so
    assigning one onto it would state a surface its class does not declare.
    Wrapping states the surface once and leaves the base the sole owner of
    everything it does build.
    """

    def __init__(self, joint_names: list[str], limits: list[tuple[float, float] | None], get_world_pose: Any):
        self._inner = _FakeArticulation(joint_names, limits)
        self.get_world_pose = get_world_pose

    def __getattr__(self, name: str) -> Any:
        # ``_inner`` is listed so a missing wrapper state reports itself
        # rather than recursing here.
        if name in ("get_world_pose", "_inner"):
            raise AttributeError(name)
        return getattr(self._inner, name)


def _sim_with_base(pose_fn: Any) -> tuple[Any, Any]:
    """A sim whose articulation answers *pose_fn* and whose robot has no ``data_config``."""
    art = _BasePoseArticulation(ARM_JOINTS, ARM_LIMITS, pose_fn)
    return _sim_with(art), art


class TestTheBasePoseReadbackAnswersEveryUnreadableRoute:
    """``None`` for every route, the documented surfaces, and a normalized quaternion."""

    @pytest.mark.parametrize("route", _ROUTES)
    def test_an_unreadable_pose_reports_none(self, route):
        art = types.SimpleNamespace(get_world_pose=_UNREADABLE_BASE_POSES[route])
        assert _base_pose(art) is None

    @pytest.mark.parametrize(
        "exc",
        [RuntimeError("torn down"), ValueError("bad shape"), AttributeError("surface drift"), TypeError("wrong type")],
        ids=["RuntimeError", "ValueError", "AttributeError", "TypeError"],
    )
    def test_every_exception_a_torn_down_articulation_raises_reports_none(self, exc):
        art = types.SimpleNamespace(get_world_pose=_pose_raising(exc))
        assert _base_pose(art) is None

    def test_a_torch_tensor_pose_is_read_through_cpu_numpy(self):
        pose = (_TorchTensor(GOOD_BASE_POS), _TorchTensor(GOOD_BASE_QUAT))
        read = _base_pose(types.SimpleNamespace(get_world_pose=_pose_returning(pose)))
        assert read is not None
        pos, quat = read
        assert pos.tolist() == pytest.approx(GOOD_BASE_POS.tolist())
        assert quat.tolist() == pytest.approx(GOOD_BASE_QUAT.tolist())

    def test_a_non_unit_quaternion_is_normalized_without_turning_the_base(self):
        pose = (GOOD_BASE_POS, GOOD_BASE_QUAT * 7.0)
        read = _base_pose(types.SimpleNamespace(get_world_pose=_pose_returning(pose)))
        assert read is not None
        _, quat = read
        assert float(np.linalg.norm(quat)) == pytest.approx(1.0)
        assert quat.tolist() == pytest.approx(GOOD_BASE_QUAT.tolist())

    def test_the_base_pose_fake_delegates_every_other_surface(self):
        """The write log the refusal cases read is the wrapped fake's own."""
        art = _BasePoseArticulation(ARM_JOINTS, ARM_LIMITS, _pose_returning(None))
        assert not isinstance(art, _FakeArticulation)
        assert art.applied is art._inner.applied
        assert np.asarray(art.positions).tolist() == np.asarray(art._inner.positions).tolist()

    def test_every_route_the_readback_documents_has_a_case(self):
        """A route added to the readback fails here until it is driven above."""
        arms = [
            node
            for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(_base_pose))))
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant) and node.value.value is None
        ]
        assert arms, "no `return None` arm found - the scan did not resolve the readback"
        assert len(arms) == len(_UNREADABLE_BASE_POSES), (
            f"the readback has {len(arms)} `could not be read` routes but {len(_UNREADABLE_BASE_POSES)} are driven here"
        )


# ---------------------------------------------------------------------------
# The consumers answer a failed read or write loudly.
# ---------------------------------------------------------------------------


class TestThePrimitivesReportAFailedReadOrWrite:
    """Never a zero-valued success, and never a raise through the tool surface."""

    def test_set_gripper_reports_a_write_that_failed_mid_drive(self):
        art = _FailingIoArticulation(ARM_JOINTS, REAL_LIMITS, apply_fails=True)
        result = _sim_with(art).set_gripper(robot_name="arm", state="close", steps=4)
        assert result["status"] == "error"
        assert "failed to set joint position targets" in result["content"][0]["text"]

    def test_set_gripper_reports_an_unverified_final_state(self):
        # The drive runs; only the readback the success payload promises is gone.
        art = _FailingIoArticulation(ARM_JOINTS, REAL_LIMITS, read_fails_after=0)
        result = _sim_with(art).set_gripper(robot_name="arm", state="close", steps=4)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "unverified" in text and "could not read" in text
        assert art.applied, "the drive did run - the failure is the readback"

    def test_rotate_wrist_reports_a_read_that_failed_before_the_servo(self):
        art = _FailingIoArticulation(ARM_JOINTS, REAL_LIMITS, read_fails_after=0)
        result = _sim_with(art).rotate_wrist(robot_name="arm", target_yaw=0.5)
        assert result["status"] == "error"
        assert "did not report a usable joint-position vector" in result["content"][0]["text"]
        assert art.applied == []  # refused before any write

    def test_rotate_wrist_reports_a_write_that_failed_mid_servo(self):
        art = _FailingIoArticulation(ARM_JOINTS, REAL_LIMITS, apply_fails=True)
        result = _sim_with(art).rotate_wrist(robot_name="arm", target_yaw=0.5)
        assert result["status"] == "error"
        assert "failed to set joint position targets" in result["content"][0]["text"]

    def test_rotate_wrist_aborts_when_the_read_stops_working_mid_servo(self):
        # The setup read succeeds; the first in-loop read does not.
        art = _FailingIoArticulation(ARM_JOINTS, REAL_LIMITS, read_fails_after=1)
        result = _sim_with(art).rotate_wrist(robot_name="arm", target_yaw=0.5)
        assert result["status"] == "error"
        assert "mid-run; aborting" in result["content"][0]["text"]
        assert art.reads == 2  # setup, then the in-loop read that failed

    def test_rotate_wrist_propagates_the_gripper_resolution_error(self):
        # rotate_wrist must exclude the gripper DOFs, so a registry entry
        # that resolves to no joint is the same loud error set_gripper
        # answers with - the wrist is never guessed from a classification
        # that failed. The shipped so101 metadata names actuator "6", which
        # this generic vocabulary does not have, so no patching is needed.
        sim, art = _make_sim(data_config="so101")
        result = sim.rotate_wrist(robot_name="arm", target_yaw=0.5)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "none match a joint on the articulation" in text
        assert "stale for this robot" in text
        assert art.applied == []


class TestMoveToRefusesAnUnreadableBaseRatherThanSubstitutingTheOrigin:
    """The readback's ``None`` reaches the caller as a refusal that commands nothing.

    The robot deliberately carries no ``data_config``, which is the *next*
    thing ``move_to`` needs after the base pose. A run that substituted an
    origin base would carry on and fail there instead, so "``data_config`` is
    not mentioned" is what separates a refusal from a substitution - the same
    discriminator ``test_move_to_ik.py`` uses to prove the sanity box measures
    from the base rather than the origin.
    """

    def test_a_readable_base_carries_on_past_the_readback(self):
        sim, _ = _sim_with_base(_pose_returning((GOOD_BASE_POS, GOOD_BASE_QUAT)))
        result = sim.move_to(robot_name="arm", position=WORLD_TARGET)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "data_config" in text
        assert "base pose" not in text

    @pytest.mark.parametrize("route", _ROUTES)
    def test_an_unreadable_base_is_refused_and_commands_nothing(self, route):
        sim, art = _sim_with_base(_UNREADABLE_BASE_POSES[route])
        result = sim.move_to(robot_name="arm", position=WORLD_TARGET)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "could not read the articulation base pose" in text
        assert "data_config" not in text
        assert art.applied == []


# ---------------------------------------------------------------------------
# The limit-source fake composes the articulation rather than inheriting it.
# ---------------------------------------------------------------------------


def _self_assigned(cls: ast.ClassDef) -> dict[str, list[int]]:
    """``{attribute: [line, ...]}`` for every ``self.<attr> = ...`` in *cls*."""
    found: dict[str, list[int]] = {}
    for node in ast.walk(cls):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                found.setdefault(target.attr, []).append(node.lineno)
    return found


def _classes(path: pathlib.Path) -> dict[str, ast.ClassDef]:
    return {n.name: n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.ClassDef)}


def _base_owned_assignments(module: pathlib.Path, base_module: pathlib.Path) -> dict[str, list[tuple[int, str]]]:
    """``{subclass: [(line, attribute), ...]}`` per class in *module* based in *base_module*.

    A key with an empty list is a subclass assigning nothing its base assigns;
    the key set is what proves the base was resolved at all.
    """
    bases = _classes(base_module)
    out: dict[str, list[tuple[int, str]]] = {}
    for name, cls in _classes(module).items():
        inherited: set[str] = set()
        resolved = False
        for base in cls.bases:
            if isinstance(base, ast.Name) and base.id in bases:
                resolved = True
                inherited |= set(_self_assigned(bases[base.id]))
        if not resolved:
            continue
        out[name] = sorted(
            (line, attr) for attr, lines in _self_assigned(cls).items() if attr in inherited for line in lines
        )
    return out


class TestTheLimitSourceFakeComposesRatherThanInherits:
    """What this module varies is state ``_FakeArticulation``'s own ``__init__`` writes.

    A subclass can only vary that by deleting and re-assigning what its base
    just set, which ties the fake to *how* the base stores the attribute - an
    instance-dict entry it can pop - rather than to what the articulation
    exposes. Exposure is all :meth:`_articulation_dof_limits` can see: it probes
    both limit surfaces with ``getattr``. Composing states each surface once and
    leaves the base the sole owner of what it builds.
    """

    def test_the_limit_source_fake_does_not_inherit_the_fake_it_wraps(self):
        assert not issubclass(_LimitSourceArticulation, _FakeArticulation)
        art = _LimitSourceArticulation(ARM_JOINTS, REAL_LIMITS, source="fallback")
        assert isinstance(art._inner, _FakeArticulation)

    def test_every_other_articulation_surface_delegates_to_the_wrapped_fake(self):
        """Only the two limit surfaces are stated here; the rest is the fake's."""
        art = _LimitSourceArticulation(ARM_JOINTS, REAL_LIMITS, source="fallback")
        assert art.servo_rate == art._inner.servo_rate
        assert art.applied is art._inner.applied  # the write log the cases above read
        assert np.asarray(art.positions).tolist() == np.asarray(art._inner.positions).tolist()

    def test_a_fake_that_does_subclass_assigns_nothing_its_base_builds(self):
        found = _base_owned_assignments(pathlib.Path(__file__), pathlib.Path(inspect.getfile(_FakeArticulation)))
        assert set(found) == {"_FailingIoArticulation"}, f"unexpected subclasses of the fake: {sorted(found)}"
        offenders = {name: hits for name, hits in found.items() if hits}
        assert offenders == {}, f"these assignments overwrite state _FakeArticulation owns: {offenders}"

    def test_the_scan_detects_a_planted_overwrite(self, tmp_path):
        planted = tmp_path / "planted.py"
        planted.write_text(
            "class _Planted(_FakeArticulation):\n    def __init__(self):\n        self.dof_properties = None\n"
        )
        found = _base_owned_assignments(planted, pathlib.Path(inspect.getfile(_FakeArticulation)))
        assert found == {"_Planted": [(3, "dof_properties")]}
