"""A caller value whose length cannot be read is reported, never raised.

Every validator that accepts a vector first asks "how many components is this?".
The obvious spelling - ``hasattr(value, "__len__")`` followed by ``len(value)`` -
is unsafe for a value class this library receives routinely: a 0-d numpy array
(``np.array(0.5)``, the result of a reduction such as ``np.mean(...)``) and a 0-d
torch tensor both *declare* ``__len__`` and then raise from it. The ``hasattr``
probe passes and the ``len()`` call escapes with a bare ``len() of unsized
object`` naming neither the parameter nor the method.

That escape matters because the surfaces doing the probing all publish a
no-raise contract: the MuJoCo agent-tool router returns a structured error for
every rejected parameter, :meth:`SimEngine.get_world_point` documents its
structural checks as being there "to keep the never-raises envelope", and
:func:`strands_robots.rendering.video.mjpeg_frames` documents ``ValueError`` with
an actionable message as its only failure mode for a malformed ``size``.

Not every affected surface publishes an envelope, though, and the ``policies/``
sites added here fail a second way: they contradict *themselves*. ``MockPolicy``
already documents ``else 6`` for a state that carries no width, and
``WBCPolicy._read_vec`` already returns ``None`` for every value it cannot use -
a 0-d array is exactly that value, and it was the one spelling that never
reached the branch written for it.

:func:`strands_robots.utils.sequence_length` is the single owner of the rule -
it answers "no readable length" for a 0-d array and for a plain scalar alike -
and these tests pin every surface that reads a caller-supplied length through
it, plus the accepted values that must keep working.

Being the single owner turned out to be two claims, and both needed work
(#1888). The owner reported only a ``TypeError`` ``__len__`` as "no length",
which is not the superset it was documented as: ``len()`` converts whatever
``__len__`` returns into an index, so a negative length is a ``ValueError`` and
one past ``sys.maxsize`` an ``OverflowError`` - raised by CPython, with the value
raising nothing at all and its ``__len__`` returning an ordinary ``int``. And
``pose_vector_error`` answered the same question a second way, with its own
``except TypeError``, so the rule had two implementations and the duplicate
carried the same gap. Every unreadable length now reports as ``None`` from one
probe, which is what the guards' callers document, and the tests below pin that
domain at the owner rather than at each of its readers.

"Every surface that reads a caller-supplied length" turned out to exclude the one
that reads no length at all. :func:`~strands_robots.utils.finite_vector_error` is
the verdict half of the shared component read - it returns a message rather than
the floats, so its callers count the components themselves, by reading the value
again. A value with no readable length cannot be read twice, and the read behind
the verdict consumes it, so the caller's own count saw an empty vector. The guard
asked nothing, and both of its live call sites reported the consequence instead
of the cause: ``add_object`` described a three-component extent as ``got 0
(size=[])`` and ``patch_scene_mjcf`` carried ``object of type 'generator' has no
len()`` into its envelope. The last two classes below pin that probe and the two
surfaces, and their values are the other shape of "unreadable" - a value that
reads cleanly and only then cannot be counted, rather than one the read never
reaches.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.mock import MockPolicy
from strands_robots.policies.wbc.policy import WBCPolicy
from strands_robots.rendering.video import mjpeg_frames
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine
from strands_robots.utils import (
    coerce_rgba,
    coerce_size_vector,
    finite_vector_error,
    pose_vector_error,
    sequence_length,
)

# A 0-d array: declares ``__len__``, raises from it, holds exactly one scalar.
UNSIZED = np.array(0.5)


# The three ways a length is unreadable *without* a ``TypeError``. Each carries a
# ``__getitem__`` as well, which is the 0-d array's own shape (``simulation/
# base.py`` notes such a value "declares ``__len__`` and ``__getitem__`` but
# raises from ``len()``") and is what the call sites gating on ``__getitem__``
# before they probe a length require to be reached at all.
class _NegativeLength:
    """``len()`` raises ``ValueError``: a length may not be negative.

    Nothing here raises. ``__len__`` returns an ordinary ``int`` and CPython
    refuses to convert it, which is why a computed length - ``self._end -
    self._start`` on a proxy whose window is inverted - reaches this row without
    anything being hostile.
    """

    def __len__(self) -> int:
        return -1

    def __getitem__(self, index: int) -> float:
        return 0.5


class _OversizedLength:
    """``len()`` raises ``OverflowError``: a length must fit in an index."""

    def __len__(self) -> int:
        return sys.maxsize + 1

    def __getitem__(self, index: int) -> float:
        return 0.5


class _RefusingLength:
    """``__len__`` refuses on its own account, as any third-party type may.

    The #1873 argument applied to ``__len__``: a method like any other, owing a
    caller nothing, on the one path whose purpose is to answer instead of raise.
    """

    def __len__(self) -> int:
        raise RuntimeError("length unavailable")

    def __getitem__(self, index: int) -> float:
        return 0.5


# ``UNSIZED`` first: the value the rule was written for, so a parametrization
# over this tuple is a widening of an existing pin rather than a separate one.
UNREADABLE_LENGTHS = (UNSIZED, _NegativeLength(), _OversizedLength(), _RefusingLength())
UNREADABLE_IDS = ("zero_d_array", "negative_len", "oversized_len", "refusing_len")


# A minimal actuated arm: enough for send_action to reach the action coercion,
# with no downloaded asset and no rendering (the model is compiled, never drawn).
_ARM_MJCF = """<mujoco model="unsized_arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <geom type="box" size="0.05 0.05 0.05"/>
      <body name="link" pos="0 0 0.05">
        <joint name="pan" type="hinge" axis="0 0 1" range="-2 2" limited="true" damping="4"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.03"/>
      </body>
    </body>
  </worldbody>
  <actuator><position name="pan_act" joint="pan" kp="50" ctrlrange="-2 2"/></actuator>
</mujoco>
"""


def _call(target: Any, **kwargs: Any) -> Any:
    """Invoke ``target`` with values whose *shape* is the subject of the test.

    These parameters are annotated for the shapes a caller *should* pass
    (``pixels`` as ``Sequence[Sequence[SupportsFloat]]``, ``size`` as a
    ``tuple[int, int]``), while the point of this module is what happens for the
    shapes a caller *does* pass - a 0-d NumPy array, and a correctly sized NumPy
    array that those annotations do not describe either. Routing every such call
    through one ``**kwargs: Any`` funnel states that intent once instead of
    scattering per-call type suppressions.
    """
    return target(**kwargs)


def _text(result: dict[str, Any]) -> str:
    """Concatenate the text blocks of an agent-tool envelope."""
    return " ".join(block["text"] for block in result.get("content", []) if "text" in block)


# --------------------------------------------------------------------------- #
# The engine premise the whole rule rests on.
# --------------------------------------------------------------------------- #
class TestUnsizedValuePremise:
    """Pin the numpy/torch behaviour that makes the ``hasattr`` probe unsafe."""

    def test_a_zero_dimensional_array_declares_a_length_then_refuses_it(self) -> None:
        """``hasattr`` says yes and ``len()`` raises - the whole defect in two lines."""
        assert hasattr(UNSIZED, "__len__")
        with pytest.raises(TypeError):
            len(UNSIZED)

    def test_a_zero_dimensional_tensor_behaves_the_same_way(self) -> None:
        """torch shares the property, so the rule is not numpy-specific."""
        torch = pytest.importorskip("torch")
        scalar = torch.tensor(0.5)
        assert hasattr(scalar, "__len__")
        with pytest.raises(TypeError):
            len(scalar)

    @pytest.mark.parametrize(
        ("value", "raised"),
        [
            (_NegativeLength(), ValueError),
            (_OversizedLength(), OverflowError),
            (_RefusingLength(), RuntimeError),
        ],
        ids=["negative_len", "oversized_len", "refusing_len"],
    )
    def test_len_refuses_in_more_ways_than_a_type_error(self, value: Any, raised: type[Exception]) -> None:
        """``TypeError`` is not a superset of what ``len()`` raises (#1888).

        The first two values raise nothing themselves: ``len()`` converts what
        ``__len__`` returned into an index and CPython refuses, so a probe
        catching only ``TypeError`` is escaped by ordinary Python returning an
        ordinary ``int``. The third is the third-party case.
        """
        assert not issubclass(raised, TypeError), "the premise is that this is not the caught type"
        with pytest.raises(raised):
            len(value)

    def test_a_numpy_scalar_declares_no_length_at_all(self) -> None:
        """The other half of the domain: ``len()`` raises ``TypeError`` here too.

        Both spellings answer a validator's question identically - this value
        carries no component count - which is why one branch covers them.
        """
        assert not hasattr(np.float64(0.5), "__len__")
        with pytest.raises(TypeError):
            len(np.float64(0.5))  # type: ignore[arg-type]


class TestSequenceLength:
    """The shared owner of the rule."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ([1.0, 2.0, 3.0], 3),
            ((1.0, 2.0), 2),
            (np.array([1.0, 2.0, 3.0]), 3),
            (np.zeros((2, 3)), 2),
            ("abc", 3),
            ({"a": 1}, 1),
            ([], 0),
        ],
    )
    def test_reports_the_component_count_of_a_sized_value(self, value: Any, expected: int) -> None:
        """A readable length is returned unchanged."""
        assert sequence_length(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            np.array(0.5),
            np.array(True),
            np.float64(0.5),
            np.int64(3),
            0.5,
            None,
            object(),
            _NegativeLength(),
            _OversizedLength(),
            _RefusingLength(),
        ],
        ids=[
            "zero_d_float",
            "zero_d_bool",
            "np_float64",
            "np_int64",
            "float",
            "none",
            "object",
            "negative_len",
            "oversized_len",
            "refusing_len",
        ],
    )
    def test_reports_none_for_a_value_without_a_readable_length(self, value: Any) -> None:
        """No readable length is ``None``, never an exception.

        The last three rows are #1888: a length is unreadable whether ``__len__``
        is absent, refuses with a ``TypeError``, refuses some other way, or
        returns an ``int`` CPython will not convert. All four answer the caller's
        question the same way, so all four report through the one branch and the
        exception's type is not part of the question.
        """
        assert sequence_length(value) is None


class TestEveryUnreadableLengthAnswersTheSameWay:
    """The widened domain where this change is: the guards, and one live envelope.

    ``UNSIZED`` is pinned at every surface further down this module, and those
    surfaces reach the shared owner by the same route these values do - so
    re-asserting all of them four times over would measure one branch four
    times. What is worth pinning here is the owner's own domain (above), the
    guard that answered the length question a *second* way and therefore had to
    be fixed separately, and that a public envelope still closes over a value
    whose length refuses rather than merely being absent.
    """

    @pytest.mark.parametrize("value", UNREADABLE_LENGTHS, ids=UNREADABLE_IDS)
    def test_pose_vector_error_reports_a_value_with_no_readable_length(self, value: Any) -> None:
        """The second probe is gone; the verdict and its wording are not.

        ``pose_vector_error`` carried its own ``try: len(vec) except TypeError``,
        so it inherited nothing when the owner was widened. It now reads through
        the owner, and this is the text it already produced for a value carrying
        no readable length.
        """
        message = pose_vector_error("add_object", "position", value, 3)
        assert message is not None, "a value with no readable length was accepted as a pose"
        assert "'position' must be a list/tuple of 3 numbers" in message

    @pytest.mark.parametrize("value", UNREADABLE_LENGTHS, ids=UNREADABLE_IDS)
    def test_coerce_rgba_reports_a_value_with_no_readable_length(self, value: Any) -> None:
        """The colour guard reads the shared probe with no gate in front of it."""
        colour, reason = coerce_rgba("add_object", "color", value)
        assert colour is None
        assert reason is not None and "'color' must be a sequence of numbers" in reason

    @pytest.mark.parametrize("value", UNREADABLE_LENGTHS, ids=UNREADABLE_IDS)
    def test_the_router_still_answers_through_its_envelope(self, value: Any) -> None:
        """End to end: a dispatched vector parameter, refused rather than raised.

        The router is the surface #1844 was written for and the one whose
        contract is documented as never raising, so it is the envelope worth
        re-measuring for a length that refuses rather than one that is absent.
        """
        engine = MuJoCoSimEngine(tool_name="refusing_length_router", mesh=False)
        signature = inspect.signature(MuJoCoSimEngine.add_object)
        _, error = engine._validate_and_build_kwargs(
            "add_object", "add_object", signature, {"name": "crate", "position": value}
        )
        assert error is not None, "add_object(position=<unreadable length>) was accepted"
        assert error["status"] == "error"
        assert "Parameter 'position' must be a list of" in _text(error)


# --------------------------------------------------------------------------- #
# The agent-tool router: one structured error per rejected vector parameter.
# --------------------------------------------------------------------------- #
# Every router vector param, paired with a method whose signature declares it
# (the router rejects a parameter the target method does not accept before it
# ever reaches the dimension check).
_VECTOR_PARAM_OWNERS: dict[str, tuple[str, dict[str, Any]]] = {
    "position": ("add_object", {"name": "crate"}),
    "target": ("add_camera", {"name": "cam"}),
    "origin": ("raycast", {"direction": [0.0, 0.0, -1.0]}),
    "force": ("apply_force", {"body_name": "crate"}),
    "torque": ("apply_force", {"body_name": "crate"}),
    "gravity": ("set_gravity", {}),
    "direction": ("raycast", {"origin": [0.0, 0.0, 1.0]}),
    "point": ("apply_force", {"body_name": "crate"}),
    "orientation": ("add_object", {"name": "crate"}),
    "color": ("add_object", {"name": "crate"}),
}


class TestAgentToolRouterVectorParams:
    """No dispatched vector parameter may escape the envelope on a length probe."""

    def test_every_router_vector_param_is_covered_here(self) -> None:
        """Exhaustiveness: a twelfth vector param must be pinned too.

        ``_FIELD_ALIASES`` entries are remapped to their canonical name before
        the dimension check runs, so they are covered by the name they alias.
        """
        aliases = set(MuJoCoSimEngine._FIELD_ALIASES)
        assert set(_VECTOR_PARAM_OWNERS) == set(MuJoCoSimEngine._VECTOR_PARAM_LENGTHS) - aliases

    @pytest.mark.parametrize("param", sorted(_VECTOR_PARAM_OWNERS))
    def test_an_unsized_vector_is_reported_through_the_envelope(self, param: str) -> None:
        """The caller gets a structured error naming the parameter, not a raise."""
        action, extra = _VECTOR_PARAM_OWNERS[param]
        engine = MuJoCoSimEngine(tool_name="unsized_router", mesh=False)
        signature = inspect.signature(getattr(MuJoCoSimEngine, action))
        _, error = engine._validate_and_build_kwargs(action, action, signature, {**extra, param: UNSIZED})
        assert error is not None, f"{action}({param}=<0-d array>) was accepted"
        assert error["status"] == "error"
        assert f"Parameter '{param}' must be a list of" in _text(error)

    @pytest.mark.parametrize("param", sorted(_VECTOR_PARAM_OWNERS))
    def test_a_correctly_sized_numpy_vector_is_still_accepted(self, param: str) -> None:
        """Over-reach control: numpy vectors of the right width keep dispatching."""
        action, extra = _VECTOR_PARAM_OWNERS[param]
        width = MuJoCoSimEngine._VECTOR_PARAM_LENGTHS[param][0]
        engine = MuJoCoSimEngine(tool_name="unsized_router_ok", mesh=False)
        signature = inspect.signature(getattr(MuJoCoSimEngine, action))
        _, error = engine._validate_and_build_kwargs(
            action, action, signature, {**extra, param: np.linspace(0.1, 0.4, width)}
        )
        assert error is None, _text(error or {})


# --------------------------------------------------------------------------- #
# get_world_point: pixels, and each [u, v] pair.
# --------------------------------------------------------------------------- #
class TestGetWorldPointPixels:
    """Both length probes on the pixel list keep the never-raises envelope.

    These are the structural checks the method runs before any render work, so
    they are reachable without a compiled world.
    """

    def test_an_unsized_pixels_container_is_reported(self) -> None:
        """A 0-d array in place of the pixel list is refused with the usage hint."""
        engine = MuJoCoSimEngine(tool_name="unsized_pixels", mesh=False)
        result = _call(engine.get_world_point, pixels=np.array(320), camera_name="cam")
        assert result["status"] == "error"
        assert "get_world_point requires 'pixels'" in _text(result)

    def test_an_unsized_pixel_pair_is_reported(self) -> None:
        """A 0-d array in place of one [u, v] pair names that pair's index."""
        engine = MuJoCoSimEngine(tool_name="unsized_pixel_pair", mesh=False)
        result = _call(engine.get_world_point, pixels=[np.array(320)], camera_name="cam")
        assert result["status"] == "error"
        assert "pixels[0] must be a [u, v] pair" in _text(result)

    def test_a_numpy_pixel_array_still_passes_structural_validation(self) -> None:
        """Over-reach control: a correctly shaped numpy pixel array reaches the render.

        With no world compiled the render is what fails, which is exactly the
        evidence that the structural checks accepted the pixels.
        """
        engine = MuJoCoSimEngine(tool_name="sized_pixels", mesh=False)
        result = _call(engine.get_world_point, pixels=np.array([[320, 240]]), camera_name="cam")
        assert result["status"] == "error"
        assert "failed to render camera frame" in _text(result)


# --------------------------------------------------------------------------- #
# send_action's ordered-vector form.
# --------------------------------------------------------------------------- #
class TestSendActionVectorForm:
    """An unsized action reports the mapping-or-vector contract it violated."""

    def test_an_unsized_action_names_the_two_accepted_shapes(self, tmp_path: Path) -> None:
        """The caller is told what an action may be, not that a length failed."""
        model = tmp_path / "arm.xml"
        model.write_text(_ARM_MJCF)
        engine = MuJoCoSimEngine(tool_name="unsized_action", mesh=False)
        engine.create_world()
        engine.add_robot(name="arm", urdf_path=str(model))
        result = _call(engine.send_action, action=np.array(0.5))
        assert result["status"] == "error"
        message = _text(result)
        assert "must be a mapping" in message
        assert "ordered numeric" in message


# --------------------------------------------------------------------------- #
# mjpeg_frames: the documented ValueError, not a bare TypeError.
# --------------------------------------------------------------------------- #
class TestMjpegFrameSize:
    """``size`` fails through the documented ``Raises: ValueError`` channel."""

    def _frame(self) -> np.ndarray:
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def test_an_unsized_size_raises_the_documented_valueerror(self) -> None:
        """A 0-d array is refused by the eager validator, message included."""
        with pytest.raises(ValueError, match="size must be a .width, height. pair"):
            _call(mjpeg_frames, frame_fn=self._frame, size=np.array(640), max_frames=1)

    def test_a_numpy_size_pair_is_still_accepted(self) -> None:
        """Over-reach control: a 2-element numpy size keeps working."""
        stream = _call(mjpeg_frames, frame_fn=self._frame, size=np.array([640, 480]), max_frames=1)
        assert next(iter(stream)).startswith(b"--frame")


# --------------------------------------------------------------------------- #
# policies/: the sites #1844's scan root could not see (#1883).
# --------------------------------------------------------------------------- #
class TestMockPolicyStateWidth:
    """``MockPolicy`` derives its joint count without raising on a 0-d state.

    The canonical reference implementation every provider is pointed at, so the
    idiom surviving here is also the one most likely to be copied.
    """

    def _dim(self, state: Any) -> int:
        actions = asyncio.run(MockPolicy().get_actions({"observation.state": state}, "go"))
        return len(actions[0])

    def test_an_unsized_state_takes_the_documented_fallback(self) -> None:
        """A 0-d state carries no width, so the ``else 6`` branch applies to it.

        It used to raise ``TypeError: len() of unsized object`` instead - past the
        branch written for a state that does not declare a width.
        """
        assert self._dim(UNSIZED) == 6

    def test_a_scalar_state_answers_the_same_either_way(self) -> None:
        """A plain float and a 0-d array are both scalars; both mean "no width"."""
        assert self._dim(0.5) == self._dim(UNSIZED) == 6

    def test_a_sized_state_still_sets_the_width(self) -> None:
        """Over-reach control: a real state vector keeps deriving its own width."""
        assert self._dim(np.zeros(4)) == 4
        assert self._dim([0.0, 0.0, 0.0]) == 3


class TestWBCObservationVectors:
    """``WBCPolicy``'s two observation readers answer, rather than raise, for a 0-d entry."""

    def _flat(self, state: Any, names: list[str]) -> np.ndarray:
        """``_read_joint_vector`` on an instance carrying only the attribute it reads.

        ``__new__`` rather than ``__init__``: the reader needs one attribute and
        the constructor needs an ONNX runtime and a checkpoint, so building a
        real policy here would make an observation-parsing test depend on
        ``onnxruntime`` being installed. It is a genuine ``WBCPolicy``, so the
        call is type-checked like any other caller's would be.
        """
        policy = WBCPolicy.__new__(WBCPolicy)
        policy._robot_state_keys = []
        return policy._read_joint_vector({"observation.state": state}, "position", names)

    def test_read_vec_reports_an_unsized_entry_as_absent(self) -> None:
        """``_read_vec`` returns ``None`` for every value it cannot use, 0-d included.

        Both callers (``base_ang_vel``, ``base_quat``) are written against that
        ``None``; a 0-d entry used to raise out of ``get_actions`` mid-rollout.
        """
        assert WBCPolicy._read_vec({"k": UNSIZED}, ("k",), 3) is None

    def test_read_vec_still_returns_a_correctly_sized_vector(self) -> None:
        """Over-reach control: the accepted width is unchanged, and so is a wrong one."""
        got = WBCPolicy._read_vec({"k": [1.0, 2.0, 3.0]}, ("k",), 3)
        assert got is not None and np.allclose(got, [1.0, 2.0, 3.0])
        assert WBCPolicy._read_vec({"k": [1.0, 2.0]}, ("k",), 3) is None

    def test_a_scalar_state_reads_the_same_either_spelling(self) -> None:
        """The flat/per-joint discriminator no longer splits the two scalars.

        A ``hasattr`` probe sent a 0-d array down the flat branch and a plain
        float down the per-joint one, so one observation produced two different
        joint vectors depending on how the scalar was spelled. Neither carries a
        component per joint, so both read as the per-joint form.
        """
        names = ["a", "b", "c"]
        assert np.allclose(self._flat(UNSIZED, names), self._flat(0.5, names))
        assert np.allclose(self._flat(UNSIZED, names), np.zeros(3))

    def test_a_flat_state_vector_is_still_consumed_positionally(self) -> None:
        """Over-reach control: the flat form the discriminator exists for is unchanged."""
        names = ["a", "b", "c"]
        assert np.allclose(self._flat([0.1, 0.2, 0.3], names), [0.1, 0.2, 0.3])
        assert np.allclose(self._flat(np.array([0.1, 0.2, 0.3]), names), [0.1, 0.2, 0.3])


# --------------------------------------------------------------------------- #
# Structural: no module may probe a caller length with hasattr again.
# --------------------------------------------------------------------------- #


def _hasattr_len_probes(tree: ast.AST) -> list[int]:
    """Line numbers of every ``hasattr(<x>, "__len__")`` call in ``tree``."""
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or getattr(node.func, "id", "") != "hasattr":
            continue
        if len(node.args) != 2 or not isinstance(node.args[1], ast.Constant):
            continue
        if node.args[1].value == "__len__":
            lines.append(node.lineno)
    return lines


def _package_modules() -> list[Path]:
    """Every module of ``strands_robots``.

    The scan was ``("simulation", "rendering")`` when #1844 introduced it, and
    the sweep was therefore exactly as wide as the scan: three call sites in
    ``policies/`` were never converted and went on raising for two more months
    (#1883). Naming packages is what allowed that, so the root is the package -
    a value whose length cannot be read is not a simulation concern, and a
    module added under a new subpackage is covered on the day it lands rather
    than when someone remembers to extend a tuple.
    """
    return sorted(Path(inspect.getfile(sequence_length)).parent.rglob("*.py"))


class TestNoDirectLengthProbe:
    """The shared owner cannot be bypassed by reintroducing the unsafe idiom."""

    def test_the_scan_root_resolves_to_real_modules(self) -> None:
        """Non-vacuity: a mislocated root would make the scan below pass trivially."""
        modules = _package_modules()
        assert len(modules) > 20, modules
        assert any(path.name == "base.py" for path in modules)

    def test_the_scan_reaches_every_subpackage_not_a_named_few(self) -> None:
        """The root covers the packages a named tuple left out, ``policies/`` included.

        #1883's three unconverted sites were all in ``policies/``, which the
        original ``("simulation", "rendering")`` root could not see. Asserting on
        the subpackages present keeps a future narrowing of the root from
        silently reopening that gap.
        """
        package_dir = Path(inspect.getfile(sequence_length)).parent
        reached = {
            path.relative_to(package_dir).parts[0]
            for path in _package_modules()
            if len(path.relative_to(package_dir).parts) > 1
        }
        assert {"policies", "simulation", "rendering"} <= reached, sorted(reached)

    def test_no_module_probes_a_length_with_hasattr(self) -> None:
        """``hasattr(x, "__len__")`` is never a safe stand-in for a length probe.

        A 0-d array passes it and then raises from ``len()``. Ask
        :func:`strands_robots.utils.sequence_length` for the length instead and
        branch on ``None``; use ``__getitem__`` when indexability is the question.
        """
        offenders = {
            f"{path.name}:{line}"
            for path in _package_modules()
            for line in _hasattr_len_probes(ast.parse(path.read_text()))
        }
        assert not offenders, f"use sequence_length() instead of a hasattr length probe: {sorted(offenders)}"

    def test_the_scanner_detects_a_planted_probe(self) -> None:
        """Meta: an empty result means clean sources, not a scanner matching nothing."""
        planted = ast.parse('if not hasattr(value, "__len__"):\n    pass\n')
        assert _hasattr_len_probes(planted) == [1]


# --------------------------------------------------------------------------- #
# Structural: the rule keeps exactly one owner.
# --------------------------------------------------------------------------- #
def _own_length_probes(tree: ast.AST) -> list[tuple[str, int]]:
    """``(function, line)`` for every ``len(<param>)`` on a parameter annotated ``Any``.

    ``Any`` is how ``utils.py`` spells "the caller's value" - its guards' other
    parameters are the ``str`` labels a call site supplies, which are literals -
    so it is the same key #1873's rendering scan used, for the same reason.

    The scope is ``utils.py`` rather than the package. Measured package-wide the
    same scan reports a dozen legitimate reads, all taken *after* a type check
    has established the value is a sized container (``mesh/security.py``,
    ``tools/use_lerobot.py``, the LIBERO parsers), so a package-wide assertion
    would have to enumerate exceptions and would stop meaning anything. Inside
    ``utils.py`` the question is unambiguous: this module *is* where the rule
    lives, so a second ``len()`` on a caller value here is a second answer to it.
    """
    hits: list[tuple[str, int]] = []
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = function.args
        untyped = {
            argument.arg
            for argument in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
            if isinstance(argument.annotation, ast.Name) and argument.annotation.id == "Any"
        }
        if not untyped:
            continue
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "len"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in untyped
            ):
                hits.append((function.name, node.lineno))
    return hits


class TestTheLengthRuleHasOneOwner:
    """A second probe cannot reappear, having been the other half of #1888.

    ``sequence_length`` exists because the rule cannot be kept in step in two
    places, and it was in two places: ``pose_vector_error`` carried its own
    ``except TypeError``, so widening the owner left it untouched. A list of
    guards would not have caught that - the duplicate was added beside the owner,
    in the same module, by someone reading the same docstring - so the invariant
    is scanned rather than enumerated.
    """

    def test_sequence_length_is_the_only_function_asking_len_of_a_caller_value(self) -> None:
        """One owner, by name, so the offender is named rather than merely counted."""
        source = ast.parse(Path(inspect.getfile(sequence_length)).read_text())
        owners = {name for name, _ in _own_length_probes(source)}
        assert owners == {"sequence_length"}, (
            f"ask sequence_length() for the length and branch on None, rather than a second len(): {sorted(owners)}"
        )

    def test_the_scanner_reports_the_probe_that_was_removed(self) -> None:
        """Meta: the scan matched the duplicate it is here to prevent.

        Planted in the shape ``pose_vector_error`` actually carried, so a scanner
        rewrite that stopped matching it would fail here rather than pass
        vacuously against clean sources.
        """
        planted = ast.parse(
            "def pose_vector_error(method: str, param_name: str, vec: Any, expected_len: int) -> str | None:\n"
            "    try:\n"
            "        length = len(vec)\n"
            "    except TypeError:\n"
            "        return method\n"
            "    return None\n"
        )
        assert _own_length_probes(planted) == [("pose_vector_error", 3)]

    def test_the_scanner_does_not_match_a_read_of_this_modules_own_value(self) -> None:
        """Non-over-reach: a length taken of a value the function built is fine.

        ``require_optionals`` reads ``len(missing)`` and the name-list guard reads
        ``len(shown)``; neither is a caller value and neither may be reported.
        """
        planted = ast.parse(
            "def guard(value: Any) -> str:\n    missing = [value]\n    return 'is' if len(missing) == 1 else 'are'\n"
        )
        assert _own_length_probes(planted) == []


# --------------------------------------------------------------------------- #
# The other half of "unreadable": a value that reads cleanly and then has no
# length. Every row of UNREADABLE_LENGTHS above is a value the component read
# never reaches - a 0-d array is not iterable at all, and the three hostile
# ``__len__`` rows are legacy sequences whose ``__getitem__`` never stops, so
# they are refused before any component is produced. A generator is the opposite
# shape: it yields finite components and only then cannot say how many there
# were. Factories, because each of these is one-shot - a generator handed to two
# guards is already empty for the second (#1906's reason, applied to the fixture).
# --------------------------------------------------------------------------- #
class _RefusingLengthIterable:
    """Iterates cleanly to completion and refuses to report a length.

    The unsized-iterable counterpart of :class:`_RefusingLength`: that one is
    read through ``__getitem__`` and never terminates, this one is a genuine
    iterable whose read finishes normally, so it reaches the point where the
    length matters.
    """

    def __iter__(self) -> Iterator[float]:
        return iter((0.1, 0.2, 0.3))

    def __len__(self) -> int:
        raise RuntimeError("length unavailable")


class _LazySizedVector:
    """A legacy sequence: readable ``__len__``, components produced one at a time.

    The over-reach control for the rule below. Laziness is not what makes a value
    unusable here - being unable to answer "how many components?" is - so a value
    that is read a component at a time and *can* answer must keep working.
    """

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> float:
        if index >= 3:
            raise IndexError(index)
        return 0.1 * (index + 1)


def _without_object_ids(message: str) -> str:
    """Mask CPython object addresses so two verdicts can be compared byte for byte.

    These values are one-shot, so each guard is handed its own instance and the
    ``repr`` the refusal quotes carries a different address. The addresses are the
    only part that may differ; masking them is what lets the comparison below be
    an equality rather than a substring.
    """
    return re.sub(r"0x[0-9a-fA-F]+", "0xADDR", message)


UNSIZED_ITERABLES: list[Any] = [
    pytest.param(lambda: (component for component in (0.1, 0.2, 0.3)), id="generator"),
    pytest.param(lambda: iter([0.1, 0.2, 0.3]), id="list_iterator"),
    pytest.param(lambda: map(float, (1, 2, 3)), id="map"),
    pytest.param(_RefusingLengthIterable, id="refusing_len_iterable"),
]

# Values whose components read as finite and whose length *is* readable. The
# accepted side of the same rule, so a guard cannot satisfy it by refusing
# everything lazy.
SIZED_VECTORS: list[Any] = [
    pytest.param(lambda: [0.1, 0.2, 0.3], id="list"),
    pytest.param(lambda: (0.1, 0.2, 0.3), id="tuple"),
    pytest.param(lambda: np.array([0.1, 0.2, 0.3]), id="np_array"),
    pytest.param(lambda: range(3), id="range"),
    pytest.param(_LazySizedVector, id="lazy_sized_sequence"),
]


class TestTheVerdictHalfAnswersAnUnreadableLength:
    """The two guards over the shared read, and what their callers do next.

    :func:`~strands_robots.utils.finite_vector_error` is the verdict half: it
    returns a message and not the floats, so a caller holding one counts the
    components by reading the value a second time. It documents deferring that
    count, and deferring it is exactly what makes a *readable* length part of the
    verdict - a value with no length cannot be read twice, because the read
    behind the verdict consumed it.

    It was the one guard in the family not asking. Its own coercing sibling over
    the same read (:func:`~strands_robots.utils.coerce_size_vector`),
    :func:`~strands_robots.utils.coerce_rgba` and
    :func:`~strands_robots.utils.pose_vector_error` all refuse an unreadable
    length, the middle one with a comment naming this precise value class - "it
    is what refuses a generator, whose components a read would consume before
    anything could count them". The verdict half accepted it, and both of its
    real call sites then read the consumed value: ``add_object`` reported a
    three-component extent as ``got 0 (size=[])``, and ``patch_scene_mjcf``
    raised ``object of type 'generator' has no len()`` into its envelope - naming
    neither the field nor the method, while the sibling fields of that same op
    named both.
    """

    @pytest.mark.parametrize("make_value", UNSIZED_ITERABLES)
    def test_the_verdict_half_reports_an_unreadable_length(self, make_value: Any) -> None:
        """A message naming the parameter, not ``None`` and not a raise."""
        message = finite_vector_error("add_object", "size", make_value())
        assert message is not None, "a value with no readable length passed the verdict"
        assert "add_object: 'size' must be a list/tuple of numbers" in message

    @pytest.mark.parametrize("make_value", UNSIZED_ITERABLES)
    def test_the_coercing_half_reports_it_in_the_same_words(self, make_value: Any) -> None:
        """One value, one verdict: the two guards over one read must not disagree.

        Byte equality rather than a substring, because the point is that neither
        guard states the rule in its own words - a second wording is how two
        halves of one contract drift apart.
        """
        verdict = finite_vector_error("add_object", "size", make_value())
        _floats, coerced = coerce_size_vector("add_object", "size", make_value())
        assert verdict is not None and coerced is not None
        assert _without_object_ids(verdict) == _without_object_ids(coerced)

    @pytest.mark.parametrize("make_value", SIZED_VECTORS)
    def test_a_lazily_read_value_with_a_readable_length_is_still_accepted(self, make_value: Any) -> None:
        """Over-reach control: laziness is not the defect, an unanswerable count is.

        ``_LazySizedVector`` is read one component at a time through the legacy
        protocol and reports its length, so it must keep passing - otherwise the
        rule would be "refuse anything not already a list", which is not the rule
        the sibling guards apply.
        """
        assert finite_vector_error("add_object", "size", make_value()) is None
        floats, reason = coerce_size_vector("add_object", "size", make_value())
        assert reason is None
        assert floats is not None and len(floats) == 3

    def test_every_refusal_this_guard_already_gave_is_unchanged(self) -> None:
        """Non-regression, stated as the exact texts rather than as prose.

        A value whose ``__iter__`` raises, or whose read fails part-way, has no
        readable length either. Those verdicts describe what actually happened,
        and reporting them as "not a list/tuple" would claim a domain check that
        never ran (#1875, #1878) - which is why the length probe runs after the
        component read and not before it.
        """

        def fails_part_way() -> Iterator[float]:
            yield 0.1
            raise RuntimeError("stream truncated")

        class HostileIter:
            def __iter__(self) -> Iterator[float]:
                raise RuntimeError("no iteration for you")

        part_way = finite_vector_error("raycast", "origin", fails_part_way())
        never_started = finite_vector_error("raycast", "origin", HostileIter())
        assert part_way is not None and "'origin[1]' could not be read" in part_way
        assert never_started is not None and "'origin' could not be iterated" in never_started
        for message in (part_way, never_started):
            assert "must be a list/tuple of numbers" not in message
        # The plain non-iterable and the non-numeric element are untouched too.
        assert finite_vector_error("m", "size", 0.5) == "m: 'size' must be a list/tuple of numbers, got 0.5"
        assert finite_vector_error("m", "size", ["a"]) == "m: 'size' elements must be numbers, got ['a']"
        assert finite_vector_error("m", "size", [0.1, float("nan")]) == (
            "m: 'size' must contain finite numbers (no nan/inf), got [0.1, nan]"
        )


class TestTheCallersThatCountByReadingAgain:
    """End to end: the two live surfaces that hold only the verdict.

    Both take the message, then count the components themselves - the one thing a
    consumed value cannot answer. These pin what each reported instead, so a
    guard that stops asking about the length is caught at the surface a caller
    actually touches and not only at the helper.
    """

    def test_add_object_reports_the_extent_it_was_given(self) -> None:
        """It must not report a supplied extent as an omitted one.

        ``add_object`` counts ``size`` against the shape after the verdict, by
        reading it again. For a consumed value that count is zero, so the refusal
        said ``got 0 (size=[])`` - describing a caller who passed nothing, about a
        caller who passed three components.
        """
        engine = MuJoCoSimEngine(tool_name="unsized_size_add_object", mesh=False)
        assert engine.create_world()["status"] == "success"
        # Bound through ``Any`` rather than suppressed: ``size`` is declared
        # ``list[float] | None`` and the measurement is what the runtime does with a
        # value outside that declaration, which is the caller mistake this guards.
        unsized: Any = (edge for edge in (0.3, 0.3, 0.3))
        result = engine.add_object(name="crate", shape="box", size=unsized, position=[0.0, 0.0, 0.4])
        assert result["status"] == "error"
        message = _text(result)
        assert "'size' must be a list/tuple of numbers" in message
        assert "got 0" not in message, "a supplied extent was reported as an empty one"
        assert "size=[]" not in message

    def test_a_patch_op_size_field_answers_like_its_sibling_fields(self) -> None:
        """The op's four numeric fields must not answer in two different ways.

        ``size`` is the only field in the patch-op domain table held by the
        verdict half; ``pos``, ``quat`` and ``rgba`` are held by guards that
        already refuse an unreadable length. Without the probe, ``size`` reached
        the compiler's own ``len()`` and the envelope carried ``object of type
        'generator' has no len()`` - a message naming neither the field nor the op,
        beside a sibling field naming both.
        """
        engine = MuJoCoSimEngine(tool_name="unsized_size_patch_op", mesh=False)
        assert engine.create_world()["status"] == "success"

        def op(field: str, value: Any) -> dict[str, Any]:
            base: dict[str, Any] = {
                "op": "add_geom",
                "body": "world",
                "name": f"patched_{field}",
                "type": "box",
                "size": [0.2, 0.2, 0.2],
                "pos": [1.0, 0.0, 0.3],
            }
            base[field] = value
            return base

        size_result = engine.patch_scene_mjcf(ops=[op("size", (edge for edge in (0.25, 0.25, 0.25)))])
        pos_result = engine.patch_scene_mjcf(ops=[op("pos", (axis for axis in (1.5, 0.0, 0.3)))])
        size_message, pos_message = _text(size_result), _text(pos_result)
        assert size_result["status"] == "error" and pos_result["status"] == "error"
        assert "has no len()" not in size_message, "the compiler's own TypeError reached the envelope"
        for field, message in (("size", size_message), ("pos", pos_message)):
            assert f"add_geom: '{field}' must be a list/tuple of" in message, message
