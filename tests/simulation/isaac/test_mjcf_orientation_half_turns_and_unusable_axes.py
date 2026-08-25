"""The orientation arms only a half turn, or an axis the reader cannot use, reaches.

MJCF's five orientation spellings are already graded against MuJoCo's compiler,
but every declaration those suites use names a rotation of less than a third of a
turn and is well formed. Two families of the reader's arms are therefore reached
by no fixture.

An ``xyaxes`` whose rotation is at or near a half turn is the first. Converting a
rotation matrix to a quaternion from its trace alone degenerates there -- the
term under the square root goes to zero -- so the conversion branches on
whichever diagonal term is largest instead, which is the case its own docstring
says the branching exists for. Three of those four arms ran in no test, and each
assembles the quaternion from a different set of matrix entries with different
signs, so a sign or index slip in one of them reports a rotated link or object at
a *different* rotation while the load still reports success. A half turn is not
an exotic declaration: it is what an ``xyaxes`` that flips two axes states, which
is how a model mounts a gripper or a camera facing backwards.

A declaration whose axes the reader cannot use is the second: too few or too many
numbers for the spelling, an axis of zero length, or an ``xyaxes`` whose y axis
is parallel to its x axis, leaving nothing after orthogonalizing. MuJoCo refuses
every one of these models and the reader tolerates them as identity, the reading
it documents for a declaration it cannot use. Only a value that is not a number
at all was graded, and that takes the earlier ``float()`` failure rather than any
of these arms, so the arms that keep a zero-length axis from being divided by
were never run.

Every expectation for a rotation MuJoCo compiles is derived from
``mujoco.MjModel``: the fixture is compiled and the reader is compared against
the ``body_quat`` the compiler stored, so no expected quaternion is restated by
hand. A declaration MuJoCo refuses has no compiled answer to compare against, so
those are graded against the tolerated reading instead, with the refusal itself
asserted as a premise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from strands_robots.simulation.isaac.loaders import load_mjcf, load_mjcf_scene_objects

mujoco = pytest.importorskip("mujoco")

#: The orientation a reader that could not use a declaration reports.
IDENTITY = (1.0, 0.0, 0.0, 0.0)


def _ids(rows: list[tuple[str, ...]] | list[tuple]) -> list[str]:
    """Single-token parametrize ids from each row's leading label."""
    return [str(row[0]).replace(" ", "-") for row in rows]


#: Rotations declared as ``xyaxes``, as ``(label, axis, degrees)``. Each family
#: earns its place, and the reasons are complementary:
#:
#: * The exact half turns are the boundary itself -- the trace reaches its minimum
#:   there, so the term the trace form divides by is exactly zero.
#: * The half turn about the xy diagonal leaves two diagonal terms equal, which is
#:   the tie the choice of arm has to break.
#: * Ten degrees short of a half turn gives the real part a small but non-zero
#:   value. At exactly a half turn the real part is zero, and a sign slip on zero
#:   is not visible, so these are what grade the signs.
#: * The skew rotations put all four quaternion components at distinct non-zero
#:   magnitudes, so reading the wrong matrix entry into one of them shows up.
#: * The quarter turn keeps the trace form itself in the set as a control.
#:
#: Between them they span the four cases a rotation-to-quaternion conversion
#: branches on, which :class:`TestTheFixturesAreWhatTheyClaim` asserts rather than
#: assumes.
ROTATIONS = [
    ("a half turn about x", (1.0, 0.0, 0.0), 180.0),
    ("a half turn about y", (0.0, 1.0, 0.0), 180.0),
    ("a half turn about z", (0.0, 0.0, 1.0), 180.0),
    ("a half turn about the xy diagonal", (1.0, 1.0, 0.0), 180.0),
    ("ten degrees short of a half turn about x", (1.0, 0.0, 0.0), 170.0),
    ("ten degrees short of a half turn about y", (0.0, 1.0, 0.0), 170.0),
    ("ten degrees short of a half turn about z", (0.0, 0.0, 1.0), 170.0),
    ("a skew rotation leaning on x", (4.0, 1.0, 2.0), 155.0),
    ("a skew rotation leaning on y", (1.0, 4.0, 2.0), 155.0),
    ("a skew rotation leaning on z", (2.0, 1.0, 4.0), 155.0),
    ("a quarter turn about z", (0.0, 0.0, 1.0), 90.0),
]

#: ``zaxis`` spellings naming the axis opposite ``+z``. There is no unique minimal
#: rotation taking ``+z`` onto ``-z``, so the reader answers a half turn about x
#: and this is what says that is the half turn MuJoCo also picks. The second
#: spelling is off the axis by less than the reader's own tolerance, so it must be
#: read the same way as the exact one rather than through the general formula.
ANTIPODAL_ZAXES = [
    ("exactly opposite z", "0 0 -1"),
    ("within tolerance of opposite z", "1e-14 0 -1"),
]

#: Declarations naming no axis the reader can use, as
#: ``(label, body attributes, compiler element)``. MuJoCo refuses every one, which
#: :class:`TestTheFixturesAreWhatTheyClaim` asserts.
UNUSABLE = [
    ("euler with two angles", 'euler="10 20"', ""),
    ("euler with four angles", 'euler="10 20 30 40"', ""),
    ("euler with a sequence naming a non-axis", 'euler="10 20 30"', '<compiler eulerseq="xyq"/>'),
    ("euler with a sequence of two axes", 'euler="10 20 30"', '<compiler eulerseq="xy"/>'),
    ("axisangle with no angle", 'axisangle="0 0 1"', ""),
    ("axisangle about a zero-length axis", 'axisangle="0 0 0 90"', ""),
    ("zaxis with two components", 'zaxis="0 1"', ""),
    ("xyaxes with five components", 'xyaxes="1 0 0 0 1"', ""),
    ("xyaxes whose x axis is zero", 'xyaxes="0 0 0 0 1 0"', ""),
    ("xyaxes whose y axis is parallel to x", 'xyaxes="1 0 0 2 0 0"', ""),
    ("xyaxes whose y axis is antiparallel to x", 'xyaxes="1 0 0 -1 0 0"', ""),
    ("xyaxes whose y axis is zero", 'xyaxes="1 0 0 0 0 0"', ""),
]


def _rotation_matrix(axis: tuple[float, float, float], degrees: float) -> np.ndarray:
    """The rotation of ``degrees`` about ``axis``, by Rodrigues' formula."""
    unit = np.asarray(axis, dtype=float)
    unit /= float(np.linalg.norm(unit))
    angle = math.radians(degrees)
    cross = np.array(
        [[0.0, -unit[2], unit[1]], [unit[2], 0.0, -unit[0]], [-unit[1], unit[0], 0.0]],
        dtype=float,
    )
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def _xyaxes(axis: tuple[float, float, float], degrees: float) -> str:
    """That rotation as an ``xyaxes`` declaration: its x column, then its y column."""
    matrix = _rotation_matrix(axis, degrees)
    return " ".join(f"{value:.17g}" for value in (*matrix[:, 0], *matrix[:, 1]))


def _write_object(tmp_path, body_attrs: str = "", compiler: str = "", *, name: str = "scene") -> str:
    """A one-body scene the scene reader reports as a single object.

    A box geom, not a mesh: MuJoCo bakes a mesh's principal-inertia alignment into
    the compiled frame, so a mesh's value is not the frame the file declared.
    """
    path = tmp_path / f"{name}.xml"
    path.write_text(
        f"<mujoco>{compiler}<worldbody>"
        f'<body name="obj" pos="0.1 0.2 0.3" {body_attrs}>'
        f'<geom name="g" type="box" size="0.1 0.05 0.02"/>'
        f"</body></worldbody></mujoco>",
        encoding="utf-8",
    )
    return str(path)


def _write_link(tmp_path, body_attrs: str, *, name: str = "robot") -> str:
    """The same body as one link of an articulated robot, for ``load_mjcf``.

    The link carries a hinge joint because ``load_mjcf`` refuses a model with no
    articulation.
    """
    path = tmp_path / f"{name}.xml"
    path.write_text(
        f"<mujoco><worldbody>"
        f'<body name="obj" pos="0.1 0.2 0.3" {body_attrs}>'
        f'<joint name="j" type="hinge" axis="0 0 1"/>'
        f'<geom name="g" type="box" size="0.1 0.05 0.02"/>'
        f"</body></worldbody></mujoco>",
        encoding="utf-8",
    )
    return str(path)


def _mujoco_body_quat(path: str) -> np.ndarray:
    """The orientation MuJoCo's compiler stored for the fixture's body."""
    model = mujoco.MjModel.from_xml_path(path)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj")
    assert body > 0, "premise: the fixture declares a body named 'obj'"
    return np.asarray(model.body_quat[body], dtype=float)


def _same_rotation(got, expected) -> float:
    """Agreement of two wxyz quaternions, treating ``q`` and ``-q`` as one rotation."""
    a = np.asarray(got, dtype=float)
    b = np.asarray(expected, dtype=float)
    return float(min(np.abs(a - b).max(), np.abs(a + b).max()))


def _object_quat(path: str) -> tuple[float, ...]:
    """The orientation the scene reader reports, via the public loader."""
    objects = load_mjcf_scene_objects(path)
    assert len(objects) == 1, f"premise: the fixture declares one object, got {[o.name for o in objects]}"
    return tuple(objects[0].quat)


def _link_quat(path: str) -> tuple[float, ...]:
    """The orientation the robot reader reports for the link, via the public loader."""
    matches = [body for body in load_mjcf(path).bodies if body.name == "obj"]
    assert len(matches) == 1, "premise: the fixture declares one link named 'obj'"
    return tuple(matches[0].orientation)


def _conversion_arm(quat: np.ndarray) -> str:
    """Which of the four cases a rotation-to-quaternion conversion must branch on.

    The trace form is usable only while the trace is positive; below that the
    quaternion has to be built around whichever diagonal term is largest, and each
    choice is a different expression. Deriving the label from the rotation MuJoCo
    compiled -- not from the reader -- lets the fixture set be checked for
    reaching all four rather than assumed to.
    """
    matrix = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=float))
    diagonal = (matrix[0], matrix[4], matrix[8])
    if sum(diagonal) > 0.0:
        return "trace"
    if diagonal[0] > diagonal[1] and diagonal[0] > diagonal[2]:
        return "x"
    if diagonal[1] > diagonal[2]:
        return "y"
    return "z"


class TestTheFixturesAreWhatTheyClaim:
    """The premises the gradings below rest on, each asserted rather than assumed."""

    def test_the_rotations_reach_every_conversion_arm(self, tmp_path):
        """All four arms are exercised, so none of them is graded vacuously.

        Without this, a change to how the conversion picks its arm could route
        every fixture down one of them and the suite would still be green.
        """
        reached = {
            _conversion_arm(
                _mujoco_body_quat(_write_object(tmp_path, f'xyaxes="{_xyaxes(axis, degrees)}"', name=label))
            )
            for label, axis, degrees in ROTATIONS
        }
        assert reached == {"trace", "x", "y", "z"}, (
            f"the fixtures reach {sorted(reached)}; a conversion arm no fixture reaches is ungraded"
        )

    @pytest.mark.parametrize(("label", "axis", "degrees"), ROTATIONS, ids=_ids(ROTATIONS))
    def test_mujoco_compiles_every_rotation(self, tmp_path, label, axis, degrees):
        """Each rotation fixture is a model MuJoCo accepts, so it has an oracle."""
        quat = _mujoco_body_quat(_write_object(tmp_path, f'xyaxes="{_xyaxes(axis, degrees)}"', name=label))
        assert np.isclose(float(np.linalg.norm(quat)), 1.0), f"MuJoCo compiled {label} to a non-unit {quat}"

    @pytest.mark.parametrize(("label", "axis", "degrees"), ROTATIONS, ids=_ids(ROTATIONS))
    def test_every_rotation_is_far_from_identity(self, tmp_path, label, axis, degrees):
        """A reader that fell back to identity cannot pass any grading by coincidence."""
        quat = _mujoco_body_quat(_write_object(tmp_path, f'xyaxes="{_xyaxes(axis, degrees)}"', name=label))
        assert _same_rotation(quat, IDENTITY) > 0.1, f"{label} compiles to {quat}, too close to identity to grade"

    @pytest.mark.parametrize(("label", "declaration"), ANTIPODAL_ZAXES, ids=_ids(ANTIPODAL_ZAXES))
    def test_mujoco_compiles_every_antipodal_zaxis(self, tmp_path, label, declaration):
        """The axis opposite ``+z`` is a model MuJoCo accepts, not one it refuses."""
        quat = _mujoco_body_quat(_write_object(tmp_path, f'zaxis="{declaration}"', name=f"z_{label}"))
        assert _same_rotation(quat, IDENTITY) > 0.1, f"zaxis {declaration!r} compiles to {quat}, too close to identity"

    @pytest.mark.parametrize(("label", "body_attrs", "compiler"), UNUSABLE, ids=_ids(UNUSABLE))
    def test_mujoco_refuses_every_unusable_declaration(self, tmp_path, label, body_attrs, compiler):
        """These have no compiled answer to grade against, which is why they are
        graded against the tolerated reading instead."""
        path = _write_object(tmp_path, body_attrs, compiler, name=f"u_{label}")
        with pytest.raises(ValueError):
            mujoco.MjModel.from_xml_path(path)


class TestARotationAtAHalfTurnIsReadAsTheCompilerReadsIt:
    """The arms a rotation past a third of a turn reaches, graded against MuJoCo."""

    @pytest.mark.parametrize(("label", "axis", "degrees"), ROTATIONS, ids=_ids(ROTATIONS))
    def test_the_scene_reader_matches_the_compiler(self, tmp_path, label, axis, degrees):
        declaration = _xyaxes(axis, degrees)
        path = _write_object(tmp_path, f'xyaxes="{declaration}"', name=label)
        expected = _mujoco_body_quat(path)
        got = _object_quat(path)
        delta = _same_rotation(got, expected)
        assert delta < 1e-6, (
            f'{label} declared as xyaxes="{declaration}" was reported as '
            f"{tuple(round(v, 6) for v in got)}, but MuJoCo compiles that model with "
            f"{tuple(round(float(v), 6) for v in expected)} (|delta| {delta:.3e})"
        )

    @pytest.mark.parametrize(("label", "axis", "degrees"), ROTATIONS, ids=_ids(ROTATIONS))
    def test_the_robot_reader_matches_the_compiler(self, tmp_path, label, axis, degrees):
        """The same declaration on a robot link, whose orientation Isaac consumes."""
        declaration = _xyaxes(axis, degrees)
        path = _write_link(tmp_path, f'xyaxes="{declaration}"', name=label)
        expected = _mujoco_body_quat(path)
        got = _link_quat(path)
        delta = _same_rotation(got, expected)
        assert delta < 1e-6, (
            f'link {label} declared as xyaxes="{declaration}" was reported as '
            f"{tuple(round(v, 6) for v in got)}, but MuJoCo compiles that model with "
            f"{tuple(round(float(v), 6) for v in expected)} (|delta| {delta:.3e})"
        )

    @pytest.mark.parametrize(("label", "axis", "degrees"), ROTATIONS, ids=_ids(ROTATIONS))
    def test_the_reported_orientation_is_a_unit_quaternion(self, tmp_path, label, axis, degrees):
        """A quaternion off unit norm scales whatever it rotates, so norm is graded
        on its own: the trace form divides by a term that vanishes at a half turn,
        and an arm that divided by it anyway would drift off the unit sphere before
        it drifted far enough to fail the comparison above."""
        got = _object_quat(_write_object(tmp_path, f'xyaxes="{_xyaxes(axis, degrees)}"', name=label))
        assert float(np.linalg.norm(np.asarray(got, dtype=float))) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize(("label", "declaration"), ANTIPODAL_ZAXES, ids=_ids(ANTIPODAL_ZAXES))
    def test_an_antipodal_zaxis_matches_the_compiler(self, tmp_path, label, declaration):
        path = _write_object(tmp_path, f'zaxis="{declaration}"', name=f"z_{label}")
        expected = _mujoco_body_quat(path)
        got = _object_quat(path)
        delta = _same_rotation(got, expected)
        assert delta < 1e-6, (
            f'zaxis="{declaration}" ({label}) was reported as {tuple(round(v, 6) for v in got)}, '
            f"but MuJoCo compiles that model with {tuple(round(float(v), 6) for v in expected)} "
            f"(|delta| {delta:.3e})"
        )


class TestADeclarationWithNoUsableAxisIsIdentity:
    """The tolerated reading for a declaration MuJoCo refuses, arm by arm.

    Both readers resolve orientation through one function, and that the two agree
    on a well-formed declaration is already graded, so the arms below are reached
    through the scene reader alone.
    """

    @pytest.mark.parametrize(("label", "body_attrs", "compiler"), UNUSABLE, ids=_ids(UNUSABLE))
    def test_the_reading_is_identity(self, tmp_path, label, body_attrs, compiler):
        got = _object_quat(_write_object(tmp_path, body_attrs, compiler, name=f"u_{label}"))
        assert got == pytest.approx(IDENTITY), (
            f"{label} was reported as {tuple(round(v, 6) for v in got)}; a declaration naming no "
            "axis the reader can use stays identity, the reading it documents, rather than being "
            "divided by its own zero length"
        )
