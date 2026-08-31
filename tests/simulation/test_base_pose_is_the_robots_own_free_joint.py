"""The reported base pose is the robot's own free joint, not a prop's.

A floating-base robot surfaces its 6-DoF base through four additive keys on
``get_observation`` (``base_pos``, ``base_quat``, ``base_lin_vel``,
``base_ang_vel``) and through ``get_robot_state``'s structured ``base`` entry.
Both read one free joint, and both used to pick it the same wrong way: the loop
that walks ``robot.joint_names`` recorded every free joint it met so it could
skip the degenerate scalar, and the LAST write won. A scene that ships a
free-jointed task object under the robot's own namespace - a kick ball, a
Menagerie grasping cube - puts a second named free joint in that list, after the
base, so the prop won and the robot reported the PROP's pose as its own.

Nothing refused it. Every value stays finite and plausibly shaped, so the rollout
reports success; only the numbers are another body's. ``base_pos`` is documented
for "fall/height tracking", so a duck standing at 0.12 m reported the ball's
0.035 m and read as permanently fallen, and ``base_ang_vel`` - documented as
matching "the IMU-gyro frame WBC/locomotion controllers consume" - carried the
prop's spin.

:meth:`~strands_robots.simulation.mujoco.rendering.RenderingMixin._robot_base_free_joint`
already owned this question and already answered it correctly: its docstring
states the guarantee in as many words, that "a sibling task object -- a
free-jointed cube, including one shipped inside the robot's own MJCF under its
namespace, which is how every Menagerie grasping scene is authored -- carries
neither a declared joint of the robot nor any actuator of it, so it is on no
seed's ancestor chain". The guarantee held; it was simply skipped, because it was
consulted only when the loop had found nothing at all. Both sites now let it
choose, and keep the loop's find only when it declines to name one - so an
UNNAMED base (a mobile base like LeKiwi), which is the case the resolver was
written for, still resolves.

The fixture is the shape the guarantee names rather than the asset that exposed
it: one MJCF carrying a named floating base, a hinge, and a free-jointed prop
declared after them. It needs MuJoCo and nothing else - no registry entry, no
downloaded asset and no policy weights.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots import Robot  # noqa: E402
from strands_robots.simulation.mujoco import rendering as rendering_mod  # noqa: E402
from strands_robots.simulation.mujoco import simulation as simulation_mod  # noqa: E402

# The prop is declared AFTER the base, which is the order that made the last
# write win. Both free joints are named, so both land in ``joint_names``. Its
# orientation is deliberately NOT the identity the trunk starts at: with both
# quaternions equal, a cell reading ``base_quat`` cannot tell the two bodies
# apart and passes whichever one is reported.
_WITH_PROP = """
<mujoco model="probe">
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="trunk" pos="0 0 0.5">
      <freejoint name="trunk_free"/>
      <geom name="trunk_geom" type="box" size="0.05 0.05 0.05" mass="1"/>
      <body name="link" pos="0 0 -0.1">
        <joint name="elbow" type="hinge" axis="0 1 0"/>
        <geom name="link_geom" type="capsule" fromto="0 0 0 0 0 -0.1" size="0.02" mass="0.2"/>
      </body>
    </body>
    <body name="prop" pos="0.7 0.3 0.2" quat="0.7071068 0 0 0.7071068">
      <freejoint name="prop_free"/>
      <geom name="prop_geom" type="sphere" size="0.04" mass="0.05"/>
    </body>
  </worldbody>
  <actuator><position name="elbow_act" joint="elbow"/></actuator>
</mujoco>
"""

_NO_PROP = _WITH_PROP.replace(
    """    <body name="prop" pos="0.7 0.3 0.2" quat="0.7071068 0 0 0.7071068">
      <freejoint name="prop_free"/>
      <geom name="prop_geom" type="sphere" size="0.04" mass="0.05"/>
    </body>
""",
    "",
)

_BASE_HEIGHT = 0.5
_PROP_POS = (0.7, 0.3, 0.2)


def _scene(tmp_path: Path, body: str, name: str) -> Any:
    """Build and reset a sim on ``body``.

    The handle is ``Any``: :func:`~strands_robots.Robot` is a factory function
    rather than a class, so naming it in a type position is a ``valid-type``
    error and its inferred return exposes none of the engine's methods.
    """
    path = tmp_path / name
    path.write_text(body)
    sim: Any = Robot("probe", urdf_path=str(path), mesh=False)
    sim.reset()
    return sim


def _state_base(sim: Any) -> dict[str, Any] | None:
    """The ``base`` entry of ``get_robot_state``, or ``None`` when absent."""
    reported = sim.get_robot_state(robot_name="probe")
    payload = next((b["json"] for b in reported.get("content", []) if "json" in b), reported)
    return payload.get("base") if isinstance(payload, dict) else None


def _dedent(source: str) -> str:
    import textwrap

    return textwrap.dedent(source)


class TestThePropIsNotTheBase:
    """The regression: a namespaced free-jointed prop is not the robot's base."""

    def test_get_observation_reports_the_robots_own_base(self, tmp_path: Path) -> None:
        """``base_pos`` is the trunk's height, not the prop's."""
        sim = _scene(tmp_path, _WITH_PROP, "with_prop.xml")
        try:
            reported = sim.get_observation(robot_name="probe", skip_images=True)["base_pos"]
            assert reported == pytest.approx([0.0, 0.0, _BASE_HEIGHT], abs=1e-6), reported
            assert reported != pytest.approx(list(_PROP_POS), abs=1e-6)
        finally:
            sim.destroy()

    def test_get_robot_state_reports_the_robots_own_base(self, tmp_path: Path) -> None:
        """The structured ``base`` entry names the trunk too."""
        sim = _scene(tmp_path, _WITH_PROP, "with_prop.xml")
        try:
            base = _state_base(sim)
            assert base is not None
            assert base["position"] == pytest.approx([0.0, 0.0, _BASE_HEIGHT], abs=1e-6)
        finally:
            sim.destroy()

    def test_the_two_surfaces_agree(self, tmp_path: Path) -> None:
        """One base, two readers: they must report the same body."""
        sim = _scene(tmp_path, _WITH_PROP, "with_prop.xml")
        try:
            observed = sim.get_observation(robot_name="probe", skip_images=True)
            base = _state_base(sim)
            assert base is not None
            assert observed["base_pos"] == pytest.approx(base["position"], abs=1e-9)
            assert observed["base_quat"] == pytest.approx(base["quaternion"], abs=1e-9)
        finally:
            sim.destroy()

    def test_every_base_key_comes_from_the_same_free_joint(self, tmp_path: Path) -> None:
        """All four keys read one address, so the prop must not supply any.

        The prop carries a quarter turn about z, so ``base_quat`` distinguishes
        the two bodies rather than agreeing with either by coincidence.
        """
        sim = _scene(tmp_path, _WITH_PROP, "with_prop.xml")
        try:
            observed = sim.get_observation(robot_name="probe", skip_images=True)
            assert observed["base_quat"] == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-6)
            assert observed["base_lin_vel"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
            assert observed["base_ang_vel"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
        finally:
            sim.destroy()


class TestThePremiseTheDefectNeeded:
    """The fixture really is the shape that made the last write win."""

    def test_the_scene_declares_two_named_free_joints_with_the_prop_last(self, tmp_path: Path) -> None:
        """Both free joints reach ``joint_names``, and the prop's is later."""
        import mujoco

        sim = _scene(tmp_path, _WITH_PROP, "with_prop.xml")
        try:
            model = sim._world._model
            robot = sim._world.robots["probe"]
            prefix = robot.namespace or ""
            free = [
                (index, name)
                for index, name in enumerate(robot.joint_names)
                if (jid := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)) >= 0
                and model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE
            ]
            assert [name for _, name in free] == ["trunk_free", "prop_free"], free
            assert free[0][0] < free[1][0], "the prop must come last for last-wins to pick it"
        finally:
            sim.destroy()


class TestTheChoiceHasOneOwner:
    """Neither reader re-derives which free joint is the base."""

    @pytest.mark.parametrize(
        ("label", "function"),
        [
            ("get_observation", rendering_mod.RenderingMixin._get_sim_observation),
            ("get_robot_state", simulation_mod.MuJoCoSimEngine.get_robot_state),
        ],
    )
    def test_the_resolvers_answer_wins(self, label: str, function: Any) -> None:
        """The ownership resolver is consulted unconditionally, and last.

        Two properties, because either alone is satisfiable by the defect. The
        call must not sit under a test on ``free_jnt_id`` - that is exactly the
        ``if free_jnt_id < 0`` that skipped it whenever the loop had already
        picked a prop - and its adoption must come after the loop, so that the
        resolver's answer is the one that survives.
        """
        source = _dedent(inspect.getsource(function))
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and "_robot_base_free_joint" in ast.unparse(node.func)
        ]
        assert len(calls) == 1, (label, len(calls))

        gated = [
            ast.unparse(branch.test)
            for branch in ast.walk(tree)
            if isinstance(branch, ast.If)
            and "free_jnt_id" in ast.unparse(branch.test)
            and any(node is calls[0] for node in ast.walk(branch))
        ]
        assert gated == [], (label, "the resolver is consulted conditionally", gated)

        assignments = {
            ast.unparse(node.value): node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "free_jnt_id"
        }
        assert "jnt_id" in assignments, (label, assignments)
        assert "owned_free_jnt_id" in assignments, (label, assignments)
        assert assignments["owned_free_jnt_id"] > assignments["jnt_id"], (label, assignments)


class TestWhatIsUnchanged:
    """A scene with one free joint, and a fixed-base arm, read as before."""

    def test_a_single_free_joint_scene_is_unaffected(self, tmp_path: Path) -> None:
        """With no prop the answer is the same trunk."""
        sim = _scene(tmp_path, _NO_PROP, "no_prop.xml")
        try:
            observed = sim.get_observation(robot_name="probe", skip_images=True)
            assert observed["base_pos"] == pytest.approx([0.0, 0.0, _BASE_HEIGHT], abs=1e-6)
        finally:
            sim.destroy()

    def test_a_fixed_base_arm_still_reports_no_base_keys(self) -> None:
        """The base keys stay absent for a robot that has no floating base."""
        sim: Any = Robot("so101", mesh=False)
        try:
            sim.reset()
            observed = sim.get_observation(robot_name="so101", skip_images=True)
            assert "base_pos" not in observed
            assert _state_base(sim) is None
        finally:
            sim.destroy()
