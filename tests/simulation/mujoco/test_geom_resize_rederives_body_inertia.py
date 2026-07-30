# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Resizing a geom re-derives the owning body's mass, center of mass and inertia.

A body that declares no ``<inertial>`` takes its whole inertial row from the shapes
it owns, integrated once by the compiler. ``model.body_mass`` / ``body_ipos`` /
``body_iquat`` / ``body_inertia`` are therefore derived from ``geom_size``, and no
``mj_forward`` / ``mj_step`` recomputes them - so ``set_geom_properties(size=...)``
resized the geom and left the body describing the shape it used to have:

    add_object("cube", box, size=0.1, mass=1.0)     body_inertia 0.0016667
    set_geom_properties(size=[0.2] * 3)             body_inertia 0.0016667  (!)
    a fresh compile at those extents                body_inertia 0.0266667

The body then collided as the new shape while resisting rotation as the old one:
0.1 Nm for 1 s spun the resized cube to 60.0 rad/s where its geometry implies
3.75 rad/s. On a body whose geom carries a density rather than a mass, the stale
row also kept the old mass and the old balance point - a dumbbell whose left
weight was grown stayed 2.22 kg and perfectly balanced instead of 15.52 kg with
its center of mass 17 cm off axis.

Worse, it was order-dependent. The size is recorded in the spec, so any later
scene mutation recompiles it and the compiler silently corrects the tensor: the
same two calls produced 16x different physics depending on whether an unrelated
``add_object`` happened afterwards.

These pin the contract: the inertial row the setter leaves behind is the one a
fresh compile of that geometry produces, for every size-defined primitive, for a
body with several geoms, and whether or not anything follows the resize. A body
that declares its own ``<inertial>`` takes nothing from geometry and is left
alone, and a resize whose geometry cannot be compiled is refused with the spec and
the model still describing the same shape.

The refresh also has to leave the running scene alone. ``set_geom_properties`` is
the documented mid-run domain-randomization path, so it is issued on a scene that
has already been stepped, and re-deriving the row must not move anything: the
reference constants the solver scales with are evaluated at the model's reference
configuration, so the MuJoCo call that recomputes them writes ``qpos0`` into
whatever ``mjData`` it is handed. Handed the scene's own, a resize rewound every
joint and body pose to its declared value while leaving ``qvel`` untouched - a
state the scene never occupied, reported as ``status="success"``. Every test above
resizes at ``t=0``, where ``qpos == qpos0``, so none of them can observe it; the
two below run the scene first.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from strands_robots.simulation.mujoco import Simulation

# The helpers below take a sim un-annotated: ``Simulation`` is re-exported lazily,
# so it is a module attribute rather than a name a type checker can resolve.
_INERTIAL_ARRAYS = ("body_mass", "body_ipos", "body_iquat", "body_inertia")

# Derived from the inertial rows at compile time and not refreshed by a step:
# body_subtreemass, and the reference constants the constraint solver scales with.
_INERTIA_DERIVED_ARRAYS = ("body_subtreemass", "body_invweight0", "dof_invweight0", "dof_M0")

# One free body carrying one geom, declared with a density so that a resize moves
# the body's mass as well as its inertia.
_ONE_GEOM = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <geom name="ground" type="plane" size="5 5 0.1"/>
    <body name="obj" pos="0 0 1"><freejoint/>
      <geom name="obj_g" type="{gtype}" size="{size}" density="1200"/>
    </body>
  </worldbody>
</mujoco>
"""

# A body whose inertia comes from several geoms, beside one that declares its own
# <inertial> and so takes nothing from geometry.
_MULTI_GEOM = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <geom name="ground" type="plane" size="5 5 0.1"/>
    <body name="dumbbell" pos="0 0 1"><freejoint/>
      <geom name="left_weight" type="sphere" size="0.05" pos="-0.2 0 0" density="2000"/>
      <geom name="right_weight" type="sphere" size="0.05" pos="0.2 0 0" density="2000"/>
    </body>
    <body name="declared" pos="1 0 1"><freejoint/>
      <inertial pos="0 0 0" mass="2.0" diaginertia="0.5 0.5 0.5"/>
      <geom name="declared_g" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

# An articulated body beside two free ones, under gravity so that running the scene
# moves every joint away from the pose it was declared at.
_RUNNING_SCENE = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="ground" type="plane" size="5 5 0.1"/>
    <body name="arm" pos="0 0 0.55">
      <joint name="pan" type="hinge" axis="0 0 1"/>
      <geom name="link" type="capsule" fromto="0 0 0 0.34 0 0" size="0.035" density="900"/>
    </body>
    <body name="crate" pos="0.42 0.3 0.09"><freejoint/>
      <geom name="crate_g" type="box" size="0.07 0.07 0.07" density="900"/>
    </body>
    <body name="ball" pos="-0.3 -0.26 0.07"><freejoint/>
      <geom name="ball_g" type="sphere" size="0.06" density="700"/>
    </body>
  </worldbody>
</mujoco>
"""

# (geom type, declared size, resized size) for every size-defined primitive.
_PRIMITIVES = [
    ("box", "0.1 0.15 0.2", [0.2, 0.05, 0.3]),
    ("sphere", "0.1", [0.17]),
    ("capsule", "0.05 0.2", [0.09, 0.31]),
    ("cylinder", "0.05 0.2", [0.11, 0.07]),
    ("ellipsoid", "0.1 0.2 0.3", [0.25, 0.06, 0.14]),
]


def _scene(sim, xml: str):
    assert sim.create_world()["status"] == "success"
    assert sim.replace_scene_mjcf(xml)["status"] == "success"
    return sim


def _body_id(sim, name: str) -> int:
    return int(mujoco.mj_name2id(sim._world._model, mujoco.mjtObj.mjOBJ_BODY, name))


def _inertial_row(sim, name: str) -> dict[str, np.ndarray]:
    model, body_id = sim._world._model, _body_id(sim, name)
    # body_mass is a scalar per body; atleast_1d keeps every row uniformly indexable.
    return {array: np.atleast_1d(np.array(getattr(model, array)[body_id], dtype=float)) for array in _INERTIAL_ARRAYS}


def _pose(sim, name: str) -> dict[str, list[float]]:
    """The body's pose as a caller reads it, rather than as the model stores it."""
    state = sim.get_body_state(body_name=name)
    payload = next(block["json"] for block in state["content"] if "json" in block)
    return {"position": list(payload["position"]), "quaternion": list(payload["quaternion"])}


def _run_scene(sim):
    """Move the scene off the pose it was declared at, then let it settle.

    The hinge is driven away from zero and the two free bodies fall onto the ground,
    so every element of ``qpos`` differs from ``qpos0`` - which is the only state in
    which a rewind is observable.
    """
    _scene(sim, _RUNNING_SCENE)
    assert sim.set_joint_positions({"pan": 1.05})["status"] == "success"
    assert sim.step(400)["status"] == "success"
    data = sim._world._data
    assert not np.allclose(data.qpos, sim._world._model.qpos0), "the scene never left its declared pose"
    return data


def _spin(sim, body: str, torque: float, steps: int) -> tuple[float, float]:
    """Spin a free body with a constant torque about x.

    The scenes are declared with zero gravity, so the body is in free space and its
    angular velocity is the torque divided by the inertia the model holds - which is
    the quantity under test. A body resting on the ground would instead measure its
    contact, and would read near zero whatever the inertia was.

    Returns:
        ``(angular velocity about x, elapsed sim seconds)``.
    """
    assert sim.apply_force(body_name=body, torque=[torque, 0.0, 0.0])["status"] == "success"
    assert sim.step(steps)["status"] == "success"
    state = sim.get_body_state(body_name=body)
    payload = next(block["json"] for block in state["content"] if "json" in block)
    elapsed = steps * float(sim._world._model.opt.timestep)
    return float(payload["angular_velocity"][0]), elapsed


@pytest.fixture
def resized_and_declared(request):
    """A resized scene and an independent one declaring the resized geometry.

    The second scene never calls the setter, so it is the compiler's own answer for
    those extents rather than a hand-derived formula - which is what makes the
    comparison able to catch a per-primitive mistake.
    """
    gtype, declared, resized = request.param
    grown = Simulation(tool_name=f"grown_{gtype}", mesh=False)
    reference = Simulation(tool_name=f"reference_{gtype}", mesh=False)
    _scene(grown, _ONE_GEOM.format(gtype=gtype, size=declared))
    _scene(reference, _ONE_GEOM.format(gtype=gtype, size=" ".join(str(v) for v in resized)))
    yield grown, reference, resized
    grown.cleanup()
    reference.cleanup()


@pytest.mark.parametrize("resized_and_declared", _PRIMITIVES, ids=[p[0] for p in _PRIMITIVES], indirect=True)
def test_a_resized_primitive_carries_the_inertia_of_its_new_shape(resized_and_declared):
    grown, reference, resized = resized_and_declared
    stale = _inertial_row(grown, "obj")

    assert grown.set_geom_properties(geom_name="obj_g", size=resized)["status"] == "success"

    expected = _inertial_row(reference, "obj")
    actual = _inertial_row(grown, "obj")
    for array in _INERTIAL_ARRAYS:
        assert actual[array] == pytest.approx(expected[array], abs=1e-12), array
    # The resize has to have moved something, or the comparison proves nothing.
    assert actual["body_inertia"] != pytest.approx(stale["body_inertia"], abs=1e-12)


@pytest.mark.parametrize("resized_and_declared", _PRIMITIVES, ids=[p[0] for p in _PRIMITIVES], indirect=True)
def test_the_constants_the_solver_scales_with_follow_the_new_inertia(resized_and_declared):
    grown, reference, resized = resized_and_declared

    assert grown.set_geom_properties(geom_name="obj_g", size=resized)["status"] == "success"

    for array in _INERTIA_DERIVED_ARRAYS:
        assert np.asarray(getattr(grown._world._model, array), dtype=float) == pytest.approx(
            np.asarray(getattr(reference._world._model, array), dtype=float), abs=1e-12
        ), array


def test_a_resized_body_rotates_as_its_new_geometry_implies():
    """The observable consequence: a constant torque spins it at torque / inertia.

    A cube's inertia is isotropic, so a torque about x stays on a principal axis and
    the exact answer is ``torque * t / inertia``. Asserting that identity - rather
    than only that the two runs agree - is what makes the measurement independent of
    the value under test, and what fails loudly if the body ever stops spinning
    freely and the comparison starts measuring something else.
    """
    grown = Simulation(tool_name="grown_spin", mesh=False)
    reference = Simulation(tool_name="reference_spin", mesh=False)
    try:
        _scene(grown, _ONE_GEOM.format(gtype="box", size="0.05 0.05 0.05"))
        _scene(reference, _ONE_GEOM.format(gtype="box", size="0.2 0.2 0.2"))
        stale_inertia = _inertial_row(grown, "obj")["body_inertia"][0]
        assert grown.set_geom_properties(geom_name="obj_g", size=[0.2, 0.2, 0.2])["status"] == "success"

        measured, elapsed = _spin(grown, "obj", 0.1, 500)

        resized_inertia = _inertial_row(reference, "obj")["body_inertia"][0]
        assert measured == pytest.approx(0.1 * elapsed / resized_inertia, rel=1e-3)
        assert measured == pytest.approx(_spin(reference, "obj", 0.1, 500)[0], rel=1e-9)
        # The stale row would have spun it 16x faster, so the two are far apart and
        # the assertion above cannot be satisfied by both.
        assert 0.1 * elapsed / stale_inertia > 10 * measured
    finally:
        grown.cleanup()
        reference.cleanup()


def test_the_resize_means_the_same_thing_whether_or_not_anything_follows_it():
    """A later unrelated mutation recompiles the spec; both paths must agree."""
    alone = Simulation(tool_name="resize_alone", mesh=False)
    followed = Simulation(tool_name="resize_followed", mesh=False)
    try:
        for sim in (alone, followed):
            _scene(sim, _ONE_GEOM.format(gtype="box", size="0.05 0.05 0.05"))
            assert sim.set_geom_properties(geom_name="obj_g", size=[0.2, 0.2, 0.2])["status"] == "success"
        assert (
            followed.add_object(name="unrelated", shape="sphere", size=[0.02] * 3, position=[2.0, 2.0, 2.0])["status"]
            == "success"
        )

        for array in _INERTIAL_ARRAYS:
            assert _inertial_row(alone, "obj")[array] == pytest.approx(
                _inertial_row(followed, "obj")[array], abs=1e-12
            ), array
        spun_alone, elapsed = _spin(alone, "obj", 0.1, 500)
        assert spun_alone == pytest.approx(_spin(followed, "obj", 0.1, 500)[0], rel=1e-9)
        # Holding the exact torque / inertia identity is what rules out the agreement
        # being two bodies that both failed to rotate: a body resting on the ground
        # reads four orders of magnitude below it.
        assert spun_alone == pytest.approx(0.1 * elapsed / _inertial_row(alone, "obj")["body_inertia"][0], rel=1e-3)
    finally:
        alone.cleanup()
        followed.cleanup()


def test_growing_one_geom_of_a_body_moves_its_mass_and_balance_point():
    grown = Simulation(tool_name="grown_dumbbell", mesh=False)
    reference = Simulation(tool_name="reference_dumbbell", mesh=False)
    try:
        _scene(grown, _MULTI_GEOM)
        _scene(
            reference,
            _MULTI_GEOM.replace(
                'name="left_weight" type="sphere" size="0.05"', 'name="left_weight" type="sphere" size="0.12"'
            ),
        )
        before = _inertial_row(grown, "dumbbell")

        assert grown.set_geom_properties(geom_name="left_weight", size=[0.12])["status"] == "success"

        expected, actual = _inertial_row(reference, "dumbbell"), _inertial_row(grown, "dumbbell")
        for array in _INERTIAL_ARRAYS:
            assert actual[array] == pytest.approx(expected[array], abs=1e-12), array
        # The two properties a single-geom body would not have exercised.
        assert actual["body_mass"][0] > before["body_mass"][0] * 2
        assert abs(actual["body_ipos"][0]) > 0.1
    finally:
        grown.cleanup()
        reference.cleanup()


def test_a_body_that_declares_its_own_inertial_is_left_alone():
    """Its geoms do not define its inertia, so a resize cannot make the row stale."""
    sim = Simulation(tool_name="declared_inertial", mesh=False)
    try:
        _scene(sim, _MULTI_GEOM)
        before = _inertial_row(sim, "declared")

        assert sim.set_geom_properties(geom_name="declared_g", size=[0.2, 0.2, 0.2])["status"] == "success"

        after = _inertial_row(sim, "declared")
        for array in _INERTIAL_ARRAYS:
            assert after[array] == pytest.approx(before[array], abs=1e-12), array
        assert after["body_mass"][0] == pytest.approx(2.0)
    finally:
        sim.cleanup()


def test_only_a_resize_of_geometry_derived_inertia_pays_a_compile(monkeypatch):
    """The compile is the cost of re-deriving; nothing else should be charged it."""
    sim = Simulation(tool_name="compile_budget", mesh=False)
    try:
        _scene(sim, _MULTI_GEOM)
        compiles: list[int] = []
        original = mujoco.MjSpec.copy

        def counting_copy(self):
            compiles.append(1)
            return original(self)

        monkeypatch.setattr(mujoco.MjSpec, "copy", counting_copy)

        assert sim.set_geom_properties(geom_name="left_weight", color=[1.0, 0.0, 0.0])["status"] == "success"
        assert sim.set_geom_properties(geom_name="left_weight", friction=[1.0, 0.5, 0.001])["status"] == "success"
        assert compiles == []

        assert sim.set_geom_properties(geom_name="declared_g", size=[0.2, 0.2, 0.2])["status"] == "success"
        assert compiles == []

        assert sim.set_geom_properties(geom_name="left_weight", size=[0.07])["status"] == "success"
        assert len(compiles) == 1
    finally:
        sim.cleanup()


def test_a_resize_that_cannot_be_re_derived_is_refused_with_both_forms_in_step(monkeypatch):
    """Neither representation may be left describing a shape the other does not."""
    sim = Simulation(tool_name="refresh_refused", mesh=False)
    try:
        _scene(sim, _MULTI_GEOM)
        model = sim._world._model
        geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_weight"))
        spec_geom = sim._world._backend_state["spec"].geoms[geom_id]
        before_model = np.array(model.geom_size[geom_id], dtype=float)
        before_spec = np.array(spec_geom.size, dtype=float)
        before_row = _inertial_row(sim, "dumbbell")

        def refusing_copy(self):
            raise ValueError("synthetic compiler refusal")

        monkeypatch.setattr(mujoco.MjSpec, "copy", refusing_copy)

        result = sim.set_geom_properties(geom_name="left_weight", size=[0.12])

        assert result["status"] == "error"
        assert "synthetic compiler refusal" in result["content"][0]["text"]
        assert np.array(model.geom_size[geom_id], dtype=float) == pytest.approx(before_model)
        assert np.array(spec_geom.size, dtype=float) == pytest.approx(before_spec)
        for array in _INERTIAL_ARRAYS:
            assert _inertial_row(sim, "dumbbell")[array] == pytest.approx(before_row[array], abs=1e-12), array
    finally:
        sim.cleanup()


def test_a_mid_run_resize_leaves_the_scene_where_it_was():
    """A resize reports the new geometry; it must not also move the scene.

    Recomputing the solver's reference constants evaluates them at the model's
    reference configuration, so the MuJoCo call that does it overwrites whatever
    ``mjData`` it is handed with ``qpos0``. Handed the scene's own, this resize
    teleported the swung arm back to zero and dropped both objects back to their
    declared heights while their velocities stayed as they were.
    """
    sim = Simulation(tool_name="mid_run_resize", mesh=False)
    try:
        data = _run_scene(sim)
        before_poses = {name: _pose(sim, name) for name in ("arm", "crate", "ball")}
        before_qpos = np.array(data.qpos, dtype=float)
        before_qvel = np.array(data.qvel, dtype=float)

        assert sim.set_geom_properties(geom_name="crate_g", size=[0.12, 0.12, 0.12])["status"] == "success"

        for name, before in before_poses.items():
            after = _pose(sim, name)
            assert after["position"] == pytest.approx(before["position"], abs=1e-9), name
            assert after["quaternion"] == pytest.approx(before["quaternion"], abs=1e-9), name
        assert np.array(data.qpos, dtype=float) == pytest.approx(before_qpos, abs=1e-12)
        assert np.array(data.qvel, dtype=float) == pytest.approx(before_qvel, abs=1e-12)
    finally:
        sim.cleanup()


def test_a_mid_run_resize_still_re_derives_the_inertia():
    """Leaving the scene alone may not cost the refresh the resize exists for.

    Pinned beside the rewind so the state above cannot be preserved by simply not
    refreshing: a resize issued on a running scene owes the same inertial row, and
    the same solver reference constants, as one issued before the first step.
    """
    sim = Simulation(tool_name="mid_run_refresh", mesh=False)
    reference = Simulation(tool_name="mid_run_reference", mesh=False)
    try:
        _run_scene(sim)
        _scene(reference, _RUNNING_SCENE.replace('size="0.07 0.07 0.07"', 'size="0.12 0.12 0.12"'))

        assert sim.set_geom_properties(geom_name="crate_g", size=[0.12, 0.12, 0.12])["status"] == "success"

        expected = _inertial_row(reference, "crate")
        for array in _INERTIAL_ARRAYS:
            assert _inertial_row(sim, "crate")[array] == pytest.approx(expected[array], abs=1e-12), array
        for array in _INERTIA_DERIVED_ARRAYS:
            assert np.asarray(getattr(sim._world._model, array)) == pytest.approx(
                np.asarray(getattr(reference._world._model, array)), abs=1e-12
            ), array
    finally:
        sim.cleanup()
        reference.cleanup()
