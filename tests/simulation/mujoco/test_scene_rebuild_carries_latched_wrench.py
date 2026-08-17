"""A scene rebuild keeps every latched ``apply_force`` wrench latched.

:meth:`~strands_robots.simulation.mujoco.MuJoCoSimEngine.apply_force` latches a
wrench in the target body's own ``xfrc_applied`` row and documents exactly two
ways that latch ends: the next ``apply_force`` on the same body, or a
``reset()``. Five other operations ended it too, because every one of them
rebuilds the model and ``xfrc_applied`` is not carried across that rebuild --
``spec.recompile`` returns the whole buffer zeroed, and the eject path allocates
a fresh ``MjData``. Each op reported ``"success"``, nothing was logged, and the
wrench was simply not applied on any later step: an object a thruster or a wind
field was holding up began to fall one ``add_camera`` later.

The contract is stated three times over, from three directions:

* ``apply_force`` names the two things that end a latch, and a scene rebuild is
  neither of them.
* ``add_robot`` documents that its composition preserves the scene's dynamic
  state and says so in these words: "a latched ``apply_force`` wrench persists".
* ``_SceneState`` -- the eject path's carry -- documents itself as "the same
  state ``save_state`` checkpoints, minus the entries a rebuilt scene cannot
  have a surviving instance of", and it already carried ``qfrc_applied``, the
  joint-space sibling of this very buffer. A body survives a rebuild, so its
  wrench had no reason to be the omitted one.

Each preservation test below is paired with a control that the latch still ends
where it is documented to, so the fix is pinned to carry the wrench across a
rebuild rather than to make it unclearable.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# A one-hinge arm, only present so ``remove_robot`` has something to eject.
_ARM_XML = """
<mujoco model="stub_arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link0" pos="0 0 0.1">
      <joint name="pan" type="hinge" axis="0 0 1" damping="1.0"/>
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

_PUCK_MASS = 0.5
_PUCK_Z = 0.5
# The wrench that exactly cancels gravity on the puck. With it latched the puck
# hovers; without it the puck is in free fall, so "is the wrench still latched"
# is answered by where the puck is rather than by reading a buffer.
_HOVER_FORCE = [0.0, 0.0, _PUCK_MASS * 9.81]
# 0.2 s of free fall is ~0.196 m - two orders of magnitude above the hover
# tolerance, and short enough that the puck cannot reach the ground from _PUCK_Z.
_FALL_STEPS = 100
_HOVER_TOL = 1e-3

# Every operation that rebuilds the model. ``set_geom_properties`` deliberately
# does not appear: it mutates the compiled model in place, so it never lost the
# wrench and is a control below.
_REBUILD_OPS = ("add_robot", "add_object", "add_camera", "remove_object", "remove_robot")


@pytest.fixture
def sim():
    s = Simulation(tool_name="devx_rebuild_wrench", mesh=False)
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.5)


def _arm_path(tmp_path) -> str:
    path = tmp_path / "stub_arm.xml"
    path.write_text(_ARM_XML)
    return str(path)


def _wrench(sim: Simulation, body: str) -> list[float]:
    """Read the wrench latched on ``body``, straight out of the live data."""
    world = sim._world
    assert world is not None and world._model is not None and world._data is not None
    mj = sim._mj
    bid = mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_BODY, body)
    assert bid >= 0, f"body {body!r} missing from the compiled model"
    return [float(v) for v in world._data.xfrc_applied[bid]]


def _latched_bodies(sim: Simulation) -> dict[str, list[float]]:
    """Every body carrying a non-zero wrench, keyed by name."""
    world = sim._world
    assert world is not None and world._model is not None and world._data is not None
    mj = sim._mj
    model, data = world._model, world._data
    return {
        (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) or f"<unnamed {bid}>"): [
            float(v) for v in data.xfrc_applied[bid]
        ]
        for bid in range(int(model.nbody))
        if data.xfrc_applied[bid].any()
    }


def _height(sim: Simulation, body: str) -> float:
    result = sim.get_body_state(body)
    assert result["status"] == "success", result
    return float([block["json"] for block in result["content"] if "json" in block][0]["position"][2])


def _scene_with_a_hovering_puck(sim: Simulation, tmp_path) -> None:
    """Build a scene whose puck hovers only because a wrench is holding it.

    Everything a later rebuild needs to act on is added FIRST, so the wrench is
    latched after the last setup-driven rebuild and nothing but the operation
    under test can clear it.
    """
    sim.create_world()
    assert sim.add_robot(name="doomed", urdf_path=_arm_path(tmp_path))["status"] == "success"
    assert sim.add_object(name="spare", shape="sphere", size=[0.04], position=[-0.6, 0.0, 0.3])["status"] == "success"
    assert (
        sim.add_object(name="puck", shape="box", size=[0.05] * 3, position=[0.0, 0.0, _PUCK_Z], mass=_PUCK_MASS)[
            "status"
        ]
        == "success"
    )
    assert sim.apply_force("puck", force=_HOVER_FORCE)["status"] == "success"
    # Premise: the wrench really is holding the puck up, so a fall below means
    # the wrench went missing rather than that it never held anything.
    start = _height(sim, "puck")
    sim.step(_FALL_STEPS)
    assert _height(sim, "puck") == pytest.approx(start, abs=_HOVER_TOL), "premise: the puck is hovering"


def _rebuild(sim: Simulation, op: str, tmp_path) -> None:
    """Run one model-rebuilding operation and assert it reported success."""
    if op == "add_robot":
        result = sim.add_robot(name="newcomer", urdf_path=_arm_path(tmp_path), position=[0.8, 0.0, 0.0])
    elif op == "add_object":
        result = sim.add_object(name="newcomer", shape="sphere", size=[0.03], position=[0.6, 0.0, 0.3])
    elif op == "add_camera":
        result = sim.add_camera(name="newcomer", position=[1.0, -1.0, 0.7], target=[0.0, 0.0, 0.1])
    elif op == "remove_object":
        result = sim.remove_object("spare")
    elif op == "remove_robot":
        result = sim.remove_robot("doomed")
    else:  # pragma: no cover - guards the parametrization against a typo
        raise AssertionError(f"unknown rebuild op {op!r}")
    assert result["status"] == "success", result


class TestARebuildKeepsTheWrenchHoldingTheBodyUp:
    """The observable half: a body a wrench was holding does not start to fall.

    Pre-fix every one of these ops dropped the wrench while reporting success,
    so the puck resumed free fall and was ~0.196 m lower 0.2 s later.
    """

    @pytest.mark.parametrize("op", _REBUILD_OPS)
    def test_the_puck_keeps_hovering_across(self, sim: Simulation, tmp_path, op: str) -> None:
        _scene_with_a_hovering_puck(sim, tmp_path)
        held = _height(sim, "puck")

        _rebuild(sim, op, tmp_path)
        sim.step(_FALL_STEPS)

        assert _height(sim, "puck") == pytest.approx(held, abs=_HOVER_TOL), (
            f"{op} dropped the latched wrench: the puck fell {held - _height(sim, 'puck'):.4f} m"
        )

    @pytest.mark.parametrize("op", _REBUILD_OPS)
    def test_the_wrench_row_survives(self, sim: Simulation, tmp_path, op: str) -> None:
        _scene_with_a_hovering_puck(sim, tmp_path)

        _rebuild(sim, op, tmp_path)

        assert _wrench(sim, "puck") == pytest.approx([*_HOVER_FORCE, 0.0, 0.0, 0.0])


class TestPerBodyIsolationSurvivesTheRebuild:
    """Each body keeps its OWN wrench across a rebuild, not a shared one.

    ``xfrc_applied`` is indexed by body and a rebuild renumbers bodies, so a
    carry that used the old index would re-latch one body's wrench onto another.
    """

    def test_two_bodies_keep_their_own_wrenches(self, sim: Simulation, tmp_path) -> None:
        sim.create_world(gravity=[0.0, 0.0, 0.0])
        for name, x in (("puck_a", 0.0), ("puck_b", 1.0)):
            assert (
                sim.add_object(name=name, shape="box", size=[0.05] * 3, position=[x, 0.0, 0.3], mass=0.1)["status"]
                == "success"
            )
        assert sim.apply_force("puck_a", force=[1.0, 0.0, 0.0])["status"] == "success"
        assert sim.apply_force("puck_b", torque=[0.0, 0.0, 2.0])["status"] == "success"

        assert sim.add_robot(name="newcomer", urdf_path=_arm_path(tmp_path))["status"] == "success"

        assert _wrench(sim, "puck_a") == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert _wrench(sim, "puck_b") == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 2.0])


class TestTheLatchStillEndsWhereItIsDocumentedTo:
    """Controls: carrying the wrench across a rebuild must not make it unclearable.

    These three are the documented ways a latch begins, persists and ends. They
    hold both before and after the fix - they are here so that widening the
    carry into "the wrench can never be cleared" fails.
    """

    def test_reset_clears_every_latched_wrench(self, sim: Simulation, tmp_path) -> None:
        _scene_with_a_hovering_puck(sim, tmp_path)

        assert sim.reset()["status"] == "success"

        assert _latched_bodies(sim) == {}

    def test_stepping_keeps_the_wrench_latched(self, sim: Simulation, tmp_path) -> None:
        _scene_with_a_hovering_puck(sim, tmp_path)

        sim.step(_FALL_STEPS)

        assert _wrench(sim, "puck") == pytest.approx([*_HOVER_FORCE, 0.0, 0.0, 0.0])

    def test_a_zero_wrench_on_the_same_body_releases_it(self, sim: Simulation, tmp_path) -> None:
        _scene_with_a_hovering_puck(sim, tmp_path)
        held = _height(sim, "puck")

        assert sim.apply_force("puck", force=[0.0, 0.0, 0.0])["status"] == "success"
        assert (
            sim.add_object(name="newcomer", shape="sphere", size=[0.03], position=[0.6, 0.0, 0.3])["status"]
            == "success"
        )
        sim.step(_FALL_STEPS)

        assert _height(sim, "puck") < held - 0.1, "a released puck must fall, rebuild or not"


class TestAWrenchIsNeverInventedOrMisattributed:
    """Controls on the two ways a name-keyed carry could go wrong.

    Both hold before the fix - before it, no wrench was carried at all - so they
    pin that the carry adds only the wrenches that were really latched.
    """

    def test_a_removed_robots_wrench_does_not_land_on_a_survivor(self, sim: Simulation, tmp_path) -> None:
        sim.create_world(gravity=[0.0, 0.0, 0.0])
        arm = _arm_path(tmp_path)
        assert sim.add_robot(name="keeper", urdf_path=arm)["status"] == "success"
        assert sim.add_robot(name="doomed", urdf_path=arm, position=[0.5, 0.0, 0.0])["status"] == "success"
        assert sim.apply_force("doomed/link0", force=[3.0, 0.0, 0.0])["status"] == "success"

        assert sim.remove_robot("doomed")["status"] == "success"

        assert _latched_bodies(sim) == {}, "the ejected body's wrench must not be re-latched anywhere"

    def test_a_wrenchless_scene_gains_no_wrench(self, sim: Simulation, tmp_path) -> None:
        sim.create_world()
        assert (
            sim.add_object(name="puck", shape="box", size=[0.05] * 3, position=[0.0, 0.0, 0.3])["status"] == "success"
        )

        assert sim.add_robot(name="newcomer", urdf_path=_arm_path(tmp_path))["status"] == "success"

        assert _latched_bodies(sim) == {}
