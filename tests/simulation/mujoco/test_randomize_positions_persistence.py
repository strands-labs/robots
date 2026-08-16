"""``randomize_positions`` must perturb the pose a rollout actually starts from.

``randomize()`` has four axes. Colour, friction and mass write ``model`` arrays,
so they persist for the rest of the scene's life and are undone only by a
recompile. The position axis wrote only ``data.qpos``, which is the one place a
:meth:`reset` overwrites -- ``mj_resetData`` restores ``data.qpos`` from
``model.qpos0``. Every rollout entry point resets before an episode's first step
(``PolicyRunner.evaluate`` calls ``sim.reset()`` at the top of each episode), so
the axis was reverted before the policy ever observed it: a
``randomize(randomize_positions=True)`` call reported success and every episode
still began at the canonical authored pose.

Making the offset persist also requires a fixed reference to measure it from.
The registry pose (what ``add_object`` / ``move_object`` commanded) is fixed;
the live pose is not, so measuring from the live pose turns a per-episode loop
into a random walk that leaves the requested bound behind -- 50 episodes at a
0.03 m half-width reach 0.12 m and put a table-top object under the floor.

These tests pin the fix -- the perturbed pose is the pose the next episode
starts from, and every offset stays inside ``position_noise`` of the commanded
pose no matter how many episodes run -- plus the boundaries it must not cross:
a static object is never moved, a recompile still undoes the axis (as it does
the other three), and a call with the axis off leaves both pose stores
untouched.
"""

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_randomize_positions_persistence", mesh=False)
    assert s.create_world(gravity=[0, 0, -9.81])["status"] == "success"
    yield s
    s.cleanup()


def _freejoint_qpos_address(sim: Simulation, name: str) -> int:
    """qpos offset of ``name``'s freejoint, asserting the joint resolves."""
    assert sim._world is not None
    model = sim._world._model
    jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f"{name}_joint")
    assert jnt_id >= 0, f"no freejoint for dynamic object {name!r}"
    return int(model.jnt_qposadr[jnt_id])


def _live_position(sim: Simulation, name: str) -> np.ndarray:
    adr = _freejoint_qpos_address(sim, name)
    assert sim._world is not None
    return np.asarray(sim._world._data.qpos[adr : adr + 3], dtype=np.float64).copy()


def _reset_reference_position(sim: Simulation, name: str) -> np.ndarray:
    """The pose ``mj_resetData`` restores for ``name`` -- ``model.qpos0``."""
    adr = _freejoint_qpos_address(sim, name)
    assert sim._world is not None
    return np.asarray(sim._world._model.qpos0[adr : adr + 3], dtype=np.float64).copy()


def _add_cube(sim: Simulation, name: str = "cube", position=(0.30, 0.00, 0.05)) -> None:
    assert (
        sim.add_object(name=name, shape="box", position=list(position), size=[0.03, 0.03, 0.03], mass=0.2)["status"]
        == "success"
    )


def test_randomized_position_survives_the_reset_that_begins_a_rollout(sim):
    """The perturbed pose is the pose the next episode starts from.

    Pre-fix the noise reached ``data.qpos`` only, so ``reset()`` -- which every
    rollout entry point calls before its first step -- restored the canonical
    authored pose and the randomization never reached the policy.
    """
    _add_cube(sim)
    authored = _live_position(sim, "cube")

    assert (
        sim.randomize(
            randomize_colors=False,
            randomize_lighting=False,
            randomize_positions=True,
            position_noise=0.05,
            seed=7,
        )["status"]
        == "success"
    )
    perturbed = _live_position(sim, "cube")
    assert not np.allclose(perturbed, authored), "randomize did not move the object at all"

    assert sim.reset()["status"] == "success"
    after_reset = _live_position(sim, "cube")
    assert np.allclose(after_reset, perturbed), (
        f"reset() reverted the randomized pose: perturbed {perturbed} -> {after_reset} "
        f"(authored {authored}); every episode would start at the canonical pose"
    )


def test_three_randomize_then_reset_cycles_yield_three_distinct_start_poses(sim):
    """The documented per-episode randomization loop actually varies the scene.

    ``randomize(seed=episode)`` followed by the reset each rollout performs is
    the loop the docs prescribe; pre-fix all N episodes began at the identical
    authored pose.
    """
    _add_cube(sim)
    starts = []
    for episode in range(3):
        assert (
            sim.randomize(
                randomize_colors=False,
                randomize_lighting=False,
                randomize_positions=True,
                position_noise=0.04,
                seed=episode,
            )["status"]
            == "success"
        )
        assert sim.reset()["status"] == "success"
        starts.append(tuple(np.round(_live_position(sim, "cube"), 6)))

    assert len(set(starts)) == 3, f"expected a distinct start pose per episode, got {starts}"


def test_offsets_stay_inside_the_bound_over_many_episodes(sim):
    """Every episode's start pose is within ``position_noise`` of the commanded pose.

    The offset is measured from the registry pose, which does not move, so the
    bound holds for episode 50 as it does for episode 0. Measuring from the live
    pose compounds instead: the same loop reaches 0.12 m and drops the object
    through the floor.
    """
    commanded = [0.30, 0.00, 0.05]
    _add_cube(sim, position=tuple(commanded))
    noise = 0.03
    worst = 0.0
    for episode in range(50):
        assert (
            sim.randomize(
                randomize_colors=False,
                randomize_lighting=False,
                randomize_positions=True,
                position_noise=noise,
                seed=episode,
            )["status"]
            == "success"
        )
        assert sim.reset()["status"] == "success"
        offset = _live_position(sim, "cube") - np.asarray(commanded, dtype=np.float64)
        worst = max(worst, float(np.max(np.abs(offset))))

    assert worst <= noise + 1e-12, (
        f"start pose drifted {worst:.4f} m from the commanded pose over 50 episodes, "
        f"outside the requested {noise} m half-width"
    )


def test_positions_result_reports_how_many_objects_were_perturbed(sim):
    """A count, as the colour axis reports -- a static-only scene perturbs nothing."""
    assert (
        sim.add_object(name="table", shape="box", position=[0.0, 0.4, 0.02], size=[0.2, 0.2, 0.02], is_static=True)[
            "status"
        ]
        == "success"
    )
    result = sim.randomize(
        randomize_colors=False,
        randomize_lighting=False,
        randomize_positions=True,
        position_noise=0.05,
        seed=3,
    )
    assert result["status"] == "success"
    assert "Positions: 0 dynamic objects perturbed" in result["content"][0]["text"], result["content"][0]["text"]

    _add_cube(sim)
    result = sim.randomize(
        randomize_colors=False,
        randomize_lighting=False,
        randomize_positions=True,
        position_noise=0.05,
        seed=3,
    )
    assert result["status"] == "success"
    assert "Positions: 1 dynamic objects perturbed" in result["content"][0]["text"], result["content"][0]["text"]


def test_static_object_is_never_moved(sim):
    """A welded body has no pose DOF; its registry pose must stay authored."""
    authored = [0.0, 0.4, 0.02]
    assert (
        sim.add_object(name="table", shape="box", position=list(authored), size=[0.2, 0.2, 0.02], is_static=True)[
            "status"
        ]
        == "success"
    )
    _add_cube(sim)
    assert sim._world is not None
    bid = mj.mj_name2id(sim._world._model, mj.mjtObj.mjOBJ_BODY, "table")
    assert bid >= 0
    xpos_before = np.asarray(sim._world._data.xpos[bid], dtype=np.float64).copy()

    assert (
        sim.randomize(
            randomize_colors=False,
            randomize_lighting=False,
            randomize_positions=True,
            position_noise=0.05,
            seed=5,
        )["status"]
        == "success"
    )

    assert np.allclose(sim._world.objects["table"].position, authored)
    assert np.allclose(sim._world._data.xpos[bid], xpos_before)


def test_axis_off_leaves_live_pose_reset_reference_and_registry_untouched(sim):
    """``randomize_positions=False`` is an exact no-op on both pose stores."""
    _add_cube(sim)
    assert sim._world is not None
    live_before = _live_position(sim, "cube")
    reference_before = _reset_reference_position(sim, "cube")
    registered_before = list(sim._world.objects["cube"].position)

    assert (
        sim.randomize(
            randomize_colors=True,
            randomize_lighting=True,
            randomize_physics=True,
            randomize_positions=False,
            seed=2,
        )["status"]
        == "success"
    )

    assert np.array_equal(_live_position(sim, "cube"), live_before)
    assert np.array_equal(_reset_reference_position(sim, "cube"), reference_before)
    assert sim._world.objects["cube"].position == registered_before


def test_perturbation_stays_within_the_requested_half_width(sim):
    """Each axis moves by at most ``position_noise`` -- the documented bound."""
    commanded = [0.30, 0.00, 0.05]
    _add_cube(sim, position=tuple(commanded))
    authored = np.asarray(commanded, dtype=np.float64)
    noise = 0.02
    assert (
        sim.randomize(
            randomize_colors=False,
            randomize_lighting=False,
            randomize_positions=True,
            position_noise=noise,
            seed=13,
        )["status"]
        == "success"
    )
    delta = _live_position(sim, "cube") - authored
    assert np.all(np.abs(delta) <= noise + 1e-12), delta
    assert np.allclose(_reset_reference_position(sim, "cube"), _live_position(sim, "cube"))


def test_a_recompile_still_undoes_the_position_axis(sim):
    """Recompiling is the documented undo, and it must undo this axis too.

    A recompile rebuilds ``model.qpos0`` from the spec, exactly as it rebuilds
    the colour / friction / mass arrays, so the axis is no more sticky than its
    three siblings.
    """
    _add_cube(sim)
    authored = _reset_reference_position(sim, "cube")
    assert (
        sim.randomize(
            randomize_colors=False,
            randomize_lighting=False,
            randomize_positions=True,
            position_noise=0.05,
            seed=17,
        )["status"]
        == "success"
    )
    assert not np.allclose(_reset_reference_position(sim, "cube"), authored)

    # Any scene mutation recompiles the model.
    assert (
        sim.add_object(name="marker", shape="sphere", position=[0.8, 0.8, 0.05], size=[0.02], mass=0.05)["status"]
        == "success"
    )
    assert np.allclose(_reset_reference_position(sim, "cube"), authored), (
        "a recompile must rebuild the reset reference from the spec, as it does for colour/friction/mass"
    )
