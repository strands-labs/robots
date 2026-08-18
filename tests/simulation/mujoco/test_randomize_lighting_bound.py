"""``randomize_lighting`` must offset each light from a fixed reference.

``randomize()`` documents this axis as "light pos jittered +/-0.5m": a bound on
how far a light may sit from where the scene authored it. The sampler wrote
``model.light_pos[i] += rng.uniform(-0.5, 0.5, size=3)``, so every call started
from wherever the previous one left the light. That is a random walk, not a
bounded jitter -- the displacement grows without limit while each individual
call still draws inside the advertised half-width, so nothing reports the
breach. On a light authored 3.5 m above the scene the bound is already gone by
the third call, and 50 per-episode calls (the documented sim2real loop) reach
4.7 m, which is 9.4x the bound and far outside the scene the light is meant to
illuminate. ``data.light_xpos`` -- the array the renderer actually reads --
follows, so the frames a dataset records go dark or acquire a wildly wrong
key light.

The same function already litigated this for the position axis, whose
``position_noise`` documentation states that the offset is measured from each
object's commanded pose "so repeated calls draw independent offsets inside this
bound instead of compounding into a random walk". The Newton backend's lighting
axis composes ``base + jitter`` around a constant base direction and is bounded
by construction. Lighting on MuJoCo was the one axis with no fixed reference:
``light_diffuse`` is assigned (bounded), colour is assigned (bounded), and the
position axis measures from the registry, leaving ``light_pos`` alone in
accumulating.

The reference used here is the scene spec's authored light pose -- the same
source a recompile regenerates ``model.light_pos`` from, so it stays correct for
the life of the scene however many times the axis has run.

These tests pin the bound and the reproducibility it makes possible, plus the
boundaries the fix must not cross: the axis still moves the light, the diffuse
and colour halves are untouched, an axis-off call writes nothing, a recompile
still undoes the axis as it does the other three, and a world whose authored
light poses cannot be read is refused with nothing applied rather than silently
falling back to the accumulating behaviour.
"""

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

#: The half-width ``randomize``'s docstring advertises for this axis.
DOCUMENTED_BOUND_M = 0.5


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_randomize_lighting_bound", mesh=False)
    _ok(s.create_world(gravity=[0, 0, -9.81]), "create_world")
    yield s
    s.cleanup()


def _ok(result: dict, what: str) -> dict:
    """Return ``result`` after checking it succeeded.

    A plain ``assert`` would strip the scene setup under ``python -O``.
    """
    if result.get("status") != "success":
        raise AssertionError(f"{what} failed: {result}")
    return result


def _light_positions(sim: Simulation) -> np.ndarray:
    assert sim._world is not None
    return np.asarray(sim._world._model.light_pos, dtype=np.float64).copy()


def _light_diffuse(sim: Simulation) -> np.ndarray:
    assert sim._world is not None
    return np.asarray(sim._world._model.light_diffuse, dtype=np.float64).copy()


def _geom_rgba(sim: Simulation) -> np.ndarray:
    assert sim._world is not None
    return np.asarray(sim._world._model.geom_rgba, dtype=np.float64).copy()


def _add_cube(sim: Simulation, name: str = "cube") -> None:
    """A non-ground geom, so the colour axis has something to recolor."""
    _ok(
        sim.add_object(name=name, shape="box", position=[0.3, 0.0, 0.05], size=[0.03, 0.03, 0.03], mass=0.2),
        f"add_object({name})",
    )


def _jitter_lighting(sim: Simulation, seed: int) -> dict:
    return _ok(
        sim.randomize(randomize_colors=False, randomize_lighting=True, seed=seed),
        f"randomize(randomize_lighting=True, seed={seed})",
    )


def test_repeated_lighting_randomization_stays_inside_the_documented_bound(sim):
    """Every light stays within the advertised half-width of its authored pose.

    Pre-fix each call offset the *live* position, so the displacement was a
    random walk: 50 calls reached 4.7 m against a 0.5 m bound.
    """
    authored = _light_positions(sim)
    assert authored.size, "premise: the world must declare at least one light"

    worst = 0.0
    for episode in range(50):
        _jitter_lighting(sim, seed=1000 + episode)
        worst = max(worst, float(np.abs(_light_positions(sim) - authored).max()))

    assert worst <= DOCUMENTED_BOUND_M, (
        f"after 50 randomize(randomize_lighting=True) calls a light sits {worst:.4f} m from its "
        f"authored position, {worst / DOCUMENTED_BOUND_M:.1f}x the documented +/-{DOCUMENTED_BOUND_M} m bound"
    )


def test_a_seed_reproduces_the_same_light_pose_whatever_ran_before_it(sim):
    """A seed names an absolute pose, not an increment onto the current one.

    ``randomize`` documents ``seed`` as giving "a reproducible stream". Pre-fix
    the resulting pose depended on how many calls had already run, so replaying
    a seed could not reproduce a scene's lighting.
    """
    _jitter_lighting(sim, seed=42)
    first = _light_positions(sim)

    for _ in range(7):
        _jitter_lighting(sim, seed=99)
    _jitter_lighting(sim, seed=42)
    replayed = _light_positions(sim)

    assert np.allclose(first, replayed), (
        "the same seed produced a different light pose because seven unrelated calls ran in "
        f"between; max difference {np.abs(first - replayed).max():.4f} m"
    )


def test_the_axis_still_moves_every_light(sim):
    """The bound is not honoured by refusing to move the light at all."""
    authored = _light_positions(sim)
    _jitter_lighting(sim, seed=7)
    moved = _light_positions(sim)

    per_light = np.abs(moved - authored).max(axis=1)
    assert (per_light > 1e-9).all(), f"a light was left exactly where it was authored: offsets {per_light.tolist()}"


def test_a_single_call_offsets_each_light_inside_the_bound(sim):
    """The first call was always inside the bound; it must stay that way."""
    authored = _light_positions(sim)
    _jitter_lighting(sim, seed=3)
    assert np.abs(_light_positions(sim) - authored).max() <= DOCUMENTED_BOUND_M


def test_the_diffuse_half_stays_inside_its_resample_range(sim):
    """``light_diffuse`` is resampled, not accumulated, and stays in 0.3..1.0."""
    for episode in range(20):
        _jitter_lighting(sim, seed=500 + episode)
    diffuse = _light_diffuse(sim)
    assert diffuse.min() >= 0.3 - 1e-9, f"diffuse fell below the documented floor: {diffuse.min()}"
    assert diffuse.max() <= 1.0 + 1e-9, f"diffuse rose above the documented ceiling: {diffuse.max()}"


def test_the_colour_axis_keeps_sampling_inside_color_range(sim):
    """Recoloring is unaffected: it was already an assignment, so bounded."""
    _add_cube(sim)
    for episode in range(20):
        _ok(
            sim.randomize(randomize_colors=True, randomize_lighting=False, color_range=(0.1, 0.9), seed=600 + episode),
            "randomize(randomize_colors=True)",
        )
    rgb = _geom_rgba(sim)[:, :3]
    assert rgb.min() >= 0.1 - 1e-9 and rgb.max() <= 0.9 + 1e-9, f"geom RGB left color_range: {rgb.min()}..{rgb.max()}"


def test_lighting_off_leaves_every_light_where_it_was(sim):
    """An axis-off call writes neither light position nor diffuse."""
    _add_cube(sim)
    before_pos, before_diffuse = _light_positions(sim), _light_diffuse(sim)
    _ok(sim.randomize(randomize_colors=True, randomize_lighting=False, seed=11), "randomize(lighting off)")
    assert np.array_equal(_light_positions(sim), before_pos)
    assert np.array_equal(_light_diffuse(sim), before_diffuse)


def test_a_recompile_restores_the_authored_pose_and_the_axis_keeps_working(sim):
    """A recompile undoes this axis, as it undoes the other three.

    The docstring's "recompile the scene to undo" must keep holding, and the
    axis must still be bounded on the model the recompile produced.
    """
    authored = _light_positions(sim)
    _jitter_lighting(sim, seed=21)
    assert not np.allclose(_light_positions(sim), authored), "premise: the jitter must have moved the light"

    _add_cube(sim)  # any scene mutation recompiles from the spec
    assert np.allclose(_light_positions(sim), authored), "a recompile no longer restores the authored light pose"

    _jitter_lighting(sim, seed=22)
    assert np.abs(_light_positions(sim) - authored).max() <= DOCUMENTED_BOUND_M


def test_a_world_without_a_readable_spec_refuses_the_axis_with_nothing_applied(sim):
    """No authored reference means the bound cannot be honoured: refuse.

    Falling back to the live position would silently reinstate the accumulating
    behaviour. The refusal also has to land before the colour axis runs, or a
    rejected call would still have recolored the scene.
    """
    _add_cube(sim)
    assert sim._world is not None
    before_rgba, before_pos = _geom_rgba(sim), _light_positions(sim)
    sim._world._backend_state.pop("spec", None)

    result = sim.randomize(randomize_colors=True, randomize_lighting=True, seed=1)

    assert result["status"] == "error", f"expected a refusal, got {result}"
    text = result["content"][0]["text"]
    assert "randomize_lighting" in text and "authored" in text, text
    assert "randomize_lighting=False" in text, f"the refusal must name a way forward: {text}"
    assert np.array_equal(_geom_rgba(sim), before_rgba), "the refused call still recolored the scene"
    assert np.array_equal(_light_positions(sim), before_pos), "the refused call still moved a light"


def test_the_success_text_still_reports_the_lights_it_randomized(sim):
    """The reported axis summary is unchanged."""
    assert sim._world is not None
    nlight = int(sim._world._model.nlight)
    result = _jitter_lighting(sim, seed=31)
    assert f"Lighting: {nlight} lights randomized" in result["content"][0]["text"]
