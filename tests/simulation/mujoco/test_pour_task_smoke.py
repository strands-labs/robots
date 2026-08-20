"""The pouring task works end to end on the real MuJoCo backend.

Pins the articulated-container acceptance criteria from the particle-proxy
pouring feature: the bundled task objects attach into a live scene through the
existing ``add_robot(urdf_path=...)`` route, the carton caps actuate and are
scored by the joint predicates, a scripted pour flips ``particles_inside``
from False to True with ``particles_spilled`` staying quiet, and the whole
benchmark scores bit-identically when the same seeded evaluation is run twice
(the "deterministic under seed" pin).

The pour mount is the SLIDING carton: lid-down, the load of resting beads is
orthogonal to the slide axis, so the closed cap is a stable state. A lid-down
hinge creeps open under any resting load (MuJoCo dof friction is a soft
constraint that saturates under persistent torque), which is why the hinged
carton is tested upright here - and why its asset comment says to mount it
that way.
"""

from __future__ import annotations

from typing import Any

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation import create_simulation  # noqa: E402
from strands_robots.simulation.benchmark import register_benchmark, unregister_benchmark  # noqa: E402
from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark  # noqa: E402
from strands_robots.simulation.predicates import make_predicate  # noqa: E402
from strands_robots.simulation.task_objects import list_task_objects, task_object_path  # noqa: E402

BEADS = ["bead_0", "bead_1", "bead_2", "bead_3"]

# A minimal actuated arm so the seeded evaluation has a robot to drive - the
# same inline-MJCF pattern as test_scene_joint_predicates.py, plus a position
# actuator so the policy's actions resolve.
ARM_MJCF = """<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link" pos="0 0 0.05">
      <joint name="shoulder" type="hinge" axis="0 0 1" range="-1 1"/>
      <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.01" mass="0.2"/>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_act" joint="shoulder" kp="5" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

POUR_SPEC: dict[str, Any] = {
    "name": "pour_smoke",
    "default_robot": "arm",
    "max_steps": 40,
    "success": {
        "all": [
            {"predicate": "joint_above", "joint": "cap_slide", "value": 0.06},
            {
                "predicate": "particles_inside",
                "particles": BEADS,
                "container": "tray",
                "min_fraction": 0.75,
                "xy_tol": 0.12,
                "z_tol": 0.08,
            },
        ]
    },
    "failure": {
        "any": [
            {
                "predicate": "particles_spilled",
                "particles": BEADS,
                "containers": ["tray", "carton"],
                "max_spilled": 1,
                "xy_tol": 0.12,
                "z_tol": 0.30,
            }
        ]
    },
    "dense_reward": [
        {"predicate": "particles_inside_fraction", "particles": BEADS, "container": "tray"},
    ],
}


def _build_pour_scene(sim) -> None:
    """Tray on the ground, sliding carton lid-down above it, beads inside."""
    assert (
        sim.add_robot(name="tray", urdf_path=task_object_path("open_tray"), position=[0.55, 0.0, 0.0])["status"]
        == "success"
    )
    assert (
        sim.add_robot(
            name="carton",
            urdf_path=task_object_path("sliding_carton"),
            position=[0.55, 0.0, 0.30],
            orientation=[0.0, 1.0, 0.0, 0.0],
        )["status"]
        == "success"
    )
    positions = [[0.53, 0.0, 0.19], [0.57, 0.0, 0.19], [0.53, 0.0, 0.22], [0.57, 0.0, 0.22]]
    for name, pos in zip(BEADS, positions, strict=True):
        assert (
            sim.add_object(name=name, shape="sphere", size=[0.024] * 3, position=pos, mass=0.05)["status"] == "success"
        )
    assert sim.step(200)["status"] == "success"


@pytest.fixture
def sim():
    engine = create_simulation(backend="mujoco", tool_name="pour_smoke", mesh=False)
    assert engine.create_world(ground_plane=True)["status"] == "success"
    try:
        yield engine
    finally:
        engine.destroy()


def _bead_positions(sim) -> list[list[float]]:
    out = []
    for name in BEADS:
        result = sim.get_body_state(body_name=name)
        assert result["status"] == "success"
        payload = next(c["json"] for c in result["content"] if "json" in c)
        out.append([float(c) for c in payload["position"]])
    return out


def test_every_bundled_task_object_attaches(sim):
    for i, name in enumerate(list_task_objects()):
        result = sim.add_robot(name=f"obj_{i}", urdf_path=task_object_path(name), position=[0.5 * i, 2.0, 0.0])
        assert result["status"] == "success", f"{name} failed to attach: {result}"


def test_hinged_cap_actuates(sim):
    """Acceptance pin: the hinge cap actuates and the joint predicate scores it."""
    assert sim.add_robot(name="carton", urdf_path=task_object_path("hinged_carton"))["status"] == "success"
    opened = make_predicate("joint_above", joint="cap_hinge", value=1.0)
    assert opened(sim) is False
    assert sim.set_joint_positions({"cap_hinge": 2.0}, robot_name="carton")["status"] == "success"
    assert opened(sim) is True


def test_scripted_pour_flips_the_predicates(sim):
    _build_pour_scene(sim)
    opened = make_predicate("joint_above", joint="cap_slide", value=0.06)
    poured = make_predicate(
        "particles_inside", particles=BEADS, container="tray", min_fraction=0.75, xy_tol=0.12, z_tol=0.08
    )
    spilled = make_predicate(
        "particles_spilled", particles=BEADS, containers=["tray", "carton"], max_spilled=1, xy_tol=0.12, z_tol=0.30
    )
    # Closed cap: beads held in the carton, nothing poured, nothing spilled.
    assert opened(sim) is False
    assert poured(sim) is False
    assert spilled(sim) is False
    # Slide the cap open; gravity pours the beads into the tray below.
    assert sim.set_joint_positions({"cap_slide": 0.09}, robot_name="carton")["status"] == "success"
    assert sim.step(500)["status"] == "success"
    assert opened(sim) is True
    assert poured(sim) is True
    assert spilled(sim) is False


@pytest.mark.parametrize("shape", [list, tuple], ids=["list", "tuple"])
def test_a_typod_bead_is_refused_before_the_rollout_whatever_the_sequence_shape(tmp_path, shape):
    """A pour clause naming a bead that is not in the scene is refused at step 0.

    ``run_policy`` probes a ``stop_when`` clause's entity names against the live
    scene before arming it, because a typo'd name compiles clean and degrades to
    a constant False: the rollout would run its whole step budget and report
    ``stopped_reason="budget"``, which is what an honest miss reports too. The
    bead names reach that probe through the sequence-valued ``particles`` kwarg,
    so every shape the predicate factory accepts for it has to arrive there -
    otherwise the spelling of the name list decides whether the typo is caught.
    """
    arm = tmp_path / "arm.xml"
    arm.write_text(ARM_MJCF)
    engine = create_simulation(backend="mujoco", tool_name="pour_probe", mesh=False)
    try:
        assert engine.create_world(ground_plane=True)["status"] == "success"
        assert engine.add_robot(name="arm", urdf_path=str(arm))["status"] == "success"
        _build_pour_scene(engine)
        result = engine.run_policy(
            robot_name="arm",
            policy_provider="mock",
            n_steps=40,
            control_frequency=50.0,
            stop_when={
                "predicate": "particles_inside",
                "particles": shape([*BEADS, "bead_typo"]),
                "container": "tray",
            },
        )
        payload = next(c["json"] for c in result["content"] if "json" in c)
        text = result["content"][0]["text"]
        assert result["status"] == "error", (
            f"the clause was armed with an unresolvable bead and the rollout ran anyway: "
            f"stopped_reason={payload.get('stopped_reason')!r} steps_used={payload.get('steps_used')!r}"
        )
        assert "bead_typo" in text, f"the refusal does not name the unresolvable bead: {text}"
        assert payload["steps_used"] == 0
    finally:
        engine.destroy()


def test_the_closed_cap_is_a_stable_state(sim):
    """The failure mode this asset choice exists for: the cap must not creep open."""
    _build_pour_scene(sim)
    opened = make_predicate("joint_above", joint="cap_slide", value=0.01)
    assert sim.step(800)["status"] == "success"
    assert opened(sim) is False


def test_scripted_pour_is_bit_deterministic():
    """Two fresh builds of the same pour produce bit-identical bead positions."""
    runs = []
    for _ in range(2):
        engine = create_simulation(backend="mujoco", tool_name="pour_det", mesh=False)
        try:
            assert engine.create_world(ground_plane=True)["status"] == "success"
            _build_pour_scene(engine)
            assert engine.set_joint_positions({"cap_slide": 0.09}, robot_name="carton")["status"] == "success"
            assert engine.step(500)["status"] == "success"
            runs.append(_bead_positions(engine))
        finally:
            engine.destroy()
    assert runs[0] == runs[1]


def test_pour_benchmark_scores_deterministically_under_seed(tmp_path):
    """Acceptance pin: the same seeded evaluation reports the same score twice."""
    arm = tmp_path / "arm.xml"
    arm.write_text(ARM_MJCF)
    benchmark = DeclarativeBenchmark.from_dict(POUR_SPEC)
    register_benchmark(benchmark.name, benchmark)
    try:
        metrics = []
        for _ in range(2):
            engine = create_simulation(backend="mujoco", tool_name="pour_seed", mesh=False)
            try:
                assert engine.create_world(ground_plane=True)["status"] == "success"
                assert engine.add_robot(name="arm", urdf_path=str(arm))["status"] == "success"
                _build_pour_scene(engine)
                result = engine.evaluate_benchmark(
                    benchmark.name, robot_name="arm", policy_provider="mock", n_episodes=1, seed=7
                )
                assert result["status"] == "success", result
                metrics.append(next(c["json"] for c in result["content"] if "json" in c))
            finally:
                engine.destroy()
        assert metrics[0]["success_rate"] == metrics[1]["success_rate"]
        assert metrics[0]["avg_reward"] == metrics[1]["avg_reward"]
        assert metrics[0]["avg_steps"] == metrics[1]["avg_steps"]
    finally:
        unregister_benchmark(benchmark.name)
