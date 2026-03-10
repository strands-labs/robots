#!/usr/bin/env python3
"""
Test suite for NewtonBackend — validates all Phase 1-4 features.

Run on Thor (GPU):
    python3 -m strands_robots.newton.test_newton_backend

Run on CPU (limited):
    NEWTON_DEVICE=cpu python3 -m strands_robots.newton.test_newton_backend
"""

import os
import sys
import time
import traceback

# Add parent to path for direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

DEVICE = os.getenv("NEWTON_DEVICE", "cuda:0")
PASS = 0
FAIL = 0
SKIP = 0


def run_test(name, fn):
    global PASS, FAIL, SKIP
    try:
        result = fn()
        if result == "SKIP":
            SKIP += 1
            print(f"  ⏭️  {name} — SKIPPED")
        else:
            PASS += 1
            print(f"  ✅ {name}")
    except Exception as e:
        FAIL += 1
        print(f"  ❌ {name} — {e}")
        traceback.print_exc()


def test_config_validation():
    from strands_robots.newton import NewtonConfig

    # Valid config
    c = NewtonConfig(solver="mujoco", device=DEVICE, num_envs=1)
    assert c.solver == "mujoco"

    # Invalid solver
    try:
        NewtonConfig(solver="invalid")
        assert False, "Should have raised"
    except ValueError:
        pass

    # Invalid broad_phase
    try:
        NewtonConfig(broad_phase="invalid")
        assert False, "Should have raised"
    except ValueError:
        pass


def test_create_world():
    from strands_robots.newton import NewtonBackend, NewtonConfig

    b = NewtonBackend(NewtonConfig(solver="mujoco", device=DEVICE))
    r = b.create_world()
    assert r["success"]
    assert r["world_info"]["solver"] == "mujoco"
    assert r["world_info"]["broad_phase"] == "explicit"
    b.destroy()


def test_add_robot_urdf():
    from strands_robots.newton import NewtonBackend, NewtonConfig

    try:
        import newton.examples as ne
    except ImportError:
        return "SKIP"

    b = NewtonBackend(NewtonConfig(solver="mujoco", device=DEVICE))
    b.create_world()
    r = b.add_robot("quad", urdf_path=ne.get_asset("quadruped.urdf"))
    assert r["success"], r.get("message")
    assert r["robot_info"]["num_joints"] > 0
    b.destroy()


def test_step():
    from strands_robots.newton import NewtonBackend, NewtonConfig

    try:
        import newton.examples as ne
    except ImportError:
        return "SKIP"

    b = NewtonBackend(NewtonConfig(solver="mujoco", device=DEVICE))
    b.create_world()
    b.add_robot("quad", urdf_path=ne.get_asset("quadruped.urdf"))
    r = b.step()
    assert r["success"], r.get("error")
    assert r["step_count"] == 1
    assert r["sim_time"] > 0
    b.destroy()


def test_observation():
    from strands_robots.newton import NewtonBackend, NewtonConfig

    try:
        import newton.examples as ne
    except ImportError:
        return "SKIP"

    b = NewtonBackend(NewtonConfig(solver="mujoco", device=DEVICE))
    b.create_world()
    b.add_robot("quad", urdf_path=ne.get_asset("quadruped.urdf"))
    for _ in range(10):
        b.step()
    obs = b.get_observation("quad")
    assert obs["success"]
    assert "quad" in obs["observations"]
    jp = obs["observations"]["quad"]["joint_positions"]
    assert jp is not None
    assert len(jp) > 0
    b.destroy()


def test_replicate():
    from strands_robots.newton import NewtonBackend, NewtonConfig

    try:
        import newton.examples as ne
    except ImportError:
        return "SKIP"

    b = NewtonBackend(NewtonConfig(solver="mujoco", device=DEVICE))
    b.create_world()
    b.add_robot("quad", urdf_path=ne.get_asset("quadruped.urdf"))
    r = b.replicate(num_envs=16)
    assert r["success"], r.get("message")
    assert r["env_info"]["num_envs"] == 16
    assert int(r["env_info"]["bodies_total"]) > 16  # 16 envs × bodies_per_world + ground

    for _ in range(10):
        b.step()
    assert b._step_count == 10
    b.destroy()


def test_replicate_4096():
    from strands_robots.newton import NewtonBackend, NewtonConfig

    try:
        import newton.examples as ne
    except ImportError:
        return "SKIP"

    if DEVICE == "cpu":
        return "SKIP"

    b = NewtonBackend(NewtonConfig(solver="mujoco", device=DEVICE))
    b.create_world()
    b.add_robot("quad", urdf_path=ne.get_asset("quadruped.urdf"))
    r = b.replicate(num_envs=4096)
    assert r["success"]

    t0 = time.time()
    for _ in range(100):
        b.step()
    elapsed = time.time() - t0
    throughput = int(100 * 4096 / elapsed)
    print(f"    → 4096 envs × 100 steps: {elapsed:.2f}s ({throughput} env-steps/s)")
    assert throughput > 1000, f"Throughput too low: {throughput}"
    b.destroy()


def test_reset_full():
    from strands_robots.newton import NewtonBackend, NewtonConfig

    try:
        import newton.examples as ne
    except ImportError:
        return "SKIP"

    b = NewtonBackend(NewtonConfig(solver="mujoco", device=DEVICE))
    b.create_world()
    b.add_robot("quad", urdf_path=ne.get_asset("quadruped.urdf"))
    for _ in range(50):
        b.step()
    assert b._sim_time > 0

    r = b.reset()
    assert r["success"]
    assert b._sim_time == 0.0
    assert b._step_count == 0
    b.destroy()


def test_reset_per_env():
    from strands_robots.newton import NewtonBackend, NewtonConfig

    try:
        import newton.examples as ne
    except ImportError:
        return "SKIP"

    b = NewtonBackend(NewtonConfig(solver="mujoco", device=DEVICE))
    b.create_world()
    b.add_robot("quad", urdf_path=ne.get_asset("quadruped.urdf"))
    b.replicate(num_envs=4)
    for _ in range(20):
        b.step()

    r = b.reset(env_ids=[0, 2])
    assert r["success"]
    # sim_time should NOT reset for per-env reset
    assert b._sim_time > 0
    b.destroy()


def test_get_state():
    from strands_robots.newton import NewtonBackend, NewtonConfig

    try:
        import newton.examples as ne
    except ImportError:
        return "SKIP"

    b = NewtonBackend(NewtonConfig(solver="mujoco", device=DEVICE))
    b.create_world()
    b.add_robot("quad", urdf_path=ne.get_asset("quadruped.urdf"))
    b.step()
    state = b.get_state()
    assert state["success"]
    assert state["config"]["solver"] == "mujoco"
    assert state["config"]["broad_phase"] == "explicit"
    assert "quad" in state["robots"]
    assert "joint_q" in state["state"]
    b.destroy()


def test_diffsim_particle():

    from strands_robots.newton import NewtonBackend, NewtonConfig

    b = NewtonBackend(NewtonConfig(enable_differentiable=True, solver="semi_implicit", device=DEVICE))
    b.create_world()
    r = b.add_particles("ball", positions=[(0, 0, 1)], velocities=[(0, 5, -5)], mass=1.0)
    assert r["success"]
    b._finalize_model()
    assert b._model.particle_count == 1

    for _ in range(50):
        b.step()
    s = b.get_state()
    pq = s["state"].get("particle_q")
    assert pq is not None
    # Particle should have moved from (0,0,1) due to velocity + gravity
    assert abs(pq[0][0]) < 10  # sanity check
    b.destroy()


def test_collision_pipeline():
    from strands_robots.newton import NewtonBackend, NewtonConfig

    try:
        import newton.examples as ne
    except ImportError:
        return "SKIP"

    b = NewtonBackend(NewtonConfig(solver="mujoco", device=DEVICE, broad_phase="explicit"))
    b.create_world()
    b.add_robot("quad", urdf_path=ne.get_asset("quadruped.urdf"))
    b._finalize_model()
    assert b._collision_pipeline is not None, "CollisionPipeline should be created"
    assert b._contacts is not None, "Contacts should be allocated"
    b.destroy()


def test_destroy():
    from strands_robots.newton import NewtonBackend, NewtonConfig

    b = NewtonBackend(NewtonConfig(solver="mujoco", device=DEVICE))
    b.create_world()
    r = b.destroy()
    assert r["success"]
    assert not b._world_created
    assert b._model is None


def test_multiple_robots():
    from strands_robots.newton import NewtonBackend, NewtonConfig

    try:
        import newton.examples as ne
    except ImportError:
        return "SKIP"

    b = NewtonBackend(NewtonConfig(solver="mujoco", device=DEVICE))
    b.create_world()
    r1 = b.add_robot("quad1", urdf_path=ne.get_asset("quadruped.urdf"), position=(0, 0, 0))
    r2 = b.add_robot("quad2", urdf_path=ne.get_asset("quadruped.urdf"), position=(2, 0, 0))
    assert r1["success"] and r2["success"]
    assert len(b._robots) == 2
    assert r2["robot_info"]["joint_offset"] > 0

    for _ in range(20):
        b.step()

    obs = b.get_observation()
    assert "quad1" in obs["observations"]
    assert "quad2" in obs["observations"]
    b.destroy()


def test_dual_solver():
    import warp as wp

    from strands_robots.newton import NewtonBackend, NewtonConfig

    b = NewtonBackend(NewtonConfig(solver="mujoco", device=DEVICE))
    b.create_world()

    # Add robot
    try:
        import newton.examples as ne

        b.add_robot("quad", urdf_path=ne.get_asset("quadruped.urdf"))
    except ImportError:
        return "SKIP"

    # Add cloth
    b._lazy_init()
    grid = 6
    verts, indices = [], []
    for y in range(grid):
        for x in range(grid):
            verts.append(wp.vec3(float(x) * 0.1, float(y) * 0.1, 1.0))
    for y in range(grid - 1):
        for x in range(grid - 1):
            i = y * grid + x
            indices.extend([i, i + 1, i + grid, i + 1, i + grid + 1, i + grid])

    b._builder.add_cloth_mesh(
        pos=wp.vec3(0, 0, 0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0, 0, 0),
        vertices=verts,
        indices=indices,
        density=0.1,
        tri_ke=1e3,
        tri_ka=1e3,
        tri_kd=1e-4,
    )
    b._cloths["test_cloth"] = {"name": "test_cloth"}
    b._builder.color()
    b._model = None

    b._finalize_model()

    # Enable dual solver
    r = b.enable_dual_solver(rigid_solver="mujoco", cloth_solver="vbd")
    assert r["success"], r.get("message")
    assert b._secondary_solver is not None

    # Step should use both solvers
    for _ in range(20):
        b.step()
    assert b._step_count == 20
    b.destroy()


if __name__ == "__main__":
    print(f"\n🧪 Newton Backend Test Suite (device={DEVICE})")
    print("=" * 55)

    tests = [
        ("Config validation", test_config_validation),
        ("Create world", test_create_world),
        ("Add robot (URDF)", test_add_robot_urdf),
        ("Step simulation", test_step),
        ("Observation", test_observation),
        ("Replicate (16 envs)", test_replicate),
        ("Replicate (4096 envs)", test_replicate_4096),
        ("Full reset", test_reset_full),
        ("Per-env reset", test_reset_per_env),
        ("Get state", test_get_state),
        ("DiffSim particle", test_diffsim_particle),
        ("CollisionPipeline", test_collision_pipeline),
        ("Destroy", test_destroy),
        ("Multiple robots", test_multiple_robots),
        ("Dual solver (MuJoCo+VBD)", test_dual_solver),
    ]

    for name, fn in tests:
        run_test(name, fn)

    print("=" * 55)
    print(f"Results: ✅ {PASS} passed, ❌ {FAIL} failed, ⏭️  {SKIP} skipped")

    if FAIL > 0:
        sys.exit(1)
    print("\n🎉 All tests passed!")
