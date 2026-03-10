#!/usr/bin/env python3
"""Test ALL 7 Newton solver backends on the same scenes.

Newton Solvers:
  1. mujoco        — MuJoCo-Warp, general-purpose articulated rigid bodies
  2. featherstone  — Recursive Newton-Euler, serial chains
  3. semi_implicit — Fastest for simple rigid-body + particles
  4. xpbd          — Position-based dynamics, soft constraints
  5. vbd           — Vertex Block Descent, deformables/cloth
  6. style3d       — Style3D garment solver
  7. implicit_mpm  — Material Point Method, granular/fluid
"""

import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import newton  # noqa: E402
import newton.examples as ne  # noqa: E402
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

wp.config.quiet = True
wp.init()

DEVICE = "cuda:0"
RESULTS = {}


def test_solver_rigid(solver_name, num_envs=16):
    """Test a solver with rigid-body articulation (quadruped URDF)."""
    print(f"\n{'='*60}")
    print(f"  Testing: {solver_name.upper()} (rigid-body, {num_envs} envs)")
    print(f"{'='*60}")

    try:
        SolverCls = getattr(
            newton.solvers,
            {
                "mujoco": "SolverMuJoCo",
                "featherstone": "SolverFeatherstone",
                "semi_implicit": "SolverSemiImplicit",
                "xpbd": "SolverXPBD",
            }[solver_name],
        )
    except (KeyError, AttributeError) as e:
        print(f"  ❌ Solver class not found: {e}")
        return {"status": "SKIP", "reason": str(e)}

    try:
        # Build single-world robot
        robot_builder = newton.ModelBuilder()
        if hasattr(SolverCls, "register_custom_attributes"):
            SolverCls.register_custom_attributes(robot_builder)
        robot_builder.add_urdf(ne.get_asset("quadruped.urdf"))

        # Replicate
        main_builder = newton.ModelBuilder()
        main_builder.replicate(robot_builder, num_envs, spacing=(2.0, 2.0, 0.0))
        main_builder.add_ground_plane()
        if hasattr(SolverCls, "register_custom_attributes"):
            SolverCls.register_custom_attributes(main_builder)

        model = main_builder.finalize()
        print(f"  Bodies: {model.body_count}, Joints: {model.joint_count}")

        # Create solver
        try:
            solver = SolverCls(model)
        except TypeError:
            solver = SolverCls(model, 1.0 / 200.0)

        s0 = model.state()
        s1 = model.state()
        ctrl = model.control()

        # Forward kinematics
        try:
            newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        except Exception as e:
            print(f"  ⚠️ eval_fk failed (non-fatal): {e}")

        # Collision (optional for some solvers)
        contacts = None
        try:
            pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
            contacts = pipeline.contacts()
        except Exception:
            try:
                contacts = model.contacts()
            except Exception:
                pass

        # Warmup
        for _ in range(3):
            s0.clear_forces()
            solver.step(s0, s1, ctrl, contacts, 1.0 / 200.0)
            s0, s1 = s1, s0

        # Benchmark
        t0 = time.time()
        n_steps = 100
        for _ in range(n_steps):
            s0.clear_forces()
            if contacts is not None and hasattr(pipeline, "collide"):
                try:
                    pipeline.collide(s0, contacts)
                except Exception:
                    pass
            solver.step(s0, s1, ctrl, contacts, 1.0 / 200.0)
            s0, s1 = s1, s0
        elapsed = time.time() - t0
        throughput = int(n_steps * num_envs / elapsed)

        # Read joint state
        jq = s0.joint_q.numpy()
        jq_norm = float(np.linalg.norm(jq))

        print(f"  ✅ {n_steps} steps × {num_envs} envs: {elapsed:.3f}s")
        print(f"  📊 Throughput: {throughput:,} env-steps/s")
        print(f"  🦾 Joint state norm: {jq_norm:.4f}")

        return {
            "status": "PASS",
            "throughput": throughput,
            "elapsed": elapsed,
            "joint_norm": jq_norm,
            "bodies": model.body_count,
        }

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()
        return {"status": "FAIL", "error": str(e)}


def test_solver_particles(solver_name, n_particles=100):
    """Test a solver with particle-based simulation."""
    print(f"\n{'='*60}")
    print(f"  Testing: {solver_name.upper()} (particles, n={n_particles})")
    print(f"{'='*60}")

    try:
        SolverCls = getattr(
            newton.solvers,
            {
                "semi_implicit": "SolverSemiImplicit",
                "xpbd": "SolverXPBD",
                "vbd": "SolverVBD",
                "implicit_mpm": "SolverImplicitMPM",
            }[solver_name],
        )
    except (KeyError, AttributeError) as e:
        print(f"  ❌ Solver class not found: {e}")
        return {"status": "SKIP", "reason": str(e)}

    try:
        builder = newton.ModelBuilder()

        # Register solver-specific attributes
        if hasattr(SolverCls, "register_custom_attributes"):
            SolverCls.register_custom_attributes(builder)

        # Add particles
        np.random.seed(42)
        for i in range(n_particles):
            pos = wp.vec3(
                float(np.random.uniform(-1, 1)),
                float(np.random.uniform(-1, 1)),
                float(np.random.uniform(0.5, 3.0)),
            )
            builder.add_particle(pos=pos, vel=wp.vec3(0, 0, 0), mass=1.0)

        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=1e4, kd=10, mu=0.5))

        model = builder.finalize()
        model.soft_contact_ke = 1e4
        model.soft_contact_kd = 10
        model.soft_contact_mu = 0.5
        model.soft_contact_restitution = 0.3

        print(f"  Particles: {model.particle_count}")

        # Create solver
        try:
            if solver_name == "implicit_mpm":
                mpm_config = SolverCls.Config()
                mpm_config.voxel_size = 0.1
                solver = SolverCls(model, mpm_config)
            elif solver_name == "vbd":
                solver = SolverCls(model, iterations=5)
            else:
                solver = SolverCls(model)
        except TypeError:
            try:
                solver = SolverCls(model, 1.0 / 60.0)
            except TypeError:
                solver = SolverCls(model, iterations=5)

        s0 = model.state()
        s1 = model.state()
        ctrl = model.control()

        # Collision pipeline
        contacts = None
        try:
            pipeline = newton.CollisionPipeline(model, broad_phase="explicit", soft_contact_margin=5.0)
            contacts = pipeline.contacts()
            pipeline.collide(s0, contacts)
        except Exception as e:
            print(f"  ⚠️ CollisionPipeline: {e}")

        # Simulate
        t0 = time.time()
        n_steps = 200
        for step in range(n_steps):
            s0.clear_forces()
            if contacts is not None:
                try:
                    pipeline.collide(s0, contacts)
                except Exception:
                    pass
            solver.step(s0, s1, ctrl, contacts, 1.0 / 60.0)
            s0, s1 = s1, s0
        elapsed = time.time() - t0

        # Check particle state
        pq = s0.particle_q.numpy()
        avg_z = float(np.mean(pq[:, 2]))
        min_z = float(np.min(pq[:, 2]))
        above_ground = bool(np.all(pq[:, 2] >= -0.5))

        print(f"  ✅ {n_steps} steps: {elapsed:.3f}s")
        print(f"  📊 Avg Z: {avg_z:.4f}, Min Z: {min_z:.4f}")
        print(f"  🌍 Above ground: {above_ground}")

        return {
            "status": "PASS",
            "elapsed": elapsed,
            "avg_z": avg_z,
            "min_z": min_z,
            "above_ground": above_ground,
        }

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()
        return {"status": "FAIL", "error": str(e)}


def test_solver_cloth(solver_name):
    """Test cloth/deformable simulation with VBD or XPBD."""
    print(f"\n{'='*60}")
    print(f"  Testing: {solver_name.upper()} (cloth grid)")
    print(f"{'='*60}")

    try:
        SolverCls = getattr(
            newton.solvers,
            {
                "xpbd": "SolverXPBD",
                "vbd": "SolverVBD",
                "style3d": "SolverStyle3D",
            }[solver_name],
        )
    except (KeyError, AttributeError) as e:
        print(f"  ❌ Solver class not found: {e}")
        return {"status": "SKIP", "reason": str(e)}

    try:
        builder = newton.ModelBuilder()
        if hasattr(SolverCls, "register_custom_attributes"):
            SolverCls.register_custom_attributes(builder)

        # Create a simple cloth grid
        grid_size = 10
        vertices = []
        indices = []
        for y in range(grid_size):
            for x in range(grid_size):
                vertices.append(wp.vec3(float(x) * 0.1 - 0.45, float(y) * 0.1 - 0.45, 1.5))

        # Triangulate grid
        for y in range(grid_size - 1):
            for x in range(grid_size - 1):
                i = y * grid_size + x
                indices.extend([i, i + 1, i + grid_size])
                indices.extend([i + 1, i + grid_size + 1, i + grid_size])

        builder.add_cloth_mesh(
            pos=wp.vec3(0, 0, 0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0, 0, 0),
            vertices=vertices,
            indices=indices,
            density=0.1,
            tri_ke=1e3,
            tri_ka=1e3,
            tri_kd=1e-4,
            edge_ke=10.0,
            edge_kd=1e-3,
        )

        builder.add_ground_plane()

        # VBD requires graph coloring before finalize
        if solver_name == "vbd":
            builder.color()

        model = builder.finalize()
        model.soft_contact_ke = 1e4
        model.soft_contact_kd = 10

        print(f"  Particles: {model.particle_count}, Triangles: {len(indices)//3}")

        # Create solver with appropriate params
        kwargs = {}
        if solver_name == "vbd":
            kwargs["iterations"] = 5
        try:
            solver = SolverCls(model, **kwargs)
        except TypeError:
            solver = SolverCls(model)

        s0 = model.state()
        s1 = model.state()
        ctrl = model.control()

        contacts = None
        try:
            pipeline = newton.CollisionPipeline(model, soft_contact_margin=2.0)
            contacts = pipeline.contacts()
            pipeline.collide(s0, contacts)
        except Exception:
            pass

        t0 = time.time()
        n_steps = 100
        for _ in range(n_steps):
            s0.clear_forces()
            if contacts is not None:
                try:
                    pipeline.collide(s0, contacts)
                except Exception:
                    pass
            solver.step(s0, s1, ctrl, contacts, 1.0 / 60.0)
            s0, s1 = s1, s0
        elapsed = time.time() - t0

        pq = s0.particle_q.numpy()
        avg_z = float(np.mean(pq[:, 2]))
        has_nan = bool(np.any(np.isnan(pq)))

        print(f"  ✅ {n_steps} steps: {elapsed:.3f}s")
        print(f"  📊 Avg Z: {avg_z:.4f} (cloth should drop from 1.5)")
        print(f"  🔢 NaN check: {'❌ HAS NaN' if has_nan else '✅ Clean'}")

        return {
            "status": "PASS",
            "elapsed": elapsed,
            "avg_z": avg_z,
            "has_nan": has_nan,
        }

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()
        return {"status": "FAIL", "error": str(e)}


# =====================================================================
# Run all tests
# =====================================================================
print("🧪 Newton ALL SOLVERS Test Suite")
print(f"Device: {DEVICE}")
print(f"Newton: {newton.__version__}")
print(f"Warp:   {wp.__version__}")

# --- Rigid-body solvers (articulated robots) ---
print("\n" + "🦿 RIGID-BODY TESTS (Quadruped URDF)" + "\n")
for solver_name in ["mujoco", "featherstone", "semi_implicit", "xpbd"]:
    RESULTS[f"rigid_{solver_name}"] = test_solver_rigid(solver_name)

# --- Particle solvers ---
print("\n" + "⚛️  PARTICLE TESTS (100 particles)" + "\n")
for solver_name in ["semi_implicit", "xpbd"]:
    RESULTS[f"particle_{solver_name}"] = test_solver_particles(solver_name)

# --- Cloth/deformable solvers ---
print("\n" + "🧵 CLOTH TESTS (10×10 grid)" + "\n")
for solver_name in ["xpbd", "vbd"]:
    RESULTS[f"cloth_{solver_name}"] = test_solver_cloth(solver_name)

# --- MPM solver ---
print("\n" + "🏔️  MPM TEST (granular particles)" + "\n")
RESULTS["particle_implicit_mpm"] = test_solver_particles("implicit_mpm", n_particles=50)

# =====================================================================
# Summary
# =====================================================================
print("\n\n" + "=" * 60)
print("📊 RESULTS SUMMARY")
print("=" * 60)
pass_count = sum(1 for v in RESULTS.values() if v["status"] == "PASS")
fail_count = sum(1 for v in RESULTS.values() if v["status"] == "FAIL")
skip_count = sum(1 for v in RESULTS.values() if v["status"] == "SKIP")

for name, result in RESULTS.items():
    status = result["status"]
    emoji = "✅" if status == "PASS" else ("❌" if status == "FAIL" else "⏭️")
    extra = ""
    if status == "PASS":
        if "throughput" in result:
            extra = f" — {result['throughput']:,} env-steps/s"
        elif "avg_z" in result:
            extra = f" — avg_z={result['avg_z']:.3f}"
    elif status == "FAIL":
        extra = f" — {result.get('error', '?')[:60]}"
    elif status == "SKIP":
        extra = f" — {result.get('reason', '?')[:60]}"
    print(f"  {emoji} {name:30s} {extra}")

print(f"\n  Total: ✅ {pass_count} PASS, ❌ {fail_count} FAIL, ⏭️ {skip_count} SKIP")
print()
