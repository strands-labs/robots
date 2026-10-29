# Paired script for notebooks/10_gpu_backends.ipynb. Run: .venv/bin/python notebooks/scripts/10_gpu_backends.py
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # nb: script-only

# %% [markdown]
# # 10 · GPU backends
#
# The engine is an argument to `Simulation`; robots, policies and `robot.run` do not change.
# The Isaac Sim and MuJoCo-Warp results below were measured on an L40S and are shown, not re-run (this Mac has no CUDA).
#
# ![one Simulation, three engines](media/10/one_world_three_engines.svg)

# %% [markdown]
# ## The same world on this Mac
#
# MuJoCo on the CPU: the SO-101 and the G1 in one world, one arm move, one render.

# %%
from PIL import Image
import _nb
from strands_robots import Robot, Simulation

sim = Simulation("mujoco")
arm = Robot("so101", sim=sim)
g1 = Robot("g1", sim=sim, position=[1.0, 0.0, 0.0])
arm.act({"shoulder_pan": 0.5}, steps=200)
frame = sim.render()
Image.fromarray(frame).save(_nb.MEDIA / "10" / "mujoco_so101_g1.png")
print(sim.backend_name, sim.physics_name, frame.shape, [r.id for r in (arm, g1)])
sim.close()

# %% [markdown]
# ![MuJoCo render](media/10/mujoco_so101_g1.png)

# %% [markdown]
# ## On a CUDA box
#
# ```python
# # Isaac Sim (PhysX, RTX): uv pip install -e ".[isaac]", OMNI_KIT_ACCEPT_EULA=YES
# sim = Simulation("isaac")
# arm = Robot("so101", sim=sim)
# g1 = Robot("g1", sim=sim, position=[1.0, 0.0, 0.0])
# arm.act({"shoulder_pan": 0.5}, steps=200)
# frame = sim.render()
# sim.close()  # one SimulationApp per process: a second open Simulation("isaac") is refused
#
# # MuJoCo stepped by MuJoCo-Warp, N copies: uv pip install -e ".[newton]"
# sim = Simulation("mujoco", physics="newton", num_envs=1024)
# arm, g1 = Robot("so101", sim=sim), Robot("g1", sim=sim, position=[1.0, 0.0, 0.0])
# sim.step(200)  # env 0 is mirrored to the CPU for observe/render
# ```
#
# Without CUDA the same calls refuse by name, before anything is built:

# %%
for kw in [dict(backend="isaac"), dict(physics="newton"), dict(num_envs=4)]:
    try:
        Simulation(**kw).close()
        print(kw, "-> opened")
    except (ImportError, ValueError) as exc:
        print(f"{kw}\n  {type(exc).__name__}: {exc}\n")

# %% [markdown]
# ## Isaac Sim on the L40S
#
# MJCF to USD through Isaac's importers (cached), MuJoCo servo gains authored onto PhysX drives, one articulation per robot.
#
# ![Isaac Sim: SO-101 and G1](media/isaac/isaac_so101_g1.png)
#
# ## MuJoCo-Warp: env 0 of 1,024
#
# MuJoCo builds, reads and renders; MuJoCo-Warp steps every copy in one CUDA graph.
#
# ![newton, 1024 envs](media/isaac/newton_so101_g1_1024envs.gif)
#
# ## Measured on the L40S (isaacsim 6.1.0, mujoco-warp 3.11)
#
# | what | result |
# |---|---|
# | Isaac: so101 + g1, 1,000 physics steps | 4,336 ms = 231 steps/s (CPU PhysX), render 480x640x3 |
# | Isaac: app boot / load 2 robots | 10.2 s cold / 1.8 s (USD cache warm) |
# | Isaac: `Robot.run`, 100 steps at 50 Hz | 2.00 s sim time in 5.05 s wall (0.40x real time) |
# | Isaac: shared-world tests | 15 passed in 61.26 s; the G1 drifts 0.37° while the arm moves |
# | Newton: so101 + g1, `num_envs=1024` | 200 steps in 117 ms = 1,754,372 env-steps/s |
# | Newton: `num_envs=1` | 2,162 steps/s |
#
# Not there yet on Isaac: `contacts()` returns `[]`; `randomize(friction=, mass=)`, `set_gains` and objects from a file
# raise `NotImplementedError`; poses agree to ±1e-3 (PhysX float32); GPU PhysX not tried.

# %% [markdown]
# ## Every LeRobot policy family over Zenoh, on the L40S
#
# One real `lerobot-policy-server` per row, `Policy("zenoh://…/<row>")`, `robot.run(seconds=11)` at 30 Hz.
# Contract level only (admitted, finite, inside the joint limits): base checkpoints are not trained for these scenes,
# so no row is a task-success claim. ACT gave identical actions locally and remotely (max abs diff 0.0).

# %%
rows = []
for line in (_nb.MEDIA / "gpu-matrix" / "MATRIX.md").read_text().splitlines():
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) == 16 and cells[0] not in ("family", "---"):
        rows.append({"family": cells[0], "mode": cells[3], "status": cells[4],
                     "p50 ms": cells[7] or "-", "Hz": cells[9] or "-", "VRAM MiB": cells[14] or "-"})
admitted = {r["family"] for r in rows if r["status"].startswith("admitted")}
print(f'{len(admitted)} of {len({r["family"] for r in rows})} families admitted, {len(rows)} rows')
_nb.show(_nb.table(rows))
