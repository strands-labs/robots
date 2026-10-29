#!/usr/bin/env python3
"""Spawn a robot in MuJoCo and run a policy - zero hardware required.

Goal: Show that Robot("so100") gives you a fully configured MuJoCo simulation
with one call - world created and the robot already added. You just add objects
and cameras, then run a policy.

Dependencies: pip install "strands-robots[sim-mujoco]"
Expected output: "run_policy status: success", then the rollout's own summary -
    "Policy complete on 'so100'", the policy and instruction, step count and sim
    time, and the note that MockPolicy does not read the instruction (so the
    "pick up the red cube" above shaped nothing). The summary is content[0], the
    text block an agent mounting this as a tool reads; the bare status alone
    hides the note.
Runtime: ~2 seconds on CPU.
"""

from strands_robots import MockPolicy, Robot

# Robot("so100") defaults to mode="sim" - safe, no hardware interaction.
# The factory already calls create_world() and adds the "so100" robot, so
# the scene is ready to use immediately (no create_world/add_robot needed).
sim = Robot("so100", mesh=False)

# Build a scene around the robot: red cube + front camera.
sim.add_object(
    name="cube",
    shape="box",
    position=[0.2, 0.0, 0.05],
    size=[0.025, 0.025, 0.025],
    color=[1, 0, 0, 1],
    mass=0.05,
)
sim.add_camera(name="front", position=[0.5, 0.0, 0.4], target=[0.2, 0, 0.05])

# Run a policy - MockPolicy produces sinusoidal test actions.
# robot_name="so100" is the robot the factory created.
result = sim.run_policy(
    robot_name="so100",
    policy_object=MockPolicy(),
    instruction="pick up the red cube",
    n_steps=50,
)

print(f"run_policy status: {result['status']}")
print(result["content"][0]["text"])
