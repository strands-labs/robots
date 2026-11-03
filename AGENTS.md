# AGENTS.md — the `dev` branch

You are working on strands-robots v1 on the `dev` branch. Read the design
(strands-labs/robots#4749) and the schedule (#3818) first.

- `dev` started from an empty package. Code arrives one part at a time; a module
  lives at its final path (`strands_robots/simulation.py`, `strands_robots/robot.py`, …).
- One PR per part, base `dev`. The part's gate from #4749's plan is the test that
  must be green.
- Core imports nothing heavy at module load (`mujoco`, `torch`, `lerobot`, `zenoh`,
  `isaacsim`, `warp`): each lives behind an extra and is imported where it is used.
- No `**kwargs` swallowing. A real robot moves only through its operator gate.
- Drawings: `docs/drawings/_tools/scene.py` (`README.txt` next to it). `v0*.py` in
  `docs/drawings/scenes/` are the design drawings.
