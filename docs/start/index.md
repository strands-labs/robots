---
description: Five pages, in order, from an empty environment to an agent that moves a robot.
cta: true
copy_prompt: true
---

# Start

You leave with a simulated SO-101 you can command from Python, the two lines that move the physical one, an agent that calls the robot as a tool, and a way to check the machine when any of that fails.

| page | you leave with |
|---|---|
| [Install](install.md) | `strands-robots[sim-mujoco]` in a Python `{{extras:python}}` environment, the extras table, robot models on disk, `MUJOCO_GL` sorted |
| [First robot](first-robot.md) | an SO-101 in MuJoCo: joints moved, state read, a PNG saved, a cube on the table |
| [First real arm](first-real-arm.md) | the arm's USB port, the native driver rehearsed on the arm's model, the native and lerobot lines that move it, calibration, what is refused |
| [First agent](first-agent.md) | `Agent(tools=[robot])`, a sentence that moves the sim arm, the approval gate stopping a real one |
| [Doctor](doctor.md) | `strands-robots doctor`: what each of the fifteen probes checks and what its verdict means |

Every `python` fence on these pages ran against this commit on a laptop with no GPU. Fences that need an arm on USB are marked `sketch`.
