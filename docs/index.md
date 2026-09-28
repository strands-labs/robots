---
hide: [navigation, toc]
template_class: sr-home
---

# Strands Robots

<div class="sr-hero" markdown>
<div markdown>
<p class="sr-hero__title">One <em>Robot</em> object. Any robot.</p>
<p class="sr-hero__lead">Simulated or physical, arm or humanoid, the same call. Hand it to a Strands Agent and it becomes a tool the model can use. A real arm does not move until an operator says yes.</p>
<div class="sr-hero__actions">
<a class="sr-btn sr-btn--primary" href="start/">Start</a>
<a class="sr-btn" href="robots/">Pick a robot</a>
<span class="sr-install">pip install "strands-robots[sim-mujoco]"<button class="sr-copy" data-clipboard-text='pip install "strands-robots[sim-mujoco]"'>copy</button></span>
</div>
</div>
<div class="sr-hero__stage" markdown>
<robot-viewer name="so101" autoload></robot-viewer>
<label class="sr-pick">Try another robot <select data-robot-pick><option value="so101">so101</option></select></label>
</div>
</div>

<div class="sr-proof" markdown>
<div><strong>{{n:robots}}</strong><span>robots in the registry</span></div>
<div><strong>{{n:policy_providers}}</strong><span>policy providers behind one <code>run_policy</code> call</span></div>
<div><strong>{{n:native_drivers}}</strong><span>native hardware drivers that need no lerobot install</span></div>
</div>

<div class="sr-grid" markdown>
<div class="sr-card" markdown>
### [Start](start/index.md)
Install, move a simulated SO-101, plug in the real one, hand it to an agent, run the doctor.
</div>
<div class="sr-card" markdown>
### [Robots](robots/index.md)
The catalog: arms, hands, humanoids, quadrupeds, mobile bases. Which ones simulate, which ones have a driver.
</div>
<div class="sr-card" markdown>
### [Agents](learn/agents.md)
`Agent(tools=[robot])`. What the model sees, what it can call, and the approval gate in front of real motion.
</div>
<div class="sr-card" markdown>
### [Simulation](learn/simulation/index.md)
MuJoCo on the CPU by default, Newton and Isaac on a GPU. Worlds, objects, cameras, predicates, recordings.
</div>
<div class="sr-card" markdown>
### [Hardware](learn/hardware/drivers.md)
Native and lerobot drivers, teleoperation, cameras, calibration.
</div>
<div class="sr-card" markdown>
### [Mesh](learn/mesh/fleet.md)
Several robots on several machines with one e-stop. Zenoh, mTLS, an access-control list.
</div>
</div>

## One API, sim or real

<div class="sr-pair" markdown>

```python
from strands_robots import Robot

robot = Robot("so101", mode="sim")
robot.send_action({"1": 0.5}, n_substeps=200)
print(robot.get_robot_state()["content"][0]["text"])
robot.cleanup()
```

```python title="sketch"
from strands_robots import Robot

robot = Robot("so101", mode="real", driver="strands", port="/dev/ttyACM0")
robot.send_action({"shoulder_pan": 30.0})
print(robot.tool_spec["description"])
robot.cleanup()
```

</div>

The left fence runs on a laptop. The right one needs an SO-101 on USB; same tool, same verbs. The sim addresses joints by the model's names in radians; the native driver addresses servos by name in degrees. [Start here](start/index.md).
