# Sample 02: Policy Playground 🎮

**Level:** K12 Level 1 (Elementary)  
**Time:** 30-45 minutes  
**Prerequisites:** Sample 01 (Your First Robot)

## 🎯 Learning Objectives

By the end of this lesson, you will:
- Understand what a **policy** is and how it controls robots
- Learn how policies take observations and return actions
- Explore different policy providers (18 providers, 52 aliases!)
- Compare how the same policy behaves on different robots
- Create your own custom robot policy

---

## 🧠 Key Concept: What is a Policy?

A **policy** is like the "brain" of a robot. It's a function that:
1. **Takes in** observations (what the robot sees/senses)
2. **Decides** what to do
3. **Returns** actions (how the robot should move)

Think of it like this:
- **Observations** = "What's happening around me?"
- **Policy** = "What should I do about it?"
- **Actions** = "Move these motors!"

### The Policy Interface

In code, a policy looks like this:

```python
async def get_actions(self, observations, instruction=None):
    # observations: Current robot state (joint positions, sensors, etc.)
    # instruction: Optional text command from a human
    # Returns: List of actions (one per robot)
    return [{"joint_positions": [...], "gripper": [...]}]
```

Every policy in the strands-robots SDK follows this pattern!

---

## 🎪 The Mock Policy

The **mock policy** is a simple testing policy that creates sinusoidal (wave-like) motion:

```python
# Mock policy makes robots move in smooth waves
action = amplitude * sin(frequency * time + phase)
```

- **Amplitude:** How far to move (default: 0.1 radians ≈ 5.7 degrees)
- **Frequency:** How fast to wave (default: 1.0 Hz = 1 cycle per second)
- **Phase:** Starting position in the wave

This creates smooth, predictable motion perfect for testing!

---

## 🗺️ Policy Providers: The Big Picture

The strands-robots SDK supports **18 providers** with **52 aliases** total:

| Provider | Description | Requires GPU? |
|----------|-------------|---------------|
| `mock` | Sinusoidal test motion | ❌ No |
| `stopped` | No motion (all zeros) | ❌ No |
| `random` | Random actions | ❌ No |
| `pi0` | π₀ (Pi Zero) VLA | ✅ Yes |
| `act` | Action Chunking Transformer | ✅ Yes |
| `smolvla` | Small Vision-Language-Action | ✅ Yes |
| `openvla` | Open VLA | ✅ Yes |
| `groot` | NVIDIA Isaac GR00T | ✅ Yes |
| ... and more! | | |

Use `list_providers()` to see all available providers and their capabilities!

---

## 📖 Code Walkthrough

### 1. Creating a Robot and Running a Policy

```python
from strands_robots import Robot

# Create a simulation with the so100 robot
sim = Robot("so100")

# Run the mock policy for 3 seconds
sim.run_policy(
    robot_name="so100",
    policy_provider="mock",
    duration=3.0
)
```

### 2. Recording Video

```python
# Record a video of the robot moving
sim.record_video(
    robot_name="so100",
    policy_provider="mock",
    duration=3.0,
    fps=30,
    output_path="mock_policy_demo.mp4"
)
```

### 3. Exploring Available Providers

```python
from strands_robots.policy import list_providers

# Get all available policy providers
providers = list_providers()

# Providers that work locally (no GPU needed)
local_providers = ["mock", "stopped", "random"]

# Providers that need GPU/special hardware
gpu_providers = ["pi0", "act", "smolvla", "openvla", "groot"]
```

### 4. Creating a Policy Directly

```python
from strands_robots.policy import create_policy

# Create a mock policy instance
policy = create_policy("mock")

# Policies inherit from the Policy abstract base class (ABC)
# They must implement: async get_actions(observations, instruction)
```

---

## 🎯 Exercises

### Exercise 1: Try Different Robots (Easy)

**Goal:** See how the same policy works on different robots.

Modify `policy_playground.py` to try these robots:
- `"so100"` (Stanford Humanoid)
- `"unitree_g1"` (Unitree G1 Humanoid)
- `"panda"` (Franka Panda Arm)
- `"aloha"` (ALOHA Bimanual System)

**Question:** Does the mock policy create the same motion on all robots? Why or why not?

<details>
<summary>💡 Hint</summary>

Different robots have different numbers of joints! A humanoid has ~20 joints, but Panda only has 7. The mock policy creates sinusoidal motion for each joint, so you'll see different overall behavior.
</details>

---

### Exercise 2: Compare Mock vs Stopped (Medium)

**Goal:** Understand the difference between active and inactive policies.

1. Run `compare_policies.py` to see mock policy on multiple robots
2. Change the policy from `"mock"` to `"stopped"`
3. Record videos of both

**Question:** What's the difference? When would you use the "stopped" policy?

<details>
<summary>💡 Hint</summary>

The "stopped" policy returns all zeros — no motion! This is useful for:
- Testing if your robot is set up correctly
- Creating a baseline comparison
- Safely initializing a robot before running a real policy
</details>

---

### Exercise 3: Create a Custom Wave Policy (Advanced)

**Goal:** Build your own policy from scratch!

1. Open `custom_policy.py`
2. Modify the `WavePolicy` to change:
   - The wave speed (frequency)
   - The wave size (amplitude)
   - Which joints move
3. Test it on different robots

**Challenge:** Can you make the robot "wave hello" with just its arm?

<details>
<summary>💡 Hint</summary>

You can control individual joints by creating an action array where only some joints move:

```python
actions = np.zeros(num_joints)  # Start with all zeros
actions[3] = 0.3 * np.sin(2.0 * time)  # Only move joint 3
```

For humanoid robots, joints 3-6 are usually the right arm!
</details>

---

## 🎓 What You Learned

- ✅ A **policy** takes observations and returns actions
- ✅ The **mock policy** creates sinusoidal test motion
- ✅ There are **18 providers** with different capabilities
- ✅ **Local providers** (mock, stopped, random) work without GPU
- ✅ **GPU providers** (pi0, act, smolvla) need special hardware
- ✅ You can create **custom policies** using `register_policy()`
- ✅ The same policy behaves differently on different robots

---

## 🚀 What's Next?

In **Sample 03: Vision-Language-Action (VLA) Models**, you'll:
- Use real AI policies that understand language and vision
- Give robots text instructions like "pick up the cube"
- Learn about transformer models and neural networks
- See how VLAs generalize across different tasks

**Ready to level up?** Head to `samples/03_vla_basics/` next!

---

## 📚 Additional Resources

- [LeRobot Policy Documentation](https://huggingface.co/docs/lerobot)
- [Strands Robots API Reference](https://github.com/strands-agents/strands-robots)
- [Understanding Robot Policies (Blog Post)](https://www.example.com)
- [Sinusoidal Motion in Robotics](https://www.example.com)

---

## 🆘 Troubleshooting

**"Robot not found" error?**
- Check the robot name is spelled correctly
- Run `Robot.list_robots()` to see available robots

**Video recording fails?**
- Make sure you have write permissions in the directory
- Check that the simulation is running (X11/display required)

**Policy not found?**
- Use `list_providers()` to see available policies
- Some policies need extra dependencies (`pip install lerobot`)

---

**Happy Policy Playing! 🎮🤖**
