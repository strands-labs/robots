# Sample 04: Gymnasium Training — Reinforcement Learning Basics

**Level:** K12 Level 2 (Middle School) 🎓  
**Prerequisite:** [Sample 03: Multi-Robot Coordination](../03_multi_robot_coordination/)  
**Time:** 45-60 minutes

---

## 🎯 Learning Objectives

By the end of this lesson, you will:

1. **Understand what Gymnasium is** and why it's the standard for RL environments
2. **Learn how to wrap strands-robots** into a Gymnasium-compatible environment
3. **Explore the core RL concepts**: observation space, action space, reward signals, and done flags
4. **Train a simple RL agent** using Proximal Policy Optimization (PPO)
5. **Create custom reward functions** to guide robot learning

---

## 🧠 Key Concept: StrandsSimEnv

`StrandsSimEnv` is a wrapper that transforms a `strands-robots` MuJoCo simulation into a standard **Gymnasium environment**. This means:

- ✅ Works with any RL library (Stable-Baselines3, Ray RLlib, etc.)
- ✅ Follows the standard `reset()`, `step(action)`, `render()` API
- ✅ Defines clear observation/action spaces
- ✅ Provides reward signals for learning

```python
from strands_robots.envs import StrandsSimEnv

# Create a Gymnasium-compatible environment
env = StrandsSimEnv(data_config="so100")

# Standard Gym API
obs, info = env.reset()
action = env.action_space.sample()  # Random action
obs, reward, terminated, truncated, info = env.step(action)
```

---

## 📚 Core RL Concepts

### 1. **Observation Space**
The robot's "view" of the world (joint positions, velocities, sensor data).

```python
print(env.observation_space)
# Box(low=-inf, high=inf, shape=(N,), dtype=float32)
```

### 2. **Action Space**
What the robot can control (joint torques, velocities, or positions).

```python
print(env.action_space)
# Box(low=-1.0, high=1.0, shape=(M,), dtype=float32)
```

### 3. **Reward Signal**
A scalar value that tells the agent "how well it's doing." Examples:
- `+1.0` for reaching a goal
- `-0.01 * distance_to_goal` for getting closer
- `-0.1` for falling over

### 4. **Done Flags**
- **Terminated:** Episode ended naturally (goal reached, robot fell)
- **Truncated:** Episode ended due to time limit

---

## 🛠️ Code Walkthrough

### **File 1: `gym_basics.py`** — Random Agent

This script demonstrates the basic Gymnasium API:

1. **Create the environment** with `StrandsSimEnv(data_config="so100")`
2. **Inspect spaces** to understand input/output dimensions
3. **Run a random agent** for 100 steps
4. **Collect and visualize rewards**

**Key Code:**
```python
env = StrandsSimEnv(data_config="so100")
obs, info = env.reset()

for step in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}: Reward = {reward:.4f}")
    
    if terminated or truncated:
        obs, info = env.reset()
```

---

### **File 2: `train_ppo.py`** — Training with PPO

Proximal Policy Optimization (PPO) is a popular RL algorithm. We use `RLTrainer` from `strands_robots.training.rl_trainer` to train a policy:

1. **Create RLTrainer** with configuration for SO-100
2. **Train for 1000 steps** (small demo)
3. **Evaluate the trained policy**
4. **Save/load checkpoints**

**Key Code:**
```python
from strands_robots.training.rl_trainer import RLTrainer

trainer = RLTrainer(
    env_name="StrandsSimEnv",
    data_config="so100",
    algorithm="PPO",
    total_timesteps=1000
)

trainer.train()
trainer.save("ppo_so100_checkpoint")
trainer.evaluate(num_episodes=5)
```

**Note:** If no GPU is available, the script will demonstrate the API setup without full training.

---

### **File 3: `custom_reward.py`** — Custom Rewards

You can define custom reward functions to teach the robot specific behaviors:

```python
def reach_target_reward(obs, action, info):
    """
    Reward the robot for getting close to a target position.
    """
    target_pos = np.array([1.0, 0.5, 0.0])
    current_pos = obs[:3]  # Assume first 3 obs are position
    
    distance = np.linalg.norm(target_pos - current_pos)
    reward = -distance  # Negative distance (closer = higher reward)
    
    # Bonus for reaching target
    if distance < 0.1:
        reward += 10.0
    
    return reward
```

Then hook it into the environment:
```python
env = StrandsSimEnv(data_config="so100", reward_fn=reach_target_reward)
```

---

## 🏃 Running the Code

### 1. **Random Agent (Baseline)**
```bash
cd samples/04_gymnasium_training
python gym_basics.py
```

**Expected Output:**
```
Observation Space: Box(low=-inf, high=inf, shape=(18,), dtype=float32)
Action Space: Box(low=-1.0, high=1.0, shape=(6,), dtype=float32)

Step 0: Reward = -0.0234
Step 1: Reward = -0.0189
...
Average Reward: -1.234
```

### 2. **Train PPO Agent**
```bash
python train_ppo.py
```

**Expected Output:**
```
Training PPO on SO-100...
Timestep 100/1000 | Reward: -5.23
Timestep 200/1000 | Reward: -3.45
...
Training complete! Saved to ppo_so100_checkpoint/
```

### 3. **Custom Reward Function**
```bash
python custom_reward.py
```

---

## 🎓 Exercises

### **Exercise 1: Tune the Random Agent**
Modify `gym_basics.py` to run 500 steps instead of 100. Plot the cumulative reward over time.

**Hint:** Use `matplotlib` to visualize the reward curve.

```python
import matplotlib.pyplot as plt

rewards = []
for step in range(500):
    # ... run step ...
    rewards.append(reward)

plt.plot(rewards)
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Random Agent Performance")
plt.show()
```

---

### **Exercise 2: Increase Training Steps**
In `train_ppo.py`, increase `total_timesteps` to 10,000 and observe the improvement.

**Question:** How does the average reward change? Can you see the agent learning?

---

### **Exercise 3: Design Your Own Reward**
Create a new reward function in `custom_reward.py` that:
- Penalizes high joint velocities (to encourage smooth motion)
- Rewards keeping the robot upright (penalize if z-position < threshold)

**Starter Code:**
```python
def smooth_upright_reward(obs, action, info):
    # Extract joint velocities (assume obs[9:15] are velocities)
    velocities = obs[9:15]
    velocity_penalty = -0.01 * np.sum(np.abs(velocities))
    
    # Extract z-position (height)
    z_pos = obs[2]
    upright_reward = 1.0 if z_pos > 0.3 else -10.0
    
    return velocity_penalty + upright_reward
```

**Test it:**
```bash
python custom_reward.py
```

---

## 🎉 What You Learned

✅ **Gymnasium API**: `reset()`, `step()`, observation/action spaces  
✅ **RL Training Loop**: Collect experience → Update policy → Repeat  
✅ **PPO Algorithm**: A stable, beginner-friendly RL method  
✅ **Custom Rewards**: How to shape robot behavior through reward engineering  
✅ **Practical Skills**: Train, evaluate, and save RL policies

---

## 🚀 What's Next?

In **Sample 05: Advanced RL Techniques**, you'll explore:
- **Curriculum Learning**: Start simple, gradually increase difficulty
- **Multi-Task Learning**: Train one policy for multiple tasks
- **Sim-to-Real Transfer**: Bridge the gap between simulation and real robots
- **Hyperparameter Tuning**: Optimize learning rates, batch sizes, etc.

**Next Sample:** [05_advanced_rl](../05_advanced_rl/) 🔥

---

## 📖 Additional Resources

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3 Tutorials](https://stable-baselines3.readthedocs.io/)
- [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com/)
- [strands-robots RL Trainer API](https://github.com/strands-agents/strands-robots)

---

## 🛟 Troubleshooting

**Q: "ModuleNotFoundError: No module named 'strands_robots'"**  
A: Install the package: `pip install strands-robots`

**Q: "No GPU available, training is slow"**  
A: Use smaller timesteps (e.g., 1000 instead of 100k) or run on Google Colab with GPU.

**Q: "Reward is always negative"**  
A: This is normal! Many RL tasks start with negative rewards. The agent learns to maximize (less negative = better).

**Q: "MuJoCo license error"**  
A: MuJoCo is now free! Update to `mujoco>=2.3.0` with `pip install --upgrade mujoco`.

---

**Happy Learning! 🤖🎓**
