# 🦆 GR00T N1 Training Pipeline — G1 × Cagatay's Room

## Pipeline Architecture

```
┌──────────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  Newton Sim  │───▶│ LeRobot Data│───▶│ Cosmos Trans │───▶│ GR00T N1.6   │
│  G1 x16      │    │ 64 episodes │    │ Sim → Real   │    │ Fine-tune    │
│  MuJoCo Warp │    │ 43 DOF      │    │ DreamZero    │    │ unitree_g1   │
└──────────────┘    └─────────────┘    └──────────────┘    └──────────────┘
```

## Status: ✅ Step 1 Complete

### Step 1: Data Collection (DONE ✅)
- Newton + MuJoCo Warp, 16 parallel worlds on L40S  
- 64 episodes × 400 steps = 25,600 frames
- 43 DOF (29 joints + 14 hand joints)
- LeRobot v2.1 format with parquet + metadata
- Location: `/home/ubuntu/room_sim/gr00t_dataset/`

### Step 2: Add Camera Rendering
Newton's `SensorTiledCamera` needs to be imported from the sensors submodule:
```bash
cd /tmp/newton
python3 -c "from newton.sensors import SensorTiledCamera; print('OK')"
```
Then re-run collection with camera frames → save as mp4 videos.

### Step 3: Cosmos Transfer 2.5 (Sim → Real)
```bash
# Domain transfer: sim renders → photorealistic
cd /home/ubuntu/cosmos-transfer2.5
python3 -m cosmos_transfer2.examples.inference \
  -i /home/ubuntu/room_sim/gr00t_dataset/cosmos_input.json \
  --setup.checkpoint_dir /home/ubuntu/checkpoints/DreamZero-DROID \
  --setup.output_dir /home/ubuntu/room_sim/gr00t_dataset/cosmos_output
```

### Step 4: GR00T N1.6 Fine-tune
```bash
source /home/ubuntu/groot_venv/bin/activate
python3 -m gr00t.experiment.launch_finetune \
  --base_model_path nvidia/GR00T-N1.6-3B \
  --dataset_path /home/ubuntu/room_sim/gr00t_dataset \
  --embodiment_tag unitree_g1 \
  --output_dir /home/ubuntu/room_sim/gr00t_finetuned \
  --max_steps 5000 \
  --global_batch_size 32 \
  --learning_rate 1e-4 \
  --num_gpus 1
```

### Step 5: Deploy to G1 Robot
```python
from strands_robots import Robot, Gr00tPolicy, create_policy

# Load fine-tuned model  
policy = create_policy("gr00t",
    model_path="/home/ubuntu/room_sim/gr00t_finetuned",
    embodiment_tag="unitree_g1")

# Connect to real G1
robot = Robot("unitree_g1")
robot.connect()

# Run policy
robot.run_policy(policy, task="walk_around_room")
```

## Key Components Used

| Component | Version | Purpose |
|-----------|---------|---------|
| Newton | 1.1.0.dev0 | GPU physics simulation |
| MuJoCo Warp | 3.5.0.2 | MuJoCo solver backend |
| Warp | 1.12.0 | GPU compute kernels |
| Cosmos Transfer 2.5 | 1.5.0 | Sim-to-real domain transfer |
| GR00T N1.6 | 3B | Foundation model for robot actions |
| L40S | 45GB VRAM | GPU compute |

## Dataset Format (LeRobot v2.1)
```
gr00t_dataset/
├── meta/
│   ├── info.json          # Dataset metadata
│   ├── episodes.jsonl     # Episode index
│   ├── tasks.jsonl        # Task definitions
│   └── modality.json      # GR00T modality config
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── videos/                # (Step 2: camera renders)
    └── chunk-000/
        └── observation.images.ego_view/
            └── episode_000000.mp4
```

## 43 DOF Joint Names
```
Left leg:  left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll
Right leg: right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll  
Waist:     waist_yaw, waist_roll, waist_pitch
Left arm:  left_shoulder_pitch/roll/yaw, left_elbow, left_wrist_roll/pitch/yaw
Right arm: right_shoulder_pitch/roll/yaw, right_elbow, right_wrist_roll/pitch/yaw
Left hand: index_0/1, middle_0/1, thumb_0/1/2
Right hand: index_0/1, middle_0/1, thumb_0/1/2
```
