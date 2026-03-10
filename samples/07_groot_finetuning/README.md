# 🧠 Sample 07: GR00T N1.6 Fine-Tuning — Real Model Checkpoints

**Level:** 3 (Advanced / High School) | **Time:** 30 minutes | **Hardware:** GPU Required (24GB+ VRAM)

> Fine-tune NVIDIA's GR00T N1.6 vision-language-action model on custom robot data.

---

## 🎯 What You'll Learn

1. Download and inspect GR00T N1.6 checkpoints from HuggingFace
2. Understand the GR00T architecture (Eagle vision + Qwen3 LLM + Diffusion head)
3. Run inference: camera image + joint state + language → robot actions
4. Fine-tune GR00T on data you collected in Sample 05
5. Evaluate fine-tuned vs base model performance
6. Explore all 21 embodiment data configs for multi-robot support

## 📋 Prerequisites

- Samples 01–06 completed (especially Sample 05 for dataset)
- NVIDIA GPU with **24GB+ VRAM** (L40S recommended, A100/H100 ideal)
- Python packages:
  ```bash
  pip install strands-robots[vla]  # Core VLA support
  pip install isaac-gr00t          # NVIDIA GR00T package
  huggingface-cli login            # For model downloads
  ```

## 🏗️ GR00T Architecture

```
Camera Image (224×224)   Robot Joint State   Language Instruction
        │                      │                     │
        ▼                      ▼                     ▼
  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
  │ Eagle Vision  │    │  State MLP    │    │ Qwen3 Tokenizer  │
  │ (SigLIP2,     │    │  Encoder      │    │ (vocab=151680)   │
  │  27 layers,   │    │              │    │                  │
  │  hidden=1152) │    │              │    │                  │
  └──────────────┘    └──────────────┘    └──────────────────┘
        │                      │                     │
        └──────────────────────┼─────────────────────┘
                               ▼
                   ┌─────────────────────┐
                   │     Qwen3 LLM       │
                   │  (16 layers,        │
                   │   hidden=2048,      │
                   │   intermediate=6144)│
                   └─────────────────────┘
                               │
                               ▼
                   ┌─────────────────────┐
                   │   Diffusion Head    │  → Action Chunk (16 steps)
                   │  (4 denoise steps)  │    = future joint positions
                   └─────────────────────┘
```

**Key insight:** GR00T combines a vision encoder (sees the world), a language model
(understands instructions), and a diffusion head (generates smooth action trajectories).

## 🔄 N1.5 vs N1.6

| Feature | N1.5 | N1.6 |
|---------|------|------|
| Vision backbone | SigLIP | Eagle-Block2A-2B-v2 (SigLIP2) |
| Language model | – | Qwen3 (16 layers) |
| Action head | Flow matching | 4-step diffusion |
| Model size | ~1B | 2B–3B |
| HuggingFace ID | (checkpoint only) | `nvidia/GR00T-N1-2B` |
| Multi-embodiment | Limited | 21 data configs |
| Import path | `gr00t.model.policy` | `gr00t.policy.gr00t_policy` |

## 📁 Files in This Sample

| File | What It Does |
|------|-------------|
| `download_checkpoint.py` | Download GR00T N1.6 from HuggingFace, inspect architecture |
| `inference_demo.py` | Load model → run inference in MuJoCo simulation |
| `finetune_groot.py` | Full fine-tuning pipeline with progress logging |
| `evaluate_policy.py` | Compare base vs fine-tuned model (50 episodes) |
| `data_config_explorer.py` | Explore all 21 embodiment data configurations |
| `configs/finetune_so100.yaml` | SO-100 pick-and-place training config |
| `configs/finetune_g1.yaml` | G1 humanoid locomotion training config |

## 🚀 Quick Start

```bash
# 1. Explore what data configs exist (no GPU needed)
python samples/07_groot_finetuning/data_config_explorer.py

# 2. Download the base model (needs HuggingFace login)
python samples/07_groot_finetuning/download_checkpoint.py

# 3. Run inference demo (GPU required)
python samples/07_groot_finetuning/inference_demo.py

# 4. Fine-tune on your data (GPU required, ~30 min)
python samples/07_groot_finetuning/finetune_groot.py \
    --dataset ./datasets/my_pick_and_place \
    --epochs 50

# 5. Evaluate
python samples/07_groot_finetuning/evaluate_policy.py \
    --checkpoint ./checkpoints/groot_finetuned/best \
    --episodes 50
```

## 🎓 Exercises

### Exercise 1: Multi-Robot Inference
Run inference with the base model on 3 different robots. Edit `inference_demo.py`
to try `so100_dualcam`, `single_panda_gripper`, and `unitree_g1` configs.
What differences do you notice in the action shapes?

### Exercise 2: Data Efficiency
Fine-tune on 10 episodes vs 100 episodes from Sample 05. Compare success rates.
How many episodes does GR00T need to learn a new task?

### Exercise 3: Component Ablation
Try these fine-tuning configurations and compare:
- Projector + Diffusion only (default, fastest)
- Projector + Diffusion + LLM (slower, better generalization)
- Full model (slowest, maximum adaptation)

Record the training loss curves and final success rates for each.

## 🔗 Connections

| Direction | Sample | Why |
|-----------|--------|-----|
| ← Previous | [Sample 06](../06_ray_traced_training/) | Ray-traced training provides GPU simulation context |
| → Next | [Sample 08](../08_sim_to_real/) | Deploy this fine-tuned model to real hardware |
| ← Data from | [Sample 05](../05_data_collection/) | Your recorded dataset is the training data |
| Related | [GR00T N1.6 Lifecycle](#65) | Full production pipeline in the SDK |

## 📚 SDK Surface Covered

| Class/Function | Module | Purpose |
|---------------|--------|---------|
| `Gr00tPolicy` | `strands_robots.policies.groot` | Policy (3 modes: service/local N1.5/local N1.6) |
| `BaseDataConfig` | `strands_robots.policies.groot.data_config` | Embodiment config base class |
| `DATA_CONFIG_MAP` | `strands_robots.policies.groot.data_config` | Registry of 21 configs |
| `load_data_config()` | `strands_robots.policies.groot.data_config` | Load config by name |
| `create_custom_data_config()` | `strands_robots.policies.groot.data_config` | Register new configs |
| `create_trainer("groot")` | `strands_robots.training` | GR00T fine-tuning trainer |
| `Gr00tTrainer` | `strands_robots.training` | Direct trainer class |
| `TrainConfig` | `strands_robots.training` | Training hyperparameters |
| `evaluate()` | `strands_robots.training` | Policy evaluation harness |

## 🤝 Partners

- **NVIDIA** — Isaac-GR00T package, model weights, training infrastructure
- **HuggingFace** — Model hosting (`nvidia/GR00T-N1-2B`), dataset format (LeRobot v3)
