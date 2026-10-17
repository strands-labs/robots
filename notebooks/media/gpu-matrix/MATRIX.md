# Part 4 §4 matrix (L40S, lerobot-policy-server over Zenoh, strands sim)

Contract-level: admitted + finite + inside limits. Only the task gates claim behaviour. Base checkpoints
(e.g. lerobot/smolvla_base) are not so101-trained: a row here is never a task-success claim.

| family | checkpoint | robot | mode | status | admission_ms | first_action_ms | turnaround_p50_ms | turnaround_p95_ms | hz | steps | max_abs_action_rad | max_joint_travel_rad | inside_limits | vram_mib | parity_max_abs_diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ACT (core) | alexis779/so101_ball_cup_act@31ff9e7e | so101 | chunk | admitted | 408.2 | 72.7 | 18.5 | 39.8 | 30.0 | 330 | 0.172 | 0.172 | True | 816 | 0.0 |
| Diffusion (core) | Lebruhbruh/diffusion_so101_pick_cube@f38d5537 | so101 | chunk | refused (lerobot-policy-server) |  |  |  |  |  |  |  |  |  |  |  |
| DM05 (dm05) | alexhegit/so101-simstudio-lab01-pnp-dm05-agent193-10k@a02640ce | so101 | chunk | fault | 227.5 | 704.8 | 690.1 | 690.1 | 29.9 | 23 | 0.158 | 0.158 | True | 20288 |  |
| DM05 (dm05) | alexhegit/so101-simstudio-lab01-pnp-dm05-agent193-10k@a02640ce | so101 | chunk | admitted | 231.0 | 755.6 | 651.5 | 665.5 | 8.0 | 88 | 0.158 | 0.158 | True | 20288 |  |
| EO1 (eo1) | crislmfroes/eo1-openarm-lift-fanta-bottle-clean@74fcfe47 | openarm | chunk | refused (strands, server_config) |  |  |  |  |  |  |  |  |  |  |  |
| EVO1 (evo1) | quangnd58/evo1-so101-multitask@5b6d832c | so101 | chunk | refused (strands, server_config) |  |  |  |  |  |  |  |  |  |  |  |
| EVO1 (evo1) | imstevenpmwork/super_chatton_evo1_stage1_pro@80cd23cd | so101 | chunk | admitted | 226.8 | 338.5 | 293.7 | 310.5 | 30.0 | 330 | 0.993 | 0.993 | True | 4386 |  |
| FastWAM (fastwam) | H2Ozone/fastwam_new_green_2026-07-19_16-26-08@239e1573 | so101 | chunk | refused (LeRobot chunk_inference_spec, config-level) |  |  |  |  |  |  |  |  |  |  |  |
| FineART (fineart_vla) | pepijn223/fineart_vla_super_poulain@ddb98404 | so101 | chunk | admitted | 227.1 | 272.6 | 238.0 | 247.5 | 30.0 | 330 | 0.98 | 0.98 | True | 18856 |  |
| FLUX.3 | 5hadytru/so101_bench_flux-3-action-so101@ac49157c | so101 | chunk | refused (lerobot-policy-server) |  |  |  |  |  |  |  |  |  |  |  |
| G1 SONIC (unitree_g1) | nepyope/sonic_walk@c37a5e42 | unitree_g1 | chunk | refused (strands, server_config) |  |  |  |  |  |  |  |  |  |  |  |
| Gaussian (gaussian_actor) | tiny:gaussian_actor (random, LeRobot defaults)@ | so101 | chunk | refused (lerobot-policy-server) |  |  |  |  |  |  |  |  |  |  |  |
| GR00T (groot) | fatdove/so101-cube-bowl_GR00T17@71dc7eae | so101 | chunk | admitted | 229.3 | 205.6 | 172.4 | 174.2 | 30.0 | 330 | 0.878 | 0.902 | True | 13698 |  |
| LaWAM (lawam) | 5hadytru/so101_bench_LaWAM_full_fine_tune@a7ddd367 | so101 | chunk | admitted | 223.9 | 238.3 | 213.0 | 218.7 | 30.0 | 330 | 1.713 | 1.713 | True | 7566 |  |
| LingBot-VA | lerobot/lingbot_va_base@a9b1da7a | aloha | chunk | refused (lerobot-policy-server) |  |  |  |  |  |  |  |  |  |  |  |
| MolmoAct2 (molmoact2) | XvKuoMing/so101_chess_molmoact2@136a2462 | so101 | chunk | admitted | 418.1 | 273.2 | 244.5 | 247.0 | 30.0 | 330 | 1.57 | 1.57 | True | 14942 |  |
| Multi-task DiT | seriintan/multi_task_dit_frazier@7c38d407 | so101 | chunk | refused (lerobot-policy-server) |  |  |  |  |  |  |  |  |  |  |  |
| π0 (pi) | tonghuiwang123/so100-pi0-new40@54c1c770 | so100 | chunk | admitted | 125.9 | 272.7 | 225.2 | 232.8 | 30.0 | 330 | 2.16 | 1.961 | True | 9474 |  |
| π0.5 (pi) | pdhinaka/pi05_so101_blue_cube_gray_box_cleaned_20k@ffbee39f | so101 | chunk | admitted | 242.6 | 305.4 | 262.0 | 281.1 | 30.0 | 330 | 1.05 | 1.252 | True | 9858 |  |
| π0-FAST (pi) | gpudad/pi0fast-so101-pick-cube@44d2a385 | so101 | chunk | refused (lerobot-policy-server) |  |  |  |  |  |  |  |  |  |  |  |
| SmolVLA (smolvla) | lerobot/smolvla_base@5e8d12a6 | so101 | chunk | admitted | 232.9 | 269.9 | 241.6 | 247.3 | 30.0 | 330 | 0.169 |  | True | 1466 |  |
| SmolVLA (smolvla) | lerobot/smolvla_base@5e8d12a6 | so101 | local | ran |  |  |  |  | 158.4 | 330 | 0.138 | 0.136 | True | 929 |  |
| SmolVLA (smolvla) | lerobot/smolvla_base@5e8d12a6 | so101 | rtc_guided | admitted | 227.6 | 303.0 | 360.0 | 371.4 | 30.0 | 330 | 0.144 |  | True | 2122 |  |
| TD-MPC (core) | tiny:tdmpc (random, LeRobot defaults)@ | so101 | chunk | refused (lerobot-policy-server) |  |  |  |  |  |  |  |  |  |  |  |
| VLA-JEPA (vla_jepa) | charlott-rgb/so101_acrylic_vla_jepa_b8_15000@1e612363 | so101 | chunk | admitted | 218.0 | 138.0 | 105.3 | 107.0 | 30.0 | 330 | 1.521 | 1.521 | True | 6562 |  |
| VQ-BeT (core) | tgibbons/SO-101-VQBET-ball2-claw_30k@5bb580f9 | so101 | chunk | refused (lerobot-policy-server) |  |  |  |  |  |  |  |  |  |  |  |
| WALL-X (wallx) | imstevenpmwork/wall_x_super_chatton@de41b2df | so101 | chunk | refused (lerobot-policy-server) |  |  |  |  |  |  |  |  |  |  |  |
| XVLA (xvla) | imstevenpmwork/xvla_super_chatton_2@36fbf138 | so101 | chunk | admitted* | 225.9 | 971.8 | 107.6 | 109.2 | 30.0 | 330 | 1.23 | 1.23 | True | 2340 |  |

\* ran on a local copy whose config.json was patched (the published config is stale):

- **xvla**: `{"input_features": {"observation.state": {"type": "STATE", "shape": [6]}, "observation.images.image": {"type": "VISUAL", "shape": [3, 256, 256]}, "observation.images.image2": {"type": "VISUAL", "shape": [3, 256, 256]}}}`

## Robustness (cuda, real lerobot-policy-server processes; scripts/matrix_robust.py)

- **robust_busy** pass=True: `{"second_client": "AdmissionDeniedError: Remote admission denied for deployment 'busy': another client owns the deployment; stop that client before retrying.", "second_client_steps": 0, "first_client_steps": 92, "sessions_admitted": 1}`
- **robust_contract** pass=True: `{"wrong_camera": {"refused": true, "message": "ValueError: Policy('zenoh://127.0.0.1:7822/contract') needs camera(s) ['wrist'] but the observation has ['front']: add them with sim.camera(name, size=(w, h), attach=robot) or cameras={name: index} on hardware"}, "wrong_state_width": {"refused": true, "message": "ValueError: deployment state component 'shoulder_pan.pos' is not a joint of aloha (action keys: ['left/waist', 'left/shoulder', 'left/elbow', 'left/forearm_roll', 'left/wrist_angle', 'left/wrist_rotate', 'left/left_finger', 'right/waist', 'right/shoulder', 'right/elbow', 'right/forearm_roll', 'right/wrist_angle', 'right/wrist_rotate', 'right/left_finger']); a LeRobot name is '<action key>.pos'"}, "sessions_admitted": 0, "wrong_semantics": {"refused": true, "message": "ValueError: unknown LeRobot semantics 'furlongs-v1': add it to strands_robots/units/semantics.json (known: koch-range-gripper-percent-v1, radians-v1, so100-degrees-gripper-percent-v1, so100-range-gripper-percent-v1, so101-degrees-gripper-percent-v1, so101-range-gripper-percent-v1) or pass units= explicitly"}, "sessions_admitted_sem": 0}`
- **robust_kill** pass=True: `{"checkpoint": "alexis779/so101_ball_cup_act@31ff9e7e", "grace_s": 1.0, "stopped_by": "fault", "fault": "Policy('zenoh://127.0.0.1:7821/kill'): Server presence was lost during control acknowledgement", "steps": 155, "kill_to_fault_s": 0.573, "kill_to_end_s": 0.576, "acts_after_fault": 0, "steps_after_kill": 17, "held_target_max_dev_rad": 0.0011474129688067158}`
- **robust_two** pass=True: `{"ids": ["so101", "so101_2"], "wall_s": 10.93, "left": {"steps": 300, "stopped_by": "steps"}, "right": {"steps": 300, "stopped_by": "steps"}, "sessions": [["2dd4fc2198cd4398b78137a73841c543"], ["d0ca3de6885a4505aa1e1e7b16371b3c"]], "distinct_sessions": true}`

## Refusals and faults (verbatim)

**diffusion** (refused (lerobot-policy-server)):
```
    raise ValueError("Chunk inference requires n_obs_steps=1; observation history is unsupported.")
ValueError: Chunk inference requires n_obs_steps=1; observation history is unsupported.
exit=1 last client error: ConnectionError: Cannot open remote connection to 'tcp/127.0.0.1:7820' for deployment 'diffusion': Unable to connect to any of [tcp/127.0.0.1:7820]. Timeout! at /root/.cargo/git/checkouts/zenoh-9c599d5ef3e0480e/81c6c93/zenoh/src/net/runtime/orchestrator.rs:345.. Check that the server/router is running and the endpoint, network access and Zenoh security configuration are correct.
```

**dm05** (fault):
```
Policy('zenoh://127.0.0.1:7820/dm05'): Invalid RTC delay
```

**eo1** (refused (strands, server_config)):
```
ValueError: crislmfroes/eo1-openarm-lift-fanta-bottle-clean state is (8,); openarm has 18 joints ('openarm_left_joint1.pos', 'openarm_left_joint2.pos', 'openarm_left_joint3.pos', 'openarm_left_joint4.pos', 'openarm_left_joint5.pos', 'openarm_left_joint6.pos', 'openarm_left_joint7.pos', 'openarm_left_finger_joint1.pos', 'openarm_left_finger_joint2.pos', 'openarm_right_joint1.pos', 'openarm_right_joint2.pos', 'openarm_right_joint3.pos', 'openarm_right_joint4.pos', 'openarm_right_joint5.pos', 'openarm_right_joint6.pos', 'openarm_right_joint7.pos', 'openarm_right_finger_joint1.pos', 'openarm_right_finger_joint2.pos')
```

**evo1** (refused (strands, server_config)):
```
DecodingError: The fields `drop_last` are not valid for Evo1Config
```

**fastwam** (refused (LeRobot chunk_inference_spec, config-level)):
```
ValueError: Chunk inference cannot sample temporal history (observation_delta_indices).
```

Note: Evaluated LeRobot 53feaef PreTrainedPolicy.chunk_inference_spec (FastWAMPolicy does not override it) on this checkpoint's config.json: n_obs_steps=1, observation_delta_indices=[0,4,...,32]. Not served: the checkpoint needs Wan2.2-TI2V-5B + umt5-xxl beside its 12 GB model, over the 20 GB VRAM budget on the shared L40S. The server runs this same check at warmup.

**flux3** (refused (lerobot-policy-server)):
```
PermissionError: [Errno 13] Permission denied: '/root/flux3/weights/so101_ourstats'
exit=1 last client error: ConnectionError: Cannot open remote connection to 'tcp/127.0.0.1:7820' for deployment 'flux3': Unable to connect to any of [tcp/127.0.0.1:7820]. Timeout! at /root/.cargo/git/checkouts/zenoh-9c599d5ef3e0480e/81c6c93/zenoh/src/net/runtime/orchestrator.rs:345.. Check that the server/router is running and the endpoint, network access and Zenoh security configuration are correct.
```

**g1_sonic** (refused (strands, server_config)):
```
ValueError: <ckpt> state is (64,); unitree_g1 has 29 joints ('left_hip_pitch_joint.pos', 'left_hip_roll_joint.pos', ...)
```

Note: pi05 checkpoint whose 64-dim state/action are SONIC motion tokens; LeRobot drives them through its SONIC whole-body controller (lerobot/robots/unitree_g1/controllers/sonic_whole_body.py), which the strands G1 sim body does not have. Refused before any server or download (config.json only). Stretch row: strands' message names the dimension mismatch, not SONIC — a motion-token semantics entry would be the fix.

**gaussian** (refused (lerobot-policy-server)):
```
    raise ValueError(
ValueError: The default chunk runner requires a positive chunk_size; provide a custom runner.
exit=1 last client error: ConnectionError: Cannot open remote connection to 'tcp/127.0.0.1:7820' for deployment 'gaussian': Unable to connect to any of [tcp/127.0.0.1:7820]. Timeout! at /root/.cargo/git/checkouts/zenoh-9c599d5ef3e0480e/81c6c93/zenoh/src/net/runtime/orchestrator.rs:345.. Check that the server/router is running and the endpoint, network access and Zenoh security configuration are correct.
```

**lingbot_va** (refused (lerobot-policy-server)):
```
    raise ValueError(f"Chunk inference cannot sample temporal history ({field_name}).")
ValueError: Chunk inference cannot sample temporal history (observation_delta_indices).
exit=1 last client error: ConnectionError: Cannot open remote connection to 'tcp/127.0.0.1:7820' for deployment 'lingbot_va': Unable to connect to any of [tcp/127.0.0.1:7820]. Timeout! at /root/.cargo/git/checkouts/zenoh-9c599d5ef3e0480e/81c6c93/zenoh/src/net/runtime/orchestrator.rs:345.. Check that the server/router is running and the endpoint, network access and Zenoh security configuration are correct.
```

**multi_task_dit** (refused (lerobot-policy-server)):
```
    raise ValueError("Chunk inference requires n_obs_steps=1; observation history is unsupported.")
ValueError: Chunk inference requires n_obs_steps=1; observation history is unsupported.
exit=1 last client error: ConnectionError: Cannot open remote connection to 'tcp/127.0.0.1:7820' for deployment 'multi_task_dit': Unable to connect to any of [tcp/127.0.0.1:7820]. Timeout! at /root/.cargo/git/checkouts/zenoh-9c599d5ef3e0480e/81c6c93/zenoh/src/net/runtime/orchestrator.rs:345.. Check that the server/router is running and the endpoint, network access and Zenoh security configuration are correct.
```

**pi0fast** (refused (lerobot-policy-server)):
```
    raise ValueError(
ValueError: Couldn't instantiate the backend tokenizer from one of: 
    raise RuntimeError("Failed to load required tokenizers for PI0FastPolicy initialization") from e
RuntimeError: Failed to load required tokenizers for PI0FastPolicy initialization
```

**tdmpc** (refused (lerobot-policy-server)):
```
    raise ValueError(f"Chunk inference cannot sample temporal history ({field_name}).")
ValueError: Chunk inference cannot sample temporal history (observation_delta_indices).
exit=1 last client error: ConnectionError: Cannot open remote connection to 'tcp/127.0.0.1:7820' for deployment 'tdmpc': Unable to connect to any of [tcp/127.0.0.1:7820]. Timeout! at /root/.cargo/git/checkouts/zenoh-9c599d5ef3e0480e/81c6c93/zenoh/src/net/runtime/orchestrator.rs:345.. Check that the server/router is running and the endpoint, network access and Zenoh security configuration are correct.
```

**vqbet** (refused (lerobot-policy-server)):
```
    raise ValueError("Chunk inference requires n_obs_steps=1; observation history is unsupported.")
ValueError: Chunk inference requires n_obs_steps=1; observation history is unsupported.
exit=1 last client error: ConnectionError: Cannot open remote connection to 'tcp/127.0.0.1:7820' for deployment 'vqbet': Unable to connect to any of [tcp/127.0.0.1:7820]. Timeout! at /root/.cargo/git/checkouts/zenoh-9c599d5ef3e0480e/81c6c93/zenoh/src/net/runtime/orchestrator.rs:345.. Check that the server/router is running and the endpoint, network access and Zenoh security configuration are correct.
```

**wall_x** (refused (lerobot-policy-server)):
```
    raise ValueError(
ValueError: WALL-X requires tokenized inputs from its policy preprocessor; missing ['attention_mask', 'image_grid_thw', 'input_ids', 'moe_token_types', 'pixel_values'].
exit=1 last client error: ConnectionError: Cannot open remote connection to 'tcp/127.0.0.1:7820' for deployment 'wall_x': Unable to connect to any of [tcp/127.0.0.1:7820]. Timeout! at /root/.cargo/git/checkouts/zenoh-9c599d5ef3e0480e/81c6c93/zenoh/src/net/runtime/orchestrator.rs:345.. Check that the server/router is running and the endpoint, network access and Zenoh security configuration are correct.
```

