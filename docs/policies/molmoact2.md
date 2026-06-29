---
description: MolmoAct2 SO-100/101 in MuJoCo - concrete action_dim, range, units, the SO-arm embodiment mapping, and the "runs but does not move" diagnostics.
---

# MolmoAct2 (SO-100 / SO-101)

MolmoAct2 runs through the [LeRobot Local](lerobot-local.md) provider
(`create_policy("lerobot_local", ...)`). This page documents the concrete
action/observation contract for the SO-100/101 checkpoints and how to debug the
common "the policy runs but the arm does not move in MuJoCo" report.

For install (lerobot from source + the `[molmoact2]` extra), caching, the
processor/`norm_stats.json` bridge and camera routing, see the
[LeRobot Local](lerobot-local.md) page.

## Action / observation contract (SO-100/101)

`allenai/MolmoAct2-SO100_101` is a 6-DoF SO-arm checkpoint:

| Quantity | Value |
| --- | --- |
| `action_dim` | 6 (5 arm joints + 1 gripper) |
| `observation.state` dim | 6 |
| Arm joint units | DEGREES (LeRobot `MotorNormMode.DEGREES`) |
| Gripper units | RANGE_0_100 (LeRobot `MotorNormMode.RANGE_0_100`), **not** degrees |
| Joint order | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper |
| Cameras | `observation.images.image` (front), `observation.images.wrist_image` |

The MuJoCo `so101` sim, by contrast, expresses revolute joints in **RADIANS**
with bare numeric joint names `1..6` (the gripper is joint `6`, range
`[-0.175, 1.745]` rad). The mismatch in both units and naming is what the
embodiment map reconciles.

## SO-arm embodiment mapping

The `so101` (and `so100`) entry in
`strands_robots/policies/lerobot_local/embodiments.json` declares the mapping
that bridges the model's training units to the sim:

```json
"so101": {
  "obs_rename": {"front": "observation.images.image", "wrist": "observation.images.wrist_image"},
  "state_keys":  ["1", "2", "3", "4", "5", "6"],
  "action_keys": ["1", "2", "3", "4", "5", "6"],
  "state_units": "degrees",
  "action_units": "degrees",
  "gripper_index": 5,
  "gripper_joint_range": [-0.175, 1.745]
}
```

At inference the policy converts in both directions:

* **state (sim -> model)**: arm radians -> degrees; gripper joint radians -> 0..100.
* **action (model -> sim)**: arm degrees -> radians; gripper 0..100 -> joint radians.

So a model action of `[30, 30, 30, 30, 30, 50]` (degrees / 0..100) maps to
`[0.524, 0.524, 0.524, 0.524, 0.524, 0.785]` radians before it reaches the sim.
Without this conversion, `30.0` is interpreted as 30 radians, which saturates the
joint's `[-pi, pi]`-scale limit and the arm appears frozen.

### Calibration mid-point (`joint_mids`)

LeRobot's `MotorNormMode.DEGREES` is **mid-point-centered**: the value a
checkpoint trains on is the angular displacement from each motor's calibration
mid-point, not the absolute joint angle. Ground truth
(`lerobot/motors/motors_bus.py`, `_normalize` / `_unnormalize`):

```python
mid = (range_min + range_max) / 2          # calibration mid, encoder ticks
degrees = (raw - mid) * 360 / max_res      # reported state/action
```

On real hardware this is correct automatically because the driver owns the
conversion and knows each servo's calibration mid. In sim, `qpos = 0` (the MJCF
home pose) is generally **not** the calibration mid, so the absolute
`deg = rad * 180/pi` conversion is offset per joint from the training
distribution. After LeRobot's `MIN_MAX` state normalization that offset can push
`observation.state` outside the dataset range, degrading the policy in sim while
the same checkpoint works on the arm.

Supply the per-joint mid-points (in degrees, aligned to `state_keys` /
`action_keys`) via `joint_mids` so the conversion mid-centers like
`motors_bus`. The gripper column (`gripper_index`) is exempt (RANGE_0_100 has no
mid). Empty (the default) treats every mid as `0` -- i.e. sim `qpos = 0` is
assumed to coincide with the calibration mid, preserving the prior
absolute-degrees behavior:

```json
"so101": {
  "state_units": "degrees",
  "action_units": "degrees",
  "gripper_index": 5,
  "gripper_joint_range": [-0.175, 1.745],
  "joint_mids": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}
```

Populate `joint_mids` from the calibration the checkpoint was trained against
(`mid = (range_min + range_max) / 2` per motor, expressed as a sim-frame joint
angle in degrees). Leave it empty only when the sim home pose already coincides
with the calibration mid.

Pass `embodiment="so101"` to the policy (or `"so100"`, `"so_real"` for hardware):

```python
policy = create_policy(
    "lerobot_local",
    pretrained_name_or_path="allenai/MolmoAct2-SO100_101",
    embodiment="so101",
    inference_action_mode="continuous",
)
```

### Sim vs hardware embodiment

The embodiment string is **not** interchangeable between MuJoCo and a physical
arm - they declare different `state_keys`/`action_keys`, so the wrong one yields
an empty `observation.state` and the policy fails with
`MolmoAct2 requires observation.state for discrete state prompting`:

| Context | Robot | Embodiment | `state_keys` |
| --- | --- | --- | --- |
| MuJoCo sim | `Robot("so101")` (`mode="sim"`, default) | `"so101"` | `["1", "2", "3", "4", "5", "6"]` (bare-numeric MJCF joints) |
| MuJoCo sim | `Robot("so100")` | `"so100"` | `["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]` |
| Hardware | `Robot("so101", mode="real", ...)` | `"so_real"` | `["shoulder_pan.pos", ..., "gripper.pos"]` (SOFollower driver) |

The `so101`/`so100` keys match the MuJoCo model's joint names; `so_real` (and
its `so101_real`/`so100_follower` aliases) match the lerobot SOFollower driver's
`<motor>.pos` keys reported over the serial bus. Pick the embodiment for the
robot you actually instantiate: `embodiment="so101"` for the sim examples,
`embodiment="so_real"` for the hardware example
(`examples/molmoact2_so101_pickplace.py`).

## Debugging "runs but does not move"

The provider surfaces two previously-silent failure modes as `WARNING` logs from
the `lerobot_local` logger:

1. **Action-dim mismatch.** Actions are mapped onto actuators by index. If the
   model emits fewer values than the embodiment declares actuators, the
   unmatched actuators are zero-filled (frozen). The policy now logs once, e.g.:

   ```
   lerobot_local: Policy action dim 4 < embodiment 'so101' actuator count 6:
   the 2 unmatched actuator(s) are zero-filled and will not move. ...
   ```

2. **Persistent near-zero actions.** If `max(abs(action)) < 1e-3` for 10
   consecutive steps (a starved obs/rename pipeline, an all-zero
   `observation.state`), the policy logs once:

   ```
   lerobot_local: Policy emitted near-zero actions (max abs < 0.001) for 10
   consecutive steps: the robot will not move. ...
   ```

These do not raise (a near-zero action can be legitimate mid-trajectory); they
point you at the embodiment / rename config. The warnings re-arm on
`policy.reset()` so each episode is evaluated independently.

### Repro / debug script

`examples/molmoact2_so101_debug.py` pushes a known degree-space action through
the `so101` mapping into MuJoCo and logs per-step joint deltas - no model
weights needed - then contrasts it with feeding raw degrees (which saturates):

```bash
MUJOCO_GL=egl python examples/molmoact2_so101_debug.py
# ... add --checkpoint allenai/MolmoAct2-SO100_101 to roll out the real policy
```

Expected: >5 deg cumulative motion on at least one joint within 20 steps.

## See also

- [LeRobot Local](lerobot-local.md) - install, caching, processor bridge, RTC
- [Policy providers](overview.md)
- [LeRobot project](https://github.com/huggingface/lerobot)
