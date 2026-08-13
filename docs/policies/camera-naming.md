---
description: Camera naming for VLA policies in sim - the model-card to embodiment obs_rename translation table, the pre-flight check, and the obs_rename_override escape hatch.
---

# Camera naming (model card vs embodiment)

VLA checkpoints declare image input features (e.g.
`observation.images.image`, `observation.images.wrist_image`). The
[LeRobot Local](lerobot-local.md) provider routes a robot/sim camera onto each
of those features using the embodiment's `obs_rename` map
(`{runtime_camera_name: "observation.images.*"}`).

The trap: a model's HuggingFace card often describes its cameras in
human terms ("top + side", "third-person + wrist") that do NOT match the
runtime source key the embodiment expects. If you name your sim cameras after
the card ("realsense_top"), the rename step never produces the model's image
features, and inference fails with a confusing
`image_keys missing from observation`.

`run_policy` / `eval_policy` catch this with a cheap pre-flight check BEFORE any
model weights download, returning a `status=error` that names the expected
source camera keys and how to fix it. See
[the pre-flight check](lerobot-local.md#embodiment-obs_rename-and-the-pre-flight-check).

## Translation table

Use the **embodiment source key** column when naming sim cameras
(`sim.add_camera(name=...)`). The model-feature column is what the checkpoint
declares internally.

| Model | Embodiment | Camera source key (`add_camera` name) | Model image feature |
| --- | --- | --- | --- |
| `allenai/MolmoAct2-SO100_101` | `so100` / `so101` | `front` | `observation.images.image` |
| `allenai/MolmoAct2-SO100_101` | `so100` / `so101` | `wrist` | `observation.images.wrist_image` |
| SmolVLA (single-cam SO arm) | `so100` / `so101` | `front` | `observation.images.image` |
| pi0 / pi0-FAST (Aloha) | `aloha` | `cam_high` | `observation.images.cam_high` |
| pi0 / pi0-FAST (Aloha) | `aloha` | `cam_left_wrist` | `observation.images.cam_left_wrist` |
| pi0 / pi0-FAST (Aloha) | `aloha` | `cam_right_wrist` | `observation.images.cam_right_wrist` |

The authoritative source for every embodiment is its entry in
`strands_robots/policies/lerobot_local/embodiments.json` (`obs_rename`). When in
doubt, read the `obs_rename` keys for your embodiment - those are exactly the
camera names the runtime observation must contain.

## Two ways to satisfy the check

1. **Rename your cameras** to the expected source keys:

   ```python
   sim.add_camera(name="front", position=[0.22, 0.025, 0.6], target=[0.22, 0.025, 0])
   sim.add_camera(name="wrist", parent_body="so101/gripper")
   ```

   `parent_body` mounts the camera ON a body so the wrist view rides with the
   arm. It is supported on the **mujoco** and **newton** backends; the isaac
   backend refuses it with an error naming this alternative, because it parents
   camera prims to the stage camera scope rather than to an articulation link.
   On isaac, place the wrist camera in world coordinates (omit `parent_body`)
   and accept that it does not track the arm, or run the wrist-stream policy on
   mujoco/newton.

2. **Keep your names and override** - `obs_rename_override` merges OVER the
   embodiment's `obs_rename`, so a custom camera name routes onto the model's
   image feature without renaming:

   ```python
   sim.run_policy(
       robot_name="so101",
       policy_provider="lerobot_local",
       policy_config={
           "pretrained_name_or_path": "allenai/MolmoAct2-SO100_101",
           "embodiment": "so101",
           "obs_rename_override": {
               "realsense_top": "observation.images.image",
               "realsense_side": "observation.images.wrist_image",
           },
       },
   )
   ```

## Bare camera keys are canonicalized on the declarative path

On the declarative `embodiment` path the camera rename
(`front` -> `observation.images.front`) happens INSIDE the preprocessor
pipeline, so the observation still carries the bare source key when the
frame is normalized to channel-first `float32`. Frames keyed by a bare
name whose embodiment `obs_rename` target is an image feature are now
recognized and canonicalized, so a single-camera checkpoint (e.g. an ACT
policy declaring only `observation.images.front`) driven via an embodiment
is handled the same as the legacy (no-embodiment) path. You do not need to
name your camera with an `image` substring for this to work.

## Bare declared image keys bind by exact name

Most ACT/diffusion checkpoints declare image features in the prefixed form
(`observation.images.<cam>`). Some VLAs - notably MolmoAct2 - instead declare
BARE image keys (`base`, `wrist`) in their `input_features`. When you route sim
cameras onto such a policy by name (via `image_keys` or simply by naming the
sim cameras to match), a camera whose name equals a bare declared key binds to
it BY NAME, exactly like the prefixed form does. Positional fallback (with its
`Camera '<name>' does not match any declared policy image key` warning) only
kicks in for cameras whose names match neither a prefixed nor a bare declared
key.

This matters when the scene carries an extra camera the policy does not
consume (for example the MuJoCo `default` free camera): the named views still
bind to their own slots and the extra camera is dropped, instead of the extra
camera positionally displacing a real view. Set `strict_keys=True` to turn an
unresolved camera name into a hard error rather than a positional guess.

## See also

- [LeRobot Local](lerobot-local.md) - camera routing, the pre-flight check, `obs_rename_override`.
- [MolmoAct2](molmoact2.md) - the SO-100/101 action/observation contract.
