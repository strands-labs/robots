### Documentation

- **docs/data/annotation.md**: document which camera `lerobot-annotate` derives
  its steerable labels from. The `plan` and `interjections` modules read only the
  dataset's first video key, and `start_recording` writes video features in the
  caller's `cameras=` order, so that list decides the view. Records the
  gripper-mounted hazard - a camera parented to the gripper renders a carried
  object as image-static and a stationary object as sweeping out of frame - and
  names the `--vlm.camera_key` escape hatch.
