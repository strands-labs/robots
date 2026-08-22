### Fixed

- **policies**: a config dataclass's `Attributes:` block now names every field it
  declares. `EmbodimentMap` documented five of ten, omitting the unit-conversion
  group (`state_units`, `action_units`, `gripper_index`, `gripper_joint_range`,
  `joint_mids`), so an embodiment written from that block converted nothing and a
  degrees-trained SO-arm checkpoint's values reached the sim unconverted;
  `PackStateProcessorStep`, which receives four of them, and `ProtoMotionsConfig`
  had the same gap. Graded for every dataclass with such a block by
  `tests/test_attributes_docstring_completeness.py`, the attribute-list counterpart
  to the existing `Args:` completeness guard.
