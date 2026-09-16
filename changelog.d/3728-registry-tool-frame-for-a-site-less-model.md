### Registry `tool_frame` - a tool point for a model that ships no site

`move_to` drives the frame end-effector discovery finds: a tool-point site
first, else a hand/wrist body. The SO-100's Menagerie model (`trs_so_arm100`)
has zero sites, so discovery settled on `Wrist_Pitch_Roll` - about 16 cm short
of the jaw tips - and the README's first robot missed every low target it was
sent to (0/3 in the v0.5.2 devx replay) while the SO-101, whose model ships a
`gripper` site, landed the same prompt 4/4. A registry entry may now declare
the tool point the model lacks - `"tool_frame": {"body": "Fixed_Jaw", "pos":
[0.0, -0.0995, 0.001], "site": "tcp"}` - and the MuJoCo backend adds that site
to the robot before the attach, so discovery, `get_robot_state`'s
`end_effector` line and `move_to` follow it (`so100` now lands those three
targets 3/3 through `site 'arm/tcp'`). The shipped `so100` entry carries one; a
model with its own site needs none. A malformed block, or one naming a body
the model lacks, refuses `add_robot` with the reason instead of falling back
to the wrist.
