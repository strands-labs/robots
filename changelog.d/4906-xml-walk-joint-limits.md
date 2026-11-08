### Fixed: without MuJoCo, joint limits are read from the model the way MuJoCo reads them

When the `mujoco` extra is not installed (for example a `feetech`-only install driving a real arm), joint limits come
from a walk of the MJCF file. That walk now matches MuJoCo's compiler: hinge ranges are degrees unless
`<compiler angle="radian">` says otherwise, ranges set in `<default>` classes apply through `class=` and
`childclass=`, `limited="false"` and `autolimits="false"` are honoured, `<include>` files are read where they appear,
and joints declared inside `<default>` are no longer counted as joints. Before, a degree model got limits about 57
times too wide and joints limited only through a default class (all of `so100`) got no limit at all, so the safety
chain's range check passed targets far outside the real range. Every MJCF model in the registry now gives the same
joints, actuators and limits with and without MuJoCo.
