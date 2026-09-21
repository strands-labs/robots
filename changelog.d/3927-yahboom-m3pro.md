### Added: `yahboom_m3pro` - Yahboom ROSMASTER M3 Pro mobile manipulator (sim)

`Robot("yahboom_m3pro")` builds the Yahboom ROSMASTER M3 Pro - a mecanum
chassis carrying the DOFBOT-Pro arm (five bus-servo joints plus a gripper), an
Orbbec camera on the wrist and a second camera on the chassis - in MuJoCo. The
asset auto-downloads from `dimwael/yahboom_m3pro_description`, an MJCF
generated from the vendor SolidWorks URDF: five position-servo arm hinges with
the URDF limits, a two-crank gripper (`gripper` drives `rlink1`, an equality
mirrors `llink1`; `closed="low"`), a kinematic planar base (`base_x`, `base_y`
world-frame slides and `base_yaw`, each with a velocity actuator; wheels are
visual only), a `tcp` site at the jaw tips, a `home` keyframe at the vendor
grasp-init pose, and both cameras declaring the measured RGB intrinsics so
`get_camera_params` returns the physical camera's `K`. The model declares
`implicitfast` / elliptic cones / `impratio=10`, which `add_robot` adopts.

Sim only: the entry has no `hardware` block, so `mode="real"` is refused at
the factory until a driver for Yahboom's expansion-board servo bus exists.
`docs/robots/mobile.md` states the three modelling simplifications (kinematic
base, rotating jaws, nominal wrist-camera pose) and the description repo's
`DESIGN.md` carries the open questions that need the physical robot.
