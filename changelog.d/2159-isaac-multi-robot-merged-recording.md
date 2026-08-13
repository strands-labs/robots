### Added: Isaac `run_multi_policy` records merged multi-robot frames (one frame per timestep)

The Isaac synchronized multi-robot loop now feeds an active dataset recording
session instead of refusing to run inside one: each loop iteration emits
exactly ONE `add_frame` carrying every driven robot's namespaced state/action
columns (`alice__shoulder_pan` ...) plus all camera images (scene-global,
read once per step), mirroring the MuJoCo merged-frame semantics - so a
2-robot Isaac dataset has both arms co-observed in every frame, usable for
bimanual / multi-agent policy training. LeRobot stores one task per frame, so
with distinct per-robot instructions the first robot's instruction is recorded
(the shared normalizer's warning covers the limitation). The loop also gains
the shared recording-rate guard (`control_frequency` must match the open
dataset's fps) and MuJoCo's partial-episode discard: a mid-loop failure drops
the dangling frames so the next episode starts at frame 0.
