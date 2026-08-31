### Added: the Microduck page names the stance every weight was trained in

`docs/policies/microduck.md` carries the skill-to-scene table, and a weight and
its stance are a pair in the same way a weight and its scene are.
`changelog.d/2900-microduck-skill-scenes.md` closed the scene half; the stance
half was still undocumented, and the page's own example spawns without it.

Every shipped Pollen weight bakes the stance it was trained in into its ONNX
metadata as `default_joint_pos`, and all nine declare the pairwise-identical
fourteen values. `MicroduckPolicy` reads them into `default_pose` and decodes
every action relative to that origin - the module's own docstring states it as
`motor_target = default_pose + raw_action * action_scale` - so the stance is not
advice, it is what the network's output is measured from. The same values ship as
`strands_robots.policies.microduck.MICRODUCK_DEFAULT_POSE`.

The asset ships the stance too, and `add_robot(keyframe=...)` seats a robot
there:

    scene.xml          keyframes=['STAND']  spawn at STAND -> 4e-05 rad from MICRODUCK_DEFAULT_POSE
    scene_rollers.xml  keyframes=['STAND']  spawn without   -> 0.458 rad away, reported success
    scene_ball.xml     keyframes=[]         keyframe="STAND" refused, naming the keyframes declared

A keyframe spawn is sticky: `reset()` restores the pose and the actuator command
that holds it, so every episode of a `run_policy` plus `reset` loop begins from
the same stance. Spawning without `keyframe` is not an error - the robot starts
at the zero configuration, 0.458 rad from the trained stance at the widest joint,
while the policy still decodes relative to the stance it expects, so the first
inference reads a pose no shipped weight was trained on.

Two details the section records because a caller meets both. The live keyframe is
named `STAND`: the asset's own comment calls the current values "STAND2" because
they supersede an earlier `STAND` commented out beside them, so `"STAND2"` is
refused. And `scene_ball.xml` declares no keyframe at all, so the route is
unavailable on the one scene a ball kick needs.

The page names the constant rather than repeating the numbers, because the asset
has revised this pose once already: the superseded `STAND` sits 0.066 rad from
the current one at the hip and flips the sign of `head_pitch`. A cell now asserts
that the shipped keyframe and the exported constant still agree to within 1e-3,
so revising either fails that cell rather than leaving the page describing a pose
the asset no longer declares. No library code changes.
