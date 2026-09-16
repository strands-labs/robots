### Fixed: `render_video.py` places the ball where the kick weights were trained to find it

`scene_ball.xml` declares its ball 0.3 m straight ahead; `ball_kick_left` and
`ball_kick_right` were trained with it 0.09 m ahead and 0.042 m to the side of
the kicking foot, in the robot's yaw frame - which `docs/policies/microduck.md`
already says, while the example's own recipe (`--onnx ball_kick_left.onnx
--scene scene_ball.xml`) rendered four seconds of the duck kicking air: over a
4 s rollout no robot geom came within 0.189 m of the ball's surface, nothing
touched it and it travelled 0 m.

The example now teleports the ball to the trained offset before the rollout,
the way Pollen's runtime does before every kick - in front of the foot
`--kick-foot left|right` names, inferred from the weight's file name when
omitted - and says where it put it. Same rollout after the change: contact,
ball travel 0.485 m. A scene without a ball is left alone, and the rollout says
the kick swings at nothing rather than rendering a duck that looks like it is
standing still; `--kick-foot` on such a scene is refused outright. Both point at
`--scene scene_ball.xml`.
