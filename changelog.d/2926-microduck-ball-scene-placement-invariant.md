### Docs: the Microduck ball scene carries the ball, not the kick geometry

`docs/policies/microduck.md` points the two ball-kick weights at `scene_ball.xml`
in its Skill scenes table, and the guard beside it grades the ball's presence: the
row names a scene, and that scene carries a body named `ball`. Neither says where
the ball sits, and the position is the half that decides whether a kick connects.

The scene declares the ball 0.3 m straight ahead. Pollen's training reset placed it
0.09 m ahead and 0.042 m to the side of the kicking foot, in the robot's yaw frame.
Driven from the shipped position through `MicroduckPolicy`, `ball_kick_left`
completes and reports success while no robot body ever touches the ball: across a
four-second rollout at 50 Hz the closest any robot geom comes to the ball centre is
0.109 m, against a 0.035 m radius. The ball still travels 0.474 m forward on its
own - its geom sets a deliberately low rolling resistance - which is why the miss
reads as a weak kick rather than as a miss.

The page now says that naming the scene is necessary and not sufficient, and states
both numbers so a reader can close the gap. It also records that the joint is
`ball_free` in the file and that `add_robot(name=...)` prefixes every joint with the
name the caller passed, so a caller resolves the name rather than assuming either
spelling. Two guard docstrings that claimed the trained position while their cells
graded only presence are corrected to say what they check.
