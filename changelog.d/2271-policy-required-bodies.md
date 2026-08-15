### Feature: a policy can declare the body poses it needs

A whole-body motion-mimic tracker is conditioned on the world orientation of one
anchor link. On a Unitree G1 that link is `torso_link`, and the observation
schema could not supply it: per-joint scalars carry no world frame, and the
floating-base signals (`base_pos` / `base_quat` / `base_lin_vel` /
`base_ang_vel`) describe the pelvis. The waist's three joints separate the two,
so a tracker written against `base_quat` reads the wrong frame whenever the
waist is not neutral -- measured on a G1 sweeping its waist, they diverge by up
to 42 degrees.

`get_body_state` already answered this on MuJoCo and Isaac, but it is a
`SimEngine` method and a `Policy` is never handed the engine, so the value was
reachable from everywhere except the place that needed it.

`Policy.required_bodies` is the declaration, on the same "policy declares,
runtime supplies" contract as `requires_images`: `PolicyRunner` resolves the
names once before the rollout and merges each body's pose into every observation
as `body.<name>.pos` / `.quat` / `.lin_vel` / `.ang_vel`, spelled like the
`base_*` signals already in the schema so a policy reads one convention for the
base and for any other link.

Resolution happens up front, so a body the scene does not contain raises there,
carrying the backend's available-names message, rather than going absent from
every observation and being read as a zero pose for the whole episode. A bare
`str` is refused rather than iterated into one entry per character.

The nine `get_observation` reads in `simulation/policy_runner.py` now go through
one `_observe` helper, so the contract holds identically on the synchronous,
chunked and async-RTC paths rather than only whichever one was patched.

Declaring nothing -- every policy that exists today -- is unchanged: no extra
backend call, no new observation keys, and no new dataset columns, since the
recording schema is an explicit allowlist of scalar joints plus expanded base
components.
