---
description: What randomize() actually samples - colors, lighting, physics, positions.
---

# Domain randomization

```python
sim.randomize(
    randomize_colors=True,      # resample object/floor RGB from color_range
    randomize_lighting=True,    # perturb directional + ambient light
    randomize_physics=False,    # mass (mass_range) + friction (friction_range) + damping
    randomize_positions=False,  # add position_noise (m) to every object position
    position_noise=0.02,
    color_range=(0.1, 1.0),
    friction_range=(0.5, 1.5),
    mass_range=(0.5, 2.0),
    seed=42,                    # deterministic sequence
)
```

**Unknown parameters are rejected.** `randomize()` and `set_obs_noise()` both
declare `**kwargs` to match the backend-agnostic `SimEngine` signature, so a
keyword they do not honor (`randomize_position` singular, `position_range`,
`joint_pos_stdev`) would otherwise be dropped and the call still reported as
applied. Instead they return `status=error` naming the unusable keys and the
valid set - a misspelled axis can never look like a successful randomization:

```python
sim.randomize(randomize_position=True)   # singular
# status=error: Unknown parameter(s) ['randomize_position'] for action 'randomize'.
#              Valid: ['color_range', 'friction_range', 'mass_range', 'position_noise',
#                      'randomize_colors', 'randomize_lighting', 'randomize_physics',
#                      'randomize_positions', 'seed']
```

**Destructive** - writes into MuJoCo model arrays. To restore: `load_scene(...)` or recreate the sim.

**Every axis survives `reset()`.** A reset restores the world's initial state,
and each axis writes that initial state rather than only the live state - the
colour, friction and mass axes write `model` arrays, and `randomize_positions`
writes `model.qpos0` (the pose a reset restores) alongside the live `data.qpos`.
This is what makes randomization reach a rollout at all: `run_policy` and
`eval_policy` reset before an episode's first step, so an axis a reset undid
would be gone before the policy ever saw it.

`randomize_positions` measures its offset from each object's **commanded** pose -
where `add_object` / `move_object` placed it - not from wherever physics has left
it. The commanded pose is a fixed reference, so calling `randomize()` once per
episode draws independent offsets that always stay inside `position_noise`
instead of compounding into a random walk that eventually leaves the workspace.

`randomize()` leaves the sim in a forwarded, render-ready state: the next `render()` / `get_observation()` reflects the perturbation immediately, with no manual `step()` in between. This matters for lighting in particular - the renderer reads light positions from the derived `data.light_xpos`, not `model.light_pos`, so a light-position jitter only reaches a render after a forward.

## Categories

| Flag | What changes | Range param |
|------|-------------|-------------|
| `randomize_colors` | Object + floor RGB (alpha fixed at 1.0) | `color_range` |
| `randomize_lighting` | Directional direction, intensity, ambient | - |
| `randomize_physics` | Per-object mass (mult), per-geom friction (scale), joint damping | `mass_range`, `friction_range` |
| `randomize_positions` | Dynamic-object position offsets (metres); static objects have no pose DOF and are skipped | `position_noise` |

Defaults: `colors=True`, `lighting=True`; `physics` and `positions` default `False`.

## Use in an eval loop

```python
for episode in range(N):
    sim.randomize(randomize_colors=True, randomize_physics=True,
                  randomize_positions=True, position_noise=0.03, seed=episode)
    # eval_policy has no randomize= kwarg - call sim.randomize() before each episode.
    # It resets at the start of every episode, which is why the perturbation has to
    # survive a reset; no explicit sim.reset() is needed here.
    result = sim.eval_policy(robot_name="so100", n_episodes=1, max_steps=300,
                             success_fn=my_fn)
```

## Targeted per-geom / per-body perturbation

`randomize()` perturbs the whole scene; `set_geom_properties` /
`set_body_properties` perturb one entity, which is what you want when only the
manipuland's friction or the table's height should change between episodes.

```python
sim.set_geom_properties(geom_name="crate", color=[0.8, 0.2, 0.2],   # RGB or RGBA
                        friction=[0.6, 0.01, 0.001],                # sliding, torsional, rolling
                        size=[0.2, 0.2, 0.05])                      # box: three half-extents
sim.set_body_properties(body_name="crate", mass=1.4)                # inertia scales with it
```

**Every vector must carry the exact component count its target defines.** There
is no meaningful value to invent for a component you omit, so a partial vector is
rejected instead of being mixed with the compiled one:

| Parameter | Accepted components |
|---|---|
| `color` | 3 (RGB, alpha set to 1.0) or 4 (RGBA) |
| `friction` | 3 (sliding, torsional, rolling) |
| `size` | whatever the geom's type defines: sphere 1, capsule/cylinder 2, box/ellipsoid/plane 3 |

`size` here is the geom's own `geom_size` - half-extents for a box - and not
`add_object`'s full extents, so the same vector means two different objects
depending on which call it is passed to. `add_object(size=[0.2, 0.2, 0.2])`
builds a 20 cm box; `set_geom_properties(size=[0.2, 0.2, 0.2])` resizes that box
to 40 cm. A capsule is `[radius, half-length]` here and
`[diameter, unused, height]` there.

```python
sim.set_geom_properties(geom_name="crate", size=[0.4])
# status=error: 'size' must have exactly 3 component(s) (box: three half-extents),
#               got 1: [0.4]. Pass every component - a partial 'size' cannot be
#               applied without inventing the missing values.
```

A mesh / height-field / SDF geom takes its extent from asset data and defines no
`geom_size` component, so `size` is refused for it (resize the asset instead).
Growing a size-defined primitive refreshes its broadphase and mid-phase collision
bounds, so other bodies collide with the new extent rather than passing through it.
It also re-derives the owning body's mass, center of mass and inertia tensor from
the new shape - those are integrated from the body's geoms at compile time and are
never recomputed by a step, so without this a resized body would collide as its new
shape while resisting rotation as the old one. The values are read from a compile of
the persisted spec, so a resize means the same thing whether or not another scene
mutation follows it. A body that declares its own `<inertial>` takes nothing from
geometry and is left alone.

## Sensor noise

`set_obs_noise` adds Gaussian measurement noise to observations so a policy is
not trained (or evaluated) on noise-free sensing - a cheap sim-to-real
robustness lever that is orthogonal to `randomize()` (which perturbs the world;
this perturbs the *sensor*).

```python
sim.set_obs_noise(
    joint_pos_std=0.01,      # rad, added to joint positions
    joint_vel_std=0.05,      # rad/s, added to per-joint velocities
    camera_jitter_px=2,      # max integer pixel shift per axis on rendered frames
    seed=0,                  # reproducible noise stream
)
```

Once configured, the noise is applied on every `get_observation`
(joint positions, the `<joint>.vel` entries, and camera frames),
`get_robot_state` (position + velocity), and `render` until reconfigured. Pass
all-zero std to disable; leaving it unconfigured (the default) is an exact
no-op, so existing observations and renders are unchanged. Floating-base
`base_quat` / `base_ang_vel` signals are left untouched (a quaternion would need
renormalization). Values must be finite and non-negative or the call returns
`status=error`.

## Newton backend

The Newton (GPU) backend mirrors both the `randomize` contract for the axes it
supports (colors, lighting, physics) and the `set_obs_noise` sensor-noise
contract, so an identical call behaves the same on either backend. See
[Newton backend](newton.md#domain-randomization-and-sensor-noise).

## See also

- [Simulation overview](overview.md)
- [World building](world-building.md)
- [Recording](../recording.md)
- [Real hardware](../hardware/robot-control.md)
