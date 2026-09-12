### Fixed: Isaac's `reset(env_ids=...)` is refused instead of silently resetting everything

`IsaacSimulation.reset` accepted an `env_ids` selection, documented it as
"Specific environment indices to reset. If None, reset all.", and then reset all
of them anyway. The parameter reached exactly one thing - the wording:

```python
sim.reset(env_ids=[0])
# {'status': 'success', 'content': [{'text': 'Partial reset complete for 1 envs.'}]}
# ... having called world.reset(), which re-initialized every environment
```

`world.reset()` is called unconditionally and takes no environment selection, so
there was never a partial reset to perform. Two things compound from that, and
neither raises.

**A vectorized caller is told the opposite of what happened.** Resetting the one
environment that terminated is the ordinary inner loop of a vectorized RL
rollout, and it teleported every *other* environment back to its initial state
mid-trajectory. The success message affirmed the caller's belief that it had not.
Nothing else in the result distinguishes the two.

**The recording path reasoned from the ignored parameter, which made it the
harmful half.** `reset()` is an episode boundary on every backend: it flushes an
open recording as its own episode before the teleport. Isaac skipped that flush
whenever `env_ids` was given, on stated grounds that were sound about a partial
reset - whether the one recorded robot's rollout ended is not knowable from
`env_ids` alone - and false about the whole-world reset that actually ran, where
it certainly ended. So the frames either side of a teleport were concatenated
into one open episode, and the dataset got a physically impossible transition
with no error, no warning, and a correct-looking frame count. A policy trained on
that episode learns a discontinuity as a dynamics sample.

`env_ids` is now refused, ahead of every check and every side effect, so a
refused call neither resets nor touches the recording - not to flush it and not
to discard it. `reset()` with no argument is unchanged and remains the boundary.

Refusing beats implementing here. A real per-environment reset means tracking
per-environment initial state and writing it back through the articulation view
per index - a feature, not the repair of a wrong answer - and MuJoCo, Newton and
the `SimEngine` ABC all declare `reset()` with no `env_ids` at all, so nothing
cross-backend was relying on it. Adding a parameter a sibling backend lacks is
allowed; advertising one that cannot be honoured is what this removes.

An empty selection is refused too. `env_ids=[]` names no environment rather than
being absent, so reading it by truthiness would have sent it to the whole-world
reset and reported that as the caller's own request - the subset-selector rule,
which is what the ignored-parameter bug looked like from the other end.

`docs/recording.md` claimed the partial reset as documented behaviour and now
records the refusal. `TestAPartialIsaacResetIsNotABoundary` is **replaced**
rather than renamed: it pinned the buffer staying open, and that conclusion does
not survive for another reason - the premise it rested on was the false one.
