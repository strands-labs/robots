### Fixed: a NaN delta-EEF channel no longer reaches PhysX as `nan` joint targets

`IsaacDeltaEEFController` coerced each GR00T action channel through a helper that
logged a WARNING and substituted `0.0` (`0.5` for the gripper) for anything it
could not read. Two separate problems, and only one of them was a posture choice.

**A NaN channel was never caught at all.** `float("nan")` coerces
successfully, so it walked straight past the fallback, and the finiteness guard in
`_solve_arm_targets` covers only the *injected callables*. Measured:

```python
controller.compute_joint_targets({"x": float("nan")})
# {'j1': nan, 'j2': nan}
```

Non-finite targets for **every** arm joint - not a held axis - handed to the
articulation's PD targets. PhysX reports those from a *later* step as
`Illegal BroadPhaseUpdateData - non-finite bounds`, attributed to whatever is
running by then rather than to the action that produced them. This is the exact
failure mode `set_joint_positions` already refuses `nan` to prevent, and the
principle is already stated in this same file - `_solve_arm_targets` raises for a
non-finite Jacobian so that "a broken solve must surface, never degrade into a
silent zero-motion step" (#1812). It was applied to one input and not the other.

It is also the case the pre-existing test for this branch could not have caught
while asserting what it asserts: it requires `all(np.isfinite(...))` of the
returned targets, and never passed a `nan` in. `NaN` now raises `ValueError` in
**both** postures, because there is no reading of it that holds an axis - and it is
pinned on *every* channel, not just one, so a later change that special-cases the
gripper coercion cannot slip past a class that tested only `x`.

**`±inf` is accepted, and that correction came out of adversarial review.** An
earlier version of this fix refused every non-finite value, on the stated grounds
that a non-finite delta "solves to non-finite targets for every arm joint". That is
true of `nan` and **false of `±inf`**: `np.clip(±inf, -1, 1)` is `±1.0`, so an
infinite channel saturates to the maximum per-step delta - exactly the
clip-then-scale contract this controller documents. Measured against clean main:

```python
compute_joint_targets({"x": float("inf")})   # {'j1': 0.0499, 'j2': 0.0}  - finite
compute_joint_targets({"x": float("nan")})   # {'j1': nan,    'j2': nan}
```

Refusing `inf` would have converted a correct saturation into a hard failure for
any policy head that saturates. `inf` and `1.0` are pinned as producing identical
targets, which is what makes accepting it correct rather than merely lenient.

**An unreadable channel is now refused by default, with the old behaviour behind
`strict=False`.** The previous posture was deliberate and documented - *"a
malformed channel degrades that one axis, it does not abort the step"* - so it is
preserved rather than deleted, as `strict=False`. What changed is which posture is
the default, on two grounds. It returns a zero-valued action on failure. And the
substitution is not the "degrade one axis" it reads as for the gripper: `0.5`
becomes `-sign(2*0.5-1)` = `-0.0`, which the `command != 0.0` guard drops
entirely, so a commanded grasp or release silently did nothing and the fingers
held their previous target. Measured, `{"gripper": "abc"}` returned `{}`.

`strict` is checked with `boolean_flag_error`, not read by truthiness, so
`strict="no"` cannot select the permissive posture while reading as the strict one.

An **absent** channel is untouched and is not an error: absence means "hold this
axis" and is the documented default. Keeping that split explicit at the call site
is what lets the coercion refuse at all - the zero for a missing axis is a
documented default, and the zero that stood in for an unreadable one was a
fabricated command. The refusal names the offending channel, so a caller fixes the
axis rather than searching the action dict.

`TestToScalarFallback` is **replaced** by three classes - refused by default, the
degraded posture under the flag, and non-finite refused in both - rather than
deleted. Its conclusion still holds; it holds under a flag now, and its stated
invariant is what the `nan` case violated.
