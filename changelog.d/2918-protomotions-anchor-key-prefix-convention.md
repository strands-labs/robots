### Fixed: the ProtoMotions anchor ladder reads its `observation.`-prefixed key spelling

Every observation-key fallback ladder in `strands_robots` pairs a bare key with
`observation.<that key>`.  `ProtoMotionsPolicy._extract_root_local_ang_vel`
reads `("root_ang_vel_local", "observation.root_ang_vel_local")`, and both of
`WBCPolicy`'s ladders read `("base_ang_vel", "observation.base_ang_vel")` and
`("base_quat", "observation.base_quat")`.  The anchor-rotation ladder on that
same tracker was the exception: it read `("anchor_rot_xyzw",
"observation.anchor_rot")` - prefixed, but with the `_xyzw` suffix dropped - so
the spelling the convention implies was absent.

Measured on one observation dict whose two keys are both written to the
convention, `observation.root_ang_vel_local` resolved and
`observation.anchor_rot_xyzw` raised `KeyError`.  One dict, two ladders on one
class, one convention, two answers, and the refusal that fired named the
runtime's `body.<anchor>.quat` key, the `anchor_rot_xyzw=` kwarg and
`body_rot_xyzw` - so a caller whose prefixed key had just been refused was not
told that a prefixed spelling was read at all, nor which one.

The ladder now reads `("anchor_rot_xyzw", "observation.anchor_rot_xyzw",
"observation.anchor_rot")` and the refusal names the prefixed form.  The change
is additive: the suffix-less `observation.anchor_rot` still resolves, because it
is accepted today, and the runtime's declared-body rung is still checked ahead
of the whole ladder, so a simulation rollout resolves exactly the rotation it
always did.  All three anchor spellings now return the same `xyzw` quaternion -
this rung does not reorder components, only the `body.<anchor>.quat` rung
converts from MuJoCo's `wxyz`.

The convention itself is now derived from the package rather than restated in a
list, so a ladder added later is held to it the hour it lands.  The derivation
reads a tuple or list of string literals that is *consumed as keys* - the
iterable of a `for`, or a call argument - and that mixes a bare key with a
prefixed one; the consumption site is what keeps a module-level list of default
camera feature names and a membership test over dataset columns out of scope,
neither of which is a fallback ladder.
