### Fixed: a `patch_scene_mjcf` batch keeps the scene's dynamic state, and a refused batch costs only that batch

`patch_scene_mjcf` was the one scene mutation that recompiled the spec itself
rather than through the shared `_recompile_preserving_state`, so it inherited
neither of that helper's two jobs.

It kept none of the state the helper carries. `apply_force` documents exactly
two things that end a latched wrench -- the next `apply_force` on that body, or
a `reset()` -- and `spec.recompile` returns `xfrc_applied` zeroed, so a patch
ended it too. A 1 kg crate held against gravity fell 470 mm on the first step
after a patch that reported success and only added an unrelated body. The
sibling suite's enumeration of every operation that rebuilds the model listed
five ops and omitted this sixth, which is why the gap stayed invisible.

Its refusal also landed past the batch's own rollback. The ops all apply to the
spec and MuJoCo then refuses the model they add up to; the raw recompile raised
from outside the `try` that restores the pre-patch snapshot, so the whole batch
stayed on the live spec. Every later mutation recompiles from that spec, so one
refused batch made the world permanently unusable -- an unrelated `add_object`
afterwards failed too, naming the leftover element from a batch the caller had
been told had failed. A zero `size` on `add_site` is finite, so it clears the
op-level domain and reaches the compiler, making this reachable through the
published action alone.

The call now goes through the helper, inside the rollback. Measured identically
on mujoco 3.5.0 and 3.12.0, restoring the snapshot is the whole rollback: the
compiled model and data objects are untouched and the cached XML is
byte-identical, because it is only re-synced after a successful install. The
refusal message is unchanged and still claims no op index, since every op applied.
