### Fixed: `[sim-newton]` resolves a MuJoCo pair the pinned Newton accepts

Installing the documented `[sim-newton]` extra produced a dependency set Newton
itself reports as violating its own requirements, on the first call that builds
a solver:

```
MuJoCo dependency version mismatch with Newton's declared requirements:
mujoco==3.12.0 (requires ~=3.11.0); mujoco-warp==3.12.0 (requires ~=3.11.0).
Reinstall Newton dependencies, for example `uv pip install -e ".[examples]"`.
```

The remedy that warning names is Newton's own extra, so a caller following it
steps outside this project's dependency management to satisfy this project's
extra. The simulation still runs -- `create_world`, `add_robot` and `step` all
succeed, and no physics difference was measured or is claimed -- but the
official install path produced a combination the solver's own preflight
declares unsupported.

Newton enforces that requirement at **runtime** from a declaration the
**resolver** never sees, which is why nothing bounded either package. Newton
declares it only under its own `[sim]` extra (`mujoco~=3.11.0; extra == "sim"`),
which `[sim-newton]` does not install, so uv is free to take both to 3.12.0 --
while `solver_mujoco._warn_if_mujoco_versions_mismatch` reads Newton's own
`METADATA` and matches `^mujoco(?=[<>=!~])([^;]+)`, truncating at the `;` and so
discarding the `extra == "sim"` marker. The pins are unreachable by the resolver
and binding at import, so declaring them in `[sim-newton]` is the only fix
available on this side.

The `newton` range moves with them rather than after them. The required series
is chosen per Newton **minor**, and the three the old `>=1.3.0,<2.0.0` admitted
disagree -- 1.3.0 requires `~=3.8.0`, 1.4.0 `~=3.10.0`, 1.5.0 and 1.5.1
`~=3.11.0` -- so no single MuJoCo pin satisfies that range, and capping MuJoCo
alone could not have closed this. For the same reason the cap is now a minor
one: `<2.0.0` would admit a Newton 1.6 free to require a different series,
reopening this with no edit to attribute it to, where a minor cap turns that
into an unresolved bump somebody has to look at. `mujoco` is named explicitly
rather than inherited because `[sim-mujoco]`'s `>=3.5.0,<4.0.0` admits 3.12.0,
so only the intersection narrows it; it stays inside that range, so no
`[sim-mujoco]`-only install is refused a version its own extra allows, and
`_MUJOCO_API_FLOOR` is untouched -- this floor is Newton's requirement, not an
API floor of ours.

The lockfile was mismatched on the same axis and by a different pair, which no
report had reached: it pinned `newton` 1.3.0 (requires `~=3.8.0`) beside
`mujoco` and `mujoco-warp` 3.10.0, so a locked install warned too, with
different numbers than the unlocked one. The relock moves `mujoco` and
`mujoco-warp` to 3.11.0, `newton` to 1.5.1 and `warp-lang` to 1.16.0, and
touches nothing else.

What is pinned is the **shape**, not Newton's table. Which series a Newton
release requires is published metadata rather than anything in this tree, so
copying it into a test would rot; instead the guard grades the three properties
that have to hold for any Newton the extra admits -- both MuJoCo distributions
on one shared series, a `newton` range admitting one minor, and that series
sitting inside `[sim-mujoco]`'s own range. Each of the three pins that
`[sim-newton]`'s pre-fix specifiers fail it, so a half-applied bump cannot pass:
capping only `mujoco-warp`, the direction the issue warned about, leaves the
`mujoco` half of the warning standing and is refused.

One guard elsewhere fired on this change and was removed rather than repaired,
which its own docstring prescribed. `check_lockfile_parity.py` maps
`(extra, name, extras)` to a *set* of specifiers because uv's own lock was once
observed recording `scipy` twice under `[all]`, and
`test_the_live_pair_carries_one_specifier_per_key` asserted that nothing in the
live files exercised that grouping -- "if this assertion ever fails, the
grouping has become live and this test should be deleted rather than repaired."
It has: `[sim-newton]` narrows `mujoco` while still entering `[sim-mujoco]`
through a self-reference, so uv records that one key at both `>=3.5.0,<4.0.0`
and `>=3.11.0,<3.12.0`. Both sides carry both, the grouping compares them as a
set, and the pair agrees in both directions -- the `scipy` shape stays pinned on
the planted pair beside it, so the observed-input case is still graded. The
script's own docstring claimed the live pair carried a unique specifier per key
and now names this case instead.
