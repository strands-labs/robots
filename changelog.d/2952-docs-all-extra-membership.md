### Docs: the pages describing `[all]` name the extras it leaves opt-in

`[all]` is a convenience bundle rather than a union, and three pages told a
reader otherwise. `docs/getting-started/installation.md` enumerated it as five
extras -- `groot-service` plus `lerobot` plus `sim-mujoco` plus `mesh` plus
`mesh-iot` -- while the bundle installs nineteen of the thirty-one declared
extras. The enumeration was a strict subset, so a reader deciding whether
`[all]` covered the policy they wanted was told it did not for fourteen extras
it does install: `cosmos3`, `kimodo`, `protomotions`, `wbc`, `openpi`, `pi`,
`smolvla`, `vera` and six more. Five lines below that table the same bundle was
called `# everything`.

The two remaining claims were wrong in the other direction. `docs/architecture.md`
gave the bundle's whole description as "union", and `docs/index.md`'s quickstart
comment as "every policy", while ten capability extras stay opt-in --
`sim-isaac`, `sim-newton`, `sim-gs`, `curobo`, `ros2`,
`microduck`, `vera-sim` and the three `cosmos3-*` service and simulation extras.
A reader who installed `[all]` expecting a union got a `ModuleNotFoundError` from
the extra they actually needed, which is exactly the failure the extras table
exists to prevent.

The three cells now state the count, and installation.md names each excluded
extra, because that is the actionable half: a reader wants to know what they
must still add. A guard derives both the count and the excluded set from
`[project.optional-dependencies]`, walking the transitive closure rather than
the literal list -- an extra can name siblings, and eight of the nineteen are
reached that way -- so a new extra is graded the hour it lands. Each failure
message prints the text the page should carry, so the count maintains itself
rather than rotting again. The completeness rule drops negated forms before it
grades a line, because a corrected page says the bundle is *not* a union and a
bare substring test would report the very wording that fixes this.

Deliberately out of scope: *why* a given extra sits outside `[all]`. The
excluded set has no single rule -- `sim-isaac` and `sim-gs` need a GPU and say
so, while `cosmos3-service` is two pure-Python packages -- so documenting a
reason per extra is a maintainer's call rather than a derivation.
