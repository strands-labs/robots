### Quality: pin the `create_world` timestep domain on every backend, not just MuJoCo

`tests/simulation/test_timestep_domain_across_surfaces.py` states that all three
backends' `create_world` route through the one shared timestep domain, and each
backend validates the *effective* dt so an unusable engine default is reported
under its own knob (`default_timestep` on MuJoCo and Newton, `physics_dt` on
Isaac). That is six cells; only the two MuJoCo ones were driven, by a module that
is `importorskip("mujoco")`-gated and MuJoCo-only by construction. The structural
scan in the same file covered `set_timestep` alone, so the claim was
behaviourally unmeasured for two backends and structurally unenforced for all
three.

Both builders are reachable with no Newton, Warp, Isaac Sim or GL: each
validates before it takes its lock. This drives the four undriven cells, pins
that the refusal names the knob the value came from, and pins the three-way
verdict against the MuJoCo builder that was already covered. The scan is widened
from `set_timestep` to every timestep-installing surface, and its non-vacuity
test now names the five-surface matrix the module asserts in prose.

Also pins why the effective-dt check is load-bearing rather than defensive:
`NewtonSimEngine.__init__` stores `default_timestep` unvalidated, and
`IsaacConfig.__post_init__` tests `physics_dt <= 0` - a bare comparison, so
`IsaacConfig(physics_dt=float("nan"))` constructs and `create_world()` is the
only thing between it and a world built on a dt no integrator can advance by.
Tests only; no library behaviour changes.
