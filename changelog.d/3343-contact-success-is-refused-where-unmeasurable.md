### Fixed: a contact success criterion is refused, not silently scored 0%, on a backend without a contact query

`eval_policy(success_fn="contact")` resolves to the predicate DSL's
`contact_any`, whose never-raise contract turned a backend's
`NotImplementedError` into `False` - every tick, logged at DEBUG. On a backend
whose `get_contacts` is still the `SimEngine` raising stub (Isaac, Newton), the
full evaluation therefore ran to completion - GPU-hours on Isaac - and reported
`success_rate: 0.0` with `success_measured: True`: a wrong answer shaped exactly
like a policy that failed every episode, with nothing anywhere saying success was
never measurable. `evaluate`'s own docstring names `success_fn="contact"` as the
way to measure real task success, which is what sent callers into it.

Three layers, one change:

* **The resolver refuses up front.** `_resolve_success_fn("contact")` now raises
  `ValueError` - which `evaluate` already returns as its structured error
  envelope - when the backend's `get_contacts` is the base stub, *before* any
  rollout is spent. The message names the backend, the wrong answer it prevents,
  and both remedies (an observation-based callable, or MuJoCo). The test is
  structural (did the subclass override the stub?) rather than a probe call,
  because a real backend's `get_contacts` can fail for world-lifecycle reasons
  that say nothing about the capability.
* **The predicates say why they answer False.** The contact read now has one
  owner (`_read_contacts`, shared by `contact_any` and `contact_between`), which
  treats `NotImplementedError` as the permanent fact it is - a WARNING once per
  backend class, naming the remedy - instead of a per-tick DEBUG line
  indistinguishable from a transient read failure. The DSL stays never-raise:
  it is reachable directly through benchmark specs, where refusing is not this
  layer's call to make. Transient failures keep their DEBUG-level degraded mode.
* **`describe()` stops advertising raising stubs.** The base advertisement of
  `load_scene` / `randomize` / `set_obs_noise` / `get_contacts` is now
  conditional on the subclass overriding the stub. The Isaac backend re-published
  all of them for months while their calls raised `NotImplementedError`; the
  Newton backend only avoided the false advertisement by building its
  `describe()` from scratch - a per-backend workaround for a base-class defect.
  Gated structurally rather than by a hand-kept list, so a backend that gains one
  of these starts advertising it with no second edit. MuJoCo, which implements
  all four, advertises all four exactly as before.

The pre-existing premise test that asserted the unconditional advertisement is
replaced rather than deleted: the discoverability it pinned still holds, on the
backends that implement the methods, and it now pins the conditional rule in
both directions.
