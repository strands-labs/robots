### Tests: the seeded-replay measurement is taken off the process-global RNG

`tests/simulation/test_rollout_seed_is_applied_or_refused.py` measures the
applied half of the rollout seed contract with a policy whose actions depend on
the RNG state, and the RNG a rollout seed sets is the process-global one. The
policy drew straight from `random.random()`, so the comparison of two seeded
evals also assumed the rollout was that object's only reader for the length of
the rollout. It is not: any other thread in the interpreter that draws from it
between two of the policy's queries shifts every action after it, and the
comparison reports that as an unapplied seed. The test failed that way once on a
branch whose diff touched no RNG code, taking a required check with it.

The policy now reads the seeded stream through a private `random.Random` whose
state is copied from the global one in `reset` - the statement after
`set_eval_seed` on every rollout surface. The draws are still the ones the global
RNG would have yielded, so the seed is still what the comparison observes, but a
later reader of the global object cannot reach them. A new case pins it: one run
of a seeded pair has a `success_fn` that draws once from the global RNG mid
rollout, and the two runs must still replay identically while deriving identical
episode seeds.

The unseeded case is measured at the appliers instead of at the state they write.
`assert random.getstate() != expected` could not see the side effect it guarded -
reseeding from entropy and merely consuming draws both leave a state that differs
from the one before the call, so it passed either way. A reseed is a call, and
`set_eval_seed` is the only thing on this path that makes one, so the call is
what is counted.
