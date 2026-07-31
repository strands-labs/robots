### Fixed: `success_fn="contact"` reads the payload the engine returns

`PolicyRunner.evaluate(success_fn="contact")` -- the sparse-success mode
`eval_policy`'s own docstring recommends -- indexed the `get_contacts` result as
if the result were the payload (`result["n_contacts"]`, `result["contacts"]`).
Backends return the agent-tool envelope, whose only top-level keys are `status`
and `content`, so both lookups missed and every episode scored a failure no
matter what the arm did: a cube resting on a plate under 4.9 N of solver force
evaluated to `success_rate: 0.0`. The bare mapping a test double returns did
satisfy that reader, so the divergence never surfaced in the suite.

The mode now shares the predicate DSL's `contact_any`, whose docstring already
claimed to match it, so there is one reader for both surfaces. `_extract_json`
reads a result with no `content` blocks as the payload itself, keeping a minimal
engine that returns a plain reading understood, and
`SimEngine.get_contacts` now documents where its records actually live.
