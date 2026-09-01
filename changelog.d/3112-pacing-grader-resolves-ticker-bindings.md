### Fixed: the pacing sweep grades every ticker, not every ticker spelled `Ticker`

`tests/test_mesh_pacing_ticker.py` requires every paced loop to acquire its
`mesh.pacing.Ticker` structurally, because the ticker owns a selector and a
socketpair and a hand-rolled `try/finally` release is a discipline each loop has
to remember. It found its population by matching the *local name* `Ticker`:

```python
isinstance(node.func, ast.Name) and node.func.id == "Ticker"
```

That reads one spelling, so a ticker imported under an alias, or reached as
`pacing.Ticker(...)`, was not merely ungraded -- it was invisible, and the sweep
reported a **clean tree** over one that held it. The distinction matters because
a rule nobody can see failing looks exactly like a rule nothing violates.

`strands_robots/simulation/policy_runner.py` was the live instance. It binds
`Ticker` at module scope for its `ticker: Ticker | None` annotation, so the
runtime import inside the rollout is necessarily aliased -- and the alias is
what makes the evasion a side effect of *correct* code, with nothing at the call
site for a reviewer to pause on. Measured on the tree before this change: all 14
name-visible constructions were `with`-acquired and the sweep passed, while a
sweep that resolves bindings finds a 15th that was not.

The sweep now follows the binding: names bound to `Ticker` by an `ImportFrom`
(alias included) and the attribute spelling through a `pacing` module alias, so
the population is 15 rather than 14. It also accepts an argument to
`enter_context` on a `with`-acquired `ExitStack` as structural release, which is
how the standard library expresses a resource acquired *conditionally* -- a
`with` item cannot, since the loop would have to construct a ticker to decide
not to use one. The stack's own acquisition is checked rather than assumed: a
hand-rolled `ExitStack()` closed in a `finally` is the same discipline one layer
up and does not launder a ticker through it.

`PolicyRunner.run` acquires its pacer on such a stack. Release was already
correct there -- a `finally` on every path, covered by the descriptor-leak
control -- so this closes a coverage hole rather than a leak: the exposure was
that a future edit to a 400-line `try` could drop the `close()` and the guard
that exists to catch exactly that would stay green. Rollout timing, descriptor
accounting and `fast_mode`'s skipped mesh import are unchanged.
