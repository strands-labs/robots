### Fixed

- **docs/policies/wbc**: the intro no longer calls `CompositePolicy` future and out of scope.
  Layering an upper-body manipulation policy on WBC locomotion has been supported since
  `CompositePolicy` landed, and the same page documents it - the routing rules, a runnable
  `CompositePolicy(...)` snippet, `examples/wbc/wbc_g1_composite.py`, and a rollout of a G1
  walking under WBC while the upper body moves its arms - 260 lines below the sentence saying
  it was not possible. A reader who stopped at the intro walked away believing the page's own
  worked example described something unavailable. The intro now states that layering is
  supported and links to that section. `tests/test_docs_no_stale_future_symbol_claims.py`
  grades every "future" / "out of scope" claim in `docs/` against the symbols the package
  defines, so a page cannot keep calling a shipped symbol a plan.
