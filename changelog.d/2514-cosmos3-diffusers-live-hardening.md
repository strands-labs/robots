### Tests: the cosmos3 diffusers live test asserts finite, sanely-scaled actions from an embodiment-derived layout

`tests_integ/policies/cosmos3/test_diffusers_backend_live.py` asserted
`isinstance(v, float)` on the first step only, which passes on NaN and inf -
exactly what a bad checkpoint or dtype drift emits - and restated the DROID
unified action layout as a hard-coded key set that could silently desync from
`strands_robots/policies/cosmos3/embodiments.py`. The step keys and the chunk's
second dimension are now derived from `get_embodiment("droid")`
(`raw_action_layout` / `raw_action_dim`), every value in every step is asserted
finite, the whole raw chunk is asserted finite, of sane magnitude
(`|v| <= 10`: the quantile normalization maps q01/q99 to -1/+1 and nothing
clamps the sampler, so tail values legitimately exceed 1, while a mis-scaled
chunk misses by orders of magnitude), and not identically zero, and the MuJoCo tracking
test additionally requires every solved IK joint target to be finite - a solve
that diverges to NaN previously passed the shape checks.
