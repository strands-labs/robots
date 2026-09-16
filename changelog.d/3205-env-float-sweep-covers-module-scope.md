### Fixed: the package-wide env-float sweep classifies the position an unusable value costs the most

`tests/test_env_float_knobs_resolve_to_a_finite_value.py` states a package-wide
rule and derived its population from every *function* that reads the environment
and coerces with `float()`. A knob resolved by a module-level statement was not a
function that failed that scan - it was invisible to it, so the sweep read as a
clean tree, which is the same blind spot its own docstring records for a
per-package root. That position is where an unusable value is worst, because the
coercion runs during import. Module-level statements are now classified too, keyed
`module::<module>`, and a guard is credited only to the scope that applies it: a
walk that descended into an enclosed `def` read a block as bounded by an
`isfinite` it never called.
