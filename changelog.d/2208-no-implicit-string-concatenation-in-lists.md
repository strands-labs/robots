### Quality: refuse a list element built from two adjacent string literals

Python folds adjacent string literals into one value, so inside a list display
`["a" "b", "c"]` and `["a", "b", "c"]` differ by one character and mean different
things. A dropped comma does not fail - it silently merges two neighbouring
elements, so a parametrize case disappears from a suite that still reports green.

`tests/test_source_no_implicit_string_concatenation_in_lists.py` walks every
Python file under `strands_robots`, `tests`, `tests_integ`, `examples` and
`scripts` (1279 files) and refuses a list element whose extent holds two or more
string tokens. It is clean on the tree as it stands, so it needs no exemptions,
and it is clean on all 11 open pull-request heads.

`ruff` cannot express the rule: `ISC002` is silent because `allow-multiline`
defaults to true, and setting it false reports 4056 sites - almost all ordinary
wrapped prose. CodeQL does report it as `py/implicit-string-concatenation-in-list`,
but only after a push and only via a review thread the branch ruleset then
requires resolving, which is the situation `.github/codeql/codeql-config.yml`
answers by moving a capability to the merge-blocking local gate.
