### Fixed: the documented Policy ABC contract lists every public member of the class

`docs/policies/custom-policies.md` is the page a subclass author reads before writing a
provider, and its "ABC contract" table named 7 of the 15 public members of `Policy`. The eight
missing ones were `preflight`, `children`, `execution_horizon`, `is_chunk_emitting`,
`control_frequency`, `set_control_frequency`, `rtc_observed_delay_steps` and
`set_rtc_observed_delay` - so the fail-fast validation hook, the re-query interval that is the
single source of truth for chunk consumption, the wrapper-delegation surface and the whole
Real-Time Chunking handshake were invisible on the page whose job is to enumerate the
contract. `control_frequency` and `rtc_observed_delay_steps` carry no docstring either, so
they were undiscoverable from the source as well.

The table now lists all fifteen with their default, and the members the runtime supplies to a
policy rather than the reverse are called out as such. The expectation is derived from the
class in `tests/policies/test_documented_abc_contract_matches_the_policy_class.py`, as a
biconditional in both directions plus an abstract-column check, so a member added to `Policy`
joins the requirement without an edit to the test and a row naming a member the class no
longer has is refused too.
