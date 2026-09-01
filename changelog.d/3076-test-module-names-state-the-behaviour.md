### Tests: a test module name states the behaviour, not the tracker item that birthed it

`tests/drivers/test_g1_send_action_success_is_the_acceptance_criterion_for_harness_361.py`
named the item it was filed against rather than what it grades, so its failure
line told a reader nothing about the broken behaviour, and the coordinate it
carried - `harness#361` - is an unowned slug that resolves to nothing, the same
shape `tests/test_source_strings_resolve_their_issue_references.py` already
refuses in an operator-facing string.

The file is now `test_g1_send_action_succeeds_on_a_healthy_wired_driver.py`,
which is what it verifies, and its docstrings state the two contracts they
always carried - the publisher must be populated, and the wire must be exercised
rather than the status alone believed - without the review history around them.

`tests/test_test_module_names_do_not_spell_a_tracker_coordinate.py` keeps the
shape out of the tree. A module name spells a coordinate when a word is followed
by a standalone run of digits (`..._for_harness_361` is `harness#361` written
for a filesystem), and whether that coordinate resolves is decided by the
predicate that already grades the string surface, so the two guards cannot drift
apart. The rule needs no exemption list: a vendor model or version number is
spelled attached to the word it qualifies (`so101`, `gr00t`, `n17`, `cosmos3`,
`go2`, `ipv6`, `moveit2`, `ros2`, `molmoact2`), so all eleven digit-bearing
tokens across the 1,500 test modules pass. The grader joins the roster in
`scripts/check_whole_tree_graders.py`, since its input is the tree rather than
the file under change.
