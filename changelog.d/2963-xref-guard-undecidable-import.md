### Fixed: one absent extra no longer ends the docstring cross-reference sweep

`tests/test_docstring_xref_roles_resolve.py::test_qualified_strands_robots_xref_roles_resolve`
separates the two causes of a failed import that need opposite verdicts -- a
`strands_robots` path that names nothing (the rot it grades) versus a
third-party module this environment does not have (a fact about the
environment) -- and it did so by reading `exc.name`. That field is populated
only on an exception the interpreter raises itself. It is `None` on one that
code *constructs*, which is what 28 raise sites in the package do, including
both raises in `strands_robots.utils.require_optional` -- the mechanism
`AGENTS.md` convention 7 makes mandatory for every optional dependency. So a
third-party absence took the branch reserved for package rot, and the member
walk answers that branch with `raise`.

Measured on a base install without `[all]`: the sweep aborted at target 676 of
795 on `strands_robots.tools.lerobot_camera`, leaving **120 targets ungraded**
and reporting one `ImportError` instead of a verdict. A contributor without the
extras saw a red guard they did not break, could not distinguish it from a dead
pointer they did introduce, and got no grading of the roles they had touched.
The same install now grades all 795 (759 resolved, 0 dead, 36 undecidable).

The classification is now made on the chained exception the raise site was
handling rather than on the surface exception alone. `raise ... from None`
clears `__cause__` and suppresses the *rendering* of `__context__`; it does not
clear the attribute, so `require_optional` is reached by the same walk and no
raise site needs to change. An `ImportError` carrying neither a name nor a
chained one is reported as undecidable rather than as rot, because the
interpreter never reports an absent module without naming it -- that backstop is
what makes aborting the sweep unreachable whatever a future raise site does.

The guard is unchanged where it matters: a `ModuleNotFoundError` naming a
`strands_robots` path is still rot, and the existing control assertion that an
absent extra and absent rot must not become the same verdict still holds. CI
installs `[all,dev]`, so this was invisible there -- which is also why the
guard's coverage in CI was order-dependent and unverified: any `[all,dev]`
module failing to import for an unrelated reason stopped the sweep at that
point instead of skipping that one target.

Refs #2963, #2940 (whole-tree graders that stop covering the tree).
