### Tests: the g1 live-handle sweep now fails when its own discovery narrows

`tests/tools/g1/test_a_live_handle_verb_refuses_a_wrong_handle.py` derives the
population it grades from the package, and every rule in the file iterates that
one scan. A scan that narrows therefore does not fail - it grades fewer verbs
and still reports green. The only self-check was `test_the_population_is_not_empty`,
which fires on an *empty* population; the narrowing that actually shipped left
seven of nine verbs in place, so it passed, and `g1_get_task_status` reached main
raising `AttributeError` past the structured response on all six wrong handles
while the file read green.

The population is now derived a second time from the package *source* by AST and
the two routes are compared. The routes share no mechanism - the source scan
never imports, so it cannot inherit a narrowing from the `pkgutil` enumeration,
from `__wrapped__` being absent on a verb, or from `__wrapped__.__module__`
disagreeing with the module a verb is defined in. It is deliberately wider than
the runtime scan in two ways, so that either kind of narrowing is loud: it reads
private modules too, and it accepts every spelling of `Any` rather than only the
bare one.

The disagreement is reported by naming the verbs rather than by a count, and no
verb is named in the assertion, so a verb the package gains is covered without
this file being edited. Measured against the narrowing this replaces, the check
fails naming exactly `g1_get_state` and `g1_get_task_status`; measured against a
new verb annotated `typing.Any`, the four existing population rules all pass and
this one fails naming it.
