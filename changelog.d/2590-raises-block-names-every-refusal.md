### Fixed: `build_lerobot_command` names the refusal a caller cannot otherwise handle

The `Raises:` block documented `ValueError` alone, while the DAgger preflight
added 28 days later raises `RuntimeError` when the installed lerobot has no
`lerobot.scripts.lerobot_rollout`. A `Raises:` block is the only place a caller
learns which `except` clause to write, so a caller who wrote exactly the handler
the docstring licensed took that refusal unhandled, with the source as the only
remedy. The class is right - a missing module is an environment problem rather
than a bad argument - and an existing test already pinned it, so only the block
was wrong.

Nothing had compared a `Raises:` block against the classes its function raises.
`tests/test_raises_docstring_completeness.py` now does, for every function whose
docstring already has one: 258 surfaces graded, 1 uncovered. It is deliberately
one-directional, because a documented class raised by a helper the function
delegates to is legitimate and common - 80 surfaces do exactly that - and three
shapes that are not refusals the function chose are filtered with a control each:
a re-raise of an exception captured elsewhere, a same-module factory whose return
annotation is the class, and a class raised and caught in one function. Unlike
its `Args:` sibling the population includes module-level functions, which is what
made this surface visible at all.
