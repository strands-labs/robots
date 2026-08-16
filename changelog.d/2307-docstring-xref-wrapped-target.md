### Fixed

The docstring cross-reference guard now checks a `:class:`/`:func:`/`:meth:` target
whose dotted path wraps over a line break. Its pattern stopped at the newline, so
such a role was never extracted and was exempt from the guard entirely; five live
roles in `strands_robots.policies` were dead pointers while the guard passed. A
target is now graded on being a contiguous dotted path as well as resolving, and
the refusal names the contiguous path to rewrap to. The five offenders are rewrapped.
