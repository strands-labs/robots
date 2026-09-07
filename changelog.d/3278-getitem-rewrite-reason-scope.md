### Docs: the `seqiter` reason for refusing a `__getitem__` alert is scoped to the probe shape it holds on

`AGENTS.md`'s CI Security Baseline refused the rewrite
`py/unexpected-raise-in-special-method` suggests for a hostile `__getitem__` with one
mechanism, stated as a general fact about the rule: `IndexError` is what CPython's
`seqiter` clears to terminate legacy-protocol iteration, so taking the suggestion leaves
the probe raising nothing and the cell measuring nothing.

That holds for the #1890 probe -- a sequence reached through the legacy protocol -- and
for neither of the two other shapes the rule keeps arriving on. `KeyError` is the other
exception the query names for indexing and it is *not* cleared, so one spelling of the
suggestion preserves the measurement intact. And a `str` subclass supplies its own
`__iter__`, so `seqiter` is never constructed and the overridden `__getitem__` is not
consulted at all -- the mechanism is absent rather than adverse. Alert 1168 on #3272 is
that third shape, and because the 280-character dismissal comment cites this file instead
of restating an argument, a reason that does not hold there becomes a false claim in a
dismissal that outlives the branch.

The passage now carries a three-row table of the measured behaviour, the one-line
discriminator (`list(probe)`) a contributor can run before choosing a reason at all, and
the argument that generalises: refuse the rewrite on the property under test -- the query's
harm is a user of the class meeting an exception the protocol did not lead them to expect,
which is exactly what a hostile probe exists to be -- rather than on a mechanism that
depends on the probe's iteration protocol. The table is parsed and executed by
`TestTheGetItemRewriteIsNotOneBehaviour` rather than transcribed, on the same terms as the
`except BaseException` handler census below it, so a row that stops being true fails there.
