### Fixed: a mesh subscription never silently fails to exist

`Mesh.subscribe` has three `None` returns. One - a raising `declare_subscriber` -
has always logged a WARNING naming the topic, in line with the six other
client-side refusals in `strands_robots/mesh/core.py`: `send`, `broadcast`, the
invalid-command and lockout rejections, the startup declare failure, and
`declare_publisher`. The other two, the peer not being on the mesh and there
being no session, answered `None` with no record at all.

Those two are the pair a caller meets on the only rejoin the class offers.
`stop()` drops every subscription `subscribe` records and clears `inbox`;
`start()` re-declares only the peer's own built-in topics, and the
`(topic, callback)` pairs behind a user subscription are not retained anywhere,
so a consumer that leaves and rejoins after a config change or a hub restart
re-subscribes. If the mesh has not come back up, that re-subscribe answered
`None` and said nothing - and `stop()` had not said the subscription was gone in
the first place, so an empty `_user_subs` was the only evidence either had
happened.

Both refusals now say why at WARNING, naming the topic and which of the three it
was, and `stop()` reports at INFO how many subscriptions it dropped and their
names when there were any, so a rejoining caller is told what it has to
re-declare. `subscribe`'s docstring gains a `Returns:` block accounting for all
three refusals and states that a subscription does not survive `stop()`;
`stop`'s states what it drops and why `start()` cannot restore it, and
`docs/mesh.md` says the same to a consumer under **Rejoining the mesh**.

Nothing about the round trip changes. Measured end to end against a live Zenoh
session, the subscription attaches and fires once, `stop()` empties `user_subs`
and `inbox`, the early re-subscribe answers `None`, and the rejoin restores the
`peer_id` and leaves an engaged e-stop lockout engaged - identically before and
after. A network blip was never a way to forget a stop; what it was is a way to
lose a subscription without being told.
