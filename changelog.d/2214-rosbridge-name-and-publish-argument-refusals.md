### Quality: drive the two `use_rosbridge` refusals nothing reached

`use_rosbridge` refuses a mistyped name for `topic` and for `service` from one
`_NAME_RE`, and refuses each action that is missing a required argument. Two
members of those families were never driven: the `service` name refusal, and the
`publish` refusal for a missing topic or interface type - the one action in that
family that writes to the robot. Their siblings were pinned, so both looked
covered.

Fifteen cases close them and take the module to 100%: the service name refused
for exactly the spellings its topic sibling is (compared, not restated, so the
two cannot drift apart on one rule), that refusal landing before the bridge is
dialed rather than after a client is cached, and an incomplete `publish`
advertising no publisher - asserted against a complete publish on the same
connection, so an empty topic list means "refused before advertising" rather
than "never advertises".

The `type` parameter is now documented as required for `publish`, and each name
parameter records that it shares one rule with the other.
