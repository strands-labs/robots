### Fixed: the declared eclipse-zenoh floor now ships the wire source attribution the mesh safety path depends on

The mesh safety handlers authenticate an e-stop / resume publisher below the JSON
body using `zenoh.SourceInfo` on the publisher and `Sample.source_info` on the
receiver. Both first ship in eclipse-zenoh 1.6.1, but `[mesh]` declared
`>=1.0.0`, so ten of the audited releases satisfied packaging while exposing
neither name. Nothing raised: the publisher side probes
`hasattr(zenoh, "SourceInfo")` and the receiver side
`getattr(sample, "source_info", None)`, and both answer "no attribution
available" and continue.

That left two silent outcomes. An all-old fleet published every safety envelope
unattributed, so the cross-session forgery defence was not in effect and nothing
said so. A mixed fleet failed closed on the wrong peer: a publisher on 1.6.1
attaches the wire zid and the body `source_zid`, a receiver on 1.5.1 sees the
body field with no wire counterpart, takes the "publisher misconfigured or
attacker stripped SourceInfo" branch, and does not engage the lockout -- so a
fleet-wide emergency stop did not stop that robot, and the warning blamed the
publisher. Both peers satisfied the declared range.

`[mesh]` now floors at `eclipse-zenoh>=1.6.1`, and a guard owns the measurement:
it fails if the mesh reaches for a zenoh name with no recorded first-shipped
release, so a newer API cannot leave the floor behind.
