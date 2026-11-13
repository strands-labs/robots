### Fixed: a Feetech servo that does not answer a read is named by its own ID, not the next servo's

A sync_read took the replies in the order asked, so when servo 3 was silent the frame read for it was servo 4's and
the error said `reply from servo 4, expected 3`: the operator checked the wrong servo. Each reply is now filed under
the ID it carries, the read continues to the last servo, and the error lists every silent one:
`sync_read: no reply from servo(s) [3] (got 5 of 6)`. A silent servo before the last costs no timeout, a reply from a
servo that was not asked is refused, and a reading that lands on a retry is used as before.
