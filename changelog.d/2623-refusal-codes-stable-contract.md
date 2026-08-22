### Added: a stable code and subject on the refusals an operator can answer

Some refusals are *continuable*: the request was well formed, and an operator who
accepts the risk can grant something that makes the identical request succeed.
Anything that offers that choice - a consent card, an approval endpoint, a
supervising agent - has to recognise which refusal it is and what it is about,
and the only thing on offer was the message text. So recognition meant prose
matching: an env-var name appearing somewhere in the sentence, the subject pulled
back out with an anchored regex. Two of the refusals name no env var at all.

That made every wording improvement silently breaking, and this package could
not detect it. Rewording the teleop refusals from "out of range" to "exceeds the
safety envelope" - a plain improvement, and nothing in `tests/` asserts on either
phrase - leaves the mesh suite at 270 passed while a consumer's classification of
both refusals drops to `None`.

Continuable refusals now carry a stable machine-readable `code` from the new
`strands_robots.refusal_codes` and the `subject` they are about, so a consumer
switches on identity and the message stays free to improve.
`REFUSAL_GRANTS` names the environment variable that lifts each refusal, which a
consumer otherwise hard-codes. The codes are additive: all eight measured refusal
messages are byte-identical to before, and a rejection with no operator grant
behind it - a schema or bounds failure - stays code-less on purpose, so
`code is None` means "show the message and stop" rather than "unknown code".
