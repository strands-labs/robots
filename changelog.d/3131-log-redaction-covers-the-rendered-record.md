### Fixed: the dashboard log filter redacts every part a formatter renders, once

`logging.Formatter.format` renders three parts and appends the last two -- the message,
`exc_text` (from `exc_info`) and `stack_info` -- and both are appended after every filter has
run. `RedactingFilter` read `getMessage()` alone, so a failing socket wrote its request URL a
second time inside a traceback, with the credential verbatim, underneath a redacted access-log
line. `exc_info` is now rendered and redacted in the filter (the formatter reuses an `exc_text`
that is already set) and `stack_info` is redacted too.

Redaction is also idempotent per record. `install_redaction` attaches the filter at both the
logger and its handlers, so a record was redacted twice, and a fingerprint's own text matches
the value pattern: the second pass reported the fingerprint's length rather than the
credential's (`?token=<redacted:18:aco>>` for a 43-character token). An unformattable message
no longer exempts the rest of the record either.

Rendering and redacting those parts is guarded the way the message rail already was: a part this
filter cannot render or redact is withheld with a marker saying so, rather than raised out of the
caller's own logging call. A `Formatter` runs inside `Handler.emit`, where a broken record degrades
to a note on stderr, but a filter has no such guard - and the default target set includes the root
logger, so an escape there would reach any logging call in the process.
