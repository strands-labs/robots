### Fixed: the duplicate-claim sweep reads every open pull request

`scripts/check_duplicate_claim.py --all-open` resolved each open pull request's
added paths from a single `files(first: 100)` page. GitHub caps that connection,
so one long pull request returned a truncated file list, and the sweep refused
the whole board rather than the one node it could not read: `open pull requests
read: 0`, `pairs compared: 0`, exit code 0. A 472-file draft therefore blinded
the relation for every other open pull request, and a caller reading only the
exit code saw a pass.

The resolver now follows `pageInfo.hasNextPage` per node, completing any
truncated file list before the bound check reads it, so the refusal it raises is
about a pull request whose list genuinely exceeds the bound rather than about a
list that was merely paginated.
