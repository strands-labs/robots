### Docs: a GraphQL mutation's node ID is checkable before the write, and a wrong one does not fail

`AGENTS.md` > PR Workflow > step 8 covered a mutation *reporting* the wrong
thing and not a mutation *addressing* the wrong thing, which is the direction
with no undo. A mutation names its subject by node ID and by nothing else -
`createIssue` takes a `repositoryId`, not an owner and a name - so a well-formed
but wrong ID succeeds against whatever object it does name. A `repositoryId`
carried over from an earlier response created an issue in an unrelated
third-party repository and returned success; `deleteIssue` needs admin on the
target, so it could not be undone.

Records that the ID is not opaque, which is what makes the write checkable
rather than merely regrettable: it decodes to a type prefix and, for anything a
repository owns, to that repository's `databaseId`, with no network call. All
three guessed IDs in the run at issue carried one wrong repository, so a single
stale value contaminated every mutation and the two that failed did so only
because their own `databaseId` happened not to exist there.

`tests/test_graphql_node_id_targeting.py` executes the claim rather than
asserting the prose says it, decoding this repository's own node IDs against the
`databaseId`s the API publishes beside them, so guidance that has become
impossible fails loudly instead of reading plausibly.
