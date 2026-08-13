### Fixed: an unresolvable `policy_provider` is reported instead of escaping the tool envelope

`run_policy` and `eval_policy` let `create_policy`'s `ValueError` raise **past**
the `status=error` envelope an agent tool is documented to return, so a caller
who guessed a provider name got a traceback rather than a result.
`start_policy` was worse: it builds the policy on a worker thread, so the raise
was captured in the future and never surfaced -- the call reported
`status="success"`, "Policy started", while `list_policies_running` reported
none. A rollout that could never build its policy was reported as started.

`preflight_policy` swallows resolution failures deliberately, on the stated
grounds that "create_policy raises the authoritative error". That premise holds
for a library caller, which sees the raise, and not for these surfaces. A new
`policy_provider_error` probes the same resolution path `create_policy` uses --
so every registered name, HuggingFace model ID, transport URL and `host:port`
still resolves -- and each surface now returns the reason on its own channel.
The message already named every registered provider, so a guess now gets the
available set back. The `start_policy` check runs before the executor submit.

A non-string `policy_provider` is refused there too: resolution indexes the
registry with it, so it previously arrived as a bare `TypeError` naming neither
the parameter nor the problem ("argument of type 'NoneType' is not iterable").
