### Fixed: a policy provider whose optional dependency is missing now names the remedy instead of raising a bare ModuleNotFoundError

`create_policy("lerobot_local", ...)` on an install without `torch` raised
`ModuleNotFoundError: No module named 'torch'` -- an error naming neither the
provider the caller asked for nor the way to fix it. Every other provider defers
its heavy import and reports through `require_optional` / `require_optionals`,
which name the extra that ships the dependency; `lerobot_local` imports `torch`
at module level, so the failure happened while importing the provider's module,
before any provider machinery could run.

`import_policy_class` -- the single funnel every provider class is imported
through -- now translates a failed provider import into an error naming the
provider, the missing module and the install command, using the `extra` the
provider declares in `policies.json`. The remedy therefore no longer depends on
WHERE a provider happens to import its dependency.

The auto-discovery branch swallowed the same `ImportError` and reported
`Unknown policy provider`, sending a caller whose provider name was correct to go
and check the name; a module that exists but cannot satisfy its dependency now
reports the dependency. An unknown provider is still reported as an unknown
provider.

No provider has ever been substituted for another when its dependency was
missing, and that is now pinned: a failed import reports rather than returning a
different provider's policy whose action space happens to fit.
