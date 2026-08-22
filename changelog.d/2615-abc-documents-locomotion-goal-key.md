### Fixed: the policy contract documents the locomotion goal key its providers read

`Policy.get_actions` is where the issue #300 well-known `**kwargs` goal
vocabulary is defined - the docstring a provider author reads to learn which
keys a caller may pass without coupling to a backend, and the one
`strands_robots.policies` points at by name for the list. It documented three
keys.

`target_velocity` was the fourth everywhere else: read by two independent
provider families (`wbc`, including its `gait` variant, and `motionbricks`),
forwarded by `run_policy` / `eval_policy` / `run_benchmark`, admitted by the
mesh wire validator and forwarded by the mesh dispatcher since #2613, and named
by `docs/policies/wbc.md` as "one of the issue #300 well-known goal keys". The
contract's own definition was the only place it was missing, and it is the place
a new provider author looks first.

Nothing executes a docstring, which is the mechanism: the one list in the set
that is not a consumer is the one a new key can be forgotten on while every
consumer keeps working. No guard caught it because the guards compared the
vocabulary against *another docstring* - #2613 graded the wire and dispatcher
against `run_policy`, a consumer - so the tree held two competing definitions
with the downstream pins anchored to the wrong one. The nearest thing to a
contract test hard-coded the same three keys and therefore agreed with the
omission.

Three enumerations that each claim to be the present, whole vocabulary are
corrected: the ABC, the package docstring that cites it, and `Cosmos3Policy`'s
disclaimer, which restates the list in order to say none of it is read and so
needs to keep covering every key it names. Provider-scoped lists (cuRobo reads
pose / joints / `world_update`, which is what cuRobo reads) and historical ones
("the contract landed in #300 (...)", true of #300 as shipped) are deliberately
left alone, because holding either to the current vocabulary would demand a
false statement.

The new guards do not compare the ABC against another docstring. They derive
what the vocabulary must contain from shipped provider code - the keys providers
actually read out of `**kwargs` - and hold the ABC to it in both directions. The
discriminator between shared vocabulary and a provider-private kwarg is the
second-caller rule AGENTS.md convention 11 already codifies, with a provider
family being the provider package so a provider's own variants count once
between them. On this tree that rule separates the four vocabulary keys from 26
private ones with no special cases, so the assertion is a set equality rather
than a containment, and `target_orientation` / `target_heading` / `height` are
pinned as single-family controls: they are `target_`-shaped and sit in the same
`get_actions` bodies, so a guard keyed on name shape would have promoted them.
