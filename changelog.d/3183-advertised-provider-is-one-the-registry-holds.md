### Fixed: the provider a robot's tool schema advertises is one the registry holds

`Robot.tool_spec` described its `policy_provider` parameter as `"Policy
provider (groot, openai, etc.)"`. `openai` is not a registered provider - it is
in none of the 31 spellings the JSON and runtime registries hold - and that
schema is the only thing the model driving the tool reads, so it is a name the
model will pass.

Neither resolver names the schema that supplied it. `create_policy("openai")`
raises `Unknown policy provider`, and on this path that happens inside
`Robot._get_policy` on the executor thread, after `start_task` has already
answered `Task started` and the arm has connected. `resolve_policy("openai")`
returns `("lerobot_local", {"pretrained_name_or_path": "openai"})`, so on the
surfaces that reach it the name resolves - to the wrong provider, pointed at a
checkpoint repository that does not exist.

The refusal that arrives first is not about the provider either. `_get_policy`
reads the registry's `requires` field to decide whether a port is owed and
falls back to demanding one when the provider is unknown, so
`start_task(..., policy_provider="openai")` with no port is refused with
`policy_port is required to build a policy` - byte-identical to the same call
naming the real `groot`. Nothing on that path says the provider does not exist.

The description now names registered providers and points at `list_providers()`
for the enumeration, which is the form the MuJoCo tool spec already uses.

`tests/registry/test_advertised_providers_are_registered.py` exists for this
class and its docstring names both failure modes, but its population was one
file - `simulation/mujoco/tool_spec.json`, the surface that prompted it.
Measured over the package there are two `policy_provider` schema entries; that
one advertises five names and all five are registered, and the other is the one
above. The sweep now walks the package and grades every schema entry it finds -
the `e.g.` examples and the `default` - so a third surface is graded on arrival
rather than needing the guard to be widened again. The walk is rooted at an
imported symbol, so `scripts/check_whole_tree_graders.py` rosters the module
with no second edit.

The same sweep grades the 30 `policy_provider=` signature defaults in the
package, which is the half #3075 asks for: those are reached by exactly the
calls that pass the fewest arguments, so a provider rename or removal leaves
them naming nothing and the loss stays invisible until a caller omits the
argument. All 30 pass today, and `None` is skipped as the not-supplied
sentinel.
