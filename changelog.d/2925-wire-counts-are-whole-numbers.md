### Fixed: a wire-side count is refused when fractional, never truncated

`validate_command` coerces four caller-supplied integers through one helper,
`_coerce_int`: `step.steps`, and `policy_port` / `action_horizon` / `n_steps` on
`start` / `execute`. That helper's docstring says it "mirrors the defences in
`_coerce_float`", and it mirrored two of them -- the `math.isfinite` check and
the coercion-error wrap -- while missing the property that makes `_coerce_float`
safe at all: it returns the caller's value or refuses it, and never a third
number. `int(...)` rounds toward zero, so the helper answered with a value
nobody sent.

Three consequences on the wire, every one silent, all reachable from an ordinary
payload -- `json.dumps` renders an integer held in a Python float as `2.0`, so a
float here is a normal shape rather than a contrived one. A count nobody asked
for was honoured: `{"steps": 2.5}` validated as `2`, and `{"policy_port":
5556.7}` as `5556`, which names a different endpoint rather than rounding a
magnitude. A below-floor refusal quoted a number the caller never wrote:
`{"steps": 0.9}` reported `steps=0 out of bounds [1, 10000]` -- the verdict was
right and the value in it was invented by the coercion.

The third is the one that mattered most: **the ceiling stopped refusing.** The
coercion runs before the bounds compare, so every one of the four fields carried
a value from above `hi` down to exactly `hi` and accepted it, while the integer
one step further was refused by name. `{"n_steps": 10000000.5}` was accepted as
`10000000` and `{"n_steps": 10000001}` was refused -- the same intent, two
answers, decided by how the JSON number was spelled. The comment above those
bounds says they "match the `SimEngine.run_policy` surface so a wire-side
`tell()` cannot drive the runner to absurd frequencies / step counts", so a
value over the cap is exactly what they exist to refuse.

A float with a fractional part is now refused under its own name, before the
coercion and after the finiteness check -- `float("nan").is_integer()` is False,
so a whole-number check placed first would answer a non-finite value with the
wrong reason. An integral float is still accepted: nothing is lost in coercing
`3.0`, and refusing it would refuse ordinary wire payloads for no gain. That is
exactly the split `positive_whole_number_error` applies to the same quantity on
the simulation side, which is the surface these bounds are declared to match --
`SimEngine.step` and `run_policy` already refuse a fractional horizon by name,
and `_coerce_float` is untouched because a fractional rate or duration is
perfectly usable.

The `step.steps` type case in the sibling suite had recorded the old behaviour
as an aside -- "a float is accepted and truncated, matching `int(...)`
semantics" -- in a file that refuses `bool` precisely because it "must not be
accepted as a count". That sentence has been corrected.
